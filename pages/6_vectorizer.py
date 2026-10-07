"""Страница векторизации — загрузка чанков и изображений в Qdrant.

6 вкладок:
  1. 📝 Векторизация текста — текстовые чанки → dense + sparse векторы
  2. 📋 Векторизация таблиц — табличные чанки → dense + sparse векторы
  3. 🖼️ Векторизация изображений — описания/картинки → image_dense векторы
  4. 📊 Коллекции — управление коллекциями Qdrant
  5. 💾 Снапшоты — создание/восстановление бэкапов коллекций
  6. 🧩 Модели — просмотр, скачивание, статус моделей эмбеддингов
"""

import logging
import threading
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.config import DATA_DIR, get_secret
from src.vectorizer import (
    DENSE_TEXT_MODELS,
    SPARSE_MODELS,
    delete_collection,
    download_model,
    find_book_chunk_dirs,
    find_book_image_dirs,
    find_book_table_dirs,
    get_all_models_status,
    get_model_dimension,
    get_qdrant_client,
    list_collections,
    ping_qdrant,
    validate_collection_dims,
)
from src.run_vectorizer import run_text_vectorizer, run_table_vectorizer, run_image_vectorizer
from src.qdrant_snapshots import (
    check_qdrant_health, get_collection_info,
    create_snapshot, list_snapshots, restore_from_snapshot,
    delete_snapshot, snapshot_all_collections,
)

logger = logging.getLogger(__name__)

# JS для автообновления страницы (threading progress pattern)
_AUTORELOAD_JS = (
    "<script>setTimeout(function(){window.parent.document.querySelector"
    "('[data-testid=\"stAppViewContainer\"]').click();}, 1500);</script>"
)
_QUICK_RELOAD_JS = (
    "<script>setTimeout(function(){window.parent.document.querySelector"
    "('[data-testid=\"stAppViewContainer\"]').click();}, 500);</script>"
)


# ══════════════════════════════════════════════════════════════════════
# Кэширование Qdrant клиента
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_cached_qdrant_client(host: str = "localhost", port: int = 6333):
    """Кэшированный Qdrant клиент (singleton)."""
    return get_qdrant_client(host, port)


# ══════════════════════════════════════════════════════════════════════
# Страница
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Векторизация", page_icon="🧠", layout="wide")
st.title("🧠 Векторизация")
st.markdown("Загрузка чанков и изображений в векторную базу данных Qdrant.")

# ── Подключение к Qdrant ────────────────────────────────────────────
with st.expander("⚙️ Подключение к Qdrant", expanded=True):
    col1, col2 = st.columns([3, 1])
    with col1:
        qdrant_host = st.text_input(
            "Хост Qdrant",
            value=st.session_state.get("vec_qdrant_host", "localhost"),
            key="vec_qdrant_host",
        )
    with col2:
        qdrant_port = st.number_input(
            "Порт",
            value=st.session_state.get("vec_qdrant_port", 6333),
            min_value=1,
            max_value=65535,
            key="vec_qdrant_port",
        )

    # Проверка подключения
    ok, msg = ping_qdrant(qdrant_host, int(qdrant_port))
    if ok:
        st.success(f"✅ {msg}")
    else:
        st.error(f"❌ {msg}")
        st.info("💡 Запустите Qdrant: `docker compose up -d` в папке проекта")
        st.stop()

client = get_cached_qdrant_client(qdrant_host, int(qdrant_port))

# ── Вкладки ─────────────────────────────────────────────────────────
tab_text, tab_tables, tab_images, tab_collections, tab_snapshots, tab_models = st.tabs([
    "📝 Векторизация текста",
    "📋 Векторизация таблиц",
    "🖼️ Векторизация изображений",
    "📊 Коллекции",
    "💾 Снапшоты",
    "🧩 Модели",
])


# ══════════════════════════════════════════════════════════════════════
# Вкладка 1: Векторизация текста
# ══════════════════════════════════════════════════════════════════════

with tab_text:
    st.subheader("📝 Векторизация текстовых чанков")

    st.markdown("""
    **Dense-эмбеддинги** — преобразуют текст в числовой вектор фиксированной размерности.
    Capture семантическое сходство: похожие по смыслу тексты имеют близкие векторы.
    Используются для семантического поиска (найти «что-то про сушку угля» → вернёт релевантные абзацы).

    **Sparse-эмбеддинги** — создают разреженные векторы на основе весов токенов (BM42).
    Capture точное совпадение ключевых слов.
    Используются для лексического поиска и гибридного поиска (dense + sparse вместе дают лучший результат).
    """)

    # Настройки
    col_model, col_sparse = st.columns(2)

    with col_model:
        st.markdown("#### Dense-модель (обязательно)")
        model_keys = list(DENSE_TEXT_MODELS.keys())
        model_labels = {
            k: f"{DENSE_TEXT_MODELS[k]['name']} ({DENSE_TEXT_MODELS[k]['dim']}d)"
            for k in model_keys
        }

        selected_dense = st.selectbox(
            "Модель dense-эмбеддингов",
            options=model_keys,
            format_func=lambda k: model_labels[k],
            index=0,  # e5-small по умолчанию
            key="vec_dense_model",
        )

        # Описание выбранной модели
        dense_info = DENSE_TEXT_MODELS[selected_dense]
        st.caption(f"ℹ️ {dense_info['description']}")
        st.caption(f"📐 Размерность вектора: **{dense_info['dim']}**")

        # Предупреждение / статус для выбранного провайдера
        if dense_info["provider"] == "gigachat":
            _gc_cred = get_secret("GIGACHAT_AUTH_TOKEN") or get_secret("GIGACHAT_SECRET")
            if _gc_cred:
                st.success("✅ GigaChat credentials найдены")
            else:
                st.error("❌ GIGACHAT_AUTH_TOKEN или GIGACHAT_SECRET не заданы в secrets")
            st.caption("⚠️ GigaChat ограничивает ~1900 символов на текст.")
        elif dense_info["provider"] == "neuroapi":
            _na_key = get_secret("NEUROAPI_API_KEY")
            if _na_key:
                st.success("✅ NeuroAPI API ключ найден")
            else:
                st.error("❌ NEUROAPI_API_KEY не задан в secrets")
            st.caption("📡 Эмбеддинги через API — нужен интернет.")

    with col_sparse:
        st.markdown("#### Sparse-эмбеддинги")
        sparse_enabled = st.checkbox(
            "Включить sparse-эмбеддинги",
            value=True,
            key="vec_sparse_enabled",
            help="BM42 — улучшает поиск по ключевым словам",
        )

        if sparse_enabled:
            sparse_keys = list(SPARSE_MODELS.keys())
            selected_sparse = st.selectbox(
                "Sparse-модель",
                options=sparse_keys,
                format_func=lambda k: SPARSE_MODELS[k]["name"],
                key="vec_sparse_model",
            )
            st.caption(f"ℹ️ {SPARSE_MODELS[selected_sparse]['description']}")
        else:
            selected_sparse = "bm42"

    # Дополнительные настройки
    with st.expander("🔧 Дополнительные настройки"):
        collection_name = st.text_input(
            "Имя коллекции Qdrant",
            value="book_chunks",
            key="vec_text_collection",
        )

        base_dir = st.text_input(
            "Папка с книгами",
            value=str(DATA_DIR / "books"),
            key="vec_text_base_dir",
        )

        col_batch, col_recreate = st.columns(2)
        with col_batch:
            batch_size = st.number_input(
                "Размер батча",
                value=32,
                min_value=1,
                max_value=256,
                key="vec_text_batch_size",
                help="Количество текстов за один вызов модели",
            )
        with col_recreate:
            recreate = st.checkbox(
                "Пересоздать коллекцию ⚠️",
                value=False,
                key="vec_text_recreate",
                help="Удалит все данные в коллекции и создаст заново!",
            )

        if recreate:
            st.error("🚨 При пересоздании коллекции ВСЕ данные будут удалены!")

    # Предпросмотр
    book_dirs = find_book_chunk_dirs(Path(base_dir))
    if book_dirs:
        st.info(f"📚 Найдено книг с чанками: **{len(book_dirs)}**")

        # Выбор книг для векторизации
        _all_text_book_names = [bn for bn, _ in book_dirs]
        _selected_text_books = st.multiselect(
            "Выберите книги для векторизации",
            options=_all_text_book_names,
            default=_all_text_book_names,
            key="vec_text_book_select",
            help="Оставьте все выбранными для векторизации всех книг",
        )
    else:
        st.warning(f"📂 В папке `{base_dir}` нет книг с чанками")
        _selected_text_books = None

    # Проверка размерностей
    if not recreate:
        _dims_ok, _dims_msg = validate_collection_dims(
            client, collection_name,
            get_model_dimension(selected_dense),
        )
        if not _dims_ok:
            st.error(f"🚨 {_dims_msg}")
    else:
        st.caption("ℹ️ Коллекция будет пересоздана — проверка размерностей не требуется")

    # ── Запуск ───────────────────────────────────────────────────────
    st.divider()

    _thread = st.session_state.get("vec_text_thread")
    _is_running = _thread is not None and _thread.is_alive()

    if _is_running:
        _progress = st.session_state.get("vec_text_progress", {"pct": 0, "msg": ""})
        _pct = _progress.get("pct", 0)
        if _pct >= 0:
            st.progress(_pct, text=_progress.get("msg", ""))

        with st.status("📋 Ход выполнения", expanded=True):
            st.write(f"📖 {_progress.get('msg', '...')}")

        if st.button("❌ Отменить векторизацию текста", key="vec_text_cancel_btn"):
            cancel_ev = st.session_state.get("vec_text_cancel")
            if cancel_ev:
                cancel_ev.set()

        components.html(_AUTORELOAD_JS, height=0)
    else:
        # Результаты после завершения
        _result = st.session_state.get("vec_text_result")
        if _result:
            # Ошибка верхнего уровня (exception в thread — ключ "error", ед.ч.)
            if _result.get("error"):
                st.error(f"❌ {_result['error']}")
            elif _result.get("errors"):
                for _err in _result["errors"]:
                    st.error(f"❌ {_err}")
                if _result.get("total_upserted", 0) > 0:
                    st.success(
                        f"Частично загружено: {_result.get('total_upserted', 0)} чанков"
                    )
            else:
                st.success(
                    f"✅ Готово! Книг: {_result.get('books_processed', 0)}, "
                    f"чанков загружено: {_result.get('total_upserted', 0)}"
                )

        # Проверка выбора книг
        _text_books_to_vec = _selected_text_books if _selected_text_books else None
        if _text_books_to_vec is not None and len(_text_books_to_vec) == 0:
            st.warning("⚠️ Выберите хотя бы одну книгу для векторизации")
        elif st.button(
            "🚀 Запустить векторизацию текста",
            key="vec_text_run_btn",
            use_container_width=True,
            type="primary",
        ):
            # Сброс
            st.session_state.vec_text_result = None
            st.session_state.vec_text_progress = {"pct": 0, "msg": "Запуск..."}

            cancel_event = threading.Event()
            st.session_state.vec_text_cancel = cancel_event

            def _progress_cb(pct: float, msg: str):
                st.session_state.vec_text_progress = {"pct": pct, "msg": msg}

            def _run():
                try:
                    result = run_text_vectorizer(
                        client=client,
                        base_dir=Path(base_dir),
                        collection_name=collection_name,
                        dense_model_key=selected_dense,
                        sparse_enabled=sparse_enabled,
                        sparse_model_key=selected_sparse,
                        batch_size=int(batch_size),
                        recreate=recreate,
                        progress_callback=_progress_cb,
                        cancel_event=cancel_event,
                        book_names=_text_books_to_vec,
                    )
                    st.session_state.vec_text_result = result
                except Exception as e:
                    st.session_state.vec_text_result = {"error": str(e)}
                    st.session_state.vec_text_progress = {"pct": -1, "msg": f"❌ Ошибка: {e}"}

            thread = threading.Thread(target=_run, daemon=True)
            st.session_state.vec_text_thread = thread
            thread.start()

        components.html(_QUICK_RELOAD_JS, height=0)


# ══════════════════════════════════════════════════════════════════════
# Вкладка 2: Векторизация таблиц
# ══════════════════════════════════════════════════════════════════════

with tab_tables:
    st.subheader("📋 Векторизация табличных чанков")

    st.markdown("""
    Табличные чанки извлекаются на этапе деления на чанки и сохраняются в `chunks/table_chunks/*.json`.
    Они содержат таблицы Markdown с контекстом (`context_before`) — текстом перед таблицей.

    Векторизуются **теми же моделями**, что и текст (dense + sparse), но с `chunk_type="table"` в Qdrant.
    Это позволяет при поиске фильтровать или отдельно ранжировать таблицы.
    """)

    # Настройки
    col_tbl_model, col_tbl_sparse = st.columns(2)

    with col_tbl_model:
        st.markdown("#### Dense-модель (обязательно)")
        tbl_model_keys = list(DENSE_TEXT_MODELS.keys())
        tbl_model_labels = {
            k: f"{DENSE_TEXT_MODELS[k]['name']} ({DENSE_TEXT_MODELS[k]['dim']}d)"
            for k in tbl_model_keys
        }

        tbl_selected_dense = st.selectbox(
            "Модель dense-эмбеддингов",
            options=tbl_model_keys,
            format_func=lambda k: tbl_model_labels[k],
            index=0,
            key="vec_table_dense_model",
        )

        tbl_dense_info = DENSE_TEXT_MODELS[tbl_selected_dense]
        st.caption(f"ℹ️ {tbl_dense_info['description']}")
        st.caption(f"📐 Размерность вектора: **{tbl_dense_info['dim']}**")

        if tbl_dense_info["provider"] == "gigachat":
            _gc_cred2 = get_secret("GIGACHAT_AUTH_TOKEN") or get_secret("GIGACHAT_SECRET")
            if _gc_cred2:
                st.success("✅ GigaChat credentials найдены")
            else:
                st.error("❌ GIGACHAT_AUTH_TOKEN или GIGACHAT_SECRET не заданы в secrets")
            st.caption("⚠️ GigaChat ограничивает ~1900 символов на текст.")
        elif tbl_dense_info["provider"] == "neuroapi":
            _na_key2 = get_secret("NEUROAPI_API_KEY")
            if _na_key2:
                st.success("✅ NeuroAPI API ключ найден")
            else:
                st.error("❌ NEUROAPI_API_KEY не задан в secrets")
            st.caption("📡 Эмбеддинги через API — нужен интернет.")

    with col_tbl_sparse:
        st.markdown("#### Sparse-эмбеддинги")
        tbl_sparse_enabled = st.checkbox(
            "Включить sparse-эмбеддинги",
            value=True,
            key="vec_table_sparse_enabled",
            help="BM42 — улучшает поиск по ключевым словам",
        )

        if tbl_sparse_enabled:
            tbl_sparse_keys = list(SPARSE_MODELS.keys())
            tbl_selected_sparse = st.selectbox(
                "Sparse-модель",
                options=tbl_sparse_keys,
                format_func=lambda k: SPARSE_MODELS[k]["name"],
                key="vec_table_sparse_model",
            )
            st.caption(f"ℹ️ {SPARSE_MODELS[tbl_selected_sparse]['description']}")
        else:
            tbl_selected_sparse = "bm42"

    # Дополнительные настройки
    with st.expander("🔧 Дополнительные настройки"):
        tbl_collection = st.text_input(
            "Имя коллекции Qdrant",
            value="book_chunks",
            key="vec_table_collection",
        )

        tbl_base_dir = st.text_input(
            "Папка с книгами",
            value=str(DATA_DIR / "books"),
            key="vec_table_base_dir",
        )

        col_tbl_batch, col_tbl_recreate = st.columns(2)
        with col_tbl_batch:
            tbl_batch_size = st.number_input(
                "Размер батча",
                value=32,
                min_value=1,
                max_value=256,
                key="vec_table_batch_size",
                help="Количество таблиц за один вызов модели",
            )
        with col_tbl_recreate:
            tbl_recreate = st.checkbox(
                "Пересоздать коллекцию ⚠️",
                value=False,
                key="vec_table_recreate",
                help="Удалит все данные в коллекции и создаст заново!",
            )

        if tbl_recreate:
            st.error("🚨 При пересоздании коллекции ВСЕ данные будут удалены!")

    # Предпросмотр
    table_dirs = find_book_table_dirs(Path(tbl_base_dir))
    if table_dirs:
        st.info(f"📋 Найдено книг с табличными чанками: **{len(table_dirs)}**")

        # Выбор книг для векторизации
        _all_table_book_names = [bn for bn, _ in table_dirs]
        _selected_table_books = st.multiselect(
            "Выберите книги для векторизации",
            options=_all_table_book_names,
            default=_all_table_book_names,
            key="vec_table_book_select",
            help="Оставьте все выбранными для векторизации всех книг",
        )
    else:
        st.warning(f"📂 В папке `{tbl_base_dir}` нет книг с `chunks/table_chunks/`")
        _selected_table_books = None

    # Проверка размерностей
    if not tbl_recreate:
        _tbl_dims_ok, _tbl_dims_msg = validate_collection_dims(
            client, tbl_collection,
            get_model_dimension(tbl_selected_dense),
        )
        if not _tbl_dims_ok:
            st.error(f"🚨 {_tbl_dims_msg}")
    else:
        st.caption("ℹ️ Коллекция будет пересоздана — проверка размерностей не требуется")

    # ── Запуск ───────────────────────────────────────────────────────
    st.divider()

    _tbl_thread = st.session_state.get("vec_table_thread")
    _tbl_is_running = _tbl_thread is not None and _tbl_thread.is_alive()

    if _tbl_is_running:
        _tbl_progress = st.session_state.get("vec_table_progress", {"pct": 0, "msg": ""})
        _tbl_pct = _tbl_progress.get("pct", 0)
        if _tbl_pct >= 0:
            st.progress(_tbl_pct, text=_tbl_progress.get("msg", ""))

        with st.status("📋 Ход выполнения", expanded=True):
            st.write(f"📋 {_tbl_progress.get('msg', '...')}")

        if st.button("❌ Отменить векторизацию таблиц", key="vec_table_cancel_btn"):
            tbl_cancel_ev = st.session_state.get("vec_table_cancel")
            if tbl_cancel_ev:
                tbl_cancel_ev.set()

        components.html(_AUTORELOAD_JS, height=0)
    else:
        _tbl_result = st.session_state.get("vec_table_result")
        if _tbl_result:
            if _tbl_result.get("error"):
                st.error(f"❌ {_tbl_result['error']}")
            elif _tbl_result.get("errors"):
                for _tbl_err in _tbl_result["errors"]:
                    st.error(f"❌ {_tbl_err}")
                if _tbl_result.get("total_upserted", 0) > 0:
                    st.success(
                        f"Частично загружено: {_tbl_result.get('total_upserted', 0)} таблиц"
                    )
            else:
                st.success(
                    f"✅ Готово! Книг: {_tbl_result.get('books_processed', 0)}, "
                    f"таблиц загружено: {_tbl_result.get('total_upserted', 0)}"
                )

        # Проверка выбора книг
        _tbl_books_to_vec = _selected_table_books if _selected_table_books else None
        if _tbl_books_to_vec is not None and len(_tbl_books_to_vec) == 0:
            st.warning("⚠️ Выберите хотя бы одну книгу для векторизации")
        elif st.button(
            "🚀 Запустить векторизацию таблиц",
            key="vec_table_run_btn",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.vec_table_result = None
            st.session_state.vec_table_progress = {"pct": 0, "msg": "Запуск..."}

            tbl_cancel_event = threading.Event()
            st.session_state.vec_table_cancel = tbl_cancel_event

            def _tbl_progress_cb(pct: float, msg: str):
                st.session_state.vec_table_progress = {"pct": pct, "msg": msg}

            def _tbl_run():
                try:
                    result = run_table_vectorizer(
                        client=client,
                        base_dir=Path(tbl_base_dir),
                        collection_name=tbl_collection,
                        dense_model_key=tbl_selected_dense,
                        sparse_enabled=tbl_sparse_enabled,
                        sparse_model_key=tbl_selected_sparse,
                        batch_size=int(tbl_batch_size),
                        recreate=tbl_recreate,
                        progress_callback=_tbl_progress_cb,
                        cancel_event=tbl_cancel_event,
                        book_names=_tbl_books_to_vec,
                    )
                    st.session_state.vec_table_result = result
                except Exception as e:
                    st.session_state.vec_table_result = {"error": str(e)}
                    st.session_state.vec_table_progress = {"pct": -1, "msg": f"❌ Ошибка: {e}"}

            tbl_thread = threading.Thread(target=_tbl_run, daemon=True)
            st.session_state.vec_table_thread = tbl_thread
            tbl_thread.start()

        components.html(_QUICK_RELOAD_JS, height=0)


# ══════════════════════════════════════════════════════════════════════
# Вкладка 3: Векторизация изображений
# ══════════════════════════════════════════════════════════════════════

with tab_images:
    st.subheader("🖼️ Векторизация изображений")

    st.markdown("""
    Описания изображений векторизуются **той же dense-моделью**, что и текст/таблицы.
    Все типы контента хранятся в одном dense-векторном пространстве — это позволяет
    искать текст, таблицы и описания картинок одним поисковым запросом.

    Источник данных: `parsed/images_descriptions.json` — результаты AI-парсинга страниц.
    """)

    # Выбор модели (из DENSE_TEXT_MODELS)
    img_model_keys = list(DENSE_TEXT_MODELS.keys())
    img_model_labels = {
        k: f"{DENSE_TEXT_MODELS[k]['name']} ({DENSE_TEXT_MODELS[k]['dim']}d)"
        for k in img_model_keys
    }

    selected_image_model = st.selectbox(
        "Dense-модель для описаний",
        options=img_model_keys,
        format_func=lambda k: img_model_labels[k],
        key="vec_image_model",
    )

    img_info = DENSE_TEXT_MODELS[selected_image_model]
    st.caption(f"ℹ️ {img_info['description']}")
    st.caption(f"📐 Размерность вектора: **{img_info['dim']}**")

    # Дополнительные настройки
    with st.expander("🔧 Дополнительные настройки"):
        img_collection = st.text_input(
            "Имя коллекции Qdrant",
            value="book_chunks",
            key="vec_image_collection",
        )

        img_base_dir = st.text_input(
            "Папка с книгами",
            value=str(DATA_DIR / "books"),
            key="vec_image_base_dir",
        )

        img_batch_size = st.number_input(
            "Размер батча",
            value=16,
            min_value=1,
            max_value=128,
            key="vec_image_batch_size",
        )

        img_recreate = st.checkbox(
            "Пересоздать коллекцию ⚠️",
            value=False,
            key="vec_image_recreate",
        )

    # Предпросмотр
    image_dirs = find_book_image_dirs(Path(img_base_dir))
    if image_dirs:
        st.info(f"📚 Найдено книг с изображениями: **{len(image_dirs)}**")

        # Выбор книг для векторизации
        _all_image_book_names = [bn for bn, _, _ in image_dirs]
        _selected_image_books = st.multiselect(
            "Выберите книги для векторизации",
            options=_all_image_book_names,
            default=_all_image_book_names,
            key="vec_image_book_select",
            help="Оставьте все выбранными для векторизации всех книг",
        )
    else:
        st.warning(f"📂 В папке `{img_base_dir}` нет книг с `parsed/images_descriptions.json`")
        _selected_image_books = None

    # Проверка размерностей
    if not img_recreate:
        _img_dims_ok, _img_dims_msg = validate_collection_dims(
            client, img_collection,
            get_model_dimension(selected_image_model),
        )
        if not _img_dims_ok:
            st.error(f"🚨 {_img_dims_msg}")
    else:
        st.caption("ℹ️ Коллекция будет пересоздана — проверка размерностей не требуется")

    # ── Запуск ───────────────────────────────────────────────────────
    st.divider()

    _img_thread = st.session_state.get("vec_image_thread")
    _img_is_running = _img_thread is not None and _img_thread.is_alive()

    if _img_is_running:
        _img_progress = st.session_state.get("vec_image_progress", {"pct": 0, "msg": ""})
        _img_pct = _img_progress.get("pct", 0)
        if _img_pct >= 0:
            st.progress(_img_pct, text=_img_progress.get("msg", ""))

        with st.status("📋 Ход выполнения", expanded=True):
            st.write(f"🖼️ {_img_progress.get('msg', '...')}")

        if st.button("❌ Отменить векторизацию изображений", key="vec_image_cancel_btn"):
            img_cancel_ev = st.session_state.get("vec_image_cancel")
            if img_cancel_ev:
                img_cancel_ev.set()

        components.html(_AUTORELOAD_JS, height=0)
    else:
        _img_result = st.session_state.get("vec_image_result")
        if _img_result:
            if _img_result.get("error"):
                st.error(f"❌ {_img_result['error']}")
            elif _img_result.get("errors"):
                for _img_err in _img_result["errors"]:
                    st.error(f"❌ {_img_err}")
                if _img_result.get("total_upserted", 0) > 0:
                    st.success(
                        f"Частично загружено: {_img_result.get('total_upserted', 0)} изображений"
                    )
            else:
                st.success(
                    f"✅ Готово! Книг: {_img_result.get('books_processed', 0)}, "
                    f"изображений загружено: {_img_result.get('total_upserted', 0)}"
                )

        # Проверка выбора книг
        _img_books_to_vec = _selected_image_books if _selected_image_books else None
        if _img_books_to_vec is not None and len(_img_books_to_vec) == 0:
            st.warning("⚠️ Выберите хотя бы одну книгу для векторизации")
        elif st.button(
            "🚀 Запустить векторизацию изображений",
            key="vec_image_run_btn",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.vec_image_result = None
            st.session_state.vec_image_progress = {"pct": 0, "msg": "Запуск..."}

            img_cancel_event = threading.Event()
            st.session_state.vec_image_cancel = img_cancel_event

            def _img_progress_cb(pct: float, msg: str):
                st.session_state.vec_image_progress = {"pct": pct, "msg": msg}

            def _img_run():
                try:
                    result = run_image_vectorizer(
                        client=client,
                        base_dir=Path(img_base_dir),
                        collection_name=img_collection,
                        dense_model_key=selected_image_model,
                        batch_size=int(img_batch_size),
                        recreate=img_recreate,
                        progress_callback=_img_progress_cb,
                        cancel_event=img_cancel_event,
                        book_names=_img_books_to_vec,
                    )
                    st.session_state.vec_image_result = result
                except Exception as e:
                    st.session_state.vec_image_result = {"error": str(e)}
                    st.session_state.vec_image_progress = {"pct": -1, "msg": f"❌ Ошибка: {e}"}

            img_thread = threading.Thread(target=_img_run, daemon=True)
            st.session_state.vec_image_thread = img_thread
            img_thread.start()

        components.html(_QUICK_RELOAD_JS, height=0)


# ══════════════════════════════════════════════════════════════════════
# Вкладка 4: Коллекции
# ══════════════════════════════════════════════════════════════════════

with tab_collections:
    st.subheader("📊 Управление коллекциями Qdrant")

    # Список коллекций
    collections = list_collections(client)

    if not collections:
        st.info("📭 Нет коллекций. Запустите векторизацию для создания.")
    else:
        for coll in collections:
            with st.container():
                col_info, col_del = st.columns([4, 1])

                with col_info:
                    name = coll.get("name", "?")
                    points = coll.get("points_count", 0)
                    status = coll.get("status", "?")
                    vectors = coll.get("vectors", {})

                    st.markdown(f"**📦 `{name}`** — {points} точек, статус: {status}")

                    if vectors:
                        vec_str = ", ".join(
                            f"{vname}: {vinfo.get('size', '?')}d"
                            for vname, vinfo in vectors.items()
                        )
                        st.caption(f"Векторы: {vec_str}")

                    if coll.get("error"):
                        st.error(f"Ошибка: {coll['error']}")

                with col_del:
                    if st.button(
                        "🗑️ Удалить",
                        key=f"del_coll_{coll.get('name', 'unknown')}",
                    ):
                        if delete_collection(client, coll["name"]):
                            st.success(f"✅ Коллекция '{coll['name']}' удалена")
                            st.rerun()
                        else:
                            st.error("❌ Ошибка удаления")

                st.divider()


# ══════════════════════════════════════════════════════════════════════
# Вкладка 5: Снапшоты
# ══════════════════════════════════════════════════════════════════════

with tab_snapshots:
    st.subheader("💾 Снапшоты (бэкапы коллекций)")
    st.markdown("""
    **Снапшот** — полная копия коллекции в момент времени. Позволяет восстановить данные
    при повреждении базы (например, после некорректного завершения Docker).

    ⚠️ **Создавайте снапшот ПЕРЕД выключением компьютера** — это гарантирует
    восстановление данных при следующем запуске.
    """)

    # Кнопка создания снапшотов всех коллекций
    col_snap1, col_snap2 = st.columns([1, 3])
    with col_snap1:
        if st.button(
            "📸 Создать снапшоты всех коллекций",
            key="snap_create_all",
            type="primary",
            use_container_width=True,
        ):
            results = snapshot_all_collections()
            for r in results:
                if "error" in r:
                    st.error(f"❌ {r['collection']}: {r['error']}")
                else:
                    st.success(f"✅ {r['collection']}: {r.get('name', 'OK')}")
            if not results:
                st.warning("📭 Нет коллекций для снапшота")

    with col_snap2:
        st.caption("💡 Также: `python scripts_/shutdown_qdrant.py` — снапшот + graceful shutdown")

    st.divider()

    # Список снапшотов по коллекциям
    _snap_collections = list_collections(client)
    if not _snap_collections:
        st.info("📭 Нет коллекций — снапшотов нет.")
    else:
        for _coll_info in _snap_collections:
            _coll_name = _coll_info.get("name", "?")
            _points = _coll_info.get("points_count", 0)

            st.markdown(f"#### 📦 `{_coll_name}` ({_points} точек)")

            # Кнопка создания снапшота одной коллекции
            col_s1, col_s2 = st.columns([1, 4])
            with col_s1:
                if st.button("📸 Снапшот", key=f"snap_create_{_coll_name}"):
                    res = create_snapshot(_coll_name)
                    if "error" in res:
                        st.error(f"❌ {res['error']}")
                    else:
                        st.success(f"✅ Создан: {res.get('name', 'OK')}")
                        st.rerun()

            with col_s2:
                # Список существующих снапшотов
                snaps = list_snapshots(_coll_name)
                if not snaps:
                    st.caption("Снапшотов нет")
                else:
                    for snap in snaps:
                        snap_name = snap.get("name", "?")
                        snap_size = snap.get("size", 0)
                        snap_date = snap.get("creation_time", "")
                        size_str = f"{snap_size / 1024 / 1024:.1f} MB" if snap_size else "?"

                        col_sn, col_rest, col_del = st.columns([3, 1, 1])
                        with col_sn:
                            st.text(f"  {snap_name} ({size_str}) {snap_date}")
                        with col_rest:
                            if st.button("↩️", key=f"snap_restore_{_coll_name}_{snap_name}",
                                         help="Восстановить из снапшота"):
                                res = restore_from_snapshot(_coll_name, snap_name)
                                if "error" in res:
                                    st.error(f"❌ {res['error']}")
                                else:
                                    st.success(f"✅ Восстановлено из {snap_name}")
                        with col_del:
                            if st.button("🗑️", key=f"snap_del_{_coll_name}_{snap_name}",
                                         help="Удалить снапшот"):
                                res = delete_snapshot(_coll_name, snap_name)
                                if "error" in res:
                                    st.error(f"❌ {res['error']}")
                                else:
                                    st.success(f"Удалён {snap_name}")

            st.divider()

    # Инструкция
    with st.expander("📖 Инструкция по восстановлению"):
        st.markdown("""
        **Если Qdrant не запускается после перезагрузки:**

        1. Запустите Docker: `docker compose up -d`
        2. Откройте эту вкладку → нажмите «📸 Снапшот» рядом с коллекцией
        3. Если коллекция повреждена — удалите её на вкладке «📊 Коллекции»
        4. Нажмите «↩️» рядом с нужным снапшотом для восстановления

        **Предотвращение:**
        - Перед выключением ПК: `python scripts_/shutdown_qdrant.py`
        - Или хотя бы: `docker stop -t 30 qdrant_rag`
        - Docker Compose настроен на 30 сек graceful shutdown (не убивайте процесс принудительно!)
        """)


# ══════════════════════════════════════════════════════════════════════
# Вкладка 6: Управление моделями
# ══════════════════════════════════════════════════════════════════════

with tab_models:
    st.subheader("🧩 Управление моделями эмбеддингов")
    st.markdown("""
    Здесь отображаются все доступные модели. Локальные модели нужно **скачать** перед первым использованием.
    API-модели (GigaChat, NeuroAPI) не требуют скачивания — нужен только ключ в secrets.
    """)

    models = get_all_models_status()

    # Счётчики
    cached_count = sum(1 for m in models if m["cached"])
    local_count = sum(1 for m in models if m["is_local"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего моделей", len(models))
    col2.metric("Локальных", local_count)
    col3.metric("Готово к работе", cached_count)

    st.divider()

    # Группировка по типу
    type_names = {
        "dense_text": "📝 Dense-модели (текст, таблицы, описания)",
        "sparse": "🔍 Sparse-модели (лексический поиск)",
    }
    for mtype, type_label in type_names.items():
        type_models = [m for m in models if m["type"] == mtype]
        if not type_models:
            continue

        st.markdown(f"### {type_label}")

        for m in type_models:
            with st.container():
                col_status, col_info, col_btn = st.columns([1, 4, 1])

                with col_status:
                    if m["cached"]:
                        st.success("✅ Готово")
                    elif not m["is_local"]:
                        # API модель — проверяем credentials
                        if m["provider"] == "gigachat":
                            has_cred = bool(get_secret("GIGACHAT_AUTH_TOKEN") or get_secret("GIGACHAT_SECRET"))
                        elif m["provider"] == "neuroapi":
                            has_cred = bool(get_secret("NEUROAPI_API_KEY"))
                        else:
                            has_cred = True
                        if has_cred:
                            st.success("✅ Готово")
                        else:
                            st.error("❌ Нет ключа")
                    else:
                        st.warning("⬇️ Не скачана")

                with col_info:
                    dim_str = f", **{m['dim']}d**" if m["dim"] else ""
                    st.markdown(f"**{m['name']}**`{m['model_id']}`{dim_str}")
                    st.caption(f"ℹ️ {m['description']}")
                    if m["is_local"]:
                        st.caption(f"📁 Провайдер: `{m['provider']}` | Размер: **{m['size_str']}**")
                    else:
                        st.caption(f"📡 Провайдер: `{m['provider']}` | Тип: API (размер не применим)")

                with col_btn:
                    if m["is_local"] and not m["cached"]:
                        if st.button(
                            "⬇️ Скачать",
                            key=f"dl_model_{m['key']}",
                            type="primary",
                        ):
                            with st.spinner(f"Скачивание {m['name']}... Это может занять несколько минут."):
                                try:
                                    msg = download_model(m["key"])
                                    st.success(msg)
                                    st.cache_resource.clear()
                                except Exception as e:
                                    st.error(f"❌ Ошибка при скачивании {m['name']}: {e}")
                                    logger.error("Ошибка скачивания модели %s: %s", m["key"], e, exc_info=True)
                            st.info("💡 Нажмите «🔄 Обновить статус моделей» для обновления")
                    elif not m["is_local"]:
                        # API модель — проверить ключ
                        if m["provider"] == "gigachat":
                            has_cred = bool(get_secret("GIGACHAT_AUTH_TOKEN") or get_secret("GIGACHAT_SECRET"))
                        elif m["provider"] == "neuroapi":
                            has_cred = bool(get_secret("NEUROAPI_API_KEY"))
                        else:
                            has_cred = True
                        if not has_cred:
                            st.caption("🔑 Задайте ключ в secrets")
                    elif m["is_local"] and m["cached"]:
                        st.caption("✓ В кэше")

                st.divider()

    # Кнопка обновления статуса
    if st.button("🔄 Обновить статус моделей", key="refresh_models_btn"):
        st.rerun()
