"""Страница AI-парсинга — обработка JPG страниц через VLM и YOLO.

Вкладки:
1. 🤖 Парсинг — 3-шаговый AI-пайплайн (VLM)
2. 🎯 Точечный парсинг — один JPG → результат в UI (для повторного парсинга)
3. 🎯 YOLO-детекция — обнаружение таблиц и изображений через YOLO
4. 👁️ Визуальный обзор — просмотр результатов YOLO с рамками
5. 🔧 Управление моделями — CRUD моделей VLM

Каждый шаг AI-пайплайна: выбор модели + промпта + температура.
YOLO-детекция: выбор модели, пороги, вырезка объектов.
"""

import json
import threading

import streamlit as st
from pathlib import Path

from src.llm_clients import (
    load_models,
    add_model,
    remove_model,
    check_provider_available,
)
from src.prompts import load_prompts, list_prompt_names, get_prompt_by_name
from src.run_page_parser import (
    run_page_parser,
    run_step1_only,
    run_step2_only,
    run_step3_only,
    _find_jpg_files,
    _find_parsed_md_files,
    _find_pages_with_crops,
)
from src.run_yolo_detector import run_yolo_detection, find_jpg_files as find_yolo_jpg_files, find_available_books


# ══════════════════════════════════════════════════════════════════════
# Кэширование YOLO модели
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource
def _cached_load_model(model_path: str):
    """Загрузить YOLO модель (кэшируется на уровне приложения)."""
    from src.yolo_detector import load_model
    return load_model(model_path)


@st.cache_data
def _cached_model_info(model_path: str) -> dict:
    """Информация о модели (клaccы, задача)."""
    from src.yolo_detector import get_model_info
    return get_model_info(model_path)


# ══════════════════════════════════════════════════════════════════════
# Вспомагательные функции UI
# ══════════════════════════════════════════════════════════════════════

def _get_model_options() -> list[str]:
    """Получить список моделей с отметкой провайдера."""
    models = load_models()
    options = []
    for m in models:
        options.append(f"{m['name']}  ({m['provider']})")
    return options


def _parse_model_option(option: str) -> str:
    """Извлечь имя модели из строки 'name  (provider)'."""
    return option.split("  (")[0].strip()


def _render_step_config(step_num: int, label: str, default_prompt: str) -> dict:
    """Отрисовать конфигурацию одного шага пайплайна.

    Returns:
        dict с ключами: model_name, prompt_text, temperature
    """
    st.markdown(f"**{label}**")

    model_options = _get_model_options()
    if not model_options:
        st.warning("⚠️ Нет моделей. Добавьте на вкладке «Управление моделями».")
        return {"model_name": "", "prompt_text": "", "temperature": 0.1}

    # Определяем индекс по умолчанию
    default_model_idx = st.session_state.get(
        f"parsing_step{step_num}_model_idx", 0
    )
    if default_model_idx >= len(model_options):
        default_model_idx = 0

    col_model, col_prompt = st.columns(2)

    with col_model:
        selected_model = st.selectbox(
            "Модель",
            options=model_options,
            index=default_model_idx,
            key=f"parsing_step{step_num}_model",
            help="Выберите модель из реестра.",
        )
        model_name = _parse_model_option(selected_model)

    prompt_names = list_prompt_names()
    # Добавляем пустой вариант
    prompt_options = ["— не выбран —"] + prompt_names

    # Определяем индекс промпта по умолчанию
    default_prompt_name = st.session_state.get(
        f"parsing_step{step_num}_prompt_name", default_prompt
    )
    default_prompt_idx = 0
    if default_prompt_name in prompt_names:
        default_prompt_idx = prompt_options.index(default_prompt_name)
    elif default_prompt_name == "— не выбран —":
        default_prompt_idx = 0

    with col_prompt:
        selected_prompt = st.selectbox(
            "Промпт",
            options=prompt_options,
            index=default_prompt_idx,
            key=f"parsing_step{step_num}_prompt",
            help="Выберите промпт из библиотеки.",
        )

    # Получаем текст промпта
    prompt_text = ""
    if selected_prompt != "— не выбран —":
        p = get_prompt_by_name(selected_prompt)
        if p:
            prompt_text = p["text"]

    temperature = st.slider(
        "Температура",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get(f"parsing_step{step_num}_temp", 0.1),
        step=0.05,
        key=f"parsing_step{step_num}_temperature",
        help="0.0 — детерминировано, 1.0 — креативно.",
    )

    return {
        "model_name": model_name,
        "prompt_text": prompt_text,
        "temperature": temperature,
    }


# ══════════════════════════════════════════════════════════════════════
# Страница
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Парсинг", page_icon="🤖", layout="wide")
st.title("🤖 AI-парсинг страниц")

st.markdown(
    "Обработка JPG-страниц: VLM-пайплайн (текст/BBox/описания), "
    "YOLO-детекция таблиц и изображений, визуальный обзор результатов."
)

# ── Вкладки ────────────────────────────────────────────────────────────
tab_parser, tab_single, tab_yolo, tab_visual, tab_models = st.tabs([
    "🤖 Парсинг (VLM)",
    "🎯 Точечный парсинг",
    "🎯 YOLO-детекция",
    "👁️ Визуальный обзор",
    "🔧 Управление моделями",
])

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Парсинг (VLM) — выбор режима (полный / отдельные шаги)
# ══════════════════════════════════════════════════════════════════════
with tab_parser:
    # ── Путь к базовой папке ────────────────────────────────────────
    base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("parsing_base_dir", "data/books"),
        key="parsing_base_dir",
        help="Структура: папка/название_книги/convert_in_jpg/*.jpg",
    )

    # ── Выбор книги ─────────────────────────────────────────────────
    _pbase_for_books = Path(base_dir) if base_dir else Path("data/books")
    _pavailable_books = find_available_books(_pbase_for_books) if _pbase_for_books.exists() else []

    parsing_book_name = None  # None = все книги
    if _pavailable_books:
        _pbook_options = ["📚 Все книги"] + _pavailable_books
        _pbook_selected = st.selectbox(
            "📖 Выберите книгу",
            options=_pbook_options,
            key="parsing_book_select",
            help="Выберите конкретную книгу или обработайте все.",
        )
        if _pbook_selected != "📚 Все книги":
            parsing_book_name = _pbook_selected
    elif _pbase_for_books.exists():
        st.caption("⚠️ В указанной папке нет книг с `convert_in_jpg/`.")

    # ── Режим запуска ───────────────────────────────────────────────
    st.divider()
    parsing_mode = st.radio(
        "🔄 Режим запуска",
        options=["full", "step1", "step2", "step3"],
        format_func=lambda x: {
            "full": "📦 Полный пайплайн (шаги 1→2→3)",
            "step1": "📝 Шаг 1: Извлечение текста",
            "step2": "🎯 Шаг 2: BBox + вырезка изображений",
            "step3": "🖼️ Шаг 3: Описание изображений + очистка тегов",
        }[x],
        key="parsing_mode",
        horizontal=True,
    )

    # ── Описание и требования для выбранного режима ─────────────────
    _mode_descriptions = {
        "full": "Все 3 шага выполняются для каждой страницы последовательно.",
        "step1": (
            "**Вход:** JPG-изображения страниц (`convert_in_jpg/*.jpg`)\n\n"
            "**Выход:** `parsed/{stem}.md` — Markdown с тегами `[IMAGE_N]`, `[CAPTION_N]`\n\n"
            "💡 Результаты понадобятся для запуска шагов 2 и 3."
        ),
        "step2": (
            "**Вход (от шага 1):**\n"
            "- `convert_in_jpg/*.jpg` — исходные изображения\n"
            "- `parsed/{stem}.md` — Markdown с тегами `[IMAGE_N]`\n\n"
            "**Выход:**\n"
            "- `parsed/bboxes/{stem}.json` — BBox координаты\n"
            "- `parsed/images/{stem}_image_N.jpg` — вырезанные картинки\n\n"
            "⚠️ Сначала запустите шаг 1!"
        ),
        "step3": (
            "**Вход (от шагов 1 и 2):**\n"
            "- `parsed/{stem}.md` — Markdown с тегами `[IMAGE_N]` (шаг 1)\n"
            "- `parsed/bboxes/{stem}.json` — BBox координаты (шаг 2)\n"
            "- `parsed/images/{stem}_image_N.jpg` — вырезанные картинки (шаг 2)\n\n"
            "**Выход:**\n"
            "- `parsed/{stem}.md` — перезаписан, теги удалены\n"
            "- `parsed/images_descriptions.json` — AI-описания\n\n"
            "⚠️ Сначала запустите шаги 1 и 2!"
        ),
    }
    st.markdown(_mode_descriptions[parsing_mode])

    # ── Настройки шагов (показываем только нужные) ──────────────────
    if parsing_mode == "full":
        with st.expander("⚙️ Настройки шагов", expanded=True):
            st.markdown("---")
            step1 = _render_step_config(1, "📝 Шаг 1: Извлечение текста", "PROMPT_TEXT_EXTRACTION")
            st.markdown("---")
            step2 = _render_step_config(2, "🎯 Шаг 2: Поиск BBox", "PROMPT_GET_BBOXES")
            st.markdown("---")
            step3 = _render_step_config(3, "🖼️ Шаг 3: Описание изображений", "PROMPT_IMAGE_DESCRIPTION")

    elif parsing_mode == "step1":
        with st.expander("⚙️ Настройки шага 1", expanded=True):
            step1 = _render_step_config(1, "📝 Шаг 1: Извлечение текста", "PROMPT_TEXT_EXTRACTION")
        step2 = {"model_name": "", "prompt_text": "", "temperature": 0.0}
        step3 = {"model_name": "", "prompt_text": "", "temperature": 0.1}

    elif parsing_mode == "step2":
        with st.expander("⚙️ Настройки шага 2", expanded=True):
            step2 = _render_step_config(2, "🎯 Шаг 2: Поиск BBox", "PROMPT_GET_BBOXES")
        step1 = {"model_name": "", "prompt_text": "", "temperature": 0.1}
        step3 = {"model_name": "", "prompt_text": "", "temperature": 0.1}

    elif parsing_mode == "step3":
        with st.expander("⚙️ Настройки шага 3", expanded=True):
            step3 = _render_step_config(3, "🖼️ Шаг 3: Описание изображений", "PROMPT_IMAGE_DESCRIPTION")
        step1 = {"model_name": "", "prompt_text": "", "temperature": 0.1}
        step2 = {"model_name": "", "prompt_text": "", "temperature": 0.0}

    # ── Расширенные настройки ───────────────────────────────────────
    with st.expander("🔧 Расширенные настройки", expanded=False):
        batch_size = st.number_input(
            "Размер батча (сохранение прогресса)",
            min_value=1,
            max_value=100,
            value=st.session_state.get("parsing_batch_size", 10),
            key="parsing_batch_size",
            help="Каждые N файлов прогресс сохраняется.",
        )

    # ── Предпросмотр файлов ─────────────────────────────────────────
    st.divider()
    _base = Path(base_dir) if base_dir else Path("data/books")

    if parsing_mode in ("full", "step1"):
        # Показываем JPG файлы
        if _base.exists():
            _preview_files = _find_jpg_files(_base, book_name=parsing_book_name)
            if _preview_files:
                _book_names = sorted(set(b for b, _, _ in _preview_files))
                _book_label = parsing_book_name or f"{len(_book_names)} книг"
                st.info(
                    f"📁 Будет обработано: **{len(_preview_files)}** JPG файлов "
                    f"из **{_book_label}**"
                )
            else:
                st.warning("В папке не найдено `convert_in_jpg/`. Сначала выполните конвертацию PDF → JPG.")
        else:
            st.error(f"❌ Папка не найдена: `{base_dir}`")

    elif parsing_mode == "step2":
        # Показываем .md файлы в parsed/
        if _base.exists():
            _preview_mds = _find_parsed_md_files(_base, book_name=parsing_book_name)
            if _preview_mds:
                _book_names = sorted(set(b for b, _, _ in _preview_mds))
                _book_label = parsing_book_name or f"{len(_book_names)} книг"
                st.info(
                    f"📁 Найдено: **{len(_preview_mds)}** .md файлов в `parsed/` "
                    f"из **{_book_label}**"
                )
            else:
                st.warning("Нет `.md` файлов в `parsed/`. Сначала запустите шаг 1.")
        else:
            st.error(f"❌ Папка не найдена: `{base_dir}`")

    elif parsing_mode == "step3":
        # Показываем страницы с вырезанными картинками
        if _base.exists():
            _preview_crops = _find_pages_with_crops(_base, book_name=parsing_book_name)
            if _preview_crops:
                _book_names = sorted(set(b for b, _, _, _ in _preview_crops))
                _book_label = parsing_book_name or f"{len(_book_names)} книг"
                st.info(
                    f"📁 Найдено: **{len(_preview_crops)}** страниц с BBox и вырезками "
                    f"из **{_book_label}**"
                )
            else:
                st.warning("Нет страниц с вырезанными картинками. Сначала запустите шаги 1 и 2.")
        else:
            st.error(f"❌ Папка не найдена: `{base_dir}`")

    # ── Валидация ───────────────────────────────────────────────────
    _validation_errors = []
    if parsing_mode in ("full", "step1"):
        if not step1["model_name"]:
            _validation_errors.append("Шаг 1: не выбрана модель")
        if not step1["prompt_text"]:
            _validation_errors.append("Шаг 1: не выбран промпт")
    if parsing_mode in ("full", "step2"):
        if not step2["model_name"]:
            _validation_errors.append("Шаг 2: не выбрана модель")
        if not step2["prompt_text"]:
            _validation_errors.append("Шаг 2: не выбран промпт")
    if parsing_mode in ("full", "step3"):
        if not step3["model_name"]:
            _validation_errors.append("Шаг 3: не выбрана модель")
        if not step3["prompt_text"]:
            _validation_errors.append("Шаг 3: не выбран промпт")
    if _validation_errors:
        for err in _validation_errors:
            st.warning(f"⚠️ {err}")

    # ── Инициализация состояния прогресса ───────────────────────────
    if "parsing_thread" not in st.session_state:
        st.session_state.parsing_thread = None
    if "parsing_cancel" not in st.session_state:
        st.session_state.parsing_cancel = None
    if "parsing_progress" not in st.session_state:
        st.session_state.parsing_progress = None
    if "parsing_result" not in st.session_state:
        st.session_state.parsing_result = None

    @st.fragment(run_every="2s")
    def _render_parsing_progress():
        """Автообновляемый фрагмент: прогресс парсинга + кнопка отмены."""
        _thread = st.session_state.parsing_thread
        _is_running = _thread is not None and _thread.is_alive()

        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.parsing_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0

            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"✅ Успешно: **{_progress.get('success', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**  |  "
                    f"⏩ Пропущено: **{_progress.get('skipped', 0)}**"
                )
                st.write(f"🔢 Токенов потрачено: **{_progress.get('total_tokens', 0)}**")

        if st.button(
            "❌ Отменить парсинг",
            key="parsing_cancel_btn",
            width="stretch",
            type="secondary",
        ):
            if st.session_state.parsing_cancel:
                st.session_state.parsing_cancel.set()
            st.rerun()


    with tab_parser:
        _thread = st.session_state.parsing_thread
        _is_running = _thread is not None and _thread.is_alive()

        if _is_running:
            _render_parsing_progress()

        elif st.session_state.parsing_result:
            _res = st.session_state.parsing_result
            _mode_label = st.session_state.get("parsing_result_mode", "парсинг")
            if _res.get("cancelled"):
                st.warning(
                    f"⚠️ {_mode_label} отменён. "
                    f"Успешно: {_res.get('success', 0)}/{_res.get('total', 0)}, "
                    f"Ошибок: {_res.get('errors', 0)}, "
                    f"Пропущено: {_res.get('skipped', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с. "
                    f"Токенов: {_res.get('total_tokens', 0)}"
                )
            elif _res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ {_mode_label} завершён с ошибками. "
                    f"Успешно: {_res.get('success', 0)}/{_res.get('total', 0)}, "
                    f"Ошибок: {_res.get('errors', 0)}, "
                    f"Пропущено: {_res.get('skipped', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с. "
                    f"Токенов: {_res.get('total_tokens', 0)}"
                )
            else:
                st.success(
                    f"✅ {_mode_label} завершён! "
                    f"Обработано: {_res.get('success', 0)} файлов, "
                    f"Пропущено: {_res.get('skipped', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с. "
                    f"Токенов: {_res.get('total_tokens', 0)}"
                )
            if st.button("🗑️ Сбросить результат", key="parsing_clear_result"):
                st.session_state.parsing_result = None
                st.session_state.parsing_progress = None
                st.rerun()

        else:
            _can_run = (
                base_dir
                and Path(base_dir).exists()
                and not _validation_errors
            )

            # Кнопки запуска
            _btn_labels = {
                "full": "📦 Запустить полный пайплайн",
                "step1": "📝 Запустить шаг 1 (извлечение текста)",
                "step2": "🎯 Запустить шаг 2 (BBox + вырезка)",
                "step3": "🖼️ Запустить шаг 3 (описание + очистка)",
            }

            if _can_run:
                if st.button(
                    _btn_labels[parsing_mode],
                    key="parsing_run_btn",
                    width="stretch",
                    type="primary",
                ):
                    cancel_event = threading.Event()
                    st.session_state.parsing_cancel = cancel_event
                    st.session_state.parsing_progress = {
                        "current": 0,
                        "total": 0,
                        "current_file": "подготовка...",
                        "success": 0,
                        "errors": 0,
                        "skipped": 0,
                        "total_tokens": 0,
                    }
                    st.session_state.parsing_result = None
                    st.session_state.parsing_result_mode = _btn_labels[parsing_mode].split(" ", 1)[1].split("(")[0].strip()

                    def _progress_cb(current, total, current_file, stats):
                        st.session_state.parsing_progress = {
                            "current": current,
                            "total": total,
                            "current_file": current_file,
                            "success": stats["success"],
                            "errors": stats["errors"],
                            "skipped": stats.get("skipped", 0),
                            "total_tokens": stats.get("total_tokens", 0),
                        }

                    def _run():
                        if parsing_mode == "full":
                            result = run_page_parser(
                                base_dir=base_dir,
                                book_name=parsing_book_name,
                                step1_model=step1["model_name"],
                                step1_prompt=step1["prompt_text"],
                                step1_temperature=step1["temperature"],
                                step2_model=step2["model_name"],
                                step2_prompt=step2["prompt_text"],
                                step2_temperature=step2["temperature"],
                                step3_model=step3["model_name"],
                                step3_prompt=step3["prompt_text"],
                                step3_temperature=step3["temperature"],
                                batch_size=batch_size,
                                cancel_event=cancel_event,
                                progress_callback=_progress_cb,
                            )
                        elif parsing_mode == "step1":
                            result = run_step1_only(
                                base_dir=base_dir,
                                book_name=parsing_book_name,
                                model_name=step1["model_name"],
                                prompt_text=step1["prompt_text"],
                                temperature=step1["temperature"],
                                batch_size=batch_size,
                                cancel_event=cancel_event,
                                progress_callback=_progress_cb,
                            )
                        elif parsing_mode == "step2":
                            result = run_step2_only(
                                base_dir=base_dir,
                                book_name=parsing_book_name,
                                model_name=step2["model_name"],
                                prompt_text=step2["prompt_text"],
                                temperature=step2["temperature"],
                                batch_size=batch_size,
                                cancel_event=cancel_event,
                                progress_callback=_progress_cb,
                            )
                        elif parsing_mode == "step3":
                            result = run_step3_only(
                                base_dir=base_dir,
                                book_name=parsing_book_name,
                                model_name=step3["model_name"],
                                prompt_text=step3["prompt_text"],
                                temperature=step3["temperature"],
                                batch_size=batch_size,
                                cancel_event=cancel_event,
                                progress_callback=_progress_cb,
                            )
                        st.session_state.parsing_result = result

                    thread = threading.Thread(target=_run, daemon=True)
                    st.session_state.parsing_thread = thread
                    thread.start()
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════
# Вкладка: Точечный парсинг (один JPG → результат в UI)
# ══════════════════════════════════════════════════════════════════════
with tab_single:
    st.subheader("🎯 Точечный парсинг одной страницы")
    st.markdown(
        "Выберите JPG-файл, модель и промпт — результат появится прямо здесь. "
        "Удобно для повторного парсинга страницы с ошибкой."
    )

    # ── Выбор файла ─────────────────────────────────────────────────
    col_file1, col_file2 = st.columns(2)

    with col_file1:
        single_file_path = st.text_input(
            "📂 Путь к JPG файлу",
            value=st.session_state.get("single_file_path", ""),
            key="single_file_path",
            placeholder="data/books/Книга/convert_in_jpg/001.jpg",
        )

    with col_file2:
        single_uploaded = st.file_uploader(
            "... или загрузите файл",
            type=["jpg", "jpeg", "png"],
            key="single_file_upload",
            help="Альтернатива: загрузить файл через браузер.",
        )

    # Предпросмотр изображения
    _single_img_bytes = None
    if single_uploaded is not None:
        _single_img_bytes = single_uploaded.read()
    elif single_file_path and Path(single_file_path).exists():
        _single_img_bytes = Path(single_file_path).read_bytes()

    if _single_img_bytes:
        st.image(_single_img_bytes, caption="Предпросмотр", width=400)

    # ── Настройки ───────────────────────────────────────────────────
    st.divider()
    col_model, col_prompt = st.columns(2)

    with col_model:
        _model_options = _get_model_options()
        if _model_options:
            _single_model_idx = st.session_state.get("single_model_idx", 0)
            if _single_model_idx >= len(_model_options):
                _single_model_idx = 0
            _single_model_sel = st.selectbox(
                "🤖 Модель",
                options=_model_options,
                index=_single_model_idx,
                key="single_model",
            )
            _single_model_name = _parse_model_option(_single_model_sel)
        else:
            st.warning("⚠️ Нет моделей. Добавьте на вкладке «Управление моделями».")
            _single_model_name = ""

    with col_prompt:
        _prompt_names = list_prompt_names()
        _prompt_options = ["✏️ Свой промпт"] + _prompt_names
        _single_prompt_sel = st.selectbox(
            "📝 Промпт",
            options=_prompt_options,
            key="single_prompt_select",
        )

        # Текст промпта: из библиотеки или ручной ввод
        if _single_prompt_sel == "✏️ Свой промпт":
            _single_prompt_text = st.text_area(
                "Текст промпта",
                value=st.session_state.get("single_prompt_manual", ""),
                key="single_prompt_manual",
                height=150,
                placeholder="Введите промпт для парсинга страницы...",
            )
        else:
            _p = get_prompt_by_name(_single_prompt_sel)
            _single_prompt_text = st.text_area(
                "Текст промпта (можно отредактировать)",
                value=_p["text"] if _p else "",
                key="single_prompt_text_edit",
                height=150,
            )

    _single_temperature = st.slider(
        "🌡️ Температура",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        key="single_temperature",
    )

    # ── Запуск ──────────────────────────────────────────────────────
    st.divider()

    _can_run_single = (
        _single_img_bytes is not None
        and _single_model_name
        and _single_prompt_text.strip()
    )

    if not _can_run_single and _single_img_bytes is None and not single_file_path:
        st.info("👆 Укажите путь к JPG или загрузите файл.")
    elif _single_img_bytes is None:
        st.warning("⚠️ Файл не найден.")
    elif not _single_model_name:
        st.warning("⚠️ Выберите модель.")
    elif not _single_prompt_text.strip():
        st.warning("⚠️ Введите или выберите промпт.")

    if _can_run_single:
        if st.button(
            "🚀 Запустить парсинг",
            key="single_run_btn",
            type="primary",
            width="stretch",
        ):
            from src.page_parser import encode_image_bytes, step1_extract_text

            with st.spinner("⏳ Парсинг..."):
                try:
                    _b64 = encode_image_bytes(_single_img_bytes)
                    _result = step1_extract_text(
                        image_b64=_b64,
                        model_name=_single_model_name,
                        prompt_text=_single_prompt_text,
                        temperature=_single_temperature,
                    )
                    st.session_state["single_result"] = _result
                except Exception as e:
                    st.session_state["single_error"] = str(e)
            st.rerun()

    # ── Результат ───────────────────────────────────────────────────
    if st.session_state.get("single_error"):
        st.error(f"❌ Ошибка: {st.session_state.single_error}")
        if st.button("🗑️ Сбросить", key="single_clear_error"):
            st.session_state.pop("single_error", None)
            st.rerun()

    elif st.session_state.get("single_result"):
        _res = st.session_state.single_result

        st.success(
            f"✅ Готово за {_res.get('total_tokens', 0)} токенов "
            f"(prompt: {_res.get('prompt_tokens', 0)}, "
            f"completion: {_res.get('completion_tokens', 0)})"
        )

        # Результат в text_area для удобного копирования
        st.markdown("**📋 Результат (выделите и скопируйте):**")
        st.text_area(
            "Результат парсинга",
            value=_res.get("content", ""),
            height=500,
            key="single_result_output",
            disabled=False,
        )

        if st.button("🗑️ Сбросить результат", key="single_clear_result"):
            st.session_state.pop("single_result", None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════
# Вкладка: YOLO-детекция
# ══════════════════════════════════════════════════════════════════════
with tab_yolo:
    st.subheader("🎯 YOLO-детекция таблиц и изображений")
    st.markdown(
        "Автоматическое обнаружение таблиц и изображений на страницах "
        "с помощью предобученной YOLO модели. "
        "Результаты: аннотированные страницы, вырезанные объекты, JSON с координатами."
    )

    # ── Настройки модели ────────────────────────────────────────────
    col_model, col_params = st.columns(2)

    with col_model:
        yolo_model_path = st.text_input(
            "📦 Путь к модели YOLO (.pt)",
            value=st.session_state.get("yolo_model_path", "models/yolo/book_parsing_n/weights/best.pt"),
            key="yolo_model_path",
            help="Путь к весам обученной YOLO модели.",
        )

    with col_params:
        yolo_base_dir = st.text_input(
            "📂 Папка с книгами",
            value=st.session_state.get("yolo_base_dir", "data/books"),
            key="yolo_base_dir",
            help="Структура: папка/название_книги/convert_in_jpg/*.jpg",
        )

    # ── Выбор книги ─────────────────────────────────────────────────
    _ybase_for_books = Path(yolo_base_dir) if yolo_base_dir else Path("data/books")
    _yavailable_books = find_available_books(_ybase_for_books) if _ybase_for_books.exists() else []

    yolo_book_name = None  # None = все книги
    if _yavailable_books:
        _ybook_options = ["📚 Все книги"] + _yavailable_books
        _ybook_selected = st.selectbox(
            "📖 Выберите книгу",
            options=_ybook_options,
            key="yolo_book_select",
            help="Выберите конкретную книгу или обработайте все.",
        )
        if _ybook_selected != "📚 Все книги":
            yolo_book_name = _ybook_selected
    elif _ybase_for_books.exists():
        st.caption("⚠️ В указанной папке нет книг с `convert_in_jpg/`.")

    # ── Пороги ──────────────────────────────────────────────────────
    with st.expander("⚙️ Настройки детекции", expanded=True):
        col_conf, col_iou, col_pad = st.columns(3)

        with col_conf:
            yolo_conf = st.slider(
                "Порог Confidence",
                min_value=0.05,
                max_value=0.95,
                value=st.session_state.get("yolo_conf", 0.25),
                step=0.05,
                key="yolo_conf",
                help="Минимальная уверенность для детекции.",
            )

        with col_iou:
            yolo_iou = st.slider(
                "Порог IoU (NMS)",
                min_value=0.1,
                max_value=0.95,
                value=st.session_state.get("yolo_iou", 0.7),
                step=0.05,
                key="yolo_iou",
                help="Порог IoU для подавления дубликатов.",
            )

        with col_pad:
            yolo_padding = st.slider(
                "Отступ при вырезке (px)",
                min_value=0,
                max_value=30,
                value=st.session_state.get("yolo_padding", 5),
                key="yolo_padding",
                help="Отступ вокруг рамки при вырезке объекта.",
            )

        yolo_skip = st.checkbox(
            "Пропускать уже обработанные страницы",
            value=True,
            key="yolo_skip_existing",
            help="Страницы с результатами в detections.json будут пропущены.",
        )

    # ── Инфо о модели ───────────────────────────────────────────────
    st.divider()

    _model_path_obj = Path(yolo_model_path)
    if _model_path_obj.exists():
        info = _cached_model_info(yolo_model_path)
        classes = info.get("classes", {})
        classes_str = ", ".join(f"{v} (id={k})" for k, v in classes.items())
        st.caption(f"📋 Классы модели: {classes_str}")
    else:
        st.error(f"❌ Файл модели не найден: `{yolo_model_path}`")

    # ── Предпросмотр ────────────────────────────────────────────────
    if yolo_base_dir:
        _ybase = Path(yolo_base_dir)
        if _ybase.exists():
            _yfiles = find_yolo_jpg_files(_ybase, book_name=yolo_book_name)
            if _yfiles:
                _ybook_names = sorted(set(b for b, _ in _yfiles))
                _book_label = yolo_book_name if yolo_book_name else f"{len(_ybook_names)} книг"
                st.info(
                    f"Будет обработано: **{len(_yfiles)}** JPG файлов "
                    f"из **{_book_label}**"
                )
            else:
                st.warning(
                    f"В `{yolo_base_dir}` не найдено папок с `convert_in_jpg/`."
                )
        else:
            st.error(f"❌ Папка не найдена: `{yolo_base_dir}`")

    # ── Инициализация состояния YOLO ────────────────────────────────
    if "yolo_thread" not in st.session_state:
        st.session_state.yolo_thread = None
    if "yolo_cancel" not in st.session_state:
        st.session_state.yolo_cancel = None
    if "yolo_progress" not in st.session_state:
        st.session_state.yolo_progress = None
    if "yolo_result" not in st.session_state:
        st.session_state.yolo_result = None


    @st.fragment(run_every="2s")
    def _render_yolo_progress():
        """Автообновляемый фрагмент: прогресс YOLO + кнопка отмены."""
        _thread = st.session_state.yolo_thread
        _is_running = _thread is not None and _thread.is_alive()

        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.yolo_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0

            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"✅ Успешно: **{_progress.get('success', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**"
                )
                st.write(
                    f"📊 Таблиц: **{_progress.get('tables', 0)}**  |  "
                    f"🖼️ Фигур: **{_progress.get('figures', 0)}**"
                )

        if st.button(
            "❌ Отменить детекцию",
            key="yolo_cancel_btn",
            width="stretch",
            type="secondary",
        ):
            if st.session_state.yolo_cancel:
                st.session_state.yolo_cancel.set()
            st.rerun()


    with tab_yolo:
        _ythread = st.session_state.yolo_thread
        _yis_running = _ythread is not None and _ythread.is_alive()

        if _yis_running:
            _render_yolo_progress()

        elif st.session_state.yolo_result:
            _yres = st.session_state.yolo_result
            if _yres.get("cancelled"):
                st.warning(
                    f"⚠️ Детекция отменена. "
                    f"Успешно: {_yres.get('success', 0)}/{_yres.get('total', 0)}, "
                    f"Ошибок: {_yres.get('errors', 0)}. "
                    f"Время: {_yres.get('elapsed_sec', '?')}с"
                )
            elif _yres.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Успешно: {_yres.get('success', 0)}/{_yres.get('total', 0)}, "
                    f"Ошибок: {_yres.get('errors', 0)}. "
                    f"Время: {_yres.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Детекция завершена! "
                    f"Обработано: {_yres.get('success', 0)} файлов. "
                    f"Таблиц: {_yres.get('tables', 0)}, Фигур: {_yres.get('figures', 0)}. "
                    f"Время: {_yres.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="yolo_clear_result"):
                st.session_state.yolo_result = None
                st.session_state.yolo_progress = None
                st.rerun()

        else:
            _ycan_run = (
                yolo_base_dir
                and Path(yolo_base_dir).exists()
                and _model_path_obj.exists()
            )
            if _ycan_run:
                if st.button(
                    "🎯 Запустить YOLO-детекцию",
                    key="yolo_run_btn",
                    width="stretch",
                    type="primary",
                ):
                    cancel_event = threading.Event()
                    st.session_state.yolo_cancel = cancel_event
                    st.session_state.yolo_progress = {
                        "current": 0,
                        "total": 0,
                        "current_file": "загрузка модели...",
                        "success": 0,
                        "errors": 0,
                        "tables": 0,
                        "figures": 0,
                    }
                    st.session_state.yolo_result = None

                    def _yolo_progress_cb(current, total, current_file, stats):
                        st.session_state.yolo_progress = {
                            "current": current,
                            "total": total,
                            "current_file": current_file,
                            "success": stats.get("success", 0),
                            "errors": stats.get("errors", 0),
                            "tables": stats.get("tables", 0),
                            "figures": stats.get("figures", 0),
                        }

                    def _yolo_run():
                        result = run_yolo_detection(
                            base_dir=yolo_base_dir,
                            model_path=yolo_model_path,
                            book_name=yolo_book_name,
                            conf=yolo_conf,
                            iou=yolo_iou,
                            padding=yolo_padding,
                            skip_existing=yolo_skip,
                            cancel_event=cancel_event,
                            progress_callback=_yolo_progress_cb,
                        )
                        st.session_state.yolo_result = result

                    thread = threading.Thread(target=_yolo_run, daemon=True)
                    st.session_state.yolo_thread = thread
                    thread.start()
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════
# Вкладка: Визуальный обзор YOLO-результатов
# ══════════════════════════════════════════════════════════════════════
with tab_visual:
    st.subheader("👁️ Визуальный обзор YOLO-результатов")
    st.markdown(
        "Просмотр страниц с нарисованными рамками детекций. "
        "Оригинал и аннотированное изображение рядом для сравнения."
    )

    # ── Выбор книги ─────────────────────────────────────────────────
    _vis_base = Path(st.session_state.get("yolo_base_dir", "data/books"))

    # Ищем книги с результатами YOLO
    _vis_books = []
    if _vis_base.exists():
        for bd in sorted(_vis_base.iterdir()):
            if bd.is_dir() and (bd / "yolo_detected" / "detections.json").exists():
                _vis_books.append(bd.name)

    if not _vis_books:
        st.info("📭 Нет книг с результатами YOLO-детекции. Сначала запустите детекцию.")
    else:
        _selected_book = st.selectbox(
            "📚 Выберите книгу",
            options=_vis_books,
            key="vis_book_select",
        )

        if _selected_book:
            _vis_book_dir = _vis_base / _selected_book
            _vis_det_dir = _vis_book_dir / "yolo_detected"
            _vis_json = _vis_det_dir / "detections.json"
            _vis_annotated_dir = _vis_det_dir / "annotated"
            _vis_jpg_dir = _vis_book_dir / "convert_in_jpg"

            # Загружаем detections.json
            with open(_vis_json, "r", encoding="utf-8") as f:
                _vis_data = json.load(f)

            _vis_pages = sorted(_vis_data.get("pages", {}).keys())

            if not _vis_pages:
                st.warning("Нет обработанных страниц.")
            else:
                # Статистика
                _total_dets = sum(
                    len(p.get("detections", []))
                    for p in _vis_data["pages"].values()
                )
                _tables_count = sum(
                    1 for p in _vis_data["pages"].values()
                    for d in p.get("detections", [])
                    if d.get("class_name") == "table"
                )
                _figures_count = _total_dets - _tables_count

                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("📄 Страниц", len(_vis_pages))
                col_s2.metric("📊 Таблиц", _tables_count)
                col_s3.metric("🖼️ Фигур", _figures_count)

                st.divider()

                # ── Навигация по страницам ──────────────────────────
                # Инициализация текущего индекса
                _vis_max_idx = len(_vis_pages) - 1
                if "vis_current_idx" not in st.session_state:
                    st.session_state.vis_current_idx = 0
                # Сброс при выходе за границы (смена книги)
                if st.session_state.vis_current_idx > _vis_max_idx:
                    st.session_state.vis_current_idx = 0

                # Кнопки перелистывания
                _nav_col1, _nav_col2, _nav_col3 = st.columns([1, 3, 1])

                with _nav_col1:
                    _prev_disabled = st.session_state.vis_current_idx <= 0
                    if st.button(
                        "◀️ Назад",
                        disabled=_prev_disabled,
                        key="vis_prev_btn",
                        width="stretch",
                    ):
                        st.session_state.vis_current_idx -= 1
                        st.rerun()

                with _nav_col2:
                    st.markdown(
                        f"<div style='text-align: center; font-size: 1.1em; padding-top: 4px;'>"
                        f"📄 Стр. <b>{_vis_pages[st.session_state.vis_current_idx]}</b> "
                        f"&nbsp;|&nbsp; {st.session_state.vis_current_idx + 1} / {len(_vis_pages)}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                with _nav_col3:
                    _next_disabled = st.session_state.vis_current_idx >= _vis_max_idx
                    if st.button(
                        "Вперёд ▶️",
                        disabled=_next_disabled,
                        key="vis_next_btn",
                        width="stretch",
                    ):
                        st.session_state.vis_current_idx += 1
                        st.rerun()

                # Выбор по номеру (прямой переход)
                _vis_page_idx = st.selectbox(
                    "📄 Перейти к странице",
                    options=range(len(_vis_pages)),
                    format_func=lambda i: f"Стр. {_vis_pages[i]}",
                    index=st.session_state.vis_current_idx,
                    key="vis_page_idx",
                )
                # Синхронизация: selectbox → session_state
                if _vis_page_idx != st.session_state.vis_current_idx:
                    st.session_state.vis_current_idx = _vis_page_idx
                    st.rerun()

                _vis_stem = _vis_pages[st.session_state.vis_current_idx]
                _vis_page_data = _vis_data["pages"][_vis_stem]

                # ── Статус детекций на странице ──────────────────────
                _detections = _vis_page_data.get("detections", [])
                if _detections:
                    st.success(
                        f"🔍 На странице **{_vis_stem}** обнаружено "
                        f"**{len(_detections)}** объектов"
                    )
                else:
                    st.info(f"На странице {_vis_stem} объектов не обнаружено.")

                # ── Оригинал и аннотация рядом ──────────────────────
                col_orig, col_annot = st.columns(2)

                with col_orig:
                    st.markdown("**📄 Оригинал**")
                    _orig_path = _vis_jpg_dir / f"{_vis_stem}.jpg"
                    if _orig_path.exists():
                        st.image(str(_orig_path), width="stretch")
                    else:
                        st.warning(f"Файл не найден: {_orig_path}")

                with col_annot:
                    st.markdown("**🎯 Аннотация (рамки YOLO)**")
                    _annot_path = _vis_annotated_dir / f"{_vis_stem}.jpg"
                    if _annot_path.exists():
                        st.image(str(_annot_path), width="stretch")
                    else:
                        st.warning("Аннотация не найдена. Запустите YOLO-детекцию.")

                # ── Детали детекций ─────────────────────────────────
                if _detections:
                    st.markdown(f"**Детекции на странице {_vis_stem}** ({len(_detections)} шт.)")

                    _det_rows = []
                    for i, det in enumerate(_detections):
                        _det_rows.append({
                            "#": i + 1,
                            "Класс": det.get("class_name", "?"),
                            "Confidence": f"{det.get('confidence', 0):.3f}",
                            "BBox (x1,y1,x2,y2)": str(det.get("bbox_xyxy", [])),
                        })
                    st.dataframe(_det_rows, width="stretch", hide_index=True)

                    # ── Превью вырезок ───────────────────────────────
                    _tables_dir = _vis_det_dir / "tables"
                    _figures_dir = _vis_det_dir / "figures"

                    _page_tables = sorted(_tables_dir.glob(f"{_vis_stem}_*.jpg")) if _tables_dir.exists() else []
                    _page_figures = sorted(_figures_dir.glob(f"{_vis_stem}_*.jpg")) if _figures_dir.exists() else []

                    if _page_tables:
                        st.markdown(f"**📊 Вырезанные таблицы** ({len(_page_tables)})")
                        _table_cols = st.columns(min(len(_page_tables), 4))
                        for idx, tpath in enumerate(_page_tables):
                            with _table_cols[idx % len(_table_cols)]:
                                st.image(str(tpath), caption=tpath.name, width="stretch")

                    if _page_figures:
                        st.markdown(f"**🖼️ Вырезанные фигуры** ({len(_page_figures)})")
                        _fig_cols = st.columns(min(len(_page_figures), 4))
                        for idx, fpath in enumerate(_page_figures):
                            with _fig_cols[idx % len(_fig_cols)]:
                                st.image(str(fpath), caption=fpath.name, width="stretch")


# ══════════════════════════════════════════════════════════════════════
# Вкладка: Управление моделями
# ══════════════════════════════════════════════════════════════════════
with tab_models:
    st.subheader("🔧 Управление моделями")
    st.markdown(
        "Добавляйте и удаляйте модели из реестра. "
        "Провайдеры: `neuroapi` (NeuroAPI), `zai` (Z.ai/GLM), `gigachat` (Сбер — заглушка)."
    )

    models = load_models()

    # ── Текущие модели ──────────────────────────────────────────────
    if models:
        st.markdown("**Зарегистрированные модели:**")
        for m in models:
            col_name, col_provider, col_status, col_btn = st.columns([3, 2, 2, 1])

            with col_name:
                st.code(m["name"])
            with col_provider:
                st.text(m["provider"])
            with col_status:
                avail, msg = check_provider_available(m["provider"])
                if avail:
                    st.caption(f"✅ {msg}")
                else:
                    st.caption(f"❌ {msg}")
            with col_btn:
                if st.button("🗑️", key=f"del_model_{m['name']}", help=f"Удалить {m['name']}"):
                    try:
                        remove_model(m["name"])
                        st.success(f"🗑️ Модель «{m['name']}» удалена.")
                        st.rerun()
                    except ValueError as e:
                        st.error(f"❌ {e}")

    else:
        st.info("📭 Нет зарегистрированных моделей.")

    # ── Добавление модели ───────────────────────────────────────────
    st.divider()
    st.markdown("**➕ Добавить модель**")

    with st.form("add_model_form", clear_on_submit=True):
        new_model_name = st.text_input(
            "Имя модели *",
            key="new_model_name",
            placeholder="Напр.: gemini-2.5-flash",
            help="Точное название модели, как оно требуется в API.",
        )
        new_model_provider = st.selectbox(
            "Провайдер *",
            options=["neuroapi", "zai", "gigachat"],
            key="new_model_provider",
            help=(
                "neuroapi — NeuroAPI (OpenAI-совместимый API). "
                "zai — Z.ai (GLM модели). "
                "gigachat — Сбер GigaChat (заглушка)."
            ),
        )

        submitted = st.form_submit_button("💾 Добавить модель", type="primary")
        if submitted:
            if not new_model_name.strip():
                st.error("❌ Имя модели обязательно.")
            else:
                try:
                    add_model(new_model_name, new_model_provider)
                    st.success(f"✅ Модель «{new_model_name}» добавлена!")
                    st.rerun()
                except ValueError as e:
                    st.error(f"❌ {e}")

    # ── Статус провайдеров ──────────────────────────────────────────
    st.divider()
    st.markdown("**📡 Статус провайдеров**")

    for provider_name in ["neuroapi", "zai", "gigachat"]:
        avail, msg = check_provider_available(provider_name)
        if avail:
            st.caption(f"✅ **{provider_name}**: {msg}")
        else:
            st.caption(f"❌ **{provider_name}**: {msg}")