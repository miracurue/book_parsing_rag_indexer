"""Страница загрузки статей с arXiv.org.

Две вкладки:
1. 🔍 Поиск — ввод запроса, просмотр результатов, выбор статей
2. 📥 Загрузка — скачивание PDF + метаданные в XLSX по подпапкам
"""

import threading

import arxiv
import streamlit as st
from pathlib import Path

from src.arxiv_searcher import search_articles, ArxivArticle
from src.arxiv_downloader import (
    run_download_articles,
    get_existing_subfolders,
    get_subfolder_stats,
    load_downloaded_ids,
    XLSX_FILENAME,
)

st.set_page_config(page_title="Загрузчик arXiv", page_icon="📥", layout="wide")
st.title("📥 Загрузчик статей с arXiv")
st.markdown(
    "Поиск и скачивание PDF-статей с **arXiv.org**. "
    "Метаданные сохраняются в XLSX рядом с PDF."
)

# ════════════════════════════════════════════════════════════════════
# Инициализация session_state
# ════════════════════════════════════════════════════════════════════
if "arxiv_search_results" not in st.session_state:
    st.session_state.arxiv_search_results = []
if "arxiv_selected_articles" not in st.session_state:
    st.session_state.arxiv_selected_articles = []
if "dl_thread" not in st.session_state:
    st.session_state.dl_thread = None
if "dl_cancel" not in st.session_state:
    st.session_state.dl_cancel = None
if "dl_progress" not in st.session_state:
    st.session_state.dl_progress = None
if "dl_result" not in st.session_state:
    st.session_state.dl_result = None

tab_search, tab_download = st.tabs(["🔍 Поиск", "📥 Загрузка"])

# ════════════════════════════════════════════════════════════════════
# ВКЛАДКА 1: ПОИСК
# ════════════════════════════════════════════════════════════════════
with tab_search:
    st.subheader("🔍 Поиск статей на arXiv")

    col_query, col_settings = st.columns([3, 1])

    with col_query:
        query = st.text_input(
            "Поисковый запрос (формат arXiv API)",
            value=st.session_state.get("arxiv_query", ""),
            key="arxiv_query",
            placeholder='Например: ti:rag AND abs:medicine',
            help="Введите запрос в формате arXiv API. "
                 "Справка по синтаксису — ниже в раскрывающемся блоке.",
        )

    with col_settings:
        max_results = st.number_input(
            "Макс. результатов",
            min_value=1,
            max_value=100,
            value=st.session_state.get("arxiv_max_results", 10),
            key="arxiv_max_results",
        )

    # ── Справка по синтаксису ──────────────────────────────────
    with st.expander("📖 Справка по синтаксису arXiv API"):
        st.markdown(
            """
            Запрос передаётся напрямую в arXiv API. Используйте операторы и префиксы полей.

            **Префиксы полей:**

            | Префикс | Поле               | Пример                     |
            |---------|--------------------|----------------------------|
            | `ti:`   | Title (заголовок)  | `ti:transformer`           |
            | `au:`   | Author (автор)     | `au:Vaswani`               |
            | `abs:`  | Abstract (аннотация)| `abs:attention mechanism` |
            | `cat:`  | Category (категория)| `cat:cs.AI`               |
            | `all:`  | Все поля           | `all:briquetting`          |
            | `jr:`   | Journal reference  | `jr:Nature`                |

            **Логические операторы:**

            | Оператор   | Значение | Пример                                      |
            |------------|----------|---------------------------------------------|
            | `AND`      | И        | `ti:rag AND abs:medicine`                   |
            | `OR`       | Или      | `ti:iron OR abs:steelmaking`                |
            | `ANDNOT`   | Не       | `ti:rag ANDNOT abs:chatbot`                 |
            | `"..."`    | Фраза    | `ti:"retrieval augmented generation"`       |
            | `( )`      | Группы   | `(ti:rag OR ti:retrieval) AND abs:medicine` |

            **Категории (примеры):** `cs.AI`, `cs.CL`, `cs.IR`, `physics.geo-ph`, `q-bio.QM`

            **Примеры запросов:**
            - `all:mineral processing AND cat:cs.AI`
            - `ti:ore AND abs:beneficiation`
            - `ti:briquetting OR ti:pelletizing`
            - `"large language model" AND abs:RAG`
            """
        )

    with st.expander("⚙️ Настройки сортировки"):
        col_sort1, col_sort2 = st.columns(2)
        with col_sort1:
            sort_by = st.selectbox(
                "Сортировать по",
                options=["relevance", "lastUpdatedDate", "submittedDate"],
                format_func=lambda x: {
                    "relevance": "Релевантность",
                    "lastUpdatedDate": "Дата обновления",
                    "submittedDate": "Дата подачи",
                }[x],
                key="arxiv_sort_by",
            )
        with col_sort2:
            sort_order = st.selectbox(
                "Порядок",
                options=["descending", "ascending"],
                format_func=lambda x: {"descending": "По убыванию", "ascending": "По возрастанию"}[x],
                key="arxiv_sort_order",
            )

    # Кнопка поиска
    if st.button(
        "🔍 Найти статьи",
        key="arxiv_search_btn",
        use_container_width=True,
        type="primary",
        disabled=not query.strip(),
    ):
        with st.spinner("Поиск на arXiv..."):
            try:
                results = search_articles(
                    query=query.strip(),
                    max_results=max_results,
                    sort_by=sort_by,
                    sort_order=sort_order,
                )
                st.session_state.arxiv_search_results = results
                if not results:
                    st.warning("Ничего не найдено. Попробуйте другой запрос.")
                else:
                    st.success(f"Найдено **{len(results)}** статей")
            except arxiv.HTTPError as e:
                _err_text = str(e)
                if "429" in _err_text:
                    st.warning(
                        "⏳ **arXiv перегружен** (HTTP 429 — слишком много запросов).\n\n"
                        "Скрипт уже сделал 3 попытки с задержками. "
                        "Подождите 1–2 минуты и попробуйте снова."
                    )
                else:
                    st.error(f"Ошибка arXiv API: {e}")
            except Exception as e:
                st.error(f"Ошибка поиска: {e}")

    # Результаты поиска
    _results = st.session_state.arxiv_search_results
    if _results:
        st.divider()
        st.subheader(f"📄 Результаты ({len(_results)})")

        # Выбор статей
        _article_labels = [
            f"{r.arxiv_id} | {r.title[:80]}{'...' if len(r.title) > 80 else ''} ({r.published})"
            for r in _results
        ]

        selected_indices = st.multiselect(
            "Выберите статьи для скачивания",
            options=list(range(len(_results))),
            format_func=lambda i: _article_labels[i],
            default=list(range(len(_results))),
            key="arxiv_selected_indices",
            help="По умолчанию выбраны все статьи.",
        )

        # Сохраняем выбранные статьи в session_state
        st.session_state.arxiv_selected_articles = [_results[i] for i in selected_indices]

        # Детальная информация по каждой статье
        for idx, r in enumerate(_results):
            _is_selected = idx in selected_indices
            _icon = "✅" if _is_selected else "⬜"
            with st.expander(f"{_icon} **{r.arxiv_id}** — {r.title}", expanded=False):
                col_meta, col_abstract = st.columns([1, 2])

                with col_meta:
                    st.write(f"**Авторы:** {', '.join(r.authors[:5])}{'...' if len(r.authors) > 5 else ''}")
                    st.write(f"**Опубликовано:** {r.published}")
                    st.write(f"**Обновлено:** {r.updated}")
                    st.write(f"**Категории:** {', '.join(r.categories)}")
                    st.link_button("🔗 Открыть на arXiv", url=r.entry_id)

                with col_abstract:
                    st.write(r.abstract)

        if selected_indices:
            st.info(
                f"Выбрано **{len(selected_indices)}** из **{len(_results)}** статей. "
                "Перейдите на вкладку **📥 Загрузка** для скачивания."
            )

# ════════════════════════════════════════════════════════════════════
# ВКЛАДКА 2: ЗАГРУЗКА
# ════════════════════════════════════════════════════════════════════
with tab_download:
    st.subheader("📥 Скачивание PDF + метаданные")

    # ── Настройки папок ────────────────────────────────────────────
    _default_base = "data/articles/raw"

    base_dir = st.text_input(
        "📂 Корневая папка для статей",
        value=st.session_state.get("arxiv_base_dir", _default_base),
        key="arxiv_base_dir",
        help="PDF и XLSX сохраняются в подпапки этой директории.",
    )

    # Существующие подпапки
    _existing_sf = get_existing_subfolders(base_dir) if Path(base_dir).exists() else []

    col_new, col_existing = st.columns(2)

    with col_new:
        new_subfolder = st.text_input(
            "📁 Новая подпапка",
            value="",
            key="arxiv_new_subfolder",
            placeholder="Например: metallurgy",
            help="Оставьте пустым, если хотите выбрать существующую.",
        )

    with col_existing:
        existing_subfolder = st.selectbox(
            "📁 Или выберите существующую",
            options=[""] + _existing_sf,
            key="arxiv_existing_subfolder",
            help="Выберите подпапку, если она уже существует.",
        )

    # Определяем итоговую подпапку
    _subfolder = new_subfolder.strip() if new_subfolder.strip() else existing_subfolder
    if _subfolder:
        _save_dir = str(Path(base_dir) / _subfolder)
        _xlsx_path = str(Path(_save_dir) / XLSX_FILENAME)

        # Статистика подпапки
        if Path(_save_dir).exists():
            _stats = get_subfolder_stats(base_dir, _subfolder)
            st.caption(
                f"📊 В папке `{_subfolder}`: "
                f"{_stats['pdf_count']} PDF, "
                f"{_stats['xlsx_rows']} записей в XLSX"
            )
    else:
        _save_dir = base_dir
        _xlsx_path = str(Path(base_dir) / XLSX_FILENAME)
        st.caption("⚠️ Подпапка не выбрана — файлы сохранятся в корневую папку")

    # ── Список выбранных статей ────────────────────────────────────
    _selected_articles: list[ArxivArticle] = st.session_state.arxiv_selected_articles

    st.divider()

    if _selected_articles:
        st.write(f"**Выбрано для скачивания:** {len(_selected_articles)} статей")

        # Показать краткий список
        with st.expander("📋 Список статей", expanded=False):
            for a in _selected_articles:
                st.write(f"- `{a.arxiv_id}` — {a.title[:100]}")

        # Дедупликация: показываем, сколько новых
        _already_ids = load_downloaded_ids(_xlsx_path) if Path(_xlsx_path).exists() else set()
        _new_articles = [a for a in _selected_articles if a.arxiv_id not in _already_ids]
        _dup_count = len(_selected_articles) - len(_new_articles)

        if _dup_count > 0:
            st.info(f"🆕 Новых: **{len(_new_articles)}**, уже скачано: **{_dup_count}**")
        else:
            st.info(f"🆕 Все **{len(_new_articles)}** статей — новые")
    else:
        st.warning(
            "⚠️ Нет выбранных статей. Перейдите на вкладку **🔍 Поиск**, "
            "найдите статьи и выберите нужные."
        )

    # ── Прогресс / Результат / Запуск ──────────────────────────────
    _thread = st.session_state.dl_thread
    _is_running = _thread is not None and _thread.is_alive()

    # Фрагмент автообновления прогресса
    @st.fragment(run_every="2s")
    def _render_dl_progress():
        _t = st.session_state.dl_thread
        _running = _t is not None and _t.is_alive()

        if not _running:
            st.rerun()
            return

        _prog = st.session_state.dl_progress
        if _prog:
            _pct = _prog.get("progress", 0.0)
            _msg = _prog.get("message", "")
            st.progress(_pct, text=_msg)

        if st.button(
            "❌ Отменить скачивание",
            key="dl_cancel_btn",
            use_container_width=True,
            type="secondary",
        ):
            if st.session_state.dl_cancel:
                st.session_state.dl_cancel.set()
            st.rerun()

    if _is_running:
        _render_dl_progress()

    elif st.session_state.dl_result:
        # ── Результат ──────────────────────────────────────────────
        _res = st.session_state.dl_result
        _dl = _res.get("downloaded", 0)
        _sk = _res.get("skipped", 0)
        _err = _res.get("errors", 0)
        _total = _res.get("total", 0)

        if _err > 0:
            st.warning(
                f"⚠️ Завершено с ошибками. "
                f"Скачано: **{_dl}**, пропущено: **{_sk}**, ошибок: **{_err}** (из {_total})"
            )
        else:
            st.success(
                f"✅ Готово! Скачано: **{_dl}**, пропущено: **{_sk}** (из {_total})"
            )

        _error_details = _res.get("error_details", [])
        if _error_details:
            with st.expander("❌ Ошибки", expanded=True):
                for e in _error_details:
                    st.write(f"- {e}")

        if st.button("🗑️ Сбросить результат", key="dl_clear_result"):
            st.session_state.dl_result = None
            st.session_state.dl_progress = None
            st.rerun()

    else:
        # ── Кнопка запуска ─────────────────────────────────────────
        _can_run = len(_selected_articles) > 0

        if _can_run:
            if st.button(
                f"📥 Скачать {len(_new_articles)} "
                f"{'статью' if len(_new_articles) == 1 else 'статьи' if len(_new_articles) < 5 else 'статей'}"
                + (f" (из {len(_selected_articles)} выбранных)" if _dup_count > 0 else ""),
                key="dl_run_btn",
                use_container_width=True,
                type="primary",
            ):
                cancel_event = threading.Event()
                st.session_state.dl_cancel = cancel_event
                st.session_state.dl_progress = {"progress": 0.0, "message": "Подготовка..."}
                st.session_state.dl_result = None

                _articles_copy = list(_selected_articles)
                _sd = _save_dir
                _xp = _xlsx_path

                def _progress_cb(progress: float, message: str):
                    st.session_state.dl_progress = {
                        "progress": progress,
                        "message": message,
                    }

                def _run():
                    result = run_download_articles(
                        articles=_articles_copy,
                        save_dir=_sd,
                        xlsx_path=_xp,
                        progress_callback=_progress_cb,
                        cancel_event=cancel_event,
                    )
                    st.session_state.dl_result = result

                thread = threading.Thread(target=_run, daemon=True)
                st.session_state.dl_thread = thread
                thread.start()
                st.rerun()

    # ── Обзор уже скачанных статей в подпапке ──────────────────────
    st.divider()
    st.subheader("📁 Скачанные статьи в папке")

    if _subfolder and Path(_save_dir).exists():
        _pdf_files = sorted(Path(_save_dir).glob("*.pdf"))
        if _pdf_files:
            st.write(f"**{len(_pdf_files)}** PDF в `{_subfolder}/`")
            with st.expander("📋 Список файлов", expanded=False):
                for p in _pdf_files:
                    _size_kb = p.stat().st_size / 1024
                    st.write(f"- `{p.name}` ({_size_kb:.0f} KB)")
        else:
            st.info("PDF-файлов пока нет.")
    elif not _subfolder:
        st.info("Выберите подпапку выше.")
    else:
        st.info(f"Папка `{_save_dir}` не существует — создастся при скачивании.")