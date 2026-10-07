"""Страница постпроцессинга — обработка результатов парсинга.

Вкладки:
- Очистка артефактов парсинга: #, маркеры code block, LaTeX-разделители
- Выравнивание заголовков: исправление уровней по оглавлению
- Нумерация страниц: добавление тегов <N> перед абзацами в .md файлах
- Соединение по главам: склейка пронумерованных файлов в главы по заголовкам
- Извлечение таблиц: выделение таблиц и описаний в отдельные файлы
- Извлечение описаний рисунков: абзацы «Рис. N» в отдельные файлы
- Деление на чанки: разбиение Markdown на текстовые и табличные чанки
"""

import threading

import streamlit as st
from pathlib import Path

from src.run_page_numbering import run_page_numbering, _find_md_files
from src.run_chapter_merger import run_chapter_merger, _find_book_dirs as _find_chapter_books
from src.run_chunking import (
    run_chunking, run_text_chunking, run_table_chunking, run_figure_chunking,
    _find_book_dirs as _find_chunk_books,
)
from src.run_clean_table_headers import run_clean_table_headers, count_files_with_artifacts
from src.clean_table_headers import ARTIFACT_TYPES, ALL_ARTIFACT_KEYS
from src.run_fix_heading_levels import run_fix_heading_levels, find_books_with_toc, count_all_mismatches
from src.run_extract_tables_md import (
    run_extract_tables_from_chapters, count_all_chapter_tables,
    find_books_with_chapters,
)
from src.run_extract_figure_captions import (
    run_extract_figures_from_chapters,
    count_all_chapter_figures,
    find_books_with_chapters as _find_ef_books,
)
from src.run_clean_tagged import (
    run_clean_tagged_content, count_all_tagged_blocks,
    find_books_with_chapters as _find_ct_books,
)
from src.extract_figure_captions import DEFAULT_PATTERN as _DEFAULT_FIG_PATTERN
from src.run_normalize_tables import (
    run_normalize_tables, find_books_with_chapters as _find_norm_books,
    count_all_tables_detailed,
)
from src.normalize_tables import count_markers as _count_markers
from src.run_normalize_figures import (
    run_normalize_figures, find_books_with_chapters as _find_nf_books,
    count_all_figures_detailed,
)
from src.run_table_summary import (
    run_table_summary, find_books_with_extracted_tables, count_all_summaries,
)
from src.book_metadata import (
    load_all as load_books_metadata, save_all as save_books_metadata, list_book_folders,
)
from src.check_chunk_sizes import (
    check_raw_table_sizes, find_book_extracted_table_dirs,
    check_created_chunk_sizes, find_book_chunk_dirs,
)
from src.run_llm_table_summary import (
    run_llm_table_summary, find_oversized_tables,
    DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT_TEMPLATE,
)
from src.llm_clients import list_model_names


# ══════════════════════════════════════════════════════════════════════
# Страница
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Постпроцессинг", page_icon="🔧", layout="wide")
st.title("🔧 Постпроцессинг")

st.markdown(
    "Обработка результатов AI-парсинга: расстановка номеров страниц, "
    "объединение файлов, деление на чанки и другие преобразования."
)

# ── Вкладки ────────────────────────────────────────────────────────────
tab_clean_headers, tab_fix_headings, tab_numbering, tab_chapters, tab_normalize, tab_normalize_figures, tab_extract_tables, tab_extract_figures, tab_clean_tagged, tab_table_summary, tab_check_sizes, tab_llm_summary, tab_chunks, tab_biblio = st.tabs([
    "🧹 Очистка артефактов",
    "📑 Выравнивание заголовков",
    "🔢 Нумерация страниц",
    "📚 Соединение по главам",
    "📍 Нормализация таблиц",
    "🖼️ Нормализация рисунков",
    "📊 Извлечение таблиц",
    "🖼️ Описания рисунков",
    "🧹 Очистка по маркерам",
    "📋 Сводки таблиц",
    "📏 Проверка размеров",
    "🤖 LLM-сводки таблиц",
    "✂️ Деление на чанки",
    "📚 Библиография",
])

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Очистка артефактов парсинга
# ══════════════════════════════════════════════════════════════════════
with tab_clean_headers:
    st.subheader("🧹 Очистка артефактов парсинга")
    st.markdown(
        "Удаляет артефакты VLM-парсинга из Markdown-файлов. "
        "Выберите нужные типы артефактов ниже.\n\n"
        "Файлы изменяются **in-place** в папке `parsed/`."
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    clean_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("clean_base_dir", "data/books"),
        key="clean_base_dir",
        help="Структура: папка/название_книги/parsed/*.md",
    )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _clean_available_books: list[str] = []
    _clean_base_path = Path(clean_base_dir) if clean_base_dir else None

    if _clean_base_path and _clean_base_path.exists() and _clean_base_path.is_dir():
        for _book_dir in sorted(_clean_base_path.iterdir()):
            if _book_dir.is_dir() and (_book_dir / "parsed").is_dir():
                _clean_available_books.append(_book_dir.name)

    # ── Выбор типов артефактов ──────────────────────────────────────
    # Формируем метки для multiselect: "ключ — описание"
    _artifact_options = {k: f"{k} — {v}" for k, v in ARTIFACT_TYPES.items()}
    _all_artifact_labels = list(_artifact_options.values())

    clean_selected_artifacts_labels = st.multiselect(
        "🎯 Выберите артефакты для очистки",
        options=_all_artifact_labels,
        default=_all_artifact_labels,
        key="clean_selected_artifacts",
        help="Выберите, какие типы артефактов нужно удалить.",
    )

    # Конвертируем метки обратно в ключи → enabled dict
    _label_to_key = {v: k for k, v in _artifact_options.items()}
    clean_enabled_keys = [
        _label_to_key[lbl] for lbl in clean_selected_artifacts_labels
        if lbl in _label_to_key
    ]
    clean_enabled: dict[str, bool] = {k: (k in clean_enabled_keys) for k in ALL_ARTIFACT_KEYS}

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    clean_selected_books: list[str] = []
    if _clean_available_books:
        clean_selected_books = st.multiselect(
            "📚 Выберите книги для очистки",
            options=_clean_available_books,
            default=_clean_available_books,
            key="clean_selected_books",
            help="Выберите книги, в которых нужно очистить артефакты парсинга.",
        )
    elif clean_base_dir and _clean_base_path and _clean_base_path.exists():
        st.warning(
            f"В `{clean_base_dir}` не найдено папок с `parsed/`. "
            f"Сначала выполните парсинг."
        )

    # Предпросмотр: сколько файлов с артефактами выбранного типа (по кнопке)
    if clean_selected_books and _clean_base_path and _clean_base_path.exists():
        if st.button("📊 Подсчитать артефакты", key="clean_preview_btn"):
            with st.spinner("Подсчёт файлов с артефактами парсинга..."):
                _artifact_stats = count_files_with_artifacts(
                    _clean_base_path,
                    book_names=clean_selected_books,
                    enabled=clean_enabled if any(clean_enabled.values()) else None,
                )
            _files_with_headers = _artifact_stats["files_with_artifacts"]
            _artifact_counts = _artifact_stats["artifact_counts"]

            if _files_with_headers > 0:
                st.info(
                    f"Найдено **{_files_with_headers}** "
                    f"{'файлов' if _files_with_headers > 1 else 'файл'} "
                    f"с выбранными артефактами "
                    f"в **{len(clean_selected_books)}** "
                    f"{'книгах' if len(clean_selected_books) > 1 else 'книге'}"
                )
                # Детализация по типам
                _detail_parts = []
                for key, count in _artifact_counts.items():
                    if count > 0 and key in ARTIFACT_TYPES:
                        _detail_parts.append(f"  - {ARTIFACT_TYPES[key]}: **{count}**")
                if _detail_parts:
                    with st.expander("📋 Детализация по типам", expanded=False):
                        for _part in _detail_parts:
                            st.markdown(_part)
            else:
                st.success("✅ В выбранных книгах нет выбранных артефактов.")
    elif _clean_available_books and not clean_selected_books:
        st.warning("⚠️ Выберите хотя бы одну книгу.")

    # Инициализация состояния прогресса
    if "clean_thread" not in st.session_state:
        st.session_state.clean_thread = None
    if "clean_cancel" not in st.session_state:
        st.session_state.clean_cancel = None
    if "clean_progress" not in st.session_state:
        st.session_state.clean_progress = None
    if "clean_result" not in st.session_state:
        st.session_state.clean_result = None


    @st.fragment(run_every="2s")
    def _render_clean_progress():
        """Автообновляемый фрагмент: прогресс очистки артефактов + отмена."""
        _thread = st.session_state.clean_thread
        _is_running = _thread is not None and _thread.is_alive()

        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.clean_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0

            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"✏️ Изменено: **{_progress.get('modified', 0)}**  |  "
                    f"⏩ Без изменений: **{_progress.get('unchanged', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**"
                )
                _repl = _progress.get("total_replacements", 0)
                if _repl > 0:
                    st.write(f"🔢 Всего замен: **{_repl}**")

        if st.button(
            "❌ Отменить очистку",
            key="clean_cancel_btn",
            use_container_width=True,
            type="secondary",
        ):
            if st.session_state.clean_cancel:
                st.session_state.clean_cancel.set()
            st.rerun()


    with tab_clean_headers:
        _clean_thread = st.session_state.clean_thread
        _clean_is_running = _clean_thread is not None and _clean_thread.is_alive()

        if _clean_is_running:
            _render_clean_progress()

        elif st.session_state.clean_result:
            _clean_res = st.session_state.clean_result
            if _clean_res.get("cancelled"):
                st.warning(
                    f"⚠️ Очистка отменена. "
                    f"Изменено: {_clean_res.get('modified', 0)}/{_clean_res.get('total', 0)}, "
                    f"Замен: {_clean_res.get('total_replacements', 0)}. "
                    f"Время: {_clean_res.get('elapsed_sec', '?')}с"
                )
            elif _clean_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Изменено: {_clean_res.get('modified', 0)}, "
                    f"Без изменений: {_clean_res.get('unchanged', 0)}, "
                    f"Ошибок: {_clean_res.get('errors', 0)}. "
                    f"Замен: {_clean_res.get('total_replacements', 0)}. "
                    f"Время: {_clean_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Очистка завершена! "
                    f"Изменено: {_clean_res.get('modified', 0)} файлов, "
                    f"замен: {_clean_res.get('total_replacements', 0)}. "
                    f"Время: {_clean_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="clean_clear_result"):
                st.session_state.clean_result = None
                st.session_state.clean_progress = None
                st.rerun()

        else:
            # Кнопка запуска
            _can_run_clean = (
                clean_base_dir
                and _clean_base_path
                and _clean_base_path.exists()
                and clean_selected_books
            )

            if _can_run_clean:
                if st.button(
                    f"🧹 Очистить артефакты в {len(clean_selected_books)} "
                    f"{'книге' if len(clean_selected_books) == 1 else 'книгах'}",
                    key="clean_run_btn",
                    use_container_width=True,
                    type="primary",
                ):
                    cancel_event = threading.Event()
                    st.session_state.clean_cancel = cancel_event
                    st.session_state.clean_progress = {
                        "current": 0,
                        "total": 0,
                        "current_file": "подготовка...",
                        "modified": 0,
                        "unchanged": 0,
                        "errors": 0,
                        "total_replacements": 0,
                    }
                    st.session_state.clean_result = None

                    _clean_sel = list(clean_selected_books)
                    _clean_base = clean_base_dir
                    _clean_enabled = dict(clean_enabled)

                    def _clean_progress_cb(current, total, current_file, stats):
                        st.session_state.clean_progress = {
                            "current": current,
                            "total": total,
                            "current_file": current_file,
                            "modified": stats["modified"],
                            "unchanged": stats["unchanged"],
                            "errors": stats["errors"],
                            "total_replacements": stats["total_replacements"],
                        }

                    def _clean_run():
                        result = run_clean_table_headers(
                            base_dir=_clean_base,
                            book_names=_clean_sel,
                            enabled=_clean_enabled,
                            cancel_event=cancel_event,
                            progress_callback=_clean_progress_cb,
                        )
                        st.session_state.clean_result = result

                    thread = threading.Thread(target=_clean_run, daemon=True)
                    st.session_state.clean_thread = thread
                    thread.start()
                    st.rerun()

            elif clean_base_dir and _clean_base_path and not _clean_base_path.exists():
                st.error(f"❌ Папка не найдена: `{clean_base_dir}`")


# ══════════════════════════════════════════════════════════════════════
# Вкладка: Нормализация границ таблиц
# ══════════════════════════════════════════════════════════════════════
with tab_normalize:
    st.subheader("📍 Нормализация границ таблиц")
    st.markdown(
        "Добавляет маркеры `[TABLE_START]`/`[TABLE_END]` вокруг каждой таблицы "
        "в файлах `chapters/*.md` — включая caption перед таблицей и сноски "
        "после неё.\n\n"
        "**Зачем:** при извлечении таблицы могут захватывать лишний текст "
        "или терять описание. Маркеры дают 100% точность границ.\n\n"
        "**Что делает:**\n"
        "1. Разделяет строки, где `</table>` и текст на одной строке\n"
        "2. Находит caption («Таблица N...», «Окончание табл. N»)\n"
        "3. Находит сноски после таблицы (`* текст`, `** текст`)\n"
        "4. Оборачивает всё в маркеры\n\n"
        "Файлы изменяются **in-place** в `chapters/`. "
        "Перед запуском можно посмотреть превью (подсчёт таблиц и маркеров)."
    )

    # ── Путь ────────────────────────────────────────────────────────
    nm_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("nm_base_dir", "data/books"),
        key="nm_base_dir",
        help="Структура: папка/название_книги/chapters/*.md",
    )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _nm_base_path = Path(nm_base_dir) if nm_base_dir else None
    _nm_available_books: list[str] = []

    if _nm_base_path and _nm_base_path.exists():
        _nm_books = _find_norm_books(_nm_base_path)
        _nm_available_books = [name for name, _ in _nm_books]

    nm_selected_books: list[str] = []
    if _nm_available_books:
        nm_selected_books = st.multiselect(
            "📚 Выберите книги",
            options=_nm_available_books,
            default=_nm_available_books,
            key="nm_selected_books",
        )

        # Предпросмотр: подсчёт таблиц и маркеров
        if nm_selected_books:
            if st.button("📊 Подсчитать таблицы и маркеры", key="nm_preview_btn"):
                with st.spinner("Подсчёт таблиц в главах..."):
                    _nm_counts = count_all_tables_detailed(
                        _nm_base_path, book_names=nm_selected_books,
                    )
                if _nm_counts["total_tables"] > 0:
                    st.info(
                        f"Найдено **{_nm_counts['total_tables']}** маркеров таблиц "
                        f"в **{_nm_counts['total_files']}** файлах"
                    )
                else:
                    st.warning(
                        "Маркеры `[TABLE_START]`/`[TABLE_END]` не найдены. "
                        "Сначала запустите нормализацию."
                    )
    elif _nm_base_path and _nm_base_path.exists():
        st.warning(
            f"В `{nm_base_dir}` нет книг с `chapters/`. "
            f"Сначала выполните соединение по главам."
        )

    # ── Состояние прогресса ─────────────────────────────────────────
    if "nm_thread" not in st.session_state:
        st.session_state.nm_thread = None
    if "nm_cancel" not in st.session_state:
        st.session_state.nm_cancel = None
    if "nm_progress" not in st.session_state:
        st.session_state.nm_progress = None
    if "nm_result" not in st.session_state:
        st.session_state.nm_result = None

    @st.fragment(run_every="2s")
    def _render_nm_progress():
        """Автообновляемый фрагмент: прогресс нормализации + отмена."""
        _thread = st.session_state.nm_thread
        _is_running = _thread is not None and _thread.is_alive()
        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.nm_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0
            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"📍 Маркеров: **{_progress.get('markers_placed', 0)}**  |  "
                    f"✏️ Изменено: **{_progress.get('modified', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**"
                )

        if st.button("❌ Отменить нормализацию", key="nm_cancel_btn",
                      use_container_width=True, type="secondary"):
            if st.session_state.nm_cancel:
                st.session_state.nm_cancel.set()
            st.rerun()

    with tab_normalize:
        _nm_thread = st.session_state.nm_thread
        _nm_is_running = _nm_thread is not None and _nm_thread.is_alive()

        if _nm_is_running:
            _render_nm_progress()

        elif st.session_state.nm_result:
            _nm_res = st.session_state.nm_result
            if _nm_res.get("cancelled"):
                st.warning(
                    f"⚠️ Нормализация отменена. "
                    f"Маркеров: {_nm_res.get('markers_placed', 0)}, "
                    f"Файлов: {_nm_res.get('modified', 0)}. "
                    f"Время: {_nm_res.get('elapsed_sec', '?')}с"
                )
            elif _nm_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Маркеров: {_nm_res.get('markers_placed', 0)}, "
                    f"Ошибок: {_nm_res.get('errors', 0)}. "
                    f"Время: {_nm_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Нормализация завершена! "
                    f"Добавлено **{_nm_res.get('markers_placed', 0)}** пар маркеров "
                    f"в **{_nm_res.get('modified', 0)}** файлов. "
                    f"Время: {_nm_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="nm_clear_result"):
                st.session_state.nm_result = None
                st.session_state.nm_progress = None
                st.rerun()

        else:
            _can_run_nm = (
                nm_base_dir and _nm_base_path and _nm_base_path.exists()
                and nm_selected_books
            )
            if _can_run_nm:
                if st.button(
                    f"📍 Нормализовать таблицы в {len(nm_selected_books)} "
                    f"{'книге' if len(nm_selected_books) == 1 else 'книгах'}",
                    key="nm_run_btn", use_container_width=True, type="primary",
                ):
                    _nm_cancel_ev = threading.Event()
                    st.session_state.nm_cancel = _nm_cancel_ev
                    st.session_state.nm_progress = {
                        "current": 0, "total": 0,
                        "current_file": "подготовка...",
                        "modified": 0, "markers_placed": 0, "errors": 0,
                    }
                    st.session_state.nm_result = None

                    _nm_sel = list(nm_selected_books)
                    _nm_base = nm_base_dir

                    def _nm_progress_cb(current, total, current_file, stats):
                        st.session_state.nm_progress = {
                            "current": current, "total": total,
                            "current_file": current_file, **stats,
                        }

                    def _nm_run():
                        result = run_normalize_tables(
                            base_dir=_nm_base,
                            book_names=_nm_sel,
                            cancel_event=_nm_cancel_ev,
                            progress_callback=_nm_progress_cb,
                        )
                        st.session_state.nm_result = result

                    _nm_t = threading.Thread(target=_nm_run, daemon=True)
                    st.session_state.nm_thread = _nm_t
                    _nm_t.start()
                    st.rerun()
            elif nm_base_dir and _nm_base_path and not _nm_base_path.exists():
                st.error(f"❌ Папка не найдена: `{nm_base_dir}`")

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Нормализация границ рисунков
# ══════════════════════════════════════════════════════════════════════
with tab_normalize_figures:
    st.subheader("🖼️ Нормализация границ рисунков")
    st.markdown(
        "Добавляет маркеры `[FIGURE_START]`/`[FIGURE_END]` вокруг каждого "
        "описания рисунка в файлах `chapters/*.md`.\n\n"
        "**Что делает:**\n"
        "1. Обрабатывает HTML `<figure>` блоки — вычищает обёртки, "
        "оставляет только caption\n"
        "2. Находит текстовые описания «Рис. N. ...» и continuation lines "
        "(позиции `1 — ...; 2 — ...`)\n"
        "3. Оборачивает всё в маркеры\n\n"
        "**Безопасность:** не трогает области внутри `[TABLE_START]`/`[TABLE_END]`. "
        "Если `<figure>` содержит `<table>` — убирает только HTML-обёртку, "
        "FIGURE-маркеры не ставятся.\n\n"
        "Файлы изменяются **in-place** в `chapters/`. "
        "⚠️ **Сначала запустите «Нормализация таблиц»**."
    )

    nf_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("nf_base_dir", "data/books"),
        key="nf_base_dir",
        help="Структура: папка/название_книги/chapters/*.md",
    )

    st.divider()

    _nf_base_path = Path(nf_base_dir) if nf_base_dir else None
    _nf_available_books: list[str] = []

    if _nf_base_path and _nf_base_path.exists():
        _nf_books = _find_nf_books(_nf_base_path)
        _nf_available_books = [name for name, _ in _nf_books]

    nf_selected_books: list[str] = []
    if _nf_available_books:
        nf_selected_books = st.multiselect(
            "📚 Выберите книги",
            options=_nf_available_books,
            default=_nf_available_books,
            key="nf_selected_books",
        )

        if nf_selected_books:
            if st.button("📊 Подсчитать рисунки и маркеры", key="nf_preview_btn"):
                with st.spinner("Подсчёт рисунков в главах..."):
                    _nf_counts = count_all_figures_detailed(
                        _nf_base_path, book_names=nf_selected_books,
                    )
                if _nf_counts["total_figures"] > 0:
                    st.info(
                        f"**{_nf_counts['total_files']}** файлов, "
                        f"**{_nf_counts['total_figures']}** рисунков "
                        f"(<figure>: {_nf_counts['figure_tags']}, "
                        f"plain «Рис. N.»: {_nf_counts['plain_captions']})"
                    )
                    if _nf_counts["files_with_markers"] > 0:
                        st.warning(
                            f"⚠️ **{_nf_counts['files_with_markers']}** файлов "
                            f"уже содержат маркеры (будут пересозданы)"
                        )
                else:
                    st.success("✅ Нет файлов с рисунками.")
    elif _nf_base_path and _nf_base_path.exists():
        st.warning(
            f"В `{nf_base_dir}` нет книг с `chapters/`. "
            f"Сначала выполните соединение по главам."
        )

    # ── Состояние прогресса ─────────────────────────────────────────
    if "nf_thread" not in st.session_state:
        st.session_state.nf_thread = None
    if "nf_cancel" not in st.session_state:
        st.session_state.nf_cancel = None
    if "nf_progress" not in st.session_state:
        st.session_state.nf_progress = None
    if "nf_result" not in st.session_state:
        st.session_state.nf_result = None

    @st.fragment(run_every="2s")
    def _render_nf_progress():
        """Автообновляемый фрагмент: прогресс нормализации рисунков + отмена."""
        _thread = st.session_state.nf_thread
        _is_running = _thread is not None and _thread.is_alive()
        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.nf_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0
            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"🖼️ Маркеров: **{_progress.get('markers_placed', 0)}**  |  "
                    f"✏️ Изменено: **{_progress.get('modified', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**"
                )

        if st.button("❌ Отменить нормализацию рисунков", key="nf_cancel_btn",
                      use_container_width=True, type="secondary"):
            if st.session_state.nf_cancel:
                st.session_state.nf_cancel.set()
            st.rerun()

    with tab_normalize_figures:
        _nf_thread = st.session_state.nf_thread
        _nf_is_running = _nf_thread is not None and _nf_thread.is_alive()

        if _nf_is_running:
            _render_nf_progress()

        elif st.session_state.nf_result:
            _nf_res = st.session_state.nf_result
            if _nf_res.get("error"):
                st.error(
                    f"❌ Критическая ошибка: `{_nf_res['error']}`"
                )
            elif _nf_res.get("cancelled"):
                st.warning(
                    f"⚠️ Нормализация отменена. "
                    f"Маркеров: {_nf_res.get('markers_placed', 0)}, "
                    f"Файлов: {_nf_res.get('modified', 0)}. "
                    f"Время: {_nf_res.get('elapsed_sec', '?')}с"
                )
            elif _nf_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Маркеров: {_nf_res.get('markers_placed', 0)}, "
                    f"Ошибок: {_nf_res.get('errors', 0)}. "
                    f"Время: {_nf_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Нормализация рисунков завершена! "
                    f"Добавлено **{_nf_res.get('markers_placed', 0)}** пар маркеров "
                    f"в **{_nf_res.get('modified', 0)}** файлов. "
                    f"Время: {_nf_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="nf_clear_result"):
                st.session_state.nf_result = None
                st.session_state.nf_progress = None
                st.rerun()

        else:
            _can_run_nf = (
                nf_base_dir and _nf_base_path and _nf_base_path.exists()
                and nf_selected_books
            )
            if _can_run_nf:
                if st.button(
                    f"🖼️ Нормализовать рисунки в {len(nf_selected_books)} "
                    f"{'книге' if len(nf_selected_books) == 1 else 'книгах'}",
                    key="nf_run_btn", use_container_width=True, type="primary",
                ):
                    _nf_cancel_ev = threading.Event()
                    st.session_state.nf_cancel = _nf_cancel_ev
                    st.session_state.nf_progress = {
                        "current": 0, "total": 0,
                        "current_file": "подготовка...",
                        "modified": 0, "markers_placed": 0, "errors": 0,
                    }
                    st.session_state.nf_result = None

                    _nf_sel = list(nf_selected_books)
                    _nf_base = nf_base_dir

                    def _nf_progress_cb(current, total, current_file, stats):
                        st.session_state.nf_progress = {
                            "current": current, "total": total,
                            "current_file": current_file, **stats,
                        }

                    def _nf_run():
                        try:
                            result = run_normalize_figures(
                                base_dir=_nf_base,
                                book_names=_nf_sel,
                                cancel_event=_nf_cancel_ev,
                                progress_callback=_nf_progress_cb,
                            )
                            st.session_state.nf_result = result
                        except Exception as exc:
                            import traceback
                            traceback.print_exc()
                            st.session_state.nf_result = {
                                "cancelled": False,
                                "error": str(exc),
                                "errors": 1,
                                "modified": 0,
                                "markers_placed": 0,
                                "elapsed_sec": 0,
                            }

                    _nf_t = threading.Thread(target=_nf_run, daemon=True)
                    st.session_state.nf_thread = _nf_t
                    _nf_t.start()
                    st.rerun()
            elif nf_base_dir and _nf_base_path and not _nf_base_path.exists():
                st.error(f"❌ Папка не найдена: `{nf_base_dir}`")

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Извлечение таблиц (из chapters/)
# ══════════════════════════════════════════════════════════════════════
with tab_extract_tables:
    st.subheader("📊 Извлечение таблиц из глав")
    st.markdown(
        "Извлекает таблицы из файлов глав (`chapters/*.md`) **по маркерам** "
        "`[TABLE_START]`/`[TABLE_END]`.\n\n"
        "**Формат результата:** каждая таблица → файл с **заголовками главы** "
        "в начале + описание (caption) + тело таблицы. Теги `<NNN>` удаляются.\n\n"
        "Оригинальные `chapters/` **не изменяются**. "
        "Для очистки используйте вкладку «🧹 Очистка по маркерам».\n\n"
        "⚠️ **Сначала запустите «📍 Нормализация таблиц»** — без маркеров "
        "извлечение невозможно."
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    et_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("et_base_dir", "data/books"),
        key="et_base_dir",
        help="Структура: папка/название_книги/chapters/*.md",
    )

    et_recreate = st.checkbox(
        "🔄 Пересоздать (удалить extracted_tables/)",
        value=False,
        key="et_recreate",
        help="Если включено — папка extracted_tables/ удаляется перед запуском.",
    )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _et_base_path = Path(et_base_dir) if et_base_dir else None
    _et_available_books: list[str] = []

    if _et_base_path and _et_base_path.exists():
        _et_books = find_books_with_chapters(_et_base_path)
        _et_available_books = [name for name, _ in _et_books]

    et_selected_books: list[str] = []
    if _et_available_books:
        et_selected_books = st.multiselect(
            "📚 Выберите книги",
            options=_et_available_books,
            default=_et_available_books,
            key="et_selected_books",
        )

        # Предпросмотр: подсчёт таблиц (по кнопке)
        if et_selected_books:
            if st.button("📊 Подсчитать таблицы", key="et_preview_btn"):
                with st.spinner("Подсчёт таблиц в главах..."):
                    _et_counts = count_all_chapter_tables(_et_base_path, book_names=et_selected_books)
                if _et_counts["total_tables"] > 0:
                    st.info(
                        f"Найдено **{_et_counts['total_tables']}** промаркированных таблиц "
                        f"в **{_et_counts['files_with_tables']}** главах "
                        f"из **{_et_counts['total_files']}**"
                    )
                else:
                    st.success("✅ В выбранных главах нет таблиц.")
    elif _et_base_path and _et_base_path.exists():
        st.warning(
            f"В `{et_base_dir}` нет книг с `chapters/`. "
            f"Сначала выполните соединение по главам."
        )

    # ── Инициализация состояния прогресса ───────────────────────────
    if "et_thread" not in st.session_state:
        st.session_state.et_thread = None
    if "et_cancel" not in st.session_state:
        st.session_state.et_cancel = None
    if "et_progress" not in st.session_state:
        st.session_state.et_progress = None
    if "et_result" not in st.session_state:
        st.session_state.et_result = None

    @st.fragment(run_every="2s")
    def _render_et_progress():
        """Автообновляемый фрагмент: прогресс извлечения + отмена."""
        _thread = st.session_state.et_thread
        _is_running = _thread is not None and _thread.is_alive()
        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.et_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0
            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"📊 Таблиц извлечено: **{_progress.get('tables_extracted', 0)}**  |  "
                    f"📁 Файлов с таблицами: **{_progress.get('files_with_tables', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**"
                )

        if st.button("❌ Отменить извлечение", key="et_cancel_btn",
                      use_container_width=True, type="secondary"):
            if st.session_state.et_cancel:
                st.session_state.et_cancel.set()
            st.rerun()

    with tab_extract_tables:
        _et_thread = st.session_state.et_thread
        _et_is_running = _et_thread is not None and _et_thread.is_alive()

        if _et_is_running:
            _render_et_progress()

        elif st.session_state.et_result:
            _et_res = st.session_state.et_result
            if _et_res.get("cancelled"):
                st.warning(
                    f"⚠️ Извлечение отменено. "
                    f"Таблиц: {_et_res.get('tables_extracted', 0)}, "
                    f"Файлов: {_et_res.get('files_with_tables', 0)}. "
                    f"Время: {_et_res.get('elapsed_sec', '?')}с"
                )
            elif _et_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Таблиц: {_et_res.get('tables_extracted', 0)}, "
                    f"Ошибок: {_et_res.get('errors', 0)}. "
                    f"Время: {_et_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Извлечение завершено! "
                    f"Извлечено **{_et_res.get('tables_extracted', 0)}** таблиц "
                    f"из **{_et_res.get('files_with_tables', 0)}** файлов. "
                    f"Время: {_et_res.get('elapsed_sec', '?')}с"
                )
            # Страницы без описания
            _no_cap = _et_res.get("no_caption_pages", [])
            if _no_cap:
                st.warning(
                    f"⚠️ **{len(_no_cap)}** таблиц без описания (caption): "
                    + ", ".join(f"`{p}`" for p in _no_cap)
                )
            if st.button("🗑️ Сбросить результат", key="et_clear_result"):
                st.session_state.et_result = None
                st.session_state.et_progress = None
                st.rerun()

        else:
            _can_run_et = (
                et_base_dir and _et_base_path and _et_base_path.exists()
                and et_selected_books
            )
            if _can_run_et:
                if st.button(
                    f"📊 Извлечь таблицы из {len(et_selected_books)} "
                    f"{'книги' if len(et_selected_books) == 1 else 'книг'}",
                    key="et_run_btn", use_container_width=True, type="primary",
                ):
                    _et_cancel_ev = threading.Event()
                    st.session_state.et_cancel = _et_cancel_ev
                    st.session_state.et_progress = {
                        "current": 0, "total": 0,
                        "current_file": "подготовка...",
                        "files_with_tables": 0,
                        "tables_extracted": 0,
                        "errors": 0,
                    }
                    st.session_state.et_result = None

                    _et_sel = list(et_selected_books)
                    _et_base = et_base_dir
                    _et_recreate = et_recreate

                    def _et_progress_cb(current, total, current_file, stats):
                        st.session_state.et_progress = {
                            "current": current, "total": total,
                            "current_file": current_file, **stats,
                        }

                    def _et_run():
                        result = run_extract_tables_from_chapters(
                            base_dir=_et_base,
                            book_names=_et_sel,
                            recreate=_et_recreate,
                            cancel_event=_et_cancel_ev,
                            progress_callback=_et_progress_cb,
                        )
                        st.session_state.et_result = result

                    _et_t = threading.Thread(target=_et_run, daemon=True)
                    st.session_state.et_thread = _et_t
                    _et_t.start()
                    st.rerun()
            elif et_base_dir and _et_base_path and not _et_base_path.exists():
                st.error(f"❌ Папка не найдена: `{et_base_dir}`")

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Извлечение описаний рисунков
# ══════════════════════════════════════════════════════════════════════
with tab_extract_figures:
    st.subheader("🖼️ Описания рисунков из глав (по маркерам)")
    st.markdown(
        "Извлекает описания рисунков из файлов глав (`chapters/*.md`) "
        "**по маркерам** `[FIGURE_START]`/`[FIGURE_END]`.\n\n"
        "**Формат результата:** каждое описание → файл с **заголовками главы** "
        "в начале + текст описания. Теги `<NNN>` удаляются.\n\n"
        "**Имя файла:** приоритет — номер рисунка (`Рис. 77` → `077.md`), "
        "fallback — номер страницы из `<NNN>`.\n\n"
        "Оригинальные `chapters/` **не изменяются**. "
        "Для очистки используйте вкладку «🧹 Очистка по маркерам».\n\n"
        "⚠️ **Сначала запустите «🖼️ Нормализация рисунков»** — без маркеров "
        "извлечение невозможно."
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    ef_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("ef_base_dir", "data/books"),
        key="ef_base_dir",
        help="Структура: папка/название_книги/chapters/*.md",
    )

    ef_pattern = st.text_input(
        "🔍 Regex-паттерн для описания",
        value=st.session_state.get("ef_pattern", _DEFAULT_FIG_PATTERN),
        key="ef_pattern",
        help="Regex для первой строки абзаца-описания. "
             "По умолчанию: «Рис.» + номер.",
    )

    ef_recreate = st.checkbox(
        "🔄 Пересоздать (удалить parsed/extracted_figures/)",
        value=False,
        key="ef_recreate",
        help="Если включено — папка parsed/extracted_figures/ удаляется перед запуском. "
             "clear_chapters/ не удаляется (описания удаляются in-place).",
    )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _ef_base_path = Path(ef_base_dir) if ef_base_dir else None
    _ef_available_books: list[str] = []

    if _ef_base_path and _ef_base_path.exists():
        _ef_books = _find_ef_books(_ef_base_path)
        _ef_available_books = [name for name, _ in _ef_books]

    ef_selected_books: list[str] = []
    if _ef_available_books:
        ef_selected_books = st.multiselect(
            "📚 Выберите книги",
            options=_ef_available_books,
            default=_ef_available_books,
            key="ef_selected_books",
        )

        # Предпросмотр: подсчёт описаний (по кнопке)
        if ef_selected_books:
            if st.button("📊 Подсчитать описания рисунков", key="ef_preview_btn"):
                with st.spinner("Подсчёт описаний рисунков в clear_chapters..."):
                    _ef_counts = count_all_chapter_figures(
                        _ef_base_path, book_names=ef_selected_books,
                    )
                if _ef_counts["total_captions"] > 0:
                    st.info(
                        f"Найдено **{_ef_counts['total_captions']}** описаний рисунков "
                        f"в **{_ef_counts['files_with_captions']}** файлах "
                        f"из **{_ef_counts['total_files']}**"
                    )
                else:
                    st.success("✅ В выбранных очищенных главах нет описаний рисунков.")
    elif _ef_base_path and _ef_base_path.exists():
        st.warning(
            f"В `{ef_base_dir}` нет книг с `chapters/`. "
            f"Сначала выполните соединение по главам."
        )

    # ── Инициализация состояния прогресса ───────────────────────────
    if "ef_thread" not in st.session_state:
        st.session_state.ef_thread = None
    if "ef_cancel" not in st.session_state:
        st.session_state.ef_cancel = None
    if "ef_progress" not in st.session_state:
        st.session_state.ef_progress = None
    if "ef_result" not in st.session_state:
        st.session_state.ef_result = None

    @st.fragment(run_every="2s")
    def _render_ef_progress():
        """Автообновляемый фрагмент: прогресс извлечения рисунков + отмена."""
        _thread = st.session_state.ef_thread
        _is_running = _thread is not None and _thread.is_alive()
        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.ef_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0
            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"🖼️ Описаний извлечено: **{_progress.get('captions_extracted', 0)}**  |  "
                    f"📁 Файлов с описаниями: **{_progress.get('files_with_captions', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**"
                )

        if st.button("❌ Отменить извлечение рисунков", key="ef_cancel_btn",
                      use_container_width=True, type="secondary"):
            if st.session_state.ef_cancel:
                st.session_state.ef_cancel.set()
            st.rerun()

    with tab_extract_figures:
        _ef_thread = st.session_state.ef_thread
        _ef_is_running = _ef_thread is not None and _ef_thread.is_alive()

        if _ef_is_running:
            _render_ef_progress()

        elif st.session_state.ef_result:
            _ef_res = st.session_state.ef_result
            if _ef_res.get("cancelled"):
                st.warning(
                    f"⚠️ Извлечение отменено. "
                    f"Описаний: {_ef_res.get('captions_extracted', 0)}, "
                    f"Файлов: {_ef_res.get('files_with_captions', 0)}. "
                    f"Время: {_ef_res.get('elapsed_sec', '?')}с"
                )
            elif _ef_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Описаний: {_ef_res.get('captions_extracted', 0)}, "
                    f"Ошибок: {_ef_res.get('errors', 0)}. "
                    f"Время: {_ef_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Извлечение завершено! "
                    f"Извлечено **{_ef_res.get('captions_extracted', 0)}** описаний "
                    f"из **{_ef_res.get('files_with_captions', 0)}** файлов. "
                    f"Время: {_ef_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="ef_clear_result"):
                st.session_state.ef_result = None
                st.session_state.ef_progress = None
                st.rerun()

        else:
            _can_run_ef = (
                ef_base_dir and _ef_base_path and _ef_base_path.exists()
                and ef_selected_books
            )
            if _can_run_ef:
                if st.button(
                    f"🖼️ Извлечь описания рисунков из {len(ef_selected_books)} "
                    f"{'книги' if len(ef_selected_books) == 1 else 'книг'}",
                    key="ef_run_btn", use_container_width=True, type="primary",
                ):
                    _ef_cancel_ev = threading.Event()
                    st.session_state.ef_cancel = _ef_cancel_ev
                    st.session_state.ef_progress = {
                        "current": 0, "total": 0,
                        "current_file": "подготовка...",
                        "files_with_captions": 0,
                        "captions_extracted": 0,
                        "errors": 0,
                    }
                    st.session_state.ef_result = None

                    _ef_sel = list(ef_selected_books)
                    _ef_base = ef_base_dir
                    _ef_recreate = ef_recreate
                    _ef_pattern = ef_pattern

                    def _ef_progress_cb(current, total, current_file, stats):
                        st.session_state.ef_progress = {
                            "current": current, "total": total,
                            "current_file": current_file, **stats,
                        }

                    def _ef_run():
                        result = run_extract_figures_from_chapters(
                            base_dir=_ef_base,
                            book_names=_ef_sel,
                            recreate=_ef_recreate,
                            pattern=_ef_pattern,
                            cancel_event=_ef_cancel_ev,
                            progress_callback=_ef_progress_cb,
                        )
                        st.session_state.ef_result = result

                    _ef_t = threading.Thread(target=_ef_run, daemon=True)
                    st.session_state.ef_thread = _ef_t
                    _ef_t.start()
                    st.rerun()
            elif ef_base_dir and _ef_base_path and not _ef_base_path.exists():
                st.error(f"❌ Папка не найдена: `{ef_base_dir}`")

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Выравнивание заголовков по оглавлению
# ══════════════════════════════════════════════════════════════════════
with tab_fix_headings:
    st.subheader("📑 Выравнивание заголовков по оглавлению")
    st.markdown(
        "Исправляет уровни заголовков (`#`–`######`) в файлах `parsed/*.md` "
        "по данным из оглавления (`parsed/toc/*.md`). "
        "Заголовки, не найденные в оглавлении, лишаются `#` "
        "(становятся обычным текстом) — это убирает ложные заголовки "
        "(подписи к таблицам, маркеры продолжения и т.д.)."
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    fh_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("fh_base_dir", "data/books"),
        key="fh_base_dir",
        help="Структура: папка/название_книги/parsed/toc/*.md + parsed/*.md",
    )

    remove_unmatched = st.checkbox(
        "Удалять `#` у заголовков, не найденных в оглавлении",
        value=True,
        key="fh_remove_unmatched",
        help="Если включено, заголовки без совпадения в TOC "
             "станут обычным текстом (убирается `#`).",
    )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _fh_base_path = Path(fh_base_dir) if fh_base_dir else None
    _fh_available_books: list[str] = []

    if _fh_base_path and _fh_base_path.exists():
        _fh_books_with_toc = find_books_with_toc(_fh_base_path)
        _fh_available_books = [name for name, _ in _fh_books_with_toc]

    fh_selected_books: list[str] = []
    if _fh_available_books:
        fh_selected_books = st.multiselect(
            "📚 Выберите книги",
            options=_fh_available_books,
            default=_fh_available_books,
            key="fh_selected_books",
        )

        # Предпросмотр: подсчёт несовпадений + покрытие TOC (по кнопке)
        if fh_selected_books:
            if st.button("📊 Подсчитать несовпадения", key="fh_preview_btn"):
                with st.spinner("Подсчёт несовпадений заголовков..."):
                    _mismatches = count_all_mismatches(
                        _fh_base_path, book_names=fh_selected_books,
                    )
                if _mismatches["total_headings"] > 0:
                    _wrong = _mismatches["wrong_level"]
                    _not_toc = _mismatches["not_in_toc"]
                    st.info(
                        f"**{_mismatches['total_headings']}** заголовков в "
                        f"**{_mismatches['total_books']}** книгах: "
                        f"✅ {_mismatches['correct']} корректных, "
                        f"⚠️ {_wrong} неверный уровень, "
                        f"❌ {_not_toc} нет в оглавлении"
                    )
                else:
                    st.success("✅ Все заголовки уже корректны.")

                # Покрытие TOC: какие записи оглавления не найдены в тексте
                _unmatched_toc = _mismatches.get("unmatched_toc_count", 0)
                _total_toc = _mismatches.get("total_toc_entries", 0)
                if _total_toc > 0:
                    _matched_toc = _total_toc - _unmatched_toc
                    if _unmatched_toc > 0:
                        st.warning(
                            f"📖 Оглавление → текст: "
                            f"**{_matched_toc}/{_total_toc}** найдено, "
                            f"**❌ {_unmatched_toc}** не найдено в заголовках"
                        )
                        with st.expander(
                            f"📋 Не найденные записи оглавления ({_unmatched_toc})",
                            expanded=False,
                        ):
                            _details = _mismatches.get("unmatched_toc_details", [])
                            for _det in _details:
                                _prefix = "#" * _det["toc_level"]
                                st.markdown(
                                    f"- `{_det['book']}`: "
                                    f"`{_prefix}` `{_det['toc_text']}`"
                                )
                    else:
                        st.success(
                            f"📖 Оглавление → текст: "
                            f"**все {_total_toc}** записей найдены ✅"
                        )
    elif _fh_base_path and _fh_base_path.exists():
        st.warning(
            f"В `{fh_base_dir}` нет книг с `parsed/toc/`. "
            f"Сначала извлеките оглавление."
        )

    # ── Инициализация состояния прогресса ───────────────────────────
    if "fh_thread" not in st.session_state:
        st.session_state.fh_thread = None
    if "fh_cancel" not in st.session_state:
        st.session_state.fh_cancel = None
    if "fh_progress" not in st.session_state:
        st.session_state.fh_progress = None
    if "fh_result" not in st.session_state:
        st.session_state.fh_result = None

    @st.fragment(run_every="2s")
    def _render_fh_progress():
        """Автообновляемый фрагмент: прогресс выравнивания + отмена."""
        _thread = st.session_state.fh_thread
        _is_running = _thread is not None and _thread.is_alive()
        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.fh_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0
            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"✏️ Изменено: **{_progress.get('modified', 0)}**  |  "
                    f"⏩ Без изменений: **{_progress.get('unchanged', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**"
                )
                st.write(
                    f"🔧 Уровней исправлено: **{_progress.get('fixed_level', 0)}**  |  "
                    f"🗑️ # удалено: **{_progress.get('removed_hash', 0)}**"
                )

        if st.button("❌ Отменить выравнивание", key="fh_cancel_btn",
                      use_container_width=True, type="secondary"):
            if st.session_state.fh_cancel:
                st.session_state.fh_cancel.set()
            st.rerun()

    with tab_fix_headings:
        _fh_thread = st.session_state.fh_thread
        _fh_is_running = _fh_thread is not None and _fh_thread.is_alive()

        if _fh_is_running:
            _render_fh_progress()

        elif st.session_state.fh_result:
            _fh_res = st.session_state.fh_result
            if _fh_res.get("cancelled"):
                st.warning(
                    f"⚠️ Выравнивание отменено. "
                    f"Изменено: {_fh_res.get('modified', 0)}/{_fh_res.get('total', 0)}. "
                    f"Время: {_fh_res.get('elapsed_sec', '?')}с"
                )
            elif _fh_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Изменено: {_fh_res.get('modified', 0)}, "
                    f"Ошибок: {_fh_res.get('errors', 0)}. "
                    f"Время: {_fh_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Выравнивание завершено! "
                    f"Изменено: {_fh_res.get('modified', 0)} файлов "
                    f"(уровней: {_fh_res.get('fixed_level', 0)}, "
                    f"# удалено: {_fh_res.get('removed_hash', 0)}). "
                    f"Время: {_fh_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="fh_clear_result"):
                st.session_state.fh_result = None
                st.session_state.fh_progress = None
                st.rerun()

        else:
            _can_run_fh = (
                fh_base_dir and _fh_base_path and _fh_base_path.exists()
                and fh_selected_books
            )
            if _can_run_fh:
                if st.button(
                    f"📑 Выровнять заголовки в {len(fh_selected_books)} "
                    f"{'книге' if len(fh_selected_books) == 1 else 'книгах'}",
                    key="fh_run_btn", use_container_width=True, type="primary",
                ):
                    _fh_cancel_ev = threading.Event()
                    st.session_state.fh_cancel = _fh_cancel_ev
                    st.session_state.fh_progress = {
                        "current": 0, "total": 0,
                        "current_file": "подготовка...",
                        "modified": 0, "unchanged": 0, "errors": 0,
                        "fixed_level": 0, "removed_hash": 0,
                    }
                    st.session_state.fh_result = None

                    _fh_sel = list(fh_selected_books)
                    _fh_base = fh_base_dir
                    _fh_remove = remove_unmatched

                    def _fh_progress_cb(current, total, current_file, stats):
                        st.session_state.fh_progress = {
                            "current": current, "total": total,
                            "current_file": current_file, **stats,
                        }

                    def _fh_run():
                        result = run_fix_heading_levels(
                            base_dir=_fh_base,
                            book_names=_fh_sel,
                            remove_unmatched=_fh_remove,
                            cancel_event=_fh_cancel_ev,
                            progress_callback=_fh_progress_cb,
                        )
                        st.session_state.fh_result = result

                    _fh_t = threading.Thread(target=_fh_run, daemon=True)
                    st.session_state.fh_thread = _fh_t
                    _fh_t.start()
                    st.rerun()
            elif fh_base_dir and _fh_base_path and not _fh_base_path.exists():
                st.error(f"❌ Папка не найдена: `{fh_base_dir}`")

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Нумерация страниц
# ══════════════════════════════════════════════════════════════════════
with tab_numbering:
    st.subheader("🔢 Расстановка номеров страниц")
    st.markdown(
        "Добавляет тег `<N>` перед каждым абзацем в Markdown-файлах. "
        "Номер `N` извлекается из имени файла (напр. `005.md` → `<5>`). "
        "Заголовки (`# ...`) и пустые строки пропускаются.\n\n"
        "Источники: `название_книги/parsed/*.md` → "
        "результат: `название_книги/numbered/*.md`"
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    num_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("numbering_base_dir", "data/books"),
        key="numbering_base_dir",
        help="Структура: папка/название_книги/parsed/*.md → "
             "папка/название_книги/numbered/*.md",
    )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _num_base_path = Path(num_base_dir) if num_base_dir else None
    _num_available_books: list[str] = []

    if _num_base_path and _num_base_path.exists() and _num_base_path.is_dir():
        for _book_dir in sorted(_num_base_path.iterdir()):
            if _book_dir.is_dir() and (_book_dir / "parsed").is_dir():
                _num_available_books.append(_book_dir.name)

    num_selected_books: list[str] = []
    if _num_available_books:
        num_selected_books = st.multiselect(
            "📚 Выберите книги для нумерации",
            options=_num_available_books,
            default=_num_available_books,
            key="num_selected_books",
            help="Выберите книги, в которых нужно расставить номера страниц.",
        )
    elif num_base_dir and _num_base_path and _num_base_path.exists():
        st.warning(
            f"В `{num_base_dir}` не найдено папок с `parsed/`. "
            f"Сначала выполните AI-парсинг."
        )

    # Предпросмотр: сколько файлов в выбранных книгах (по кнопке)
    if num_selected_books and _num_base_path and _num_base_path.exists():
        if st.button("📊 Подсчитать файлы", key="num_preview_btn"):
            _preview_files = _find_md_files(_num_base_path, book_names=num_selected_books)
            if _preview_files:
                st.info(
                    f"Будет обработано: **{len(_preview_files)}** .md файлов "
                    f"из **{len(num_selected_books)}** "
                    f"{'книг' if len(num_selected_books) > 1 else 'книги'}"
                )
            else:
                st.warning("В выбранных книгах нет .md файлов в `parsed/`.")
    elif _num_available_books and not num_selected_books:
        st.warning("⚠️ Выберите хотя бы одну книгу.")
    elif num_base_dir and _num_base_path and not _num_base_path.exists():
        st.error(f"❌ Папка не найдена: `{num_base_dir}`")

    # ── Инициализация состояния прогресса ───────────────────────────
    if "numbering_thread" not in st.session_state:
        st.session_state.numbering_thread = None
    if "numbering_cancel" not in st.session_state:
        st.session_state.numbering_cancel = None
    if "numbering_progress" not in st.session_state:
        st.session_state.numbering_progress = None
    if "numbering_result" not in st.session_state:
        st.session_state.numbering_result = None


    @st.fragment(run_every="2s")
    def _render_numbering_progress():
        """Автообновляемый фрагмент: прогресс нумерации + кнопка отмены."""
        _thread = st.session_state.numbering_thread
        _is_running = _thread is not None and _thread.is_alive()

        if not _is_running:
            st.rerun()
            return

        # ── Прогресс ────────────────────────────────────────────────
        _progress = st.session_state.numbering_progress
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

        # ── Кнопка отмены ───────────────────────────────────────────
        if st.button(
            "❌ Отменить нумерацию",
            key="numbering_cancel_btn",
            use_container_width=True,
            type="secondary",
        ):
            if st.session_state.numbering_cancel:
                st.session_state.numbering_cancel.set()
            st.rerun()


    with tab_numbering:
        _thread = st.session_state.numbering_thread
        _is_running = _thread is not None and _thread.is_alive()

        if _is_running:
            _render_numbering_progress()

        elif st.session_state.numbering_result:
            # ── Результат ───────────────────────────────────────────
            _res = st.session_state.numbering_result
            if _res.get("cancelled"):
                st.warning(
                    f"⚠️ Нумерация отменена. "
                    f"Успешно: {_res.get('success', 0)}/{_res.get('total', 0)}, "
                    f"Ошибок: {_res.get('errors', 0)}, "
                    f"Пропущено: {_res.get('skipped', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с"
                )
            elif _res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Успешно: {_res.get('success', 0)}/{_res.get('total', 0)}, "
                    f"Ошибок: {_res.get('errors', 0)}, "
                    f"Пропущено: {_res.get('skipped', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Нумерация завершена! "
                    f"Обработано: {_res.get('success', 0)} файлов, "
                    f"Пропущено: {_res.get('skipped', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="numbering_clear_result"):
                st.session_state.numbering_result = None
                st.session_state.numbering_progress = None
                st.rerun()

        else:
            # ── Кнопка запуска ──────────────────────────────────────
            _can_run = (
                num_base_dir
                and _num_base_path
                and _num_base_path.exists()
                and num_selected_books
            )
            if _can_run:
                if st.button(
                    f"🔢 Расставить номера в {len(num_selected_books)} "
                    f"{'книге' if len(num_selected_books) == 1 else 'книгах'}",
                    key="numbering_run_btn",
                    use_container_width=True,
                    type="primary",
                ):
                    cancel_event = threading.Event()
                    st.session_state.numbering_cancel = cancel_event
                    st.session_state.numbering_progress = {
                        "current": 0,
                        "total": 0,
                        "current_file": "подготовка...",
                        "success": 0,
                        "errors": 0,
                        "skipped": 0,
                    }
                    st.session_state.numbering_result = None

                    _num_sel = list(num_selected_books)
                    _num_base = num_base_dir

                    def _progress_cb(current, total, current_file, stats):
                        st.session_state.numbering_progress = {
                            "current": current,
                            "total": total,
                            "current_file": current_file,
                            "success": stats["success"],
                            "errors": stats["errors"],
                            "skipped": stats.get("skipped", 0),
                        }

                    def _run():
                        result = run_page_numbering(
                            base_dir=_num_base,
                            book_names=_num_sel,
                            cancel_event=cancel_event,
                            progress_callback=_progress_cb,
                        )
                        st.session_state.numbering_result = result

                    thread = threading.Thread(target=_run, daemon=True)
                    st.session_state.numbering_thread = thread
                    thread.start()
                    st.rerun()

            elif num_base_dir and _num_base_path and not _num_base_path.exists():
                st.error(f"❌ Папка не найдена: `{num_base_dir}`")

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Соединение по главам
# ══════════════════════════════════════════════════════════════════════
with tab_chapters:
    st.subheader("📚 Соединение файлов по главам")
    st.markdown(
        "Склеивает пронумерованные `.md` файлы в главы по заголовкам. "
        "Блоки, разорванные на границе страниц, объединяются с учётом тегов `<N>` "
        "и переносов слов через дефис.\n\n"
        "Родительские заголовки (уровни выше выбранного) дублируются в каждый файл. "
        "Имя файла — только из заголовка выбранного уровня.\n\n"
        "Источники: `название_книги/numbered/*.md` → "
        "результат: `название_книги/chapters/01_Название.md`"
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    ch_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("chapters_base_dir", "data/books"),
        key="chapters_base_dir",
        help="Структура: папка/название_книги/numbered/*.md → "
             "папка/название_книги/chapters/*.md",
    )

    # ── Уровень разбиения ────────────────────────────────────────────
    _split_options = {
        "🔄 Авто (самый нижний уровень)": 0,
        "# (уровень 1)": 1,
        "## (уровень 2)": 2,
        "### (уровень 3)": 3,
        "#### (уровень 4)": 4,
        "##### (уровень 5)": 5,
        "###### (уровень 6)": 6,
    }
    _split_label = st.selectbox(
        "✂️ Уровень разбиения",
        options=list(_split_options.keys()),
        index=0,
        key="chapters_split_level_label",
        help="«Авто» — разделение по самому глубокому уровню заголовков в тексте. "
             "При ручном выборе заголовки более высоких уровней дублируются в каждый файл, "
             "а заголовки более низких уровней — очищаются от маркеров #.",
    )
    ch_split_level = _split_options[_split_label]
    if ch_split_level == 0:
        st.info("🔄 Авто: для каждой книги будет определён самый глубокий уровень заголовков.")

    # ── Выбор книг ──────────────────────────────────────────────────
    st.divider()

    _ch_base_path = Path(ch_base_dir) if ch_base_dir else None
    _ch_available_books: list[str] = []

    if _ch_base_path and _ch_base_path.exists() and _ch_base_path.is_dir():
        _ch_books = _find_chapter_books(_ch_base_path)
        _ch_available_books = sorted(b[0] for b in _ch_books)

    ch_selected_books: list[str] = []
    if _ch_available_books:
        ch_selected_books = st.multiselect(
            "📚 Выберите книги для соединения",
            options=_ch_available_books,
            default=_ch_available_books,
            key="ch_selected_books",
            help="Выберите книги, в которых нужно соединить файлы по главам.",
        )
    elif ch_base_dir and _ch_base_path and _ch_base_path.exists():
        st.warning(
            f"В `{ch_base_dir}` не найдено папок с `numbered/`. "
            f"Сначала выполните нумерацию страниц."
        )
    elif ch_base_dir and _ch_base_path and not _ch_base_path.exists():
        st.error(f"❌ Папка не найдена: `{ch_base_dir}`")

    # ── Инициализация состояния прогресса ───────────────────────────
    if "chapters_thread" not in st.session_state:
        st.session_state.chapters_thread = None
    if "chapters_cancel" not in st.session_state:
        st.session_state.chapters_cancel = None
    if "chapters_progress" not in st.session_state:
        st.session_state.chapters_progress = None
    if "chapters_result" not in st.session_state:
        st.session_state.chapters_result = None

    @st.fragment(run_every="2s")
    def _render_chapters_progress():
        """Автообновляемый фрагмент: прогресс соединения + кнопка отмены."""
        _thread = st.session_state.chapters_thread
        _is_running = _thread is not None and _thread.is_alive()

        if not _is_running:
            st.rerun()
            return

        # ── Прогресс ────────────────────────────────────────────────
        _progress = st.session_state.chapters_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0

            st.progress(_pct, text=f"Обработано {_current}/{_total} книг")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущая книга: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"✅ Успешно: **{_progress.get('success', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**  |  "
                    f"⏩ Пропущено: **{_progress.get('skipped', 0)}**"
                )

        # ── Кнопка отмены ───────────────────────────────────────────
        if st.button(
            "❌ Отменить соединение",
            key="chapters_cancel_btn",
            use_container_width=True,
            type="secondary",
        ):
            if st.session_state.chapters_cancel:
                st.session_state.chapters_cancel.set()
            st.rerun()

    with tab_chapters:
        _ch_thread = st.session_state.chapters_thread
        _ch_is_running = _ch_thread is not None and _ch_thread.is_alive()

        if _ch_is_running:
            _render_chapters_progress()

        elif st.session_state.chapters_result:
            # ── Результат ───────────────────────────────────────────
            _res = st.session_state.chapters_result
            if _res.get("cancelled"):
                st.warning(
                    f"⚠️ Соединение отменено. "
                    f"Успешно: {_res.get('success', 0)}/{_res.get('total', 0)}, "
                    f"Ошибок: {_res.get('errors', 0)}, "
                    f"Пропущено: {_res.get('skipped', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с"
                )
            elif _res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Успешно: {_res.get('success', 0)}/{_res.get('total', 0)}, "
                    f"Ошибок: {_res.get('errors', 0)}, "
                    f"Пропущено: {_res.get('skipped', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Соединение завершено! "
                    f"Обработано: {_res.get('success', 0)} книг, "
                    f"Пропущено: {_res.get('skipped', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="chapters_clear_result"):
                st.session_state.chapters_result = None
                st.session_state.chapters_progress = None
                st.rerun()

        else:
            # ── Кнопка запуска ──────────────────────────────────────
            _ch_can_run = (
                ch_base_dir
                and _ch_base_path
                and _ch_base_path.exists()
                and ch_selected_books
            )
            if _ch_can_run:
                if st.button(
                    f"📚 Соединить по главам в {len(ch_selected_books)} "
                    f"{'книге' if len(ch_selected_books) == 1 else 'книгах'}",
                    key="chapters_run_btn",
                    use_container_width=True,
                    type="primary",
                ):
                    ch_cancel_event = threading.Event()
                    st.session_state.chapters_cancel = ch_cancel_event
                    st.session_state.chapters_progress = {
                        "current": 0,
                        "total": 0,
                        "current_file": "подготовка...",
                        "success": 0,
                        "errors": 0,
                        "skipped": 0,
                    }
                    st.session_state.chapters_result = None

                    def _ch_progress_cb(current, total, current_file, stats):
                        st.session_state.chapters_progress = {
                            "current": current,
                            "total": total,
                            "current_file": current_file,
                            "success": stats["success"],
                            "errors": stats["errors"],
                            "skipped": stats.get("skipped", 0),
                        }

                    _ch_sel = list(ch_selected_books)

                    def _ch_run():
                        result = run_chapter_merger(
                            base_dir=ch_base_dir,
                            split_level=ch_split_level,
                            book_names=_ch_sel,
                            cancel_event=ch_cancel_event,
                            progress_callback=_ch_progress_cb,
                        )
                        st.session_state.chapters_result = result

                    ch_thread = threading.Thread(target=_ch_run, daemon=True)
                    st.session_state.chapters_thread = ch_thread
                    ch_thread.start()
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Очистка по маркерам
# ══════════════════════════════════════════════════════════════════════
with tab_clean_tagged:
    st.subheader("🧹 Очистка по маркерам")
    st.markdown(
        "Удаляет блоки `[TABLE_START]`/`[TABLE_END]` и "
        "`[FIGURE_START]`/`[FIGURE_END]` из файлов `chapters/*.md`, "
        "сохраняя очищенные файлы в `parsed/clear_chapters/`.\n\n"
        "Оригинальные `chapters/` **не изменяются** — результат записывается "
        "в отдельную папку.\n\n"
        "⚠️ **Запускайте после** «Нормализация таблиц» и "
        "«Нормализация рисунков» — именно они создают маркеры."
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    ct_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("ct_base_dir", "data/books"),
        key="ct_base_dir",
        help="Структура: папка/название_книги/chapters/*.md",
    )

    ct_recreate = st.checkbox(
        "🔄 Пересоздать (удалить clear_chapters/)",
        value=False,
        key="ct_recreate",
        help="Если включено — папка parsed/clear_chapters/ удаляется перед запуском.",
    )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _ct_base_path = Path(ct_base_dir) if ct_base_dir else None
    _ct_available_books: list[str] = []

    if _ct_base_path and _ct_base_path.exists():
        _ct_books = _find_ct_books(_ct_base_path)
        _ct_available_books = [name for name, _ in _ct_books]

    ct_selected_books: list[str] = []
    if _ct_available_books:
        ct_selected_books = st.multiselect(
            "📚 Выберите книги",
            options=_ct_available_books,
            default=_ct_available_books,
            key="ct_selected_books",
        )

        # Предпросмотр: подсчёт маркированных блоков (по кнопке)
        if ct_selected_books:
            if st.button("📊 Подсчитать маркированные блоки", key="ct_preview_btn"):
                with st.spinner("Подсчёт блоков в chapters..."):
                    _ct_counts = count_all_tagged_blocks(
                        _ct_base_path, book_names=ct_selected_books,
                    )
                _ct_total = _ct_counts["total_tables"] + _ct_counts["total_figures"]
                if _ct_total > 0:
                    st.info(
                        f"Найдено **{_ct_total}** маркированных блоков "
                        f"в **{_ct_counts['total_files']}** файлах: "
                        f"📊 {_ct_counts['total_tables']} таблиц, "
                        f"🖼️ {_ct_counts['total_figures']} рисунков"
                    )
                else:
                    st.success("✅ В выбранных главах нет маркированных блоков.")
    elif _ct_base_path and _ct_base_path.exists():
        st.warning(
            f"В `{ct_base_dir}` нет книг с `chapters/`. "
            f"Сначала выполните соединение по главам."
        )

    # ── Инициализация состояния прогресса ───────────────────────────
    if "ct_thread" not in st.session_state:
        st.session_state.ct_thread = None
    if "ct_cancel" not in st.session_state:
        st.session_state.ct_cancel = None
    if "ct_progress" not in st.session_state:
        st.session_state.ct_progress = None
    if "ct_result" not in st.session_state:
        st.session_state.ct_result = None

    @st.fragment(run_every="2s")
    def _render_ct_progress():
        """Автообновляемый фрагмент: прогресс очистки по маркерам + отмена."""
        _thread = st.session_state.ct_thread
        _is_running = _thread is not None and _thread.is_alive()
        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.ct_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0
            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"📊 Таблиц удалено: **{_progress.get('tables_removed', 0)}**  |  "
                    f"🖼️ Рисунков удалено: **{_progress.get('figures_removed', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**"
                )

        if st.button("❌ Отменить очистку", key="ct_cancel_btn",
                      use_container_width=True, type="secondary"):
            if st.session_state.ct_cancel:
                st.session_state.ct_cancel.set()
            st.rerun()

    with tab_clean_tagged:
        _ct_thread = st.session_state.ct_thread
        _ct_is_running = _ct_thread is not None and _ct_thread.is_alive()

        if _ct_is_running:
            _render_ct_progress()

        elif st.session_state.ct_result:
            _ct_res = st.session_state.ct_result
            if _ct_res.get("cancelled"):
                st.warning(
                    f"⚠️ Очистка отменена. "
                    f"Обработано: {_ct_res.get('files_processed', 0)}/{_ct_res.get('total', 0)}, "
                    f"Таблиц: {_ct_res.get('tables_removed', 0)}, "
                    f"Рисунков: {_ct_res.get('figures_removed', 0)}. "
                    f"Время: {_ct_res.get('elapsed_sec', '?')}с"
                )
            elif _ct_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Обработано: {_ct_res.get('files_processed', 0)}, "
                    f"Таблиц: {_ct_res.get('tables_removed', 0)}, "
                    f"Рисунков: {_ct_res.get('figures_removed', 0)}, "
                    f"Ошибок: {_ct_res.get('errors', 0)}. "
                    f"Время: {_ct_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Очистка завершена! "
                    f"Обработано **{_ct_res.get('files_processed', 0)}** файлов: "
                    f"удалено {_ct_res.get('tables_removed', 0)} таблиц, "
                    f"{_ct_res.get('figures_removed', 0)} рисунков. "
                    f"Время: {_ct_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="ct_clear_result"):
                st.session_state.ct_result = None
                st.session_state.ct_progress = None
                st.rerun()

        else:
            _can_run_ct = (
                ct_base_dir and _ct_base_path and _ct_base_path.exists()
                and ct_selected_books
            )
            if _can_run_ct:
                if st.button(
                    f"🧹 Очистить по маркерам в {len(ct_selected_books)} "
                    f"{'книге' if len(ct_selected_books) == 1 else 'книгах'}",
                    key="ct_run_btn", use_container_width=True, type="primary",
                ):
                    _ct_cancel_ev = threading.Event()
                    st.session_state.ct_cancel = _ct_cancel_ev
                    st.session_state.ct_progress = {
                        "current": 0, "total": 0,
                        "current_file": "подготовка...",
                        "files_processed": 0,
                        "tables_removed": 0,
                        "figures_removed": 0,
                        "errors": 0,
                    }
                    st.session_state.ct_result = None

                    _ct_sel = list(ct_selected_books)
                    _ct_base = ct_base_dir
                    _ct_recreate = ct_recreate

                    def _ct_progress_cb(current, total, current_file, stats):
                        st.session_state.ct_progress = {
                            "current": current, "total": total,
                            "current_file": current_file, **stats,
                        }

                    def _ct_run():
                        result = run_clean_tagged_content(
                            base_dir=_ct_base,
                            book_names=_ct_sel,
                            recreate=_ct_recreate,
                            cancel_event=_ct_cancel_ev,
                            progress_callback=_ct_progress_cb,
                        )
                        st.session_state.ct_result = result

                    _ct_t = threading.Thread(target=_ct_run, daemon=True)
                    st.session_state.ct_thread = _ct_t
                    _ct_t.start()
                    st.rerun()
            elif ct_base_dir and _ct_base_path and not _ct_base_path.exists():
                st.error(f"❌ Папка не найдена: `{ct_base_dir}`")

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Генерация сводок таблиц
# ══════════════════════════════════════════════════════════════════════
with tab_table_summary:
    st.subheader("📋 Генерация структурных сводок таблиц")
    st.markdown(
        "Создаёт **компактную сводку** каждой таблицы для векторизации.\n\n"
        "**Что делает:** из полной таблицы (HTML/Markdown) извлекает:\n"
        "1. 📌 Название (caption: «Таблица N. ...»)\n"
        "2. 📌 Заголовки столбцов\n"
        "3. 📌 Группы строк (подзаголовки)\n"
        "4. 📌 Ключевые значения (первый столбец: модели, минералы и т.д.)\n\n"
        "**Зачем:** сводка (~300-800 символов) даёт **информативные вектора** "
        "для поиска, а полная таблица остаётся в исходном файле для контекста LLM.\n\n"
        "Источник: `parsed/extracted_tables/*.md` → "
        "результат: `parsed/table_summaries/*.md`"
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    ts_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("ts_base_dir", "data/books"),
        key="ts_base_dir",
        help="Структура: папка/название_книги/parsed/extracted_tables/*.md",
    )

    ts_recreate = st.checkbox(
        "🔄 Пересоздать (удалить table_summaries/)",
        value=False,
        key="ts_recreate",
    )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _ts_base_path = Path(ts_base_dir) if ts_base_dir else None
    _ts_available_books: list[str] = []

    if _ts_base_path and _ts_base_path.exists():
        _ts_books = find_books_with_extracted_tables(_ts_base_path)
        _ts_available_books = [name for name, _ in _ts_books]

    ts_selected_books: list[str] = []
    if _ts_available_books:
        ts_selected_books = st.multiselect(
            "📚 Выберите книги",
            options=_ts_available_books,
            default=_ts_available_books,
            key="ts_selected_books",
        )

        # Предпросмотр
        if ts_selected_books:
            if st.button("📊 Подсчитать таблицы", key="ts_preview_btn"):
                with st.spinner("Подсчёт..."):
                    _ts_counts = count_all_summaries(_ts_base_path, book_names=ts_selected_books)
                if _ts_counts["total_files"] > 0:
                    st.info(
                        f"**{_ts_counts['total_files']}** таблиц. "
                        f"Сводок уже есть: **{_ts_counts['existing_summaries']}**, "
                        f"Ожидание: **{_ts_counts['pending']}**"
                    )
                else:
                    st.success("✅ Нет извлечённых таблиц.")
    elif _ts_base_path and _ts_base_path.exists():
        st.warning(
            "Нет книг с `parsed/extracted_tables/`. "
            "Сначала извлеките таблицы."
        )

    # ── Состояние прогресса ─────────────────────────────────────────
    if "ts_thread" not in st.session_state:
        st.session_state.ts_thread = None
    if "ts_cancel" not in st.session_state:
        st.session_state.ts_cancel = None
    if "ts_progress" not in st.session_state:
        st.session_state.ts_progress = None
    if "ts_result" not in st.session_state:
        st.session_state.ts_result = None

    @st.fragment(run_every="2s")
    def _render_ts_progress():
        _thread = st.session_state.ts_thread
        _is_running = _thread is not None and _thread.is_alive()
        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.ts_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0
            st.progress(_pct, text=f"Обработано {_current}/{_total} файлов")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"✅ Создано: **{_progress.get('generated', 0)}**  |  "
                    f"⏩ Пропущено: **{_progress.get('skipped', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**"
                )

        if st.button("❌ Отменить", key="ts_cancel_btn",
                      use_container_width=True, type="secondary"):
            if st.session_state.ts_cancel:
                st.session_state.ts_cancel.set()
            st.rerun()

    with tab_table_summary:
        _ts_thread = st.session_state.ts_thread
        _ts_is_running = _ts_thread is not None and _ts_thread.is_alive()

        if _ts_is_running:
            _render_ts_progress()

        elif st.session_state.ts_result:
            _ts_res = st.session_state.ts_result
            if _ts_res.get("cancelled"):
                st.warning(
                    f"⚠️ Отменено. "
                    f"Создано: {_ts_res.get('generated', 0)}, "
                    f"Пропущено: {_ts_res.get('skipped', 0)}. "
                    f"Время: {_ts_res.get('elapsed_sec', '?')}с"
                )
            elif _ts_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Создано: {_ts_res.get('generated', 0)}, "
                    f"Ошибок: {_ts_res.get('errors', 0)}. "
                    f"Время: {_ts_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Сводки созданы! "
                    f"**{_ts_res.get('generated', 0)}** сводок "
                    f"(пропущено: {_ts_res.get('skipped', 0)}). "
                    f"Время: {_ts_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить", key="ts_clear_result"):
                st.session_state.ts_result = None
                st.session_state.ts_progress = None
                st.rerun()

        else:
            _can_run_ts = (
                ts_base_dir and _ts_base_path and _ts_base_path.exists()
                and ts_selected_books
            )
            if _can_run_ts:
                if st.button(
                    f"📋 Создать сводки для {len(ts_selected_books)} "
                    f"{'книги' if len(ts_selected_books) == 1 else 'книг'}",
                    key="ts_run_btn", use_container_width=True, type="primary",
                ):
                    _ts_cancel_ev = threading.Event()
                    st.session_state.ts_cancel = _ts_cancel_ev
                    st.session_state.ts_progress = {
                        "current": 0, "total": 0,
                        "current_file": "подготовка...",
                        "generated": 0, "skipped": 0, "errors": 0,
                    }
                    st.session_state.ts_result = None

                    _ts_sel = list(ts_selected_books)
                    _ts_base = ts_base_dir
                    _ts_rec = ts_recreate

                    def _ts_progress_cb(current, total, current_file, stats):
                        st.session_state.ts_progress = {
                            "current": current, "total": total,
                            "current_file": current_file, **stats,
                        }

                    def _ts_run():
                        result = run_table_summary(
                            base_dir=_ts_base,
                            book_names=_ts_sel,
                            recreate=_ts_rec,
                            cancel_event=_ts_cancel_ev,
                            progress_callback=_ts_progress_cb,
                        )
                        st.session_state.ts_result = result

                    _ts_t = threading.Thread(target=_ts_run, daemon=True)
                    st.session_state.ts_thread = _ts_t
                    _ts_t.start()
                    st.rerun()
            elif ts_base_dir and _ts_base_path and not _ts_base_path.exists():
                st.error(f"❌ Папка не найдена: `{ts_base_dir}`")

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Проверка размеров (сырые таблицы + готовые чанки)
# ══════════════════════════════════════════════════════════════════════
with tab_check_sizes:
    st.subheader("📏 Проверка размеров")

    # ── Режим проверки ──────────────────────────────────────────────
    _chk_mode = st.radio(
        "Режим проверки",
        options=["Сырые таблицы", "Готовые чанки"],
        index=0,
        key="chk_mode",
        horizontal=True,
        help=(
            "«Сырые таблицы» — размеры .md файлов из extracted/extracted_tables/. "
            "«Готовые чанки» — размеры текста в JSON из chunks/."
        ),
    )

    # ── Общие настройки ─────────────────────────────────────────────
    col_chk_dir, col_chk_size = st.columns([3, 1])

    with col_chk_dir:
        chk_base_dir = st.text_input(
            "Папка с книгами",
            value="data/books",
            key="chk_base_dir",
        )

    with col_chk_size:
        chk_chunk_size = st.number_input(
            "Порог (символы)",
            value=1200,
            min_value=100,
            max_value=10000,
            key="chk_chunk_size",
            help="Файлы/чанки длиннее этого значения будут отмечены",
        )

    _chk_base_path = Path(chk_base_dir) if chk_base_dir else None

    # ════════════════════════════════════════════════════════════════
    # Режим 1: Сырые таблицы
    # ════════════════════════════════════════════════════════════════
    if _chk_mode == "Сырые таблицы":
        st.markdown("""
        Подсчитывает **все символы** файлов таблиц из `extracted/extracted_tables/` —
        заголовки глав, описание таблицы (caption), сама таблица.
        Всё, что войдёт в чанк: пробелы, переносы строк, HTML-теги.

        Помогает оценить, какие таблицы превышают порог и будут векторизоваться через сводку (summary).
        """)

        _chk_table_dirs = find_book_extracted_table_dirs(_chk_base_path) if _chk_base_path and _chk_base_path.exists() else []
        if _chk_table_dirs:
            _chk_all_books = [bn for bn, _ in _chk_table_dirs]
            _chk_selected_books = st.multiselect(
                "Выберите книги",
                options=_chk_all_books,
                default=_chk_all_books,
                key="chk_book_select",
            )
        else:
            st.warning(f"📂 В папке `{chk_base_dir}` нет книг с `extracted/extracted_tables/`")
            _chk_selected_books = None

        st.divider()

        if st.button("📊 Проверить размеры", key="chk_run_btn",
                     use_container_width=True, type="primary"):
            if not _chk_selected_books:
                st.warning("⚠️ Выберите хотя бы одну книгу")
            elif _chk_base_path and _chk_base_path.exists():
                with st.spinner("🔍 Подсчёт размеров сырых таблиц..."):
                    _chk_result = check_raw_table_sizes(
                        base_dir=_chk_base_path,
                        chunk_size=int(chk_chunk_size),
                        book_names=_chk_selected_books,
                    )
                st.session_state.chk_result = _chk_result

        _chk_result = st.session_state.get("chk_result")
        if _chk_result:
            _total = _chk_result["total_chunks"]
            _over = _chk_result["oversized_count"]
            _stats = _chk_result.get("stats", {})

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Всего файлов", _total)
            col_m2.metric("Превышающих", _over,
                           delta=f"{_over / _total * 100:.1f}%" if _total else "0%",
                           delta_color="inverse")
            if _stats:
                col_m3.metric("Медиана", f"{_stats['median']} симв.")
                col_m4.metric("Макс", f"{_stats['max']} симв.")

            st.markdown("#### 📊 Распределение по размерам")
            _dist = _chk_result["distribution"]
            st.bar_chart(
                {"Диапазон": list(_dist.keys()), "Кол-во": list(_dist.values())},
                x="Диапазон", y="Кол-во", height=300,
            )

            _per_book = _chk_result.get("per_book", {})
            if _per_book:
                st.markdown("#### 📚 По книгам")
                st.dataframe(
                    [{"Книга": bn, "Всего": info["total"], "Превышающих": info["oversized"]}
                     for bn, info in _per_book.items()],
                    use_container_width=True, hide_index=True,
                )

            if _over > 0:
                st.markdown(f"#### ⚠️ Превышающие {chk_chunk_size} символов")
                st.dataframe(
                    [{"Книга": it["book"], "Файл": it["file"],
                      "Символов": it["chars"], "Превышение": f"+{it['over']}"}
                     for it in _chk_result["oversized"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success(f"✅ Все {_total} файлов укладываются в {chk_chunk_size} символов!")

    # ════════════════════════════════════════════════════════════════
    # Режим 2: Готовые чанки
    # ════════════════════════════════════════════════════════════════
    else:
        st.markdown("""
        Подсчитывает размеры **текста** (`chunk["text"]`) в готовых JSON-чанках
        из `chunks/{text_chunks,table_chunks,figure_chunks}/`.
        Это тот текст, который реально пойдёт в векторизацию.
        """)

        _chk_chunk_dirs = find_book_chunk_dirs(_chk_base_path) if _chk_base_path and _chk_base_path.exists() else []
        if _chk_chunk_dirs:
            _chk_all_chunk_books = [bn for bn, _ in _chk_chunk_dirs]
            _chk_selected_chunk_books = st.multiselect(
                "Выберите книги",
                options=_chk_all_chunk_books,
                default=_chk_all_chunk_books,
                key="chk_chunk_book_select",
            )
        else:
            st.warning(f"📂 В папке `{chk_base_dir}` нет книг с `chunks/`")
            _chk_selected_chunk_books = None

        st.divider()

        if st.button("📊 Проверить чанки", key="chk_chunk_run_btn",
                     use_container_width=True, type="primary"):
            if not _chk_selected_chunk_books:
                st.warning("⚠️ Выберите хотя бы одну книгу")
            elif _chk_base_path and _chk_base_path.exists():
                with st.spinner("🔍 Подсчёт размеров чанков..."):
                    _chk_chunk_result = check_created_chunk_sizes(
                        base_dir=_chk_base_path,
                        chunk_size=int(chk_chunk_size),
                        book_names=_chk_selected_chunk_books,
                    )
                st.session_state.chk_chunk_result = _chk_chunk_result

        _chk_chunk_result = st.session_state.get("chk_chunk_result")
        if _chk_chunk_result:
            _ct_total = _chk_chunk_result["total_chunks"]
            _ct_over = _chk_chunk_result["oversized_count"]
            _ct_stats = _chk_chunk_result.get("stats", {})
            _ct_by_type = _chk_chunk_result.get("by_type", {})

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("Всего чанков", _ct_total)
            col_m2.metric("Превышающих", _ct_over,
                           delta=f"{_ct_over / _ct_total * 100:.1f}%" if _ct_total else "0%",
                           delta_color="inverse")
            if _ct_stats:
                col_m3.metric("Медиана", f"{_ct_stats['median']} симв.")
                col_m4.metric("Макс", f"{_ct_stats['max']} симв.")
            if _ct_by_type:
                col_m5.metric(
                    "Текст / Таблицы / Рис.",
                    f"{_ct_by_type.get('text', 0)} / "
                    f"{_ct_by_type.get('table', 0)} / "
                    f"{_ct_by_type.get('figure', 0)}",
                )

            st.markdown("#### 📊 Распределение по размерам")
            _ct_dist = _chk_chunk_result["distribution"]
            st.bar_chart(
                {"Диапазон": list(_ct_dist.keys()), "Кол-во": list(_ct_dist.values())},
                x="Диапазон", y="Кол-во", height=300,
            )

            _ct_per_book = _chk_chunk_result.get("per_book", {})
            if _ct_per_book:
                st.markdown("#### 📚 По книгам")
                st.dataframe(
                    [{"Книга": bn,
                      "Всего": info["total"],
                      "Текст": info.get("text", 0),
                      "Таблицы": info.get("table", 0),
                      "Рис.": info.get("figure", 0),
                      "Превышающих": info["oversized"]}
                     for bn, info in _ct_per_book.items()],
                    use_container_width=True, hide_index=True,
                )

            if _ct_over > 0:
                st.markdown(f"#### ⚠️ Превышающие {chk_chunk_size} символов")
                st.dataframe(
                    [{"Книга": it["book"], "Файл": it["file"],
                      "Тип": it.get("chunk_type", "?"),
                      "Символов": it["chars"], "Превышение": f"+{it['over']}"}
                     for it in _chk_chunk_result["oversized"]],
                    use_container_width=True, hide_index=True,
                )
            else:
                st.success(f"✅ Все {_ct_total} чанков укладываются в {chk_chunk_size} символов!")


# ══════════════════════════════════════════════════════════════════════
# Вкладка: LLM-сводки таблиц
# ══════════════════════════════════════════════════════════════════════
with tab_llm_summary:
    st.subheader("🤖 LLM-сводки таблиц")
    st.markdown(
        "Генерирует **текстовое описание** (summary) oversized таблиц через LLM. "
        "Результат сохраняется в `extracted/table_summaries/` отдельными файлами.\n\n"
        "**Зачем:** большие таблицы (>chunk_size) дают «размытые» вектора. "
        "LLM-сводка описывает таблицу словами, сохраняя все термины и значения.\n\n"
        "**Порядок:** после «📏 Проверка размеров» → перед «✂️ Деление на чанки»."
    )

    # ── Путь ────────────────────────────────────────────────────────
    ls_base_dir = st.text_input(
        "📂 Папка с книгами", value="data/books",
        key="ls_base_dir",
    )

    _ls_base_path = Path(ls_base_dir) if ls_base_dir else None

    # ── chunk_size ──────────────────────────────────────────────────
    ls_chunk_size = st.number_input(
        "chunk_size (порог)", value=1200, min_value=100, max_value=10000,
        key="ls_chunk_size",
        help="Таблицы длиннее этого значения будут отправлены в LLM",
    )

    # ── Выбор модели ────────────────────────────────────────────────
    _ls_model_names = list_model_names()
    ls_model_name = st.selectbox(
        "🧠 Модель LLM",
        options=_ls_model_names if _ls_model_names else ["—"],
        index=0,
        key="ls_model_name",
        help="Выберите модель из реестра (data/models.json)",
    )

    # ── Температура ─────────────────────────────────────────────────
    ls_temperature = st.slider(
        "🌡️ Температура", min_value=0.0, max_value=1.0,
        value=0.1, step=0.05, key="ls_temperature",
    )


    # ── Промпты ─────────────────────────────────────────────────────
    with st.expander("📝 Промпты", expanded=False):
        ls_system_prompt = st.text_area(
            "Системный промпт",
            value=DEFAULT_SYSTEM_PROMPT,
            key="ls_system_prompt", height=150,
        )
        ls_user_template = st.text_area(
            "Шаблон пользовательского промпта ({table} = тело таблицы)",
            value=DEFAULT_USER_PROMPT_TEMPLATE,
            key="ls_user_template", height=80,
        )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _ls_table_dirs = find_book_extracted_table_dirs(_ls_base_path) if _ls_base_path and _ls_base_path.exists() else []
    ls_selected_books: list[str] = []

    if _ls_table_dirs:
        _ls_all_books = list(dict.fromkeys(bn for bn, _ in _ls_table_dirs))
        ls_selected_books = st.multiselect(
            "📚 Выберите книги",
            options=_ls_all_books, default=_ls_all_books,
            key="ls_selected_books",
        )

        # Предпросмотр: подсчёт oversized без LLM-сводки
        if ls_selected_books:
            if st.button("📊 Подсчитать oversized таблицы", key="ls_preview_btn"):
                _ls_oversized = find_oversized_tables(
                    _ls_base_path, int(ls_chunk_size), ls_selected_books,
                )
                if _ls_oversized:
                    st.info(
                        f"Найдено **{len(_ls_oversized)}** oversized таблиц "
                        f"без LLM-сводки в `table_summaries/`"
                    )
                    with st.expander("📋 Список", expanded=False):
                        for _it in _ls_oversized[:30]:
                            st.markdown(
                                f"- `{_it['book']}/{_it['file'].name}` "
                                f"— **{_it['chars']}** симв."
                            )
                        if len(_ls_oversized) > 30:
                            st.caption(f"... и ещё {len(_ls_oversized) - 30}")
                else:
                    st.success("✅ Все oversized таблицы уже имеют LLM-сводку.")
    elif _ls_base_path and _ls_base_path.exists():
        st.warning(f"📂 В `{ls_base_dir}` нет книг с `extracted/extracted_tables/`")

    # ── Состояние прогресса ─────────────────────────────────────────
    if "ls_thread" not in st.session_state:
        st.session_state.ls_thread = None
    if "ls_cancel" not in st.session_state:
        st.session_state.ls_cancel = None
    if "ls_progress" not in st.session_state:
        st.session_state.ls_progress = None
    if "ls_result" not in st.session_state:
        st.session_state.ls_result = None

    @st.fragment(run_every="2s")
    def _render_ls_progress():
        _thread = st.session_state.ls_thread
        _is_running = _thread is not None and _thread.is_alive()
        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.ls_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0
            st.progress(_pct, text=f"Обработано {_current}/{_total} таблиц")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущий файл: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"✅ Обработано: **{_progress.get('processed', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**  |  "
                    f"🔢 Токенов: **{_progress.get('total_tokens', 0)}**"
                )

        if st.button("❌ Отменить", key="ls_cancel_btn",
                      use_container_width=True, type="secondary"):
            if st.session_state.ls_cancel:
                st.session_state.ls_cancel.set()
            st.rerun()

    with tab_llm_summary:
        _ls_thread = st.session_state.ls_thread
        _ls_is_running = _ls_thread is not None and _ls_thread.is_alive()

        if _ls_is_running:
            _render_ls_progress()

        elif st.session_state.ls_result:
            _ls_res = st.session_state.ls_result
            if _ls_res.get("cancelled"):
                st.warning(
                    f"⚠️ Отменено. "
                    f"Обработано: {_ls_res.get('processed', 0)}, "
                    f"Ошибок: {_ls_res.get('errors', 0)}. "
                    f"Время: {_ls_res.get('elapsed_sec', '?')}с"
                )
            elif _ls_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Обработано: {_ls_res.get('processed', 0)}, "
                    f"Ошибок: {_ls_res.get('errors', 0)}. "
                    f"Время: {_ls_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ LLM-сводки созданы! "
                    f"**{_ls_res.get('processed', 0)}** таблиц, "
                    f"токенов: {_ls_res.get('total_tokens', 0)}. "
                    f"Время: {_ls_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить", key="ls_clear_result"):
                st.session_state.ls_result = None
                st.session_state.ls_progress = None
                st.rerun()

        else:
            _can_run_ls = (
                ls_base_dir and _ls_base_path and _ls_base_path.exists()
                and ls_selected_books and _ls_model_names
            )
            if _can_run_ls:
                if st.button(
                    f"🤖 Генерировать LLM-сводки для {len(ls_selected_books)} "
                    f"{'книги' if len(ls_selected_books) == 1 else 'книг'}",
                    key="ls_run_btn", use_container_width=True, type="primary",
                ):
                    _ls_cancel_ev = threading.Event()
                    st.session_state.ls_cancel = _ls_cancel_ev
                    st.session_state.ls_progress = {
                        "current": 0, "total": 0,
                        "current_file": "подготовка...",
                        "processed": 0, "errors": 0, "total_tokens": 0,
                    }
                    st.session_state.ls_result = None

                    _ls_sel = list(ls_selected_books)
                    _ls_base = ls_base_dir
                    _ls_model = ls_model_name

                    def _ls_progress_cb(current, total, current_file, stats):
                        st.session_state.ls_progress = {
                            "current": current, "total": total,
                            "current_file": current_file, **stats,
                        }

                    def _ls_run():
                        result = run_llm_table_summary(
                            base_dir=_ls_base,
                            chunk_size=int(ls_chunk_size),
                            model_name=_ls_model,
                            system_prompt=ls_system_prompt,
                            user_prompt_template=ls_user_template,
                            temperature=ls_temperature,
                            book_names=_ls_sel,
                            cancel_event=_ls_cancel_ev,
                            progress_callback=_ls_progress_cb,
                        )
                        st.session_state.ls_result = result

                    _ls_t = threading.Thread(target=_ls_run, daemon=True)
                    st.session_state.ls_thread = _ls_t
                    _ls_t.start()
                    st.rerun()
            elif ls_base_dir and _ls_base_path and not _ls_base_path.exists():
                st.error(f"❌ Папка не найдена: `{ls_base_dir}`")
            elif not _ls_model_names:
                st.warning("⚠️ Нет моделей в реестре. Добавьте на странице «Парсинг» → «Управление моделями».")

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Деление на чанки
# ══════════════════════════════════════════════════════════════════════
with tab_chunks:
    st.subheader("✂️ Деление на чанки")
    st.markdown(
        "Независимое чанкование **текста**, **таблиц** и **описаний рисунков**. "
        "Каждый тип — в отдельную папку с единой схемой JSON.\n\n"
        "Единая схема метаданных: `book_name`, `source_file`, `chunk_type`, "
        "`page_numbers`, `headings`.\n\n"
        "Результаты в `название_книги/chunks/`:\n"
        "- `text_chunks/*.json` — из `extracted/clear_chapters/*.md` (очищенные главы без таблиц и рисунков)\n"
        "- `table_chunks/*.json` — из `extracted/extracted_tables/*.md` (номер страницы из имени файла)\n"
        "- `figure_chunks/*.json` — из `extracted/extracted_figures/*.md` (номер страницы из имени файла)"
    )

    # ── Режим чанкования ─────────────────────────────────────────────
    _ck_modes = {
        "📝 Текст": "text",
        "📋 Таблицы": "table",
        "🖼️ Описания рисунков": "figure",
    }
    _ck_mode_label = st.radio(
        "Тип чанкования",
        options=list(_ck_modes.keys()),
        index=0,
        key="chunks_mode",
        horizontal=True,
    )
    ck_mode = _ck_modes[_ck_mode_label]

    # ── Путь к базовой папке ────────────────────────────────────────
    ck_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("chunks_base_dir", "data/books"),
        key="chunks_base_dir",
    )

    # ── Параметры чанкования ─────────────────────────────────────────
    with st.expander("⚙️ Параметры чанкования", expanded=False):
        _ck_size_help = {
            "text": "Максимальный размер текстового чанка в символах",
            "table": "Порог: таблицы ≤ threshold векторизуются целиком, "
                     "> threshold — через сводку (summary)",
            "figure": "Максимальный размер чанка описания рисунка",
        }
        ck_chunk_size = st.slider(
            "Размер чанка (символы)"
            if ck_mode == "text" else
            "Порог размера таблицы (символы)"
            if ck_mode == "table" else
            "Макс. размер описания (символы)",
            min_value=200, max_value=5000, value=1200, step=100,
            key="chunks_chunk_size",
            help=_ck_size_help.get(ck_mode, ""),
        )
        ck_overlap = st.slider(
            "Перекрытие (overlap)",
            min_value=0, max_value=1000, value=200, step=50,
            key="chunks_overlap",
        )
        ck_min_chunk_size = st.slider(
            "Мин. размер последнего чанка",
            min_value=0, max_value=1000, value=400, step=50,
            key="chunks_min_chunk_size",
        )
    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _ck_base = Path(ck_base_dir) if ck_base_dir else None
    _ck_available_books: list[str] = []

    if _ck_base and _ck_base.exists():
        if ck_mode == "text":
            _ck_books = _find_chunk_books(_ck_base, "extracted/clear_chapters")
            _ck_available_books = sorted(b[0] for b in _ck_books)
            if not _ck_available_books:
                st.warning("Нет книг с папкой `extracted/clear_chapters/`. Сначала выполните очистку по маркерам.")
        elif ck_mode == "table":
            _ck_t_books = _find_chunk_books(_ck_base, "extracted/extracted_tables")
            _ck_available_books = sorted(n for n, _ in _ck_t_books)
            if not _ck_available_books:
                st.warning("Нет книг с `extracted/extracted_tables/`. Сначала извлеките таблицы.")
        elif ck_mode == "figure":
            _ck_f_books = _find_chunk_books(_ck_base, "extracted/extracted_figures")
            _ck_available_books = sorted(n for n, _ in _ck_f_books)
            if not _ck_available_books:
                st.warning("Нет книг с `extracted/extracted_figures/`. Сначала извлеките описания.")
    elif ck_base_dir:
        st.error(f"❌ Папка не найдена: `{ck_base_dir}`")

    ck_selected_books: list[str] = []
    if _ck_available_books:
        ck_selected_books = st.multiselect(
            "📚 Выберите книги для чанкования",
            options=_ck_available_books,
            default=_ck_available_books,
            key="ck_selected_books",
        )

    # ── Инициализация состояния прогресса ───────────────────────────
    if "chunks_thread" not in st.session_state:
        st.session_state.chunks_thread = None
    if "chunks_cancel" not in st.session_state:
        st.session_state.chunks_cancel = None
    if "chunks_progress" not in st.session_state:
        st.session_state.chunks_progress = None
    if "chunks_result" not in st.session_state:
        st.session_state.chunks_result = None

    @st.fragment(run_every="2s")
    def _render_chunks_progress():
        """Автообновляемый фрагмент: прогресс чанкования + кнопка отмены."""
        _thread = st.session_state.chunks_thread
        _is_running = _thread is not None and _thread.is_alive()

        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.chunks_progress
        if _progress:
            _current = _progress.get("current", 0)
            _total = _progress.get("total", 1)
            _pct = _current / _total if _total > 0 else 0
            st.progress(_pct, text=f"Обработано {_current}/{_total} книг")

            with st.status("📋 Ход выполнения", expanded=True):
                st.write(f"📁 Текущая книга: `{_progress.get('current_file', '...')}`")
                st.write(
                    f"✅ Успешно: **{_progress.get('success', 0)}**  |  "
                    f"❌ Ошибок: **{_progress.get('errors', 0)}**  |  "
                    f"⏩ Пропущено: **{_progress.get('skipped', 0)}**"
                )
                st.write(f"📊 Чанков создано: **{_progress.get('total_chunks', 0)}**")

        if st.button("❌ Отменить чанкование", key="chunks_cancel_btn",
                      use_container_width=True, type="secondary"):
            if st.session_state.chunks_cancel:
                st.session_state.chunks_cancel.set()
            st.rerun()

    with tab_chunks:
        _ck_thread = st.session_state.chunks_thread
        _ck_is_running = _ck_thread is not None and _ck_thread.is_alive()

        if _ck_is_running:
            _render_chunks_progress()

        elif st.session_state.chunks_result:
            _res = st.session_state.chunks_result
            _total_chunks = _res.get("total_chunks", 0)
            _mode_label = _res.get("chunk_type", "")

            if _res.get("cancelled"):
                st.warning(
                    f"⚠️ Чанкование отменено. "
                    f"Успешно: {_res.get('success', 0)}/{_res.get('total', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с"
                )
            elif _res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Успешно: {_res.get('success', 0)}/{_res.get('total', 0)}, "
                    f"Ошибок: {_res.get('errors', 0)}. "
                    f"Время: {_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Чанкование ({_mode_label}) завершено! "
                    f"Обработано: {_res.get('success', 0)} книг, "
                    f"Создано: **{_total_chunks}** чанков. "
                    f"Время: {_res.get('elapsed_sec', '?')}с"
                )

            # Статистика по сводкам (только для таблиц)
            _from_summary = _res.get("from_summary", 0)
            _from_full = _res.get("from_full", 0)
            if _from_summary or _from_full:
                _total_tables = _from_summary + _from_full
                st.info(
                    f"📊 **Таблицы через сводку (summary):** {_from_summary} из {_total_tables} "
                    f"({round(_from_summary / _total_tables * 100) if _total_tables else 0}%). "
                    f"Без сводки: {_from_full}."
                )

            if st.button("🗑️ Сбросить результат", key="chunks_clear_result"):
                st.session_state.chunks_result = None
                st.session_state.chunks_progress = None
                st.rerun()

        else:
            _ck_can_run = (
                ck_base_dir and _ck_base and _ck_base.exists()
                and ck_selected_books
            )
            if _ck_can_run:
                _btn_label = {
                    "text": f"📝 Чанковать текст в {len(ck_selected_books)} "
                            f"{'книге' if len(ck_selected_books) == 1 else 'книгах'}",
                    "table": f"📋 Чанковать таблицы в {len(ck_selected_books)} "
                             f"{'книге' if len(ck_selected_books) == 1 else 'книгах'}",
                    "figure": f"🖼️ Чанковать описания в {len(ck_selected_books)} "
                              f"{'книге' if len(ck_selected_books) == 1 else 'книгах'}",
                }[ck_mode]

                if st.button(_btn_label, key="chunks_run_btn",
                             use_container_width=True, type="primary"):
                    ck_cancel_event = threading.Event()
                    st.session_state.chunks_cancel = ck_cancel_event
                    st.session_state.chunks_progress = {
                        "current": 0, "total": 0,
                        "current_file": "подготовка...",
                        "success": 0, "errors": 0, "skipped": 0,
                        "total_chunks": 0,
                    }
                    st.session_state.chunks_result = None

                    def _ck_progress_cb(current, total, current_file, stats):
                        st.session_state.chunks_progress = {
                            "current": current, "total": total,
                            "current_file": current_file,
                            "success": stats["success"],
                            "errors": stats["errors"],
                            "skipped": stats.get("skipped", 0),
                            "total_chunks": stats.get("total_chunks", 0),
                        }

                    _ck_sel = list(ck_selected_books)

                    def _ck_run():
                        if ck_mode == "text":
                            result = run_text_chunking(
                                base_dir=ck_base_dir,
                                book_names=_ck_sel,
                                chunk_size=ck_chunk_size,
                                overlap=ck_overlap,
                                min_chunk_size=ck_min_chunk_size,
                                cancel_event=ck_cancel_event,
                                progress_callback=_ck_progress_cb,
                            )
                        elif ck_mode == "table":
                            result = run_table_chunking(
                                base_dir=ck_base_dir,
                                book_names=_ck_sel,
                                chunk_size=ck_chunk_size,
                                cancel_event=ck_cancel_event,
                                progress_callback=_ck_progress_cb,
                            )
                        elif ck_mode == "figure":
                            result = run_figure_chunking(
                                base_dir=ck_base_dir,
                                book_names=_ck_sel,
                                chunk_size=ck_chunk_size,
                                cancel_event=ck_cancel_event,
                                progress_callback=_ck_progress_cb,
                            )
                        st.session_state.chunks_result = result

                    ck_thread = threading.Thread(target=_ck_run, daemon=True)
                    st.session_state.chunks_thread = ck_thread
                    ck_thread.start()
                    st.rerun()

# ══════════════════════════════════════════════════════════════════════
# Вкладка: Библиография
# ══════════════════════════════════════════════════════════════════════
with tab_biblio:
    st.subheader("📚 Библиография книг")
    st.markdown(
        "Ручной ввод автора, названия и года издания для каждой книги. "
        "Эти данные попадают в метаданные чанков и результатов поиска.\n\n"
        "При пустых полях используется авто-парсинг из имени папки "
        "(напр. `«Крохин, брикетирование углей»` → автор: Крохин, "
        "название: брикетирование углей)."
    )

    biblio_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("biblio_base_dir", "data/books"),
        key="biblio_base_dir",
    )

    _biblio_base = Path(biblio_base_dir) if biblio_base_dir else None

    if _biblio_base and _biblio_base.exists():
        # Загрузить метаданные
        _metadata = load_books_metadata()
        _folders = list_book_folders(_biblio_base)

        if _folders:
            st.info(f"Найдено **{len(_folders)}** папок с книгами.")

            # Загрузить существующие данные в session_state при первом запуске
            _edited = st.session_state.get("biblio_edited", False)

            rows = []
            for folder_name in _folders:
                _existing = _metadata.get(folder_name, {})
                _auto_author, _auto_title = _existing.get("author", ""), _existing.get("title", "")
                if not _auto_author and not _auto_title:
                    from src.chunking import parse_book_name
                    _auto_author, _auto_title = parse_book_name(folder_name)

                _year = _existing.get("year", "")

                cols = st.columns([3, 2, 3, 1, 1])
                with cols[0]:
                    st.markdown(f"📁 `{folder_name}`")
                with cols[1]:
                    _author_val = st.text_input(
                        "Автор", value=_auto_author,
                        key=f"biblio_author_{folder_name}", label_visibility="collapsed",
                        placeholder="Автор",
                    )
                with cols[2]:
                    _title_val = st.text_input(
                        "Название", value=_auto_title,
                        key=f"biblio_title_{folder_name}", label_visibility="collapsed",
                        placeholder="Название",
                    )
                with cols[3]:
                    _year_val = st.text_input(
                        "Год", value=_year,
                        key=f"biblio_year_{folder_name}", label_visibility="collapsed",
                        placeholder="Год",
                    )
                with cols[4]:
                    _saved = _existing.get("author") is not None
                    if _saved:
                        st.markdown("✅")

                rows.append({
                    "folder": folder_name,
                    "author": _author_val,
                    "title": _title_val,
                    "year": _year_val,
                })

            if st.button("💾 Сохранить библиографию", type="primary",
                         use_container_width=True, key="biblio_save_btn"):
                _new_meta = {}
                for row in rows:
                    _new_meta[row["folder"]] = {
                        "author": row["author"],
                        "title": row["title"],
                        "year": row["year"],
                    }
                save_books_metadata(_new_meta)
                st.success(f"✅ Сохранено метаданных для **{len(_new_meta)}** книг.")
                st.session_state.biblio_edited = True

        else:
            st.warning("Папка с книгами пуста.")
    elif _biblio_base:
        st.error(f"❌ Папка не найдена: `{biblio_base_dir}`")
