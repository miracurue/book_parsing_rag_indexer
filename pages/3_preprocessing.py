"""Страница препроцессинга — скрипты обработки PDF и изображений.

Содержит вкладки для каждого скрипта:
- Разделение PDF на страницы
- Конвертация PDF → JPEG
- Извлечение оглавления (VLM)
"""

import threading

import streamlit as st

from src.ui_helpers import render_source_selector, render_path_inputs, render_run_button
from src.run_pdf_split import run_pdf_split
from src.run_pdf_to_jpg import run_pdf_to_jpg, _find_split_pdf_files
from src.pdf_to_jpg import get_pdf_page_info
from src.sources import PDF_EXTS
from src.run_extract_toc import run_extract_toc, find_books_with_oglavlenie, count_toc_images
from src.llm_clients import list_model_names
from src.prompts import list_prompt_names, get_prompt_by_name

from pathlib import Path
from io import BytesIO


def _fmt_size(num_bytes: int) -> str:
    """Форматировать размер файла в читаемый вид."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"

st.set_page_config(page_title="Препроцессинг", page_icon="📂", layout="wide")
st.title("📂 Препроцессинг")

# ── Вкладки скриптов ────────────────────────────────────────────────
tab_pdf_split, tab_pdf_to_jpg, tab_extract_toc = st.tabs([
    "✂️ Разделение PDF",
    "🖼️ Конвертация PDF → JPG",
    "📖 Извлечение оглавления",
])

# ════════════════════════════════════════════════════════════════════
# Вкладка: Разделение PDF на страницы
# ════════════════════════════════════════════════════════════════════
with tab_pdf_split:
    st.subheader("✂️ Разделение PDF на страницы")
    st.markdown(
        "Разбивает PDF файлы на отдельные страницы. "
        "Для каждого файла создаётся папка с именем файла, "
        "внутрь — подпапка `split_pdf/`, а в ней — страницы: `000.pdf`, `001.pdf` и т.д."
    )

    # Выбор источника
    source_type, dest_type = render_source_selector(
        key_prefix="pdf_split",
        label="Откуда брать PDF",
    )

    # Поля путей
    source_dir, dest_dir = render_path_inputs(
        source_type=source_type,
        dest_type=dest_type,
        key_prefix="pdf_split",
        source_label="Папка с PDF файлами",
        dest_label="Папка для разбитых страниц",
        source_placeholder="Напр.: data/raw или Books/PDFs",
        dest_placeholder="Напр.: data/processed/pages или Books/Pages",
    )

    # ── Выбор PDF файлов ────────────────────────────────────────────
    _available_pdfs: list[str] = []
    if source_type == "local" and source_dir:
        _src_path = Path(source_dir)
        if _src_path.exists() and _src_path.is_dir():
            _available_pdfs = sorted(
                f.name for f in _src_path.iterdir()
                if f.is_file() and f.suffix.lower() in PDF_EXTS
            )

    selected_pdfs: list[str] = []
    if _available_pdfs:
        selected_pdfs = st.multiselect(
            "📄 Выберите PDF для обработки",
            options=_available_pdfs,
            default=_available_pdfs,
            key="pdf_split_selected_files",
            help="Оставьте все выбранными или снимите лишние.",
        )
    elif source_type == "local" and source_dir:
        if Path(source_dir).exists():
            st.warning("⚠️ В указанной папке нет PDF файлов.")
        else:
            st.error(f"❌ Папка не найдена: `{source_dir}`")

    # Параметры
    col1, col2 = st.columns([1, 3])
    with col1:
        start_page = st.number_input(
            "Начало нумерации страниц",
            min_value=0,
            max_value=9999,
            value=st.session_state.get("pdf_split_start_page", 0),
            key="pdf_split_start_page",
            help="Номер первой страницы (для именования файлов). "
                 "Например, 0 → файлы начнутся с 000.pdf",
        )

    # Валидация
    if not source_dir:
        st.warning("⚠️ Укажите папку с исходными PDF файлами.")
    elif not dest_dir:
        st.warning("⚠️ Укажите папку для результатов.")
    elif source_type == "local" and not selected_pdfs:
        st.warning("⚠️ Выберите хотя бы один PDF файл.")

    # Кнопка запуска
    _can_run = source_dir and dest_dir
    if source_type == "local":
        _can_run = _can_run and bool(selected_pdfs)

    if _can_run:
        render_run_button(
            label=f"✂️ Разделить {len(selected_pdfs)} PDF на страницы",
            run_fn=run_pdf_split,
            args={
                "source_type": source_type,
                "dest_type": dest_type,
                "local_source_dir": source_dir if source_type == "local" else "",
                "local_dest_dir": dest_dir if dest_type == "local" else "",
                "yandex_source_dir": source_dir if source_type == "yandex" else "",
                "yandex_dest_dir": dest_dir if dest_type == "yandex" else "",
                "start_page": start_page,
                "selected_files": selected_pdfs if source_type == "local" else None,
            },
            key_prefix="pdf_split",
        )


# ════════════════════════════════════════════════════════════════════
# Вкладка: Конвертация PDF → JPG
# ════════════════════════════════════════════════════════════════════
with tab_pdf_to_jpg:
    st.subheader("🖼️ Конвертация PDF → JPG")
    st.markdown(
        "Конвертирует одностраничные PDF из `название_книги/split_pdf/` "
        "в JPEG и сохраняет в `название_книги/convert_in_jpg/`. "
        "Папка `convert_in_jpg` создаётся автоматически."
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("pdf_to_jpg_base_dir", "data/books"),
        key="pdf_to_jpg_base_dir",
        help="Структура: папка/название_книги/split_pdf/*.pdf → "
             "папка/название_книги/convert_in_jpg/*.jpg",
    )

    # ── Настройки ───────────────────────────────────────────────────
    with st.expander("⚙️ Настройки", expanded=False):
        col_dpi, col_poppler = st.columns(2)
        with col_dpi:
            dpi = st.slider(
                "DPI (разрешение)",
                min_value=72,
                max_value=600,
                value=st.session_state.get("pdf_to_jpg_dpi", 200),
                step=10,
                key="pdf_to_jpg_dpi",
                help="Чем выше DPI, тем больше размер файла и лучше качество. "
                     "Рекомендуется 200 для чтения, 300–600 для OCR.",
            )
        with col_poppler:
            poppler_path = st.text_input(
                "Путь к Poppler (необязательно)",
                value=st.session_state.get("pdf_to_jpg_poppler", ""),
                key="pdf_to_jpg_poppler",
                placeholder="Напр.: C:/poppler/Library/bin",
                help="Оставьте пустым, если Poppler уже в PATH.",
            )

    # ── Информация о странице ───────────────────────────────────────
    st.divider()
    st.markdown("**📄 Просмотр информации о странице**")

    col_page_num, col_btn_info = st.columns([1, 1])
    with col_page_num:
        page_number = st.number_input(
            "Номер страницы (имя файла)",
            min_value=0,
            max_value=9999,
            value=st.session_state.get("pdf_to_jpg_page_number", 0),
            key="pdf_to_jpg_page_number",
            help="Введите номер страницы (например, 0 → файл 000.pdf). "
                 "Информация берётся из первого найденного файла с таким именем.",
        )

    # Ищем файл с указанным номером
    _page_filename = f"{page_number:03d}.pdf"
    _found_page = None
    if base_dir:
        _base = Path(base_dir)
        if _base.exists():
            for _book_dir in sorted(_base.iterdir()):
                if not _book_dir.is_dir():
                    continue
                _candidate = _book_dir / "split_pdf" / _page_filename
                if _candidate.exists():
                    _found_page = _candidate
                    break

    if _found_page:
        st.caption(f"Файл найден: `{_found_page}`")
        if st.button(
            "🔍 Показать информацию о странице",
            key="pdf_to_jpg_show_info_btn",
        ):
            with st.spinner("Получение информации о странице..."):
                try:
                    _poppler = poppler_path if poppler_path else None
                    pdf_buf = BytesIO(_found_page.read_bytes())
                    info = get_pdf_page_info(
                        pdf_buf, dpi=dpi, poppler_path=_poppler,
                    )
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Размер (px)", f"{info['width_px']} × {info['height_px']}")
                    with col_b:
                        pdf_sz = _fmt_size(info['pdf_size_bytes'])
                        jpg_sz = _fmt_size(info['jpg_size_bytes'])
                        st.metric("Размер файла", f"PDF {pdf_sz} → JPG {jpg_sz}")
                except Exception as e:
                    st.error(f"❌ Ошибка: {e}")
    else:
        if base_dir:
            st.info(
                f"Файл `{_page_filename}` не найден ни в одной "
                f"папке `*/split_pdf/` внутри `{base_dir}`."
            )

    # ── Выбор книг ──────────────────────────────────────────────────
    st.divider()

    _available_books: list[str] = []
    if base_dir:
        _base = Path(base_dir)
        if _base.exists() and _base.is_dir():
            for _book_dir in sorted(_base.iterdir()):
                if _book_dir.is_dir() and (_book_dir / "split_pdf").is_dir():
                    _available_books.append(_book_dir.name)

    selected_books: list[str] = []
    if _available_books:
        selected_books = st.multiselect(
            "📚 Выберите книги для конвертации",
            options=_available_books,
            default=_available_books,
            key="pdf_to_jpg_selected_books",
            help="Оставьте все выбранными или снимите лишние.",
        )
    elif base_dir and Path(base_dir).exists():
        st.warning(
            f"В `{base_dir}` не найдено папок с `split_pdf/`. "
            f"Сначала выполните разделение PDF."
        )

    # Предпросмотр: сколько файлов будет обработано
    if selected_books and base_dir:
        _base = Path(base_dir)
        if _base.exists():
            _preview_files = _find_split_pdf_files(_base, book_names=selected_books)
            if _preview_files:
                st.info(
                    f"Будет обработано: **{len(_preview_files)}** PDF файлов "
                    f"из **{len(selected_books)}** "
                    f"{'книги' if len(selected_books) == 1 else 'книг'} "
                    f"({', '.join(selected_books[:5])}"
                    f"{', ...' if len(selected_books) > 5 else ''})"
                )
            else:
                st.warning("Выбранные книги не содержат PDF файлов в `split_pdf/`.")
    elif _available_books and not selected_books:
        st.warning("⚠️ Выберите хотя бы одну книгу для конвертации.")

    # Инициализация состояния прогресса
    if "pdf_to_jpg_thread" not in st.session_state:
        st.session_state.pdf_to_jpg_thread = None
    if "pdf_to_jpg_cancel" not in st.session_state:
        st.session_state.pdf_to_jpg_cancel = None
    if "pdf_to_jpg_progress" not in st.session_state:
        st.session_state.pdf_to_jpg_progress = None
    if "pdf_to_jpg_result" not in st.session_state:
        st.session_state.pdf_to_jpg_result = None


@st.fragment(run_every="2s")
def _render_pdf_to_jpg_progress():
    """Автообновляемый фрагмент: прогресс конвертации PDF→JPG + кнопка отмены.

    Обновляется каждые 2 сек, пока поток жив.
    Когда поток завершается — вызывает st.rerun() для полной перерисовки.
    """
    _thread = st.session_state.pdf_to_jpg_thread
    _is_running = _thread is not None and _thread.is_alive()

    if not _is_running:
        st.rerun()
        return

    # ── Прогресс ────────────────────────────────────────────────────
    _progress = st.session_state.pdf_to_jpg_progress
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

    # ── Кнопка отмены ───────────────────────────────────────────────
    if st.button(
        "❌ Отменить конвертацию",
        key="pdf_to_jpg_cancel_btn",
        use_container_width=True,
        type="secondary",
    ):
        if st.session_state.pdf_to_jpg_cancel:
            st.session_state.pdf_to_jpg_cancel.set()
        st.rerun()


with tab_pdf_to_jpg:
    # (продолжение — рендер результата и кнопки запуска)
    _thread = st.session_state.pdf_to_jpg_thread
    _is_running = _thread is not None and _thread.is_alive()

    if _is_running:
        # Фрагмент с автообновлением
        _render_pdf_to_jpg_progress()

    elif st.session_state.pdf_to_jpg_result:
        # ── Результат ───────────────────────────────────────────────
        _res = st.session_state.pdf_to_jpg_result
        if _res.get("cancelled"):
            st.warning(
                f"⚠️ Конвертация отменена. "
                f"Успешно: {_res.get('success', 0)}/{_res.get('total', 0)}, "
                f"Ошибок: {_res.get('errors', 0)}. "
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
                f"✅ Конвертация завершена! "
                f"Обработано: {_res.get('success', 0)} файлов. "
                f"Время: {_res.get('elapsed_sec', '?')}с"
            )
        if st.button("🗑️ Сбросить результат", key="pdf_to_jpg_clear_result"):
            st.session_state.pdf_to_jpg_result = None
            st.session_state.pdf_to_jpg_progress = None
            st.rerun()

    else:
        # ── Кнопка запуска ──────────────────────────────────────────
        if base_dir and Path(base_dir).exists() and selected_books:
            if st.button(
                f"🖼️ Конвертировать {len(selected_books)} "
                f"{'книгу' if len(selected_books) == 1 else 'книги' if len(selected_books) < 5 else 'книг'} в JPG",
                key="pdf_to_jpg_run_btn",
                use_container_width=True,
                type="primary",
            ):
                # Создаём событие отмены
                cancel_event = threading.Event()
                st.session_state.pdf_to_jpg_cancel = cancel_event
                st.session_state.pdf_to_jpg_progress = {
                    "current": 0,
                    "total": 0,
                    "current_file": "подготовка...",
                    "success": 0,
                    "errors": 0,
                }
                st.session_state.pdf_to_jpg_result = None

                # Замыкаем текущие значения
                _selected = list(selected_books)
                _base = base_dir
                _dpi = dpi
                _poppler = poppler_path if poppler_path else None

                def _progress_cb(current, total, current_file, stats):
                    st.session_state.pdf_to_jpg_progress = {
                        "current": current,
                        "total": total,
                        "current_file": current_file,
                        "success": stats["success"],
                        "errors": stats["errors"],
                    }

                def _run():
                    result = run_pdf_to_jpg(
                        base_dir=_base,
                        book_names=_selected,
                        dpi=_dpi,
                        poppler_path=_poppler,
                        cancel_event=cancel_event,
                        progress_callback=_progress_cb,
                    )
                    st.session_state.pdf_to_jpg_result = result

                thread = threading.Thread(target=_run, daemon=True)
                st.session_state.pdf_to_jpg_thread = thread
                thread.start()
                st.rerun()
        elif base_dir and not Path(base_dir).exists():
            st.error(f"❌ Папка не найдена: `{base_dir}`")


# ════════════════════════════════════════════════════════════════════
# Вкладка: Извлечение оглавления
# ════════════════════════════════════════════════════════════════════
with tab_extract_toc:
    st.subheader("📖 Извлечение оглавления")
    st.markdown(
        "Извлекает оглавление из JPG-сканов страниц в папке "
        "`название_книги/convert_in_jpg/toc/` с помощью VLM. "
        "Результат сохраняется в `название_книги/parsed/toc/`."
    )

    # ── Путь к базовой папке ────────────────────────────────────────
    toc_base_dir = st.text_input(
        "📂 Папка с книгами",
        value=st.session_state.get("toc_base_dir", "data/books"),
        key="toc_base_dir",
        help="Структура: папка/название_книги/convert_in_jpg/toc/*.jpg",
    )

    # ── Настройки модели и промпта ──────────────────────────────────
    col_model, col_prompt = st.columns(2)

    _model_names = list_model_names()
    _default_model_idx = _model_names.index("glm-4.6v") if "glm-4.6v" in _model_names else 0

    with col_model:
        toc_model = st.selectbox(
            "🤖 Модель VLM",
            options=_model_names,
            index=_default_model_idx,
            key="toc_model",
            help="Модель для распознавания оглавления.",
        )

    _prompt_names = list_prompt_names()
    _default_prompt_idx = (
        _prompt_names.index("EXTRACT_CHUPTERS")
        if "EXTRACT_CHUPTERS" in _prompt_names
        else 0
    )

    with col_prompt:
        toc_prompt_name = st.selectbox(
            "📝 Промпт",
            options=_prompt_names,
            index=_default_prompt_idx,
            key="toc_prompt_name",
            help="Промпт из реестра. По умолчанию — EXTRACT_CHUPTERS.",
        )

    # Показать текст промпта
    if toc_prompt_name:
        _prompt_data = get_prompt_by_name(toc_prompt_name)
        if _prompt_data:
            with st.expander("📄 Текст промпта", expanded=False):
                st.text(_prompt_data["text"])

    # Температура
    toc_temperature = st.slider(
        "🌡️ Температура",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.get("toc_temperature", 0.1),
        step=0.05,
        key="toc_temperature",
    )

    st.divider()

    # ── Выбор книг ──────────────────────────────────────────────────
    _toc_available_books: list[str] = []
    if toc_base_dir:
        _toc_base = Path(toc_base_dir)
        if _toc_base.exists() and _toc_base.is_dir():
            _toc_available_books = find_books_with_oglavlenie(_toc_base)

    toc_selected_books: list[str] = []
    if _toc_available_books:
        toc_selected_books = st.multiselect(
            "📚 Выберите книги для извлечения оглавления",
            options=_toc_available_books,
            default=_toc_available_books,
            key="toc_selected_books",
            help="Выберите книги, у которых есть папка convert_in_jpg/toc/.",
        )
    elif toc_base_dir and Path(toc_base_dir).exists():
        st.info(
            "В указанной папке нет книг с подпапкой `convert_in_jpg/toc/`. "
            "Сначала поместите сканы оглавления в эту папку."
        )

    # Предпросмотр
    if toc_selected_books and toc_base_dir:
        _toc_base = Path(toc_base_dir)
        if _toc_base.exists():
            _toc_count = count_toc_images(_toc_base, toc_selected_books)
            if _toc_count > 0:
                st.info(
                    f"Будет обработано: **{_toc_count}** "
                    f"{'страниц' if _toc_count > 1 else 'страница'} оглавления "
                    f"из **{len(toc_selected_books)}** "
                    f"{'книг' if len(toc_selected_books) > 1 else 'книги'}"
                )
    elif _toc_available_books and not toc_selected_books:
        st.warning("⚠️ Выберите хотя бы одну книгу.")

    # Инициализация состояния прогресса
    if "toc_thread" not in st.session_state:
        st.session_state.toc_thread = None
    if "toc_cancel" not in st.session_state:
        st.session_state.toc_cancel = None
    if "toc_progress" not in st.session_state:
        st.session_state.toc_progress = None
    if "toc_result" not in st.session_state:
        st.session_state.toc_result = None


    @st.fragment(run_every="2s")
    def _render_toc_progress():
        """Автообновляемый фрагмент: прогресс извлечения оглавления + отмена."""
        _thread = st.session_state.toc_thread
        _is_running = _thread is not None and _thread.is_alive()

        if not _is_running:
            st.rerun()
            return

        _progress = st.session_state.toc_progress
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

        if st.button(
            "❌ Отменить извлечение",
            key="toc_cancel_btn",
            use_container_width=True,
            type="secondary",
        ):
            if st.session_state.toc_cancel:
                st.session_state.toc_cancel.set()
            st.rerun()


    with tab_extract_toc:
        _toc_thread = st.session_state.toc_thread
        _toc_is_running = _toc_thread is not None and _toc_thread.is_alive()

        if _toc_is_running:
            _render_toc_progress()

        elif st.session_state.toc_result:
            _toc_res = st.session_state.toc_result
            if _toc_res.get("cancelled"):
                st.warning(
                    f"⚠️ Извлечение отменено. "
                    f"Успешно: {_toc_res.get('success', 0)}/{_toc_res.get('total', 0)}, "
                    f"Ошибок: {_toc_res.get('errors', 0)}. "
                    f"Книг: {_toc_res.get('books_processed', 0)}. "
                    f"Время: {_toc_res.get('elapsed_sec', '?')}с"
                )
            elif _toc_res.get("errors", 0) > 0:
                st.warning(
                    f"⚠️ Завершено с ошибками. "
                    f"Успешно: {_toc_res.get('success', 0)}/{_toc_res.get('total', 0)}, "
                    f"Ошибок: {_toc_res.get('errors', 0)}. "
                    f"Книг: {_toc_res.get('books_processed', 0)}. "
                    f"Время: {_toc_res.get('elapsed_sec', '?')}с"
                )
            else:
                st.success(
                    f"✅ Извлечение завершено! "
                    f"Обработано: {_toc_res.get('success', 0)} страниц, "
                    f"книг: {_toc_res.get('books_processed', 0)}. "
                    f"Время: {_toc_res.get('elapsed_sec', '?')}с"
                )
            if st.button("🗑️ Сбросить результат", key="toc_clear_result"):
                st.session_state.toc_result = None
                st.session_state.toc_progress = None
                st.rerun()

        else:
            # Кнопка запуска
            _can_run_toc = (
                toc_base_dir
                and Path(toc_base_dir).exists()
                and toc_selected_books
                and toc_model
                and toc_prompt_name
            )

            if _can_run_toc:
                if st.button(
                    f"📖 Извлечь оглавление из {len(toc_selected_books)} "
                    f"{'книги' if len(toc_selected_books) == 1 else 'книг'}",
                    key="toc_run_btn",
                    use_container_width=True,
                    type="primary",
                ):
                    # Получаем текст промпта
                    _prompt_data = get_prompt_by_name(toc_prompt_name)
                    _prompt_text = _prompt_data["text"] if _prompt_data else ""

                    cancel_event = threading.Event()
                    st.session_state.toc_cancel = cancel_event
                    st.session_state.toc_progress = {
                        "current": 0,
                        "total": 0,
                        "current_file": "подготовка...",
                        "success": 0,
                        "errors": 0,
                    }
                    st.session_state.toc_result = None

                    _toc_selected = list(toc_selected_books)
                    _toc_base = toc_base_dir
                    _toc_model = toc_model
                    _toc_prompt = _prompt_text
                    _toc_temp = toc_temperature

                    def _toc_progress_cb(current, total, current_file, stats):
                        st.session_state.toc_progress = {
                            "current": current,
                            "total": total,
                            "current_file": current_file,
                            "success": stats["success"],
                            "errors": stats["errors"],
                        }

                    def _toc_run():
                        result = run_extract_toc(
                            base_dir=_toc_base,
                            book_names=_toc_selected,
                            model_name=_toc_model,
                            prompt_text=_toc_prompt,
                            temperature=_toc_temp,
                            cancel_event=cancel_event,
                            progress_callback=_toc_progress_cb,
                        )
                        st.session_state.toc_result = result

                    thread = threading.Thread(target=_toc_run, daemon=True)
                    st.session_state.toc_thread = thread
                    thread.start()
                    st.rerun()

            elif toc_base_dir and not Path(toc_base_dir).exists():
                st.error(f"❌ Папка не найдена: `{toc_base_dir}`")
