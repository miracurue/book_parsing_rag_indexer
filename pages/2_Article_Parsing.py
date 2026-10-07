"""Страница парсинга PDF-статей через Docling.

Парсит PDF файлы из data/articles/raw → Markdown в data/articles/parsed.
Извлекает текст, заголовки, таблицы, формулы, изображения (опционально).
"""

import threading

import streamlit as st
from pathlib import Path

from src.docling_parser import get_pdf_files, get_parsed_articles, create_converter
from src.run_docling_parser import run_article_parsing

st.set_page_config(page_title="Статьи", page_icon="📰", layout="wide")
st.title("📰 Парсинг статей (Docling)")
st.markdown(
    "Конвертация PDF-статей в Markdown с помощью **Docling**. "
    "Извлекает текст, заголовки, таблицы, формулы. "
    "Изображения — опционально (заглушки `[IMAGE_N]` по умолчанию)."
)

# ════════════════════════════════════════════════════════════════════
# Настройки путей
# ════════════════════════════════════════════════════════════════════
col_raw, col_out = st.columns(2)

with col_raw:
    raw_dir = st.text_input(
        "📂 Папка с PDF",
        value=st.session_state.get("articles_raw_dir", "data/articles/raw"),
        key="articles_raw_dir",
        help="Папка с исходными PDF-файлами статей.",
    )

with col_out:
    output_dir = st.text_input(
        "📂 Папка для результатов",
        value=st.session_state.get("articles_output_dir", "data/articles/parsed"),
        key="articles_output_dir",
        help="Каждая статья → подпапка с .md файлом.",
    )

# ════════════════════════════════════════════════════════════════════
# Настройки парсинга
# ════════════════════════════════════════════════════════════════════
with st.expander("⚙️ Настройки", expanded=False):
    extract_images = st.checkbox(
        "🖼️ Извлекать изображения",
        value=st.session_state.get("articles_extract_images", False),
        key="articles_extract_images",
        help="Если выключено — в тексте будут заглушки [IMAGE_N]. "
             "Если включено — изображения вырезаются и сохраняются в images/.",
    )

    if extract_images:
        image_scale = st.slider(
            "Масштаб изображений",
            min_value=1.0,
            max_value=4.0,
            value=st.session_state.get("articles_image_scale", 2.0),
            step=0.5,
            key="articles_image_scale",
            help="DPI = 72 × scale. 2.0 = 144 DPI (по умолчанию), 4.0 = 288 DPI (высокое качество).",
        )
    else:
        image_scale = 2.0

    st.divider()

    extract_formulas = st.checkbox(
        "🔢 Извлекать формулы (VLM)",
        value=st.session_state.get("articles_extract_formulas", False),
        key="articles_extract_formulas",
        help="Если включено — Docling распознаёт формулы через VLM-модель "
             "(CodeFormulaV2) и выводит их в LaTeX-формате ($...$, $$...$$). "
             "⚠️ Замедляет парсинг в 2–5 раз. Модель ~1–2 ГБ скачивается при первом запуске.",
    )

    st.divider()

    recreate = st.checkbox(
        "🔄 Перезаписать все статьи",
        value=st.session_state.get("articles_recreate", False),
        key="articles_recreate",
        help="Если включено — уже распарсенные статьи будут перезаписаны. "
             "Иначе — пропускаются.",
    )

st.divider()

# ════════════════════════════════════════════════════════════════════
# Выбор файлов
# ════════════════════════════════════════════════════════════════════
_available_pdfs: list[str] = []
_raw_path = Path(raw_dir) if raw_dir else None

if _raw_path and _raw_path.exists():
    _pdf_paths = get_pdf_files(_raw_path)
    _available_pdfs = [p.name for p in _pdf_paths]

# Уже обработанные
_already_parsed: list[str] = []
_out_path = Path(output_dir) if output_dir else None
if _out_path and _out_path.exists():
    _already_parsed = get_parsed_articles(_out_path)

if _available_pdfs:
    selected_files = st.multiselect(
        "📄 Выберите PDF для парсинга",
        options=_available_pdfs,
        default=_available_pdfs,
        key="articles_selected_files",
        help="Оставьте все выбранными или снимите лишние.",
    )
elif _raw_path and _raw_path.exists():
    st.warning("⚠️ В указанной папке нет PDF файлов.")
elif _raw_path:
    st.error(f"❌ Папка не найдена: `{raw_dir}`")

# Предпросмотр
if _available_pdfs and selected_files:
    _new_files = [f for f in selected_files if Path(f).stem not in _already_parsed]
    _skipped = len(selected_files) - len(_new_files)

    info_parts = [f"**{len(selected_files)}** PDF выбрано"]
    if _skipped > 0:
        info_parts.append(f"({len(_new_files)} новых, {_skipped} уже обработано)")
    if _already_parsed:
        info_parts.append(f"Всего обработано ранее: **{len(_already_parsed)}**")

    st.info(" ".join(info_parts))

# ════════════════════════════════════════════════════════════════════
# Инициализация состояния прогресса
# ════════════════════════════════════════════════════════════════════
if "articles_thread" not in st.session_state:
    st.session_state.articles_thread = None
if "articles_cancel" not in st.session_state:
    st.session_state.articles_cancel = None
if "articles_progress" not in st.session_state:
    st.session_state.articles_progress = None
if "articles_result" not in st.session_state:
    st.session_state.articles_result = None


# ════════════════════════════════════════════════════════════════════
# Фрагмент прогресса (автообновление)
# ════════════════════════════════════════════════════════════════════
@st.fragment(run_every="2s")
def _render_articles_progress():
    """Автообновляемый фрагмент: прогресс парсинга + кнопка отмены."""
    _thread = st.session_state.articles_thread
    _is_running = _thread is not None and _thread.is_alive()

    if not _is_running:
        st.rerun()
        return

    _progress = st.session_state.articles_progress
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
        "❌ Отменить парсинг",
        key="articles_cancel_btn",
        use_container_width=True,
        type="secondary",
    ):
        if st.session_state.articles_cancel:
            st.session_state.articles_cancel.set()
        st.rerun()


# ════════════════════════════════════════════════════════════════════
# Рендер состояния (результат / прогресс / кнопка запуска)
# ════════════════════════════════════════════════════════════════════
_thread = st.session_state.articles_thread
_is_running = _thread is not None and _thread.is_alive()

if _is_running:
    _render_articles_progress()

elif st.session_state.articles_result:
    # ── Результат ───────────────────────────────────────────────────
    _res = st.session_state.articles_result
    if _res.get("cancelled"):
        st.warning(
            f"⚠️ Парсинг отменён. "
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
            f"✅ Парсинг завершён! "
            f"Обработано: {_res.get('success', 0)} статей. "
            f"Время: {_res.get('elapsed_sec', '?')}с"
        )

    # Детализация по файлам
    _results = _res.get("results", [])
    if _results:
        with st.expander("📊 Детализация", expanded=True):
            for r in _results:
                _stem = r.get("stem", "?")
                _status = r.get("status", "?")
                if _status == "ok":
                    _placeholders = r.get("image_placeholders", 0)
                    _imgs = r.get("images_count", 0)
                    _formulas = r.get("formula_count", 0)
                    _formulas_nd = r.get("formula_not_decoded", 0)
                    _split = r.get("split_used", False)
                    _extra = ""
                    if _split:
                        _extra += " ⚡(разделён на части)"
                    if _formulas:
                        _extra += f", {_formulas} формул"
                    if _formulas_nd:
                        _extra += f", {_formulas_nd} нераспозн. формул"
                    if _placeholders:
                        _extra += f", {_placeholders} изобр."
                    if _imgs:
                        _extra += f", сохранено {_imgs} картинок"
                    st.write(f"✅ `{_stem}`{_extra}")
                elif _status == "skipped":
                    st.write(f"⏭️ `{_stem}` (уже обработан)")
                else:
                    _err = r.get("error", "неизвестная ошибка")
                    st.write(f"❌ `{_stem}`: {_err}")

    if st.button("🗑️ Сбросить результат", key="articles_clear_result"):
        st.session_state.articles_result = None
        st.session_state.articles_progress = None
        st.rerun()

else:
    # ── Кнопка запуска ──────────────────────────────────────────────
    _can_run = (
        raw_dir
        and output_dir
        and Path(raw_dir).exists()
        and _available_pdfs
        and selected_files
    )

    if _can_run:
        if st.button(
            f"📰 Парсить {len(selected_files)} "
            f"{'статью' if len(selected_files) == 1 else 'статьи' if len(selected_files) < 5 else 'статей'}",
            key="articles_run_btn",
            use_container_width=True,
            type="primary",
        ):
            cancel_event = threading.Event()
            st.session_state.articles_cancel = cancel_event
            st.session_state.articles_progress = {
                "current": 0,
                "total": 0,
                "current_file": "подготовка...",
                "success": 0,
                "errors": 0,
            }
            st.session_state.articles_result = None

            _selected = list(selected_files)
            _raw = raw_dir
            _out = output_dir
            _extract_img = extract_images
            _img_scale = image_scale
            _extract_formulas = extract_formulas
            _recreate = recreate

            def _progress_cb(current, total, current_file, stats):
                st.session_state.articles_progress = {
                    "current": current,
                    "total": total,
                    "current_file": current_file,
                    "success": stats["success"],
                    "errors": stats["errors"],
                }

            def _run():
                try:
                    result = run_article_parsing(
                        raw_dir=_raw,
                        output_dir=_out,
                        selected_files=_selected,
                        extract_images=_extract_img,
                        image_scale=_img_scale,
                        extract_formulas=_extract_formulas,
                        recreate=_recreate,
                        remote_url=None,
                        cancel_event=cancel_event,
                        progress_callback=_progress_cb,
                    )
                    st.session_state.articles_result = result
                except Exception as _thread_err:
                    import logging as _log
                    _log.getLogger(__name__).exception(
                        "Крах потока парсинга: %s", _thread_err
                    )
                    st.session_state.articles_result = {
                        "total": len(_selected),
                        "success": 0,
                        "errors": 1,
                        "cancelled": False,
                        "elapsed_sec": 0,
                        "results": [{
                            "stem": "thread_crash",
                            "status": "error",
                            "error": f"Крах потока: {_thread_err}",
                        }],
                    }

            thread = threading.Thread(target=_run, daemon=True)
            st.session_state.articles_thread = thread
            thread.start()
            st.rerun()

    elif _available_pdfs and not selected_files:
        st.warning("⚠️ Выберите хотя бы один PDF файл.")
    elif raw_dir and not Path(raw_dir).exists():
        st.error(f"❌ Папка не найдена: `{raw_dir}`")

# ════════════════════════════════════════════════════════════════════
# Предпросмотр уже обработанных статей
# ════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📁 Обработанные статьи")

if _out_path and _out_path.exists() and _already_parsed:
    _selected_preview = st.selectbox(
        "Выберите статью для просмотра",
        options=_already_parsed,
        key="articles_preview_select",
    )

    if _selected_preview:
        _preview_dir = _out_path / _selected_preview
        _md_files = list(_preview_dir.glob("*.md"))

        if _md_files:
            _md_path = _md_files[0]
            _content = _md_path.read_text(encoding="utf-8")

            col_info, col_text = st.columns([1, 3])

            with col_info:
                st.metric("Файл", _md_path.name)
                st.metric("Размер", f"{len(_content):,} символов")

                _img_count = _content.count("[IMAGE_")
                if _img_count:
                    st.metric("Изображений", _img_count)

                # Подсчёт формул в тексте
                _formula_nd = _content.count("<!-- formula-not-decoded -->")
                if _formula_nd:
                    st.metric("Нераспозн. формулы", _formula_nd)

                # Ссылка на папку
                st.caption(f"📁 `{_preview_dir}`")

            with col_text:
                st.text_area(
                    "Предпросмотр Markdown",
                    value=_content[:5000] + ("\n\n... (обрезано)" if len(_content) > 5000 else ""),
                    height=400,
                    key="articles_preview_text",
                    disabled=True,
                )

        # Картинки (если есть)
        _images_dir = _preview_dir / "images"
        if _images_dir.exists():
            _img_files = sorted(_images_dir.glob("*.png"))
            if _img_files:
                with st.expander(f"🖼️ Изображения ({len(_img_files)})", expanded=False):
                    _img_cols = st.columns(min(4, len(_img_files)))
                    for _idx, _img_f in enumerate(_img_files[:8]):
                        with _img_cols[_idx % len(_img_cols)]:
                            st.image(str(_img_f), caption=_img_f.name)
                    if len(_img_files) > 8:
                        st.caption(f"... и ещё {len(_img_files) - 8} изображений")

else:
    st.info("Обработанных статей пока нет. Запустите парсинг выше.")

# ════════════════════════════════════════════════════════════════════
# 📥 Импорт результатов из Colab (ZIP)
# ════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📥 Импорт из Colab")

st.markdown(
    "Если вы распарсили PDF в Google Colab через ноутбук "
    "`server/colab_docling_standalone.ipynb`, загрузите ZIP-архив сюда — "
    "файлы распакуются в папку с результатами."
)

_import_dir = st.text_input(
    "📂 Папка для распаковки",
    value=st.session_state.get("articles_import_dir", output_dir or "data/articles/parsed"),
    key="articles_import_dir",
)

_uploaded_zip = st.file_uploader(
    "📦 Загрузите ZIP-архив из Colab",
    type=["zip"],
    key="articles_zip_upload",
    help="Файл вида `docling_parsed_YYYYMMDD_HHMMSS.zip` из Colab.",
)

if _uploaded_zip is not None:
    import zipfile as _zf
    import io as _io

    _target = Path(_import_dir)
    _target.mkdir(parents=True, exist_ok=True)

    try:
        with _zf.ZipFile(_io.BytesIO(_uploaded_zip.read()), "r") as _zip_ref:
            _names = _zip_ref.namelist()

            # Фильтруем: только parsed/внутри
            _md_files = [n for n in _names if n.endswith(".md") and not n.endswith("_report.txt")]
            _report = [n for n in _names if "_report" in n.lower()]

            if not _md_files:
                st.warning("⚠️ В архиве нет .md файлов.")
            else:
                _zip_ref.extractall(_target.parent)
                st.success(
                    f"✅ Распаковано {len(_md_files)} .md файлов в `{_target}`"
                )

                # Показываем отчёт
                for _rpt in _report:
                    _rpt_content = _zip_ref.read(_rpt).decode("utf-8", errors="replace")
                    with st.expander("📋 Отчёт парсинга", expanded=True):
                        st.text(_rpt_content[:3000])

    except _zf.BadZipFile:
        st.error("❌ Файл не является корректным ZIP-архивом.")
    except Exception as _import_err:
        st.error(f"❌ Ошибка распаковки: {_import_err}")
