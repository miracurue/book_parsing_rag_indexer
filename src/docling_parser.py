"""Парсинг PDF-статей через Docling.

Чистые функции для конвертации PDF → Markdown.
Извлекает текст, заголовки, таблицы, формулы.
Изображения — опционально (вырезка + сохранение).

При нехватке памяти (std::bad_alloc) — автоматический retry
с разделением PDF на 2 части и конвертацией по частям.

Поддерживается удалённый парсинг через API (Colab / GPU-сервер):
    result = parse_pdf_to_markdown("article.pdf", output_dir, remote_url="https://...")

Использование (локально):
    result = parse_pdf_to_markdown("article.pdf", output_dir, extract_images=True)
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# Создание конвертера Docling
# ════════════════════════════════════════════════════════════════════

def create_converter(extract_formulas: bool = False):
    """Создать DocumentConverter с нужными настройками.

    Создаётся один раз на партию файлов — веса моделей загружаются
    один раз, а не на каждый PDF.

    Args:
        extract_formulas: Распознавать формулы через VLM (CodeFormulaV2).

    Returns:
        Экземпляр DocumentConverter.
    """
    from docling.document_converter import DocumentConverter

    if extract_formulas:
        from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
        from docling.document_converter import FormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
        from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

        pipeline_opts = ThreadedPdfPipelineOptions(do_formula_enrichment=True)
        return DocumentConverter(
            format_options={
                InputFormat.PDF: FormatOption(
                    pipeline_options=pipeline_opts,
                    backend=DoclingParseDocumentBackend,
                    pipeline_cls=StandardPdfPipeline,
                )
            }
        )
    else:
        return DocumentConverter()


# ════════════════════════════════════════════════════════════════════
# Разделение PDF при нехватке памяти
# ════════════════════════════════════════════════════════════════════

def _split_pdf_at(pdf_path: Path, mid_page: int) -> tuple[Path, Path]:
    """Разделить PDF на две части в temp-файлы.

    Args:
        pdf_path: Путь к исходному PDF.
        mid_page: Номер страницы, по которой делить (включается во 2-ю часть).

    Returns:
        Кортеж (path_part1, path_part2) — временные файлы.
    """
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path))
    total = len(reader.pages)

    if mid_page <= 0 or mid_page >= total:
        raise ValueError(
            f"mid_page={mid_page} вне диапазона [1, {total - 1}]"
        )

    # Часть 1: страницы 0 .. mid_page-1
    writer1 = PdfWriter()
    for i in range(mid_page):
        writer1.add_page(reader.pages[i])

    # Часть 2: страницы mid_page .. end
    writer2 = PdfWriter()
    for i in range(mid_page, total):
        writer2.add_page(reader.pages[i])

    # Записываем во временные файлы (Docling нужен путь к файлу)
    tmp_dir = Path(tempfile.mkdtemp(prefix="docling_split_"))
    part1 = tmp_dir / f"{pdf_path.stem}_part1.pdf"
    part2 = tmp_dir / f"{pdf_path.stem}_part2.pdf"

    part1.write_bytes(b"")
    part2.write_bytes(b"")

    with open(part1, "wb") as f:
        writer1.write(f)
    with open(part2, "wb") as f:
        writer2.write(f)

    logger.info(
        "PDF разделён: часть 1 = стр. 1-%d, часть 2 = стр. %d-%d",
        mid_page, mid_page + 1, total,
    )

    return part1, part2


def _cleanup_split_files(part1: Path, part2: Path):
    """Удалить временные файлы и папку после разделения."""
    try:
        if part1.exists():
            part1.unlink()
        if part2.exists():
            part2.unlink()
        parent = part1.parent
        if parent.exists() and not list(parent.iterdir()):
            parent.rmdir()
    except Exception as e:
        logger.warning(f"Ошибка при удалении временных файлов: {e}")


def _is_memory_error(exc: Exception) -> bool:
    """Проверить, связана ли ошибка с нехваткой памяти.

    Ловит паттерны:
    - std::bad_alloc (C++ через pybind11)
    - "bad allocation" (ONNX Runtime RuntimeException)
    - Python MemoryError
    """
    exc_text = str(exc).lower()
    # C++ bad_alloc / ONNX "bad allocation"
    if "bad_alloc" in exc_text or "bad allocation" in exc_text:
        return True
    if isinstance(exc, MemoryError):
        return True
    # Проверяем цепочку причин (cause)
    cause = exc.__cause__
    if cause:
        cause_text = str(cause).lower()
        if "bad_alloc" in cause_text or "bad allocation" in cause_text:
            return True
    # Проверяем __context__ (исключение, возникшее при обработке другого)
    ctx = exc.__context__
    if ctx and ctx is not cause:
        ctx_text = str(ctx).lower()
        if "bad_alloc" in ctx_text or "bad allocation" in ctx_text:
            return True
    return False


# ════════════════════════════════════════════════════════════════════
# Конвертация одного PDF через Docling
# ════════════════════════════════════════════════════════════════════

def _convert_single(
    converter,
    pdf_path: Path,
) -> tuple[str, int, int, int, object]:
    """Конвертировать один PDF через готовый конвертер.

    Args:
        converter: Экземпляр DocumentConverter.
        pdf_path: Путь к PDF (оригинал или временная часть).

    Returns:
        (markdown_text, placeholder_count, formula_count, formula_not_decoded, conv_result)
    """
    result = converter.convert(str(pdf_path))
    md_content = result.document.export_to_markdown()

    # Подсчёт заглушек изображений
    placeholder_count = md_content.count("<!-- image -->")

    # Подсчёт нераспознанных формул
    formula_not_decoded = md_content.count("<!-- formula-not-decoded -->")

    # Заменяем заглушки Docling на [IMAGE_N]
    img_idx = 0
    new_lines = []
    for line in md_content.split("\n"):
        if "<!-- image -->" in line:
            img_idx += 1
            line = line.replace("<!-- image -->", f"[IMAGE_{img_idx}]")
        new_lines.append(line)
    md_content = "\n".join(new_lines)

    # Подсчёт распознанных формул
    formula_count = _count_latex_formulas(md_content)

    return md_content, placeholder_count, formula_count, formula_not_decoded, result


# ════════════════════════════════════════════════════════════════════
# Основная функция
# ════════════════════════════════════════════════════════════════════

def parse_pdf_to_markdown(
    pdf_path: str | Path,
    output_dir: str | Path,
    extract_images: bool = False,
    image_scale: float = 2.0,
    extract_formulas: bool = False,
    converter=None,
    max_pages: int = 30,
) -> dict:
    """Конвертировать один PDF-файл в Markdown через Docling.

    Если PDF содержит больше max_pages страниц — проактивно делится
    пополам и конвертируется по частям (склейка в один .md).
    Это предотвращает потерю страниц из-за bad_alloc.

    Args:
        pdf_path: Путь к PDF-файлу.
        output_dir: Папка для сохранения результата.
        extract_images: Если True — вырезать изображения из PDF.
        image_scale: Масштаб для вырезания изображений (dpi = 72 * scale).
        extract_formulas: Если True — распознавать формулы через VLM.
        converter: Внешний DocumentConverter (если None — создаётся свой).
            Передача извне экономит время — веса моделей загружаются один раз.
        max_pages: Порог страниц для проактивного деления (0 = не делить).

    Returns:
        dict с ключами:
            - markdown_path: Path до сохранённого .md файла
            - images_dir: Path до папки с картинками (или None)
            - images_count: количество извлечённых изображений
            - image_placeholders: количество заглушек [IMAGE_N] в тексте
            - formula_count: количество распознанных формул (LaTeX)
            - formula_not_decoded: количество нераспознанных формул
            - split_used: True если PDF был разбит на части
            - error: текст ошибки (если есть)
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        return {"error": f"Файл не найден: {pdf_path}"}

    # Создаём выходную папку
    stem = pdf_path.stem
    article_dir = output_dir / stem
    article_dir.mkdir(parents=True, exist_ok=True)

    # Конвертер: внешний или создаём свой
    own_converter = converter is None
    if own_converter:
        converter = create_converter(extract_formulas)

    # Проактивное деление: если страниц > max_pages — сразу делим пополам
    if max_pages > 0:
        try:
            from pypdf import PdfReader
            total_pages = len(PdfReader(str(pdf_path)).pages)
            if total_pages > max_pages:
                logger.info(
                    "PDF %s: %d страниц > max_pages=%d → проактивное деление",
                    pdf_path.name, total_pages, max_pages,
                )
                try:
                    result_dict = _try_convert_split(
                        converter, pdf_path, article_dir, stem,
                        extract_images, image_scale, extract_formulas,
                    )
                    return result_dict
                except Exception as split_err:
                    logger.error(
                        "Ошибка при проактивном разделении %s: %s",
                        pdf_path.name, split_err,
                    )
                    return {
                        "error": f"Проактивное разделение не удалось: {split_err}",
                        "stem": stem,
                    }
        except Exception as e:
            logger.warning(
                "Не удалось проверить кол-во страниц %s: %s. Конвертируем целиком.",
                pdf_path.name, e,
            )

    try:
        result_dict = _try_convert(
            converter, pdf_path, article_dir, stem,
            extract_images, image_scale,
        )
        return result_dict

    except Exception as e:
        if _is_memory_error(e):
            logger.warning(
                "Нехватка памяти при парсинге %s (%s). "
                "Пробуем разделить PDF пополам...",
                pdf_path.name, type(e).__name__,
            )
            try:
                result_dict = _try_convert_split(
                    converter, pdf_path, article_dir, stem,
                    extract_images, image_scale, extract_formulas,
                )
                return result_dict
            except Exception as split_err:
                logger.error(
                    "Ошибка даже после разделения PDF: %s", split_err
                )
                return {
                    "error": (
                        f"Нехватка памяти (bad_alloc). "
                        f"Попытка разделения тоже не удалась: {split_err}"
                    ),
                    "stem": stem,
                }
        else:
            logger.exception(
                "Ошибка при парсинге %s: %s", pdf_path.name, e
            )
            return {"error": str(e), "stem": stem}

    finally:
        pass


def _try_convert(
    converter,
    pdf_path: Path,
    article_dir: Path,
    stem: str,
    extract_images: bool,
    image_scale: float,
) -> dict:
    """Попытка конвертации PDF целиком."""
    logger.info("Конвертация целиком: %s", pdf_path.name)

    md_content, placeholder_count, formula_count, formula_not_decoded, conv_result = \
        _convert_single(converter, pdf_path)

    # Сохраняем Markdown
    md_path = article_dir / f"{stem}.md"
    md_path.write_text(md_content, encoding="utf-8")

    # Извлечение изображений
    images_dir = None
    images_count = 0
    if extract_images:
        images_dir_path = article_dir / "images"
        images_dir_path.mkdir(parents=True, exist_ok=True)
        images_count = _extract_images(conv_result, images_dir_path, image_scale)
        images_dir = images_dir_path

    return {
        "markdown_path": md_path,
        "images_dir": images_dir,
        "images_count": images_count,
        "image_placeholders": placeholder_count,
        "formula_count": formula_count,
        "formula_not_decoded": formula_not_decoded,
        "stem": stem,
        "split_used": False,
    }


def _try_convert_split(
    converter,
    pdf_path: Path,
    article_dir: Path,
    stem: str,
    extract_images: bool,
    image_scale: float,
    extract_formulas: bool,
) -> dict:
    """Конвертация PDF по частям (после разделения пополам)."""
    from pypdf import PdfReader

    # Определяем точку разделения
    total_pages = len(PdfReader(str(pdf_path)).pages)
    if total_pages < 2:
        raise RuntimeError(
            f"PDF содержит только {total_pages} стр. — разделение невозможно"
        )

    mid = total_pages // 2
    part1, part2 = _split_pdf_at(pdf_path, mid)

    try:
        all_md = []
        total_placeholders = 0
        total_formula_count = 0
        total_formula_not_decoded = 0
        total_images_count = 0
        images_dir = None

        for part_idx, part_path in enumerate([part1, part2], start=1):
            logger.info(
                "Часть %d/2: стр. %s (%s)",
                part_idx,
                f"1-{mid}" if part_idx == 1 else f"{mid + 1}-{total_pages}",
                part_path.name,
            )

            try:
                md_content, placeholders, formulas, formulas_nd, conv_result = \
                    _convert_single(converter, part_path)

                all_md.append(md_content)
                total_placeholders += placeholders
                total_formula_count += formulas
                total_formula_not_decoded += formulas_nd

                # Извлечение изображений из каждой части
                if extract_images:
                    img_dir = article_dir / "images"
                    img_dir.mkdir(parents=True, exist_ok=True)
                    count = _extract_images(conv_result, img_dir, image_scale)
                    total_images_count += count
                    images_dir = img_dir

            except Exception as e:
                if _is_memory_error(e):
                    logger.warning(
                        "Часть %d тоже вызвала not enough memory — пропускаем: %s",
                        part_idx, e,
                    )
                    all_md.append(
                        f"\n\n<!-- Часть {part_idx} пропущена: нехватка памяти -->\n\n"
                    )
                else:
                    raise

        # Склеиваем Markdown
        final_md = "\n\n".join(all_md)
        md_path = article_dir / f"{stem}.md"
        md_path.write_text(final_md, encoding="utf-8")

        return {
            "markdown_path": md_path,
            "images_dir": images_dir,
            "images_count": total_images_count,
            "image_placeholders": total_placeholders,
            "formula_count": total_formula_count,
            "formula_not_decoded": total_formula_not_decoded,
            "stem": stem,
            "split_used": True,
        }

    finally:
        _cleanup_split_files(part1, part2)


# ════════════════════════════════════════════════════════════════════
# Извлечение изображений
# ════════════════════════════════════════════════════════════════════

def _extract_images(conv_result, images_dir: Path, scale: float) -> int:
    """Извлечь изображения из результата Docling.

    Args:
        conv_result: Результат DocumentConverter.convert().
        images_dir: Папка для сохранения картинок.
        scale: Масштаб (dpi = 72 * scale).

    Returns:
        Количество сохранённых изображений.
    """
    count = 0

    # Docling предоставляет pictures в conv_result.pictures
    pictures = getattr(conv_result, "pictures", []) or []
    for pic_idx, pic in enumerate(pictures, start=1):
        try:
            img = _get_picture_image(pic, conv_result, scale)
            if img is not None:
                img_path = images_dir / f"image_{pic_idx:03d}.png"
                img.save(str(img_path), "PNG")
                count += 1
        except Exception as e:
            logger.warning(f"Ошибка при извлечении изображения {pic_idx}: {e}")

    return count


def _get_picture_image(pic, conv_result, scale: float):
    """Получить PIL-изображение из объекта picture результата Docling."""
    from PIL import Image
    import io

    # Способ 1: через метод get_image (новые версии Docling)
    if hasattr(pic, "get_image"):
        try:
            return pic.get_image(conv_result, scale=scale)
        except Exception:
            pass

    # Способ 2: через атрибут image
    if hasattr(pic, "image") and pic.image is not None:
        return pic.image

    return None


# ════════════════════════════════════════════════════════════════════
# Подсчёт формул
# ════════════════════════════════════════════════════════════════════

import re as _re

# Паттерн для поиска LaTeX-формул: $...$ (inline) и $$...$$ (display)
_LATEX_INLINE_RE = _re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_LATEX_DISPLAY_RE = _re.compile(r"\$\$(.+?)\$\$", _re.DOTALL)


def _count_latex_formulas(text: str) -> int:
    """Подсчитать количество LaTeX-формул в тексте.

    Считает отдельно display ($$...$$) и inline ($...$) формулы.
    """
    display = len(_LATEX_DISPLAY_RE.findall(text))
    inline = len(_LATEX_INLINE_RE.findall(text))
    return display + inline


# ════════════════════════════════════════════════════════════════════
# Утилиты
# ════════════════════════════════════════════════════════════════════

def get_pdf_files(directory: str | Path) -> list[Path]:
    """Найти все PDF-файлы в папке.

    Returns:
        Отсортированный список Path к PDF файлам.
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    )


def get_parsed_articles(output_dir: str | Path) -> list[str]:
    """Найти уже обработанные статьи (есть подпапка с .md файлом).

    Returns:
        Список stem-ов обработанных файлов.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []

    parsed = []
    for sub in sorted(output_dir.iterdir()):
        if sub.is_dir():
            md_files = list(sub.glob("*.md"))
            if md_files:
                parsed.append(sub.name)
    return parsed


# ════════════════════════════════════════════════════════════════════
# Удалённый парсинг через API (Colab / GPU-сервер) — Polling Pattern
# ════════════════════════════════════════════════════════════════════

# Timeout-ы для HTTP-запросов через cloudflare tunnel (секунды)
_UPLOAD_TIMEOUT = 300       # POST /parse — загрузка PDF (до 5 мин)
_POLL_TIMEOUT = 30          # GET /status — короткий polling-запрос
_RESULT_TIMEOUT = 300       # GET /result — скачивание результата (до 5 мин)
_HEALTH_TIMEOUT = 15        # GET /health — проверка доступности
_MAX_RETRIES = 3            # количество повторных попыток при сбое
_RETRY_DELAYS = [10, 20, 30]  # задержки между попытками (сек)


def _requests_session() -> "requests.Session":
    """Создать requests.Session с retry-настройками."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "book_parsing_rag_indexer/1.0"})
    return session


def check_remote_server(url: str) -> dict:
    """Проверить доступность удалённого сервера Docling.

    Args:
        url: Базовый URL сервера (например, https://xxx.trycloudflare.com).

    Returns:
        dict с ключами:
            - available: bool
            - gpu: str (информация о GPU)
            - converter_loaded: bool
            - error: str | None
    """
    import requests

    url = url.rstrip("/")

    try:
        session = _requests_session()
        resp = session.get(f"{url}/health", timeout=_HEALTH_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        return {
            "available": True,
            "gpu": data.get("gpu", "?"),
            "converter_loaded": data.get("converter_loaded", False),
            "error": None,
        }
    except Exception as e:
        return {
            "available": False,
            "gpu": None,
            "converter_loaded": False,
            "error": str(e),
        }


def parse_pdf_remote(
    pdf_path: str | Path,
    output_dir: str | Path,
    remote_url: str,
    extract_images: bool = False,
    image_scale: float = 2.0,
) -> dict:
    """Отправить PDF на удалённый сервер Docling (polling pattern).

    Схема работы:
    1. POST /parse → отправить PDF, получить job_id (мгновенно)
    2. GET /status/{job_id} каждые 5 сек → ждём завершения
    3. GET /result/{job_id} → забираем результат

    Использует requests с увеличенными timeout-ами и retry — устойчиво
    к медленным cloudflare tunnel, обрывам соединения, 502/503/504.

    Args:
        pdf_path: Путь к локальному PDF-файлу.
        output_dir: Папка для сохранения результата.
        remote_url: URL удалённого сервера (https://xxx.trycloudflare.com).
        extract_images: Если True — извлечь изображения.
        image_scale: Масштаб для изображений.

    Returns:
        dict в том же формате, что и parse_pdf_to_markdown().
    """
    import requests as _requests
    import time as _time

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if not pdf_path.exists():
        return {"error": f"Файл не найден: {pdf_path}"}

    stem = pdf_path.stem
    article_dir = output_dir / stem
    article_dir.mkdir(parents=True, exist_ok=True)

    remote_url = remote_url.rstrip("/")
    session = _requests_session()

    logger.info("Удалённый парсинг (polling): %s → %s", pdf_path.name, remote_url)

    # ── Шаг 1: POST /parse — отправить PDF, получить job_id ──────
    pdf_bytes = pdf_path.read_bytes()
    pdf_size_mb = len(pdf_bytes) / (1024 * 1024)

    logger.info(
        "Отправка PDF (%s, %.1f МБ) на удалённый сервер...",
        pdf_path.name, pdf_size_mb,
    )

    parse_resp = None
    last_err = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = session.post(
                f"{remote_url}/parse",
                files={"file": (pdf_path.name, pdf_bytes, "application/pdf")},
                data={
                    "extract_images": str(extract_images).lower(),
                    "image_scale": str(image_scale),
                },
                timeout=_UPLOAD_TIMEOUT,
            )
            resp.raise_for_status()
            parse_resp = resp.json()
            break

        except _requests.exceptions.ConnectionError as e:
            last_err = e
            logger.warning(
                "Попытка %d/%d: ошибка соединения при отправке %s (%.1f МБ): %s",
                attempt, _MAX_RETRIES, pdf_path.name, pdf_size_mb, e,
            )
        except _requests.exceptions.Timeout as e:
            last_err = e
            logger.warning(
                "Попытка %d/%d: таймаут при отправке %s (%.1f МБ): %s",
                attempt, _MAX_RETRIES, pdf_path.name, pdf_size_mb, e,
            )
        except _requests.exceptions.HTTPError as e:
            # Сервер ответил с ошибкой — повторять нет смысла для 4xx
            if resp.status_code < 500:
                logger.error(
                    "Ошибка сервера при отправке %s: HTTP %d — %s",
                    pdf_path.name, resp.status_code, resp.text[:200],
                )
                return {"error": f"HTTP {resp.status_code}: {resp.text[:300]}", "stem": stem}
            last_err = e
            logger.warning(
                "Попытка %d/%d: сервер вернул HTTP %d при отправке %s",
                attempt, _MAX_RETRIES, resp.status_code, pdf_path.name,
            )
        except Exception as e:
            last_err = e
            logger.warning(
                "Попытка %d/%d: неожиданная ошибка при отправке %s: %s",
                attempt, _MAX_RETRIES, pdf_path.name, e,
            )

        # Задержка перед следующей попыткой
        if attempt < _MAX_RETRIES:
            delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
            logger.info("Повторная отправка через %d сек...", delay)
            _time.sleep(delay)

    if parse_resp is None:
        logger.error(
            "Не удалось отправить %s после %d попыток: %s",
            pdf_path.name, _MAX_RETRIES, last_err,
        )
        return {"error": f"Отправка не удалась после {_MAX_RETRIES} попыток: {last_err}", "stem": stem}

    job_id = parse_resp.get("job_id")
    if not job_id:
        err = parse_resp.get("detail", parse_resp.get("error", "нет job_id"))
        return {"error": f"Сервер не вернул job_id: {err}", "stem": stem}

    logger.info("Задача создана: job_id=%s, ожидаю завершение...", job_id[:8])

    # ── Шаг 2: GET /status/{job_id} — polling каждые 5 сек ───────
    _POLL_INTERVAL = 5      # сек между запросами
    _MAX_POLL_TIME = 600    # максимальное время ожидания (10 мин)
    _start_poll = _time.time()
    _consecutive_poll_errors = 0
    _MAX_CONSECUTIVE_ERRORS = 10  # макс. подряд ошибок polling → фатал

    while True:
        elapsed = _time.time() - _start_poll
        if elapsed > _MAX_POLL_TIME:
            return {
                "error": f"Таймаут: парсинг не завершился за {_MAX_POLL_TIME}с",
                "stem": stem,
            }

        try:
            status_resp = session.get(
                f"{remote_url}/status/{job_id}",
                timeout=_POLL_TIMEOUT,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            _consecutive_poll_errors = 0  # сброс счётчика ошибок

        except Exception as e:
            _consecutive_poll_errors += 1
            logger.warning(
                "Ошибка polling статуса (job=%s, %.0fс, ошибок подряд: %d): %s",
                job_id[:8], elapsed, _consecutive_poll_errors, e,
            )
            if _consecutive_poll_errors >= _MAX_CONSECUTIVE_ERRORS:
                return {
                    "error": (
                        f"Polling: {_MAX_CONSECUTIVE_ERRORS} ошибок подряд. "
                        f"Вероятно, туннель или сервер недоступны: {e}"
                    ),
                    "stem": stem,
                }
            _time.sleep(_POLL_INTERVAL)
            continue

        status = status_data.get("status", "unknown")

        if status == "done":
            logger.info(
                "Парсинг завершён (job=%s, %.1fс) — забираю результат...",
                job_id[:8], elapsed,
            )
            break

        if status == "error":
            err_msg = status_data.get("error", "неизвестная ошибка")
            logger.error("Ошибка парсинга на сервере (job=%s): %s", job_id[:8], err_msg)
            return {"error": f"Ошибка сервера: {err_msg}", "stem": stem}

        # pending / processing — ждём
        if int(elapsed) % 30 < _POLL_INTERVAL:
            logger.info(
                "Ожидание... статус=%s, прошло %.0fс (job=%s)",
                status, elapsed, job_id[:8],
            )

        _time.sleep(_POLL_INTERVAL)

    # ── Шаг 3: GET /result/{job_id} — забираем результат ─────────
    resp_data = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result_resp = session.get(
                f"{remote_url}/result/{job_id}",
                timeout=_RESULT_TIMEOUT,
            )
            result_resp.raise_for_status()
            resp_data = result_resp.json()
            break
        except Exception as e:
            logger.warning(
                "Попытка %d/%d: ошибка получения результата (job=%s): %s",
                attempt, _MAX_RETRIES, job_id[:8], e,
            )
            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
                _time.sleep(delay)

    if resp_data is None:
        return {
            "error": f"Не удалось получить результат после {_MAX_RETRIES} попыток",
            "stem": stem,
        }

    # ── Обработка результата ──────────────────────────────────────
    try:
        if resp_data.get("error"):
            return {"error": f"Ошибка сервера: {resp_data['error']}", "stem": stem}

        md_content = resp_data.get("markdown", "")
        if not md_content:
            return {"error": "Сервер вернул пустой результат", "stem": stem}

        # Сохраняем Markdown
        md_path = article_dir / f"{stem}.md"
        md_path.write_text(md_content, encoding="utf-8")

        # Сохраняем изображения (если есть)
        images_dir = None
        images_count = 0
        images_data_list = resp_data.get("images", [])
        if images_data_list:
            img_dir = article_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            for img_info in images_data_list:
                img_filename = img_info.get("filename", "image.png")
                img_b64 = img_info.get("data", "")
                if img_b64:
                    img_bytes = base64.b64decode(img_b64)
                    img_path = img_dir / img_filename
                    img_path.write_bytes(img_bytes)
                    images_count += 1
            images_dir = img_dir

        result = {
            "markdown_path": md_path,
            "images_dir": images_dir,
            "images_count": images_count,
            "image_placeholders": resp_data.get("image_placeholders", 0),
            "formula_count": resp_data.get("formula_count", 0),
            "formula_not_decoded": resp_data.get("formula_not_decoded", 0),
            "stem": stem,
            "split_used": False,
            "parse_time_sec": resp_data.get("parse_time_sec", 0),
            "remote": True,
        }

        logger.info(
            "Удалённый парсинг завершён: %s (%d символов, %d формул, %.1fs)",
            pdf_path.name, len(md_content),
            result["formula_count"],
            result["parse_time_sec"],
        )

        return result

    except Exception as e:
        logger.exception(f"Ошибка обработки ответа для {pdf_path.name}: {e}")
        return {"error": str(e), "stem": stem}
