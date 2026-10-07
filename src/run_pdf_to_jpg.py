"""Оркестратор конвертации PDF → JPEG.

Обходит структуру: base_dir/название_книги/split_pdf/*.pdf
Конвертирует каждый PDF в JPG и сохраняет:
  base_dir/название_книги/convert_in_jpg/имя.jpg

Поддерживает отмену и обратный вызов прогресса.
Может вызываться из CLI или из веб-интерфейса.
"""

import logging
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Callable

from src.config import setup_logging
from src.pdf_to_jpg import convert_pdf_page_to_jpg

logger = logging.getLogger(__name__)


def _find_split_pdf_files(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> list[tuple[str, Path, Path]]:
    """Найти все PDF в подпапках */split_pdf/.

    Args:
        base_dir: Корневая папка с книгами.
        book_names: Если задано — искать только в указанных папках книг.
                    Если None — искать во всех подпапках.

    Returns:
        Список кортежей: (book_name, pdf_path, jpg_output_dir)
        book_name — имя родительской папки (название книги)
        pdf_path — полный путь к PDF файлу
        jpg_output_dir — путь к папке book_name/convert_in_jpg/
    """
    results: list[tuple[str, Path, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue

        # Фильтрация по выбранным книгам
        if book_names is not None and book_dir.name not in book_names:
            continue

        split_dir = book_dir / "split_pdf"
        if not split_dir.is_dir():
            logger.debug("Пропуск %s: нет split_pdf/", book_dir.name)
            continue

        jpg_dir = book_dir / "convert_in_jpg"

        for pdf_file in sorted(split_dir.glob("*.pdf")):
            results.append((book_dir.name, pdf_file, jpg_dir))

    return results


def run_pdf_to_jpg(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    dpi: int = 200,
    poppler_path: str | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить конвейер конвертации PDF → JPEG.

    Обходит: base_dir/*/split_pdf/*.pdf
    Сохраняет: base_dir/*/convert_in_jpg/*.jpg

    Args:
        base_dir:          Корневая папка с книгами (напр. data/books).
        book_names:        Список названий книг (имён подпапок) для обработки.
                           Если None — обрабатываются все найденные книги.
        dpi:               Разрешение рендеринга (72–600).
        poppler_path:      Путь к бинарникам Poppler (для Windows).
        cancel_event:      Событие отмены (threading.Event).
        progress_callback: Функция(progress, current, total, current_file, stats).
                           Вызывается после каждого файла.
        log_level:         Уровень логирования.

    Returns:
        dict: total, success, errors, elapsed_sec, cancelled.
    """
    setup_logging(log_level)

    logger.info("=== Запуск конвертации PDF → JPG ===")
    logger.info("Базовая папка: %s, DPI: %d, Книги: %s", base_dir, dpi,
                book_names if book_names else "все")

    base = Path(base_dir)

    # Собираем список файлов для обработки
    files = _find_split_pdf_files(base, book_names=book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет PDF файлов для обработки в %s", base_dir)
        return {
            "total": 0, "success": 0, "errors": 0,
            "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d PDF файлов", total)

    t0 = time.perf_counter()
    success = 0
    errors = 0
    cancelled = False

    for i, (book_name, pdf_path, jpg_dir) in enumerate(files, start=1):
        # Проверка отмены
        if cancel_event and cancel_event.is_set():
            logger.info("Конвертация отменена пользователем на файле %d/%d", i, total)
            cancelled = True
            break

        jpg_name = pdf_path.stem + ".jpg"
        relative_name = f"{book_name}/split_pdf/{pdf_path.name}"

        logger.info("[%d/%d] %s → %s", i, total, relative_name, jpg_name)

        try:
            # Читаем PDF
            pdf_buf = BytesIO(pdf_path.read_bytes())

            # Конвертируем
            jpg_buf, width, height = convert_pdf_page_to_jpg(
                pdf_buf, dpi=dpi, poppler_path=poppler_path,
            )

            # Создаём выходную директорию и сохраняем
            jpg_dir.mkdir(parents=True, exist_ok=True)
            out_path = jpg_dir / jpg_name
            out_path.write_bytes(jpg_buf.getvalue())

            success += 1
            logger.info(
                "[%d/%d] OK — %s/%s (%dx%d, %d байт)",
                i, total, book_name, jpg_name,
                width, height, jpg_buf.getbuffer().nbytes,
            )

        except Exception:
            errors += 1
            logger.exception("[%d/%d] ОШИБКА — %s", i, total, relative_name)

        # Обратный вызов прогресса
        if progress_callback:
            stats = {
                "total": total,
                "success": success,
                "errors": errors,
            }
            progress_callback(i, total, relative_name, stats)

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Готово: %d/%d успешно, %d ошибок%s, %.2f с ===",
        success, total, errors,
        ", ОТМЕНЕНО" if cancelled else "",
        elapsed,
    )

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }