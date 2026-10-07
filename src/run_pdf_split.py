"""Оркестратор разделения PDF на отдельные страницы.

Связывает источник, логику разделения и сохранение.
Результат: для каждого PDF создаётся подпапка с именем файла (без расширения),
внутрь — подпапка split_pdf/, а в ней — страницы 000.pdf, 001.pdf, ... в соответствии с start_page.

Может вызываться из CLI или из веб-интерфейса.
"""

import logging
import threading
import time
from pathlib import Path

from src.config import setup_logging
from src.pdf_split import split_pdf_to_pages
from src.sources import DataSource, create_source, PDF_EXTS
from src.storage_utils import save_result

logger = logging.getLogger(__name__)


def run_pdf_split(
    source_type: str = "local",
    dest_type: str = "local",
    local_source_dir: str = "",
    local_dest_dir: str = "",
    yandex_source_dir: str = "",
    yandex_dest_dir: str = "",
    start_page: int = 0,
    selected_files: list[str] | None = None,
    cancel_event: threading.Event | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить конвейер разделения PDF на страницы.

    Для каждого PDF файла создаётся подпапка dest_dir/pdf_stem/split_pdf/,
    внутрь которой сохраняются страницы: 000.pdf, 001.pdf и т.д.

    Args:
        source_type: 'local' или 'yandex'
        dest_type: 'local' или 'yandex'
        local_source_dir: путь к локальной папке с исходными PDF
        local_dest_dir: путь к локальной папке для результатов
        yandex_source_dir: путь на Яндекс.Диске к исходным PDF
        yandex_dest_dir: путь на Яндекс.Диске для результатов
        start_page: номер первой страницы (для именования файлов)
        selected_files: Список имён файлов для обработки (без пути, только имя).
                        Если None — обрабатываются все найденные PDF.
        cancel_event: событие отмены
        log_level: уровень логирования

    Returns:
        dict с результатами: total, success, errors, total_pages, elapsed_sec
    """
    setup_logging(log_level)

    logger.info("=== Запуск разделения PDF на страницы ===")
    logger.info("Откуда: %s → Куда: %s", source_type, dest_type)
    logger.info("Начало нумерации: %d", start_page)

    source = create_source(
        source_type=source_type,
        dest_type=dest_type,
        local_source_dir=local_source_dir,
        local_dest_dir=local_dest_dir,
        yandex_source_dir=yandex_source_dir,
        yandex_dest_dir=yandex_dest_dir,
    )

    # Создаём отдельный адаптер для записи в основное хранилище
    # (для save_result — локальный кэш + основное хранилище)
    dest_source = None
    if dest_type == "yandex":
        from src.sources import YandexSource

        dest_source = YandexSource(
            source_dir=yandex_dest_dir,
            dest_dir=yandex_dest_dir,
        )

    # Определяем локальную папку для кэша
    cache_dir = Path(local_dest_dir) if dest_type == "local" else None

    names = source.list_files(PDF_EXTS)

    # Фильтрация по выбранным файлам
    if selected_files is not None:
        names = [n for n in names if n in selected_files]

    total = len(names)
    if total == 0:
        logger.info("Нет PDF файлов для обработки")
        return {"total": 0, "success": 0, "errors": 0, "total_pages": 0, "elapsed_sec": 0.0}

    logger.info("К обработке: %d PDF файлов", total)

    t0 = time.perf_counter()
    success = 0
    errors = 0
    total_pages = 0
    current_page = start_page

    for i, name in enumerate(names, start=1):
        if cancel_event and cancel_event.is_set():
            logger.info("Разделение отменено на файле %d/%d", i, total)
            break

        # Имя подпапки = имя PDF без расширения
        pdf_stem = Path(name).stem
        logger.info("[%d/%d] %s (начиная со стр. %d)", i, total, name, current_page)

        try:
            pdf_buf = source.download(name)
            pages = split_pdf_to_pages(pdf_buf, start_page=current_page)

            for filename, page_buf in pages:
                # Сохраняем в подпапку: pdf_stem/split_pdf/003.pdf
                relative_path = f"{pdf_stem}/split_pdf/{filename}"
                save_result(
                    name=relative_path,
                    buffer=page_buf,
                    dest_source=dest_source,
                    local_cache_dir=cache_dir,
                )

            num_pages = len(pages)
            current_page += num_pages
            total_pages += num_pages
            success += 1
            logger.info(
                "[%d/%d] OK — %s → %d страниц (%s/%s ... %s/%s)",
                i, total, name, num_pages,
                pdf_stem, pages[0][0], pdf_stem, pages[-1][0],
            )
        except Exception:
            errors += 1
            logger.exception("[%d/%d] ОШИБКА — %s", i, total, name)

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Готово: %d файлов, %d страниц, %d ошибок, %.2f с ===",
        success, total_pages, errors, elapsed,
    )

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "total_pages": total_pages,
        "elapsed_sec": elapsed,
        "dest_dir": str(cache_dir) if cache_dir else (yandex_dest_dir or ""),
    }
