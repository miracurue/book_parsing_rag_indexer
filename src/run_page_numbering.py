"""Оркестратор расстановки номеров страниц.

Обходит структуру: base_dir/название_книги/parsed/without_figures/*.md
(если without_figures/ нет — fallback на parsed/*.md).
Обрабатывает каждый файл через page_numbering.add_page_numbers().
Сохраняет результаты: base_dir/название_книги/numbered/*.md

Поддерживает:
- Возобновление (пропускает уже обработанные файлы)
- Отмену через threading.Event
- Обратный вызов прогресса
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from src.config import setup_logging
from src.page_numbering import add_page_numbers, extract_page_number

logger = logging.getLogger(__name__)

MD_EXTS = (".md",)


def _find_md_files(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> list[tuple[str, Path, Path]]:
    """Найти все .md в подпапках */parsed/without_figures/ (приоритет) или */parsed/.

    Args:
        base_dir: Корневая папка с книгами.
        book_names: Список книг для обработки. None = все книги.

    Returns:
        Список кортежей: (book_name, md_path, numbered_output_dir)
    """
    results: list[tuple[str, Path, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue

        # Фильтр по выбранным книгам
        if book_names and book_dir.name not in book_names:
            continue

        parsed_dir = book_dir / "parsed"
        if not parsed_dir.is_dir():
            logger.debug("Пропуск %s: нет parsed/", book_dir.name)
            continue

        numbered_dir = book_dir / "numbered"

        # Приоритет: parsed/without_figures/ → fallback на parsed/
        wf_dir = parsed_dir / "without_figures"
        if wf_dir.is_dir():
            source_dir = wf_dir
            logger.debug("%s: используется without_figures/", book_dir.name)
        else:
            source_dir = parsed_dir
            logger.debug("%s: без without_figures/, используется parsed/", book_dir.name)

        for md_file in sorted(source_dir.glob("*")):
            if md_file.is_file() and md_file.suffix.lower() in MD_EXTS:
                results.append((book_dir.name, md_file, numbered_dir))

    return results


# ══════════════════════════════════════════════════════════════════════
# Публичный интерфейс
# ══════════════════════════════════════════════════════════════════════

def run_page_numbering(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить расстановку номеров страниц для всех .md файлов.

    Обходит: base_dir/*/parsed/*.md (или только выбранные книги).
    Сохраняет: base_dir/*/numbered/*.md

    Args:
        base_dir:          Корневая папка с книгами.
        book_names:        Список книг для обработки. None = все.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict: total, success, errors, skipped, elapsed_sec, cancelled.
    """
    setup_logging(log_level)

    logger.info("=== Запуск расстановки номеров страниц ===")
    logger.info("Базовая папка: %s", base_dir)
    if book_names:
        logger.info("Выбранные книги: %s", ", ".join(book_names))

    base = Path(base_dir)
    files = _find_md_files(base, book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет .md файлов для обработки в %s", base_dir)
        return {
            "total": 0, "success": 0, "errors": 0, "skipped": 0,
            "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d .md файлов", total)

    t0 = time.perf_counter()
    success = 0
    errors = 0
    skipped = 0
    cancelled = False

    for i, (book_name, md_path, numbered_dir) in enumerate(files):
        # Проверка отмены
        if cancel_event and cancel_event.is_set():
            logger.info("Нумерация отменена на файле %s", md_path.name)
            cancelled = True
            break

        filename = md_path.name
        relative_name = f"{book_name}/parsed/{filename}"

        # Проверяем, существует ли уже результат
        dest_path = numbered_dir / filename
        if dest_path.exists():
            logger.info("⏩ Пропуск: %s (уже обработан)", relative_name)
            skipped += 1
            if progress_callback:
                progress_callback(i + 1, total, relative_name, {
                    "success": success, "errors": errors, "skipped": skipped,
                })
            continue

        logger.info("[%d/%d] %s", i + 1, total, relative_name)

        try:
            # Читаем исходный файл
            content = md_path.read_text(encoding="utf-8")

            # Проверяем, что номер страницы извлекается
            page_num = extract_page_number(filename)
            if page_num is None:
                logger.warning(
                    "⚠️ Пропуск %s: номер страницы не найден в названии.",
                    filename,
                )
                skipped += 1
                if progress_callback:
                    progress_callback(i + 1, total, relative_name, {
                        "success": success, "errors": errors, "skipped": skipped,
                    })
                continue

            # Обрабатываем
            new_content = add_page_numbers(content, filename)

            # Сохраняем результат
            numbered_dir.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(new_content, encoding="utf-8")

            logger.info("Сохранён: %s/numbered/%s", book_name, filename)
            success += 1

        except Exception as e:
            errors += 1
            logger.exception("❌ Ошибка при обработке %s: %s", filename, e)

        # Обратный вызов прогресса
        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "success": success, "errors": errors, "skipped": skipped,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Нумерация завершена: %d успешно, %d ошибок, %d пропущено%s, %.2fс ===",
        success, errors, skipped,
        ", ОТМЕНЕНО" if cancelled else "",
        elapsed,
    )

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }