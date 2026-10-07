"""Оркестратор выравнивания заголовков по оглавлению.

Обходит структуру: base_dir/название_книги/parsed/toc/*.md + parsed/*.md
Для каждой книги с TOC: парсит оглавление -> исправляет заголовки в parsed/*.md.

Поддерживает:
- Выбор конкретных книг (book_names)
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
from src.fix_heading_levels import (
    parse_toc_files,
    process_file,
    count_heading_mismatches,
    find_unmatched_toc_entries,
)

logger = logging.getLogger(__name__)


def find_books_with_toc(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти книги с папкой parsed/toc/.

    Returns:
        Список кортежей: (book_name, toc_dir)
    """
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue

        toc_dir = book_dir / "parsed" / "toc"
        if toc_dir.is_dir() and any(toc_dir.glob("*.md")):
            results.append((book_dir.name, toc_dir))

    return results


def count_all_mismatches(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчитать несовпадения заголовков и покрытие TOC (без модификации).

    Returns:
        dict: total_books, total_headings, wrong_level, not_in_toc, correct,
              total_toc_entries, unmatched_toc_count, unmatched_toc_details
    """
    books = find_books_with_toc(base_dir)
    totals = {
        "total_books": 0,
        "total_headings": 0,
        "wrong_level": 0,
        "not_in_toc": 0,
        "correct": 0,
        "total_toc_entries": 0,
        "unmatched_toc_count": 0,
        "unmatched_toc_details": [],  # list of {"book": str, "toc_text": str, "toc_level": int}
    }

    for book_name, toc_dir in books:
        if book_names and book_name not in book_names:
            continue

        parsed_dir = toc_dir.parent  # parsed/
        toc_entries = parse_toc_files(toc_dir)

        if not toc_entries:
            continue

        book_has_files = False
        for md_file in sorted(parsed_dir.glob("*.md")):
            if md_file.parent.name == "toc":
                continue
            try:
                content = md_file.read_text(encoding="utf-8")
                mismatches = count_heading_mismatches(content, toc_entries)
                book_has_files = True
                totals["total_headings"] += mismatches["total"]
                totals["wrong_level"] += mismatches["wrong_level"]
                totals["not_in_toc"] += mismatches["not_in_toc"]
                totals["correct"] += mismatches["correct"]
            except Exception:
                pass

        if book_has_files:
            totals["total_books"] += 1

        # Проверка покрытия TOC: какие записи не найдены в parsed
        totals["total_toc_entries"] += len(toc_entries)
        toc_match_results = find_unmatched_toc_entries(toc_entries, parsed_dir)
        for entry in toc_match_results:
            if not entry["matched"]:
                totals["unmatched_toc_count"] += 1
                totals["unmatched_toc_details"].append({
                    "book": book_name,
                    "toc_text": entry["toc_text"],
                    "toc_level": entry["toc_level"],
                })

    return totals


# ======================================================================
# Публичный интерфейс
# ======================================================================

def run_fix_heading_levels(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    remove_unmatched: bool = True,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить выравнивание заголовков по оглавлению.

    Для каждой книги с parsed/toc/:
    1. Парсит TOC -> список (текст, уровень)
    2. Обходит parsed/*.md (исключая toc/)
    3. Исправляет уровни заголовков in-place

    Args:
        base_dir:          Корневая папка с книгами.
        book_names:        Список книг для обработки. None = все с TOC.
        remove_unmatched:  Удалять # у заголовков, не найденных в TOC.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict с результатами.
    """
    setup_logging(log_level)

    logger.info("=== Запуск выравнивания заголовков ===")
    logger.info("Базовая папка: %s", base_dir)
    logger.info("Удалять # у не-TOC заголовков: %s", remove_unmatched)

    base = Path(base_dir)
    books = find_books_with_toc(base)

    # Фильтр по выбранным книгам
    if book_names:
        books = [(n, d) for n, d in books if n in book_names]

    if not books:
        logger.info("Нет книг с parsed/toc/ в %s", base_dir)
        return {
            "total": 0, "modified": 0, "unchanged": 0, "errors": 0,
            "fixed_level": 0, "removed_hash": 0,
            "elapsed_sec": 0.0, "cancelled": False,
        }

    # Собираем все файлы для обработки
    all_files: list[tuple[str, Path, Path]] = []  # (book_name, toc_dir, md_file)
    for book_name, toc_dir in books:
        parsed_dir = toc_dir.parent
        for md_file in sorted(parsed_dir.glob("*.md")):
            if md_file.parent.name == "toc":
                continue
            all_files.append((book_name, toc_dir, md_file))

    total = len(all_files)
    if total == 0:
        logger.info("Нет .md файлов для обработки")
        return {
            "total": 0, "modified": 0, "unchanged": 0, "errors": 0,
            "fixed_level": 0, "removed_hash": 0,
            "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d файлов из %d книг", total, len(books))

    t0 = time.perf_counter()
    modified = 0
    unchanged = 0
    errors = 0
    total_fixed = 0
    total_removed = 0
    cancelled = False

    # TOC-кэш по книгам (чтобы не парсить заново)
    toc_cache: dict[str, list] = {}

    for i, (book_name, toc_dir, md_path) in enumerate(all_files):
        if cancel_event and cancel_event.is_set():
            logger.info("Выравнивание отменено на файле %s", md_path.name)
            cancelled = True
            break

        relative_name = f"{book_name}/parsed/{md_path.name}"

        # Загрузить TOC для книги (с кэшем)
        if book_name not in toc_cache:
            toc_cache[book_name] = parse_toc_files(toc_dir)
        toc_entries = toc_cache[book_name]

        if not toc_entries:
            logger.debug("Пропуск %s: пустой TOC", book_name)
            unchanged += 1
            continue

        logger.info("[%d/%d] %s", i + 1, total, relative_name)

        result = process_file(md_path, toc_entries, remove_unmatched)

        if result["error"]:
            errors += 1
            logger.error("❌ Ошибка %s: %s", relative_name, result["error"])
        elif result["modified"]:
            modified += 1
            stats = result["stats"]
            total_fixed += stats["fixed_level"]
            total_removed += stats["removed_hash"]
            logger.info(
                "✅ %s: уровень=%d, удалено #=%d",
                relative_name, stats["fixed_level"], stats["removed_hash"],
            )
        else:
            unchanged += 1

        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "modified": modified,
                "unchanged": unchanged,
                "errors": errors,
                "fixed_level": total_fixed,
                "removed_hash": total_removed,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Выравнивание завершено: %d изменено, %d без изменений, "
        "%d ошибок, уровень=%d, # удалено=%d%s, %.2fс ===",
        modified, unchanged, errors, total_fixed, total_removed,
        ", ОТМЕНЕНО" if cancelled else "",
        elapsed,
    )

    return {
        "total": total,
        "modified": modified,
        "unchanged": unchanged,
        "errors": errors,
        "fixed_level": total_fixed,
        "removed_hash": total_removed,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }