"""Оркестратор очистки файлов глав от маркированных блоков.

Обходит: base_dir/*/chapters/*.md
Сохраняет: base_dir/*/extracted/clear_chapters/*.md

Поддерживает:
- Выбор конкретных книг (book_names)
- Пересоздание (recreate=True)
- Отмену через threading.Event
- Обратный вызов прогресса
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from src.config import setup_logging
from src.clean_tagged_content import process_file, count_tagged_blocks

logger = logging.getLogger(__name__)

MD_EXTS = (".md",)
CLEAR_CHAPTERS_SUBDIR = "clear_chapters"


def _find_chapter_files(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Найти все .md в подпапках */chapters/."""
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        if book_names and book_dir.name not in book_names:
            continue
        chapters_dir = book_dir / "chapters"
        if not chapters_dir.is_dir():
            continue
        for md_file in sorted(chapters_dir.glob("*.md")):
            if md_file.is_file():
                results.append((book_dir.name, md_file))

    return results


def find_books_with_chapters(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти книги с папкой chapters/."""
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        chapters_dir = book_dir / "chapters"
        if chapters_dir.is_dir():
            results.append((book_dir.name, chapters_dir))

    return results


def count_all_tagged_blocks(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчитать маркированные блоки во всех файлах chapters/."""
    total_tables = 0
    total_figures = 0
    total_files = 0

    for _, md_path in _find_chapter_files(base_dir, book_names):
        total_files += 1
        try:
            content = md_path.read_text(encoding="utf-8")
            counts = count_tagged_blocks(content)
            total_tables += counts["tables"]
            total_figures += counts["figures"]
        except Exception:
            pass

    return {
        "total_tables": total_tables,
        "total_figures": total_figures,
        "total_files": total_files,
    }


def run_clean_tagged_content(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    recreate: bool = False,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить очистку chapters/ от маркированных блоков.

    Обходит: base_dir/*/chapters/*.md
    Сохраняет: base_dir/*/extracted/clear_chapters/*.md

    Args:
        base_dir:          Корневая папка с книгами.
        book_names:        Список книг для обработки. None = все.
        recreate:          Если True — удалить clear_chapters/ перед запуском.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict: total, files_processed, tables_removed, figures_removed,
              errors, elapsed_sec, cancelled.
    """
    setup_logging(log_level)

    logger.info("=== Запуск очистки по маркерам ===")
    logger.info("Базовая папка: %s", base_dir)
    if book_names:
        logger.info("Выбранные книги: %s", ", ".join(book_names))

    base = Path(base_dir)
    files = _find_chapter_files(base, book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет .md файлов для обработки в chapters/")
        return {
            "total": 0, "files_processed": 0,
            "tables_removed": 0, "figures_removed": 0,
            "errors": 0, "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d файлов глав", total)

    # Пересоздание выходных папок
    if recreate:
        seen_books: set[str] = set()
        for book_name, _ in files:
            if book_name in seen_books:
                continue
            seen_books.add(book_name)
            cc_dir = base / book_name / "extracted" / CLEAR_CHAPTERS_SUBDIR
            if cc_dir.exists():
                shutil.rmtree(cc_dir)
                logger.info("Удалено: %s", cc_dir)

    t0 = time.perf_counter()
    files_processed = 0
    tables_removed = 0
    figures_removed = 0
    errors = 0
    cancelled = False
    created_dirs: set[str] = set()

    for i, (book_name, md_path) in enumerate(files):
        if cancel_event and cancel_event.is_set():
            logger.info("Очистка отменена на файле %s", md_path.name)
            cancelled = True
            break

        relative_name = f"{book_name}/chapters/{md_path.name}"

        # Возобновление: пропуск если файл уже в clear_chapters/
        output_dir = base / book_name / "extracted" / CLEAR_CHAPTERS_SUBDIR
        if not recreate and (output_dir / md_path.name).exists():
            logger.debug("Пропуск %s: уже обработан", relative_name)
            continue

        logger.info("[%d/%d] %s", i + 1, total, relative_name)

        # Создать выходную папку
        if book_name not in created_dirs:
            output_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(book_name)

        result = process_file(md_path, output_dir)

        if result["error"]:
            errors += 1
            logger.error("❌ Ошибка %s: %s", result["source_file"], result["error"])
        else:
            files_processed += 1
            tables_removed += result["tables_removed"]
            figures_removed += result["figures_removed"]
            logger.info(
                "✅ %s: %d таблиц, %d рисунков удалено",
                relative_name,
                result["tables_removed"],
                result["figures_removed"],
            )

        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "files_processed": files_processed,
                "tables_removed": tables_removed,
                "figures_removed": figures_removed,
                "errors": errors,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Очистка завершена: %d файлов, %d таблиц, %d рисунков удалено, "
        "%d ошибок%s, %.2fс ===",
        files_processed, tables_removed, figures_removed, errors,
        ", ОТМЕНЕНО" if cancelled else "", elapsed,
    )

    return {
        "total": total,
        "files_processed": files_processed,
        "tables_removed": tables_removed,
        "figures_removed": figures_removed,
        "errors": errors,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }