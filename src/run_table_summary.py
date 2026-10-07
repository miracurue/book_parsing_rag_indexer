"""Оркестратор генерации структурных сводок таблиц.

Обходит: base_dir/*/extracted/extracted_tables/*.md
Сохраняет: base_dir/*/extracted/table_summaries/*.md

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
from src.table_summary import generate_table_summary

logger = logging.getLogger(__name__)

SUMMARIES_SUBDIR = "table_summaries"
SOURCE_SUBDIR = "extracted_tables"


def _find_table_files(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Найти все .md в подпапках */extracted/extracted_tables/."""
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        if book_names and book_dir.name not in book_names:
            continue
        et_dir = book_dir / "extracted" / SOURCE_SUBDIR
        if not et_dir.is_dir():
            continue
        for md_file in sorted(et_dir.glob("*.md")):
            if md_file.is_file():
                results.append((book_dir.name, md_file))

    return results


def find_books_with_extracted_tables(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти книги с папкой extracted/extracted_tables/.

    Returns:
        Список (book_name, extracted_tables_dir).
    """
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        et_dir = book_dir / "extracted" / SOURCE_SUBDIR
        if et_dir.is_dir():
            results.append((book_dir.name, et_dir))

    return results


def count_all_summaries(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчитать файлы таблиц (без генерации сводок)."""
    files = _find_table_files(base_dir, book_names)

    # Считаем уже существующие сводки
    existing = 0
    for book_name, md_path in files:
        summary_dir = base_dir / book_name / "extracted" / SUMMARIES_SUBDIR
        summary_file = summary_dir / md_path.name
        if summary_file.exists():
            existing += 1

    return {
        "total_files": len(files),
        "existing_summaries": existing,
        "pending": len(files) - existing,
    }


def run_table_summary(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    recreate: bool = False,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить генерацию сводок таблиц.

    Для каждого файла в extracted_tables/ создаётся файл с тем же именем
    в table_summaries/ содержащий компактную сводку для векторизации.

    Args:
        base_dir:          Корневая папка с книгами.
        book_names:        Список книг для обработки. None = все.
        recreate:          Если True — удалить table_summaries/.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict: total, generated, skipped, errors, elapsed_sec, cancelled.
    """
    setup_logging(log_level)

    logger.info("=== Запуск генерации сводок таблиц ===")
    logger.info("Базовая папка: %s", base_dir)
    if book_names:
        logger.info("Выбранные книги: %s", ", ".join(book_names))

    base = Path(base_dir)
    files = _find_table_files(base, book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет файлов таблиц в extracted_tables/")
        return {
            "total": 0, "generated": 0, "skipped": 0,
            "errors": 0, "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d файлов таблиц", total)

    # Пересоздание
    if recreate:
        seen_books: set[str] = set()
        for book_name, _ in files:
            if book_name in seen_books:
                continue
            seen_books.add(book_name)
            s_dir = base / book_name / "extracted" / SUMMARIES_SUBDIR
            if s_dir.exists():
                shutil.rmtree(s_dir)
                logger.info("Удалено: %s", s_dir)

    t0 = time.perf_counter()
    generated = 0
    skipped = 0
    errors = 0
    cancelled = False
    created_dirs: set[str] = set()

    for i, (book_name, md_path) in enumerate(files):
        if cancel_event and cancel_event.is_set():
            logger.info("Генерация отменена на файле %s", md_path.name)
            cancelled = True
            break

        relative_name = f"{book_name}/extracted/{SOURCE_SUBDIR}/{md_path.name}"
        logger.info("[%d/%d] %s", i + 1, total, relative_name)

        summary_dir = base / book_name / "extracted" / SUMMARIES_SUBDIR
        summary_file = summary_dir / md_path.name

        # Возобновление: пропуск если сводка уже существует
        if not recreate and summary_file.exists():
            skipped += 1
            logger.debug("Пропуск %s: сводка уже существует", relative_name)
            if progress_callback:
                progress_callback(i + 1, total, relative_name, {
                    "generated": generated, "skipped": skipped, "errors": errors,
                })
            continue

        # Создать выходную папку
        if book_name not in created_dirs:
            summary_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(book_name)

        # Генерация сводки
        try:
            content = md_path.read_text(encoding="utf-8")
            summary_text = generate_table_summary(content)

            if not summary_text.strip():
                skipped += 1
                logger.warning("⚠️ %s: пустая сводка", relative_name)
            else:
                summary_file.write_text(summary_text, encoding="utf-8")
                generated += 1
                logger.info("✅ %s (%d символов)", relative_name, len(summary_text))

        except Exception as e:
            errors += 1
            logger.error("❌ Ошибка %s: %s", relative_name, e)

        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "generated": generated, "skipped": skipped, "errors": errors,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Генерация сводок завершена: %d создано, %d пропущено, "
        "%d ошибок%s, %.2fс ===",
        generated, skipped, errors,
        ", ОТМЕНЕНО" if cancelled else "", elapsed,
    )

    return {
        "total": total,
        "generated": generated,
        "skipped": skipped,
        "errors": errors,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }