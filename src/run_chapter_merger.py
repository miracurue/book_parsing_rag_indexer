"""Оркестратор соединения файлов по главам.

Обходит структуру: base_dir/название_книги/numbered/*.md
Для каждой книги:
  1. Читает все .md из numbered/
  2. Склеивает разорванные блоки через chapter_merger.merge_blocks_from_pages()
  3. Разделяет на главы через chapter_merger.split_into_chapters()
  4. Сохраняет в base_dir/название_книги/chapters/01_Название.md

Поддерживает:
- Пересоздание chapters/ при повторном запуске (старые файлы удаляются)
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

from src.chapter_merger import (
    merge_blocks_from_pages,
    split_into_chapters,
    find_deepest_heading_level,
    _split_compound_blocks,
)
from src.config import setup_logging

logger = logging.getLogger(__name__)

MD_EXTS = (".md",)


def _find_book_dirs(base_dir: Path) -> list[tuple[str, Path, Path]]:
    """Найти все папки книг с numbered/ внутри.

    Returns:
        Список кортежей: (book_name, numbered_dir, chapters_dir)
    """
    results: list[tuple[str, Path, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue

        numbered_dir = book_dir / "numbered"
        if not numbered_dir.is_dir():
            logger.debug("Пропуск %s: нет numbered/", book_dir.name)
            continue

        chapters_dir = book_dir / "chapters"
        results.append((book_dir.name, numbered_dir, chapters_dir))

    return results


def _count_md_files(directory: Path) -> int:
    """Посчитать .md файлы в директории (рекурсивно)."""
    return sum(1 for f in directory.rglob("*") if f.is_file() and f.suffix.lower() in MD_EXTS)


# ══════════════════════════════════════════════════════════════════════
# Публичный интерфейс
# ══════════════════════════════════════════════════════════════════════

def run_chapter_merger(
    base_dir: str = "data/books",
    split_level: int = 1,
    book_names: list[str] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить соединение файлов по главам для всех книг.

    Обходит: base_dir/*/numbered/*.md
    Сохраняет: base_dir/*/chapters/01_Название.md

    Args:
        base_dir:          Корневая папка с книгами.
        split_level:       Уровень заголовков для разделения (1=#, 2=##, ...).
        book_names:        Список имён книг для обработки. None — все книги.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict: total, success, errors, skipped, elapsed_sec, cancelled.
    """
    setup_logging(log_level)

    is_auto = split_level == 0
    logger.info("=== Запуск соединения файлов по главам ===")
    logger.info("Базовая папка: %s", base_dir)
    if is_auto:
        logger.info("Уровень разбиения: АВТО (определяется для каждой книги)")
    else:
        logger.info("Уровень разбиения: %d (%s)", split_level, "#" * split_level)

    base = Path(base_dir)
    book_dirs = _find_book_dirs(base)

    # Фильтрация по выбранным книгам
    if book_names is not None:
        _names_set = set(book_names)
        book_dirs = [(n, nd, cd) for n, nd, cd in book_dirs if n in _names_set]
        logger.info("Фильтрация по книгам: %s", book_names)

    total = len(book_dirs)

    if total == 0:
        logger.info("Нет папок книг с numbered/ в %s", base_dir)
        return {
            "total": 0, "success": 0, "errors": 0, "skipped": 0,
            "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d книг", total)

    t0 = time.perf_counter()
    success = 0
    errors = 0
    skipped = 0
    cancelled = False

    for i, (book_name, numbered_dir, chapters_dir) in enumerate(book_dirs):
        # Проверка отмены
        if cancel_event and cancel_event.is_set():
            logger.info("Соединение отменено на книге %s", book_name)
            cancelled = True
            break

        relative_name = f"{book_name}/numbered/"

        # Очищаем старую папку chapters/ (соединение — быстрая операция, пересоздаём с нуля)
        if chapters_dir.exists():
            shutil.rmtree(chapters_dir, ignore_errors=True)
            logger.info("🗑️ Удалена старая папка chapters/ для %s", book_name)

        logger.info("[%d/%d] %s", i + 1, total, book_name)

        try:
            # Читаем .md файлы из корня numbered/ (без подпапок вроде others/)
            pages: dict[str, str] = {}
            for md_file in sorted(numbered_dir.glob("*")):
                if md_file.is_file() and md_file.suffix.lower() in MD_EXTS:
                    content = md_file.read_text(encoding="utf-8")
                    # Ключ — относительный путь от numbered_dir
                    rel_key = md_file.relative_to(numbered_dir).as_posix()
                    pages[rel_key] = content

            if not pages:
                logger.warning("⚠️ Нет .md файлов в %s/numbered/", book_name)
                skipped += 1
                if progress_callback:
                    progress_callback(i + 1, total, relative_name, {
                        "success": success, "errors": errors, "skipped": skipped,
                    })
                continue

            logger.info("  Прочитано %d файлов из numbered/", len(pages))

            # Шаг 1+2: склейка блоков
            merged_blocks = merge_blocks_from_pages(pages)
            logger.info("  Склеено блоков: %d", len(merged_blocks))

            # Шаг 2.5: разделение составных блоков (несколько заголовков в одном)
            merged_blocks = _split_compound_blocks(merged_blocks)
            logger.info("  После разделения составных: %d блоков", len(merged_blocks))

            # Шаг 3: разделение на главы
            chapters = split_into_chapters(merged_blocks, split_level=split_level)
            logger.info("  Найдено глав: %d", len(chapters))

            # Шаг 4: сохранение
            chapters_dir.mkdir(parents=True, exist_ok=True)
            for ch_title, ch_blocks in chapters:
                ch_path = chapters_dir / f"{ch_title}.md"
                ch_path.write_text("\n\n".join(ch_blocks), encoding="utf-8")
                logger.debug("  Сохранена глава: %s (%d блоков)", ch_title, len(ch_blocks))

            logger.info(
                "✅ %s: %d глав сохранено в chapters/",
                book_name, len(chapters),
            )
            success += 1

        except Exception as e:
            errors += 1
            logger.exception("❌ Ошибка при обработке %s: %s", book_name, e)

        # Обратный вызов прогресса
        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "success": success, "errors": errors, "skipped": skipped,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Соединение завершено: %d успешно, %d ошибок, %d пропущено%s, %.2fс ===",
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