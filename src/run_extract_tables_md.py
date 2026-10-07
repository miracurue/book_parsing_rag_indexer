"""Оркестратор извлечения таблиц из Markdown-файлов.

Два режима:
1. Из parsed/*.md — классический режим (process_file)
2. Из chapters/*.md — новый режим: заголовки + имена по тегам + clear_chapters/

Выходные данные сохраняются в: книга/extracted/extracted_tables/

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
from src.extract_tables_md import process_file, process_chapter_file, count_tables_by_markers, count_tables_in_text

logger = logging.getLogger(__name__)

MD_EXTS = (".md",)
OUTPUT_SUBDIR = "extracted_tables"
CLEAR_CHAPTERS_SUBDIR = "clear_chapters"


# ══════════════════════════════════════════════════════════════════════
# Поиск файлов (parsed/ — старый режим)
# ══════════════════════════════════════════════════════════════════════

def _find_md_files(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Найти все .md в подпапках */parsed/."""
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        if book_names and book_dir.name not in book_names:
            continue
        parsed_dir = book_dir / "parsed"
        if not parsed_dir.is_dir():
            continue
        for md_file in sorted(parsed_dir.glob("*")):
            if md_file.is_file() and md_file.suffix.lower() in MD_EXTS:
                if md_file.parent.name != "parsed":
                    continue
                results.append((book_dir.name, md_file))

    return results


# ══════════════════════════════════════════════════════════════════════
# Поиск файлов (chapters/ — новый режим)
# ══════════════════════════════════════════════════════════════════════

def _find_chapter_files(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Найти все .md в подпапках */chapters/.

    Returns:
        Список кортежей: (book_name, chapter_path)
    """
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
            logger.debug("Пропуск %s: нет chapters/", book_dir.name)
            continue
        for md_file in sorted(chapters_dir.glob("*.md")):
            if md_file.is_file():
                results.append((book_dir.name, md_file))

    return results


def find_books_with_chapters(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти книги с папкой chapters/.

    Returns:
        Список (book_name, chapters_dir).
    """
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


# ══════════════════════════════════════════════════════════════════════
# Подсчёт
# ══════════════════════════════════════════════════════════════════════

def count_all_tables(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчитать таблицы во всех файлах parsed/ (без сохранения)."""
    files_with = 0
    total_tables = 0
    html_tables = 0
    md_tables = 0
    total_files = 0

    for _, md_path in _find_md_files(base_dir, book_names):
        total_files += 1
        try:
            content = md_path.read_text(encoding="utf-8")
            counts = count_tables_in_text(content)
            if counts["total"] > 0:
                files_with += 1
                total_tables += counts["total"]
                html_tables += counts["html"]
                md_tables += counts["markdown"]
        except Exception:
            pass

    return {
        "files_with_tables": files_with,
        "total_tables": total_tables,
        "html_tables": html_tables,
        "md_tables": md_tables,
        "total_files": total_files,
    }


def count_all_chapter_tables(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчитать таблицы во всех файлах chapters/ по маркерам [TABLE_START]."""
    files_with = 0
    total_tables = 0
    total_files = 0

    for _, md_path in _find_chapter_files(base_dir, book_names):
        total_files += 1
        try:
            content = md_path.read_text(encoding="utf-8")
            counts = count_tables_by_markers(content)
            if counts["markers"] > 0:
                files_with += 1
                total_tables += counts["markers"]
        except Exception:
            pass

    return {
        "files_with_tables": files_with,
        "total_tables": total_tables,
        "total_files": total_files,
    }


def find_books_with_parsed(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти книги с папкой parsed/."""
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        parsed_dir = book_dir / "parsed"
        if parsed_dir.is_dir():
            results.append((book_dir.name, parsed_dir))

    return results


# ══════════════════════════════════════════════════════════════════════
# Публичный интерфейс: старый режим (parsed/)
# ══════════════════════════════════════════════════════════════════════

def run_extract_tables(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    recreate: bool = False,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить извлечение таблиц из parsed/*.md (старый режим)."""
    setup_logging(log_level)

    logger.info("=== Запуск извлечения таблиц (parsed/) ===")
    logger.info("Базовая папка: %s", base_dir)
    if book_names:
        logger.info("Выбранные книги: %s", ", ".join(book_names))

    base = Path(base_dir)
    files = _find_md_files(base, book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет .md файлов для обработки в %s", base_dir)
        return {
            "total": 0, "files_with_tables": 0, "tables_extracted": 0,
            "errors": 0, "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d .md файлов", total)

    if recreate:
        for book_name, _ in files:
            output_dir = base / book_name / "extracted" / OUTPUT_SUBDIR
            if output_dir.exists():
                shutil.rmtree(output_dir)
                logger.info("Удалено: %s", output_dir)

    t0 = time.perf_counter()
    files_with_tables = 0
    tables_extracted = 0
    errors = 0
    cancelled = False
    no_caption_pages: list[str] = []
    created_dirs: set[str] = set()

    for i, (book_name, md_path) in enumerate(files):
        if cancel_event and cancel_event.is_set():
            logger.info("Извлечение отменено на файле %s", md_path.name)
            cancelled = True
            break

        relative_name = f"{book_name}/parsed/{md_path.name}"
        logger.info("[%d/%d] %s", i + 1, total, relative_name)

        if book_name not in created_dirs:
            output_dir = base / book_name / "extracted" / OUTPUT_SUBDIR
            output_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(book_name)

        output_dir = base / book_name / "extracted" / OUTPUT_SUBDIR
        results = process_file(md_path, output_dir)

        if not results:
            pass
        else:
            has_error = any(r["error"] for r in results)
            if has_error:
                errors += 1
                for r in results:
                    if r["error"]:
                        logger.error("❌ Ошибка %s: %s", r["output_file"], r["error"])
            else:
                files_with_tables += 1
                tables_extracted += len(results)
                for r in results:
                    if not r.get("caption"):
                        no_caption_pages.append(r["output_file"])
                logger.info("✅ %s: %d таблиц", relative_name, len(results))

        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "files_with_tables": files_with_tables,
                "tables_extracted": tables_extracted,
                "errors": errors,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Извлечение завершено: %d файлов с таблицами, "
        "%d таблиц извлечено, %d ошибок%s, %.2fс ===",
        files_with_tables, tables_extracted, errors,
        ", ОТМЕНЕНО" if cancelled else "", elapsed,
    )

    return {
        "total": total,
        "files_with_tables": files_with_tables,
        "tables_extracted": tables_extracted,
        "errors": errors,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
        "no_caption_pages": no_caption_pages,
    }


# ══════════════════════════════════════════════════════════════════════
# Публичный интерфейс: новый режим (chapters/ → extracted_tables/ + clear_chapters/)
# ══════════════════════════════════════════════════════════════════════

def run_extract_tables_from_chapters(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    recreate: bool = False,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить извлечение таблиц из chapters/*.md по маркерам.

    Обходит: base_dir/*/chapters/*.md
    Сохраняет таблицы: base_dir/*/extracted/extracted_tables/*.md
    Исходные файлы chapters/ НЕ изменяются.
    Очистка (clear_chapters/) делается отдельным скриптом clean_tagged.

    Args:
        base_dir:          Корневая папка с книгами.
        book_names:        Список книг для обработки. None = все.
        recreate:          Если True — удалить extracted_tables/.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict: total, files_with_tables, tables_extracted, errors,
              elapsed_sec, cancelled, no_caption_pages, no_markers.
    """
    setup_logging(log_level)

    logger.info("=== Запуск извлечения таблиц из глав (по маркерам) ===")
    logger.info("Базовая папка: %s", base_dir)
    if book_names:
        logger.info("Выбранные книги: %s", ", ".join(book_names))

    base = Path(base_dir)
    files = _find_chapter_files(base, book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет .md файлов для обработки в chapters/")
        return {
            "total": 0, "files_with_tables": 0, "tables_extracted": 0,
            "errors": 0, "elapsed_sec": 0.0, "cancelled": False,
            "no_markers": 0, "no_caption_pages": [],
        }

    logger.info("К обработке: %d файлов глав", total)

    # Пересоздание выходной папки
    if recreate:
        seen_books: set[str] = set()
        for book_name, _ in files:
            if book_name in seen_books:
                continue
            seen_books.add(book_name)
            et_dir = base / book_name / "extracted" / OUTPUT_SUBDIR
            if et_dir.exists():
                shutil.rmtree(et_dir)
                logger.info("Удалено: %s", et_dir)

    t0 = time.perf_counter()
    files_with_tables = 0
    tables_extracted = 0
    errors = 0
    no_markers_count = 0
    cancelled = False
    no_caption_pages: list[str] = []
    created_dirs: set[str] = set()

    for i, (book_name, md_path) in enumerate(files):
        if cancel_event and cancel_event.is_set():
            logger.info("Извлечение отменено на файле %s", md_path.name)
            cancelled = True
            break

        relative_name = f"{book_name}/chapters/{md_path.name}"
        logger.info("[%d/%d] %s", i + 1, total, relative_name)

        # Возобновление: пропуск если уже есть извлечённые файлы для этой главы
        output_dir = base / book_name / "extracted" / OUTPUT_SUBDIR
        if not recreate and output_dir.exists():
            # Проверяем, есть ли файлы с таким source (по номеру страницы)
            import re as _re
            page_tag = _re.search(r"<(\d+)>", md_path.read_text(encoding="utf-8")[:500])
            if page_tag:
                existing = list(output_dir.glob(f"{page_tag.group(1)}*.md"))
                if existing:
                    logger.debug("Пропуск %s: уже извлечён", relative_name)
                    if progress_callback:
                        progress_callback(i + 1, total, relative_name, {
                            "files_with_tables": files_with_tables,
                            "tables_extracted": tables_extracted,
                            "errors": errors,
                        })
                    continue

        # Создать выходную папку
        if book_name not in created_dirs:
            output_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(book_name)

        output_dir = base / book_name / "extracted" / OUTPUT_SUBDIR

        results = process_chapter_file(md_path, output_dir)

        # Проверяем no_markers
        if results and len(results) == 1 and results[0].get("error") == "no_markers":
            no_markers_count += 1
            logger.warning("⚠️ %s: нет TABLE-маркеров — сначала нормализуйте таблицы", relative_name)
        elif not results:
            pass  # Нет таблиц
        else:
            has_error = any(r["error"] for r in results)
            if has_error:
                errors += 1
                for r in results:
                    if r["error"]:
                        logger.error("❌ Ошибка %s: %s", r["output_file"], r["error"])
            else:
                files_with_tables += 1
                tables_extracted += len(results)
                for r in results:
                    if not r.get("caption"):
                        no_caption_pages.append(r["output_file"])
                logger.info("✅ %s: %d таблиц", relative_name, len(results))

        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "files_with_tables": files_with_tables,
                "tables_extracted": tables_extracted,
                "errors": errors,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Извлечение из глав завершено: %d файлов с таблицами, "
        "%d таблиц извлечено, %d без маркеров, %d ошибок%s, %.2fс ===",
        files_with_tables, tables_extracted, no_markers_count, errors,
        ", ОТМЕНЕНО" if cancelled else "", elapsed,
    )

    return {
        "total": total,
        "files_with_tables": files_with_tables,
        "tables_extracted": tables_extracted,
        "errors": errors,
        "no_markers": no_markers_count,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
        "no_caption_pages": no_caption_pages,
    }
