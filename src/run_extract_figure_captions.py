"""Оркестратор извлечения описаний рисунков из Markdown-файлов парсинга.

Обходит структуру: base_dir/название_книги/parsed/*.md
Для каждого файла извлекает описания рисунков в parsed/extracted_figures/.

Поддерживает:
- Выбор конкретных книг (book_names)
- Пересоздание (recreate=True — удаляет extracted_figures/)
- Настраиваемый regex-паттерн
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
from src.extract_figure_captions import (
    process_file,
    process_chapter_file_figures,
    count_figure_captions_in_text,
    check_figure_sequence,
    DEFAULT_PATTERN,
)

logger = logging.getLogger(__name__)

MD_EXTS = (".md",)
OUTPUT_SUBDIR = "extracted_figures"
CLEANED_SUBDIR = "without_figures"
CLEAR_CHAPTERS_SUBDIR = "clear_chapters"


def _find_md_files(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Найти все .md в подпапках */parsed/ (без подпапок).

    Returns:
        Список кортежей: (book_name, md_path)
    """
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
            logger.debug("Пропуск %s: нет parsed/", book_dir.name)
            continue

        for md_file in sorted(parsed_dir.glob("*")):
            if md_file.is_file() and md_file.suffix.lower() in MD_EXTS:
                # Только файлы напрямую в parsed/, не в подпапках
                if md_file.parent.name != "parsed":
                    continue
                results.append((book_dir.name, md_file))

    return results


def find_books_with_parsed(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти книги с папкой parsed/.

    Returns:
        Список (book_name, parsed_dir).
    """
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


def count_all_figure_captions(
    base_dir: Path,
    book_names: list[str] | None = None,
    pattern: str = DEFAULT_PATTERN,
) -> dict:
    """Подсчитать описания рисунков во всех файлах (без сохранения).

    Returns:
        {
            "files_with_captions": int,
            "total_captions": int,
            "total_files": int,
        }
    """
    files_with = 0
    total_captions = 0
    total_files = 0

    for _, md_path in _find_md_files(base_dir, book_names):
        total_files += 1
        try:
            content = md_path.read_text(encoding="utf-8")
            count = count_figure_captions_in_text(content, pattern)
            if count > 0:
                files_with += 1
                total_captions += count
        except Exception:
            pass

    return {
        "files_with_captions": files_with,
        "total_captions": total_captions,
        "total_files": total_files,
    }


def check_all_figure_sequences(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Проверить полноту нумерации рисунков во всех книгах.

    Args:
        base_dir: Корневая папка с книгами.
        book_names: Список книг. None = все.

    Returns:
        {
            "books": {
                "Название": {
                    "found": [1, 2, ...],
                    "missing": [3, 4, ...],
                    "total_found": int,
                    "no_number_files": [str],
                },
            },
        }
    """
    books_result: dict[str, dict] = {}

    books = find_books_with_parsed(base_dir)
    for book_name, parsed_dir in books:
        if book_names and book_name not in book_names:
            continue
        figures_dir = parsed_dir / OUTPUT_SUBDIR
        result = check_figure_sequence(figures_dir)
        if result["total_found"] > 0 or result["no_number_files"]:
            books_result[book_name] = result

    return {"books": books_result}


# ══════════════════════════════════════════════════════════════════════
# Публичный интерфейс
# ══════════════════════════════════════════════════════════════════════

def run_extract_figure_captions(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    recreate: bool = False,
    pattern: str = DEFAULT_PATTERN,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить извлечение описаний рисунков из Markdown-файлов.

    Обходит: base_dir/*/parsed/*.md (или только выбранные книги).
    Сохраняет описания в: base_dir/*/parsed/extracted_figures/*.md
    Сохраняет очищенный текст (без описаний) в: base_dir/*/parsed/without_figures/*.md
    Оригинальные файлы в parsed/ остаются без изменений.

    Args:
        base_dir:          Корневая папка с книгами.
        book_names:        Список книг для обработки. None = все.
        recreate:          Если True — удалить extracted_figures/ перед запуском.
        pattern:           Regex-паттерн для строки-описания.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict: total, files_with_captions, captions_extracted, errors,
              elapsed_sec, cancelled.
    """
    setup_logging(log_level)

    logger.info("=== Запуск извлечения описаний рисунков ===")
    logger.info("Базовая папка: %s", base_dir)
    logger.info("Паттерн: %s", pattern)
    if book_names:
        logger.info("Выбранные книги: %s", ", ".join(book_names))

    base = Path(base_dir)
    files = _find_md_files(base, book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет .md файлов для обработки в %s", base_dir)
        return {
            "total": 0, "files_with_captions": 0, "captions_extracted": 0,
            "errors": 0, "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d .md файлов", total)

    # Пересоздание выходных папок
    if recreate:
        seen_books: set[str] = set()
        for book_name, _ in files:
            if book_name in seen_books:
                continue
            seen_books.add(book_name)
            for subdir in (OUTPUT_SUBDIR, CLEANED_SUBDIR):
                target = base / book_name / "extracted" / subdir
                if target.exists():
                    shutil.rmtree(target)
                    logger.info("Удалено: %s", target)

    t0 = time.perf_counter()
    files_with_captions = 0
    captions_extracted = 0
    errors = 0
    skipped = 0
    cancelled = False

    # Множество уже созданных выходных папок
    created_dirs: set[str] = set()

    for i, (book_name, md_path) in enumerate(files):
        # Проверка отмены
        if cancel_event and cancel_event.is_set():
            logger.info("Извлечение отменено на файле %s", md_path.name)
            cancelled = True
            break

        relative_name = f"{book_name}/parsed/{md_path.name}"

        # Возобновление: пропуск файлов, уже обработанных ранее
        cleaned_dir = base / book_name / "extracted" / CLEANED_SUBDIR
        if not recreate and (cleaned_dir / md_path.name).exists():
            skipped += 1
            continue

        logger.info("[%d/%d] %s", i + 1, total, relative_name)

        # Создать выходные папки при необходимости
        if book_name not in created_dirs:
            output_dir = base / book_name / "extracted" / OUTPUT_SUBDIR
            output_dir.mkdir(parents=True, exist_ok=True)
            cleaned_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(book_name)

        output_dir = base / book_name / "extracted" / OUTPUT_SUBDIR

        results = process_file(md_path, output_dir, pattern, cleaned_output_dir=cleaned_dir)

        if not results:
            # Нет описаний в файле — не ошибка
            pass
        else:
            has_error = any(r["error"] for r in results)
            if has_error:
                errors += 1
                for r in results:
                    if r["error"]:
                        logger.error("❌ Ошибка %s: %s", r["output_file"], r["error"])
            else:
                files_with_captions += 1
                captions_extracted += len(results)
                logger.info(
                    "✅ %s: %d описаний рисунков",
                    relative_name,
                    len(results),
                )

        # Обратный вызов прогресса
        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "files_with_captions": files_with_captions,
                "captions_extracted": captions_extracted,
                "errors": errors,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Извлечение завершено: %d файлов с описаниями, "
        "%d описаний извлечено, %d ошибок%s, %.2fс ===",
        files_with_captions, captions_extracted, errors,
        ", ОТМЕНЕНО" if cancelled else "",
        elapsed,
    )

    return {
        "total": total,
        "files_with_captions": files_with_captions,
        "captions_extracted": captions_extracted,
        "errors": errors,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }


# ══════════════════════════════════════════════════════════════════════
# Публичный интерфейс: режим по маркерам (chapters/ → extracted_figures/)
# ══════════════════════════════════════════════════════════════════════

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


def count_all_chapter_figures(
    base_dir: Path | str,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчёт FIGURE-маркеров в chapters/ (для UI-превью)."""
    base_dir = Path(base_dir)
    _books = find_books_with_chapters(base_dir)
    if book_names:
        _books = [(n, p) for n, p in _books if n in book_names]

    total_figures = 0
    files_with_figures = 0
    total_files = 0

    for _name, _path in _books:
        for md_file in sorted(_path.glob("*.md")):
            total_files += 1
            text = md_file.read_text(encoding="utf-8")
            count = text.count("[FIGURE_START]")
            if count > 0:
                total_figures += count
                files_with_figures += 1

    return {
        "total_figures": total_figures,
        "files_with_figures": files_with_figures,
        "total_files": total_files,
        "total_captions": total_figures,  # alias для совместимости с UI
        "files_with_captions": files_with_figures,  # alias для совместимости с UI
    }


def run_extract_figures_from_chapters(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    recreate: bool = False,
    pattern: str = DEFAULT_PATTERN,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить извлечение описаний рисунков из chapters/ по маркерам.

    Обходит: base_dir/*/chapters/*.md
    Сохраняет описания: base_dir/*/extracted/extracted_figures/*.md
    Исходные файлы chapters/ НЕ изменяются.
    Очистка (clear_chapters/) делается отдельным скриптом clean_tagged.

    Args:
        base_dir:          Корневая папка с книгами.
        book_names:        Список книг. None = все.
        recreate:          Если True — удалить extracted_figures/.
        pattern:           Regex-паттерн.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict: total, files_with_captions, captions_extracted, errors,
              elapsed_sec, cancelled, no_markers.
    """
    setup_logging(log_level)

    logger.info("=== Запуск извлечения описаний рисунков из глав (по маркерам) ===")
    logger.info("Базовая папка: %s", base_dir)
    logger.info("Паттерн: %s", pattern)
    if book_names:
        logger.info("Выбранные книги: %s", ", ".join(book_names))

    base = Path(base_dir)
    files = _find_chapter_files(base, book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет .md файлов для обработки в chapters/")
        return {
            "total": 0, "files_with_captions": 0, "captions_extracted": 0,
            "errors": 0, "elapsed_sec": 0.0, "cancelled": False,
            "no_markers": 0,
        }

    logger.info("К обработке: %d файлов глав", total)

    # Пересоздание выходной папки
    if recreate:
        seen_books: set[str] = set()
        for book_name, _ in files:
            if book_name in seen_books:
                continue
            seen_books.add(book_name)
            ef_dir = base / book_name / "extracted" / OUTPUT_SUBDIR
            if ef_dir.exists():
                shutil.rmtree(ef_dir)
                logger.info("Удалено: %s", ef_dir)

    t0 = time.perf_counter()
    files_with_captions = 0
    captions_extracted = 0
    errors = 0
    no_markers_count = 0
    cancelled = False
    created_dirs: set[str] = set()

    for i, (book_name, md_path) in enumerate(files):
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break

        relative_name = f"{book_name}/chapters/{md_path.name}"

        logger.info("[%d/%d] %s", i + 1, total, relative_name)

        # Создать выходную папку
        if book_name not in created_dirs:
            output_dir = base / book_name / "extracted" / OUTPUT_SUBDIR
            output_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(book_name)

        output_dir = base / book_name / "extracted" / OUTPUT_SUBDIR

        results = process_chapter_file_figures(md_path, output_dir, pattern)

        # Проверяем no_markers
        if results and len(results) == 1 and results[0].get("error") == "no_markers":
            no_markers_count += 1
            logger.warning("⚠️ %s: нет FIGURE-маркеров — сначала нормализуйте рисунки", relative_name)
        elif not results:
            pass  # Нет описаний
        else:
            has_error = any(r["error"] for r in results)
            if has_error:
                errors += 1
                for r in results:
                    if r["error"]:
                        logger.error("❌ Ошибка %s: %s", r["output_file"], r["error"])
            else:
                files_with_captions += 1
                captions_extracted += len(results)
                logger.info("✅ %s: %d описаний рисунков", relative_name, len(results))

        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "files_with_captions": files_with_captions,
                "captions_extracted": captions_extracted,
                "errors": errors,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Извлечение рисунков из глав завершено: "
        "%d файлов, %d описаний, %d без маркеров, %d ошибок%s, %.2fс ===",
        files_with_captions, captions_extracted, no_markers_count, errors,
        ", ОТМЕНЕНО" if cancelled else "", elapsed,
    )

    return {
        "total": total,
        "files_with_captions": files_with_captions,
        "captions_extracted": captions_extracted,
        "errors": errors,
        "no_markers": no_markers_count,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }
