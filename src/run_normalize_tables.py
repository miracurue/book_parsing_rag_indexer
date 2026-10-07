"""Оркестратор нормализации границ таблиц.

Обходит chapters/*.md, добавляет маркеры [TABLE_START]/[TABLE_END]
вокруг каждой таблицы (включая caption и сноски).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from .normalize_tables import (
    MARKER_END,
    MARKER_START,
    count_markers,
    count_tables_in_text,
    has_markers,
    normalize_table_boundaries,
)

logger = logging.getLogger(__name__)


def find_books_with_chapters(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти книги с папкой chapters/.

    Returns:
        [(book_name, chapters_path), ...] — отсортировано по имени.
    """
    books = []
    if not base_dir.exists():
        return books

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        chapters_dir = book_dir / "chapters"
        if chapters_dir.exists() and any(chapters_dir.glob("*.md")):
            books.append((book_dir.name, chapters_dir))

    return books


def count_all_tables(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчитать таблицы во всех книгах.

    Returns:
        {
            "books": {book_name: {"html": N, "markdown": N, "total": N}},
            "total_html": N,
            "total_md": N,
            "total": N,
        }
    """
    books_data = {}
    total_html = 0
    total_md = 0

    books = find_books_with_chapters(base_dir)
    book_names_set = set(book_names) if book_names else None

    for bname, chap_dir in books:
        if book_names_set and bname not in book_names_set:
            continue

        book_counts = {"html": 0, "markdown": 0, "total": 0}
        for md_file in sorted(chap_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
                counts = count_tables_in_text(text)
                book_counts["html"] += counts["html"]
                book_counts["markdown"] += counts["markdown"]
                book_counts["total"] += counts["total"]
            except Exception:
                pass

        books_data[bname] = book_counts
        total_html += book_counts["html"]
        total_md += book_counts["markdown"]

    return {
        "books": books_data,
        "total_html": total_html,
        "total_md": total_md,
        "total": total_html + total_md,
    }


def count_all_markers(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчитать маркеры во всех книгах.

    Returns:
        {
            "books": {book_name: {"starts": N, "ends": N}},
            "files_with_markers": N,
            "files_without_markers": N,
        }
    """
    books_data = {}
    files_with = 0
    files_without = 0

    books = find_books_with_chapters(base_dir)
    book_names_set = set(book_names) if book_names else None

    for bname, chap_dir in books:
        if book_names_set and bname not in book_names_set:
            continue

        book_markers = {"starts": 0, "ends": 0}
        for md_file in sorted(chap_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
                mc = count_markers(text)
                book_markers["starts"] += mc["starts"]
                book_markers["ends"] += mc["ends"]
                if mc["starts"] > 0 or mc["ends"] > 0:
                    files_with += 1
                else:
                    files_without += 1
            except Exception:
                pass

        books_data[bname] = book_markers

    return {
        "books": books_data,
        "files_with_markers": files_with,
        "files_without_markers": files_without,
    }


def count_all_tables_detailed(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчитать таблицы по маркерам [TABLE_START]/[TABLE_END].

    Считает ТОЛЬКО размеченные таблицы (после нормализации).
    Если маркеров нет — вернёт нули.

    Returns:
        {
            "total_files": int,
            "total_tables": int,
            "files_with_markers": int,
        }
    """
    total_files = 0
    total_tables = 0
    files_with_markers = 0

    books = find_books_with_chapters(base_dir)
    book_names_set = set(book_names) if book_names else None

    for bname, chap_dir in books:
        if book_names_set and bname not in book_names_set:
            continue
        for md_file in sorted(chap_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
                mc = count_markers(text)
                if mc["starts"] > 0:
                    total_files += 1
                    total_tables += mc["starts"]
                    files_with_markers += 1
            except Exception:
                pass

    return {
        "total_files": total_files,
        "total_tables": total_tables,
        "files_with_markers": files_with_markers,
    }


def run_normalize_tables(
    base_dir: str | Path,
    book_names: list[str] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    **kwargs,
) -> dict:
    """Запустить нормализацию таблиц для всех выбранных книг.

    Args:
        base_dir:           Папка с книгами (data/books/).
        book_names:         Список имён книг.
        cancel_event:       Событие отмены.
        progress_callback:  Функция(current, total, current_file, stats).

    Returns:
        {
            "modified": int,
            "markers_placed": int,
            "errors": int,
            "elapsed_sec": float,
            "cancelled": bool,
        }
    """
    import time
    t0 = time.time()

    base_dir = Path(base_dir)
    books = find_books_with_chapters(base_dir)
    book_names_set = set(book_names) if book_names else None

    if book_names_set:
        books = [(n, p) for n, p in books if n in book_names_set]

    if not books:
        return {
            "modified": 0, "markers_placed": 0, "errors": 0,
            "elapsed_sec": 0, "cancelled": False,
        }

    # Собираем все файлы для обработки
    all_files: list[tuple[str, Path]] = []
    for bname, chap_dir in books:
        for md_file in sorted(chap_dir.glob("*.md")):
            all_files.append((bname, md_file))

    if not all_files:
        return {
            "modified": 0, "markers_placed": 0, "errors": 0,
            "elapsed_sec": 0, "cancelled": False,
        }

    total = len(all_files)
    modified = 0
    markers_placed = 0
    errors_count = 0
    cancelled = False

    for idx, (bname, filepath) in enumerate(all_files):
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break

        try:
            text = filepath.read_text(encoding="utf-8")
        except Exception as e:
            logger.error("%s/%s: %s", bname, filepath.name, e)
            errors_count += 1
            continue

        # Подсчёт таблиц в файле
        counts = count_tables_in_text(text)

        if counts["total"] == 0:
            if progress_callback:
                progress_callback(idx + 1, total, filepath.name, {
                    "modified": modified, "markers_placed": markers_placed,
                    "errors": errors_count,
                })
            continue

        # Нормализация (маркеры пересоздаются — старые удаляются внутри)
        try:
            result = normalize_table_boundaries(text)
            filepath.write_text(result, encoding="utf-8")

            mc = count_markers(result)
            markers_placed += mc["starts"]
            modified += 1
        except Exception as e:
            logger.error("%s/%s: %s", bname, filepath.name, e)
            errors_count += 1

        if progress_callback:
            progress_callback(idx + 1, total, filepath.name, {
                "modified": modified, "markers_placed": markers_placed,
                "errors": errors_count,
            })

    elapsed = round(time.time() - t0, 1)
    return {
        "modified": modified,
        "markers_placed": markers_placed,
        "errors": errors_count,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }
