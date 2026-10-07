"""Оркестратор нормализации границ описаний рисунков.

Обходит chapters/*.md, добавляет маркеры [FIGURE_START]/[FIGURE_END]
вокруг каждого описания рисунка.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from .normalize_figures import (
    MARKER_END,
    MARKER_START,
    count_figures_in_text,
    count_markers,
    has_markers,
    normalize_figure_boundaries,
)
from .run_normalize_tables import find_books_with_chapters

logger = logging.getLogger(__name__)


def count_all_figures_detailed(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> dict:
    """Подсчитать рисунки и маркеры (детально для UI превью).

    Returns:
        {
            "total_files": int,          # файлов с рисунками
            "figure_tags": int,          # <figure> блоков
            "plain_captions": int,       # «Рис. N.» строк
            "total_figures": int,        # всего рисунков
            "files_with_markers": int,   # файлов с FIGURE-маркерами
        }
    """
    total_files = 0
    figure_tags = 0
    plain_captions = 0
    total_figures = 0
    files_with_markers = 0

    books = find_books_with_chapters(base_dir)
    book_names_set = set(book_names) if book_names else None

    for bname, chap_dir in books:
        if book_names_set and bname not in book_names_set:
            continue
        for md_file in sorted(chap_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
                counts = count_figures_in_text(text)
                if counts["total"] > 0:
                    total_files += 1
                    figure_tags += counts["figure_tags"]
                    plain_captions += counts["plain_captions"]
                    total_figures += counts["total"]
                if has_markers(text):
                    files_with_markers += 1
            except Exception:
                pass

    return {
        "total_files": total_files,
        "figure_tags": figure_tags,
        "plain_captions": plain_captions,
        "total_figures": total_figures,
        "files_with_markers": files_with_markers,
    }


def run_normalize_figures(
    base_dir: str | Path,
    book_names: list[str] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    **kwargs,
) -> dict:
    """Запустить нормализацию рисунков для всех выбранных книг.

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

        # Подсчёт рисунков в файле
        counts = count_figures_in_text(text)

        if counts["total"] == 0:
            if progress_callback:
                progress_callback(idx + 1, total, filepath.name, {
                    "modified": modified, "markers_placed": markers_placed,
                    "errors": errors_count,
                })
            continue

        # Нормализация (маркеры пересоздаются — старые удаляются внутри)
        try:
            result = normalize_figure_boundaries(text)
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