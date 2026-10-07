"""Подсчёт символов табличных файлов и готовых чанков — диагностика размеров.

Чистые функции — вызываются из UI pages/5_postprocessing.py.

Два режима проверки:
1. Сырые таблицы: .md файлы из extracted/extracted_tables/
2. Готовые чанки: JSON файлы из chunks/{text_chunks,table_chunks,figure_chunks}/
"""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Общие вспомогательные функции
# ══════════════════════════════════════════════════════════════════════

def _build_distribution(sizes: list[int]) -> dict[str, int]:
    """Построить распределение размеров по диапазонам."""
    dist = {
        "0–500": 0,
        "500–1000": 0,
        "1000–1200": 0,
        "1200–1500": 0,
        "1500–2000": 0,
        "2000+": 0,
    }
    for chars in sizes:
        if chars < 500:
            dist["0–500"] += 1
        elif chars < 1000:
            dist["500–1000"] += 1
        elif chars < 1200:
            dist["1000–1200"] += 1
        elif chars < 1500:
            dist["1200–1500"] += 1
        elif chars < 2000:
            dist["1500–2000"] += 1
        else:
            dist["2000+"] += 1
    return dist


def _compute_stats(sizes: list[int]) -> dict:
    """Вычислить min/max/median/mean."""
    if not sizes:
        return {}
    return {
        "min": min(sizes),
        "max": max(sizes),
        "median": int(statistics.median(sizes)),
        "mean": int(statistics.mean(sizes)),
    }


# ══════════════════════════════════════════════════════════════════════
# 1. Проверка сырых таблиц (extracted/extracted_tables/)
# ══════════════════════════════════════════════════════════════════════

def find_book_extracted_table_dirs(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти все папки книг с extracted/extracted_tables/ внутри.

    Returns:
        Список (book_name, extracted_tables_dir).
    """
    results: list[tuple[str, Path]] = []
    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        et_dir = book_dir / "extracted" / "extracted_tables"
        if et_dir.is_dir():
            md_files = list(et_dir.glob("*.md"))
            if md_files:
                results.append((book_dir.name, et_dir))

    return results


def check_raw_table_sizes(
    base_dir: Path,
    chunk_size: int = 1200,
    book_names: Optional[list[str]] = None,
) -> dict:
    """Подсчитать размеры файлов таблиц из extracted/extracted_tables/.

    Каждый .md файл — это готовая таблица (заголовки + caption + тело),
    которая целиком пойдёт в чанк. Подсчитываются ВСЕ символы включая
    пробелы, переносы строк, HTML-теги.

    Args:
        base_dir: Папка с книгами (data/books).
        chunk_size: Порог размера (символы). Файлы длиннее — «превышающие».
        book_names: Список книг для проверки. None = все найденные.

    Returns:
        {
            "total_chunks": int,
            "oversized_count": int,
            "stats": {"min", "max", "median", "mean"},
            "oversized": [{"book", "file", "chars", "over"}],
            "distribution": {"0–500": int, ...},
            "per_book": {"book_name": {"total": int, "oversized": int}},
        }
    """
    table_dirs = find_book_extracted_table_dirs(base_dir)

    # Фильтрация по выбранным книгам
    if book_names:
        table_dirs = [(bn, d) for bn, d in table_dirs if bn in book_names]

    all_sizes: list[int] = []
    oversized: list[dict] = []
    per_book: dict[str, dict] = {}

    for book_name, tables_dir in table_dirs:
        book_total = 0
        book_oversized = 0

        for md_file in sorted(tables_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError as e:
                logger.warning("Ошибка чтения %s: %s", md_file, e)
                continue

            chars = len(text)
            all_sizes.append(chars)
            book_total += 1

            if chars > chunk_size:
                oversized.append({
                    "book": book_name,
                    "file": md_file.stem,
                    "chars": chars,
                    "over": chars - chunk_size,
                })
                book_oversized += 1

        if book_total > 0:
            per_book[book_name] = {"total": book_total, "oversized": book_oversized}

    return {
        "total_chunks": len(all_sizes),
        "oversized_count": len(oversized),
        "stats": _compute_stats(all_sizes),
        "oversized": sorted(oversized, key=lambda x: x["chars"], reverse=True),
        "distribution": _build_distribution(all_sizes),
        "per_book": per_book,
    }


# Обратная совместимость — старое имя функции
check_table_chunk_sizes = check_raw_table_sizes


# ══════════════════════════════════════════════════════════════════════
# 2. Проверка готовых чанков (chunks/*.json)
# ══════════════════════════════════════════════════════════════════════

def find_book_chunk_dirs(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти все папки книг с chunks/ внутри.

    Returns:
        Список (book_name, book_dir) — book_dir содержит папку chunks/.
    """
    results: list[tuple[str, Path]] = []
    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        chunks_dir = book_dir / "chunks"
        if chunks_dir.is_dir():
            # Проверяем, что есть хотя бы один JSON
            json_files = list(chunks_dir.rglob("*.json"))
            if json_files:
                results.append((book_dir.name, book_dir))

    return results


def check_created_chunk_sizes(
    base_dir: Path,
    chunk_size: int = 1200,
    book_names: Optional[list[str]] = None,
) -> dict:
    """Подсчитать размеры текста в готовых чанках из chunks/*.json.

    Читает JSON файлы из chunks/{text_chunks,table_chunks,figure_chunks}/.
    Для каждого чанка измеряет len(chunk["text"]) — размер текста,
    который пойдёт в векторизацию.

    Args:
        base_dir: Папка с книгами (data/books).
        chunk_size: Порог размера (символы). Чанки длиннее — «превышающие».
        book_names: Список книг для проверки. None = все найденные.

    Returns:
        {
            "total_chunks": int,
            "by_type": {"text": int, "table": int, "figure": int},
            "oversized_count": int,
            "stats": {"min", "max", "median", "mean"},
            "oversized": [{"book", "file", "chunk_type", "chars", "over"}],
            "distribution": {"0–500": int, ...},
            "per_book": {"book_name": {"total": int, "text": int, "table": int, "figure": int, "oversized": int}},
        }
    """
    book_dirs = find_book_chunk_dirs(base_dir)

    # Фильтрация по выбранным книгам
    if book_names:
        _names = set(book_names)
        book_dirs = [(bn, d) for bn, d in book_dirs if bn in _names]

    all_sizes: list[int] = []
    oversized: list[dict] = []
    per_book: dict[str, dict] = {}
    by_type: dict[str, int] = {"text": 0, "table": 0, "figure": 0}

    chunk_subdirs = ["text_chunks", "table_chunks", "figure_chunks"]

    for book_name, book_dir in book_dirs:
        book_total = 0
        book_oversized = 0
        book_by_type = {"text": 0, "table": 0, "figure": 0}

        chunks_dir = book_dir / "chunks"

        for subdir_name in chunk_subdirs:
            sub_path = chunks_dir / subdir_name
            if not sub_path.is_dir():
                continue

            # Определяем тип чанка из имени подпапки
            if "table" in subdir_name:
                chunk_type = "table"
            elif "figure" in subdir_name:
                chunk_type = "figure"
            else:
                chunk_type = "text"

            for json_file in sorted(sub_path.glob("*.json")):
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning("Ошибка чтения %s: %s", json_file, e)
                    continue

                # JSON может быть списком чанков или одним чанком
                chunks_list = data if isinstance(data, list) else [data]

                for chunk in chunks_list:
                    text = chunk.get("text", "")
                    # Переопределяем тип из metadata, если есть
                    meta_type = chunk.get("metadata", {}).get("chunk_type", chunk_type)
                    chars = len(text)
                    all_sizes.append(chars)
                    book_total += 1
                    by_type[meta_type] = by_type.get(meta_type, 0) + 1
                    book_by_type[meta_type] = book_by_type.get(meta_type, 0) + 1

                    if chars > chunk_size:
                        oversized.append({
                            "book": book_name,
                            "file": json_file.stem,
                            "chunk_type": meta_type,
                            "chars": chars,
                            "over": chars - chunk_size,
                        })
                        book_oversized += 1

        if book_total > 0:
            per_book[book_name] = {
                "total": book_total,
                **book_by_type,
                "oversized": book_oversized,
            }

    return {
        "total_chunks": len(all_sizes),
        "by_type": by_type,
        "oversized_count": len(oversized),
        "stats": _compute_stats(all_sizes),
        "oversized": sorted(oversized, key=lambda x: x["chars"], reverse=True),
        "distribution": _build_distribution(all_sizes),
        "per_book": per_book,
    }