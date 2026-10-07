"""Хранилище библиографических данных книг.

Центральный файл data/books_metadata.json с ручными metadata:
  {book_folder_name: {author, title, year}}

При отсутствии ручных данных — fallback к parse_book_name() из имени папки.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

METADATA_FILE = Path("data/books_metadata.json")


def load_all() -> dict[str, dict]:
    """Загрузить все библиографические записи.

    Returns:
        {book_folder_name: {"author": str, "title": str, "year": str}}
    """
    if not METADATA_FILE.exists():
        return {}
    try:
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Ошибка чтения %s: %s", METADATA_FILE, e)
        return {}


def save_all(data: dict[str, dict]) -> None:
    """Сохранить все библиографические записи."""
    METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Сохранено %d записей в %s", len(data), METADATA_FILE)


def get_book_meta(book_name: str) -> dict:
    """Получить библиографию книги.

    Приоритет: ручные данные из books_metadata.json → fallback к parse_book_name().

    Returns:
        {"author": str, "title": str, "year": str}
        Пустые строки при отсутствии данных.
    """
    from src.chunking import parse_book_name

    all_data = load_all()

    if book_name in all_data:
        entry = all_data[book_name]
        return {
            "author": entry.get("author", ""),
            "title": entry.get("title", ""),
            "year": entry.get("year", ""),
        }

    # Fallback к парсингу имени папки
    author, title = parse_book_name(book_name)
    return {"author": author, "title": title, "year": ""}


def set_book_meta(book_name: str, author: str = "", title: str = "", year: str = "") -> None:
    """Установить библиографию для книги."""
    all_data = load_all()
    all_data[book_name] = {
        "author": author,
        "title": title,
        "year": year,
    }
    save_all(all_data)


def delete_book_meta(book_name: str) -> None:
    """Удалить библиографию книги."""
    all_data = load_all()
    all_data.pop(book_name, None)
    save_all(all_data)


def list_book_folders(base_dir: str = "data/books") -> list[str]:
    """Получить список всех папок книг.

    Returns:
        Отсортированный список имён папок.
    """
    base = Path(base_dir)
    if not base.exists():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )