"""Очистка файлов глав от промаркированных блоков таблиц и рисунков.

Удаляет из текста все блоки [TABLE_START]...[TABLE_END] и
[FIGURE_START]...[FIGURE_END] (включая сами маркеры).
Результат сохраняется в отдельную папку clear_chapters/.

Исходные файлы chapters/ НЕ изменяются.
"""

from __future__ import annotations

import re
from pathlib import Path

from .normalize_tables import MARKER_START as TABLE_START
from .normalize_tables import MARKER_END as TABLE_END

FIGURE_START = "[FIGURE_START]"
FIGURE_END = "[FIGURE_END]"

# Regex: захватывает блок от открывающего до закрывающего маркера (включая маркеры)
_TABLE_BLOCK_RE = re.compile(
    re.escape(TABLE_START) + r".*?" + re.escape(TABLE_END),
    re.DOTALL,
)
_FIGURE_BLOCK_RE = re.compile(
    re.escape(FIGURE_START) + r".*?" + re.escape(FIGURE_END),
    re.DOTALL,
)


def clean_tagged_blocks(text: str) -> str:
    """Удалить все TABLE и FIGURE блоки из текста.

    Удаляет блоки включая маркеры. Схлопывает 3+ переносов в 2.

    Args:
        text: Исходный текст с маркерами.

    Returns:
        Очищенный текст без блоков таблиц и рисунков.
    """
    # Удаляем TABLE-блоки
    cleaned = _TABLE_BLOCK_RE.sub("", text)
    # Удаляем FIGURE-блоки
    cleaned = _FIGURE_BLOCK_RE.sub("", cleaned)
    # Схлопываем 3+ \n → \n\n
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return cleaned


def process_file(
    filepath: Path,
    output_dir: Path,
) -> dict:
    """Обработать один файл: удалить маркированные блоки, сохранить в output_dir.

    Args:
        filepath:   Путь к файлу главы (chapters/*.md).
        output_dir: Папка для очищенных файлов (clear_chapters/).

    Returns:
        dict с результатом:
        - source_file, output_file, tables_removed, figures_removed, error
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return {
            "source_file": filepath.name,
            "output_file": None,
            "tables_removed": 0,
            "figures_removed": 0,
            "error": str(e),
        }

    # Подсчитываем блоки до удаления
    tables_removed = len(_TABLE_BLOCK_RE.findall(content))
    figures_removed = len(_FIGURE_BLOCK_RE.findall(content))

    # Очищаем
    cleaned = clean_tagged_blocks(content)

    # Сохраняем
    out_path = output_dir / filepath.name
    try:
        out_path.write_text(cleaned, encoding="utf-8")
    except Exception as e:
        return {
            "source_file": filepath.name,
            "output_file": filepath.name,
            "tables_removed": tables_removed,
            "figures_removed": figures_removed,
            "error": str(e),
        }

    return {
        "source_file": filepath.name,
        "output_file": filepath.name,
        "tables_removed": tables_removed,
        "figures_removed": figures_removed,
        "error": None,
    }


def count_tagged_blocks(text: str) -> dict:
    """Подсчитать количество маркированных блоков в тексте.

    Returns:
        {"tables": int, "figures": int}
    """
    return {
        "tables": len(_TABLE_BLOCK_RE.findall(text)),
        "figures": len(_FIGURE_BLOCK_RE.findall(text)),
    }