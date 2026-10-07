"""LLM-сводки для oversized таблиц — чистые функции.

Для таблиц, превышающих порог chunk_size, генерируется текстовое описание
через LLM. Результат сохраняется в отдельный файл в extracted/table_summaries/.

Используется оркестратором run_llm_table_summary.py.
"""

from __future__ import annotations

import re
from pathlib import Path


def extract_table_body(text: str) -> str:
    """Извлечь тело таблицы без заголовков # и caption.

    Поддерживает HTML-таблицы (<table>...</table>) и Markdown-таблицы (|...|).

    Returns:
        Текст таблицы (HTML или Markdown). Пустая строка, если таблица не найдена.
    """
    # HTML-таблица
    html_match = re.search(
        r"(<table>.*?</table>)", text, re.DOTALL | re.IGNORECASE,
    )
    if html_match:
        return html_match.group(1).strip()

    # Markdown-таблица: строки, начинающиеся с |
    lines = text.split("\n")
    md_lines: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2:
            in_table = True
            md_lines.append(stripped)
        elif in_table:
            # Таблица закончилась
            break

    if md_lines:
        return "\n".join(md_lines)

    return ""


