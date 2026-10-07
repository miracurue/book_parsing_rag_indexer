"""Расстановка номеров страниц в Markdown-абзацах.

Чистая функция — используется оркестратором run_page_numbering.py.

Логика:
1. Схлопываем 2+ пробелов → один пробел
2. Схлопываем 2+ подряд идущих \\n → один \\n
3. Схлопываем 4+ точек → три точки (...)
4. Разделяем по одному \\n на блоки
5. Каждый непустой блок, не являющийся заголовком (#), помечается тегом <N>
6. Объединяем обратно через \\n
"""

from __future__ import annotations

import re


def extract_page_number(filename: str) -> int | None:
    """Извлечь номер страницы из имени файла.

    Examples:
        '005.md' → 5
        '012.md' → 12
        'intro.md' → None
    """
    match = re.search(r"(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def add_page_numbers(content: str, filename: str) -> str:
    """Добавить тег номера страницы перед каждым абзацем.

    Args:
        content:   Текст Markdown-файла.
        filename:  Имя файла (для извлечения номера страницы).

    Returns:
        Обработанный текст с тегами <N> перед абзацами.
    """
    match = re.search(r"(\d+)", filename)
    if not match:
        return content

    tag = f"<{match.group(1)}> "

    # 1. Схлопываем 2+ пробелов → один пробел
    content = re.sub(r" {2,}", " ", content)

    # 2. Схлопываем 2+ подряд идущих \n → один \n
    content = re.sub(r"\n{2,}", "\n", content)

    # 3. Схлопываем 4+ точек → три точки (...)
    content = re.sub(r"\.{4,}", "...", content)

    # 4. Разделяем по одному \n
    blocks = content.split("\n")

    # 5. Обрабатываем каждый блок
    processed: list[str] = []
    for block in blocks:
        stripped = block.strip()
        # Пустые строки и заголовки (#) — без тега
        if not stripped or stripped.startswith("#"):
            processed.append(block)
        else:
            processed.append(tag + block)

    # 6. Объединяем
    return "\n".join(processed)