"""Извлечение идентификаторов таблиц и рисунков из текста — для связывания чанков.

Regex-поиск упоминаний таблиц («табл. 3.1», «в таблице 5») и
извлечение table_id из caption («Таблица 5.1 Характеристики...»).

Regex-поиск упоминаний рисунков («рис. 3.1», «на рисунке 5») и
извлечение figure_id из caption («Рис. 3.1 Схема...»).

Используется при чанковании для обогащения metadata:
  - текстовые чанки → metadata.table_refs: ["3.1", "5"]
  - табличные чанки → metadata.table_id: "5.1"
  - текстовые чанки → metadata.figure_refs: ["3.1", "5"]
  - чанки рисунков  → metadata.figure_id: "3.1"
"""

from __future__ import annotations

import re

# ── Regex-шаблоны ──────────────────────────────────────────────────────

# Упоминания таблиц в тексте:
#   «табл. 3.1», «в таблице 5», «табл.12», «таблицу 2.3.1»,
#   «Таблица 5», «табл 7» (без точки)
# Группа 1 — идентификатор: «3.1», «5», «2.3.1»
_TABLE_REF_RE = re.compile(
    r"(?:табл\.?|таблиц[а-яё]+)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# Извлечение table_id из caption:
#   «Таблица 5.1 Характеристики...» → «5.1»
#   «Таблица 12» → «12»
#   «Таблица 3.2.1» → «3.2.1»
_TABLE_ID_FROM_CAPTION_RE = re.compile(
    r"Таблица\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)


def extract_table_refs(text: str) -> list[str]:
    """Извлечь все идентификаторы таблиц, упоминаемых в тексте.

    Args:
        text: Текст чанка или абзаца.

    Returns:
        Список уникальных идентификаторов в порядке появления.
        Пример: ["3.1", "5", "12"]

    Examples:
        >>> extract_table_refs("Данные приведены в табл. 3.1 и таблице 5")
        ['3.1', '5']
        >>> extract_table_refs("см. табл.12")
        ['12']
        >>> extract_table_refs("без таблиц")
        []
    """
    seen: set[str] = set()
    result: list[str] = []

    for m in _TABLE_REF_RE.finditer(text):
        table_id = m.group(1)
        if table_id not in seen:
            seen.add(table_id)
            result.append(table_id)

    return result


def extract_table_id_from_caption(text: str) -> str | None:
    """Извлечь идентификатор таблицы из caption.

    Args:
        text: Caption таблицы (напр. «Таблица 5.1 Характеристики...»).

    Returns:
        Идентификатор («5.1») или None.

    Examples:
        >>> extract_table_id_from_caption("Таблица 5.1 Характеристики дробилок")
        '5.1'
        >>> extract_table_id_from_caption("Таблица 12")
        '12'
        >>> extract_table_id_from_caption("Описание процесса")
        None
    """
    m = _TABLE_ID_FROM_CAPTION_RE.search(text)
    return m.group(1) if m else None


def extract_table_id_from_text(text: str) -> str | None:
    """Извлечь первый table_id из любого текста (body + caption).

    Ищет первое упоминание «Таблица N...» — используется для таблиц,
    у которых caption может быть в начале тела.

    Args:
        text: Полный текст табличного чанка.

    Returns:
        Идентификатор или None.
    """
    m = _TABLE_ID_FROM_CAPTION_RE.search(text)
    return m.group(1) if m else None


# ── Regex-шаблоны для рисунков ────────────────────────────────────────

# Упоминания рисунков в тексте:
#   «рис. 3.1», «на рисунке 5», «рис.12», «рисунок 2.3.1»,
#   «Рис. 5», «рис 7» (без точки)
#   Также: «рисунке», «рисунку», «рисунком», «рисунка»
# Группа 1 — идентификатор: «3.1», «5», «2.3.1»
_FIGURE_REF_RE = re.compile(
    r"(?:рис\.?|рисунок[а-яё]*)\s*(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# Извлечение figure_id из caption:
#   «Рис. 5.1 Схема дробилки» → «5.1»
#   «Рис. 12» → «12»
#   «Рис. 3.2.1» → «3.2.1»
_FIGURE_ID_FROM_CAPTION_RE = re.compile(
    r"Рис\.?\s+(\d+(?:\.\d+)*)",
    re.IGNORECASE,
)


def extract_figure_refs(text: str) -> list[str]:
    """Извлечь все идентификаторы рисунков, упоминаемых в тексте.

    Args:
        text: Текст чанка или абзаца.

    Returns:
        Список уникальных идентификаторов в порядке появления.
        Пример: ["3.1", "5", "12"]

    Examples:
        >>> extract_figure_refs("Как показано на рис. 3.1 и рисунке 5")
        ['3.1', '5']
        >>> extract_figure_refs("см. рис.12")
        ['12']
        >>> extract_figure_refs("без рисунков")
        []
    """
    seen: set[str] = set()
    result: list[str] = []

    for m in _FIGURE_REF_RE.finditer(text):
        fig_id = m.group(1)
        if fig_id not in seen:
            seen.add(fig_id)
            result.append(fig_id)

    return result


def extract_figure_id_from_text(text: str) -> str | None:
    """Извлечь идентификатор рисунка из текста (caption).

    Ищет первое упоминание «Рис. N...» — используется для чанков
    с описаниями рисунков.

    Args:
        text: Полный текст чанка рисунка.

    Returns:
        Идентификатор («3.1») или None.

    Examples:
        >>> extract_figure_id_from_text("Рис. 5.1 Схема дробилки")
        '5.1'
        >>> extract_figure_id_from_text("Рис. 12")
        '12'
        >>> extract_figure_id_from_text("Описание процесса")
        None
    """
    m = _FIGURE_ID_FROM_CAPTION_RE.search(text)
    return m.group(1) if m else None
