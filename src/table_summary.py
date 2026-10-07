"""Генерация структурных сводок таблиц для векторизации.

Чистые функции — используются оркестратором run_table_summary.py.

Подход «Структурная сводка + Полная таблица»:
- Из HTML/Markdown таблицы извлекается компактная сводка (~300-800 символов)
  для качественной векторизации (поиск)
- Полная таблица остаётся в исходном файле (для контекста LLM)

Структура сводки:
  1. Caption (название таблицы: «Таблица N. ...»)
  2. Заголовки столбцов
  3. Группы строк (<th colspan>)
  4. Ключевые значения (первый столбец данных — модели, минералы и т.д.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Максимальная длина списка ключевых значений (обрезается с «...»)
_MAX_KEY_VALUES = 30
# Максимальная длина одного значения (обрезается с «...»)
_MAX_VALUE_LEN = 80


@dataclass
class TableSummary:
    """Результат анализа таблицы."""

    headings: list[str] = field(default_factory=list)
    caption: str = ""
    columns: list[str] = field(default_factory=list)
    row_groups: list[str] = field(default_factory=list)
    key_values: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        """Сформировать текст сводки для векторизации."""
        parts: list[str] = []
        if self.headings:
            parts.extend(self.headings)
        if self.caption:
            parts.append(self.caption)
        if self.columns:
            parts.append("Столбцы: " + ", ".join(self.columns))
        if self.row_groups:
            parts.append("Группы: " + ", ".join(self.row_groups))
        if self.key_values:
            parts.append("Значения: " + ", ".join(self.key_values))
        return "\n".join(parts)


def _strip_html(text: str) -> str:
    """Удалить HTML-теги из текста, сохранив содержимое."""
    import html as _html

    # <sup> и <sub> -- содержимое объединяется с окружающим текстом
    text = re.sub(r"<sup[^>]*>([^<]*)</sup>", r"\1", text)
    text = re.sub(r"<sub[^>]*>([^<]*)</sub>", r"\1", text)
    # Удалить все остальные теги
    text = re.sub(r"<[^>]+>", "", text)
    # Декодировать HTML-сущности через стандартный модуль
    text = _html.unescape(text)
    return text.strip()


def _truncate_value(val: str) -> str:
    """Обрезать длинное значение."""
    val = val.strip()
    if len(val) > _MAX_VALUE_LEN:
        return val[:_MAX_VALUE_LEN] + "..."
    return val


def _extract_headings(text: str) -> list[str]:
    """Извлечь все Markdown-заголовки (# ... – ###### ...) перед таблицей."""
    pre_table = text.split("<table")[0] if "<table" in text.lower() else text
    # Для Markdown-таблиц — берём текст до первой строки |
    if "<table" not in text.lower():
        lines = text.split("\n")
        pre_table_lines = []
        for line in lines:
            if line.strip().startswith("|"):
                break
            pre_table_lines.append(line)
        pre_table = "\n".join(pre_table_lines)

    headings: list[str] = []
    for line in pre_table.split("\n"):
        stripped = line.strip()
        if re.match(r"^#{1,6}\s+", stripped):
            headings.append(stripped)
    return headings


def _extract_caption(text: str) -> str:
    """Извлечь полное описание таблицы из текста перед <table>.

    Возвращает все непустые не-заголовочные строки после последнего
    заголовка и до <table> (многострочный caption).
    """
    pre_table = text.split("<table")[0]

    # Собираем все непустые строки, не являющиеся заголовками #
    lines = pre_table.split("\n")
    non_header = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]

    if non_header:
        return "\n".join(non_header)

    return ""


def _summarize_html_table(text: str) -> TableSummary:
    """Извлечь сводку из HTML-таблицы (<table>...</table>)."""
    summary = TableSummary()

    # ── Заголовки (# ... – ###### ...) ────────────────────────────
    summary.headings = _extract_headings(text)

    # ── Caption ────────────────────────────────────────────────────
    summary.caption = _extract_caption(text)

    # ── Заголовки столбцов: <th> в <thead> ────────────────────────
    thead_match = re.search(r"<thead>(.*?)</thead>", text, re.DOTALL | re.IGNORECASE)
    if thead_match:
        thead = thead_match.group(1)
        # Обычные <th> без colspan
        ths = re.findall(r"<th>(.*?)</th>", thead, re.DOTALL)
        summary.columns = [_truncate_value(_strip_html(th)) for th in ths if _strip_html(th)]

    # ── Tbody: группы + ключевые значения ─────────────────────────
    tbody_match = re.search(r"<tbody>(.*?)</tbody>", text, re.DOTALL | re.IGNORECASE)
    if tbody_match:
        tbody = tbody_match.group(1)

        # Группы строк: <th colspan="N">...</th>
        group_ths = re.findall(
            r'<th\s+colspan\s*=\s*["\']?\d+["\']?\s*>(.*?)</th>',
            tbody, re.DOTALL | re.IGNORECASE,
        )
        summary.row_groups = [
            _truncate_value(_strip_html(g))
            for g in group_ths
            if _strip_html(g)
        ]

        # Ключевые значения: первый <td> каждой <tr>
        rows = re.findall(r"<tr>(.*?)</tr>", tbody, re.DOTALL | re.IGNORECASE)
        seen: set[str] = set()
        for row in rows:
            # Первый <td ...>...</td> в строке (включая rowspan)
            td_match = re.search(
                r"<td[^>]*>(.*?)</td>", row, re.DOTALL | re.IGNORECASE,
            )
            if td_match:
                val = _truncate_value(_strip_html(td_match.group(1)))
                if val and val not in seen:
                    seen.add(val)
                    summary.key_values.append(val)
                    if len(summary.key_values) >= _MAX_KEY_VALUES:
                        summary.key_values.append("...")
                        break

    return summary


def _summarize_md_table(text: str) -> TableSummary:
    """Извлечь сводку из Markdown-таблицы (|...|)."""
    summary = TableSummary()

    # ── Заголовки (# ... – ###### ...) ────────────────────────────
    summary.headings = _extract_headings(text)

    lines = text.split("\n")
    in_table = False
    header_parsed = False
    caption_parts: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Определяем начало таблицы
        if not in_table and stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2:
            in_table = True

        if not in_table:
            # Caption — собираем весь текст перед таблицей (не заголовок #)
            if stripped and not stripped.startswith("#"):
                caption_parts.append(stripped)
            continue

        # Внутри таблицы
        cells = [c.strip() for c in stripped.split("|")[1:-1]]

        # Разделитель |---|---|
        if all(re.match(r"^[-:]+$", c) for c in cells):
            continue

        if not header_parsed:
            # Первая строка = заголовки
            summary.columns = cells
            header_parsed = True
        else:
            # Строка данных — первый столбец = ключевое значение
            if cells and cells[0]:
                val = _truncate_value(cells[0])
                if val not in summary.key_values:
                    summary.key_values.append(val)
                    if len(summary.key_values) >= _MAX_KEY_VALUES:
                        summary.key_values.append("...")
                        break

    # Склеиваем все caption-строки
    if caption_parts:
        summary.caption = "\n".join(caption_parts)

    return summary


def generate_table_summary(text: str) -> str:
    """Сгенерировать компактную сводку таблицы для векторизации.

    Автоматически определяет формат: HTML (<table>) или Markdown (|...|).
    Fallback: первые 500 символов текста без заголовков.

    Args:
        text: Полный текст .md файла (заголовки # + caption + таблица).

    Returns:
        Текст сводки (~300-800 символов).
    """
    if "<table" in text.lower():
        summary = _summarize_html_table(text)
    elif "|" in text and re.search(r"^\s*\|(?:\s*[-:]+\s*\|)+\s*$", text, re.MULTILINE):
        summary = _summarize_md_table(text)
    else:
        # Fallback: первые значимые строки без #
        lines = text.split("\n")
        non_header = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        return "\n".join(non_header)[:500]

    return summary.to_text()