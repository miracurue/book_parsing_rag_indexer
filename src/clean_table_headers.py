"""Очистка артефактов парсинга в Markdown-файлах.

Удаляет / исправляет (выборочно через параметр enabled):
1. Символы # перед описаниями таблиц: '# Таблица ...' → 'Таблица ...'
2. Маркеры code block: ```markdown, ```html, голые ```
3. LaTeX-разделители: \\(...\\) → $...$, \\[...\\] → $$...$$
4. Одиночные цифры в конце текста (нумерация страниц VLM)
5. Тройные+ переносы строк → двойные
6. Атрибуты style="..." в HTML-тегах (border, padding, width, text-align и т.д.)
7. Теги <p ...> и </p> — удаляются, текст внутри сохраняется
"""

from __future__ import annotations

import re
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# Реестр типов артефактов
# ══════════════════════════════════════════════════════════════════════

ARTIFACT_TYPES: dict[str, str] = {
    "table_hash": "# перед «Таблица»",
    "code_blocks": "Маркеры code block (```markdown, ```html, ```)",
    "latex": "LaTeX-разделители (\\(...) → $...$)",
    "standalone_digits": "Одиночные цифры в конце текста (нумерация страниц)",
    "excessive_newlines": "Тройные+ переносы строк → двойные",
    "html_styles": "Атрибуты style=\"...\" в HTML-тегах таблиц",
    "html_p_tags": "Теги <p ...> и </p> (текст внутри сохраняется)",
    "bold_markers": "Markdown bold-маркеры ** (удаляются)",
}

ALL_ARTIFACT_KEYS = list(ARTIFACT_TYPES.keys())

# ══════════════════════════════════════════════════════════════════════
# Regex-паттерны
# ══════════════════════════════════════════════════════════════════════

# 1. # перед «Таблица» (от 1 до 6 символов #)
_TABLE_HASH_RE = re.compile(r"^(#{1,6})\s+(Таблица\s)", re.MULTILINE)

# 2. Маркеры code block с языком — удаляются ДО голых бэктиков
_MD_BLOCK_RE = re.compile(r"^```markdown\s*\n?", re.MULTILINE)
_HTML_BLOCK_RE = re.compile(r"^```html\s*\n?", re.MULTILINE)

# 3. Голые тройные бэктики — удаляются ПОСЛЕДНИМИ
_BACKTICKS_RE = re.compile(r"^```\s*\n?", re.MULTILINE)

# 4. LaTeX-разделители: \(...\) → $...$  и  \[...\] → $$...$$
_INLINE_LATEX_RE = re.compile(r"\\\((.+?)\\\)")
_BLOCK_LATEX_RE = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)

# 5. Одиночные цифры (1–3 цифры) в конце текста — нумерация страниц VLM
#    Только в конце файла: \n + опциональные пробелы + 1-3 цифры + хвост
_PAGE_NUM_END_RE = re.compile(r"\n[ \t]*\d{1,3}[ \t]*\s*\Z")

# 6. Три и более подряд \n → \n\n
_EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")

# 7. Атрибуты style="..." в HTML-тегах (таблицы, figure и др.)
#    Ловит: style="...", style='...', с любым содержимым внутри кавычек
_HTML_STYLE_RE = re.compile(r"""\s+style=["'][^"']*["']""")

# 8. Теги <p ...> и </p> — открывающий тег с любыми атрибутами и закрывающий
_HTML_OPEN_P_RE = re.compile(r"<p(?:\s[^>]*)?>")
_HTML_CLOSE_P_RE = re.compile(r"</p>")

# ══════════════════════════════════════════════════════════════════════
# Функции очистки (каждая возвращает tuple[str, int])
# ══════════════════════════════════════════════════════════════════════

def remove_table_hash(text: str) -> tuple[str, int]:
    """Удалить символы # перед словом «Таблица» в тексте."""
    count = 0

    def _replacer(match: re.Match) -> str:
        nonlocal count
        count += 1
        return match.group(2)  # «Таблица ...»

    cleaned = _TABLE_HASH_RE.sub(_replacer, text)
    return cleaned, count


def remove_code_block_markers(text: str) -> tuple[str, int]:
    """Удалить маркеры code block (```markdown, ```html, ```).

    Порядок важен: сначала специфичные маркеры с языком, потом голые ```.
    """
    total = 0

    text, n1 = _MD_BLOCK_RE.subn("", text)
    total += n1

    text, n2 = _HTML_BLOCK_RE.subn("", text)
    total += n2

    text, n3 = _BACKTICKS_RE.subn("", text)
    total += n3

    return text, total


def fix_latex_delimiters(text: str) -> tuple[str, int]:
    """Заменить LaTeX-разделители на Markdown-нотацию.

    \\(expr\\) → $expr$ (инлайн)
    \\[expr\\] → $$expr$$ (блочные)
    """
    total = 0

    # Блочные формулы — первыми
    text, n1 = _BLOCK_LATEX_RE.subn(r"$$\1$$", text)
    total += n1

    # Инлайн-формулы
    text, n2 = _INLINE_LATEX_RE.subn(r"$\1$", text)
    total += n2

    return text, total


def remove_standalone_digits(text: str) -> tuple[str, int]:
    """Удалить одиночные цифры (1–3 цифры) в конце текста.

    VLM часто оставляет номер страницы как отдельную строку в конце.
    Удаляется только последняя строка-цифра в самом конце файла.
    """
    text, n = _PAGE_NUM_END_RE.subn("", text)
    return text, n


def remove_excessive_newlines(text: str) -> tuple[str, int]:
    """Заменить 3+ подряд \\n на \\n\\n (двойной перенос допустим)."""
    text, n = _EXCESSIVE_NEWLINES_RE.subn("\n\n", text)
    return text, n


def remove_html_styles(text: str) -> tuple[str, int]:
    """Удалить атрибуты style=\"...\" из HTML-тегов.

    VLM генерирует стилизованный HTML (border, padding, width, text-align и т.д.),
    но для RAG-системы эти стили — мусор, не несущий смысловой нагрузки.
    """
    text, n = _HTML_STYLE_RE.subn("", text)
    return text, n


def remove_html_p_tags(text: str) -> tuple[str, int]:
    """Удалить теги <p ...> и </p>, сохранив текст внутри.

    VLM оборачивает подписи к таблицам в <p align="...">...</p>.
    Для RAG-системы важен только текст, без HTML-обёрток.
    """
    total = 0
    text, n1 = _HTML_OPEN_P_RE.subn("", text)
    total += n1
    text, n2 = _HTML_CLOSE_P_RE.subn("", text)
    total += n2
    return text, total


def remove_bold_markers(text: str) -> tuple[str, int]:
    """Удалить markdown bold-маркеры ** из текста.

    VLM оборачивает подписи к таблицам и continuation-маркеры
    в **...** (например, **Продолжение таблицы 1.17**).
    Для RAG-системы эти маркеры — мусор.
    """
    text, n = re.subn(r"\*\*", "", text)
    return text, n


# ══════════════════════════════════════════════════════════════════════
# Подсчёт артефактов (без модификации)
# ══════════════════════════════════════════════════════════════════════

def _count_table_hash(text: str) -> int:
    return len(_TABLE_HASH_RE.findall(text))


def _count_code_blocks(text: str) -> int:
    return (
        len(_MD_BLOCK_RE.findall(text))
        + len(_HTML_BLOCK_RE.findall(text))
        + len(_BACKTICKS_RE.findall(text))
    )


def _count_latex(text: str) -> int:
    return len(_INLINE_LATEX_RE.findall(text)) + len(_BLOCK_LATEX_RE.findall(text))


def _count_standalone_digits(text: str) -> int:
    return len(_PAGE_NUM_END_RE.findall(text))


def _count_excessive_newlines(text: str) -> int:
    return len(_EXCESSIVE_NEWLINES_RE.findall(text))


_COUNT_FN = {
    "table_hash": _count_table_hash,
    "code_blocks": _count_code_blocks,
    "latex": _count_latex,
    "standalone_digits": _count_standalone_digits,
    "excessive_newlines": _count_excessive_newlines,
    "html_styles": lambda t: len(_HTML_STYLE_RE.findall(t)),
    "html_p_tags": lambda t: len(_HTML_OPEN_P_RE.findall(t)) + len(_HTML_CLOSE_P_RE.findall(t)),
    "bold_markers": lambda t: t.count("**"),
}


def count_artifacts(
    text: str,
    enabled: dict[str, bool] | None = None,
) -> dict[str, int]:
    """Посчитать артефакты в тексте по типам (без модификации).

    Args:
        text: Исходный Markdown-текст.
        enabled: Словарь {тип: True/False}. None = все типы.

    Returns:
        Словарь {тип: количество}, плюс ключ "total" с суммой.
    """
    result: dict[str, int] = {}
    for key in ALL_ARTIFACT_KEYS:
        if enabled is not None and not enabled.get(key, False):
            continue
        result[key] = _COUNT_FN[key](text)
    result["total"] = sum(v for k, v in result.items() if k != "total")
    return result


# ══════════════════════════════════════════════════════════════════════
# Обработка одного файла
# ══════════════════════════════════════════════════════════════════════

# Маппинг: ключ → функция очистки
_CLEAN_FN = {
    "table_hash": remove_table_hash,
    "code_blocks": remove_code_block_markers,
    "latex": fix_latex_delimiters,
    "standalone_digits": remove_standalone_digits,
    "excessive_newlines": remove_excessive_newlines,
    "html_styles": remove_html_styles,
    "html_p_tags": remove_html_p_tags,
    "bold_markers": remove_bold_markers,
}


def process_file(
    filepath: Path,
    enabled: dict[str, bool] | None = None,
) -> dict:
    """Обработать один .md файл: удалить выбранные артефакты парсинга.

    Файл перезаписывается на месте (in-place).

    Args:
        filepath: Путь к .md файлу.
        enabled: Словарь {тип: True/False}. None = все типы включены.

    Returns:
        dict: {
            "file": имя файла,
            "replacements": количество замен,
            "modified": True/False,
            "error": None или текст ошибки,
            "details": {тип: количество_замен},
        }
    """
    try:
        content = filepath.read_text(encoding="utf-8")

        total = 0
        details: dict[str, int] = {}

        for key in ALL_ARTIFACT_KEYS:
            # Пропуск отключённых типов
            if enabled is not None and not enabled.get(key, False):
                continue

            content, n = _CLEAN_FN[key](content)
            total += n
            if n > 0:
                details[key] = n

        if total > 0:
            filepath.write_text(content, encoding="utf-8")

        return {
            "file": filepath.name,
            "replacements": total,
            "modified": total > 0,
            "error": None,
            "details": details,
        }

    except Exception as e:
        return {
            "file": filepath.name,
            "replacements": 0,
            "modified": False,
            "error": str(e),
            "details": {},
        }


# ══════════════════════════════════════════════════════════════════════
# Обратная совместимость
# ══════════════════════════════════════════════════════════════════════

def count_table_headers(text: str) -> int:
    """Посчитать количество всех артефактов (без модификации).

    .. deprecated:: Используйте count_artifacts().
    """
    return count_artifacts(text)["total"]