"""Извлечение таблиц из Markdown-файлов парсинга.

Находит HTML-таблицы (<table>...</table>) и Markdown-таблицы (pipe-синтаксис).
Для каждой таблицы определяет описание (caption):
  1. <caption> внутри <table> — извлекается, тег удаляется из HTML
  2. Строка «Таблица N...» перед таблицей
  3. Предыдущий абзац текста (последний непустой текст до таблицы)

Если в тексте есть маркеры [TABLE_START]/[TABLE_END] (после нормализации),
границы таблиц определяются по маркерам — надёжно и точно.

Результат: каждая таблица → отдельный .md файл в parsed/extracted_tables/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .normalize_tables import MARKER_END, MARKER_START


# ── Regex-шаблоны ──────────────────────────────────────────────────────

# HTML-таблица: <table ...>...</table>
_HTML_TABLE_RE = re.compile(
    r"<table\b.*?</table>",
    re.DOTALL | re.IGNORECASE,
)

# <caption> внутри HTML
_CAPTION_TAG_RE = re.compile(
    r"<caption>(.*?)</caption>",
    re.DOTALL | re.IGNORECASE,
)

# Строка-описание таблицы: «Таблица 1.2», «Табл. 3», «Таблица 1»
# Опционально учитывает теги <NNN> перед «Таблица» (в файлах chapters/)
_TABLE_CAPTION_LINE_RE = re.compile(
    r"^(?:<\d+>\s*)*Таблица\s+\d+[\.\d]*.*$",
    re.MULTILINE,
)

# Markdown-таблица: строка с pipe-разделителем |---|---|...
# Допускаются | внутри (мультиколоночный разделитель)
_MD_SEPARATOR_RE = re.compile(
    r"^\|[\s\-:|]+\|?$",
)

# Строка markdown-таблицы: начинается с |
_MD_ROW_RE = re.compile(
    r"^\|",
)

# «Таблица N...» в любом месте строки (без привязки к началу строки)
_TABLE_CAPTION_IN_LINE_RE = re.compile(
    r"Таблица\s+\d+[\.\d]*",
)

# Теги номеров страниц <NNN> — используются в нескольких функциях
_PAGE_TAG_RE = re.compile(r"<(\d+)>")


@dataclass
class ExtractedTable:
    """Извлечённая таблица с описанием."""
    caption: str          # Описание (может быть пустым)
    body: str             # Тело таблицы (HTML или Markdown)
    table_type: str       # "html" или "markdown"
    start_pos: int        # Позиция начала в исходном тексте
    end_pos: int          # Позиция конца в исходном тексте


def _extract_html_tables(text: str) -> list[ExtractedTable]:
    """Найти все HTML-таблицы в тексте.

    Извлекает <caption> изнутри таблицы и удаляет тег.
    """
    tables: list[ExtractedTable] = []

    for match in _HTML_TABLE_RE.finditer(text):
        html = match.group(0)
        caption = ""

        # Попытка извлечь <caption>
        cap_match = _CAPTION_TAG_RE.search(html)
        if cap_match:
            caption = cap_match.group(1).strip()
            # Удалить тег <caption> из HTML
            html = _CAPTION_TAG_RE.sub("", html, count=1).strip()

        tables.append(ExtractedTable(
            caption=caption,
            body=html,
            table_type="html",
            start_pos=match.start(),
            end_pos=match.end(),
        ))

    return tables


def _extract_markdown_tables(text: str) -> list[ExtractedTable]:
    """Найти все Markdown-таблицы в тексте.

    Markdown-таблица = группа подряд идущих строк, начинающихся с |,
    внутри которой есть хотя бы одна строка-разделитель |---|...|.
    """
    tables: list[ExtractedTable] = []
    lines = text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Ищем начало группы pipe-строк
        if not _MD_ROW_RE.match(line):
            i += 1
            continue

        # Собираем подряд идущие pipe-строки
        group_start = i
        has_separator = False

        while i < len(lines) and _MD_ROW_RE.match(lines[i]):
            if _MD_SEPARATOR_RE.match(lines[i].strip()):
                has_separator = True
            i += 1

        group_end = i  # эксклюзивный

        # Группа считается таблицей, если есть разделитель и ≥3 строк
        # (заголовок + разделитель + хотя бы одна строка данных)
        if has_separator and (group_end - group_start) >= 3:
            table_lines = lines[group_start:group_end]
            table_body = "\n".join(table_lines)

            # Позиции в исходном тексте
            start_pos = sum(len(lines[j]) + 1 for j in range(group_start))
            end_pos = sum(len(lines[j]) + 1 for j in range(group_end))

            tables.append(ExtractedTable(
                caption="",
                body=table_body,
                table_type="markdown",
                start_pos=start_pos,
                end_pos=end_pos,
            ))

    return tables


def _find_preceding_paragraph(text: str, pos: int) -> str:
    """Найти последний непустой абзац текста перед позицией pos.

    Абзац = текст между пустыми строками.
    Возвращает пустую строку, если перед pos только пустота.
    """
    before = text[:pos].rstrip()

    if not before:
        return ""

    # Разделяем по двойному переводу строки
    paragraphs = re.split(r"\n\s*\n", before)
    # Последний непустой абзац
    for p in reversed(paragraphs):
        p = p.strip()
        if p:
            return p

    return ""


def _extract_caption_from_line(stripped: str) -> str | None:
    """Извлечь caption «Таблица N...» из строки.

    Сначала убирает теги <NNN>, затем ищет «Таблица N...»
    в начале строки или в любом месте (inline-описание).
    Возвращает полную строку caption или None.
    """
    # Убираем теги номеров страниц для анализа
    no_tags = _PAGE_TAG_RE.sub("", stripped).strip()
    if not no_tags:
        return None

    # Приоритет 1: caption в начале строки (после удаления тегов)
    if _TABLE_CAPTION_LINE_RE.match(no_tags):
        return no_tags

    # Приоритет 2: «Таблица N...» в середине строки
    # (например: «<p>*A – ...</p> <015> Таблица 4. ...»)
    m = _TABLE_CAPTION_IN_LINE_RE.search(no_tags)
    if m:
        return no_tags[m.start():]

    return None


def _is_prev_table_html_content(no_tags: str) -> bool:
    """Проверить, что строка — содержимое предыдущей HTML-таблицы.

    Такие строки (</table>, </tr>, <td>...) нужно пропускать при поиске
    caption для следующей таблицы.
    """
    if no_tags.startswith("</table") or no_tags.startswith("</tbody") or no_tags.startswith("</thead"):
        return True
    if no_tags.startswith("<td") or no_tags.startswith("</td") or no_tags.startswith("<th") or no_tags.startswith("</th"):
        return True
    if no_tags.startswith("<tr") or no_tags.startswith("</tr"):
        return True
    return False


def _find_table_caption_line(text: str, pos: int) -> tuple[str, int]:
    """Найти строку «Таблица N...» перед позицией pos.

    Ищет в последних 10 строках перед pos.
    Поддерживает caption как в начале строки, так и в середине.
    Пропускает строки-теги <NNN> и HTML-содержимое предыдущей таблицы.

    Возвращает (caption, caption_end_pos). Если не найдено — ("", pos).
    """
    before = text[:pos].rstrip()
    if not before:
        return "", pos

    lines = before.split("\n")

    # Проверяем до 10 последних строк (caption может быть отделён
    # содержимым предыдущей таблицы и сносками)
    caption_parts: list[str] = []

    for idx_i, line in enumerate(reversed(lines[-12:])):
        stripped = line.strip()
        no_tags = _PAGE_TAG_RE.sub("", stripped).strip()

        cap = _extract_caption_from_line(stripped)
        if cap is not None:
            caption_parts.insert(0, cap)
        elif caption_parts:
            # Строка выше caption — проверяем многострочное описание
            if (no_tags
                    and not no_tags.startswith("|")
                    and not no_tags.startswith("<table")
                    and not _is_prev_table_html_content(no_tags)):
                caption_parts.insert(0, stripped)
            else:
                break
        elif not no_tags:
            # Пустая строка или только теги <NNN>
            continue
        elif _is_prev_table_html_content(no_tags):
            # Содержимое предыдущей HTML-таблицы — пропускаем
            continue
        else:
            break

    if not caption_parts:
        return "", pos

    caption = "\n".join(caption_parts)
    return caption, pos


def extract_tables_from_text(text: str) -> list[ExtractedTable]:
    """Извлечь все таблицы из Markdown-текста с описаниями.

    Алгоритм:
    1. Найти все HTML-таблицы и Markdown-таблицы
    2. Отсортировать по позиции в тексте
    3. Для каждой таблицы определить caption:
       a. <caption> внутри <table> (уже извлечён)
       b. Строка «Таблица N...» перед таблицей
       c. Предыдущий абзац текста

    Args:
        text: Содержимое Markdown-файла.

    Returns:
        Список ExtractedTable, отсортированный по позиции.
    """
    # Шаг 1: найти все таблицы
    html_tables = _extract_html_tables(text)
    md_tables = _extract_markdown_tables(text)

    all_tables = html_tables + md_tables
    all_tables.sort(key=lambda t: t.start_pos)

    # Шаг 2: для каждой таблицы определить caption
    for table in all_tables:
        if table.caption:
            # <caption> уже извлечён из HTML — приоритет 1
            continue

        # Приоритет 2: строка «Таблица N...»
        caption, _ = _find_table_caption_line(text, table.start_pos)
        if caption:
            table.caption = caption
            continue

        # Приоритет 3: предыдущий абзац текста
        para = _find_preceding_paragraph(text, table.start_pos)
        if para:
            # Ограничиваем длину — берём последний абзац
            # Если абзац слишком длинный (>500 символов), берём последнюю строку
            if len(para) > 500:
                last_line = para.rsplit("\n", 1)[-1].strip()
                if last_line:
                    table.caption = last_line
                else:
                    # Берём последние 200 символов
                    table.caption = "..." + para[-200:].strip()
            else:
                table.caption = para

    return all_tables


def extract_page_number(filepath: Path) -> str:
    """Извлечь номер страницы из имени файла.

    Examples:
        Path("023.md") → "023"
        Path("005.md") → "005"

    Returns:
        Строка с номером (с ведущими нулями) или "000".
    """
    stem = filepath.stem
    if stem.isdigit():
        return stem
    return "000"


def format_output_filename(page_num: str, table_index: int, total_tables: int) -> str:
    """Сформировать имя выходного файла.

    Args:
        page_num: Номер страницы (напр. "023").
        table_index: Индекс таблицы на странице (0-based).
        total_tables: Всего таблиц на странице.

    Returns:
        "023.md" если одна таблица, "023-1.md", "023-2.md" если несколько.
    """
    if total_tables <= 1:
        return f"{page_num}.md"
    return f"{page_num}-{table_index + 1}.md"


def format_table_file_content(table: ExtractedTable) -> str:
    """Сформировать содержимое .md файла для одной таблицы.

    Формат:
        <caption>\n\n<body>
        или
        <body> (без caption)
    """
    parts: list[str] = []

    if table.caption:
        parts.append(table.caption)

    parts.append(table.body)

    return "\n\n".join(parts) + "\n"


def process_file(
    filepath: Path,
    output_dir: Path,
) -> list[dict]:
    """Обработать один .md файл: извлечь таблицы и сохранить.

    Args:
        filepath: Путь к исходному .md файлу.
        output_dir: Папка для сохранения извлечённых таблиц.

    Returns:
        Список dict с результатами для каждой таблицы:
        [
            {
                "source_file": "010.md",
                "output_file": "010.md",
                "table_type": "html",
                "caption": "Таблица 1.2...",
                "error": None,
            },
            ...
        ]
    """
    results: list[dict] = []

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return [{
            "source_file": filepath.name,
            "output_file": None,
            "table_type": None,
            "caption": None,
            "error": str(e),
        }]

    tables = extract_tables_from_text(content)

    if not tables:
        return []

    page_num = extract_page_number(filepath)
    total = len(tables)

    for idx, table in enumerate(tables):
        out_name = format_output_filename(page_num, idx, total)
        out_path = output_dir / out_name

        try:
            file_content = format_table_file_content(table)
            out_path.write_text(file_content, encoding="utf-8")

            results.append({
                "source_file": filepath.name,
                "output_file": out_name,
                "table_type": table.table_type,
                "caption": table.caption[:100] if table.caption else "",
                "error": None,
            })
        except Exception as e:
            results.append({
                "source_file": filepath.name,
                "output_file": out_name,
                "table_type": table.table_type,
                "caption": table.caption[:100] if table.caption else "",
                "error": str(e),
            })

    return results


# ══════════════════════════════════════════════════════════════════════
# Функции для извлечения таблиц из файлов глав (chapters/)
# ══════════════════════════════════════════════════════════════════════


# ── Извлечение по маркерам [TABLE_START]/[TABLE_END] ─────────────────────

def _split_by_markers(text: str) -> list[tuple[str, int, int]]:
    """Разделить текст на блоки по маркерам [TABLE_START]...[TABLE_END].

    Returns:
        [(block_text, start_pos, end_pos), ...] — только блоки между маркерами.
    """
    blocks = []
    search_start = 0

    while True:
        start_idx = text.find(MARKER_START, search_start)
        if start_idx == -1:
            break

        # Пропускаем сам маркер + \n
        content_start = start_idx + len(MARKER_START)
        if content_start < len(text) and text[content_start] == "\n":
            content_start += 1

        end_idx = text.find(MARKER_END, content_start)
        if end_idx == -1:
            break

        # Берём контент до маркера (без \n перед ним)
        content_end = end_idx
        if content_end > 0 and text[content_end - 1] == "\n":
            content_end -= 1

        block_text = text[content_start:content_end]
        blocks.append((block_text, start_idx, end_idx + len(MARKER_END)))

        search_start = end_idx + len(MARKER_END)

    return blocks


def _parse_marker_block(
    block_text: str,
    block_start: int,
) -> list[ExtractedTable]:
    """Разобрать блок между маркерами на таблицы.

    Блок может содержать caption перед таблицей и сноски после.
    Внутри может быть одна или несколько таблиц (HTML/Markdown).
    """
    # Находим таблицы внутри блока
    html_tables = _extract_html_tables(block_text)
    md_tables = _extract_markdown_tables(block_text)
    all_tables = html_tables + md_tables
    all_tables.sort(key=lambda t: t.start_pos)

    if not all_tables:
        # Подфункции не нашли таблицы (например, pipe-строки с тегами <NNN>)
        # Разделяем блок на caption + тело по первой pipe/HTML строке
        lines = block_text.split("\n")
        split_line = 0
        for j, ln in enumerate(lines):
            clean = _PAGE_TAG_RE.sub("", ln).strip()
            if clean.startswith("|") or clean.startswith("<table"):
                split_line = j
                break

        caption_raw = "\n".join(lines[:split_line]).strip()
        body_raw = "\n".join(lines[split_line:]).strip()

        all_tables = [ExtractedTable(
            caption=_remove_page_tags(caption_raw).strip() if caption_raw else "",
            body=body_raw if body_raw else block_text,
            table_type="block",
            start_pos=sum(len(lines[k]) + 1 for k in range(split_line)) if split_line > 0 else 0,
            end_pos=len(block_text),
        )]

    # Текст до первой таблицы — caption
    first_table_start = all_tables[0].start_pos
    before_text = block_text[:first_table_start].strip()

    # Очищаем caption от тегов <NNN>
    if before_text:
        caption = _remove_page_tags(before_text).strip()
    else:
        caption = ""

    # Назначаем caption первой таблице
    if not all_tables[0].caption and caption:
        all_tables[0].caption = caption
    elif all_tables[0].caption and caption:
        # <caption> внутри таблицы + «Таблица N...» перед ней
        if caption in all_tables[0].caption:
            pass  # уже содержит — не дублируем
        elif all_tables[0].caption in caption:
            all_tables[0].caption = caption  # before_text полнее
        else:
            all_tables[0].caption = caption + "\n" + all_tables[0].caption

    # Для последующих таблиц — <caption> уже извлечён из HTML,
    # или оставляем пустым

    # Корректируем позиции на абсолютные в тексте
    for table in all_tables:
        table.start_pos += block_start
        table.end_pos += block_start

    return all_tables


def extract_tables_by_markers(text: str) -> list[ExtractedTable]:
    """Извлечь таблицы по маркерам [TABLE_START]/[TABLE_END].

    Если маркеры есть — использует их для точного определения границ.
    Возвращает пустой список, если маркеров нет.

    Args:
        text: Текст с маркерами.

    Returns:
        Список ExtractedTable.
    """
    blocks = _split_by_markers(text)
    if not blocks:
        return []

    all_tables = []
    for block_text, block_start, _ in blocks:
        tables = _parse_marker_block(block_text, block_start)
        all_tables.extend(tables)

    return all_tables


def has_markers(text: str) -> bool:
    """Проверить, содержит ли текст маркеры таблиц."""
    return MARKER_START in text or MARKER_END in text


# ══════════════════════════════════════════════════════════════════════
# Функции для извлечения таблиц из файлов глав (chapters/)
# ══════════════════════════════════════════════════════════════════════


def _extract_headings(text: str) -> list[str]:
    """Извлечь все строки-заголовки (# ... ## ... и т.д.) из текста."""
    headings = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            headings.append(stripped)
    return headings


def _get_first_page_number(text: str) -> str:
    """Получить первый номер страницы из тега <NNN> в тексте.

    Returns: строка с номером (с ведущими нулями) или '000'.
    """
    m = _PAGE_TAG_RE.search(text)
    return m.group(1) if m else "000"


def _get_page_range(text: str) -> str:
    """Получить диапазон номеров страниц из тегов <NNN> в тексте.

    Извлекает все уникальные теги <NNN>, сортирует и склеивает через '_'.
    Examples:
        "<045> ..." → "045"
        "<045> ... <046> ..." → "045_046"
        "<045> ... <046> ... <047> ..." → "045_046_047"
        (нет тегов) → "000"

    Returns: строка с номерами страниц (с ведущими нулями).
    """
    nums = sorted(set(m.group(1) for m in _PAGE_TAG_RE.finditer(text)))
    if not nums:
        return "000"
    return "_".join(nums)


def _remove_page_tags(text: str) -> str:
    """Удалить все теги <NNN> из текста."""
    return _PAGE_TAG_RE.sub("", text)


def _is_table_note(text_line: str) -> bool:
    """Проверить, является ли строка сноской/примечанием к таблице.

    Признаки:
    - <p>*...»</p> — HTML-сноска
    - *... — текстовая сноска (начинается с *)
    - Текст в скобках, начинающийся со *
    """
    no_tags = _PAGE_TAG_RE.sub("", text_line).strip()
    if not no_tags:
        return False
    # Сноски: <p>*A...</p> или * Данные... или (*...)
    if re.match(r"^(\*|<p>\*|(\*\s))", no_tags):
        return True
    return False


def _find_table_removal_start(text: str, table: ExtractedTable) -> int:
    """Найти начало диапазона удаления для таблицы (включая caption перед ней).

    Ищет строки «Таблица N...» и сноски непосредственно перед телом таблицы.
    Пропускает строки-теги <NNN> и HTML-содержимое предыдущей таблицы.
    """
    before = text[: table.start_pos]
    if not before.strip():
        return table.start_pos

    lines = before.split("\n")

    # Идём назад от конца, ищем caption и сноски
    first_removal_idx = len(lines)
    found = False

    for i in range(len(lines) - 1, max(len(lines) - 12, -1), -1):
        no_tags = _PAGE_TAG_RE.sub("", lines[i]).strip()

        if not no_tags:
            if found:
                break
            continue

        if _is_prev_table_html_content(no_tags):
            # Содержимое предыдущей HTML-таблицы — пропускаем
            continue

        cap = _extract_caption_from_line(lines[i].strip())
        if cap is not None:
            found = True
            first_removal_idx = i
        elif found and no_tags and not no_tags.startswith("|"):
            # Продолжение описания таблицы или сноска
            first_removal_idx = i
        else:
            break

    if not found:
        return table.start_pos

    # Вычисляем позицию начала первой строки удаления
    pos = sum(len(lines[j]) + 1 for j in range(first_removal_idx))
    return pos


def _find_post_table_end(text: str, table: ExtractedTable, tables: list[ExtractedTable]) -> int:
    """Найти конец диапазона удаления после таблицы (сноски, примечания).

    После </table> или конца Markdown-таблицы могут быть:
    - Сноски: * Данные..., <p>*A – ...</p>
    - Пустые строки с тегами <NNN>

    Ищет до следующей таблицы, первого значимого текста или конца файла.
    """
    after = text[table.end_pos:]
    if not after.strip():
        return table.end_pos

    lines = after.split("\n")

    # Найдём позицию начала следующей таблицы (если есть)
    next_table_start = len(text)  # по умолчанию — конец файла
    for t in tables:
        if t.start_pos > table.end_pos:
            next_table_start = t.start_pos
            break

    # Проверяем строки после таблицы
    end_pos = table.end_pos
    for line in lines:
        line_start_in_text = end_pos
        stripped = line.strip()
        no_tags = _PAGE_TAG_RE.sub("", stripped).strip()

        # Пустая строка или строка с одними тегами <NNN> — пропускаем
        if not no_tags:
            end_pos += len(line) + 1  # +1 за \n
            continue

        # Сноска/примечание — включаем в удаление
        if _is_table_note(stripped):
            end_pos += len(line) + 1
            continue

        # Caption следующей таблицы — НЕ включаем, это belongs к след. таблице
        cap = _extract_caption_from_line(stripped)
        if cap is not None:
            break

        # Значимый текст (не сноска, не caption) — останавливаемся
        # Но проверяем, что не вышли за начало следующей таблицы
        if line_start_in_text >= next_table_start:
            break

        # Если текст на той же строке что и </table> — пропускаем
        # (например: «</table> <017> Текст...» — текст belongs к след. таблице)
        if table.table_type == "html" and "</table>" in line:
            # Текст после </table> на этой же строке
            after_close = line[line.index("</table>") + len("</table>"):]
            after_clean = _PAGE_TAG_RE.sub("", after_close).strip()
            if after_clean and not _is_table_note(after_close):
                break
        break

    return end_pos


def _remove_ranges_from_text(text: str, ranges: list[tuple[int, int]]) -> str:
    """Удалить несколько диапазонов из текста и очистить лишние переносы."""
    if not ranges:
        return text

    ranges = sorted(ranges)
    parts: list[str] = []
    prev_end = 0

    for start, end in ranges:
        if start > prev_end:
            parts.append(text[prev_end:start])
        prev_end = max(prev_end, end)

    if prev_end < len(text):
        parts.append(text[prev_end:])

    cleaned = "".join(parts)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"
    return cleaned


def process_chapter_file(
    filepath: Path,
    output_dir: Path,
) -> list[dict]:
    """Обработать файл главы: извлечь таблицы по маркерам [TABLE_START]/[TABLE_END].

    Исходный файл НЕ изменяется. Для извлечения используются только
    промаркированные блоки (после нормализации таблиц).

    Для каждой таблицы:
    - Имя файла = номер страницы из первого тега <NNN> в тексте таблицы
    - Содержимое: все заголовки главы + caption + тело таблицы (теги <NNN> удалены)

    Args:
        filepath:   Путь к файлу главы (chapters/*.md).
        output_dir: Папка для извлечённых таблиц (parsed/extracted_tables/).

    Returns:
        Список dict с результатами. Если маркеров нет — список с одним dict
        с ошибкой "no_markers".
    """
    results: list[dict] = []

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return [{
            "source_file": filepath.name, "output_file": None,
            "table_type": None, "caption": None, "error": str(e),
        }]

    # Проверяем наличие маркеров
    if not has_markers(content):
        return [{
            "source_file": filepath.name, "output_file": None,
            "table_type": None, "caption": None,
            "error": "no_markers",
        }]

    # Собираем все заголовки из файла
    headings = _extract_headings(content)
    heading_block = "\n".join(headings)

    # Извлекаем таблицы по маркерам
    tables = extract_tables_by_markers(content)

    if not tables:
        return []

    total = len(tables)
    for idx, table in enumerate(tables):
        # Ищем <NNN> во всём маркерном блоке (от [TABLE_START] до конца таблицы),
        # т.к. тег <NNN> может быть перед <table> и не попасть в table.body
        marker_pos = content.rfind(MARKER_START, 0, table.start_pos)
        if marker_pos >= 0:
            search_text = content[marker_pos + len(MARKER_START):table.end_pos]
        else:
            search_text = table.body
        page_num = _get_page_range(search_text)

        body_clean = _remove_page_tags(table.body).strip()
        parts: list[str] = []

        if heading_block:
            parts.append(heading_block)

        if table.caption:
            caption_clean = _remove_page_tags(table.caption).strip()
            parts.append(caption_clean)

        parts.append(body_clean)
        file_content = "\n\n".join(parts) + "\n"

        out_name = format_output_filename(page_num, idx, total)
        out_path = output_dir / out_name

        try:
            out_path.write_text(file_content, encoding="utf-8")
            results.append({
                "source_file": filepath.name,
                "output_file": out_name,
                "table_type": table.table_type,
                "caption": table.caption[:100] if table.caption else "",
                "page_num": page_num,
                "error": None,
            })
        except Exception as e:
            results.append({
                "source_file": filepath.name,
                "output_file": out_name,
                "table_type": table.table_type,
                "caption": table.caption[:100] if table.caption else "",
                "page_num": page_num,
                "error": str(e),
            })

    return results


def count_tables_in_text(text: str) -> dict:
    """Подсчитать количество таблиц в тексте (без сохранения).

    Returns:
        {"html": int, "markdown": int, "total": int}
    """
    html = len(_extract_html_tables(text))
    md = len(_extract_markdown_tables(text))
    return {"html": html, "markdown": md, "total": html + md}


def count_tables_by_markers(text: str) -> dict:
    """Подсчитать таблицы по маркерам [TABLE_START]/[TABLE_END].

    Использует тот же _split_by_markers(), что и при реальном извлечении —
    гарантирует совпадение числа в предпросмотре и при запуске.

    Returns:
        {"markers": int, "has_markers": bool}
        markers — количество блоков [TABLE_START]...[TABLE_END].
    """
    blocks = _split_by_markers(text)
    return {"markers": len(blocks), "has_markers": len(blocks) > 0}
