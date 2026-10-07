"""Нормализация границ таблиц в Markdown-файлах глав (chapters/).

Добавляет маркеры [TABLE_START] и [TABLE_END] вокруг таблиц,
опираясь на слово «Таблица»/«Табл.» как якорь.

Алгоритм:
  1. Предобработка: разделить смешанные строки
  2. Найти строки с caption («Таблица N», «Табл. N», «Окончание табл. N»)
  3. [TABLE_START] поставить перед <NNN>-тегом (если есть) или перед caption
  4. От caption найти конец таблицы (</table> или конец |...|)
  5. Найти сноски после таблицы
  6. [TABLE_END] поставить после сносок

Если слова «Таблица»/«Табл.» нет — маркеры не ставятся, таблица пропускается.
"""

from __future__ import annotations

import re


# ── Константы ──────────────────────────────────────────────────────────────

MARKER_START = "[TABLE_START]"
MARKER_END = "[TABLE_END]"

# Сколько строк вперёд искать тело таблицы после caption
MAX_TABLE_BODY_LOOKAHEAD = 30

# Сколько строк вперёд искать сноски после таблицы
MAX_FOOTNOTE_LOOKAHEAD = 10


# ── Regex-шаблоны ──────────────────────────────────────────────────────────

# Тег номера страницы <NNN>
_PAGE_TAG_RE = re.compile(r"<(\d+)>")

# Caption в начале строки (после необязательного <NNN>)
# «Таблица 15. ...», «Табл. 3 ...», «Таблица 1.2»
_CAPTION_START_RE = re.compile(
    r"^(?:<\d+>\s*)?(?:Таблица|Табл\.)\s+\d+[\.\d]*(?:\s|$)",
)

# Продолжение/окончание таблицы в начале строки
_CONTINUATION_START_RE = re.compile(
    r"^(?:<\d+>\s*)?\*{0,2}(?:Окончание|Продолжение)\s+(?:табл\.|таблицы)\s+\d+",
    re.IGNORECASE,
)

# Caption в середине строки (для предобработки)
_CAPTION_MID_RE = re.compile(r"(?:Таблица|Табл\.)\s+\d+[\.\d]*(?:\s|$)")
_CONTINUATION_MID_RE = re.compile(
    r"\*{0,2}(?:Окончание|Продолжение)\s+(?:табл\.|таблицы)\s+\d+",
    re.IGNORECASE,
)

# HTML-таблица (для подсчёта)
_HTML_TABLE_RE = re.compile(
    r"<table\b.*?</table>",
    re.DOTALL | re.IGNORECASE,
)

# Markdown-таблица
_MD_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|?$")
_MD_ROW_RE = re.compile(r"^\|")

# Сноска после таблицы: * текст, ** текст, *** текст
_FOOTNOTE_RE = re.compile(r"^(?:<p>)?(?:\\+)?\*")


# ── Вспомогательные ────────────────────────────────────────────────────────

def _strip_page_tags(text: str) -> str:
    """Удалить теги <NNN> из строки."""
    return _PAGE_TAG_RE.sub("", text).strip()


def _is_caption_line(line: str) -> bool:
    """Проверить, что строка — caption таблицы или continuation."""
    s = line.strip()
    return bool(_CAPTION_START_RE.match(s) or _CONTINUATION_START_RE.match(s))


# Regex для извлечения номера таблицы из caption/continuation
_TABLE_NUM_RE = re.compile(
    r"(?:Таблица|Табл\.|табл\.|таблицы|таблицу)\s+(\d+[\.\d]*)",
)


def _extract_table_number(line: str) -> str | None:
    """Извлечь номер таблицы из caption или continuation строки.

    Примеры::
        «Таблица 1.17 — Производство...» → «1.17»
        «Продолжение таблицы 1.17» → «1.17»
        «Окончание табл. 1.8» → «1.8»
        «Таблица 15» → «15»
    """
    # Сначала чистим от <NNN> тегов
    clean = _strip_page_tags(line).strip()
    m = _TABLE_NUM_RE.search(clean)
    return m.group(1) if m else None


def _is_continuation_line(line: str) -> bool:
    """Проверить, что строка — продолжение/окончание таблицы (НЕ первичный caption).

    Различает:
      - «Таблица 1.17 — ...» → False (первичный caption)
      - «Продолжение таблицы 1.17» → True
      - «Окончание табл. 1.8» → True
      - «Таблица 1.8 — Окончание таблицы 1.8» → True (составной caption)
    """
    s = line.strip()
    # Прямое продолжение: начинается с «Продолжение/Окончание»
    if _CONTINUATION_START_RE.match(s):
        return True
    # Составной caption: «Таблица N — Окончание/Продолжение таблицы N»
    clean = _strip_page_tags(s)
    if _CAPTION_START_RE.match(s) and re.search(
        r"(?:Окончание|Продолжение)\s+(?:табл\.|таблицы)",
        clean,
        re.IGNORECASE,
    ):
        return True
    return False


def _merge_continuation_ranges(
    ranges: list[tuple[int, int]], lines: list[str]
) -> list[tuple[int, int]]:
    """Объединить диапазоны продолжений одной таблицы.

    Если следующий диапазон — continuation с тем же номером таблицы,
    он объединяется с предыдущим (end расширяется).

    Returns:
        Список объединённых непересекающихся диапазонов.
    """
    if len(ranges) <= 1:
        return ranges

    merged: list[tuple[int, int]] = [ranges[0]]

    for start_idx, end_idx in ranges[1:]:
        prev_start, prev_end = merged[-1]

        # Проверяем, является ли текущий диапазон продолжением предыдущего
        caption_line = lines[start_idx]
        prev_caption_line = None

        # Ищем первую непустую строку-заголовок предыдущего диапазона
        for k in range(prev_start, min(prev_start + 5, len(lines))):
            if _is_caption_line(lines[k]):
                prev_caption_line = lines[k]
                break

        if _is_continuation_line(caption_line) and prev_caption_line:
            curr_num = _extract_table_number(caption_line)
            prev_num = _extract_table_number(prev_caption_line)

            if curr_num and prev_num and curr_num == prev_num:
                # Merge: расширяем end предыдущего диапазона
                merged[-1] = (prev_start, end_idx)
                continue

        # Не продолжение — отдельный диапазон
        merged.append((start_idx, end_idx))

    return merged


# ── Предобработка ──────────────────────────────────────────────────────────

def _preprocess_split_table_lines(text: str) -> str:
    """Разделить строки, где </table> и контент — на одной строке.

    Пример::

        <045> </table> <046> Окончание табл. 15

    → ::

        <045> </table>
        <046> Окончание табл. 15
    """
    lines = text.split("\n")
    result: list[str] = []
    for line in lines:
        m = re.search(r"</table\s*>", line, re.IGNORECASE)
        if m:
            after = line[m.end():].strip()
            if after:
                result.append(line[:m.end()].rstrip())
                result.append(after)
            else:
                result.append(line)
        else:
            result.append(line)
    return "\n".join(result)


def _preprocess_split_md_row_trailing_text(text: str) -> str:
    """Разделить MD-строку таблицы и хвостовой текст, слитый после последнего ``|``.

    Случай: последняя ячейка таблицы и текст нового абзаца на одной строке
    из-за слияния при переносе страниц::

        <002> | ... | текст | <003> Новый абзац...
        →
        <002> | ... | текст |
        <003> Новый абзац...
    """
    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        no_tags = _strip_page_tags(line).strip()

        # Только строки-кандидаты Markdown-таблицы (начинаются с |)
        if not _MD_ROW_RE.match(no_tags):
            result.append(line)
            continue

        last_pipe = line.rfind("|")
        if last_pipe < 0:
            result.append(line)
            continue

        after_pipe = line[last_pipe + 1:]
        trailing = _strip_page_tags(after_pipe).strip()

        # Разделяем только если после последнего | есть непустой текст,
        # содержащий тег <NNN> — признак слитого абзаца с другой страницы
        if trailing and _PAGE_TAG_RE.search(after_pipe):
            result.append(line[:last_pipe + 1].rstrip())
            result.append(after_pipe.strip())
        else:
            result.append(line)

    return "\n".join(result)


def _preprocess_split_caption_lines(text: str) -> str:
    """Разделить строки, где caption находится в середине, а не в начале.

    Пример::

        <011> текст... 35—37% <012> Таблица 2

    → ::

        <011> текст... 35—37%
        <012> Таблица 2
    """
    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        no_tags = _strip_page_tags(line)

        m_cap = _CAPTION_MID_RE.search(no_tags)
        m_cont = _CONTINUATION_MID_RE.search(no_tags)

        matches = []
        if m_cap:
            matches.append(m_cap)
        if m_cont:
            matches.append(m_cont)

        if not matches:
            result.append(line)
            continue

        m = min(matches, key=lambda x: x.start())

        if m.start() == 0:
            # Caption уже в начале строки — разделение не нужно
            result.append(line)
            continue

        # Маппинг чистых позиций обратно в оригинальные
        orig_pos = 0
        clean_pos = 0
        last_tag_start = -1
        last_tag_end = -1
        split_pos = None

        while orig_pos < len(line):
            tag_match = _PAGE_TAG_RE.match(line, orig_pos)
            if tag_match:
                last_tag_start = tag_match.start()
                last_tag_end = tag_match.end()
                orig_pos = tag_match.end()
                continue

            if clean_pos == m.start():
                if last_tag_start >= 0:
                    between = line[last_tag_end:orig_pos]
                    if between.strip() == "":
                        split_pos = last_tag_start
                    else:
                        split_pos = orig_pos
                else:
                    split_pos = orig_pos
                break

            clean_pos += 1
            orig_pos += 1

        # Дошли до конца строки — caption в самом конце
        if split_pos is None and clean_pos == m.start():
            if last_tag_start >= 0:
                between = line[last_tag_end:]
                if between.strip() == "":
                    split_pos = last_tag_start
                else:
                    split_pos = orig_pos
            else:
                split_pos = orig_pos

        if split_pos and split_pos > 0:
            before = line[:split_pos].rstrip()
            after = line[split_pos:]
            if before and after:
                result.append(before)
                result.append(after)
                continue

        result.append(line)

    return "\n".join(result)


def _preprocess_fix_broken_tags(text: str) -> str:
    """Починить HTML-теги, разорванные переносом строки.

    VLM иногда генерирует ``<captio\\nn>`` вместо ``<caption>``.
    Склеивает разорванные теги: ``<tag_part1\\ntag_part2>`` → ``<tag_part1tag_part2>``.
    """
    text = re.sub(r"(</?\w*)\n(\w*>)", r"\1\2", text)
    return text


def _preprocess_extract_internal_captions(text: str) -> str:
    """Вынести ``<caption>text</caption>`` изнутри ``<table>`` как отдельную строку перед таблицей.

    VLM иногда помещает ``<caption>`` внутри ``<table>`` — стандартный алгоритм
    ищет caption только ПЕРЕД таблицей. Функция извлекает текст caption,
    удаляет тег из тела таблицы и вставляет как отдельную строку перед ``<table>``.
    """
    lines = text.split("\n")
    result: list[str] = []
    i = 0

    while i < len(lines):
        no_tags = _strip_page_tags(lines[i]).strip()

        # Проверяем, что строка содержит <table>
        if re.search(r"<table\b", no_tags, re.IGNORECASE):
            table_line_idx = i
            caption_text = None
            caption_line_idx = None

            # Скан вперёд: ищем <caption>...</caption> внутри таблицы
            j = i
            while j < len(lines):
                m = re.search(
                    r"<caption>(.*?)</caption>",
                    lines[j],
                    re.IGNORECASE | re.DOTALL,
                )
                if m:
                    caption_text = m.group(1).strip()
                    caption_line_idx = j
                    break
                nt = _strip_page_tags(lines[j]).strip()
                if re.search(r"</table\s*>", nt, re.IGNORECASE):
                    break
                j += 1

            if caption_text:
                # Берём <NNN> префикс из строки <table>
                page_m = _PAGE_TAG_RE.search(lines[table_line_idx])
                prefix = (page_m.group(0) + " ") if page_m else ""
                # Вставляем caption перед таблицей
                result.append(f"{prefix}{caption_text}")

            # Добавляем строки от <table> до </table>
            j = table_line_idx
            while j < len(lines):
                if j == caption_line_idx and caption_text is not None:
                    # Удаляем <caption>...</caption> из строки
                    cleaned = re.sub(
                        r"<caption>.*?</caption>",
                        "",
                        lines[j],
                        flags=re.IGNORECASE | re.DOTALL,
                    )
                    remaining = _strip_page_tags(cleaned).strip()
                    if remaining:
                        result.append(cleaned.rstrip())
                else:
                    result.append(lines[j])

                nt = _strip_page_tags(lines[j]).strip()
                if re.search(r"</table\s*>", nt, re.IGNORECASE):
                    j += 1
                    break
                j += 1

            i = j
            continue

        result.append(lines[i])
        i += 1

    return "\n".join(result)


# ── Поиск конца таблицы от caption ────────────────────────────────────────

def _find_table_end(lines: list[str], caption_idx: int) -> int | None:
    """Найти индекс последней строки тела таблицы, начиная от caption.

    Сканирует вперёд от caption_idx + 1, ищет:
      - HTML: ``<table>`` ... ``</table>``
      - Markdown: группу ``|...|`` с разделителем ``|---|``

    Returns:
        Индекс строки конца таблицы или None.
    """
    # Редкий случай: вся таблица на одной строке с caption
    no_tags_cap = _strip_page_tags(lines[caption_idx])
    if re.search(r"<table\b.*?</table\s*>", no_tags_cap, re.IGNORECASE | re.DOTALL):
        return caption_idx

    i = caption_idx + 1
    limit = min(len(lines), caption_idx + 1 + MAX_TABLE_BODY_LOOKAHEAD)

    while i < limit:
        no_tags = _strip_page_tags(lines[i])

        # Пустая строка (или только теги <NNN>) — пропускаем
        if not no_tags:
            i += 1
            continue

        # HTML таблица: нашли <table> — ищем </table>
        if re.search(r"<table\b", no_tags, re.IGNORECASE):
            while i < len(lines):
                nt = _strip_page_tags(lines[i])
                if re.search(r"</table\s*>", nt, re.IGNORECASE):
                    return i
                i += 1
            return None  # Незакрытая таблица

        # Markdown таблица
        if _MD_ROW_RE.match(no_tags):
            group_start = i
            has_sep = False
            while i < len(lines):
                nt = _strip_page_tags(lines[i])
                if not _MD_ROW_RE.match(nt):
                    break
                if _MD_SEPARATOR_RE.match(nt):
                    has_sep = True
                i += 1
            if has_sep and (i - group_start) >= 3:
                return i - 1
            # Не настоящая MD-таблица — продолжаем сканирование
            i = group_start + 1
            continue

        # Текст между caption и таблицей (описание) — пропускаем
        i += 1

    return None


def _find_footnotes_end(lines: list[str], table_end_idx: int) -> int:
    """Найти конец сносок после таблицы.

    Returns:
        Индекс последней строки сноски, или table_end_idx если сносок нет.
    """
    i = table_end_idx + 1
    limit = min(len(lines), table_end_idx + 1 + MAX_FOOTNOTE_LOOKAHEAD)
    footnote_end = table_end_idx
    found = False

    while i < limit:
        no_tags = _strip_page_tags(lines[i])

        # Пустая строка
        if not no_tags:
            i += 1
            if found:
                break  # Пустая строка после сносок — стоп
            continue

        # Сноска
        if _FOOTNOTE_RE.match(no_tags):
            found = True
            footnote_end = i
            i += 1
            continue

        # Не сноска — стоп
        break

    return footnote_end if found else table_end_idx


# ── Основная функция ──────────────────────────────────────────────────────

def normalize_table_boundaries(text: str) -> str:
    """Добавить маркеры [TABLE_START]/[TABLE_END] вокруг таблиц в тексте.

    Якорь — слово «Таблица»/«Табл.» в начале строки (после необязательного
    тега ``<NNN>``). Если слова «Таблица»/«Табл.» нет — маркеры не ставятся.

    Args:
        text: Содержимое Markdown-файла.

    Returns:
        Текст с добавленными маркерами.
    """
    # Удаляем существующие маркеры (для повторного запуска)
    text = text.replace(MARKER_START + "\n", "")
    text = text.replace("\n" + MARKER_END, "")
    text = text.replace(MARKER_START, "")
    text = text.replace(MARKER_END, "")

    # Предобработка
    text = _preprocess_fix_broken_tags(text)
    text = _preprocess_extract_internal_captions(text)
    text = _preprocess_split_table_lines(text)
    text = _preprocess_split_md_row_trailing_text(text)
    text = _preprocess_split_caption_lines(text)

    lines = text.split("\n")

    # Собираем диапазоны (start_line, end_line) — непересекающиеся
    ranges: list[tuple[int, int]] = []

    i = 0
    while i < len(lines):
        if _is_caption_line(lines[i]):
            table_end = _find_table_end(lines, i)
            if table_end is not None:
                final_end = _find_footnotes_end(lines, table_end)
                ranges.append((i, final_end))
                i = final_end + 1  # Перепрыгиваем обработанную таблицу
                continue
        i += 1

    if not ranges:
        return text

    # Объединяем продолжения таблиц (один номер → один блок)
    ranges = _merge_continuation_ranges(ranges, lines)

    # Вставляем маркеры (с конца — чтобы не сбить индексы)
    for start_idx, end_idx in reversed(ranges):
        lines.insert(end_idx + 1, MARKER_END)
        lines.insert(start_idx, MARKER_START)

    result = "\n".join(lines)
    # Убираем возможные дубликаты пустых строк
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    return result


# ── Подсчёт ────────────────────────────────────────────────────────────────

def _count_md_tables_in_text(text: str) -> int:
    """Подсчитать Markdown-таблицы по структуре (группа |...| с |---|)."""
    lines = text.split("\n")
    count = 0
    i = 0
    while i < len(lines):
        no_tags = _strip_page_tags(lines[i].strip())
        if _MD_ROW_RE.match(no_tags):
            has_sep = False
            while i < len(lines):
                nt = _strip_page_tags(lines[i].strip())
                if not _MD_ROW_RE.match(nt):
                    break
                if _MD_SEPARATOR_RE.match(nt):
                    has_sep = True
                i += 1
            if has_sep:
                count += 1
        else:
            i += 1
    return count


def count_tables_in_text(text: str) -> dict:
    """Подсчитать количество таблиц в тексте (по структуре).

    Returns:
        {"html": int, "markdown": int, "total": int}
    """
    html = len(_HTML_TABLE_RE.findall(text))
    md = _count_md_tables_in_text(text)
    return {"html": html, "markdown": md, "total": html + md}


def has_markers(text: str) -> bool:
    """Проверить, содержит ли текст маркеры таблиц."""
    return MARKER_START in text or MARKER_END in text


def count_markers(text: str) -> dict:
    """Подсчитать количество маркеров в тексте.

    Returns:
        {"starts": int, "ends": int}
    """
    return {
        "starts": text.count(MARKER_START),
        "ends": text.count(MARKER_END),
    }