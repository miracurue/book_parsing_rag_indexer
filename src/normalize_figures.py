"""Нормализация границ описаний рисунков в Markdown-файлах глав (chapters/).

Добавляет маркеры [FIGURE_START] и [FIGURE_END] вокруг каждого описания
рисунка, включая:
  - HTML-обёртки <figure>...</figure> (вычищаются, остается только caption)
  - Markdown-изображения ![...](...)
  - Текстовые описания «Рис. N. ...» / «Рисунок N. ...» с продолжениями (позиции 1 — ...; 2 — ...)

Важно: не трогает области внутри [TABLE_START]/[TABLE_END] — таблицы приоритетнее.
Также не ставит FIGURE-маркеры вокруг <figure>, содержащих <table> — убирает только
HTML-обёртку, оставляя таблицу без изменений.

После нормализации extraction работает по маркерам — 100% надёжно.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ── Константы ──────────────────────────────────────────────────────────────

MARKER_START = "[FIGURE_START]"
MARKER_END = "[FIGURE_END]"

# Максимальное количество строк вперёд для поиска продолжения описания
MAX_CONTINUATION_LOOKAHEAD = 15

# Таблицы — уже промаркированы
TABLE_START = "[TABLE_START]"
TABLE_END = "[TABLE_END]"


# ── Regex-шаблоны ──────────────────────────────────────────────────────────

# Тег номера страницы <NNN>
_PAGE_TAG_RE = re.compile(r"<(\d+)>")

# HTML <figure>...</figure> блок
_FIGURE_BLOCK_RE = re.compile(
    r"<figure\b[^>]*>.*?</figure>",
    re.DOTALL | re.IGNORECASE,
)

# HTML <figcaption>...</figcaption>
_FIGCAPTION_RE = re.compile(
    r"<figcaption>(.*?)</figcaption>",
    re.DOTALL | re.IGNORECASE,
)

# HTML <img ...>
_IMG_TAG_RE = re.compile(r"<img\b[^>]*/?>", re.DOTALL | re.IGNORECASE)

# Markdown-изображение: ![alt](src)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")

# Тег <figure>
_FIGURE_OPEN_RE = re.compile(r"<figure\b[^>]*>", re.IGNORECASE)
_FIGURE_CLOSE_RE = re.compile(r"</figure>", re.IGNORECASE)

# Описание рисунка: «Рис. N. ...», «Рисунок N. ...»
# N может быть составным: 1.3, 1.10, 90 и т.д.
_FIGURE_CAPTION_RE = re.compile(
    r"Рис(?:унок|\.)\s+\d+[\.\d]*\s",
)

# Строка продолжения описания (позиции): «1 — ...; 2 — ...»
_CONTINUATION_RE = re.compile(
    r"^\d+\s*[—–\-]\s",
)

# Inline-описание: <NNN> Рис. N в середине строки (после значимого текста)
_INLINE_CAPTION_RE = re.compile(
    r"<\d+>\s*Рис(?:унок|\.)\s+\d+[\.\d]*\s",
)

# Пустая строка после удаления тегов <NNN>
_EMPTY_LINE_RE = re.compile(r"^(<\d+>\s*)+$")

# Кирилическая строчная буква в начале строки (после очистки тегов)
_LOWERCASE_START_RE = re.compile(r"^[а-яё]")


# ── Вспомогательные функции ────────────────────────────────────────────────

def _strip_page_tags(text: str) -> str:
    """Удалить теги <NNN> из строки."""
    return _PAGE_TAG_RE.sub("", text).strip()


def _is_empty_line(line: str) -> bool:
    """Проверить, что строка пустая или содержит только теги <NNN>."""
    stripped = line.strip()
    if not stripped:
        return True
    return bool(_EMPTY_LINE_RE.match(stripped))


def _is_enumeration_continuation(text: str) -> bool:
    """Проверить, что строка похожа на продолжение перечисления.

    Используется для многострочных описаний (caption заканчивается на «:»).
    Строка считается продолжением, если:
      - содержит разделитель «—» / «–» (ключ — значение)
      - начинается со строчной буквы (не новое предложение)
      - заканчивается на «;» или «:» (перечисление продолжается)

    NOT a continuation:
      - маркированный список «- item;» (bullet list — не часть описания)
      - новое предложение с заглавной буквы
    """
    if not text:
        return False

    # Маркированный список «- text» — НЕ продолжение описания рисунка
    if text.startswith("- "):
        return False

    # Содержит em-dash / en-dash — ключ — значение
    if "—" in text or "–" in text:
        return True

    # Заканчивается на разделитель перечисления
    if text.endswith(";") or text.endswith(":"):
        return True

    # Начинается со строчной буквы — продолжение предложения
    if _LOWERCASE_START_RE.match(text):
        return True

    return False


def _remove_standalone_md_images(text: str) -> str:
    """Удалить строки, состоящие только из markdown-изображения ![alt](src).

    VLM-парсинг иногда оставляет строки вида:
      <162> ![Схема печи](image.png)
    Это артефакт — ссылка на несуществующий файл. Удаляем такие строки целиком,
    чтобы они не ломали определение границ описаний «Рис. N. ...».

    Строки с другим текстом рядом с md-image НЕ удаляются:
      «См. ![схема](a.png) далее» → оставляем
    """
    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        no_tags = _strip_page_tags(line)
        # Если после удаления тегов остался только md-image (и ничего больше)
        remaining = _MD_IMAGE_RE.sub("", no_tags).strip()
        if _MD_IMAGE_RE.search(no_tags) and not remaining:
            # Строка состоит только из <NNN> тегов + md-image → удалить
            continue
        result.append(line)

    return "\n".join(result)


def _get_table_zones(text: str) -> list[tuple[int, int]]:
    """Получить зоны таблиц по маркерам [TABLE_START]...[TABLE_END].

    Returns:
        [(start, end), ...] — позиции зон таблиц.
    """
    zones = []
    search_start = 0
    while True:
        ts = text.find(TABLE_START, search_start)
        if ts == -1:
            break
        te = text.find(TABLE_END, ts)
        if te == -1:
            break
        zones.append((ts, te + len(TABLE_END)))
        search_start = te + len(TABLE_END)
    return zones


def _is_inside_table(pos: int, table_zones: list[tuple[int, int]]) -> bool:
    """Проверить, находится ли позиция внутри зоны таблицы."""
    for ts, te in table_zones:
        if ts <= pos < te:
            return True
    return False


# ── Шаг 1: Обработка HTML <figure> блоков ──────────────────────────────────

def _process_figure_blocks(text: str, table_zones: list[tuple[int, int]]) -> str:
    """Обработать <figure>...</figure> блоки.

    - Если внутри <figure> есть <table>: убрать обёртку <figure>,
      вынести <figcaption> как plain text. FIGURE-маркеры НЕ ставятся.
    - Если внутри <figure> нет <table>: заменить на [FIGURE_START]\ncaption\n[FIGURE_END],
      убрать <img>, <figcaption> теги.
    """
    result = text

    # Ищем все <figure> блоки от конца к началу (чтобы не сбить индексы)
    matches = list(_FIGURE_BLOCK_RE.finditer(result))
    for m in reversed(matches):
        fig_block = m.group(0)
        fig_start = m.start()
        fig_end = m.end()

        # Проверяем, внутри ли зоны таблицы
        if _is_inside_table(fig_start, table_zones):
            continue

        # Извлекаем caption
        caption_match = _FIGCAPTION_RE.search(fig_block)
        caption = _strip_page_tags(caption_match.group(1).strip()) if caption_match else ""

        # Проверяем, есть ли <table> внутри
        has_table = bool(re.search(r"<table\b", fig_block, re.IGNORECASE))

        if has_table:
            # Таблица внутри figure — убираем обёртку, выносим caption
            inner = fig_block

            # Убираем <figure ...> и </figure>
            inner = _FIGURE_OPEN_RE.sub("", inner)
            inner = _FIGURE_CLOSE_RE.sub("", inner)

            # Убираем <figcaption>...</figcaption> (тег целиком)
            inner = _FIGCAPTION_RE.sub("", inner)

            # Убираем <img ...> теги
            inner = _IMG_TAG_RE.sub("", inner)

            # Добавляем caption перед содержимым как plain text
            if caption:
                # Убираем ведущие/конечные пустые строки
                inner = inner.strip()
                replacement = f"{caption}\n{inner}"
            else:
                replacement = inner.strip()

            result = result[:fig_start] + replacement + result[fig_end:]
        else:
            # Нет таблицы — это описание рисунка
            # Формируем блок с FIGURE-маркерами
            parts = []
            if caption:
                parts.append(caption)

            # Убираем все HTML-теги из контента
            content = fig_block
            content = _FIGURE_OPEN_RE.sub("", content)
            content = _FIGURE_CLOSE_RE.sub("", content)
            content = _FIGCAPTION_RE.sub("", content)
            content = _IMG_TAG_RE.sub("", content)
            content = _MD_IMAGE_RE.sub("", content)
            content = _strip_page_tags(content)
            if content and content != caption:
                parts.append(content)

            if parts:
                replacement = f"{MARKER_START}\n" + "\n".join(parts) + f"\n{MARKER_END}"
            else:
                replacement = ""

            result = result[:fig_start] + replacement + result[fig_end:]

    return result


# ── Шаг 2: Обработка plain-text описаний ───────────────────────────────────

def _find_caption_start_backward(lines: list[str], caption_idx: int) -> int:
    """Найти начало описания рисунка, двигаясь назад от найденного «Рис. N.».

    Проверяет, есть ли markdown-изображение ![...](...) или <img> перед caption.
    Возвращает индекс строки начала описания.
    """
    start_idx = caption_idx

    # Ищем markdown-изображение или <img> над caption (1-2 строки)
    for idx in range(caption_idx - 1, max(caption_idx - 3, -1), -1):
        line = lines[idx]
        no_tags = _strip_page_tags(line)

        if _is_empty_line(line):
            break

        # Markdown-изображение
        if _MD_IMAGE_RE.search(no_tags) or _IMG_TAG_RE.search(no_tags):
            start_idx = idx
            break

        # Другой значимый текст — не включаем
        break

    return start_idx


def _find_figure_end(lines: list[str], start_idx: int, num_lines: int) -> int:
    """Найти конец описания рисунка, двигаясь вперёд от caption.

    Включает continuation lines:
      - нумерованные позиции «1 — ...; 2 — ...»
      - многострочные описания: caption заканчивается на «:», далее строки
        через «;» (феррит — ...; аустенит — ...; цементит — ...)

    Возвращает индекс строки после последней строки описания.
    """
    end_idx = start_idx + 1  # минимум caption-строка

    # Определяем, является ли описание многострочным (caption заканчивается на «:»)
    caption_clean = _strip_page_tags(lines[start_idx]).rstrip()
    multi_line = caption_clean.endswith(":")

    for idx in range(start_idx + 1, min(start_idx + 1 + MAX_CONTINUATION_LOOKAHEAD, num_lines)):
        line = lines[idx]
        no_tags = _strip_page_tags(line)

        # Пустая строка — стоп
        if _is_empty_line(line):
            break

        # Попали в зону таблицы — стоп
        if TABLE_START in line or TABLE_END in line:
            break

        # Уже есть маркер — стоп
        if MARKER_START in line or MARKER_END in line:
            break

        # Следующее описание рисунка — стоп
        if _FIGURE_CAPTION_RE.search(no_tags):
            break

        # Заголовок — стоп
        if no_tags.startswith("#"):
            break

        # Строка-продолжение (позиции: «1 — ...; 2 — ...»)
        if _CONTINUATION_RE.match(no_tags):
            end_idx = idx + 1
            continue

        # Многострочное описание: включаем строку, если предыдущая
        # заканчивалась на «:» или «;» И текущая строка похожа на
        # продолжение перечисления (а не новое предложение).
        if multi_line:
            prev_clean = _strip_page_tags(lines[idx - 1]).rstrip()
            if prev_clean.endswith(":") or prev_clean.endswith(";"):
                cur_clean = no_tags.rstrip()
                # СНАЧАЛА проверяем, похожа ли строка на продолжение
                if _is_enumeration_continuation(cur_clean):
                    end_idx = idx + 1
                    # Если текущая строка тоже заканчивается на «:» или «;» —
                    # описание продолжается дальше
                    if not (cur_clean.endswith(":") or cur_clean.endswith(";")):
                        break
                    continue
                # Строка похожа на новое предложение — не включаем
            break

        # Значимый текст (не continuation) — стоп
        break

    return end_idx


def _split_inline_page_tags(text: str) -> str:
    """Разделить строки по встроенным тегам <NNN> в середине.

    При склейке глав текст с разных страниц может оказаться на одной строке:
      «<173> текст продолжается <174> Начало нового предложения»
    После разделения:
      «<173> текст продолжается»
      «<174> Начало нового предложения»

    Regex: ноль или более пробелов между непробельным символом и <NNN> тегом.
    Не трогает <NNN> в начале строки (там нет текста перед тегом).
    """
    lines = text.split("\n")
    result: list[str] = []
    split_re = re.compile(r"(?<=\S)\s*(?=<\d+>)")

    for line in lines:
        # Проверяем, есть ли <NNN> не в начале строки
        parts = split_re.split(line)
        if len(parts) > 1:
            result.extend(parts)
        else:
            result.append(line)

    return "\n".join(result)


def _split_inline_captions(text: str) -> str:
    """Разделить строки, где «<NNN> Рис. N» находится в середине.

    Такое бывает при разрыве страницы внутри абзаца:
      «<125> ...основная масса <126> Рис. 29. Схема...»
    После разделения:
      «<125> ...основная масса»
      «<126> Рис. 29. Схема...»
    """
    lines = text.split("\n")
    result_lines: list[str] = []

    for line in lines:
        # Ищем все вхождения inline-паттерна
        m = _INLINE_CAPTION_RE.search(line)
        if not m:
            result_lines.append(line)
            continue

        # Проверяем, что перед совпадением есть значимый текст (не только теги)
        before = line[:m.start()]
        before_no_tags = _PAGE_TAG_RE.sub("", before).strip()

        if before_no_tags:
            # Разделяем: текст до → отдельная строка, <NNN> Рис. N... → новая строка
            result_lines.append(before.rstrip())
            result_lines.append(line[m.start():])
        else:
            # Рис. N в начале строки (после тегов) — не разделяем
            result_lines.append(line)

    return "\n".join(result_lines)


def _process_plain_figures(text: str, table_zones: list[tuple[int, int]]) -> str:
    """Найти и обернуть в маркеры plain-text описания «Рис. N. ...»."""
    lines = text.split("\n")
    num_lines = len(lines)

    # Предрасчитываем позиции строк один раз (O(n))
    line_pos = _build_line_positions(lines)

    # Собираем диапазоны для маркировки (от начала к концу)
    spans: list[tuple[int, int]] = []

    idx = 0
    while idx < num_lines:
        line = lines[idx]
        no_tags = _strip_page_tags(line)

        # Ищем «Рис. N. ...» в строке
        if not _FIGURE_CAPTION_RE.search(no_tags):
            idx += 1
            continue

        # Внутри зоны таблицы — пропускаем
        if _is_inside_table(line_pos[idx], table_zones):
            idx += 1
            continue

        # Уже есть маркер FIGURE_START на этой или предыдущей строке — пропускаем
        if idx > 0 and MARKER_START in lines[idx - 1]:
            idx += 1
            continue
        if MARKER_START in line or MARKER_END in line:
            idx += 1
            continue

        # Определяем начало описания (может быть md-image выше)
        start_idx = _find_caption_start_backward(lines, idx)

        # Проверяем, что начало тоже не в таблице
        if start_idx < idx:
            if _is_inside_table(line_pos[start_idx], table_zones):
                idx += 1
                continue

        # Определяем конец описания
        end_idx = _find_figure_end(lines, start_idx, num_lines)

        # Описание должно быть хотя бы 1 строка
        if end_idx <= start_idx:
            idx += 1
            continue

        spans.append((start_idx, end_idx))
        # Гарантия прогресса вперёд — если end_idx <= idx, цикл зависнет
        idx = max(end_idx, idx + 1)

    if not spans:
        return text

    # Вставляем маркеры (от конца к началу, чтобы не сбить индексы)
    for start_idx, end_idx in reversed(spans):
        lines.insert(end_idx, MARKER_END)
        lines.insert(start_idx, MARKER_START)

    return "\n".join(lines)


# ── Основная функция ───────────────────────────────────────────────────────

def normalize_figure_boundaries(text: str) -> str:
    """Добавить маркеры [FIGURE_START]/[FIGURE_END] вокруг описаний рисунков.

    Алгоритм:
    1. Удалить существующие FIGURE-маркеры (для повторного запуска)
    2. Определить зоны таблиц ([TABLE_START]...[TABLE_END])
    3. Обработать HTML <figure> блоки
    4. Найти и обернуть plain-text описания «Рис. N. ...» / «Рисунок N. ...»

    Args:
        text: Содержимое Markdown-файла.

    Returns:
        Текст с добавленными маркерами.
    """
    # 1. Удаляем существующие маркеры (для повторного запуска)
    text = text.replace(MARKER_START + "\n", "")
    text = text.replace("\n" + MARKER_END, "")
    text = text.replace(MARKER_START, "")
    text = text.replace(MARKER_END, "")

    # 1.5 Разделяем строки по встроенным <NNN> тегам (склейка глав)
    text = _split_inline_page_tags(text)

    # 2. Определяем зоны таблиц
    table_zones = _get_table_zones(text)

    # 3. Обрабатываем <figure> блоки
    text = _process_figure_blocks(text, table_zones)

    # Переопределяем зоны таблиц (могли сместиться после обработки <figure>)
    table_zones = _get_table_zones(text)

    # 4. Разделяем inline-описания (Рис. N в середине строки)
    text = _split_inline_captions(text)

    # Переопределяем зоны таблиц (смещение после разделения строк)
    table_zones = _get_table_zones(text)

    # 5. Удаляем standalone md-image строки (VLM-артефакты)
    text = _remove_standalone_md_images(text)

    # Переопределяем зоны таблиц (смещение после удаления строк)
    table_zones = _get_table_zones(text)

    # 6. Обрабатываем plain-text описания
    text = _process_plain_figures(text, table_zones)

    # Убираем возможные дубликаты пустых строк
    text = re.sub(r"\n{4,}", "\n\n\n", text)

    return text


# ── Подсчёт ────────────────────────────────────────────────────────────────

def _build_line_positions(lines: list[str]) -> list[int]:
    """Предрасчитать позиции начала каждой строки (O(n) вместо O(n²))."""
    positions = [0] * len(lines)
    pos = 0
    for i, line in enumerate(lines):
        positions[i] = pos
        pos += len(line) + 1  # +1 за \n
    return positions


def count_figures_in_text(text: str) -> dict:
    """Подсчитать количество описаний рисунков в тексте.

    Returns:
        {
            "figure_tags": int,      # <figure> блоков
            "plain_captions": int,   # «Рис. N.» строк
            "total": int,            # всего
            "markers_start": int,    # уже установленных [FIGURE_START]
            "markers_end": int,      # уже установленных [FIGURE_END]
        }
    """
    figure_tags = len(list(_FIGURE_BLOCK_RE.finditer(text)))

    # Считаем «Рис. N.» строки (вне TABLE зон)
    plain = 0
    table_zones = _get_table_zones(text)
    lines = text.split("\n")
    line_pos = _build_line_positions(lines)
    for i, line in enumerate(lines):
        no_tags = _strip_page_tags(line)
        if _FIGURE_CAPTION_RE.search(no_tags):
            if not _is_inside_table(line_pos[i], table_zones):
                plain += 1

    markers_start = text.count(MARKER_START)
    markers_end = text.count(MARKER_END)

    return {
        "figure_tags": figure_tags,
        "plain_captions": plain,
        "total": figure_tags + plain,
        "markers_start": markers_start,
        "markers_end": markers_end,
    }


def has_markers(text: str) -> bool:
    """Проверить, содержит ли текст маркеры рисунков."""
    return MARKER_START in text or MARKER_END in text


def count_markers(text: str) -> dict:
    """Подсчитать количество маркеров рисунков.

    Returns:
        {"starts": int, "ends": int}
    """
    return {
        "starts": text.count(MARKER_START),
        "ends": text.count(MARKER_END),
    }