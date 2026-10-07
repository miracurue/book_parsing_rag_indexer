"""Соединение пронумерованных Markdown-файлов по главам.

Чистые функции — используются оркестратором run_chapter_merger.py.

Логика:
1. Сбор всех блоков из .md файлов (разделение по \\n\\n+)
2. Склейка разорванных блоков — текст, перенесённый через границу страниц,
   объединяется с учётом тегов <N> и дефисов
3. Разделение на главы по заголовкам (# ...)
4. Сохранение каждой главы в отдельный файл
"""

from __future__ import annotations

import re

try:
    import pymorphy3
    _morph = pymorphy3.MorphAnalyzer()
except ImportError:
    _morph = None


# ══════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════

def is_text_block(block: str) -> bool:
    """Определяет, является ли блок обычным текстом.

    Не текст: таблицы (|), заголовки (#), картинки (![), формулы ($$).
    """
    b = block.strip()
    if b.startswith("|"):
        return False
    if b.startswith("#"):
        return False
    if b.startswith("!["):
        return False
    if b.startswith("<img"):
        return False
    if b.startswith("$$"):
        return False
    return True


def is_continuation(prev_block: str, curr_text: str) -> bool:
    """Проверяет, является ли текущий текст продолжением предыдущего блока."""
    if not prev_block or not curr_text:
        return False

    # Заголовок — жёсткий разделитель, не может быть продолжением
    if curr_text.strip().startswith("#"):
        return False

    clean_prev = re.sub(r"[*_]+$", "", prev_block.strip())

    # Строка заканчивается дефисом — перенос
    if clean_prev.endswith("-"):
        return True

    # Нет завершающего знака препинания — вероятно, перенос
    terminators = (".", "!", "?", "»", '"', ":", ";")
    if not clean_prev.endswith(terminators):
        return True

    return False


def smart_merge_blocks(prev_text: str, marker: str, curr_text: str) -> str:
    """Умное слияние блоков с учётом переносов слов через дефис."""
    clean_prev = prev_text.strip()
    clean_curr = curr_text.strip()

    if not clean_prev.endswith("-"):
        return f"{clean_prev} {marker} {clean_curr}"

    # Обработка дефиса: перенос или сложное слово?
    match_prev = re.search(r"([а-яА-ЯёЁa-zA-Z]+)-$", clean_prev)
    match_curr = re.match(r"^([а-яА-ЯёЁa-zA-Z]+)", clean_curr)

    keep_hyphen = False
    if match_prev and match_curr:
        part1, part2 = match_prev.group(1).lower(), match_curr.group(1).lower()
        particles = {"то", "либо", "нибудь", "таки", "ка", "де"}

        if part2 in particles:
            keep_hyphen = True
        elif _morph:
            whole_word = part1 + part2
            compound_word = f"{part1}-{part2}"
            if _morph.word_is_known(whole_word) and not _morph.word_is_known(compound_word):
                keep_hyphen = False
            elif _morph.word_is_known(compound_word):
                keep_hyphen = True
            elif _morph.word_is_known(part1) and _morph.word_is_known(part2):
                keep_hyphen = True

    if keep_hyphen:
        return f"{clean_prev}{marker}{clean_curr}"
    else:
        return f"{clean_prev[:-1]}{marker}{clean_curr}"


# ══════════════════════════════════════════════════════════════════════
# Основные функции пайплайна
# ══════════════════════════════════════════════════════════════════════

def merge_blocks_from_pages(pages: dict[str, str]) -> list[str]:
    """Собрать и склеить блоки из всех страниц.

    Args:
        pages: словарь {filename: content}, отсортированный по имени файла.
               Ключи — имена файлов (напр. '005.md'), значения — текст.

    Returns:
        Список склеенных блоков.
    """
    # Шаг 1: собираем все блоки из всех файлов (по порядку имён)
    all_blocks: list[str] = []
    for filename in sorted(pages.keys()):
        content = pages[filename]
        blocks = re.split(r"\n\n+", content)
        all_blocks.extend(b.strip() for b in blocks if b.strip())

    # Шаг 2: склейка разорванных блоков
    processed_blocks: list[str] = []
    last_text_idx = -1
    marker_pattern = re.compile(r"^(\s*<\d+>\s*)(.*)", re.DOTALL)

    for block in all_blocks:
        match = marker_pattern.match(block)
        if match and last_text_idx != -1:
            marker = match.group(1).strip()
            curr_text = match.group(2).strip()

            if is_continuation(processed_blocks[last_text_idx], curr_text):
                processed_blocks[last_text_idx] = smart_merge_blocks(
                    processed_blocks[last_text_idx], marker, curr_text
                )
                continue

        processed_blocks.append(block)

        # Обновляем индекс последнего текстового блока
        if block.startswith("#"):
            last_text_idx = -1  # Заголовок — жёсткий разделитель
        elif is_text_block(block):
            last_text_idx = len(processed_blocks) - 1

    return processed_blocks


def _heading_level(block: str) -> int | None:
    """Вернуть уровень заголовка (1 для #, 2 для ##, ...) или None."""
    m = re.match(r"^(#{1,6})\s+", block)
    return len(m.group(1)) if m else None


def find_deepest_heading_level(blocks: list[str]) -> int:
    """Найти самый глубокий уровень заголовка в блокам.

    Returns:
        Максимальный уровень (1-6). Если заголовков нет — возвращает 1.
    """
    max_level = 0
    for block in blocks:
        level = _heading_level(block)
        if level is not None and level > max_level:
            max_level = level
    return max_level if max_level > 0 else 1


def _split_heading_block(block: str) -> tuple[str, str | None]:
    """Отделить строки-заголовки от остального контента в блоке.

    Блок после _split_compound_blocks() может содержать заголовок
    на первой строке и текст на последующих (одиночные \\n).

    Returns:
        (heading_line, content_or_None) — заголовок и остаток блока.
    """
    lines = block.split('\n')
    heading_lines: list[str] = []
    content_lines: list[str] = []

    phase = "heading"  # сначала собираем заголовки
    for line in lines:
        stripped = line.strip()
        if phase == "heading" and stripped.startswith('#'):
            heading_lines.append(stripped)
        else:
            phase = "content"
            content_lines.append(line)

    heading = '\n'.join(heading_lines)
    content = '\n'.join(content_lines).strip() if content_lines else None
    return heading, content


def _clean_heading_text(block: str, max_len: int = 60) -> str:
    """Извлечь текст заголовка и очистить для имени файла."""
    text = re.sub(r"^#+\s+", "", block).strip()
    text = text.split("\n", 1)[0].strip()
    return re.sub(r'[\\/*?:"<>|\r\n\t]', "", text)[:max_len]


def _split_compound_blocks(blocks: list[str]) -> list[str]:
    """Разбить блоки, содержащие несколько строк-заголовков.

    Если внутри блока есть строки, начинающиеся с '#', отделённые
    только одиночными '\\n' (без '\\n\\n'), такой блок делится
    на подблоки по границам заголовков.
    """
    result: list[str] = []
    for block in blocks:
        lines = block.split('\n')
        # Собираем подблоки: новый подблок начинается с каждой '#'-строки,
        # за которой идёт пустая строка или конец блока
        sub_lines: list[str] = []
        for line in lines:
            if line.strip().startswith('#') and sub_lines:
                # Начался новый заголовок — сбрасываем накопленное
                result.append('\n'.join(sub_lines).strip())
                sub_lines = [line]
            else:
                sub_lines.append(line)
        if sub_lines:
            result.append('\n'.join(sub_lines).strip())
    return result


def _has_text_content(blocks: list[str]) -> bool:
    """Проверить, есть ли в списке блоков хотя бы одна текстовая строка.
    
    Проверяет каждую строку внутри блока, т.к. блок может содержать
    заголовок на первой строке и текст на последующих (через одиночный \\n).
    """
    for b in blocks:
        for line in b.split('\n'):
            s = line.strip()
            if s and not s.startswith('#'):
                return True
    return False


def split_into_chapters(
    blocks: list[str],
    split_level: int = 0,
) -> list[tuple[str, list[str]]]:
    """Разделить блоки на главы по заголовкам заданного уровня.

    Правила:
    - **split_level=0 (авто)**: определяется самый глубокий уровень
      заголовков в тексте, разделение происходит по нему.
    - Разделение происходит по каждому заголовку уровня == split_level.
    - Заголовки уровней < split_level отслеживаются как «родительские»
      и дублируются в начало каждого файла.
    - **Пустые главы пропускаются**: если после заголовка нет текстового
      контента до следующего заголовка того же уровня — файл не создаётся.
    - Заголовки уровней > split_level — маркеры ``#`` удаляются, текст
      попадает в текущую главу как обычный контент.
    - Контент до первого заголовка → файл «00_Введение».

    Args:
        blocks: список склеенных текстовых блоков.
        split_level: уровень заголовков для разделения (1=#, 2=##, ...).
                     0 = авто (самый глубокий уровень в тексте).

    Returns:
        Список кортежей (filename_title, list[block]).
    """
    # Авто-определение самого глубокого уровня
    if split_level == 0:
        split_level = find_deepest_heading_level(blocks)

    chapters: list[tuple[str, list[str]]] = []
    parent_headers: dict[int, str] = {}       # level -> block (< split_level)
    current_blocks: list[str] = []            # накопленные блоки текущей главы
    chapter_counter = 0
    current_split_header: str | None = None   # последний заголовок == split_level

    def _flush_chapter():
        """Сохранить текущую накопленную главу."""
        nonlocal chapter_counter, current_blocks, current_split_header
        if not current_blocks and current_split_header is None:
            return

        # Проверка: есть ли текстовый контент (не только заголовки)
        # Разрешаем создавать файл, если есть заголовок split_level + любой контент,
        # либо если есть только контент (Введение без заголовка)
        has_real_content = _has_text_content(current_blocks)
        if not has_real_content:
            # Нет текста — пропускаем, не создаём пустой файл.
            # split-level заголовки НЕ добавляем в parent_headers —
            # это siblings, не parents (иначе они попадут в начало следующего файла).
            current_blocks = []
            current_split_header = None
            return

        chapter_counter += 1
        # Содержимое: родительские заголовки + накопленный контент
        built: list[str] = []
        for lvl in sorted(parent_headers.keys()):
            built.append(parent_headers[lvl])
        built.extend(current_blocks)
        # Имя файла: из split-заголовка, либо из самого глубокого родителя
        if current_split_header:
            name = _clean_heading_text(current_split_header)
        elif parent_headers:
            deepest = max(parent_headers.keys())
            name = _clean_heading_text(parent_headers[deepest])
        else:
            name = "Введение"
        chapters.append((f"{chapter_counter:02d}_{name}", built))
        current_blocks = []
        current_split_header = None

    for block in blocks:
        level = _heading_level(block)

        # --- Обычный контент ---
        if level is None:
            current_blocks.append(block)
            continue

        # --- Заголовок уровня разделения ---
        if level == split_level:
            _flush_chapter()
            current_split_header = block
            current_blocks.append(block)

        # --- Родительский заголовок (level < split_level) ---
        elif level < split_level:
            # Если есть накопленный контент — сохранить в отдельный файл
            if current_blocks or current_split_header is not None:
                _flush_chapter()
            # Обновить родительские: убрать уровни >= текущего
            parent_headers = {k: v for k, v in parent_headers.items() if k < level}
            # Отделить заголовок от контента (может быть в одном блоке)
            heading_only, after_content = _split_heading_block(block)
            parent_headers[level] = heading_only
            # Контент после заголовка — в текущий накопитель
            if after_content:
                current_blocks.append(after_content)

        # --- Подзаголовок (level > split_level) → очистить # ---
        else:
            cleaned = re.sub(r'^#{1,6}\s+', '', block, count=1)
            current_blocks.append(cleaned)

    # Финальный сброс
    if current_blocks or current_split_header is not None:
        _flush_chapter()

    return chapters
