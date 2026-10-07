"""Выравнивание заголовков по оглавлению (TOC).

Алгоритм:
1. Парсинг TOC → структура с тремя типами якорей:
   - секционный номер:  (1, 2, 3) → (text, level)
   - специальный ключ:  ("раздел", 1) → (text, level)  для «Раздел I», «Глава 2»
   - текстовые записи:  [(normalized, text, level), ...]  для безномерных
2. Сканирование parsed/*.md — поиск заголовков (с # или голых)
3. Каскад стратегий матчинга: точный номер → родительский → спец.ключ → текст
4. Применение: найден в TOC → проставить правильный level; не найден → убрать #
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ======================================================================
# Константы
# ======================================================================

# Строка-заголовок с # маркерами (опционально с <NNN> перед)
_HEADING_RE = re.compile(
    r"^(?:<\d+>\s*)*(#{1,6})\s+(.+)$"
)

# «Голый» секционный заголовок: «1.2.3 Title» или «0.1. Title»
_BARE_SECTION_RE = re.compile(
    r"^(?:<\d+>\s*)*(\d+(?:\.\d+)+)[.\s]+(.+)$"
)

# Специальный заголовок: «Раздел I», «Глава 2», «Часть 3»
_SPECIAL_RE = re.compile(
    r"^(?:<\d+>\s*)*(?:#{1,6}\s+)?"
    r"(Раздел|Глава|Часть|раздел|глава|часть)\s+"
    r"([IVXLCDM]+|\d+)[.\s]*(.*)$",
    re.IGNORECASE,
)

# §-заголовок: «§ 1.1 Title»
_PARA_RE = re.compile(
    r"^(?:<\d+>\s*)*(?:#{1,6}\s+)?§\s*(\d+(?:\.\d+)+)[.\s]+(.+)$"
)

# Blacklist — строки, которые точно не заголовки
_BLACKLIST_RE = re.compile(
    r"^(?:<\d+>\s*)*(?:#{1,6}\s+)?"
    r"(?:Рис\.|рис\.|Таблица|таблица|Табл\.|табл\.|Окончание\s+табл\.|Продолжение\s+табл\.)",
    re.IGNORECASE,
)

# Для очистки trailing точек из номера: «1.2.» → «1.2»
_TRAIL_DOT_RE = re.compile(r"\.+$")

# Римские цифры → int
_ROMAN_MAP = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


# ======================================================================
# Структуры данных
# ======================================================================

@dataclass
class TocEntry:
    """Одна запись оглавления."""
    raw_text: str          # оригинальный текст (без #)
    level: int             # количество # (1–6)
    section_key: Optional[tuple[int, ...]] = None  # (1, 2, 3) или None
    special_key: Optional[tuple[str, int]] = None   # ("раздел", 1) или None
    normalized: str = ""   # для текстового матчинга


@dataclass
class TocIndex:
    """Индекс оглавления — быстрый lookup по разным ключам."""
    section_map: dict[tuple[int, ...], TocEntry] = field(default_factory=dict)
    special_map: dict[tuple[str, int], TocEntry] = field(default_factory=dict)
    text_entries: list[TocEntry] = field(default_factory=list)  # без номера/спец.ключа
    all_entries: list[TocEntry] = field(default_factory=list)   # все (для статистики)


# ======================================================================
# Вспомогательные функции
# ======================================================================

def _roman_to_int(s: str) -> int:
    """Конвертировать римскую цифру в int. 'XIV' → 14."""
    s = s.upper().strip()
    total = 0
    prev = 0
    for ch in reversed(s):
        val = _ROMAN_MAP.get(ch, 0)
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total


def _normalize(text: str) -> str:
    """Нормализация текста для сравнения: lower, удалить пунктуацию, схлопнуть пробелы."""
    text = text.lower().strip()
    # Удалить §
    text = re.sub(r"§\s*", "", text)
    # Удалить trailing точки из секционных номеров
    text = re.sub(r"\.{2,}", " ", text)
    # Удалить пунктуацию (кроме цифр и букв)
    text = re.sub(r"[^\w\sа-яёА-ЯЁa-zA-Z0-9]", " ", text)
    # Схлопнуть пробелы
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_with_digits(text: str) -> str:
    """Нормализация с сохранением цифр для секционного матчинга."""
    return _normalize(text)


def _extract_section_key(text: str) -> Optional[tuple[int, ...]]:
    """Извлечь секционный номер из начала текста.
    
    '1.2.3 Title' → (1, 2, 3)
    '0.1. Цель агломерации' → (0, 1)
    '2.10. Материальный баланс' → (2, 10)
    '0. Введение' → (0,)
    '5 Раздел' → None (одиночная цифра без точки)
    """
    text = text.strip()
    # Сначала пробуем составной номер: X.Y.Z
    m = re.match(r"^(\d+(?:\.\d+)+)[.\s]", text)
    if m:
        num_str = m.group(1)
        parts = num_str.split(".")
        try:
            return tuple(int(p) for p in parts)
        except ValueError:
            return None
    # Затем одиночный номер с точкой: «0. Введение»
    m2 = re.match(r"^(\d+)\.\s", text)
    if m2:
        try:
            return (int(m2.group(1)),)
        except ValueError:
            return None
    return None


def _extract_special_key(text: str) -> Optional[tuple[str, int]]:
    """Извлечь специальный ключ: «Раздел I» → («раздел», 1).
    
    Работает с '# Раздел 1 ...', 'Раздел I. ...', 'Глава 2 ...'
    """
    m = re.match(
        r"^(?:#{1,6}\s+)?(Раздел|Глава|Часть|раздел|глава|часть)\s+([IVXLCDM]+|\d+)",
        text.strip(),
    )
    if not m:
        return None
    keyword = m.group(1).lower()
    raw_num = m.group(2)
    if raw_num.upper() in _ROMAN_MAP or all(c.upper() in _ROMAN_MAP for c in raw_num):
        num = _roman_to_int(raw_num)
    else:
        num = int(raw_num)
    return (keyword, num)


def _token_overlap(s1: str, s2: str) -> float:
    """Токенное перекрытие двух строк. Доля токенов короткой строки в длинной."""
    tokens1 = set(s1.split())
    tokens2 = set(s2.split())
    if not tokens1 or not tokens2:
        return 0.0
    shorter = tokens1 if len(tokens1) <= len(tokens2) else tokens2
    longer = tokens2 if len(tokens1) <= len(tokens2) else tokens1
    overlap = shorter & longer
    return len(overlap) / len(shorter)


# ======================================================================
# Парсинг TOC
# ======================================================================

def parse_toc_files(toc_dir: Path) -> list[tuple[str, int]]:
    """Парсинг файлов оглавления.
    
    Совместимый с оркестратором интерфейс: возвращает list[(text, level)].
    Также строит внутренний TocIndex для быстрого lookup.
    """
    index = _build_toc_index(toc_dir)
    # Возвращаем простой список для обратной совместимости
    return [(e.raw_text, e.level) for e in index.all_entries]


def _build_toc_index(toc_dir: Path) -> TocIndex:
    """Построить TocIndex из файлов TOC."""
    index = TocIndex()

    if not toc_dir.exists():
        return index

    for toc_file in sorted(toc_dir.glob("*.md")):
        try:
            content = toc_file.read_text(encoding="utf-8")
        except Exception:
            continue

        for line in content.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Подсчитать # в начале (учитываем отступы: «  ## Title»)
            # Также поддерживаем «# Title» в начале
            stripped = line_stripped.lstrip()
            hash_count = 0
            while hash_count < len(stripped) and stripped[hash_count] == "#":
                hash_count += 1

            if hash_count == 0:
                continue  # не заголовок

            raw_text = stripped[hash_count:].strip()
            if not raw_text:
                continue

            level = hash_count

            entry = TocEntry(raw_text=raw_text, level=level)

            # Очистить trailing номера страниц и точек
            cleaned = re.sub(r"\s*\.{3,}\s*\d*\s*$", "", raw_text)
            cleaned = re.sub(r"\s+\d+\s*$", "", cleaned)
            cleaned = _TRAIL_DOT_RE.sub("", cleaned).strip()

            # Попробовать секционный номер
            sec_key = _extract_section_key(cleaned)
            if sec_key:
                entry.section_key = sec_key
                # Убрать секционный номер из текста: «0. Введение» → «Введение»
                stripped = re.sub(r"^\d+(?:\.\d+)*[.\s]*", "", cleaned)
                entry.normalized = _normalize(stripped)
                index.section_map[sec_key] = entry
                index.all_entries.append(entry)
                continue

            # Попробовать специальный ключ
            sp_key = _extract_special_key(cleaned)
            if sp_key:
                entry.special_key = sp_key
                entry.normalized = _normalize(re.sub(
                    r"^(?:Раздел|Глава|Часть|раздел|глава|часть)\s+[IVXLCDM\d]+[.\s]*",
                    "", cleaned
                ))
                index.special_map[sp_key] = entry
                index.all_entries.append(entry)
                continue

            # Без номера — текстовая запись
            entry.normalized = _normalize(cleaned)
            index.text_entries.append(entry)
            index.all_entries.append(entry)

    return index


def _build_toc_index_from_list(entries: list[tuple[str, int]]) -> TocIndex:
    """Построить TocIndex из простого списка [(text, level)]."""
    index = TocIndex()
    for raw_text, level in entries:
        entry = TocEntry(raw_text=raw_text, level=level)

        # Очистить trailing номера страниц
        cleaned = re.sub(r"\s*\.{3,}\s*\d*\s*$", "", raw_text)
        cleaned = re.sub(r"\s+\d+\s*$", "", cleaned)
        cleaned = _TRAIL_DOT_RE.sub("", cleaned).strip()

        sec_key = _extract_section_key(cleaned)
        if sec_key:
            entry.section_key = sec_key
            stripped = re.sub(r"^\d+(?:\.\d+)*[.\s]*", "", cleaned)
            entry.normalized = _normalize(stripped)
            index.section_map[sec_key] = entry
            index.all_entries.append(entry)
            continue

        sp_key = _extract_special_key(cleaned)
        if sp_key:
            entry.special_key = sp_key
            entry.normalized = _normalize(re.sub(
                r"^(?:Раздел|Глава|Часть|раздел|глава|часть)\s+[IVXLCDM\d]+[.\s]*",
                "", cleaned
            ))
            index.special_map[sp_key] = entry
            index.all_entries.append(entry)
            continue

        entry.normalized = _normalize(cleaned)
        index.text_entries.append(entry)
        index.all_entries.append(entry)

    return index


# ======================================================================
# Каскад стратегий матчинга
# ======================================================================

def _find_match(
    line_text: str,
    index: TocIndex,
    line_has_hash: bool = True,
) -> Optional[TocEntry]:
    """Найти соответствие строки заголовка в TOC.
    
    Каскад стратегий:
    A. Точный секционный номер
    B. Родительский секционный номер
    C. Специальный ключ (Раздел/Глава/Часть)
    D. §-заголовок
    E. Точное текстовое совпадение
    F. Префиксное текстовое совпадение
    G. Токенное перекрытие ≥ 70%
    
    Returns:
        TocEntry или None
    """
    # Убрать # маркеры если есть
    text = line_text.strip()
    if line_has_hash:
        m = re.match(r"^#{1,6}\s+(.+)$", text)
        if m:
            text = m.group(1).strip()

    # Убрать ведущие <NNN> теги
    text = re.sub(r"^<\d+>\s*", "", text).strip()

    # --- Стратегия A: точный секционный номер ---
    sec_key = _extract_section_key(text)
    if sec_key and sec_key in index.section_map:
        return index.section_map[sec_key]

    # --- Стратегия B: родительский номер ---
    if sec_key and len(sec_key) > 1:
        for parent_len in range(len(sec_key) - 1, 0, -1):
            parent_key = sec_key[:parent_len]
            if parent_key in index.section_map:
                return index.section_map[parent_key]

    # --- Стратегия C: специальный ключ ---
    sp_key = _extract_special_key(text)
    if sp_key:
        if sp_key in index.special_map:
            return index.special_map[sp_key]
        # Пробуем римскую → арабскую конверсию
        keyword, raw_num = sp_key
        # Если число, пробуем lookup
        alt_key = (keyword, raw_num)
        if alt_key in index.special_map:
            return index.special_map[alt_key]

    # --- Стратегия D: §-заголовок ---
    para_m = re.match(r"^§\s*(\d+(?:\.\d+)+)[.\s]*(.*)$", text)
    if para_m:
        para_key = tuple(int(p) for p in para_m.group(1).split("."))
        if para_key in index.section_map:
            return index.section_map[para_key]

    # --- Текстовые стратегии (E, F, G) ---
    cleaned = re.sub(r"^(\d+(?:\.\d+)+)[.\s]*", "", text)
    cleaned = re.sub(r"^(?:Раздел|Глава|Часть|раздел|глава|часть)\s+[IVXLCDM\d]+[.\s]*", "", cleaned)
    cleaned = re.sub(r"^§\s*(\d+(?:\.\d+)+)[.\s]*", "", cleaned)
    cleaned = _TRAIL_DOT_RE.sub("", cleaned).strip()
    norm = _normalize(cleaned)

    if not norm:
        return None

    # Собираем кандидатов из text_entries
    candidates = list(index.text_entries)

    # Также добавляем записи с section_key/special_key для текстового матчинга
    # (на случай, если номер был потерян VLM)
    for entry in index.all_entries:
        if entry.normalized and entry not in candidates:
            candidates.append(entry)

    # --- Стратегия E: точное текстовое совпадение ---
    for entry in candidates:
        if entry.normalized == norm:
            return entry

    # --- Стратегия F: префиксное совпадение (один — префикс другого) ---
    for entry in candidates:
        if not entry.normalized or not norm:
            continue
        if entry.normalized.startswith(norm) or norm.startswith(entry.normalized):
            # Дополнительная проверка: min длина ≥ 5 символов
            if len(min(entry.normalized, norm, key=len)) >= 5:
                return entry

    # --- Стратегия G: токенное перекрытие ≥ 70% ---
    best_entry = None
    best_score = 0.0
    for entry in candidates:
        if not entry.normalized:
            continue
        score = _token_overlap(norm, entry.normalized)
        # Требуем минимум 3 токена для надёжности
        min_tokens = min(len(norm.split()), len(entry.normalized.split()))
        if score >= 0.7 and min_tokens >= 2 and score > best_score:
            best_score = score
            best_entry = entry

    return best_entry


# ======================================================================
# Основная обработка
# ======================================================================

def fix_headings_in_text(
    content: str,
    toc_entries: list[tuple[str, int]],
    remove_unmatched: bool = True,
) -> tuple[str, dict]:
    """Исправить уровни заголовков в тексте по TOC.
    
    Args:
        content:         Текст .md файла
        toc_entries:     Список (text, level) из TOC
        remove_unmatched: Удалять # у заголовков, не найденных в TOC
    
    Returns:
        (new_content, stats_dict)
        stats: {total, fixed_level, removed_hash, added_hash, unchanged, matched, unmatched}
    """
    index = _build_toc_index_from_list(toc_entries)
    stats = {
        "total": 0,
        "fixed_level": 0,
        "removed_hash": 0,
        "added_hash": 0,
        "unchanged": 0,
        "matched": 0,
        "unmatched": 0,
    }

    lines = content.split("\n")
    new_lines: list[str] = []

    for line in lines:
        # Проверка blacklist (Рис., Таблица и т.д.)
        if _BLACKLIST_RE.match(line.strip()):
            new_lines.append(line)
            continue

        # Попробовать # -заголовок
        heading_m = _HEADING_RE.match(line)
        if heading_m:
            stats["total"] += 1
            old_level = len(heading_m.group(1))
            heading_text = heading_m.group(2).strip()

            match = _find_match(heading_text, index, line_has_hash=False)
            if match:
                stats["matched"] += 1
                new_level = match.level
                if new_level != old_level:
                    stats["fixed_level"] += 1
                    new_lines.append(f"{'#' * new_level} {heading_text}")
                else:
                    stats["unchanged"] += 1
                    new_lines.append(line)
            else:
                stats["unmatched"] += 1
                if remove_unmatched:
                    stats["removed_hash"] += 1
                    new_lines.append(heading_text)
                else:
                    new_lines.append(line)
            continue

        # Попробовать голый секционный заголовок
        bare_m = _BARE_SECTION_RE.match(line)
        if bare_m:
            stats["total"] += 1
            full_text = line.strip()

            match = _find_match(full_text, index, line_has_hash=False)
            if match:
                stats["matched"] += 1
                stats["added_hash"] += 1
                new_level = match.level
                # Убираем ведущие <NNN> если есть
                clean = re.sub(r"^(?:<\d+>\s*)*", "", full_text).strip()
                new_lines.append(f"{'#' * new_level} {clean}")
            else:
                stats["unmatched"] += 1
                new_lines.append(line)
            continue

        # Попробовать §-заголовок (без #)
        para_m = _PARA_RE.match(line)
        if para_m:
            stats["total"] += 1
            full_text = line.strip()

            match = _find_match(full_text, index, line_has_hash=False)
            if match:
                stats["matched"] += 1
                stats["added_hash"] += 1
                new_level = match.level
                clean = re.sub(r"^(?:<\d+>\s*)*(?:#{1,6}\s*)?", "", full_text).strip()
                new_lines.append(f"{'#' * new_level} {clean}")
            else:
                stats["unmatched"] += 1
                new_lines.append(line)
            continue

        # Попробовать специальный заголовок (Раздел, Глава) без #
        sp_m = _SPECIAL_RE.match(line)
        if sp_m:
            stats["total"] += 1
            full_text = line.strip()

            match = _find_match(full_text, index, line_has_hash=True)
            if match:
                stats["matched"] += 1
                stats["added_hash"] += 1
                new_level = match.level
                clean = re.sub(r"^(?:<\d+>\s*)*(?:#{1,6}\s*)?", "", full_text).strip()
                new_lines.append(f"{'#' * new_level} {clean}")
            else:
                stats["unmatched"] += 1
                new_lines.append(line)
            continue

        # Обычная строка — пропускаем
        new_lines.append(line)

    return "\n".join(new_lines), stats


def process_file(
    md_path: Path,
    toc_entries: list[tuple[str, int]],
    remove_unmatched: bool = True,
) -> dict:
    """Обработать один .md файл in-place.
    
    Returns:
        {"modified": bool, "stats": dict, "error": str|None}
    """
    try:
        content = md_path.read_text(encoding="utf-8")
    except Exception as e:
        return {"modified": False, "stats": {}, "error": str(e)}

    new_content, stats = fix_headings_in_text(content, toc_entries, remove_unmatched)

    if new_content != content:
        try:
            md_path.write_text(new_content, encoding="utf-8")
        except Exception as e:
            return {"modified": False, "stats": stats, "error": str(e)}
        return {"modified": True, "stats": stats, "error": None}

    return {"modified": False, "stats": stats, "error": None}


# ======================================================================
# Статистика (dry run)
# ======================================================================

def count_heading_mismatches(
    content: str,
    toc_entries: list[tuple[str, int]],
) -> dict:
    """Подсчитать несовпадения заголовков (dry run, без модификации).
    
    Returns:
        {total, wrong_level, not_in_toc, correct}
    """
    index = _build_toc_index_from_list(toc_entries)
    result = {"total": 0, "wrong_level": 0, "not_in_toc": 0, "correct": 0}

    for line in content.splitlines():
        if _BLACKLIST_RE.match(line.strip()):
            continue

        heading_m = _HEADING_RE.match(line)
        bare_m = _BARE_SECTION_RE.match(line) if not heading_m else None
        para_m = _PARA_RE.match(line) if not heading_m and not bare_m else None
        sp_m = _SPECIAL_RE.match(line) if not heading_m and not bare_m and not para_m else None

        matcher = heading_m or bare_m or para_m or sp_m
        if not matcher:
            continue

        result["total"] += 1

        if heading_m:
            old_level = len(heading_m.group(1))
            text = heading_m.group(2).strip()
            match = _find_match(text, index, line_has_hash=False)
        elif bare_m:
            old_level = 0  # голый — нет #
            text = line.strip()
            match = _find_match(text, index, line_has_hash=False)
        elif para_m:
            old_level = 0
            text = line.strip()
            match = _find_match(text, index, line_has_hash=False)
        else:  # sp_m
            old_level = 0
            text = line.strip()
            match = _find_match(text, index, line_has_hash=True)

        if match:
            if match.level == old_level:
                result["correct"] += 1
            else:
                result["wrong_level"] += 1
        else:
            result["not_in_toc"] += 1

    return result


def find_unmatched_toc_entries(
    toc_entries: list[tuple[str, int]],
    parsed_dir: Path,
) -> list[dict]:
    """Найти TOC-записи, не найденные в parsed-файлах.
    
    Returns:
        [{"toc_text": str, "toc_level": int, "matched": bool}, ...]
    """
    index = _build_toc_index_from_list(toc_entries)

    # Собрать все заголовки из parsed
    all_headings: list[str] = []
    for md_file in sorted(parsed_dir.glob("*.md")):
        if md_file.parent.name == "toc":
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            for line in content.splitlines():
                stripped = line.strip()
                m = _HEADING_RE.match(stripped)
                if m:
                    all_headings.append(m.group(2).strip())
                elif _BARE_SECTION_RE.match(stripped):
                    all_headings.append(re.sub(r"^(?:<\d+>\s*)*", "", stripped))
                elif _SPECIAL_RE.match(stripped):
                    all_headings.append(re.sub(r"^(?:<\d+>\s*)*(?:#{1,6}\s*)?", "", stripped))
        except Exception:
            pass

    results = []
    for entry in index.all_entries:
        # Попробовать найти хотя бы один match среди заголовков
        found = False
        for h in all_headings:
            match = _find_match(h, index, line_has_hash=False)
            if match is entry:
                found = True
                break

        results.append({
            "toc_text": entry.raw_text,
            "toc_level": entry.level,
            "matched": found,
        })

    return results