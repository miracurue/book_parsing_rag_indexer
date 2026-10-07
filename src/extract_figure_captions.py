"""Извлечение описаний рисунков из Markdown-файлов парсинга.

Находит абзацы, начинающиеся с «Рис.» (или аналогичного паттерна),
извлекает каждое описание в отдельный .md файл.
Оригинальные файлы не изменяются — очищенный текст сохраняется
в отдельную папку (without_figures/).

Именование файлов:
  - одно описание на странице → 023.md
  - несколько → 023-1.md, 023-2.md

Выходные папки:
  - parsed/extracted_figures/ — извлечённые описания
  - parsed/without_figures/ — текст без описаний
"""

from __future__ import annotations

import re
from pathlib import Path

# Дефолтный regex-паттерн: Рис. 1.2., Рис. I.14., Рис 72, Рисунок 5 и т.д.
DEFAULT_PATTERN = r"Рис(?:унок|\.)?\s*[\dIVXivx]+[\.\d\)]*.*"

# Regex для очистки обёрток: <center>, **, <figcaption>
_WRAPPER_RE = re.compile(
    r"^<(?:center|figcaption)[^>]*>\s*"   # открывающий тег
    r"(.*?)"                                # контент
    r"\s*</(?:center|figcaption)>\s*$"      # закрывающий тег
    r"|"
    r"^\*\*\s*(.*?)\s*\*\*\s*$",            # **bold**
    re.DOTALL | re.IGNORECASE,
)


def _strip_wrappers(text: str) -> str:
    """Удалить HTML/bold обёртки и теги номеров страниц из текста описания.

    Обрабатывает:
    - <NNN> теги номеров страниц (в файлах chapters/)
    - <center>text</center>, <figcaption>text</figcaption>
    - **text** (полная строка)
    - **text** rest (частичный bold — убирает маркеры **)
    """
    # Убираем теги номеров страниц <NNN>
    text = re.sub(r"<\d+>\s*", "", text)
    # Убираем HTML-теги в начале и конце
    text = re.sub(r"^<(?:center|figcaption)[^>]*>\s*", "", text)
    text = re.sub(r"\s*</(?:center|figcaption)>\s*$", "", text)
    # Убираем ** маркеры (не более 2 пар: открытие + закрытие)
    text = re.sub(r"^\*\*", "", text)
    text = re.sub(r"\*\*", "", text, count=1)
    return text.strip()


def extract_figure_captions_from_text(
    text: str,
    pattern: str = DEFAULT_PATTERN,
) -> tuple[str, list[str]]:
    """Извлечь описания рисунков из текста и удалить их.

    Абзац = текст между пустыми строками (\\n\\n).
    Каждый абзац, начинающийся с паттерна — описание.

    Args:
        text: Содержимое Markdown-файла.
        pattern: Regex-паттерн для строки-описания.

    Returns:
        (cleaned_text, captions): текст без описаний + список описаний.
    """
    regex = re.compile(pattern, re.IGNORECASE)

    # Разделяем на абзацы по \n\n
    paragraphs = re.split(r"(\n\n+)", text)

    cleaned_parts: list[str] = []
    captions: list[str] = []

    for part in paragraphs:
        # Разделители \n\n+ — пропускаем как есть
        if re.match(r"^\n\n+$", part):
            cleaned_parts.append(part)
            continue

        # Проверяем первую строку абзаца
        first_line = part.split("\n", 1)[0].strip()
        # Убираем обёртки для проверки
        first_line_clean = _strip_wrappers(first_line)

        if regex.match(first_line_clean):
            caption = _strip_wrappers(part.strip())
            captions.append(caption)
        else:
            cleaned_parts.append(part)

    cleaned = "".join(cleaned_parts)

    # Убираем 3+ подряд \n → 2 (мусор после удаления)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"

    return cleaned, captions


def extract_page_number(filepath: Path) -> str:
    """Извлечь номер страницы из имени файла.

    Path("023.md") → "023"
    """
    stem = filepath.stem
    if stem.isdigit():
        return stem
    return "000"


def format_output_filename(page_num: str, index: int, total: int) -> str:
    """Сформировать имя выходного файла.

    total=1 → «023.md», total>1 → «023-1.md», «023-2.md»
    """
    if total <= 1:
        return f"{page_num}.md"
    return f"{page_num}-{index + 1}.md"


def process_file(
    filepath: Path,
    output_dir: Path,
    pattern: str = DEFAULT_PATTERN,
    cleaned_output_dir: Path | None = None,
) -> list[dict]:
    """Обработать один .md файл: извлечь описания, сохранить отдельно.

    Оригинальный файл остаётся без изменений.
    Очищенный текст (без описаний) сохраняется в cleaned_output_dir.

    Args:
        filepath: Путь к исходному .md файлу.
        output_dir: Папка для сохранения извлечённых описаний.
        pattern: Regex-паттерн.
        cleaned_output_dir: Папка для очищенного текста (без описаний).
            Если None — очищенный текст не сохраняется.

    Returns:
        Список dict с результатами для каждого описания.
    """
    results: list[dict] = []

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return [{
            "source_file": filepath.name,
            "output_file": None,
            "caption": None,
            "error": str(e),
        }]

    cleaned_text, captions = extract_figure_captions_from_text(content, pattern)

    if not captions:
        # Нет описаний — всё равно сохраняем очищенный текст (он = оригинал)
        if cleaned_output_dir is not None:
            try:
                cleaned_output_dir.mkdir(parents=True, exist_ok=True)
                (cleaned_output_dir / filepath.name).write_text(
                    cleaned_text, encoding="utf-8"
                )
            except Exception:
                pass
        return []

    # Сохраняем очищенный текст в отдельную папку
    if cleaned_output_dir is not None:
        try:
            cleaned_output_dir.mkdir(parents=True, exist_ok=True)
            (cleaned_output_dir / filepath.name).write_text(
                cleaned_text, encoding="utf-8"
            )
        except Exception as e:
            return [{
                "source_file": filepath.name,
                "output_file": None,
                "caption": None,
                "error": f"Ошибка записи очищенного текста: {e}",
            }]

    page_num = extract_page_number(filepath)
    total = len(captions)

    for idx, caption in enumerate(captions):
        out_name = format_output_filename(page_num, idx, total)
        out_path = output_dir / out_name

        try:
            out_path.write_text(caption + "\n", encoding="utf-8")
            results.append({
                "source_file": filepath.name,
                "output_file": out_name,
                "caption": caption[:100],
                "error": None,
            })
        except Exception as e:
            results.append({
                "source_file": filepath.name,
                "output_file": out_name,
                "caption": caption[:100],
                "error": str(e),
            })

    return results


# Regex для извлечения номера рисунка: «Рис. 1.», «Рис. 1.2», «Рисунок 5»
_FIG_NUM_RE = re.compile(
    r"Рис(?:унок|\.)\s*"
    r"(\d+)",           # основное число (первое после «Рис.»)
    re.IGNORECASE,
)


def extract_figure_number(caption: str) -> int | None:
    """Извлечь номер рисунка из текста описания.

    «Рис. 1. ...» → 1, «Рис. 7. ...» → 7, «Рис. 1.2 ...» → 1.
    Возвращает None, если номер не найден.
    """
    m = _FIG_NUM_RE.search(caption)
    return int(m.group(1)) if m else None


def check_figure_sequence(figures_dir: Path) -> dict:
    """Проверить полноту нумерации рисунков в папке extracted_figures/.

    Читает все .md файлы, извлекает номера рисунков, находит пропуски.

    Args:
        figures_dir: Папка extracted_figures/ с .md файлами описаний.

    Returns:
        {
            "found": [1, 2, 7, 8, ...],  — найденные номера (sorted)
            "missing": [3, 4, 5, 6],     — недостающие номера
            "total_found": int,           — кол-во найденных
            "no_number_files": [str],     — файлы без номера рисунка
        }
    """
    if not figures_dir.exists():
        return {
            "found": [], "missing": [], "total_found": 0,
            "no_number_files": [],
        }

    found_numbers: list[int] = []
    no_number_files: list[str] = []

    for md_file in sorted(figures_dir.glob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        num = extract_figure_number(text)
        if num is not None:
            found_numbers.append(num)
        else:
            no_number_files.append(md_file.name)

    found_numbers.sort()

    # Пропуски: от 1 до max
    if found_numbers:
        full_range = set(range(1, max(found_numbers) + 1))
        missing = sorted(full_range - set(found_numbers))
    else:
        missing = []

    return {
        "found": found_numbers,
        "missing": missing,
        "total_found": len(found_numbers),
        "no_number_files": no_number_files,
    }


# ══════════════════════════════════════════════════════════════════════
# Извлечение по маркерам [FIGURE_START]/[FIGURE_END]
# ══════════════════════════════════════════════════════════════════════

_FIGURE_MARKER_START = "[FIGURE_START]"
_FIGURE_MARKER_END = "[FIGURE_END]"


def extract_by_markers(
    text: str,
    pattern: str = DEFAULT_PATTERN,
) -> tuple[str, list[str]]:
    """Извлечь описания рисунков по маркерам [FIGURE_START]/[FIGURE_END].

    Если маркеры есть в тексте — использует их для 100% точности границ.
    Удаляет маркерные блоки из текста и возвращает очищенный текст + описания.

    Returns:
        (cleaned_text, captions): текст без описаний + список описаний.
    """
    regex = re.compile(pattern, re.IGNORECASE)
    captions: list[str] = []
    result_parts: list[str] = []
    last_end = 0

    search_start = 0
    while True:
        fs = text.find(_FIGURE_MARKER_START, search_start)
        if fs == -1:
            break
        fe = text.find(_FIGURE_MARKER_END, fs)
        if fe == -1:
            break

        # Текст до маркера — сохраняем
        result_parts.append(text[last_end:fs])

        # Контент между маркерами
        block = text[fs + len(_FIGURE_MARKER_START):fe].strip()

        # Убираем теги <NNN> из блока
        block_clean = re.sub(r"<\d+>\s*", "", block).strip()

        # Проверяем, что блок содержит описание рисунка (по паттерну)
        first_line = block_clean.split("\n", 1)[0].strip()
        first_line_clean = _strip_wrappers(first_line)

        if regex.match(first_line_clean):
            captions.append(block)  # сохраняем оригинал с тегами <NNN>
        else:
            # Не рис. — возможно ложный маркер, сохраняем контент
            result_parts.append(block_clean + "\n")

        last_end = fe + len(_FIGURE_MARKER_END)
        search_start = last_end

    # Добавляем остаток текста
    result_parts.append(text[last_end:])

    cleaned = "".join(result_parts)
    # Убираем 3+ подряд \n → 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip() + "\n"

    return cleaned, captions


def extract_figure_captions_from_text_auto(
    text: str,
    pattern: str = DEFAULT_PATTERN,
) -> tuple[str, list[str]]:
    """Автоматически выбрать метод извлечения: по маркерам или по абзацам.

    Если в тексте есть [FIGURE_START]/[FIGURE_END] — использует extract_by_markers().
    Иначе — extract_figure_captions_from_text() (по абзацам).

    Returns:
        (cleaned_text, captions): текст без описаний + список описаний.
    """
    if _FIGURE_MARKER_START in text and _FIGURE_MARKER_END in text:
        return extract_by_markers(text, pattern)
    return extract_figure_captions_from_text(text, pattern)


# ══════════════════════════════════════════════════════════════════════
# Функции для извлечения описаний рисунков из файлов глав (clear_chapters/)
# ══════════════════════════════════════════════════════════════════════

# Regex для тегов номеров страниц <NNN>
_PAGE_TAG_RE = re.compile(r"<(\d+)>")


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

    Returns: строка с номером или '000'.
    """
    m = _PAGE_TAG_RE.search(text)
    return m.group(1) if m else "000"


def _remove_page_tags(text: str) -> str:
    """Удалить все теги <NNN> из текста."""
    return _PAGE_TAG_RE.sub("", text)


def _find_caption_removal_range(text: str, paragraph_start: int, paragraph_end: int) -> tuple[int, int]:
    """Определить диапазон удаления для описания рисунка.

    Включает пустые строки перед абзацем описания (чтобы не оставалось мусора).
    """
    # Находим начало — включаем предшествующие пустые строки
    start = paragraph_start
    while start > 0 and text[start - 1] == '\n':
        start -= 1
    return start, paragraph_end


def process_chapter_file_figures(
    filepath: Path,
    output_dir: Path,
    pattern: str = DEFAULT_PATTERN,
) -> list[dict]:
    """Обработать файл главы: извлечь описания рисунков по маркерам.

    Исходный файл НЕ изменяется. Для извлечения используются только
    промаркированные блоки [FIGURE_START]/[FIGURE_END] (после нормализации).

    Для каждого описания:
    - Имя файла = номер страницы из тега <NNN> (напр. 069.md, 069-1.md).
    - Содержимое: все заголовки главы + описание (теги <NNN> удалены)

    Args:
        filepath:   Путь к файлу главы (chapters/*.md).
        output_dir: Папка для извлечённых описаний (parsed/extracted_figures/).
        pattern:    Regex-паттерн для строки-описания.

    Returns:
        Список dict с результатами. Если FIGURE-маркеров нет — список
        с одним dict с ошибкой "no_markers".
    """
    results: list[dict] = []

    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception as e:
        return [{
            "source_file": filepath.name, "output_file": None,
            "caption": None, "error": str(e),
        }]

    # Проверяем наличие маркеров
    if not (_FIGURE_MARKER_START in content and _FIGURE_MARKER_END in content):
        return [{
            "source_file": filepath.name, "output_file": None,
            "caption": None, "error": "no_markers",
        }]

    # Собираем все заголовки из файла
    headings = _extract_headings(content)
    heading_block = "\n".join(headings)

    # Извлекаем описания строго по маркерам
    _, captions = extract_by_markers(content, pattern)

    if not captions:
        return []

    # Сохраняем каждое описание
    total = len(captions)
    for idx, caption in enumerate(captions):
        caption_clean = _remove_page_tags(caption).strip()
        parts: list[str] = []

        if heading_block:
            parts.append(heading_block)

        parts.append(caption_clean)
        file_content = "\n\n".join(parts) + "\n"

        # Имя выходного файла: номер страницы из тега <NNN>
        page_num = _get_first_page_number(caption)
        out_name = format_output_filename(page_num, idx, total)
        out_path = output_dir / out_name

        try:
            out_path.write_text(file_content, encoding="utf-8")
            results.append({
                "source_file": filepath.name,
                "output_file": out_name,
                "caption": caption_clean[:100],
                "page_num": page_num,
                "error": None,
            })
        except Exception as e:
            results.append({
                "source_file": filepath.name,
                "output_file": out_name,
                "caption": caption_clean[:100],
                "page_num": page_num,
                "error": str(e),
            })

    return results


def count_figure_captions_in_text(
    text: str,
    pattern: str = DEFAULT_PATTERN,
) -> int:
    """Подсчитать описания рисунков в тексте (без сохранения)."""
    regex = re.compile(pattern, re.IGNORECASE)
    count = 0
    paragraphs = re.split(r"\n\n+", text)
    for part in paragraphs:
        first_line = part.split("\n", 1)[0].strip()
        first_line_clean = _strip_wrappers(first_line)
        if regex.match(first_line_clean):
            count += 1
    return count