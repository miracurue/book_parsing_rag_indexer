"""Деление Markdown-документов на чанки.

Чистые функции — используются оркестратором run_chunking.py.

Логика:
1. Текст разделяется на блоки (текст / таблицы)
2. Текстовые блоки склеиваются в единую строку и режутся на чанки
   заданного размера с overlap
3. Таблицы выделяются отдельно, большие таблицы бьются на части
   с сохранением шапки
4. Каждый чанк содержит метаданные — заголовки (## и т.д.),
   в контексте которых он находится
5. Единая схема JSON: chunk_type (text/table/figure), source_file

Для извлечённых файлов (таблицы, рисунки):
- chunk_extracted_file() — парсит .md с заголовками + тегами <NNN>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .table_refs import (
    extract_figure_id_from_text,
    extract_figure_refs,
    extract_table_id_from_text,
    extract_table_refs,
)


def parse_book_name(book_name: str) -> tuple[str, str]:
    """Разделить имя папки книги на автора и название.

    Паттерн: 'Автор, Название' или просто 'Название'.

    Args:
        book_name: Имя папки книги (без пути).

    Returns:
        (author, title) — оба значения str. Если запятой нет — author=''.
    """
    if "," in book_name:
        parts = book_name.split(",", 1)
        return parts[0].strip(), parts[1].strip()
    return "", book_name.strip()


@dataclass
class ChunkResult:
    """Результат чанкования одного документа."""

    text_chunks: list[dict] = field(default_factory=list)
    table_chunks: list[dict] = field(default_factory=list)


class MarkdownChunker:
    """Делитель Markdown-документов на чанки.

    Args:
        chunk_size: Максимальный размер текстового чанка (символы).
        overlap: Перекрытие между соседними текстовыми чанками.
        min_chunk_size: Минимальный размер последнего чанка.
            Если меньше — объединяется с предыдущим.
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 200,
        min_chunk_size: int = 400,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    # ══════════════════════════════════════════════════════════════════════
    # Вспомогательные методы
    # ══════════════════════════════════════════════════════════════════════

    # Регулярка для удаления тега номера страницы в начале строки
    _PAGE_TAG_RE = re.compile(r"^\s*<\d+>\s*")

    # Регулярка для поиска ВСЕХ тегов <NNN> в произвольном месте текста
    _PAGE_NUM_RE = re.compile(r"<(\d+)>")

    @classmethod
    def _strip_page_tag(cls, line: str) -> str:
        """Убрать ведущий тег <N> (номер страницы) из строки."""
        return cls._PAGE_TAG_RE.sub("", line).strip()

    @classmethod
    def _strip_all_page_tags(cls, line: str) -> tuple[str, set[int]]:
        """Удалить ВСЕ теги <NNN> из строки, вернуть (чистая_строка, номера_страниц).

        В отличие от _strip_page_tag (только ведущий тег), удаляет теги
        из любого места строки и дополнительно собирает номера страниц.
        """
        page_nums: set[int] = set()
        found = cls._PAGE_NUM_RE.findall(line)
        page_nums.update(int(n) for n in found)
        clean = cls._PAGE_NUM_RE.sub("", line)
        return clean, page_nums

    @classmethod
    def is_table_line(cls, line: str) -> bool:
        """Проверяет, является ли строка строкой таблицы.

        Учитывает, что перед | может быть тег страницы: <030> | ... |
        """
        s = cls._strip_page_tag(line)
        return s.startswith("|") and s.endswith("|") and len(s) > 1

    def _split_large_table(
        self,
        table_lines: list[str],
        headers: dict[int, str],
    ) -> list[dict]:
        """Разбивает большую таблицу на части, сохраняя шапку."""
        if len(table_lines) < 3:
            return [{"text": "\n".join(table_lines), "headers": headers}]

        header_rows = table_lines[:2]  # Шапка и разделитель |--|--|
        data_rows = table_lines[2:]

        chunks: list[dict] = []
        current_lines = list(header_rows)
        current_length = sum(len(l) + 1 for l in header_rows)

        for line in data_rows:
            line_len = len(line) + 1
            if current_length + line_len > self.chunk_size and len(current_lines) > 2:
                chunks.append({"text": "\n".join(current_lines), "headers": headers})
                current_lines = list(header_rows)
                current_length = sum(len(l) + 1 for l in header_rows)

            current_lines.append(line)
            current_length += line_len

        if len(current_lines) > 2:
            chunks.append({"text": "\n".join(current_lines), "headers": headers})

        return chunks

    @classmethod
    def _extract_and_remove_page_numbers(cls, chunk: dict) -> None:
        """Извлечь теги <NNN> из чанка в метаданные и удалить их из текста.

        Ищет все вхождения <NNN> в chunk['text']. Номера сохраняются как
        отсортированный список int в chunk['metadata']['page_numbers'].
        Теги удаляются из текста.
        """
        page_nums: set[int] = set()

        if "text" in chunk:
            found = cls._PAGE_NUM_RE.findall(chunk["text"])
            page_nums.update(int(n) for n in found)
            chunk["text"] = cls._PAGE_NUM_RE.sub("", chunk["text"])

        chunk["metadata"]["page_numbers"] = sorted(page_nums)

    # ══════════════════════════════════════════════════════════════════════
    # Основной метод (текст из chapters/clear_chapters)
    # ══════════════════════════════════════════════════════════════════════

    def chunk_document(
        self,
        md_text: str,
        source_file: str = "",
    ) -> ChunkResult:
        """Разбить Markdown-документ на текстовые чанки.

        Алгоритм:
        1. Заголовки (#-######) из начала файла → metadata.headers
        2. Удалить все теги <NNN>, запоминая их позиции в чистом тексте
           (page_markers: список кортежей (позиция, номер_страницы))
        3. Нарезать чистый текст на чанки по chunk_size с overlap
        4. Для каждого чанка взять страницы из page_markers

        Таблицы обрабатываются отдельными функциями
        (chunk_extracted_file, chunk_table_with_summary).

        Args:
            md_text: Полный текст Markdown-документа.
            source_file: Имя исходного файла для метаданных.

        Returns:
            ChunkResult с текстовыми чанками (table_chunks всегда []).
        """
        # ── Шаг 1: Извлечь заголовки из начала файла ──────────────────
        headers: dict[int, str] = {}
        body_start = 0
        for line in md_text.split("\n"):
            m = re.match(r"^(#{1,6})\s+(.*)", line)
            if m:
                headers[len(m.group(1))] = m.group(2).strip()
                body_start += len(line) + 1
            else:
                break

        body = md_text[body_start:]
        if not body.strip():
            return ChunkResult(text_chunks=[], table_chunks=[])

        # ── Шаг 2: Построить clean_text и page_markers ────────────────
        # Удаляем теги <NNN>, записывая их позицию в чистом тексте
        clean_chars: list[str] = []
        page_markers: list[tuple[int, int]] = []  # (позиция, номер_страницы)

        i = 0
        while i < len(body):
            if body[i] == '<':
                m = re.match(r"<(\d+)>", body[i:])
                if m:
                    page_markers.append((len(clean_chars), int(m.group(1))))
                    i += len(m.group(0))
                    continue
            clean_chars.append(body[i])
            i += 1

        clean_text = "".join(clean_chars)
        if not clean_text.strip():
            return ChunkResult(text_chunks=[], table_chunks=[])

        # ── Шаг 3: Нарезка на чанки с overlap ─────────────────────────

        def get_pages(start: int, end: int) -> list[int]:
            """Номера страниц для диапазона [start, end) в clean_text."""
            pages: set[int] = set()
            for pos, pn in page_markers:
                if start <= pos < end:
                    pages.add(pn)
            return sorted(pages)

        text_chunks: list[dict] = []
        pos = 0

        while pos < len(clean_text):
            c_start = pos
            c_end = min(pos + self.chunk_size, len(clean_text))
            c_text = clean_text[c_start:c_end]

            if c_end >= len(clean_text):
                # Последний чанк — если мелкий, объединяем с предыдущим
                if len(c_text) < self.min_chunk_size and text_chunks:
                    last = text_chunks.pop()
                    prev_start = last.pop("_start_idx", c_start)
                    combined = clean_text[prev_start:]

                    if len(combined) <= self.chunk_size:
                        text_chunks.append({
                            "_start_idx": prev_start,
                            "metadata": {
                                "chunk_type": "text",
                                "headers": headers,
                                "page_numbers": get_pages(
                                    prev_start, len(clean_text),
                                ),
                                "source_file": source_file,
                            },
                            "text": combined,
                        })
                    else:
                        mid = len(combined) // 2
                        sp = prev_start + mid

                        text_chunks.append({
                            "_start_idx": prev_start,
                            "metadata": {
                                "chunk_type": "text",
                                "headers": headers,
                                "page_numbers": get_pages(prev_start, sp),
                                "source_file": source_file,
                            },
                            "text": clean_text[prev_start:sp],
                        })
                        text_chunks.append({
                            "metadata": {
                                "chunk_type": "text",
                                "headers": headers,
                                "page_numbers": get_pages(
                                    sp - self.overlap, len(clean_text),
                                ),
                                "source_file": source_file,
                            },
                            "text": clean_text[sp - self.overlap:],
                        })
                else:
                    text_chunks.append({
                        "_start_idx": c_start,
                        "metadata": {
                            "chunk_type": "text",
                            "headers": headers,
                            "page_numbers": get_pages(c_start, c_end),
                            "source_file": source_file,
                        },
                        "text": c_text,
                    })
                break

            text_chunks.append({
                "_start_idx": c_start,
                "metadata": {
                    "chunk_type": "text",
                    "headers": headers,
                    "page_numbers": get_pages(c_start, c_end),
                    "source_file": source_file,
                },
                "text": c_text,
            })
            pos += self.chunk_size - self.overlap

        # Удаляем техническое поле _start_idx
        for c in text_chunks:
            c.pop("_start_idx", None)

        # ── full_content: дубликат текста для универсальности ───────────
        for c in text_chunks:
            c["metadata"]["full_content"] = c["text"]

        # ── Нумерация чанков внутри главы ──────────────────────────────
        for idx, c in enumerate(text_chunks):
            c["metadata"]["chunk_index"] = idx

        # ── Шаг 4: Обогащение metadata (table_refs / figure_refs) ──────
        for c in text_chunks:
            refs = extract_table_refs(c["text"])
            if refs:
                c["metadata"]["table_refs"] = refs
            fig_refs = extract_figure_refs(c["text"])
            if fig_refs:
                c["metadata"]["figure_refs"] = fig_refs

        return ChunkResult(
            text_chunks=text_chunks,
            table_chunks=[],
        )


# ══════════════════════════════════════════════════════════════════════
# Чанкование извлечённых файлов (таблицы / описания рисунков)
# ══════════════════════════════════════════════════════════════════════

def _parse_headers_and_body(md_text: str) -> tuple[dict[int, str], str]:
    """Извлечь заголовки (#-######) из начала файла.

    Returns:
        (headers_dict, remaining_text_after_headers)
    """
    headers: dict[int, str] = {}
    body_start = 0

    for line in md_text.split("\n"):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            level = len(m.group(1))
            headers[level] = m.group(2).strip()
            body_start += len(line) + 1  # +1 за \n
        else:
            # Первая не-заголовочная строка — начало тела
            break

    return headers, md_text[body_start:]


def _extract_page_from_filename(source_file: str) -> list[int]:
    """Извлечь номер страницы из имени файла.

    Examples:
        '086.md' → [86]
        '030-2.md' → [30]
        '042-4.md' → [42]
        'unknown.txt' → []
    """
    import re as _re
    m = _re.match(r"^0*(\d+)", source_file)
    if m:
        return [int(m.group(1))]
    return []


def chunk_extracted_file(
    md_text: str,
    chunk_type: str,
    source_file: str = "",
    chunk_size: int = 1200,
) -> list[dict]:
    """Разбить извлечённый файл (таблицу/рисунок) на чанки.

    Файлы из extracted_tables/ или extracted_figures/ имеют структуру:
      # Заголовок главы
      ## Подзаголовок
      контент (таблица или описание рисунка)

    Номер страницы берётся из имени файла (086.md → 86).

    Алгоритм:
    1. Извлечь заголовки (#-######) из начала файла → metadata.headers
    2. Номер страницы из имени файла → metadata.page_numbers
    3. Удалить теги <NNN> из текста
    4. Оставшийся текст = тело чанка
    5. Если тело > chunk_size — разбить на части:
       - Для таблиц: с сохранением шапки (первые 2 строки)
       - Для рисунков/текста: простая нарезка

    Args:
        md_text: Текст Markdown-файла.
        chunk_type: "table" или "figure".
        source_file: Имя исходного файла (для номера страницы).
        chunk_size: Максимальный размер чанка.

    Returns:
        Список чанков с единой схемой метаданных.
    """
    headers, body = _parse_headers_and_body(md_text)

    # Номер страницы — из имени файла
    page_nums: set[int] = set(_extract_page_from_filename(source_file))

    # Удаляем теги <NNN> из текста (если есть)
    body = MarkdownChunker._PAGE_NUM_RE.sub("", body).strip()

    if not body:
        return []

    # Один чанк — если помещается
    if len(body) <= chunk_size:
        meta_single: dict = {
            "chunk_type": chunk_type,
            "headers": headers,
            "page_numbers": sorted(page_nums),
            "source_file": source_file,
            "full_content": body,
        }
        if chunk_type == "table":
            tid = extract_table_id_from_text(body)
            if tid:
                meta_single["table_id"] = tid
        elif chunk_type == "figure":
            fid = extract_figure_id_from_text(body)
            if fid:
                meta_single["figure_id"] = fid
        return [{"metadata": meta_single, "text": body}]

    # Нарезка большого контента
    # full_content = исходный body до нарезки (для таблиц — полная таблица)
    original_body = body
    chunks: list[dict] = []

    if chunk_type == "table":
        # Для таблиц — пытаемся сохранить шапку
        lines = body.split("\n")
        header_lines: list[str] = []
        data_start = 0

        # Ищем шапку: первые строки до разделителя |---|
        for idx, line in enumerate(lines):
            s = line.strip()
            if re.match(r"^\|[\s\-:]+\|$", s):
                header_lines = lines[:idx + 1]
                data_start = idx + 1
                break

        if header_lines and data_start < len(lines):
            header_text = "\n".join(header_lines)
            header_len = len(header_text) + 1
            data_lines = lines[data_start:]
            current_lines: list[str] = []
            current_length = header_len

            for line in data_lines:
                line_len = len(line) + 1
                if current_length + line_len > chunk_size and current_lines:
                    chunk_text = header_text + "\n" + "\n".join(current_lines)
                    chunks.append({
                        "metadata": {
                            "chunk_type": chunk_type,
                            "headers": headers,
                            "page_numbers": sorted(page_nums),
                            "source_file": source_file,
                            "full_content": original_body,
                        },
                        "text": chunk_text,
                    })
                    current_lines = []
                    current_length = header_len

                current_lines.append(line)
                current_length += line_len

            if current_lines:
                chunk_text = header_text + "\n" + "\n".join(current_lines)
                chunks.append({
                    "metadata": {
                        "chunk_type": chunk_type,
                        "headers": headers,
                        "page_numbers": sorted(page_nums),
                        "source_file": source_file,
                        "full_content": original_body,
                    },
                    "text": chunk_text,
                })
        else:
            # Нет шапки — простая нарезка
            chunks = _split_plain_text(body, chunk_type, headers, page_nums, source_file, chunk_size, full_content=original_body)
    else:
        # Для рисунков/описаний — простая нарезка
        chunks = _split_plain_text(body, chunk_type, headers, page_nums, source_file, chunk_size, full_content=original_body)

    return chunks


def chunk_table_with_summary(
    md_text: str,
    summary_text: str,
    source_file: str = "",
    threshold: int = 1200,
) -> list[dict]:
    """Создать табличный чанк с адаптивным выбором содержимого.

    Логика (двухуровневый подход):
    - Если len(body) <= threshold → векторизуется полная таблица
      (таблица достаточно маленькая для качественного вектора).
    - Если len(body) > threshold → векторизуется сводка (summary),
      а полная таблица сохраняется в metadata.full_content
      для контекста LLM при генерации ответа.

    Args:
        md_text: Полный текст файла из extracted_tables/.
        summary_text: Текст сводки из table_summaries/.
        source_file: Имя файла (для номера страницы).
        threshold: Порог размера (символы). Если полная таблица
            меньше или равна — векторизуется целиком.
            По умолчанию = chunk_size (1200).

    Returns:
        Список из одного чанка (таблица целиком).
    """
    # 1. Парсим заголовки из начала файла
    headers, body = _parse_headers_and_body(md_text)

    # 2. Номер страницы из имени файла
    page_nums = _extract_page_from_filename(source_file)

    # 3. Удаляем теги <NNN> из тела
    body = MarkdownChunker._PAGE_NUM_RE.sub("", body).strip()

    if not body and not summary_text.strip():
        return []

    # ── Адаптивный выбор: полная таблица или сводка ──────────────
    # Заголовки хранятся ТОЛЬКО в metadata.headers,
    # в текст чанка НЕ добавляются (breadcrumb формируется при векторизации)
    if len(body) <= threshold:
        # Таблица маленькая → векторизуем целиком
        meta_small: dict = {
            "chunk_type": "table",
            "headers": headers,
            "page_numbers": sorted(page_nums),
            "source_file": source_file,
            "full_content": body,
        }
        tid = extract_table_id_from_text(body)
        if tid:
            meta_small["table_id"] = tid
        return [{"metadata": meta_small, "text": body}]
    else:
        # Таблица большая → векторизуем сводку, полная таблица в metadata.full_content
        # Убираем заголовки из summary (они уже в metadata.headers → breadcrumb при векторизации)
        _, summary_body = _parse_headers_and_body(summary_text)
        chunk_text = summary_body.strip() or _extract_table_caption(body) or body[:threshold]

        return [{
            "metadata": {
                "chunk_type": "table",
                "headers": headers,
                "page_numbers": sorted(page_nums),
                "source_file": source_file,
                "full_content": body,
                "table_id": extract_table_id_from_text(md_text) or None,
            },
            "text": chunk_text,
        }]


def _extract_table_caption(body: str) -> str:
    """Извлечь описание таблицы (текст перед <table> или |...|).

    Returns:
        Строки описания таблицы (caption).
    """
    # Ищем начало таблицы
    lines = body.split("\n")
    caption_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # HTML таблица
        if stripped.lower().startswith("<table"):
            break
        # Markdown таблица: строка вида | ... |
        if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2:
            # Проверяем, не разделитель ли это (|---|)
            import re as _re
            if _re.match(r"^\|[\s\-:]+\|$", stripped):
                break
            break
        # Пустые строки в начале пропускаем
        if not stripped and not caption_lines:
            continue
        caption_lines.append(stripped)

    return "\n".join(caption_lines).strip()


def _split_plain_text(
    text: str,
    chunk_type: str,
    headers: dict[int, str],
    page_nums: set[int],
    source_file: str,
    chunk_size: int,
    full_content: str | None = None,
) -> list[dict]:
    """Нарезать текст на чанки простым split по размеру."""
    chunks: list[dict] = []
    i = 0
    while i < len(text):
        chunk_text = text[i:i + chunk_size].strip()
        if chunk_text:
            meta: dict = {
                "chunk_type": chunk_type,
                "headers": headers,
                "page_numbers": sorted(page_nums),
                "source_file": source_file,
            }
            if full_content is not None:
                meta["full_content"] = full_content
            chunks.append({"metadata": meta, "text": chunk_text})
        i += chunk_size
    return chunks
