"""Оркестратор деления Markdown-файлов на чанки.

Три независимых режима:
  1. Текст:  base_dir/*/extracted/clear_chapters/*.md → chunks/text_chunks/*.json
  2. Таблицы: base_dir/*/extracted/extracted_tables/*.md → chunks/table_chunks/*.json
  3. Рисунки: base_dir/*/extracted/extracted_figures/*.md → chunks/figure_chunks/*.json

Также: run_chunking() — все три режима сразу (обратная совместимость).

Поддерживает:
- Пересоздание подпапок chunks/ при повторном запуске
- Отмену через threading.Event
- Обратный вызов прогресса
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from src.chunking import MarkdownChunker, chunk_extracted_file, chunk_table_with_summary
from src.book_metadata import get_book_meta
from src.config import setup_logging

logger = logging.getLogger(__name__)

MD_EXTS = (".md",)


# ══════════════════════════════════════════════════════════════════════
# Поиск книг
# ══════════════════════════════════════════════════════════════════════

def _find_book_dirs(base_dir: Path, subdir: str = "chapters") -> list[tuple[str, Path]]:
    """Найти все папки книг с указанной подпапкой внутри.

    Args:
        base_dir: Корневая папка с книгами.
        subdir: Имя подпапки (chapters, clear_chapters, parsed/extracted_tables, ...).

    Returns:
        Список кортежей: (book_name, book_dir)
    """
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue

        target = book_dir / subdir
        if not target.exists():
            logger.debug("Пропуск %s: нет %s/", book_dir.name, subdir)
            continue

        results.append((book_dir.name, book_dir))

    return results


# ══════════════════════════════════════════════════════════════════════
# Вспомогательные: чтение файлов и сохранение чанков
# ══════════════════════════════════════════════════════════════════════

def _read_md_files(directory: Path) -> list[Path]:
    """Получить отсортированный список .md файлов."""
    if not directory.is_dir():
        return []
    return sorted(
        f for f in directory.glob("*")
        if f.is_file() and f.suffix.lower() in MD_EXTS
    )


def _save_chunks(chunks: list[dict], output_dir: Path, base_name: str) -> int:
    """Сохранить список чанков в JSON файл.

    Returns:
        Количество сохранённых чанков.
    """
    if not chunks:
        return 0

    out_path = output_dir / f"{base_name}.json"
    out_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(chunks)


def _inject_book_meta(chunks: list[dict], book_name: str) -> None:
    """Добавить book_name, author, title, year в metadata каждого чанка (in-place).

    Использует book_metadata.get_book_meta() — ручные данные приоритетнее
    авто-парсинга из имени папки.
    """
    meta_info = get_book_meta(book_name)
    for chunk in chunks:
        meta = chunk.setdefault("metadata", {})
        meta["book_name"] = book_name
        meta["author"] = meta_info.get("author", "")
        meta["title"] = meta_info.get("title", "")
        meta["year"] = meta_info.get("year", "")


# ══════════════════════════════════════════════════════════════════════
# 1. Чанкование текста (из clear_chapters/ или chapters/)
# ══════════════════════════════════════════════════════════════════════

def run_text_chunking(
    base_dir: str = "data/books",
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chunk_size: int = 400,
    book_names: list[str] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Чанкование текста из parsed/clear_chapters/.

    Сохраняет: base_dir/*/chunks/text_chunks/basename.json

    Args:
        book_names: Если указано — обработать только эти книги.
    """
    setup_logging(log_level)
    logger.info("=== Чанкование текста ===")

    base = Path(base_dir)

    # Источник: только parsed/clear_chapters/
    book_dirs = _find_book_dirs(base, "extracted/clear_chapters")
    source_subdir = "extracted/clear_chapters"

    # Фильтрация по выбранным книгам
    if book_names:
        _names = set(book_names)
        book_dirs = [(n, p) for n, p in book_dirs if n in _names]

    total = len(book_dirs)
    if total == 0:
        logger.info("Нет папок книг с текстом в %s", base_dir)
        return {"total": 0, "success": 0, "errors": 0, "skipped": 0,
                "elapsed_sec": 0.0, "cancelled": False, "total_chunks": 0}

    logger.info("К обработке: %d книг (источник: %s)", total, source_subdir)

    chunker = MarkdownChunker(
        chunk_size=chunk_size, overlap=overlap,
        min_chunk_size=min_chunk_size,
    )

    t0 = time.perf_counter()
    success = errors = skipped = 0
    total_chunks = 0
    cancelled = False

    for i, (book_name, book_dir) in enumerate(book_dirs):
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break

        source_dir = book_dir / source_subdir
        chunks_dir = book_dir / "chunks" / "text_chunks"
        relative_name = f"{book_name}/{source_subdir}/"

        # Пересоздаём папку
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir, ignore_errors=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        try:
            md_files = _read_md_files(source_dir)
            if not md_files:
                skipped += 1
                if progress_callback:
                    progress_callback(i + 1, total, relative_name, {
                        "success": success, "errors": errors, "skipped": skipped,
                        "total_chunks": total_chunks,
                    })
                continue

            book_chunks = 0
            for md_file in md_files:
                content = md_file.read_text(encoding="utf-8")
                result = chunker.chunk_document(content, source_file=md_file.name)

                # Добавляем book_name, author, title в метаданные
                _inject_book_meta(result.text_chunks, book_name)
                _inject_book_meta(result.table_chunks, book_name)

                # Сохраняем текстовые чанки
                count = _save_chunks(result.text_chunks, chunks_dir, md_file.stem)
                book_chunks += count

            total_chunks += book_chunks
            success += 1
            logger.info("  %s: %d чанков", book_name, book_chunks)

        except Exception as e:
            errors += 1
            logger.exception("Ошибка при обработке %s: %s", book_name, e)

        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "success": success, "errors": errors, "skipped": skipped,
                "total_chunks": total_chunks,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("=== Текст: %d успешно, %d ошибок, %d чанков, %.2fс ===",
                success, errors, total_chunks, elapsed)

    return {
        "total": total, "success": success, "errors": errors, "skipped": skipped,
        "elapsed_sec": elapsed, "cancelled": cancelled, "total_chunks": total_chunks,
    }


# ══════════════════════════════════════════════════════════════════════
# 2. Чанкование таблиц (из parsed/extracted_tables/)
# ══════════════════════════════════════════════════════════════════════

def run_table_chunking(
    base_dir: str = "data/books",
    chunk_size: int = 1200,
    book_names: list[str] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Чанкование таблиц из parsed/extracted_tables/.

    Сохраняет: base_dir/*/chunks/table_chunks/basename.json

    Args:
        book_names: Если указано — обработать только эти книги.
    """
    setup_logging(log_level)
    logger.info("=== Чанкование таблиц ===")

    base = Path(base_dir)
    book_dirs = _find_book_dirs(base, "extracted/extracted_tables")

    # Фильтрация по выбранным книгам
    if book_names:
        _names = set(book_names)
        book_dirs = [(n, p) for n, p in book_dirs if n in _names]

    total = len(book_dirs)

    if total == 0:
        logger.info("Нет папок книг с extracted_tables/ в %s", base_dir)
        return {"total": 0, "success": 0, "errors": 0, "skipped": 0,
                "elapsed_sec": 0.0, "cancelled": False, "total_chunks": 0}

    logger.info("К обработке: %d книг", total)

    t0 = time.perf_counter()
    success = errors = skipped = 0
    total_chunks = 0
    from_summary = 0
    from_full = 0
    cancelled = False

    for i, (book_name, book_dir) in enumerate(book_dirs):
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break

        source_dir = book_dir / "extracted" / "extracted_tables"
        chunks_dir = book_dir / "chunks" / "table_chunks"
        relative_name = f"{book_name}/extracted_tables/"

        # Пересоздаём папку
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir, ignore_errors=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        try:
            md_files = _read_md_files(source_dir)
            if not md_files:
                skipped += 1
                if progress_callback:
                    progress_callback(i + 1, total, relative_name, {
                        "success": success, "errors": errors, "skipped": skipped,
                        "total_chunks": total_chunks,
                    })
                continue

            # Папка со сводками
            summary_dir = book_dir / "extracted" / "table_summaries"

            book_chunks = 0
            for md_file in md_files:
                content = md_file.read_text(encoding="utf-8")

                # Пытаемся найти сводку
                summary_file = summary_dir / md_file.name
                summary_text = ""
                if summary_file.exists():
                    summary_text = summary_file.read_text(encoding="utf-8")

                if summary_text.strip():
                    # Есть сводка → адаптивный чанк (полная таблица или summary)
                    chunks = chunk_table_with_summary(
                        content,
                        summary_text=summary_text,
                        source_file=md_file.name,
                        threshold=chunk_size,
                    )
                    from_summary += 1
                else:
                    # Нет сводки → fallback к обычному чанкованию
                    chunks = chunk_extracted_file(
                        content, chunk_type="table",
                        source_file=md_file.name, chunk_size=chunk_size,
                    )
                    from_full += 1
                # Добавляем book_name, author, title в метаданные
                _inject_book_meta(chunks, book_name)
                book_chunks += _save_chunks(chunks, chunks_dir, md_file.stem)

            total_chunks += book_chunks
            success += 1
            logger.info("  %s: %d табличных чанков", book_name, book_chunks)

        except Exception as e:
            errors += 1
            logger.exception("Ошибка при обработке %s: %s", book_name, e)

        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "success": success, "errors": errors, "skipped": skipped,
                "total_chunks": total_chunks,
                "from_summary": from_summary, "from_full": from_full,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("=== Таблицы: %d успешно, %d ошибок, %d чанков, %.2fс ===",
                success, errors, total_chunks, elapsed)

    return {
        "total": total, "success": success, "errors": errors, "skipped": skipped,
        "elapsed_sec": elapsed, "cancelled": cancelled, "total_chunks": total_chunks,
        "from_summary": from_summary, "from_full": from_full,
    }


# ══════════════════════════════════════════════════════════════════════
# 3. Чанкование описаний рисунков (из parsed/extracted_figures/)
# ══════════════════════════════════════════════════════════════════════

def run_figure_chunking(
    base_dir: str = "data/books",
    chunk_size: int = 1200,
    book_names: list[str] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Чанкование описаний рисунков из parsed/extracted_figures/.

    Сохраняет: base_dir/*/chunks/figure_chunks/basename.json

    Args:
        book_names: Если указано — обработать только эти книги.
    """
    setup_logging(log_level)
    logger.info("=== Чанкование описаний рисунков ===")

    base = Path(base_dir)
    book_dirs = _find_book_dirs(base, "extracted/extracted_figures")

    # Фильтрация по выбранным книгам
    if book_names:
        _names = set(book_names)
        book_dirs = [(n, p) for n, p in book_dirs if n in _names]

    total = len(book_dirs)

    if total == 0:
        logger.info("Нет папок книг с extracted_figures/ в %s", base_dir)
        return {"total": 0, "success": 0, "errors": 0, "skipped": 0,
                "elapsed_sec": 0.0, "cancelled": False, "total_chunks": 0}

    logger.info("К обработке: %d книг", total)

    t0 = time.perf_counter()
    success = errors = skipped = 0
    total_chunks = 0
    cancelled = False

    for i, (book_name, book_dir) in enumerate(book_dirs):
        if cancel_event and cancel_event.is_set():
            cancelled = True
            break

        source_dir = book_dir / "extracted" / "extracted_figures"
        chunks_dir = book_dir / "chunks" / "figure_chunks"
        relative_name = f"{book_name}/extracted_figures/"

        # Пересоздаём папку
        if chunks_dir.exists():
            shutil.rmtree(chunks_dir, ignore_errors=True)
        chunks_dir.mkdir(parents=True, exist_ok=True)

        try:
            md_files = _read_md_files(source_dir)
            if not md_files:
                skipped += 1
                if progress_callback:
                    progress_callback(i + 1, total, relative_name, {
                        "success": success, "errors": errors, "skipped": skipped,
                        "total_chunks": total_chunks,
                    })
                continue

            book_chunks = 0
            for md_file in md_files:
                content = md_file.read_text(encoding="utf-8")
                chunks = chunk_extracted_file(
                    content, chunk_type="figure",
                    source_file=md_file.name, chunk_size=chunk_size,
                )
                # Добавляем book_name, author, title в метаданные
                _inject_book_meta(chunks, book_name)
                book_chunks += _save_chunks(chunks, chunks_dir, md_file.stem)

            total_chunks += book_chunks
            success += 1
            logger.info("  %s: %d чанков рисунков", book_name, book_chunks)

        except Exception as e:
            errors += 1
            logger.exception("Ошибка при обработке %s: %s", book_name, e)

        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "success": success, "errors": errors, "skipped": skipped,
                "total_chunks": total_chunks,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info("=== Рисунки: %d успешно, %d ошибок, %d чанков, %.2fс ===",
                success, errors, total_chunks, elapsed)

    return {
        "total": total, "success": success, "errors": errors, "skipped": skipped,
        "elapsed_sec": elapsed, "cancelled": cancelled, "total_chunks": total_chunks,
    }


# ══════════════════════════════════════════════════════════════════════
# Обратная совместимость: run_chunking() — все три режима
# ══════════════════════════════════════════════════════════════════════

def run_chunking(
    base_dir: str = "data/books",
    chunk_size: int = 1200,
    overlap: int = 200,
    min_chunk_size: int = 400,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить деление на чанки для всех книг (текст + таблицы + рисунки).

    Обратная совместимость: вызывает run_text_chunking(),
    run_table_chunking(), run_figure_chunking() последовательно.
    """
    setup_logging(log_level)
    logger.info("=== Полное чанкование (все типы) ===")

    t0 = time.perf_counter()
    total_chunks = 0
    total_errors = 0

    # Текст
    text_result = run_text_chunking(
        base_dir=base_dir, chunk_size=chunk_size, overlap=overlap,
        min_chunk_size=min_chunk_size,
        cancel_event=cancel_event, log_level=log_level,
    )
    total_chunks += text_result.get("total_chunks", 0)
    total_errors += text_result.get("errors", 0)

    if cancel_event and cancel_event.is_set():
        return {**text_result, "total_chunks": total_chunks, "errors": total_errors}

    # Таблицы
    table_result = run_table_chunking(
        base_dir=base_dir, chunk_size=chunk_size,
        cancel_event=cancel_event, log_level=log_level,
    )
    total_chunks += table_result.get("total_chunks", 0)
    total_errors += table_result.get("errors", 0)

    if cancel_event and cancel_event.is_set():
        return {**table_result, "total_chunks": total_chunks, "errors": total_errors}

    # Рисунки
    fig_result = run_figure_chunking(
        base_dir=base_dir, chunk_size=chunk_size,
        cancel_event=cancel_event, log_level=log_level,
    )
    total_chunks += fig_result.get("total_chunks", 0)
    total_errors += fig_result.get("errors", 0)

    elapsed = round(time.perf_counter() - t0, 2)

    return {
        "total": text_result.get("total", 0) + table_result.get("total", 0) + fig_result.get("total", 0),
        "success": text_result.get("success", 0) + table_result.get("success", 0) + fig_result.get("success", 0),
        "errors": total_errors,
        "skipped": text_result.get("skipped", 0) + table_result.get("skipped", 0) + fig_result.get("skipped", 0),
        "elapsed_sec": elapsed,
        "cancelled": bool(cancel_event and cancel_event.is_set()),
        "total_text_chunks": text_result.get("total_chunks", 0),
        "total_table_chunks": table_result.get("total_chunks", 0),
        "total_figure_chunks": fig_result.get("total_chunks", 0),
        "total_chunks": total_chunks,
    }