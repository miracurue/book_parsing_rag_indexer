"""Оркестратор извлечения оглавления из JPG-сканов.

Обходит структуру: base_dir/название_книги/convert_in_jpg/toc/*.jpg
Отправляет каждое изображение в VLM с выбранным промптом.
Сохраняет результаты:
  base_dir/название_книги/parsed/toc/toc.md  — итоговый Markdown

Поддерживает:
- Выбор книг через параметр book_names
- Возобновление (пропускает книги с уже существующим toc.md)
- Отмену через threading.Event
- Обратный вызов прогресса
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from src.config import setup_logging
from src.extract_toc import encode_image_bytes, extract_toc_from_page

logger = logging.getLogger(__name__)

JPG_EXTS = (".jpg", ".jpeg", ".png")


def find_toc_images(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> list[tuple[str, Path, Path]]:
    """Найти JPG файлы в подпапках */convert_in_jpg/toc/.

    Args:
        base_dir: Корневая папка с книгами.
        book_names: Список имён книг для обработки. None = все.

    Returns:
        Список кортежей: (book_name, jpg_path, toc_output_dir)
    """
    results: list[tuple[str, Path, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        if book_names and book_dir.name not in book_names:
            continue

        oglav_dir = book_dir / "convert_in_jpg" / "toc"
        if not oglav_dir.is_dir():
            logger.debug("Пропуск %s: нет convert_in_jpg/toc/", book_dir.name)
            continue

        toc_dir = book_dir / "parsed" / "toc"

        for jpg_file in sorted(oglav_dir.glob("*")):
            if jpg_file.is_file() and jpg_file.suffix.lower() in JPG_EXTS:
                results.append((book_dir.name, jpg_file, toc_dir))

    return results


def find_books_with_oglavlenie(base_dir: Path) -> list[str]:
    """Найти книги, у которых есть папка convert_in_jpg/toc/.

    Returns:
        Список имён книг.
    """
    books: list[str] = []
    if not base_dir.exists():
        return books

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        oglav_dir = book_dir / "convert_in_jpg" / "toc"
        if oglav_dir.is_dir():
            # Проверяем, что есть хотя бы один JPG
            has_jpg = any(
                f.is_file() and f.suffix.lower() in JPG_EXTS
                for f in oglav_dir.iterdir()
            )
            if has_jpg:
                books.append(book_dir.name)

    return books


def count_toc_images(base_dir: Path, book_names: list[str]) -> int:
    """Посчитать количество JPG в папках toc/ для выбранных книг."""
    files = find_toc_images(base_dir, book_names=book_names)
    return len(files)


# ══════════════════════════════════════════════════════════════════════
# Публичный интерфейс
# ══════════════════════════════════════════════════════════════════════

def run_extract_toc(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    model_name: str = "glm-4.6v",
    prompt_text: str = "",
    temperature: float = 0.1,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить извлечение оглавления для выбранных книг.

    Обходит: base_dir/*/convert_in_jpg/toc/*.jpg
    Сохраняет: base_dir/*/parsed/toc/

    Args:
        base_dir:          Корневая папка с книгами.
        book_names:        Список имён книг. None = все с toc/.
        model_name:        Имя модели VLM (из реестра).
        prompt_text:       Текст промпта.
        temperature:       Температура генерации.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict: total, success, errors, skipped, elapsed_sec, cancelled,
              total_tokens, books_processed.
    """
    setup_logging(log_level)

    logger.info("=== Запуск извлечения оглавления ===")
    logger.info("Базовая папка: %s", base_dir)
    logger.info("Модель: %s, книг: %s", model_name,
                ", ".join(book_names) if book_names else "все")

    base = Path(base_dir)
    files = find_toc_images(base, book_names=book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет JPG файлов для обработки в %s", base_dir)
        return {
            "total": 0, "success": 0, "errors": 0, "skipped": 0,
            "elapsed_sec": 0.0, "cancelled": False, "total_tokens": 0,
            "books_processed": 0,
        }

    logger.info("К обработке: %d JPG файлов", total)

    # Группируем по книгам
    book_files: dict[str, list[tuple[str, Path, Path]]] = {}
    for book_name, jpg_path, toc_dir in files:
        book_files.setdefault(book_name, []).append((book_name, jpg_path, toc_dir))

    t0 = time.perf_counter()
    success = 0
    errors = 0
    skipped = 0
    cancelled = False
    grand_total_tokens = 0
    processed_in_run = 0
    books_processed = 0

    for book_name, book_file_list in book_files.items():
        # Проверяем, есть ли уже результат для этой книги
        toc_dir = book_file_list[0][2]
        toc_md_path = toc_dir / "toc.md"

        if toc_md_path.exists() and toc_md_path.stat().st_size > 0:
            logger.info("⏩ Пропуск книги %s: оглавление уже извлечено", book_name)
            skipped += len(book_file_list)
            processed_in_run += len(book_file_list)
            continue

        # Собираем все ответы по страницам этой книги
        page_results: list[str] = []

        for _book, jpg_path, _toc_dir in book_file_list:
            filename = jpg_path.name

            # Проверка отмены
            if cancel_event and cancel_event.is_set():
                logger.info("Извлечение отменено на файле %s", filename)
                cancelled = True
                break

            relative_name = f"{book_name}/toc/{filename}"
            logger.info("[%d/%d] %s", processed_in_run + 1, total, relative_name)

            try:
                image_bytes = jpg_path.read_bytes()
                image_b64 = encode_image_bytes(image_bytes)

                result = extract_toc_from_page(
                    image_b64=image_b64,
                    model_name=model_name,
                    prompt_text=prompt_text,
                    temperature=temperature,
                )

                page_results.append(result["content"])
                grand_total_tokens += result["total_tokens"]
                success += 1
                processed_in_run += 1

            except Exception as e:
                errors += 1
                processed_in_run += 1
                logger.exception("❌ Ошибка при обработке %s: %s", filename, e)

            # Обратный вызов прогресса
            if progress_callback:
                stats = {
                    "success": success,
                    "errors": errors,
                    "skipped": skipped,
                    "total_tokens": grand_total_tokens,
                }
                progress_callback(processed_in_run, total, relative_name, stats)

        if cancelled:
            break

        # Сохраняем результат для книги
        if page_results:
            toc_dir.mkdir(parents=True, exist_ok=True)

            # Итоговый Markdown
            content = "\n\n".join(page_results)
            toc_md_path = toc_dir / "toc.md"
            toc_md_path.write_text(content, encoding="utf-8")

            books_processed += 1
            logger.info(
                "✔ Книга %s: оглавление сохранено (%d страниц, %d токенов)",
                book_name, len(page_results), grand_total_tokens,
            )

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Извлечение оглавления завершено: %d успешно, %d ошибок, "
        "%d пропущено%s, %.2fс, %d токенов, %d книг ===",
        success, errors, skipped,
        ", ОТМЕНЕНО" if cancelled else "",
        elapsed, grand_total_tokens, books_processed,
    )

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
        "total_tokens": grand_total_tokens,
        "books_processed": books_processed,
    }