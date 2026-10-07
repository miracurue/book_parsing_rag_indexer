"""Оркестратор пакетного парсинга PDF-статей через Docling.

Обходит папку с PDF, вызывает parse_pdf_to_markdown() для каждого.
Поддерживает threading, progress_callback, cancel_event, resume.

Использование:
    result = run_article_parsing(
        raw_dir="data/articles/raw",
        output_dir="data/articles/parsed",
        selected_files=None,  # все PDF
        extract_images=False,
        cancel_event=threading.Event(),
        progress_callback=lambda *a: None,
    )
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from src.docling_parser import (
    parse_pdf_to_markdown,
    parse_pdf_remote,
    create_converter,
    get_pdf_files,
    get_parsed_articles,
)

logger = logging.getLogger(__name__)


def run_article_parsing(
    raw_dir: str | Path,
    output_dir: str | Path,
    selected_files: Optional[list[str]] = None,
    extract_images: bool = False,
    image_scale: float = 2.0,
    extract_formulas: bool = False,
    recreate: bool = False,
    remote_url: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable] = None,
) -> dict:
    """Оркестратор: пакетный парсинг PDF-статей.

    Args:
        raw_dir: Папка с исходными PDF.
        output_dir: Папка для результатов (data/articles/parsed).
        selected_files: Список имён файлов (если None — все PDF).
        extract_images: Извлекать изображения.
        image_scale: Масштаб для вырезания изображений.
        extract_formulas: Распознавать формулы через VLM (LaTeX).
        recreate: Перезаписать все статьи (не пропускать уже обработанные).
        remote_url: URL удалённого сервера Docling (Colab/GPU). Если None — локально.
        cancel_event: Событие отмены.
        progress_callback: Функция (current, total, current_file, stats).

    Returns:
        dict: {
            total, success, errors, cancelled,
            elapsed_sec, results: list[dict]
        }
    """
    start_time = time.time()
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    # Найти PDF файлы
    all_pdfs = get_pdf_files(raw_dir)

    # Фильтр по выбранным файлам
    if selected_files:
        selected_set = set(selected_files)
        pdf_files = [f for f in all_pdfs if f.name in selected_set]
    else:
        pdf_files = all_pdfs

    total = len(pdf_files)
    success = 0
    errors = 0
    results = []

    # Resume: пропускаем уже обработанные (если recreate — не пропускаем)
    already_parsed = set() if recreate else set(get_parsed_articles(output_dir))
    if recreate:
        logger.info("Режим recreate: все статьи будут перезаписаны")

    def _report(current: int, current_file: str):
        if progress_callback:
            progress_callback(current, total, current_file, {
                "success": success,
                "errors": errors,
            })

    # Создаём конвертер ОДИН раз на всю партию (только для локального парсинга)
    converter = None
    if not remote_url:
        _report(0, "загрузка моделей...")
        logger.info("Создание DocumentConverter (extract_formulas=%s)...", extract_formulas)
        converter = create_converter(extract_formulas)
        logger.info("DocumentConverter готов")
    else:
        _report(0, f"удалённый сервер ({remote_url})...")
        logger.info("Удалённый режим: парсинг через %s", remote_url)

    for i, pdf_path in enumerate(pdf_files, start=1):
        # Проверка отмены
        if cancel_event and cancel_event.is_set():
            _report(i, "отмена...")
            return {
                "total": total,
                "success": success,
                "errors": errors,
                "cancelled": True,
                "elapsed_sec": round(time.time() - start_time, 1),
                "results": results,
            }

        stem = pdf_path.stem

        # Resume: пропуск уже обработанных
        if stem in already_parsed:
            logger.info(f"Пропуск (уже обработан): {stem}")
            success += 1
            results.append({"stem": stem, "status": "skipped"})
            _report(i, pdf_path.name)
            continue

        logger.info(f"Парсинг ({i}/{total}): {pdf_path.name}")
        _report(i, pdf_path.name)

        try:
            if remote_url:
                result = parse_pdf_remote(
                    pdf_path=pdf_path,
                    output_dir=output_dir,
                    remote_url=remote_url,
                    extract_images=extract_images,
                    image_scale=image_scale,
                )
            else:
                result = parse_pdf_to_markdown(
                    pdf_path=pdf_path,
                    output_dir=output_dir,
                    extract_images=extract_images,
                    image_scale=image_scale,
                    extract_formulas=extract_formulas,
                    converter=converter,
                )

            if "error" in result:
                logger.error(f"Ошибка парсинга {pdf_path.name}: {result['error']}")
                errors += 1
                result["status"] = "error"
            else:
                success += 1
                result["status"] = "ok"

            results.append(result)

        except Exception as e:
            logger.exception(f"Исключение при парсинге {pdf_path.name}: {e}")
            errors += 1
            results.append({"stem": stem, "status": "error", "error": str(e)})

    elapsed = round(time.time() - start_time, 1)

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "cancelled": False,
        "elapsed_sec": elapsed,
        "results": results,
    }