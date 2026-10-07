"""Оркестратор YOLO-детекции таблиц и изображений.

Обходит структуру: base_dir/название_книги/convert_in_jpg/*.jpg
Для каждого JPG: детекция → вырезка → аннотирование → сохранение.

Результаты:
  base_dir/название_книги/yolo_detected/
    annotated/          JPG с нарисованными рамками
    tables/             Вырезанные таблицы
    figures/            Вырезанные картинки
    detections.json     Все детекции с метаданными
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

import cv2

from src.config import setup_logging
from src.yolo_detector import (
    Detection,
    PageResult,
    crop_detection,
    detect_objects,
    draw_annotations,
    load_model,
)

logger = logging.getLogger(__name__)


def find_jpg_files(
    base_dir: Path,
    book_name: str | None = None,
) -> list[tuple[str, Path]]:
    """Найти все JPG в структуре base_dir/*/convert_in_jpg/.

    Args:
        base_dir:  Корневая папка с книгами.
        book_name: Если указано — искать только в этой книге.

    Returns:
        Список кортежей: (book_name, jpg_path)
    """
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    # Если указана конкретная книга — ищем только в ней
    if book_name:
        book_dir = base_dir / book_name
        if not book_dir.is_dir():
            return results
        dirs_to_scan = [book_dir]
    else:
        dirs_to_scan = sorted(base_dir.iterdir())

    for book_dir in dirs_to_scan:
        if not book_dir.is_dir():
            continue

        jpg_dir = book_dir / "convert_in_jpg"
        if not jpg_dir.is_dir():
            logger.debug("Пропуск %s: нет convert_in_jpg/", book_dir.name)
            continue

        for jpg_file in sorted(jpg_dir.glob("*.jpg")):
            results.append((book_dir.name, jpg_file))

    return results


def find_available_books(base_dir: Path) -> list[str]:
    """Найти все книги с папкой convert_in_jpg/.

    Returns:
        Отсортированный список имён книг.
    """
    books: list[str] = []
    if not base_dir.exists():
        return books
    for book_dir in sorted(base_dir.iterdir()):
        if book_dir.is_dir() and (book_dir / "convert_in_jpg").is_dir():
            books.append(book_dir.name)
    return books


def _page_result_to_dict(page: PageResult) -> dict:
    """Конвертировать PageResult в dict для JSON."""
    return {
        "page_file": page.page_file,
        "page_stem": page.page_stem,
        "image_shape": list(page.image_shape),
        "detections": [
            {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox_xyxy": d.bbox_xyxy,
            }
            for d in page.detections
        ],
    }


def _load_existing_detections(json_path: Path) -> dict:
    """Загрузить существующий detections.json."""
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"pages": {}}


def _get_processed_stems(data: dict) -> set[str]:
    """Получить множество уже обработанных page_stem из данных."""
    return set(data.get("pages", {}).keys())


def run_yolo_detection(
    base_dir: str = "data/books",
    model_path: str = "models/yolo/book_parsing_n/weights/best.pt",
    book_name: str | None = None,
    conf: float = 0.25,
    iou: float = 0.7,
    padding: int = 5,
    skip_existing: bool = True,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить YOLO-детекцию на всех страницах.

    Args:
        base_dir:          Корневая папка с книгами.
        model_path:        Путь к весам YOLO (.pt).
        book_name:         Имя конкретной книги (None = все книги).
        conf:              Порог confidence.
        iou:               Порог IoU для NMS.
        padding:           Отступ при вырезке (px).
        skip_existing:     Пропускать уже обработанные страницы.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict с результатами: total, success, errors, tables, figures, elapsed_sec, cancelled.
    """
    setup_logging(log_level)

    base = Path(base_dir)

    # Загружаем модель один раз
    logger.info("=== Запуск YOLO-детекции ===")
    logger.info("Модель: %s, conf=%.2f, iou=%.2f", model_path, conf, iou)

    model = load_model(model_path)
    class_names = dict(model.names)
    logger.info("Классы модели: %s", class_names)

    # Собираем файлы
    files = find_jpg_files(base, book_name=book_name)
    total = len(files)

    if total == 0:
        logger.info("Нет JPG файлов для обработки в %s", base_dir)
        return {
            "total": 0, "success": 0, "errors": 0,
            "tables": 0, "figures": 0,
            "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d JPG файлов", total)

    t0 = time.perf_counter()
    success = 0
    errors = 0
    total_tables = 0
    total_figures = 0
    cancelled = False

    # Группируем файлы по книгам для по-книжного сохранения
    books: dict[str, list[tuple[str, Path]]] = {}
    for book_name, jpg_path in files:
        books.setdefault(book_name, []).append((book_name, jpg_path))

    for book_name, book_files in books.items():
        book_dir = base / book_name
        output_dir = book_dir / "yolo_detected"
        annotated_dir = output_dir / "annotated"
        tables_dir = output_dir / "tables"
        figures_dir = output_dir / "figures"
        json_path = output_dir / "detections.json"

        # Загружаем существующий прогресс
        existing_data = _load_existing_detections(json_path)
        existing_data["model"] = model_path
        existing_data["classes"] = {str(k): v for k, v in class_names.items()}

        processed_stems = _get_processed_stems(existing_data) if skip_existing else set()

        for jpg_path in (f[1] for f in book_files):
            # Глобальный индекс для progress_callback
            current_idx = success + errors + 1
            stem = jpg_path.stem

            # Проверка отмены
            if cancel_event and cancel_event.is_set():
                logger.info("Детекция отменена на странице %s", stem)
                cancelled = True
                break

            # Пропуск уже обработанных
            if stem in processed_stems:
                logger.debug("Пропуск %s — уже обработано", stem)
                success += 1
                if progress_callback:
                    progress_callback(current_idx, total, jpg_path.name, {
                        "success": success, "errors": errors,
                        "tables": total_tables, "figures": total_figures,
                    })
                continue

            relative_name = f"{book_name}/convert_in_jpg/{jpg_path.name}"
            logger.info("[%d/%d] %s", current_idx, total, relative_name)

            try:
                # Детекция
                page_result = detect_objects(model, jpg_path, conf=conf, iou=iou)

                # Читаем изображение для вырезки и аннотирования
                image_bgr = cv2.imread(str(jpg_path))
                if image_bgr is None:
                    raise ValueError(f"Не удалось прочитать изображение: {jpg_path}")

                # Аннотированное изображение
                annotated = draw_annotations(image_bgr, page_result.detections)

                # Сохраняем аннотированное
                annotated_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(annotated_dir / jpg_path.name), annotated)

                # Вырезаем и сохраняем детекции
                for det_idx, det in enumerate(page_result.detections, start=1):
                    cropped = crop_detection(image_bgr, det.bbox_xyxy, padding=padding)

                    if cropped.size == 0:
                        logger.warning("Пустой кроп для %s bbox=%s", stem, det.bbox_xyxy)
                        continue

                    if det.class_name == "table":
                        tables_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = tables_dir / f"{stem}_{det_idx}.jpg"
                        cv2.imwrite(str(crop_path), cropped)
                        total_tables += 1
                    else:
                        # figure, image, picture и прочие → figures/
                        figures_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = figures_dir / f"{stem}_{det_idx}.jpg"
                        cv2.imwrite(str(crop_path), cropped)
                        total_figures += 1

                # Сохраняем в JSON
                existing_data["pages"][stem] = _page_result_to_dict(page_result)

                # Пишем JSON после каждой страницы (для возобновления)
                output_dir.mkdir(parents=True, exist_ok=True)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=2)

                success += 1

            except Exception:
                errors += 1
                logger.exception("[%d/%d] ОШИБКА — %s", current_idx, total, relative_name)

            # Прогресс
            if progress_callback:
                progress_callback(current_idx, total, jpg_path.name, {
                    "success": success, "errors": errors,
                    "tables": total_tables, "figures": total_figures,
                })

        if cancelled:
            break

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Готово: %d/%d успешно, %d ошибок, %d таблиц, %d фигур%s, %.2f с ===",
        success, total, errors, total_tables, total_figures,
        ", ОТМЕНЕНО" if cancelled else "",
        elapsed,
    )

    return {
        "total": total,
        "success": success,
        "errors": errors,
        "tables": total_tables,
        "figures": total_figures,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }