"""YOLO-детекция таблиц и изображений на страницах книг.

Чистые функции для:
- Загрузки YOLO модели
- Инференса (детекция объектов)
- Вырезки обнаруженных регионов
- Отрисовки рамок на изображениях для визуального контроля
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Одна детекция объекта на странице."""
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[int]  # [x1, y1, x2, y2] — пиксели
    page_file: str = ""  # имя исходного JPG


@dataclass
class PageResult:
    """Результат детекции для одной страницы."""
    page_file: str
    page_stem: str  # например "000"
    detections: list[Detection] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)  # (height, width)


def load_model(model_path: str):
    """Загрузить YOLO модель.

    Args:
        model_path: Путь к весам (.pt файл).

    Returns:
        YOLO model object.
    """
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Модель не найдена: {path}")

    logger.info("Загрузка YOLO модели: %s", path)
    model = YOLO(str(path), task="detect")
    logger.info("Модель загружена. Классы: %s", model.names)
    return model


def get_model_info(model_path: str) -> dict:
    """Получить информацию о модели без полного инференса.

    Returns:
        {"model_path": str, "classes": dict[int, str], "task": str}
    """
    from ultralytics import YOLO

    path = Path(model_path)
    if not path.exists():
        return {"model_path": str(path), "classes": {}, "task": "detect", "exists": False}

    model = YOLO(str(path), task="detect")
    return {
        "model_path": str(path),
        "classes": dict(model.names),
        "task": model.task or "detect",
        "exists": True,
    }


def detect_objects(
    model,
    image_path: Path,
    conf: float = 0.25,
    iou: float = 0.7,
) -> PageResult:
    """Запустить детекцию на одном изображении.

    Args:
        model: Загруженная YOLO модель.
        image_path: Путь к JPG файлу.
        conf: Порог confidence.
        iou: Порог IoU для NMS.

    Returns:
        PageResult с обнаруженными объектами.
    """
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
    )

    if not results:
        return PageResult(
            page_file=image_path.name,
            page_stem=image_path.stem,
        )

    result = results[0]
    h, w = result.orig_shape if hasattr(result, "orig_shape") else (0, 0)

    page_result = PageResult(
        page_file=image_path.name,
        page_stem=image_path.stem,
        image_shape=(h, w),
    )

    if result.boxes is None or len(result.boxes) == 0:
        return page_result

    boxes = result.boxes
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().numpy().astype(int).tolist()
        cls_id = int(boxes.cls[i].cpu().numpy())
        confidence = float(boxes.conf[i].cpu().numpy())
        cls_name = model.names.get(cls_id, f"class_{cls_id}")

        page_result.detections.append(Detection(
            class_id=cls_id,
            class_name=cls_name,
            confidence=confidence,
            bbox_xyxy=xyxy,
            page_file=image_path.name,
        ))

    return page_result


def crop_detection(
    image: np.ndarray,
    bbox_xyxy: list[int],
    padding: int = 5,
) -> np.ndarray:
    """Вырезать регион из изображения.

    Args:
        image: numpy массив (H, W, 3).
        bbox_xyxy: [x1, y1, x2, y2].
        padding: Отступ вокруг рамки в пикселях.

    Returns:
        Вырезанный фрагмент.
    """
    h, w = image.shape[:2]
    x1 = max(0, bbox_xyxy[0] - padding)
    y1 = max(0, bbox_xyxy[1] - padding)
    x2 = min(w, bbox_xyxy[2] + padding)
    y2 = min(h, bbox_xyxy[3] + padding)
    return image[y1:y2, x1:x2].copy()


def draw_annotations(
    image: np.ndarray,
    detections: list[Detection],
    class_names: dict[int, str] | None = None,
    line_width: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Нарисовать рамки и лейблы на копии изображения.

    Args:
        image: numpy массив (H, W, 3) — BGR.
        detections: Список детекций.
        class_names: Словарь {class_id: name} (опционально).
        line_width: Толщина линий.
        font_scale: Масштаб шрифта.

    Returns:
        Аннотированная копия изображения (BGR).
    """
    annotated = image.copy()

    # Цвета по классам (BGR)
    colors = {
        "table": (0, 0, 255),     # Красный
        "figure": (0, 255, 0),    # Зелёный
        "image": (0, 255, 0),     # Зелёный
        "picture": (255, 0, 0),   # Синий
    }
    default_color = (0, 165, 255)  # Оранжевый

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        cls_name = det.class_name

        color = colors.get(cls_name, default_color)

        # Рамка
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_width)

        # Лейбл
        label = f"{cls_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Фон для текста
        cv2.rectangle(
            annotated,
            (x1, y1 - th - baseline - 4),
            (x1 + tw, y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated