"""Чистые функции извлечения оглавления из JPG-сканов.

Использует VLM для распознавания оглавления на страницах книги.
Функции работают с байтами и base64 — чистая бизнес-логика.

Паттерн: JPG → VLM → Markdown-оглавление
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO

from src.llm_clients import call_vision_api

logger = logging.getLogger(__name__)


def encode_image_bytes(image_bytes: bytes | BytesIO) -> str:
    """Кодировать изображение в base64 строку."""
    if isinstance(image_bytes, BytesIO):
        image_bytes = image_bytes.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_toc_from_page(
    image_b64: str,
    model_name: str,
    prompt_text: str,
    temperature: float = 0.1,
) -> dict:
    """Извлечь оглавление из одного изображения.

    Args:
        image_b64: Base64-encoded изображение страницы.
        model_name: Имя модели (из реестра).
        prompt_text: Текст промпта.
        temperature: Температура генерации.

    Returns:
        {
            "content": str,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
        }
    """
    result = call_vision_api(
        model_name=model_name,
        prompt=prompt_text,
        image_b64=image_b64,
        temperature=temperature,
    )
    return result