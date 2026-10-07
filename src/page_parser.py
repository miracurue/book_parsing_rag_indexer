"""Чистые функции AI-парсинга страниц.

3-шаговый пайплайн:
1. Извлечение текста — JPG → VLM → Markdown с тегами [IMAGE_N], [CAPTION_N]
2. Поиск BBox — та же страница → VLM → JSON с координатами
3. Описание картинок — вырезанные фрагменты → VLM → RAG-описание

Все функции работают с BytesIO и base64 строками.
Не зависят от источников данных — чистая бизнес-логика.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Optional

from src.llm_clients import call_vision_api

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Вспомогательные функции
# ══════════════════════════════════════════════════════════════════════

def encode_image_bytes(image_bytes: bytes | BytesIO) -> str:
    """Кодировать изображение в base64 строку.

    Args:
        image_bytes: байты изображения или BytesIO буфер.

    Returns:
        Base64-encoded строка.
    """
    if isinstance(image_bytes, BytesIO):
        image_bytes = image_bytes.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")


def crop_image_by_bbox(
    image_bytes: bytes | BytesIO,
    bbox: list[int],
    img_size: tuple[int, int],
) -> BytesIO | None:
    """Вырезать фрагмент изображения по BBox координатам.

    BBox формат: [ymin, xmin, ymax, xmax] — целые числа 0–1000.

    Args:
        image_bytes: Исходное изображение.
        bbox: Координаты [ymin, xmin, ymax, xmax] (0–1000).
        img_size: (width, height) исходного изображения в пикселях.

    Returns:
        BytesIO с JPEG-фрагментом или None, если координаты некорректны.
    """
    from PIL import Image

    if isinstance(image_bytes, BytesIO):
        image_bytes = image_bytes.getvalue()

    ymin, xmin, ymax, xmax = bbox
    img_w, img_h = img_size

    top = max(0, int(ymin * img_h / 1000))
    left = max(0, int(xmin * img_w / 1000))
    bottom = min(img_h, int(ymax * img_h / 1000))
    right = min(img_w, int(xmax * img_w / 1000))

    if right <= left or bottom <= top:
        logger.warning("Некорректные координаты BBox: %s → (%d,%d,%d,%d)", bbox, left, top, right, bottom)
        return None

    with Image.open(BytesIO(image_bytes)) as img:
        cropped = img.crop((left, top, right, bottom))
        buf = BytesIO()
        cropped.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        return buf


def parse_bboxes_json(raw_text: str) -> dict[str, list[int]]:
    """Распарсить JSON с BBox координатами из ответа модели.

    Убирает markdown-обёртки ```json ... ```, если модель их добавила.

    Returns:
        Dict: {"IMAGE_1": [ymin, xmin, ymax, xmax], ...}
    """
    text = raw_text.strip()
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        logger.warning("BBox JSON не является dict: %s", type(data))
        return {}
    except json.JSONDecodeError as e:
        logger.warning("Не удалось распарсить BBox JSON: %s\nТекст: %.200s", e, text)
        return {}


# ══════════════════════════════════════════════════════════════════════
# Шаги пайплайна
# ══════════════════════════════════════════════════════════════════════

def step1_extract_text(
    image_b64: str,
    model_name: str,
    prompt_text: str,
    temperature: float = 0.1,
) -> dict:
    """Шаг 1: Извлечь текст со страницы.

    Returns:
        {"content": str, "prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    """
    result = call_vision_api(
        model_name=model_name,
        prompt=prompt_text,
        image_b64=image_b64,
        temperature=temperature,
    )
    return result


def step2_get_bboxes(
    image_b64: str,
    model_name: str,
    prompt_text: str,
    temperature: float = 0.0,
) -> dict:
    """Шаг 2: Получить BBox координаты изображений.

    Автоматически запрашивает response_format=json_object для
    моделей, не содержащих «glm» в названии.

    Returns:
        {"bboxes": dict, "prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    """
    api_result = call_vision_api(
        model_name=model_name,
        prompt=prompt_text,
        image_b64=image_b64,
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    bboxes = parse_bboxes_json(api_result["content"])

    return {
        "bboxes": bboxes,
        "prompt_tokens": api_result["prompt_tokens"],
        "completion_tokens": api_result["completion_tokens"],
        "total_tokens": api_result["total_tokens"],
    }


def step3_describe_image(
    crop_b64: str,
    model_name: str,
    prompt_text: str,
    caption: str,
    local_context: str,
    temperature: float = 0.1,
) -> dict:
    """Шаг 3: Описать вырезанное изображение для RAG.

    Args:
        crop_b64: Base64 вырезанного фрагмента.
        model_name: Модель для описания.
        prompt_text: Промпт описания.
        caption: Авторская подпись к картинке.
        local_context: Фрагмент текста вокруг картинки.
        temperature: Температура генерации.

    Returns:
        {"content": str, "prompt_tokens": int, "completion_tokens": int, "total_tokens": int}
    """
    full_prompt = (
        f"ОРИГИНАЛЬНАЯ ПОДПИСЬ:\n{caption}\n\n"
        f"КОНТЕКСТ ИЗ КНИГИ (вокруг картинки):\n{local_context}\n\n"
        f"ЗАДАЧА:\n{prompt_text}"
    )

    return call_vision_api(
        model_name=model_name,
        prompt=full_prompt,
        image_b64=crop_b64,
        temperature=temperature,
    )


# ══════════════════════════════════════════════════════════════════════
# Оркестратор одной страницы
# ══════════════════════════════════════════════════════════════════════

def process_single_page(
    image_bytes: bytes | BytesIO,
    filename: str,
    step1_model: str,
    step1_prompt: str,
    step1_temperature: float,
    step2_model: str,
    step2_prompt: str,
    step2_temperature: float,
    step3_model: str,
    step3_prompt: str,
    step3_temperature: float,
) -> dict:
    """Обработать одну страницу (JPG) через полный 3-шаговый пайплайн.

    Args:
        image_bytes: Байты или BytesIO изображения страницы.
        filename: Имя файла (для логов и метаданных).
        step1_model/prompt/temperature: Настройки шага 1 (извлечение текста).
        step2_model/prompt/temperature: Настройки шага 2 (BBox).
        step3_model/prompt/temperature: Настройки шага 3 (описание).

    Returns:
        {
            "text_content": str,           — итоговый Markdown (без тегов IMAGE/CAPTION)
            "duration": float,             — время обработки в секундах
            "token_stats": dict,           — токены по шагам
            "has_images": bool,            — найдены ли картинки
            "cropped_images": list[BytesIO], — вырезанные фрагменты
            "cropped_names": list[str],    — имена файлов фрагментов
            "image_data": list[dict],      — метаданные картинок (для JSON)
        }
    """
    from PIL import Image

    if isinstance(image_bytes, BytesIO):
        raw_bytes = image_bytes.getvalue()
    else:
        raw_bytes = image_bytes

    t0 = time.time()
    token_stats = {
        "s1_prompt": 0, "s1_comp": 0,
        "s2_prompt": 0, "s2_comp": 0,
        "s3_prompt": 0, "s3_comp": 0,
    }

    base64_img = encode_image_bytes(raw_bytes)

    # ── Шаг 1: Извлечение текста ───────────────────────────────────
    logger.info("[Шаг 1] %s — извлечение текста (%s)", filename, step1_model)
    r1 = step1_extract_text(base64_img, step1_model, step1_prompt, step1_temperature)
    text_content = r1["content"]
    token_stats["s1_prompt"] = r1["prompt_tokens"]
    token_stats["s1_comp"] = r1["completion_tokens"]
    logger.info("[Шаг 1] %s — OK, tokens: %d", filename, r1["total_tokens"])

    # Поиск тегов [IMAGE_N]
    img_numbers = re.findall(r'\[IMAGE_(\d+)\]', text_content)
    has_images = bool(img_numbers)
    cropped_images: list[BytesIO] = []
    cropped_names: list[str] = []
    image_data: list[dict] = []

    if not has_images:
        logger.info("[Пропуск] %s — графика не найдена", filename)
        duration = round(time.time() - t0, 2)
        token_stats["total_prompt"] = token_stats["s1_prompt"]
        token_stats["total_comp"] = token_stats["s1_comp"]
        token_stats["grand_total"] = token_stats["total_prompt"] + token_stats["total_comp"]
        return {
            "text_content": text_content,
            "duration": duration,
            "token_stats": token_stats,
            "has_images": False,
            "cropped_images": [],
            "cropped_names": [],
            "image_data": [],
        }

    logger.info("[Инфо] %s — найдено маркеров: %d", filename, len(img_numbers))

    # ── Шаг 2: BBox ────────────────────────────────────────────────
    logger.info("[Шаг 2] %s — запрос BBox (%s)", filename, step2_model)
    r2 = step2_get_bboxes(base64_img, step2_model, step2_prompt, step2_temperature)
    bboxes = r2["bboxes"]
    token_stats["s2_prompt"] = r2["prompt_tokens"]
    token_stats["s2_comp"] = r2["completion_tokens"]
    logger.info("[Шаг 2] %s — координаты: %s", filename, bboxes)

    # ── Шаг 3: Вырезка + описание ──────────────────────────────────
    base_name = os.path.splitext(filename)[0] if '.' in filename else filename

    with Image.open(BytesIO(raw_bytes)) as img:
        img_w, img_h = img.size

    for num_str in img_numbers:
        tag_img = f"[IMAGE_{num_str}]"
        tag_key = f"IMAGE_{num_str}"

        # Извлечь подпись
        caption_match = re.search(
            rf'\[CAPTION_{num_str}\]\s*(.*?)(?=\n|\[IMAGE_|\[CAPTION_|$)',
            text_content,
        )
        original_caption = caption_match.group(1).strip() if caption_match else "Нет подписи."

        # Контекст вокруг картинки
        img_pos = text_content.find(tag_img)
        local_context = text_content[max(0, img_pos - 500): min(len(text_content), img_pos + 500)]

        if tag_key not in bboxes:
            logger.warning("[Внимание] Координаты для %s не найдены — пропуск", tag_key)
            # Убираем теги из текста
            if caption_match:
                text_content = text_content.replace(caption_match.group(0), "")
            text_content = text_content.replace(tag_img, "")
            continue

        bbox = bboxes[tag_key]
        crop_buf = crop_image_by_bbox(raw_bytes, bbox, (img_w, img_h))

        if crop_buf is None:
            logger.warning("[Ошибка] Некорректные координаты для %s — пропуск", tag_key)
            if caption_match:
                text_content = text_content.replace(caption_match.group(0), "")
            text_content = text_content.replace(tag_img, "")
            continue

        crop_filename = f"{base_name}_{tag_key.lower()}.jpg"
        cropped_images.append(crop_buf)
        cropped_names.append(crop_filename)
        logger.info("[+] Вырезан: %s", crop_filename)

        # Описание через VLM
        crop_b64 = encode_image_bytes(crop_buf)
        logger.info("[AI] Описание для %s (%s)", crop_filename, step3_model)

        r3 = step3_describe_image(
            crop_b64=crop_b64,
            model_name=step3_model,
            prompt_text=step3_prompt,
            caption=original_caption,
            local_context=local_context,
            temperature=step3_temperature,
        )
        desc = r3["content"]
        token_stats["s3_prompt"] += r3["prompt_tokens"]
        token_stats["s3_comp"] += r3["completion_tokens"]

        # Записать метаданные
        image_data.append({
            "page_filename": filename,
            "image_filename": crop_filename,
            "original_caption": original_caption,
            "ai_description": desc if "<SKIP>" not in desc else None,
        })

        # Убрать теги из текста
        if caption_match:
            text_content = text_content.replace(caption_match.group(0), "")
        text_content = text_content.replace(tag_img, "")

    # Итоговая статистика
    total_prompt = token_stats["s1_prompt"] + token_stats["s2_prompt"] + token_stats["s3_prompt"]
    total_comp = token_stats["s1_comp"] + token_stats["s2_comp"] + token_stats["s3_comp"]
    token_stats["total_prompt"] = total_prompt
    token_stats["total_comp"] = total_comp
    token_stats["grand_total"] = total_prompt + total_comp

    duration = round(time.time() - t0, 2)
    logger.info(
        "✔ %s — завершено за %.2fs, токенов: %d",
        filename, duration, token_stats["grand_total"],
    )
    return {
        "text_content": text_content,
        "duration": duration,
        "token_stats": token_stats,
        "has_images": has_images,
        "cropped_images": cropped_images,
        "cropped_names": cropped_names,
        "image_data": image_data,
    }


# ══════════════════════════════════════════════════════════════════════
# Отдельные шаги пайплайна (для запуска независимо)
# ══════════════════════════════════════════════════════════════════════

def process_page_step1(
    image_bytes: bytes | BytesIO,
    filename: str,
    model_name: str,
    prompt_text: str,
    temperature: float = 0.1,
) -> dict:
    """Шаг 1 отдельно: извлечение текста со страницы.

    **Вход:** JPG-изображение страницы (из ``convert_in_jpg/``).
    **Выход:** Markdown с тегами ``[IMAGE_N]``, ``[CAPTION_N]``.
    **Сохранить:** ``parsed/{stem}.md`` — понадобится для шагов 2 и 3.

    Returns:
        {
            "text_content": str,       — Markdown с тегами изображений
            "has_images": bool,        — есть ли теги [IMAGE_N]
            "images_count": int,       — количество тегов [IMAGE_N]
            "duration": float,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
        }
    """
    if isinstance(image_bytes, BytesIO):
        raw_bytes = image_bytes.getvalue()
    else:
        raw_bytes = image_bytes

    t0 = time.time()
    base64_img = encode_image_bytes(raw_bytes)

    r1 = step1_extract_text(base64_img, model_name, prompt_text, temperature)
    duration = round(time.time() - t0, 2)

    img_numbers = re.findall(r'\[IMAGE_(\d+)\]', r1["content"])

    return {
        "text_content": r1["content"],
        "has_images": bool(img_numbers),
        "images_count": len(img_numbers),
        "duration": duration,
        "prompt_tokens": r1["prompt_tokens"],
        "completion_tokens": r1["completion_tokens"],
        "total_tokens": r1["total_tokens"],
    }


def process_page_step2(
    image_bytes: bytes | BytesIO,
    filename: str,
    markdown_text: str,
    model_name: str,
    prompt_text: str,
    temperature: float = 0.0,
) -> dict:
    """Шаг 2 отдельно: поиск BBox + вырезка изображений.

    **Вход:**
      - JPG-изображение страницы (из ``convert_in_jpg/``)
      - Markdown от **шага 1** (с тегами ``[IMAGE_N]``) — из ``parsed/{stem}.md``
    **Выход:**
      - BBox координаты (JSON)
      - Вырезанные изображения (BytesIO)
    **Сохранить:**
      - ``parsed/bboxes/{stem}.json`` — понадобится для шага 3
      - ``parsed/images/{stem}_image_N.jpg`` — понадобится для шага 3

    Returns:
        {
            "bboxes": dict,              — {IMAGE_N: [ymin,xmin,ymax,xmax]}
            "cropped_images": list[BytesIO],
            "cropped_names": list[str],
            "duration": float,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "skipped": bool,             — True если на странице нет IMAGE-тегов
        }
    """
    from PIL import Image

    if isinstance(image_bytes, BytesIO):
        raw_bytes = image_bytes.getvalue()
    else:
        raw_bytes = image_bytes

    t0 = time.time()

    # Проверяем наличие IMAGE-тегов в Markdown от шага 1
    img_numbers = re.findall(r'\[IMAGE_(\d+)\]', markdown_text)
    if not img_numbers:
        return {
            "bboxes": {},
            "cropped_images": [],
            "cropped_names": [],
            "duration": round(time.time() - t0, 2),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "skipped": True,
        }

    base64_img = encode_image_bytes(raw_bytes)

    # Получаем BBox координаты
    r2 = step2_get_bboxes(base64_img, model_name, prompt_text, temperature)
    bboxes = r2["bboxes"]

    # Вырезаем изображения
    with Image.open(BytesIO(raw_bytes)) as img:
        img_w, img_h = img.size

    base_name = os.path.splitext(filename)[0] if '.' in filename else filename

    cropped_images: list[BytesIO] = []
    cropped_names: list[str] = []

    for num_str in img_numbers:
        tag_key = f"IMAGE_{num_str}"
        if tag_key not in bboxes:
            logger.warning("[Шаг 2] Координаты для %s не найдены — пропуск", tag_key)
            continue

        bbox = bboxes[tag_key]
        crop_buf = crop_image_by_bbox(raw_bytes, bbox, (img_w, img_h))
        if crop_buf is None:
            continue

        crop_filename = f"{base_name}_{tag_key.lower()}.jpg"
        cropped_images.append(crop_buf)
        cropped_names.append(crop_filename)

    duration = round(time.time() - t0, 2)

    return {
        "bboxes": bboxes,
        "cropped_images": cropped_images,
        "cropped_names": cropped_names,
        "duration": duration,
        "prompt_tokens": r2["prompt_tokens"],
        "completion_tokens": r2["completion_tokens"],
        "total_tokens": r2["total_tokens"],
        "skipped": False,
    }


def process_page_step3(
    markdown_text: str,
    filename: str,
    cropped_images: list[BytesIO],
    cropped_names: list[str],
    bboxes: dict,
    model_name: str,
    prompt_text: str,
    temperature: float = 0.1,
) -> dict:
    """Шаг 3 отдельно: описание вырезанных изображений + очистка тегов.

    **Вход:**
      - Markdown от **шага 1** (с тегами) — из ``parsed/{stem}.md``
      - Вырезанные изображения от **шага 2** — из ``parsed/images/``
      - BBox координаты от **шага 2** — из ``parsed/bboxes/{stem}.json``
    **Выход:**
      - Очищенный Markdown (теги удалены)
      - AI-описания изображений
    **Сохранить:**
      - ``parsed/{stem}.md`` — перезаписывается без тегов
      - ``parsed/images_descriptions.json`` — пополняется

    Returns:
        {
            "text_content": str,         — очищенный Markdown
            "image_data": list[dict],    — метаданные картинок
            "duration": float,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
        }
    """
    t0 = time.time()
    text_content = markdown_text

    img_numbers = re.findall(r'\[IMAGE_(\d+)\]', text_content)

    token_prompt = 0
    token_comp = 0
    image_data: list[dict] = []

    # Словарь: имя файла вырезки → BytesIO
    crops_by_name = dict(zip(cropped_names, cropped_images))

    base_name = os.path.splitext(filename)[0] if '.' in filename else filename

    for num_str in img_numbers:
        tag_img = f"[IMAGE_{num_str}]"
        tag_key = f"IMAGE_{num_str}"

        # Извлечь подпись
        caption_match = re.search(
            rf'\[CAPTION_{num_str}\]\s*(.*?)(?=\n|\[IMAGE_|\[CAPTION_|$)',
            text_content,
        )
        original_caption = caption_match.group(1).strip() if caption_match else "Нет подписи."

        # Контекст вокруг картинки
        img_pos = text_content.find(tag_img)
        local_context = text_content[max(0, img_pos - 500): min(len(text_content), img_pos + 500)]

        # Найти соответствующую вырезку
        crop_filename = f"{base_name}_{tag_key.lower()}.jpg"
        crop_buf = crops_by_name.get(crop_filename)

        if crop_buf is None:
            # Нет вырезки — просто убрать теги
            if caption_match:
                text_content = text_content.replace(caption_match.group(0), "")
            text_content = text_content.replace(tag_img, "")
            continue

        # Описание через VLM
        crop_b64 = encode_image_bytes(crop_buf)
        logger.info("[Шаг 3] Описание для %s (%s)", crop_filename, model_name)

        r3 = step3_describe_image(
            crop_b64=crop_b64,
            model_name=model_name,
            prompt_text=prompt_text,
            caption=original_caption,
            local_context=local_context,
            temperature=temperature,
        )

        desc = r3["content"]
        token_prompt += r3["prompt_tokens"]
        token_comp += r3["completion_tokens"]

        image_data.append({
            "page_filename": filename,
            "image_filename": crop_filename,
            "original_caption": original_caption,
            "ai_description": desc if "<SKIP>" not in desc else None,
        })

        # Убрать теги из текста
        if caption_match:
            text_content = text_content.replace(caption_match.group(0), "")
        text_content = text_content.replace(tag_img, "")

    duration = round(time.time() - t0, 2)

    return {
        "text_content": text_content,
        "image_data": image_data,
        "duration": duration,
        "prompt_tokens": token_prompt,
        "completion_tokens": token_comp,
        "total_tokens": token_prompt + token_comp,
    }
