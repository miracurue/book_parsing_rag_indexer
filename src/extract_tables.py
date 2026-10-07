"""Извлечение таблиц из JPG-сканов страниц в HTML формат.

Использует VLM (GLM-4.6v через Z.ai) для распознавания таблиц на изображениях.
Каждая таблица сохраняется в отдельный HTML-файл, подписи/заголовки — в JSON.

Запуск:
    python -m src.extract_tables
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR, get_secret
from src.llm_clients import FatalAPIError, call_vision_api, get_model_provider, get_client

logger = logging.getLogger(__name__)

# ── Промпт для извлечения таблиц ───────────────────────────────────────
TABLE_EXTRACTION_PROMPT = """Ты — эксперт по распознаванию таблиц в отсканированных документах.

Внимательно рассмотри изображение страницы книги. Найди ВСЕ таблицы на этой странице.

Для каждой найденной таблицы:
1. Преобразуй её в корректный HTML (<table> с <thead>, <tbody>, <tr>, <td>, <th>). Объединённые ячейки размечай через colspan/rowspan. Числа в ячейках сохраняй как есть, без лишних символов.
2. Если перед таблицей есть заголовок (например "Таблица 1.5 — Название") или подпись после неё — укажи его в поле "caption". Если заголовка/подписи нет — верни пустую строку.

ВАЖНО: в HTML-атрибутах используй ТОЛЬКО одинарные кавычки: rowspan='2', colspan='3'. НЕ используй двойные кавычки внутри HTML.

Ответь ТОЛЬКО валидным JSON (без markdown-обёрток):
{
  "tables": [
    {
      "html": "<table>...</table>",
      "caption": "Таблица 1.5 — Название таблицы"
    }
  ]
}

Если на странице нет таблиц — верни: {"tables": []}"""


# ── Список страниц для обработки ──────────────────────────────────────
PAGE_NUMBERS = [
    "030", "037", "038", "039", "053", "056", "061", "068", "070", "072",
    "080", "082", "084", "087", "098", "099", "109", "115", "121", "130",
    "133", "140", "142", "143", "160", "163", "169", "171", "184", "188",
    "189", "194", "201", "202",
]

# ── Пути ──────────────────────────────────────────────────────────────
BOOK_DIR = DATA_DIR / "books" / "Крохин, брикетирование углей"
JPG_DIR = BOOK_DIR / "convert_in_jpg"
OUTPUT_DIR = BOOK_DIR / "parsed" / "tables_html"
CAPTIONS_FILE = OUTPUT_DIR / "tables_captions.json"


def _extract_tables_regex(text: str) -> list[dict]:
    """Fallback: извлечь <table>...</table> блоки напрямую через regex.

    Используется когда модель вернула невалидный JSON (напр. двойные кавычки в HTML).
    Также пытается найти caption перед таблицей.
    """
    tables = []
    # Ищем все <table>...</table> блоки (жадный, но с DOTALL)
    table_matches = list(re.finditer(
        r"<table\b.*?</table>", text, re.DOTALL | re.IGNORECASE
    ))

    for match in table_matches:
        html = match.group(0)
        # Попытка найти caption перед таблицей (строка с "Таблица" или "табл.")
        caption = ""
        before = text[:match.start()]
        # Ищем последнюю строку с "Таблица" перед таблицей
        caption_match = re.search(
            r"(Таблица\s+\d+[\.\d]*[^\n]*)", before, re.IGNORECASE
        )
        if caption_match:
            caption = caption_match.group(1).strip()
        tables.append({"html": html, "caption": caption})

    return tables


def parse_tables_json(raw_text: str) -> list[dict]:
    """Распарсить JSON с таблицами из ответа модели.

    Сначала пробует JSON-парсер. Если не получается — fallback на regex
    для прямого извлечения <table>...</table> блоков.

    Returns:
        Список: [{"html": "<table>...</table>", "caption": "..."}, ...]
    """
    text = raw_text.strip()
    # Убрать markdown-обёртки
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Попытка 1: стандартный JSON-парсер
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tables" in data:
            tables = data["tables"]
            if isinstance(tables, list):
                return tables
        logger.warning("Неожиданная структура JSON: %s", type(data))
    except json.JSONDecodeError as e:
        logger.warning("JSON не распарсен (%s), пробуем regex fallback", e)

    # Попытка 2: починить JSON — заменить двойные кавычки в HTML на одинарные
    try:
        # Заменяем ="цифра" на ='цифра' внутри HTML-атрибутов
        fixed = re.sub(r'=(\d+)"', r"='\1'", text)
        # Более общая замена: ="любое" после HTML-атрибутов
        fixed = re.sub(r'([a-zA-Z-]+)="([^"]{0,20})"', r"\1='\2'", fixed)
        data = json.loads(fixed)
        if isinstance(data, dict) and "tables" in data:
            tables = data["tables"]
            if isinstance(tables, list):
                logger.info("JSON починен заменой кавычек, таблиц: %d", len(tables))
                return tables
    except json.JSONDecodeError:
        pass

    # Попытка 3: regex fallback — извлечь <table> напрямую
    tables = _extract_tables_regex(text)
    if tables:
        logger.info("Regex fallback: извлечено таблиц: %d", len(tables))
        return tables

    logger.warning("Таблицы не найдены ни одним методом")
    return []


def encode_image_file(jpg_path: Path) -> str:
    """Прочитать JPG-файл и закодировать в base64."""
    with open(jpg_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_tables_from_page(
    image_b64: str,
    model_name: str,
    temperature: float = 0.1,
) -> list[dict]:
    """Извлечь таблицы из одного изображения.

    Args:
        image_b64: Base64-encoded изображение страницы.
        model_name: Имя модели (из реестра).
        temperature: Температура генерации.

    Returns:
        Список таблиц: [{"html": str, "caption": str}, ...]
    """
    result = call_vision_api(
        model_name=model_name,
        prompt=TABLE_EXTRACTION_PROMPT,
        image_b64=image_b64,
        temperature=temperature,
    )
    return parse_tables_json(result["content"])


def save_table_html(html_content: str, page_num: str, table_index: int, total_tables: int) -> str:
    """Сохранить HTML-таблицу в файл.

    Args:
        html_content: HTML-код таблицы.
        page_num: Номер страницы (например "030").
        table_index: Индекс таблицы на странице (0-based).
        total_tables: Всего таблиц на странице.

    Returns:
        Имя сохранённого файла.
    """
    if total_tables == 1:
        filename = f"{page_num}.html"
    else:
        filename = f"{page_num}_{table_index + 1}.html"

    filepath = OUTPUT_DIR / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filename


def load_existing_captions() -> list[dict]:
    """Загрузить существующие подписи из JSON."""
    if CAPTIONS_FILE.exists():
        with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_captions(captions: list[dict]) -> None:
    """Сохранить подписи в JSON."""
    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(captions, f, ensure_ascii=False, indent=2)


def get_processed_pages() -> set[str]:
    """Определить уже обработанные страницы по существующим файлам."""
    processed = set()
    if not OUTPUT_DIR.exists():
        return processed

    for f in OUTPUT_DIR.glob("*.html"):
        name = f.stem
        # "030" или "030_1" → "030"
        page = name.split("_")[0]
        processed.add(page)

    return processed


def run_extraction(
    model_name: str = "glm-4.6v",
    temperature: float = 0.1,
    skip_existing: bool = True,
) -> dict:
    """Оркестратор: обойти все страницы и извлечь таблицы.

    Args:
        model_name: Имя модели для Vision API.
        temperature: Температура генерации.
        skip_existing: Пропускать уже обработанные страницы.

    Returns:
        {
            "total_pages": int,
            "processed": int,
            "skipped": int,
            "errors": int,
            "tables_found": int,
            "duration": float,
        }
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    processed = 0
    skipped = 0
    errors = 0
    tables_found = 0

    captions = load_existing_captions()
    existing_pages = get_processed_pages() if skip_existing else set()

    logger.info("Начало извлечения таблиц: %d страниц, модель=%s", len(PAGE_NUMBERS), model_name)
    print(f"\n{'='*60}")
    print(f"Извлечение таблиц: {len(PAGE_NUMBERS)} страниц")
    print(f"Модель: {model_name}")
    print(f"Пропуск готовых: {skip_existing} (уже обработано: {len(existing_pages)})")
    print(f"{'='*60}\n")

    for i, page_num in enumerate(PAGE_NUMBERS):
        jpg_path = JPG_DIR / f"{page_num}.jpg"

        if not jpg_path.exists():
            logger.warning("Файл не найден: %s — пропуск", jpg_path)
            print(f"  [{i+1}/{len(PAGE_NUMBERS)}] {page_num} — файл не найден, пропуск")
            skipped += 1
            continue

        if page_num in existing_pages:
            logger.info("Страница %s уже обработана — пропуск", page_num)
            print(f"  [{i+1}/{len(PAGE_NUMBERS)}] {page_num} — уже обработана, пропуск")
            skipped += 1
            continue

        print(f"  [{i+1}/{len(PAGE_NUMBERS)}] {page_num} — обработка...", end=" ", flush=True)

        try:
            image_b64 = encode_image_file(jpg_path)
            tables = extract_tables_from_page(image_b64, model_name, temperature)

            if not tables:
                print("таблиц не найдено")
                # Сохраняем пустой маркер чтобы не повторять
                processed += 1
                continue

            # Удалить старые записи caption для этой страницы
            captions = [c for c in captions if c["page"] != page_num]

            total_on_page = len(tables)
            for idx, table_data in enumerate(tables):
                html = table_data.get("html", "")
                caption = table_data.get("caption", "")

                if not html.strip():
                    continue

                filename = save_table_html(html, page_num, idx, total_on_page)
                tables_found += 1

                if caption:
                    captions.append({
                        "file": filename,
                        "page": page_num,
                        "caption": caption,
                    })

                logger.info("Сохранено: %s (caption=%s)", filename, bool(caption))

            print(f"найдено таблиц: {total_on_page}")
            processed += 1

        except FatalAPIError as e:
            logger.error("Фатальная ошибка API на странице %s: %s", page_num, e)
            print(f"ФАТАЛЬНАЯ ОШИБКА: {e}")
            errors += 1
            # Сохраняем что успели
            save_captions(captions)
            break

        except Exception as e:
            logger.error("Ошибка на странице %s: %s", page_num, e)
            print(f"ОШИБКА: {e}")
            errors += 1

        # Небольшая пауза между запросами
        time.sleep(0.5)

    # Сохранить подписи
    save_captions(captions)

    duration = round(time.time() - t0, 1)

    result = {
        "total_pages": len(PAGE_NUMBERS),
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "tables_found": tables_found,
        "duration": duration,
    }

    print(f"\n{'='*60}")
    print(f"Готово за {duration}с")
    print(f"  Обработано: {processed}")
    print(f"  Пропущено: {skipped}")
    print(f"  Ошибок: {errors}")
    print(f"  Таблиц найдено: {tables_found}")
    print(f"{'='*60}")

    return result


# ── Точка входа ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    from src.config import setup_logging
    setup_logging("INFO")

    run_extraction()