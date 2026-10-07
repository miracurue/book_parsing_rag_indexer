"""Управление промптами — CRUD-операции с хранением в JSON.

Файл хранения: data/prompts.json
Структура:
{
    "prompts": [
        {
            "id": "uuid-строка",
            "name": "Название",
            "description": "Описание",
            "text": "Текст промпта",
            "created_at": "ISO-формат",
            "updated_at": "ISO-формат"
        }
    ]
}

Использование в других скриптах:
    from src.prompts import list_prompt_names, get_prompt_by_name
    names = list_prompt_names()
    prompt = get_prompt_by_name("Мой промпт")
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR

# ── Путь к файлу хранения ─────────────────────────────────────────────
PROMPTS_FILE = DATA_DIR / "prompts.json"


def _ensure_file() -> None:
    """Создать файл prompts.json, если не существует."""
    if not PROMPTS_FILE.exists():
        PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        PROMPTS_FILE.write_text(
            json.dumps({"prompts": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_prompts() -> list[dict]:
    """Загрузить все промпты из JSON-файла.

    Returns:
        Список словарей с полями: id, name, description, text, created_at, updated_at.
        Пустой список, если файл не существует.
    """
    _ensure_file()
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("prompts", [])


def save_prompts(prompts: list[dict]) -> None:
    """Сохранить список промптов в JSON-файл.

    Args:
        prompts: Полный список промптов (перезаписывает файл).
    """
    _ensure_file()
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"prompts": prompts}, f, ensure_ascii=False, indent=2)


def add_prompt(name: str, description: str, text: str) -> dict:
    """Добавить новый промпт.

    Args:
        name: Название промпта (уникальное).
        description: Описание.
        text: Текст промпта.

    Returns:
        Созданный словарь промпта.

    Raises:
        ValueError: Если промпт с таким названием уже существует
                    или name/text пустые.
    """
    name = name.strip()
    description = description.strip()
    text = text.strip()

    if not name:
        raise ValueError("Название промпта не может быть пустым.")
    if not text:
        raise ValueError("Текст промпта не может быть пустым.")

    prompts = load_prompts()

    # Проверка дубликата названия
    if any(p["name"] == name for p in prompts):
        raise ValueError(f"Промпт с названием «{name}» уже существует.")

    now = datetime.now().isoformat()
    prompt = {
        "id": str(uuid.uuid4()),
        "name": name,
        "description": description,
        "text": text,
        "created_at": now,
        "updated_at": now,
    }
    prompts.append(prompt)
    save_prompts(prompts)
    return prompt


def update_prompt(prompt_id: str, **fields) -> dict:
    """Обновить существующий промпт.

    Args:
        prompt_id: UUID промпта.
        **fields: Поля для обновления (name, description, text).

    Returns:
        Обновлённый словарь промпта.

    Raises:
        ValueError: Если промпт не найден или название дублируется.
    """
    prompts = load_prompts()

    # Найти целевой промпт
    target_idx = None
    for i, p in enumerate(prompts):
        if p["id"] == prompt_id:
            target_idx = i
            break

    if target_idx is None:
        raise ValueError(f"Промпт с id «{prompt_id}» не найден.")

    # Проверка дубликата названия (если меняется name)
    new_name = fields.get("name", prompts[target_idx]["name"]).strip()
    if new_name != prompts[target_idx]["name"]:
        if any(p["name"] == new_name for p in prompts):
            raise ValueError(f"Промпт с названием «{new_name}» уже существует.")

    # Обновить поля
    allowed = {"name", "description", "text"}
    for key, value in fields.items():
        if key in allowed:
            prompts[target_idx][key] = value.strip() if isinstance(value, str) else value

    prompts[target_idx]["updated_at"] = datetime.now().isoformat()
    save_prompts(prompts)
    return prompts[target_idx]


def delete_prompt(prompt_id: str) -> None:
    """Удалить промпт по ID.

    Args:
        prompt_id: UUID промпта.

    Raises:
        ValueError: Если промпт не найден.
    """
    prompts = load_prompts()
    new_prompts = [p for p in prompts if p["id"] != prompt_id]

    if len(new_prompts) == len(prompts):
        raise ValueError(f"Промпт с id «{prompt_id}» не найден.")

    save_prompts(new_prompts)


def get_prompt_by_name(name: str) -> Optional[dict]:
    """Найти промпт по названию.

    Args:
        name: Точное название промпта.

    Returns:
        Словарь промпта или None, если не найден.
    """
    prompts = load_prompts()
    for p in prompts:
        if p["name"] == name.strip():
            return p
    return None


def get_prompt_by_id(prompt_id: str) -> Optional[dict]:
    """Найти промпт по ID.

    Args:
        prompt_id: UUID промпта.

    Returns:
        Словарь промпта или None, если не найден.
    """
    prompts = load_prompts()
    for p in prompts:
        if p["id"] == prompt_id:
            return p
    return None


def list_prompt_names() -> list[str]:
    """Получить список названий всех промптов.

    Удобно для использования в st.selectbox на других страницах.

    Returns:
        Список строк — названий промптов.
    """
    prompts = load_prompts()
    return [p["name"] for p in prompts]