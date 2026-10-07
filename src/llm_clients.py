"""Унифицированный доступ к LLM-провайдерам и моделям.

Поддерживаемые провайдеры:
- neuroapi: OpenAI-совместимый API (https://neuroapi.host/v1)
- zai: Z.ai SDK (zai-sdk)
- gigachat: заглушка — точка расширения

Реестр моделей хранится в data/models.json (аналогично prompts.json).
Клиенты кэшируются на уровне модуля — создаются один раз.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.config import DATA_DIR, get_secret

logger = logging.getLogger(__name__)

# ── Файл хранения моделей ─────────────────────────────────────────────
MODELS_FILE = DATA_DIR / "models.json"

# ── Кэш клиентов по провайдерам ────────────────────────────────────────
_clients_cache: dict[str, object] = {}


# ══════════════════════════════════════════════════════════════════════
# Провайдеры
# ══════════════════════════════════════════════════════════════════════

def _get_neuroapi_client():
    """Создать или вернуть кэшированный OpenAI-клиент для NeuroAPI."""
    if "neuroapi" not in _clients_cache:
        from openai import OpenAI

        api_key = get_secret("NEUROAPI_API_KEY")
        if not api_key:
            raise ValueError("NEUROAPI_API_KEY не задан в секретах")

        _clients_cache["neuroapi"] = OpenAI(
            base_url="https://neuroapi.host/v1",
            api_key=api_key,
        )
        logger.info("NeuroAPI клиент создан")
    return _clients_cache["neuroapi"]


def _get_zai_client():
    """Создать или вернуть кэшированный Z.ai клиент."""
    if "zai" not in _clients_cache:
        from zai import ZaiClient

        api_key = get_secret("GLM_API_KEY")
        if not api_key:
            raise ValueError("GLM_API_KEY не задан в секретах")

        _clients_cache["zai"] = ZaiClient(api_key=api_key)
        logger.info("Z.ai клиент создан")
    return _clients_cache["zai"]


def _get_gigachat_client():
    """Заглушка для GigaChat — будет реализовано позже."""
    raise NotImplementedError(
        "GigaChat провайдер ещё не реализован. "
        "Добавьте реализацию в src/llm_clients.py::_get_gigachat_client()"
    )


class _ManualProvider:
    """Фейковый клиент для моделей с ручным вводом ответов (напр. MetalGPT-1)."""

    class _FakeCompletions:
        def create(self, **kwargs):
            raise NotImplementedError(
                "Эта модель не поддерживает API-вызовы. "
                "Ответы вводятся вручную через UI."
            )

    chat = type("Chat", (), {"completions": _FakeCompletions()})()


def _get_manual_client():
    """Вернуть фейковый клиент для ручного ввода."""
    return _ManualProvider()


# Маппинг: имя провайдера → функция создания клиента
PROVIDER_FACTORIES = {
    "neuroapi": _get_neuroapi_client,
    "zai": _get_zai_client,
    "gigachat": _get_gigachat_client,
    "manual": _get_manual_client,
}


def get_client(provider: str):
    """Получить кэшированный клиент по имени провайдера.

    Raises:
        ValueError: провайдер неизвестен или ключ не задан.
    """
    factory = PROVIDER_FACTORIES.get(provider)
    if factory is None:
        raise ValueError(f"Неизвестный провайдер: {provider}. Доступные: {list(PROVIDER_FACTORIES)}")
    return factory()


def check_provider_available(provider: str) -> tuple[bool, str]:
    """Проверить доступность провайдера (наличие ключа).

    Returns:
        (available, message)
    """
    key_map = {
        "neuroapi": "NEUROAPI_API_KEY",
        "zai": "GLM_API_KEY",
        "gigachat": "GIGACHAT_SECRET",
    }
    key_name = key_map.get(provider, "")
    if not key_name:
        return False, f"Неизвестный провайдер: {provider}"

    val = get_secret(key_name)
    if val:
        return True, f"Ключ {key_name} задан"
    return False, f"Ключ {key_name} не задан"


# ══════════════════════════════════════════════════════════════════════
# Реестр моделей (CRUD)
# ══════════════════════════════════════════════════════════════════════

def _ensure_models_file() -> None:
    """Создать файл models.json, если не существует."""
    if not MODELS_FILE.exists():
        MODELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        MODELS_FILE.write_text(
            json.dumps({"models": []}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def load_models() -> list[dict]:
    """Загрузить все модели из JSON.

    Returns:
        Список словарей: {"name": str, "provider": str}
    """
    _ensure_models_file()
    with open(MODELS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("models", [])


def save_models(models: list[dict]) -> None:
    """Сохранить список моделей (перезапись)."""
    _ensure_models_file()
    with open(MODELS_FILE, "w", encoding="utf-8") as f:
        json.dump({"models": models}, f, ensure_ascii=False, indent=2)


def add_model(name: str, provider: str) -> dict:
    """Добавить модель в реестр.

    Raises:
        ValueError: если модель с таким именем уже есть.
    """
    name = name.strip()
    provider = provider.strip()
    models = load_models()
    if any(m["name"] == name for m in models):
        raise ValueError(f"Модель «{name}» уже есть в реестре.")
    entry = {"name": name, "provider": provider}
    models.append(entry)
    save_models(models)
    logger.info("Добавлена модель: %s (%s)", name, provider)
    return entry


def remove_model(name: str) -> None:
    """Удалить модель по имени."""
    models = load_models()
    new_models = [m for m in models if m["name"] != name]
    if len(new_models) == len(models):
        raise ValueError(f"Модель «{name}» не найдена.")
    save_models(new_models)
    logger.info("Удалена модель: %s", name)


def list_model_names() -> list[str]:
    """Список имён моделей — удобно для st.selectbox."""
    return [m["name"] for m in load_models()]


def get_model_provider(model_name: str) -> str:
    """Получить провайдер по имени модели.

    Raises:
        ValueError: модель не найдена.
    """
    for m in load_models():
        if m["name"] == model_name:
            return m["provider"]
    raise ValueError(f"Модель «{model_name}» не найдена в реестре.")


# ══════════════════════════════════════════════════════════════════════
# Фатальные ошибки API
# ══════════════════════════════════════════════════════════════════════

# Подстроки, которые означают, что повторный запрос бессмысленен
_FATAL_PATTERNS = (
    "insufficient balance",
    "no resource package",
    "please recharge",
    "invalid api key",
    "incorrect api key",
    "authentication",
    "unauthorized",
    "permission denied",
    "code\":\"1113",
    "code\": \"1113",
)


class FatalAPIError(RuntimeError):
    """Ошибка API, которую не имеет смысла повторять (баланс, ключ и т.д.).

    Атрибут ``original`` хранит исходное исключение.
    """

    def __init__(self, message: str, original: Exception | None = None):
        super().__init__(message)
        self.original = original


def _is_fatal_error(exc: Exception) -> bool:
    """Определить, является ли ошибка фатальной (повтор бессмысленен)."""
    text = str(exc).lower()
    return any(p in text for p in _FATAL_PATTERNS)


# ══════════════════════════════════════════════════════════════════════
# Унифицированный вызов Vision API (с retry)
# ══════════════════════════════════════════════════════════════════════

# Настройки retry
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # секунды (экспоненциальный backoff: 2, 4, 8)


def call_vision_api(
    model_name: str,
    prompt: str,
    image_b64: str,
    temperature: float = 0.1,
    response_format: Optional[dict] = None,
    max_retries: int = _MAX_RETRIES,
) -> dict:
    """Вызвать VLM с изображением и промптом.

    Автоматически выбирает провайдера и клиента по имени модели.
    При временных ошибках (rate-limit 429, серверные 5xx) выполняет retry
    с exponential backoff. При фатальных ошибках (баланс, невалидный ключ)
    сразу выбрасывает :class:`FatalAPIError`.

    Args:
        model_name:       Имя модели (из реестра).
        prompt:           Текст промпта.
        image_b64:        Base64-encoded изображение.
        temperature:      Температура генерации.
        response_format:  Опционально {"type": "json_object"}.
        max_retries:      Максимум повторных попыток (по умолчанию 3).

    Returns:
        {
            "content": str,          — текст ответа
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "model": str,
            "provider": str,
        }

    Raises:
        ValueError:    модель не найдена / провайдер недоступен.
        FatalAPIError: фатальная ошибка (баланс, ключ и т.д.).
        RuntimeError:  ошибка API после всех попыток.
    """
    import time as _time

    provider = get_model_provider(model_name)
    client = get_client(provider)

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ],
    }]

    kwargs: dict = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }

    # response_format не поддерживается Z.ai
    if response_format and "glm" not in model_name.lower():
        kwargs["response_format"] = response_format

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            last_exc = e

            # Фатальная ошибка — нет смысла повторять
            if _is_fatal_error(e):
                logger.error(
                    "Фатальная ошибка API (%s/%s): %s", provider, model_name, e
                )
                raise FatalAPIError(
                    f"Фатальная ошибка API ({provider}/{model_name}): {e}",
                    original=e,
                ) from e

            # Временная ошибка — retry с backoff
            if attempt < max_retries:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Попытка %d/%d не удалась (%s/%s): %s. "
                    "Повтор через %.1fс...",
                    attempt, max_retries, provider, model_name, e, delay,
                )
                _time.sleep(delay)
            continue

        # Успех
        content = resp.choices[0].message.content.strip()
        usage = resp.usage

        return {
            "content": content,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            "model": model_name,
            "provider": provider,
        }

    # Все попытки исчерпаны
    raise RuntimeError(
        f"Ошибка API ({provider}/{model_name}) после {max_retries} попыток: {last_exc}"
    ) from last_exc


# ══════════════════════════════════════════════════════════════════════
# Расчёт стоимости вызова
# ══════════════════════════════════════════════════════════════════════

def calculate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> dict:
    """Рассчитать примерную стоимость вызова API в рублях.

    Читает цены из реестра моделей (data/models.json):
      - input_price_rub_per_1m: цена за 1М входных токенов
      - output_price_rub_per_1m: цена за 1М выходных токенов

    Args:
        model_name: Имя модели.
        prompt_tokens: Кол-во входных токенов.
        completion_tokens: Кол-во выходных токенов.

    Returns:
        {
            "cost_rub": float,             — итоговая стоимость в рублях
            "input_cost_rub": float,        — стоимость входных токенов
            "output_cost_rub": float,       — стоимость выходных токенов
            "prices_available": bool,       — заданы ли цены для модели
        }
    """
    models = load_models()
    model_entry = next((m for m in models if m["name"] == model_name), None)

    if not model_entry:
        return {
            "cost_rub": 0.0,
            "input_cost_rub": 0.0,
            "output_cost_rub": 0.0,
            "prices_available": False,
        }

    input_price = model_entry.get("input_price_rub_per_1m")
    output_price = model_entry.get("output_price_rub_per_1m")

    if input_price is None or output_price is None:
        return {
            "cost_rub": 0.0,
            "input_cost_rub": 0.0,
            "output_cost_rub": 0.0,
            "prices_available": False,
        }

    input_cost = (prompt_tokens / 1_000_000) * input_price
    output_cost = (completion_tokens / 1_000_000) * output_price

    return {
        "cost_rub": round(input_cost + output_cost, 6),
        "input_cost_rub": round(input_cost, 6),
        "output_cost_rub": round(output_cost, 6),
        "prices_available": True,
    }


def call_text_api(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    max_retries: int = _MAX_RETRIES,
    max_tokens: Optional[int] = None,
) -> dict:
    """Вызвать LLM с текстовым промптом (без изображения).

    Используется для суммаризации и других текстовых задач.

    Args:
        model_name:     Имя модели (из реестра).
        system_prompt:  Системный промпт.
        user_prompt:    Пользовательский промпт.
        temperature:    Температура генерации.
        max_retries:    Максимум повторных попыток.
        max_tokens:     Максимум токенов в ответе (None = без лимита).

    Returns:
        {
            "content": str,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "model": str,
            "provider": str,
        }

    Raises:
        ValueError:    модель не найдена / провайдер недоступен.
        FatalAPIError: фатальная ошибка (баланс, ключ и т.д.).
        RuntimeError:  ошибка API после всех попыток.
    """
    import time as _time

    provider = get_model_provider(model_name)
    client = get_client(provider)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    kwargs: dict = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            last_exc = e

            if _is_fatal_error(e):
                logger.error(
                    "Фатальная ошибка API (%s/%s): %s", provider, model_name, e
                )
                raise FatalAPIError(
                    f"Фатальная ошибка API ({provider}/{model_name}): {e}",
                    original=e,
                ) from e

            if attempt < max_retries:
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Попытка %d/%d не удалась (%s/%s): %s. Повтор через %.1fс...",
                    attempt, max_retries, provider, model_name, e, delay,
                )
                _time.sleep(delay)
            continue

        content = resp.choices[0].message.content.strip()
        usage = resp.usage

        if not content:
            # Reasoning-модели (glm-4.6v, glm-5.1) могут вернуть пустой content,
            # если все токены ушли на reasoning_content
            logger.warning(
                "Пустой content от %s/%s (tokens=%s, finish_reason=%s). "
                "Возможно, reasoning-модель исчерпала лимит токенов.",
                provider, model_name,
                getattr(usage, "total_tokens", "?"),
                resp.choices[0].finish_reason if resp.choices else "?",
            )

        return {
            "content": content,
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            "model": model_name,
            "provider": provider,
        }

    raise RuntimeError(
        f"Ошибка API ({provider}/{model_name}) после {max_retries} попыток: {last_exc}"
    ) from last_exc
