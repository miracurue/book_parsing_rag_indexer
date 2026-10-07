"""Базовая конфигурация проекта Book Parsing RAG Indexer.

Поддерживает два источника секретов:
1. .env (через python-dotenv)
2. .streamlit/secrets.toml (приоритет выше, если значение не пустое)
"""

import logging
import os
from pathlib import Path

# ── Пути проекта ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_VECTOR_DB_DIR = DATA_DIR / "vector_db"


def _load_secrets() -> dict[str, str]:
    """Загрузить секреты из .env и secrets.toml.

    Приоритет: secrets.toml > .env > переменные окружения.
    """
    secrets: dict[str, str] = {}

    # 1. Из .env (если запускаем не через streamlit)
    if DEFAULT_ENV_FILE.exists():
        try:
            from dotenv import load_dotenv

            load_dotenv(DEFAULT_ENV_FILE, override=False)
        except ImportError:
            pass

    # 2. Из переменных окружения (могут быть загружены dotenv)
    for key in (
        "YANDEX_DISK_TOKEN",
        "NEUROAPI_API_KEY",
        "GLM_API_KEY",
        "GIGACHAT_SECRET",
        "GIGACHAT_AUTH_TOKEN",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "HF_TOKEN",
        "ARXIV_EMAIL",
    ):
        val = os.getenv(key, "")
        if val:
            secrets[key] = val

    # 3. Из secrets.toml (приоритет выше)
    try:
        import streamlit as st

        for key in (
            "YANDEX_DISK_TOKEN",
            "NEUROAPI_API_KEY",
            "GLM_API_KEY",
            "GLM_API_KEY2",
            "GIGACHAT_SECRET",
            "GIGACHAT_AUTH_TOKEN",
            "QDRANT_HOST",
            "QDRANT_PORT",
            "HF_TOKEN",
            "ARXIV_EMAIL",
        ):
            val = st.secrets.get(key, "")
            if val:
                secrets[key] = str(val)
    except (ImportError, Exception):
        pass

    return secrets


# Глобальный кэш секретов (загружается один раз)
_SECRETS: dict[str, str] | None = None


def get_secret(key: str, default: str = "") -> str:
    """Получить секрет по ключу."""
    global _SECRETS
    if _SECRETS is None:
        _SECRETS = _load_secrets()
    return _SECRETS.get(key, default)


def setup_logging(level: str = "INFO") -> None:
    """Настроить logging для проекта."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        force=True,
    )