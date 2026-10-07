"""Векторизация чанков и изображений — загрузка в Qdrant.

Чистые функции — используются оркестратором run_vectorizer.py.

Провайдеры dense-эмбеддингов:
  - huggingface: sentence-transformers (локальные модели)
  - gigachat: Сбер GigaChat Embeddings (API, 1024d)
  - neuroapi: OpenAI-compatible /v1/embeddings (API)

Схема Qdrant (на точку):
  - dense (named):  один dense-вектор для ВСЕХ типов (текст, таблицы, описания изображений)
  - text_sparse:    опциональный sparse-вектор для гибридного поиска (BM42)

Нулевых заглушек нет — каждая точка хранит только реальный dense-вектор.
"""

from __future__ import annotations

import hashlib
import warnings
# Подавление спама transformers: "Accessing `__path__` from ..."
warnings.filterwarnings("ignore", message="Accessing.*__path__.*")
import json
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from src.config import get_secret

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Реестр моделей эмбеддингов
# ══════════════════════════════════════════════════════════════════════

# Каждая модель: {id, name, provider, dim, type, description}
DENSE_TEXT_MODELS = {
    "e5-small": {
        "id": "intfloat/multilingual-e5-small",
        "name": "E5 Small (мультиязычная)",
        "provider": "huggingface",
        "dim": 384,
        "description": "⚡ Лёгкая (~470 МБ). Быстрая на CPU. Рекомендуется для старта.",
    },
    "e5-base": {
        "id": "intfloat/multilingual-e5-base",
        "name": "E5 Base (мультиязычная)",
        "provider": "huggingface",
        "dim": 768,
        "description": "Сбалансированная (~1.1 ГБ). Хорошее качество/скорость.",
    },
    "e5-large": {
        "id": "intfloat/multilingual-e5-large",
        "name": "E5 Large (мультиязычная)",
        "provider": "huggingface",
        "dim": 1024,
        "description": "Точная (~2.2 ГБ). Медленнее без GPU.",
    },
    "minilm-l12": {
        "id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "name": "MiniLM-L12 (мультиязычная)",
        "provider": "huggingface",
        "dim": 384,
        "description": "Лёгкая (~470 МБ). Хороша для парафраз.",
    },
    "gigachat-embed": {
        "id": "Embeddings",
        "name": "GigaChat Embeddings (Сбер)",
        "provider": "gigachat",
        "dim": 1024,
        "description": "API Сбера. Требует GIGACHAT_SECRET в secrets.",
    },
    "neuroapi-small": {
        "id": "text-embedding-3-small",
        "name": "OpenAI text-embedding-3-small (NeuroAPI)",
        "provider": "neuroapi",
        "dim": 1536,
        "description": "API через NeuroAPI. 1536d, хорошее качество.",
    },
    "neuroapi-large": {
        "id": "text-embedding-3-large",
        "name": "OpenAI text-embedding-3-large (NeuroAPI)",
        "provider": "neuroapi",
        "dim": 3072,
        "description": "API через NeuroAPI. 3072d, максимальное качество.",
    },
}

SPARSE_MODELS = {
    "bm42": {
        "id": "Qdrant/bm42-all-minilm-l6-v2-attentions",
        "name": "BM42 (fastembed)",
        "description": "Sparse-эмбеддинги для лексического поиска. Усиливает dense-поиск ключевыми словами.",
    },
}

# Максимальная размерность среди всех моделей
MAX_DENSE_DIM = max(m["dim"] for m in DENSE_TEXT_MODELS.values())  # 3072


def get_model_info(model_key: str) -> dict:
    """Получить информацию о модели по ключу."""
    if model_key in DENSE_TEXT_MODELS:
        return DENSE_TEXT_MODELS[model_key]
    if model_key in SPARSE_MODELS:
        return SPARSE_MODELS[model_key]
    raise ValueError(f"Неизвестная модель: {model_key}")


def get_model_dimension(model_key: str) -> int:
    """Получить размерность вектора модели."""
    info = get_model_info(model_key)
    return info.get("dim", 0)


# ══════════════════════════════════════════════════════════════════════
# Кэш моделей (singleton per model_key)
# ══════════════════════════════════════════════════════════════════════

_hf_models: dict[str, object] = {}
_sparse_models: dict[str, object] = {}
_gigachat_model: object | None = None
_neuroapi_client: object | None = None


def _get_hf_model(model_id: str):
    """Ленивая загрузка HuggingFace модели (sentence-transformers)."""
    if model_id not in _hf_models:
        import os

        from sentence_transformers import SentenceTransformer

        # Установить HF_TOKEN из секретов проекта для авторизованных загрузок
        hf_token = get_secret("HF_TOKEN")
        if hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = hf_token
            logger.info("HF_TOKEN установлен из секретов проекта")

        logger.info("Загрузка HF модели '%s'...", model_id)
        model = SentenceTransformer(model_id, token=hf_token or None)
        _hf_models[model_id] = model
        logger.info("HF модель загружена. Размерность: %d", model.get_embedding_dimension())
    return _hf_models[model_id]


def _get_sparse_model(model_id: str):
    """Ленивая загрузка sparse-модели (fastembed)."""
    if model_id not in _sparse_models:
        import os

        from fastembed import SparseTextEmbedding

        # Установить HF_TOKEN из секретов проекта
        hf_token = get_secret("HF_TOKEN")
        if hf_token and not os.environ.get("HF_TOKEN"):
            os.environ["HF_TOKEN"] = hf_token
            logger.info("HF_TOKEN установлен из секретов проекта (sparse)")

        logger.info("Загрузка sparse-модели '%s'...", model_id)
        model = SparseTextEmbedding(model_name=model_id)
        _sparse_models[model_id] = model
        logger.info("Sparse-модель загружена.")
    return _sparse_models[model_id]


def _get_gigachat_model():
    """Ленивая загрузка GigaChat модели."""
    global _gigachat_model
    if _gigachat_model is None:
        from langchain_community.embeddings import GigaChatEmbeddings

        auth_token = get_secret("GIGACHAT_AUTH_TOKEN")
        secret = get_secret("GIGACHAT_SECRET")

        # Приоритет: GIGACHAT_AUTH_TOKEN (base64 client_id:client_secret) > GIGACHAT_SECRET (UUID)
        if auth_token:
            credentials = auth_token
            logger.info("GigaChat: используем GIGACHAT_AUTH_TOKEN (base64)")
        elif secret:
            credentials = secret
            logger.info("GigaChat: используем GIGACHAT_SECRET (raw)")
        else:
            raise ValueError("GIGACHAT_AUTH_TOKEN или GIGACHAT_SECRET не задан в секретах")

        logger.info("Подключение к GigaChat Embeddings...")
        _gigachat_model = GigaChatEmbeddings(
            credentials=credentials,
            verify_ssl_certs=False,
            scope="GIGACHAT_API_PERS",
            timeout=30,
        )
        logger.info("GigaChat модель создана.")
    return _gigachat_model


def _get_neuroapi_client():
    """Ленивая загрузка NeuroAPI клиента (OpenAI-compatible)."""
    global _neuroapi_client
    if _neuroapi_client is None:
        from openai import OpenAI

        api_key = get_secret("NEUROAPI_API_KEY")
        if not api_key:
            raise ValueError("NEUROAPI_API_KEY не задан в секретах")

        _neuroapi_client = OpenAI(
            base_url="https://neuroapi.host/v1",
            api_key=api_key,
        )
        logger.info("NeuroAPI клиент создан.")
    return _neuroapi_client


# ══════════════════════════════════════════════════════════════════════
# Генерация эмбеддингов
# ══════════════════════════════════════════════════════════════════════

def generate_dense_embeddings(
    texts: list[str],
    model_key: str = "e5-small",
    batch_size: int = 32,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable] = None,
) -> list[list[float]]:
    """Генерация dense-эмбеддингов для списка текстов.

    Автоматически выбирает провайдера по model_key.

    Args:
        texts: Список текстов для векторизации.
        model_key: Ключ модели из DENSE_TEXT_MODELS.
        batch_size: Размер батча.
        cancel_event: Событие отмены.

    Returns:
        Список векторов (каждый — list[float]).
    """
    if not texts:
        return []

    info = get_model_info(model_key)
    provider = info["provider"]
    model_id = info["id"]

    logger.info("Генерация dense-эмбеддингов: provider=%s, model=%s, texts=%d", provider, model_id, len(texts))

    if provider == "huggingface":
        return _gen_dense_hf(texts, model_id, batch_size, cancel_event, progress_callback)
    elif provider == "gigachat":
        return _gen_dense_gigachat(texts, batch_size, cancel_event, progress_callback)
    elif provider == "neuroapi":
        return _gen_dense_neuroapi(texts, model_id, batch_size, cancel_event, progress_callback)
    else:
        raise ValueError(f"Неизвестный провайдер: {provider}")


def _gen_dense_hf(
    texts: list[str],
    model_id: str,
    batch_size: int,
    cancel_event: Optional[threading.Event],
    progress_callback: Optional[Callable] = None,
) -> list[list[float]]:
    """Dense-эмбеддинги через sentence-transformers."""
    model = _get_hf_model(model_id)

    # E5 модели требуют префикс "passage: " для текстов-пассажей
    if "e5" in model_id.lower():
        prefixed = [f"passage: {t}" if not t.startswith("passage: ") else t for t in texts]
    else:
        prefixed = texts

    total_batches = (len(prefixed) + batch_size - 1) // batch_size
    all_embeddings: list[list[float]] = []
    for i in range(0, len(prefixed), batch_size):
        if cancel_event and cancel_event.is_set():
            logger.warning("Dense HF: отменено")
            return all_embeddings
        batch = prefixed[i:i + batch_size]
        embeddings = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        all_embeddings.extend(embeddings.tolist())
        batch_num = i // batch_size + 1
        logger.info("Dense HF: батч %d/%d", batch_num, total_batches)
        if progress_callback:
            progress_callback(batch_num / total_batches, f"dense HF: батч {batch_num}/{total_batches}")

    logger.info("Dense HF: %d векторов", len(all_embeddings))
    return all_embeddings


def _gen_dense_gigachat(
    texts: list[str],
    batch_size: int,
    cancel_event: Optional[threading.Event],
    progress_callback: Optional[Callable] = None,
) -> list[list[float]]:
    """Dense-эмбеддинги через GigaChat API."""
    model = _get_gigachat_model()
    # GigaChat ограничивает 514 токенов ~1900 символов
    max_chars = 1900
    truncated = [t[:max_chars] if len(t) > max_chars else t for t in texts]

    total_batches = (len(truncated) + batch_size - 1) // batch_size
    all_embeddings: list[list[float]] = []
    for i in range(0, len(truncated), batch_size):
        if cancel_event and cancel_event.is_set():
            logger.warning("Dense GigaChat: отменено")
            return all_embeddings
        batch = truncated[i:i + batch_size]
        logger.info("Dense GigaChat: вызов embed_documents (%d текстов)...", len(batch))
        embeddings = model.embed_documents(batch)
        all_embeddings.extend(embeddings)
        batch_num = i // batch_size + 1
        logger.info("Dense GigaChat: батч %d/%d (%d векторов)", batch_num, total_batches, len(embeddings))
        if progress_callback:
            progress_callback(batch_num / total_batches, f"GigaChat dense: батч {batch_num}/{total_batches}")

    logger.info("Dense GigaChat: %d векторов", len(all_embeddings))
    return all_embeddings


def _gen_dense_neuroapi(
    texts: list[str],
    model_id: str,
    batch_size: int,
    cancel_event: Optional[threading.Event],
    progress_callback: Optional[Callable] = None,
) -> list[list[float]]:
    """Dense-эмбеддинги через NeuroAPI (OpenAI-compatible)."""
    client = _get_neuroapi_client()
    total_batches = (len(texts) + batch_size - 1) // batch_size
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        if cancel_event and cancel_event.is_set():
            logger.warning("Dense NeuroAPI: отменено")
            return all_embeddings
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model_id, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        batch_num = i // batch_size + 1
        logger.info("Dense NeuroAPI: батч %d/%d", batch_num, total_batches)
        if progress_callback:
            progress_callback(batch_num / total_batches, f"NeuroAPI dense: батч {batch_num}/{total_batches}")

    logger.info("Dense NeuroAPI: %d векторов", len(all_embeddings))
    return all_embeddings




def generate_sparse_embeddings(
    texts: list[str],
    model_key: str = "bm42",
    batch_size: int = 32,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable] = None,
) -> list[dict]:
    """Генерация sparse-эмбеддингов через fastembed.

    Returns:
        Список словарей {"indices": list[int], "values": list[float]}.
    """
    if not texts:
        return []

    info = get_model_info(model_key)
    model_id = info["id"]
    model = _get_sparse_model(model_id)

    logger.info("Генерация sparse-эмбеддингов: model=%s, texts=%d", model_id, len(texts))

    all_sparse: list[dict] = []
    for i in range(0, len(texts), batch_size):
        if cancel_event and cancel_event.is_set():
            logger.warning("Sparse: отменено")
            return all_sparse

        batch = texts[i:i + batch_size]
        for sparse_embedding in model.embed(batch):
            all_sparse.append({
                "indices": sparse_embedding.indices.tolist(),
                "values": sparse_embedding.values.tolist(),
            })

        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info("Sparse: батч %d/%d", batch_num, total_batches)
        if progress_callback:
            progress_callback(batch_num / total_batches, f"Sparse: батч {batch_num}/{total_batches}")

    logger.info("Sparse: %d векторов", len(all_sparse))
    return all_sparse


# ══════════════════════════════════════════════════════════════════════
# Qdrant: клиент, DDL, DML
# ══════════════════════════════════════════════════════════════════════

def get_qdrant_client(host: str = "localhost", port: int = 6333):
    """Создать клиент Qdrant (не кэшируется — вызывать через @st.cache_resource в UI)."""
    from qdrant_client import QdrantClient
    logger.info("Подключение к Qdrant: %s:%d", host, port)
    return QdrantClient(host=host, port=port)


def ping_qdrant(host: str = "localhost", port: int = 6333) -> tuple[bool, str]:
    """Проверить доступность Qdrant.

    Returns:
        (ok, message)
    """
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port, timeout=5)
        info = client.get_collections()
        return True, f"Подключено. Коллекций: {len(info.collections)}"
    except Exception as e:
        return False, f"Ошибка подключения: {e}"


def ensure_collection(
    client,
    collection_name: str = "book_chunks",
    dense_dim: int = 384,
    recreate: bool = False,
) -> bool:
    """Создать коллекцию с одним dense вектором + optional sparse.

    Схема:
      - dense (named):  один dense-вектор для ВСЕХ типов контента
      - text_sparse:    sparse-вектор (BM42) для гибридного поиска

    Args:
        client: QdrantClient.
        collection_name: Имя коллекции.
        dense_dim: Размерность dense-вектора.
        recreate: Если True — пересоздать коллекцию.

    Returns:
        True если коллекция готова.
    """
    from qdrant_client.models import Distance, SparseVectorParams, VectorParams

    existing = client.get_collections().collections
    names = [c.name for c in existing]

    if collection_name in names:
        if recreate:
            logger.warning("Коллекция '%s' существует — пересоздаём", collection_name)
            client.delete_collection(collection_name)
        else:
            logger.info("Коллекция '%s' уже существует", collection_name)
            return True

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dense_dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "text_sparse": SparseVectorParams(),
        },
    )
    logger.info(
        "Коллекция '%s' создана: dense=%dd, text_sparse=sparse",
        collection_name, dense_dim,
    )
    return True


def create_payload_indexes(client, collection_name: str = "book_chunks") -> None:
    """Создать payload-индексы для пре-фильтрации."""
    from qdrant_client.models import PayloadSchemaType, TextIndexParams, TokenizerType

    # book_name — keyword-индекс
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="book_name",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'book_name'")
    except Exception as e:
        logger.debug("Индекс 'book_name': %s", e)

    # chunk_type — keyword-индекс (text / table / image)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="chunk_type",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'chunk_type'")
    except Exception as e:
        logger.debug("Индекс 'chunk_type': %s", e)

    # pages — keyword-индекс (для фильтрации по номерам страниц)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="page_numbers",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'page_numbers'")
    except Exception as e:
        logger.debug("Индекс 'page_numbers': %s", e)

    # author — keyword-индекс (для фильтрации по автору)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="author",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'author'")
    except Exception as e:
        logger.debug("Индекс 'author': %s", e)

    # title — keyword-индекс (для фильтрации по названию книги)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'title'")
    except Exception as e:
        logger.debug("Индекс 'title' уже существует: %s", e)

    # table_refs — keyword-индекс (список упоминаний таблиц в текстовых чанках)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="table_refs",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'table_refs'")
    except Exception as e:
        logger.debug("Индекс 'table_refs' уже существует: %s", e)

    # table_id — keyword-индекс (идентификатор таблицы в табличном чанке)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="table_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'table_id'")
    except Exception as e:
        logger.debug("Индекс 'table_id': %s", e)

    # figure_refs — keyword-индекс (список упоминаний рисунков в текстовых чанках)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="figure_refs",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'figure_refs'")
    except Exception as e:
        logger.debug("Индекс 'figure_refs' уже существует: %s", e)

    # figure_id — keyword-индекс (идентификатор рисунка в чанке-описании)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="figure_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'figure_id'")
    except Exception as e:
        logger.debug("Индекс 'figure_id': %s", e)

    # year — keyword-индекс (для фильтрации по году издания)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="year",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info("Создан keyword-индекс для 'year'")
    except Exception as e:
        logger.debug("Индекс 'year': %s", e)

    # text — текстовый индекс (для полнотекстового поиска)
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="text",
            field_schema=TextIndexParams(
                type="text",
                tokenizer=TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
            ),
        )
        logger.info("Создан текстовый индекс для 'text'")
    except Exception as e:
        logger.debug("Индекс 'text': %s", e)


# ══════════════════════════════════════════════════════════════════════
# Загрузка чанков из JSON
# ══════════════════════════════════════════════════════════════════════

def load_chunks_from_json(file_path: str | Path) -> list[dict]:
    """Загрузить чанки из JSON-файла (наш формат).

    Каждый чанк — dict с ключами:
      - metadata.headers: dict[int -> str]
      - metadata.page_numbers: list[int]
      - text: str
      - chunk_type: str (text / table / figure)
    """
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        logger.warning("Файл %s: ожидается список, получен %s", file_path.name, type(data).__name__)
        return []

    logger.info("Загружено %d чанков из %s", len(data), file_path.name)
    return data


def find_book_chunk_dirs(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти все папки книг с chunks/text_chunks/ внутри.

    Returns:
        Список (book_name, text_chunks_dir).
    """
    results: list[tuple[str, Path]] = []
    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        text_chunks_dir = book_dir / "chunks" / "text_chunks"
        if text_chunks_dir.is_dir():
            json_files = list(text_chunks_dir.glob("*.json"))
            if json_files:
                results.append((book_dir.name, text_chunks_dir))

    return results


def find_book_table_dirs(base_dir: Path) -> list[tuple[str, Path]]:
    """Найти все папки книг с chunks/table_chunks/ внутри.

    Returns:
        Список (book_name, table_chunks_dir).
    """
    results: list[tuple[str, Path]] = []
    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        table_chunks_dir = book_dir / "chunks" / "table_chunks"
        if table_chunks_dir.is_dir():
            json_files = list(table_chunks_dir.glob("*.json"))
            if json_files:
                results.append((book_dir.name, table_chunks_dir))

    return results


def find_book_image_dirs(base_dir: Path) -> list[tuple[str, Path, Path]]:
    """Найти все папки книг с parsed/images/.

    Returns:
        Список (book_name, images_dir, descriptions_file).
    """
    results: list[tuple[str, Path, Path]] = []
    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue
        images_dir = book_dir / "parsed" / "images"
        desc_file = book_dir / "parsed" / "images_descriptions.json"
        if images_dir.is_dir() or desc_file.exists():
            results.append((book_dir.name, images_dir, desc_file))

    return results


# ══════════════════════════════════════════════════════════════════════
# Подготовка текста для векторизации
# ══════════════════════════════════════════════════════════════════════

def prepare_text_for_embedding(chunk: dict) -> str:
    """Подготовить текст чанка для векторизации.

    Берёт только заголовок самого глубокого уровня (последний в иерархии)
    и прибавляет к чистому тексту чанка.
    Заголовки (#-строки) уже убраны из chunk["text"] при чанковании.
    Полный breadcrumb всех уровней в вектор не включается:
    он «размывает» вектор.
    """
    headers = chunk.get("metadata", {}).get("headers", {})
    text = chunk.get("text", "")

    # Берём только заголовок самого глубокого уровня
    header_str = ""
    if headers:
        max_level = max(int(k) for k in headers if str(k).isdigit())
        header_str = headers.get(max_level, "")

    # Собираем: заголовок последнего уровня + чистый текст
    parts = []
    if header_str:
        parts.append(header_str)
    parts.append(text.strip())

    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# Upsert в Qdrant
# ══════════════════════════════════════════════════════════════════════

def _make_point_id(text: str, book_name: str, chunk_index: int) -> int:
    """Генерировать детерминированный ID точки из текста + книга + индекс.

    SHA256 → первые 13 hex → 52-bit integer (безопасно для JS).
    """
    raw = f"{book_name}::{chunk_index}::{text[:200]}"
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:13], 16)


def upsert_points(
    client,
    collection_name: str,
    items: list[dict],
    dense_embeddings: list[list[float]],
    sparse_embeddings: Optional[list[dict]] = None,
    book_name: str = "",
    id_offset: int = 0,
    batch_size: int = 64,
    cancel_event: Optional[threading.Event] = None,
) -> int:
    """Записать точки (текст/таблицы/изображения) + эмбеддинги в Qdrant батчами.

    Единая функция для всех типов контента. Каждая точка хранит только
    один реальный dense-вектор (без нулевых заглушек).

    Args:
        items: Список словарей с payload (text, chunk_type, headers, ...).
        dense_embeddings: Dense-векторы (по одному на item).
        sparse_embeddings: Sparse-векторы (опционально).
        book_name: Имя книги для payload.
        id_offset: Смещение для генерации ID (100000 для изображений).
        batch_size: Размер батча upsert.
        cancel_event: Событие отмены.

    Returns:
        Количество записанных точек.
    """
    from qdrant_client.models import PointStruct, SparseVector

    if len(items) != len(dense_embeddings):
        raise ValueError(f"Items ({len(items)}) != dense-векторов ({len(dense_embeddings)})")

    has_sparse = sparse_embeddings is not None
    total_upserted = 0

    for i in range(0, len(items), batch_size):
        if cancel_event and cancel_event.is_set():
            logger.warning("Upsert отменён. Записано: %d", total_upserted)
            break

        batch_items = items[i:i + batch_size]
        batch_dense = dense_embeddings[i:i + batch_size]
        batch_sparse = sparse_embeddings[i:i + batch_size] if has_sparse else [None] * len(batch_items)

        points = []
        for j, (item, dense, sparse) in enumerate(zip(batch_items, batch_dense, batch_sparse)):
            text = item.get("text", "")
            pid = _make_point_id(text, book_name, id_offset + i + j)

            vector_dict: dict = {"dense": dense}
            if sparse is not None:
                vector_dict["text_sparse"] = SparseVector(
                    indices=sparse["indices"],
                    values=sparse["values"],
                )

            points.append(PointStruct(id=pid, vector=vector_dict, payload=item))

        client.upsert(collection_name=collection_name, points=points, wait=True)
        total_upserted += len(points)
        logger.info("Upsert: %d/%d точек (книга: %s)", total_upserted, len(items), book_name)

    logger.info("Итого записано для '%s': %d точек", book_name, total_upserted)
    return total_upserted


# Backward compatibility alias
upsert_text_chunks = upsert_points


def validate_collection_dims(
    client,
    collection_name: str,
    expected_dense_dim: int,
) -> tuple[bool, str]:
    """Проверить, совпадает ли размерность dense-вектора коллекции с ожидаемой.

    Returns:
        (ok, message) — ok=True если всё совпадает или коллекция не существует.
    """
    existing = client.get_collections().collections
    names = [c.name for c in existing]

    if collection_name not in names:
        return True, "Коллекция не существует — будет создана"

    try:
        info = client.get_collection(collection_name)
        vectors = info.config.params.vectors

        # Поддержка обоих схем: старая (text_dense) и новая (dense)
        actual_dense = vectors.get("dense") or vectors.get("text_dense")
        if actual_dense and actual_dense.size != expected_dense_dim:
            msg = (
                f"Несовпадение размерностей! dense: ожидается {expected_dense_dim}d, "
                f"в коллекции {actual_dense.size}d. "
                f"Поставьте галочку «Пересоздать коллекцию» или выберите другую модель."
            )
            return False, msg

        return True, "Размерности совпадают"
    except Exception as e:
        return False, f"Ошибка проверки коллекции: {e}"


def get_collection_info(client, collection_name: str) -> dict:
    """Получить информацию о коллекции.

    Returns:
        {name, points_count, vectors_config, status}
    """
    try:
        info = client.get_collection(collection_name)
        vectors = {}
        for name, cfg in info.config.params.vectors.items():
            vectors[name] = {"size": cfg.size, "distance": str(cfg.distance)}
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors": vectors,
            "status": str(info.status),
        }
    except Exception as e:
        return {"name": collection_name, "error": str(e)}


def list_collections(client) -> list[dict]:
    """Получить список коллекций с информацией."""
    collections = client.get_collections().collections
    result = []
    for c in collections:
        result.append(get_collection_info(client, c.name))
    return result


def delete_collection(client, collection_name: str) -> bool:
    """Удалить коллекцию."""
    try:
        client.delete_collection(collection_name)
        logger.info("Коллекция '%s' удалена", collection_name)
        return True
    except Exception as e:
        logger.error("Ошибка удаления коллекции '%s': %s", collection_name, e)
        return False


# ══════════════════════════════════════════════════════════════════════
# Управление моделями: проверка кэша, скачивание, статус
# ══════════════════════════════════════════════════════════════════════

def _is_hf_cached(model_id: str) -> bool:
    """Проверить, скачана ли HuggingFace модель (есть ли веса в кэше)."""
    sanitized = model_id.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{sanitized}"
    snapshots = cache_dir / "snapshots"
    if not snapshots.exists():
        return False
    for snapshot in snapshots.iterdir():
        if snapshot.is_dir():
            if any(
                (snapshot / f).exists()
                for f in ("model.safetensors", "pytorch_model.bin", "model.onnx")
            ):
                return True
    return False


def _is_fastembed_cached(model_id: str) -> bool:
    """Проверить, скачана ли fastembed модель."""
    # fastembed кэширует в ~/.cache/fastembed/
    fe_cache = Path.home() / ".cache" / "fastembed"
    if not fe_cache.exists():
        return False
    # Ищем любую подпапку содержащую модель
    sanitized = model_id.replace("/", "--").replace("\\", "--")
    for d in fe_cache.iterdir():
        if d.is_dir() and sanitized.lower() in d.name.lower():
            return True
    # Также проверяем HF cache (новые версии fastembed используют его)
    return _is_hf_cached(model_id)


def _get_hf_model_size(model_id: str) -> int:
    """Получить размер HF модели в кэше (байты)."""
    sanitized = model_id.replace("/", "--")
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{sanitized}"
    if not cache_dir.exists():
        return 0
    total = 0
    for f in cache_dir.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def _get_fastembed_model_size(model_id: str) -> int:
    """Получить размер fastembed модели в кэше (байты)."""
    sanitized = model_id.replace("/", "--").replace("\\", "--")
    fe_cache = Path.home() / ".cache" / "fastembed"
    total = 0
    if fe_cache.exists():
        for d in fe_cache.iterdir():
            if d.is_dir() and sanitized.lower() in d.name.lower():
                for f in d.rglob("*"):
                    if f.is_file():
                        try:
                            total += f.stat().st_size
                        except OSError:
                            pass
    # Также учитываем HF cache
    total += _get_hf_model_size(model_id)
    return total


def _fmt_bytes(size: int) -> str:
    """Форматировать размер в человекочитаемый вид."""
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} ТБ"


def get_all_models_status() -> list[dict]:
    """Получить статус всех моделей эмбеддингов.

    Returns:
        Список словарей с ключами:
          key, name, type, provider, model_id, dim, description,
          is_local (bool), cached (bool), size_str (str)
    """
    models = []

    for key, info in DENSE_TEXT_MODELS.items():
        provider = info["provider"]
        is_local = provider == "huggingface"
        cached = _is_hf_cached(info["id"]) if is_local else True  # API = всегда "готово"
        size_bytes = _get_hf_model_size(info["id"]) if is_local else 0
        models.append({
            "key": key,
            "name": info["name"],
            "type": "dense_text",
            "provider": provider,
            "model_id": info["id"],
            "dim": info.get("dim", 0),
            "description": info.get("description", ""),
            "is_local": is_local,
            "cached": cached,
            "size_str": _fmt_bytes(size_bytes) if is_local else "API",
        })

    for key, info in SPARSE_MODELS.items():
        model_id = info["id"]
        cached = _is_fastembed_cached(model_id)
        size_bytes = _get_fastembed_model_size(model_id)
        models.append({
            "key": key,
            "name": info["name"],
            "type": "sparse",
            "provider": "fastembed",
            "model_id": model_id,
            "dim": 0,
            "description": info.get("description", ""),
            "is_local": True,
            "cached": cached,
            "size_str": _fmt_bytes(size_bytes),
        })

    return models


def download_model(model_key: str) -> str:
    """Скачать модель в кэш.

    Returns:
        Статус сообщения.
    """
    if model_key in DENSE_TEXT_MODELS:
        info = DENSE_TEXT_MODELS[model_key]
        if info["provider"] == "huggingface":
            _get_hf_model(info["id"])
            return f"✅ {info['name']} загружена"
        elif info["provider"] == "gigachat":
            _get_gigachat_model()
            return f"✅ GigaChat подключён"
        elif info["provider"] == "neuroapi":
            _get_neuroapi_client()
            return f"✅ NeuroAPI клиент создан"
    elif model_key in SPARSE_MODELS:
        info = SPARSE_MODELS[model_key]
        logger.info("Начинаем скачивание sparse-модели '%s' (%s)...", info["name"], info["id"])
        try:
            _get_sparse_model(info["id"])
            logger.info("Sparse-модель '%s' успешно загружена", info["id"])
        except Exception as e:
            logger.error("Ошибка скачивания sparse-модели '%s': %s", info["id"], e, exc_info=True)
            raise
        return f"✅ {info['name']} загружена"

    raise ValueError(f"Неизвестная модель: {model_key}")
