"""Оркестратор векторизации — запуск через threading с прогрессом и отменой.

Используется UI-страницей pages/6_vectorizer.py.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Callable, Optional

from src.chunking import parse_book_name
from src.vectorizer import (
    create_payload_indexes,
    ensure_collection,
    find_book_chunk_dirs,
    find_book_image_dirs,
    find_book_table_dirs,
    generate_dense_embeddings,
    generate_sparse_embeddings,
    get_model_dimension,
    load_chunks_from_json,
    prepare_text_for_embedding,
    upsert_points,
    validate_collection_dims,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Векторизация текстовых чанков
# ══════════════════════════════════════════════════════════════════════

def run_text_vectorizer(
    client,
    base_dir: Path,
    collection_name: str = "book_chunks",
    dense_model_key: str = "e5-small",
    sparse_enabled: bool = True,
    sparse_model_key: str = "bm42",
    batch_size: int = 32,
    recreate: bool = False,
    progress_callback: Optional[Callable] = None,
    cancel_event: Optional[threading.Event] = None,
    book_names: Optional[list[str]] = None,
) -> dict:
    """Запуск векторизации текстовых чанков.

    Обходит base_dir/*/chunks/text_chunks/*.json, генерирует эмбеддинги, записывает в Qdrant.

    Args:
        client: QdrantClient.
        base_dir: Корневая папка с книгами (data/books).
        collection_name: Имя коллекции Qdrant.
        dense_model_key: Ключ модели dense-эмбеддингов.
        sparse_enabled: Генерировать ли sparse-эмбеддинги.
        sparse_model_key: Ключ sparse-модели.
        batch_size: Размер батча для генерации эмбеддингов.
        recreate: Пересоздать коллекцию.
        progress_callback: Функция(progress_pct, message).
        cancel_event: Событие отмены.
        book_names: Если задано — обработать только указанные книги.

    Returns:
        {"books_processed": int, "total_chunks": int, "total_upserted": int, "errors": list}
    """
    result = {"books_processed": 0, "total_chunks": 0, "total_upserted": 0, "errors": []}

    def _progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    # 1. Найти все книги с чанками
    book_dirs = find_book_chunk_dirs(base_dir)
    if not book_dirs:
        _progress(1.0, "❌ Не найдено папок с чанками")
        return result

    # 1.5. Фильтрация по выбранным книгам
    if book_names:
        book_names_set = set(book_names)
        book_dirs = [(bn, cd) for bn, cd in book_dirs if bn in book_names_set]
        if not book_dirs:
            _progress(1.0, "❌ Выбранные книги не найдены среди книг с чанками")
            return result

    _progress(0.0, f"Найдено книг: {len(book_dirs)}")

    # 2. Создать коллекцию
    dense_dim = get_model_dimension(dense_model_key)

    _progress(0.02, f"Создание коллекции '{collection_name}' (dense={dense_dim}d)...")
    ensure_collection(client, collection_name, dense_dim=dense_dim, recreate=recreate)
    create_payload_indexes(client, collection_name)

    # 2.5. Валидация размерностей (если коллекция уже существует и не пересоздаётся)
    if not recreate:
        dims_ok, dims_msg = validate_collection_dims(client, collection_name, dense_dim)
        if not dims_ok:
            _progress(1.0, f"❌ {dims_msg}")
            result["errors"].append(dims_msg)
            logger.error("Ошибка размерностей: %s", dims_msg)
            return result

    # 3. Обработать каждую книгу
    for book_idx, (book_name, chunks_dir) in enumerate(book_dirs):
        if cancel_event and cancel_event.is_set():
            _progress(-1, "⚠️ Отменено")
            break

        _progress(
            book_idx / len(book_dirs) * 0.95 + 0.05,
            f"📖 Книга {book_idx + 1}/{len(book_dirs)}: {book_name}",
        )

        try:
            _process_book_text(
                client=client,
                collection_name=collection_name,
                book_name=book_name,
                chunks_dir=chunks_dir,
                dense_model_key=dense_model_key,
                sparse_enabled=sparse_enabled,
                sparse_model_key=sparse_model_key,
                batch_size=batch_size,
                cancel_event=cancel_event,
                result=result,
                progress_callback=progress_callback,
                base_pct=book_idx / len(book_dirs) * 0.95 + 0.05,
                pct_range=0.95 / len(book_dirs),
            )
            result["books_processed"] += 1
        except Exception as e:
            logger.error("Ошибка обработки книги '%s': %s", book_name, e)
            result["errors"].append(f"{book_name}: {e}")

    _progress(1.0, f"✅ Готово. Книг: {result['books_processed']}, чанков: {result['total_upserted']}")
    return result


def _process_book_text(
    client,
    collection_name: str,
    book_name: str,
    chunks_dir: Path,
    dense_model_key: str,
    sparse_enabled: bool,
    sparse_model_key: str,
    batch_size: int,
    cancel_event: Optional[threading.Event],
    result: dict,
    progress_callback: Optional[Callable] = None,
    base_pct: float = 0.05,
    pct_range: float = 0.9,
    id_offset: int = 0,
) -> None:
    """Обработка одной книги: загрузка → эмбеддинги → upsert.

    Args:
        progress_callback: Функция(pct, msg) для обновления прогресса.
        base_pct: Начальный процент для этой книги.
        pct_range: Диапазон процентов (0..1) для этой книги.
    """
    def _book_progress(step_pct: float, msg: str):
        """Обёртка: маппит локальный 0..1 в глобальный прогресс."""
        if progress_callback:
            global_pct = base_pct + step_pct * pct_range
            progress_callback(global_pct, msg)

    # ── Шаг 1: Загрузка чанков ─────────────────────────────────────
    _book_progress(0.0, f"📖 {book_name}: загрузка чанков...")

    json_files = sorted(chunks_dir.glob("*.json"))

    all_chunks: list[dict] = []
    for jf in json_files:
        chunks = load_chunks_from_json(jf)
        all_chunks.extend(chunks)

    if not all_chunks:
        logger.warning("Книга '%s': нет чанков", book_name)
        return

    result["total_chunks"] += len(all_chunks)
    logger.info("Книга '%s': %d чанков из %d файлов", book_name, len(all_chunks), len(json_files))

    _book_progress(0.05, f"📖 {book_name}: загружено {len(all_chunks)} чанков из {len(json_files)} файлов")

    # ── Шаг 2: Подготовка текстов ──────────────────────────────────
    texts = [prepare_text_for_embedding(c) for c in all_chunks]
    _book_progress(0.1, f"📖 {book_name}: тексты подготовлены")

    # ── Шаг 3: Dense эмбеддинги ────────────────────────────────────
    _book_progress(0.12, f"📖 {book_name}: генерация dense-эмбеддингов ({len(texts)} текстов)...")
    logger.info("Книга '%s': начало генерации dense-эмбеддингов, модель=%s, батч=%d, текстов=%d",
                book_name, dense_model_key, batch_size, len(texts))

    def _dense_progress(step_pct: float, msg: str):
        """Прогресс внутри dense-генерации (0.12 → 0.55)."""
        _book_progress(0.12 + step_pct * 0.43, f"📖 {book_name}: {msg}")

    dense = generate_dense_embeddings(
        texts, model_key=dense_model_key, batch_size=batch_size,
        cancel_event=cancel_event, progress_callback=_dense_progress,
    )
    if cancel_event and cancel_event.is_set():
        return

    logger.info("Книга '%s': dense-эмбеддингов получено: %d", book_name, len(dense))
    _book_progress(0.55, f"📖 {book_name}: dense-эмбеддинги готовы ({len(dense)} векторов)")

    # ── Шаг 4: Sparse эмбеддинги ───────────────────────────────────
    sparse = None
    if sparse_enabled:
        _book_progress(0.6, f"📖 {book_name}: генерация sparse-эмбеддингов...")
        logger.info("Книга '%s': начало генерации sparse-эмбеддингов, модель=%s", book_name, sparse_model_key)

        def _sparse_progress(step_pct: float, msg: str):
            """Прогресс внутри sparse-генерации (0.6 → 0.85)."""
            _book_progress(0.6 + step_pct * 0.25, f"📖 {book_name}: {msg}")

        sparse = generate_sparse_embeddings(
            texts, model_key=sparse_model_key, batch_size=batch_size,
            cancel_event=cancel_event, progress_callback=_sparse_progress,
        )
        if cancel_event and cancel_event.is_set():
            return

        logger.info("Книга '%s': sparse-эмбеддингов получено: %d", book_name, len(sparse))
        _book_progress(0.85, f"📖 {book_name}: sparse-эмбеддинги готовы ({len(sparse)} векторов)")

    # ── Шаг 5: Подготовка payload и Upsert ──────────────────────────
    _book_progress(0.88, f"📖 {book_name}: запись в Qdrant...")
    logger.info("Книга '%s': начало upsert в Qdrant (%d точек)...", book_name, len(all_chunks))

    # Разбираем автора и название из имени книги
    author, title = parse_book_name(book_name)

    # Формируем payload для каждой точки
    payload_items = []
    for chunk in all_chunks:
        chunk_text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        headers = meta.get("headers", {})
        page_numbers = meta.get("page_numbers", [])
        chunk_type = meta.get("chunk_type", "text")

        # Берём author/title/year из метаданных чанка (если есть), иначе из book_name
        chunk_author = meta.get("author", author)
        chunk_title = meta.get("title", title)
        chunk_year = meta.get("year", "")

        item = {
            "text": chunk_text,
            "book_name": book_name,
            "author": chunk_author,
            "title": chunk_title,
            "year": chunk_year,
            "chunk_type": chunk_type,
            "headers": headers,
            "page_numbers": [str(p) for p in page_numbers],
        }

        if meta.get("chunk_index") is not None:
            item["chunk_index"] = meta["chunk_index"]
        if meta.get("full_content"):
            item["full_content"] = meta["full_content"]
        if meta.get("table_refs"):
            item["table_refs"] = meta["table_refs"]
        if meta.get("table_id"):
            item["table_id"] = meta["table_id"]
        if meta.get("figure_refs"):
            item["figure_refs"] = meta["figure_refs"]
        if meta.get("figure_id"):
            item["figure_id"] = meta["figure_id"]
        payload_items.append(item)

    upserted = upsert_points(
        client=client,
        collection_name=collection_name,
        items=payload_items,
        dense_embeddings=dense,
        sparse_embeddings=sparse,
        book_name=book_name,
        id_offset=id_offset,
        cancel_event=cancel_event,
    )
    result["total_upserted"] += upserted
    _book_progress(1.0, f"📖 {book_name}: ✅ записано {upserted} точек")


# ══════════════════════════════════════════════════════════════════════
# Векторизация табличных чанков
# ══════════════════════════════════════════════════════════════════════

def run_table_vectorizer(
    client,
    base_dir: Path,
    collection_name: str = "book_chunks",
    dense_model_key: str = "e5-small",
    sparse_enabled: bool = True,
    sparse_model_key: str = "bm42",
    batch_size: int = 32,
    recreate: bool = False,
    progress_callback: Optional[Callable] = None,
    cancel_event: Optional[threading.Event] = None,
    book_names: Optional[list[str]] = None,
) -> dict:
    """Запуск векторизации табличных чанков.

    Обходит base_dir/*/chunks/table_chunks/*.json, генерирует эмбеддинги, записывает в Qdrant.
    Табличные чанки используют те же named vectors (dense + text_sparse),
    что и текстовые, но с chunk_type="table" в payload.

    Args:
        client: QdrantClient.
        base_dir: Корневая папка с книгами (data/books).
        collection_name: Имя коллекции Qdrant.
        dense_model_key: Ключ модели dense-эмбеддингов.
        sparse_enabled: Генерировать ли sparse-эмбеддинги.
        sparse_model_key: Ключ sparse-модели.
        batch_size: Размер батча для генерации эмбеддингов.
        recreate: Пересоздать коллекцию.
        progress_callback: Функция(progress_pct, message).
        cancel_event: Событие отмены.
        book_names: Если задано — обработать только указанные книги.

    Returns:
        {"books_processed": int, "total_chunks": int, "total_upserted": int, "errors": list}
    """
    result = {"books_processed": 0, "total_chunks": 0, "total_upserted": 0, "errors": []}

    def _progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    # 1. Найти все книги с табличными чанками
    book_dirs = find_book_table_dirs(base_dir)
    if not book_dirs:
        _progress(1.0, "❌ Не найдено папок с табличными чанками")
        return result

    # 1.5. Фильтрация по выбранным книгам
    if book_names:
        book_names_set = set(book_names)
        book_dirs = [(bn, cd) for bn, cd in book_dirs if bn in book_names_set]
        if not book_dirs:
            _progress(1.0, "❌ Выбранные книги не найдены среди книг с табличными чанками")
            return result

    _progress(0.0, f"Найдено книг с таблицами: {len(book_dirs)}")

    # 2. Создать коллекцию
    dense_dim = get_model_dimension(dense_model_key)

    _progress(0.02, f"Создание коллекции '{collection_name}' (dense={dense_dim}d)...")
    ensure_collection(client, collection_name, dense_dim=dense_dim, recreate=recreate)
    create_payload_indexes(client, collection_name)

    # 2.5. Валидация размерностей (если коллекция уже существует и не пересоздаётся)
    if not recreate:
        dims_ok, dims_msg = validate_collection_dims(client, collection_name, dense_dim)
        if not dims_ok:
            _progress(1.0, f"❌ {dims_msg}")
            result["errors"].append(dims_msg)
            logger.error("Ошибка размерностей: %s", dims_msg)
            return result

    # 3. Обработать каждую книгу
    for book_idx, (book_name, chunks_dir) in enumerate(book_dirs):
        if cancel_event and cancel_event.is_set():
            _progress(-1, "⚠️ Отменено")
            break

        _progress(
            book_idx / len(book_dirs) * 0.95 + 0.05,
            f"📋 Книга {book_idx + 1}/{len(book_dirs)}: {book_name}",
        )

        try:
            _process_book_text(
                client=client,
                collection_name=collection_name,
                book_name=book_name,
                chunks_dir=chunks_dir,
                dense_model_key=dense_model_key,
                sparse_enabled=sparse_enabled,
                sparse_model_key=sparse_model_key,
                batch_size=batch_size,
                cancel_event=cancel_event,
                result=result,
                progress_callback=progress_callback,
                base_pct=book_idx / len(book_dirs) * 0.95 + 0.05,
                pct_range=0.95 / len(book_dirs),
                id_offset=200_000,
            )
            result["books_processed"] += 1
        except Exception as e:
            logger.error("Ошибка обработки таблиц книги '%s': %s", book_name, e)
            result["errors"].append(f"{book_name}: {e}")

    _progress(1.0, f"✅ Готово. Книг: {result['books_processed']}, таблиц: {result['total_upserted']}")
    return result


# ══════════════════════════════════════════════════════════════════════
# Векторизация изображений
# ══════════════════════════════════════════════════════════════════════

def run_image_vectorizer(
    client,
    base_dir: Path,
    collection_name: str = "book_chunks",
    dense_model_key: str = "e5-small",
    batch_size: int = 32,
    recreate: bool = False,
    progress_callback: Optional[Callable] = None,
    cancel_event: Optional[threading.Event] = None,
    book_names: Optional[list[str]] = None,
) -> dict:
    """Запуск векторизации описаний изображений.

    Обходит base_dir/*/parsed/images_descriptions.json.
    Описания векторизуются той же dense-моделью, что и текст.

    Args:
        dense_model_key: Ключ модели из DENSE_TEXT_MODELS.
        book_names: Если задано — обработать только указанные книги.

    Returns:
        {"books_processed": int, "total_images": int, "total_upserted": int, "errors": list}
    """
    result = {"books_processed": 0, "total_images": 0, "total_upserted": 0, "errors": []}

    def _progress(pct: float, msg: str):
        if progress_callback:
            progress_callback(pct, msg)

    # 1. Найти все книги с изображениями
    image_dirs = find_book_image_dirs(base_dir)
    if not image_dirs:
        _progress(1.0, "❌ Не найдено папок с изображениями")
        return result

    # 1.5. Фильтрация по выбранным книгам
    if book_names:
        book_names_set = set(book_names)
        image_dirs = [(bn, imd, df) for bn, imd, df in image_dirs if bn in book_names_set]
        if not image_dirs:
            _progress(1.0, "❌ Выбранные книги не найдены среди книг с изображениями")
            return result

    _progress(0.0, f"Найдено книг с изображениями: {len(image_dirs)}")

    # 2. Размерность модели
    dense_dim = get_model_dimension(dense_model_key)

    _progress(0.02, f"Проверка коллекции '{collection_name}'...")
    ensure_collection(client, collection_name, dense_dim=dense_dim, recreate=recreate)
    create_payload_indexes(client, collection_name)

    # 2.5. Валидация размерностей
    if not recreate:
        dims_ok, dims_msg = validate_collection_dims(client, collection_name, dense_dim)
        if not dims_ok:
            _progress(1.0, f"❌ {dims_msg}")
            result["errors"].append(dims_msg)
            logger.error("Ошибка размерностей: %s", dims_msg)
            return result

    # 3. Обработать каждую книгу
    for book_idx, (book_name, images_dir, desc_file) in enumerate(image_dirs):
        if cancel_event and cancel_event.is_set():
            _progress(-1, "⚠️ Отменено")
            break

        _progress(
            book_idx / len(image_dirs) * 0.95 + 0.05,
            f"🖼️ Книга {book_idx + 1}/{len(image_dirs)}: {book_name}",
        )

        try:
            _process_book_images(
                client=client,
                collection_name=collection_name,
                book_name=book_name,
                images_dir=images_dir,
                desc_file=desc_file,
                dense_model_key=dense_model_key,
                batch_size=batch_size,
                cancel_event=cancel_event,
                result=result,
            )
            result["books_processed"] += 1
        except Exception as e:
            logger.error("Ошибка обработки изображений '%s': %s", book_name, e)
            result["errors"].append(f"{book_name}: {e}")

    _progress(1.0, f"✅ Готово. Книг: {result['books_processed']}, изображений: {result['total_upserted']}")
    return result


def _process_book_images(
    client,
    collection_name: str,
    book_name: str,
    images_dir: Path,
    desc_file: Path,
    dense_model_key: str,
    batch_size: int,
    cancel_event: Optional[threading.Event],
    result: dict,
) -> None:
    """Обработка изображений одной книги — векторизация описаний через dense-модель."""
    if not desc_file.exists():
        logger.warning("Книга '%s': нет файла %s", book_name, desc_file.name)
        return

    with open(desc_file, "r", encoding="utf-8") as f:
        descriptions_data = json.load(f)

    if isinstance(descriptions_data, dict):
        items = list(descriptions_data.items())
    elif isinstance(descriptions_data, list):
        items = [(item.get("name", f"img_{i}"), item.get("description", "")) for i, item in enumerate(descriptions_data)]
    else:
        logger.warning("Книга '%s': неизвестный формат описаний", book_name)
        return

    if not items:
        return

    image_names = [name for name, _ in items]
    descriptions = [desc for _, desc in items]
    image_paths = [str(images_dir / name) for name in image_names]

    result["total_images"] += len(descriptions)

    # Dense эмбеддинги для описаний (той же моделью, что и текст)
    dense = generate_dense_embeddings(
        texts=descriptions,
        model_key=dense_model_key,
        batch_size=batch_size,
        cancel_event=cancel_event,
    )
    if cancel_event and cancel_event.is_set():
        return

    # Разбираем автора и название из имени книги
    author, title = parse_book_name(book_name)

    # Формируем payload
    payload_items = []
    for desc, img_path in zip(descriptions, image_paths):
        payload_items.append({
            "text": desc,
            "book_name": book_name,
            "author": author,
            "title": title,
            "year": "",
            "chunk_type": "image",
            "image_path": img_path,
            "headers": {},
            "page_numbers": [],
            "full_content": desc,
        })

    # Upsert (id_offset=100000 чтобы не пересекались с текстовыми/табличными)
    upserted = upsert_points(
        client=client,
        collection_name=collection_name,
        items=payload_items,
        dense_embeddings=dense,
        book_name=book_name,
        id_offset=100000,
        cancel_event=cancel_event,
    )
    result["total_upserted"] += upserted
