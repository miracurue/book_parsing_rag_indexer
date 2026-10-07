"""Утилиты для сохранения результатов работы скриптов.

Стратегия хранения:
- Яндекс.Диск — основное хранилище (результат всегда загружается туда)
- Локальная папка data/ — кэш/резерв (сохраняется параллельно)

Универсальный компонент — используется всеми оркестраторами.
"""

import logging
from io import BytesIO
from pathlib import Path

from src.config import DATA_PROCESSED_DIR
from src.sources import DataSource, LocalSource

logger = logging.getLogger(__name__)


def save_result(
    name: str,
    buffer: BytesIO,
    dest_source: DataSource | None = None,
    local_cache_dir: Path | str | None = None,
) -> None:
    """Сохранить результат обработки.

    Всегда сохраняет локально. Если dest_source передан — загружает туда тоже.

    Args:
        name: путь файла (может содержать подпапки, напр. 'book/003.pdf')
        buffer: данные файла
        dest_source: адаптер назначения (LocalSource, YandexSource и т.д.)
        local_cache_dir: папка для локального кэша.
            По умолчанию — data/processed/
    """
    # Локальный кэш
    cache_dir = Path(local_cache_dir) if local_cache_dir else DATA_PROCESSED_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / name
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(buffer.getvalue())
    logger.info("Локальный кэш: %s", cache_path)

    # Основное хранилище (если передано и это не локальное)
    if dest_source is not None:
        buffer.seek(0)
        dest_source.upload(name, buffer)

    # Возвращаем буфер в начало на всякий случай
    buffer.seek(0)


def load_file(
    name: str,
    source: DataSource,
    local_cache_dir: Path | str | None = None,
) -> BytesIO:
    """Загрузить файл из кэша или из источника.

    Сначала проверяет локальный кэш. Если файла нет — скачивает из источника.

    Args:
        name: имя файла
        source: адаптер-источник данных
        local_cache_dir: папка локального кэша

    Returns:
        BytesIO с данными файла
    """
    cache_dir = Path(local_cache_dir) if local_cache_dir else DATA_PROCESSED_DIR
    cache_path = cache_dir / name

    if cache_path.exists():
        logger.debug("Загружен из кэша: %s", cache_path)
        return BytesIO(cache_path.read_bytes())

    buf = source.download(name)
    # Сохраняем в кэш
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(buf.getvalue())
    logger.debug("Скачан и закэширован: %s", name)
    buf.seek(0)
    return buf


def check_exists(
    name: str,
    source: DataSource | None = None,
    local_cache_dir: Path | str | None = None,
) -> bool:
    """Проверить существование файла в кэше или источнике."""
    cache_dir = Path(local_cache_dir) if local_cache_dir else DATA_PROCESSED_DIR
    cache_path = cache_dir / name

    if cache_path.exists():
        return True

    if source is not None:
        try:
            return source.check_exists(name)
        except NotImplementedError:
            pass

    return False