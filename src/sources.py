"""Адаптеры источников данных: локальная папка и Яндекс.Диск.

Оба адаптера предоставляют одинаковый набор методов:
- list_files(extensions) → list[str]
- download(name) → BytesIO
- upload(name, buffer)           — поддерживает подпапки (name может содержать /)

Универсальный компонент — используется всеми скриптами обработки.
"""

import logging
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

from src.config import get_secret

logger = logging.getLogger(__name__)

# Расширения файлов
JPEG_EXTS = (".jpg", ".jpeg")
PDF_EXTS = (".pdf",)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tiff", ".bmp")


class DataSource(ABC):
    """Базовый интерфейс источника данных."""

    @abstractmethod
    def list_files(self, extensions: tuple[str, ...] = PDF_EXTS) -> list[str]:
        """Вернуть список имён файлов с указанными расширениями."""

    @abstractmethod
    def download(self, name: str) -> BytesIO:
        """Скачать файл по имени и вернуть BytesIO."""

    @abstractmethod
    def upload(self, name: str, buffer: BytesIO) -> None:
        """Загрузить обработанный файл.

        Args:
            name: имя файла, может содержать подпапку (напр. 'book_name/003.pdf')
            buffer: данные файла
        """

    def check_exists(self, name: str) -> bool:
        """Проверить, существует ли файл."""
        raise NotImplementedError


# ── Локальная файловая система ──────────────────────────────────────


class LocalSource(DataSource):
    """Чтение/запись из локальных директорий."""

    def __init__(self, source_dir: str, dest_dir: str) -> None:
        self._source_dir = Path(source_dir)
        self._dest_dir = Path(dest_dir)

        if not self._source_dir.exists():
            raise FileNotFoundError(
                f"Исходная папка не найдена: {self._source_dir}"
            )

        self._dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info("LocalSource: src=%s  dest=%s", self._source_dir, self._dest_dir)

    def list_files(self, extensions: tuple[str, ...] = PDF_EXTS) -> list[str]:
        names = sorted(
            f.name
            for f in self._source_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )
        logger.info(
            "Найдено %d файлов (%s) в %s", len(names), extensions, self._source_dir
        )
        return names

    def download(self, name: str) -> BytesIO:
        path = self._source_dir / name
        buf = BytesIO(path.read_bytes())
        logger.debug("Скачан локально: %s (%d байт)", name, buf.getbuffer().nbytes)
        return buf

    def upload(self, name: str, buffer: BytesIO) -> None:
        dest_path = self._dest_dir / name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(buffer.getvalue())
        logger.debug("Сохранён локально: %s", dest_path)

    def check_exists(self, name: str) -> bool:
        return (self._dest_dir / name).exists()


# ── Яндекс.Диск ────────────────────────────────────────────────────


class YandexSource(DataSource):
    """Чтение/запись через Яндекс.Диск (yadisk)."""

    def __init__(
        self,
        source_dir: str,
        dest_dir: str,
        token: str | None = None,
    ) -> None:
        import yadisk

        if not token:
            token = get_secret("YANDEX_DISK_TOKEN")
        if not token:
            raise ValueError("YANDEX_DISK_TOKEN не задан")

        self._client = yadisk.YaDisk(token=token)

        if not self._client.check_token():
            raise ValueError("Токен Яндекс.Диска невалиден")

        self._source_dir = source_dir.rstrip("/")
        self._dest_dir = dest_dir.rstrip("/")

        # Создаём целевую папку при необходимости
        if not self._client.exists(self._dest_dir):
            self._client.mkdir(self._dest_dir)
            logger.info("Создана папка на Яндекс.Диске: %s", self._dest_dir)

        logger.info("YandexSource: src=%s  dest=%s", self._source_dir, self._dest_dir)

    def list_files(self, extensions: tuple[str, ...] = PDF_EXTS) -> list[str]:
        items = list(self._client.listdir(self._source_dir))
        names = sorted(
            it.name
            for it in items
            if it.type == "file" and Path(it.name).suffix.lower() in extensions
        )
        logger.info(
            "Найдено %d файлов (%s) в %s", len(names), extensions, self._source_dir
        )
        return names

    def download(self, name: str) -> BytesIO:
        remote_path = f"{self._source_dir}/{name}"
        buf = BytesIO()
        self._client.download(remote_path, buf)
        buf.seek(0)
        logger.debug("Скачан с Яндекс.Диска: %s (%d байт)", name, buf.getbuffer().nbytes)
        return buf

    def upload(self, name: str, buffer: BytesIO) -> None:
        dest_path = f"{self._dest_dir}/{name}"
        # Создаём родительскую папку, если name содержит подпапку
        parent = "/".join(dest_path.split("/")[:-1])
        if parent and not self._client.exists(parent):
            self._client.mkdir(parent)
            logger.debug("Создана папка на Яндекс.Диске: %s", parent)

        buffer.seek(0)
        self._client.upload(buffer, dest_path, overwrite=True)
        logger.debug("Загружен на Яндекс.Диск: %s", dest_path)

    def check_exists(self, name: str) -> bool:
        return self._client.exists(f"{self._dest_dir}/{name}")


# ── Комбинированный источник ────────────────────────────────────────


class CompositeSource(DataSource):
    """Комбинирует независимые адаптеры для чтения и записи.

    Позволяет читать из одного хранилища и писать в другое.
    """

    def __init__(self, reader: DataSource, writer: DataSource) -> None:
        self._reader = reader
        self._writer = writer

    def list_files(self, extensions: tuple[str, ...] = PDF_EXTS) -> list[str]:
        return self._reader.list_files(extensions)

    def download(self, name: str) -> BytesIO:
        return self._reader.download(name)

    def upload(self, name: str, buffer: BytesIO) -> None:
        self._writer.upload(name, buffer)

    def check_exists(self, name: str) -> bool:
        return self._writer.check_exists(name)


# ── Фабрика ────────────────────────────────────────────────────────


def create_source(
    source_type: str,
    dest_type: str,
    local_source_dir: str = "",
    local_dest_dir: str = "",
    yandex_source_dir: str = "",
    yandex_dest_dir: str = "",
    yandex_token: str | None = None,
) -> DataSource:
    """Создать нужный адаптер по параметрам.

    Если source_type == dest_type — возвращается единый адаптер.
    Иначе — CompositeSource с раздельными reader/writer.
    """
    if source_type == dest_type:
        if source_type == "yandex":
            return YandexSource(
                source_dir=yandex_source_dir,
                dest_dir=yandex_dest_dir,
                token=yandex_token,
            )
        return LocalSource(
            source_dir=local_source_dir,
            dest_dir=local_dest_dir,
        )

    # Разные хранилища — CompositeSource
    if source_type == "yandex":
        reader = YandexSource(
            source_dir=yandex_source_dir,
            dest_dir=yandex_source_dir,
            token=yandex_token,
        )
    else:
        reader = LocalSource(
            source_dir=local_source_dir,
            dest_dir=local_source_dir,
        )

    if dest_type == "yandex":
        writer = YandexSource(
            source_dir=yandex_dest_dir,
            dest_dir=yandex_dest_dir,
            token=yandex_token,
        )
    else:
        writer = LocalSource(
            source_dir=local_dest_dir,
            dest_dir=local_dest_dir,
        )

    logger.info("CompositeSource: reader=%s, writer=%s", source_type, dest_type)
    return CompositeSource(reader, writer)