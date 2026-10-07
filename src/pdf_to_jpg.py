"""Конвертация PDF в JPEG.

Чистые функции, не зависящие от источника данных.
Работают с BytesIO — можно вызывать из любого адаптера.
"""

import logging
from io import BytesIO

from pdf2image import convert_from_bytes

logger = logging.getLogger(__name__)


def convert_pdf_page_to_jpg(
    pdf_buffer: BytesIO,
    dpi: int = 200,
    poppler_path: str | None = None,
) -> tuple[BytesIO, int, int]:
    """Конвертировать одну страницу PDF в JPEG.

    Берёт первую страницу PDF. Если страниц несколько —
    остальные игнорируются (типичный кейс: порезанные PDF).

    Args:
        pdf_buffer:    Исходный PDF в памяти.
        dpi:           Разрешение рендеринга (72–600).
        poppler_path:  Путь к директории с бинарниками Poppler (для Windows).

    Returns:
        Кортеж (BytesIO, width_px, height_px) — JPEG-буфер и размеры в пикселях.

    Raises:
        ValueError: если PDF пустой или не содержит страниц.
    """
    if not 72 <= dpi <= 600:
        raise ValueError(f"dpi должен быть 72–600, получено: {dpi}")

    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.read()

    if not pdf_bytes:
        raise ValueError("Пустой PDF-буфер")

    logger.debug(
        "Конвертация PDF → JPG, dpi=%d, размер PDF=%d байт",
        dpi, len(pdf_bytes),
    )

    kwargs: dict = {"dpi": dpi, "fmt": "jpeg", "use_cropbox": True}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    pages = convert_from_bytes(pdf_bytes, **kwargs)

    if not pages:
        raise ValueError("PDF не содержит страниц")

    # Берём первую (и обычно единственную) страницу
    page = pages[0]

    out = BytesIO()
    page.save(out, format="JPEG")
    out.seek(0)

    width, height = page.size

    logger.debug(
        "Конвертация завершена: JPEG %dx%d, %d байт",
        width, height, out.getbuffer().nbytes,
    )

    # Освобождаем память от оставшихся страниц
    del pages

    return out, width, height


def get_pdf_page_info(
    pdf_buffer: BytesIO,
    dpi: int = 200,
    poppler_path: str | None = None,
) -> dict:
    """Получить информацию о странице PDF (пиксели и размер файла).

    Рендерит первую страницу PDF и возвращает размеры
    в пикселях, размер исходного PDF и размер полученного JPG.

    Args:
        pdf_buffer:    Исходный PDF в памяти (одностраничный).
        dpi:           DPI для рендеринга.
        poppler_path:  Путь к бинарникам Poppler.

    Returns:
        dict с ключами: width_px, height_px, pdf_size_bytes, jpg_size_bytes, dpi
    """
    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.read()
    pdf_size = len(pdf_bytes)

    if not pdf_bytes:
        raise ValueError("Пустой PDF-буфер")

    kwargs: dict = {"dpi": dpi, "fmt": "jpeg", "use_cropbox": True}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path

    pages = convert_from_bytes(pdf_bytes, **kwargs)

    if not pages:
        raise ValueError("PDF не содержит страниц")

    page = pages[0]
    width_px, height_px = page.size

    # Рендерим в JPG, чтобы узнать размер итогового файла
    out = BytesIO()
    page.save(out, format="JPEG")
    jpg_size = out.getbuffer().nbytes
    out.close()

    del pages

    return {
        "width_px": width_px,
        "height_px": height_px,
        "pdf_size_bytes": pdf_size,
        "jpg_size_bytes": jpg_size,
        "dpi": dpi,
    }
