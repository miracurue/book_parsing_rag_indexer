"""Разделение PDF файла на отдельные страницы.

Чистые функции, не зависящие от источника данных.
Работают с BytesIO — можно вызывать из любого адаптера.
"""

import logging
from io import BytesIO

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def split_pdf_to_pages(
    pdf_buffer: BytesIO,
    start_page: int = 0,
) -> list[tuple[str, BytesIO]]:
    """Разделить PDF на отдельные страницы.

    Args:
        pdf_buffer:  Исходный PDF в памяти.
        start_page:  Номер первой страницы (для именования файлов).
                     По умолчанию 0.

    Returns:
        Список кортежей (filename, BytesIO), где filename вида "003.pdf".

    Raises:
        ValueError: если PDF пустой или не содержит страниц.
    """
    if start_page < 0:
        raise ValueError(f"start_page должен быть >= 0, получено: {start_page}")

    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.read()

    if not pdf_bytes:
        raise ValueError("Пустой PDF-буфер")

    logger.debug(
        "Разделение PDF на страницы, start_page=%d, размер=%d байт",
        start_page, len(pdf_bytes),
    )

    reader = PdfReader(BytesIO(pdf_bytes))
    num_pages = len(reader.pages)

    if num_pages == 0:
        raise ValueError("PDF не содержит страниц")

    logger.debug("PDF содержит %d страниц", num_pages)

    result: list[tuple[str, BytesIO]] = []

    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)

        out_buf = BytesIO()
        writer.write(out_buf)
        out_buf.seek(0)

        page_num = start_page + i
        filename = f"{page_num:03d}.pdf"
        result.append((filename, out_buf))

    logger.debug(
        "Разделение завершено: %d страниц, имена %s ... %s",
        num_pages,
        result[0][0],
        result[-1][0],
    )

    return result