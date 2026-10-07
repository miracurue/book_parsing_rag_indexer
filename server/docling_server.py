"""Удалённый сервер Docling для парсинга PDF с распознаванием формул.

Запускается на машине с GPU (Colab / удалённый сервер).
Принимает PDF через HTTP, возвращает Markdown с LaTeX-формулами.

Архитектура: polling pattern (для работы через cloudflare tunnel).
    POST /parse       — начать парсинг → job_id
    GET  /status/{id} — проверить статус (pending/processing/done/error)
    GET  /result/{id} — получить результат (markdown, формулы, метрики)
    POST /preload     — предзагрузка модели
    GET  /health      — проверка доступности
    GET  /models_info — информация о моделях

Использование (локально):
    uvicorn docling_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("docling_server")

app = FastAPI(title="Docling Parser API", version="2.0")

# ════════════════════════════════════════════════════════════════════
# Глобальное состояние
# ════════════════════════════════════════════════════════════════════

_converter = None
_gpu_info: Optional[str] = None

# Хранилище задач: job_id → {status, result, error, created_at, ...}
_jobs: dict[str, dict] = {}


def _get_converter():
    """Ленивая инициализация конвертера Docling с поддержкой формул."""
    global _converter, _gpu_info
    if _converter is not None:
        return _converter

    logger.info("Загрузка модели Docling (formula enrichment)...")
    start = time.time()

    try:
        import torch
        if torch.cuda.is_available():
            _gpu_info = f"CUDA: {torch.cuda.get_device_name(0)}"
            logger.info(f"GPU обнаружен: {_gpu_info}")
        else:
            _gpu_info = "CPU only"
            logger.warning("GPU не обнаружен")
    except ImportError:
        _gpu_info = "CPU (torch не установлен)"

    from docling.document_converter import DocumentConverter
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
    from docling.document_converter import FormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

    pipeline_opts = ThreadedPdfPipelineOptions(do_formula_enrichment=True)

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: FormatOption(
                pipeline_options=pipeline_opts,
                backend=DoclingParseDocumentBackend,
                pipeline_cls=StandardPdfPipeline,
            )
        }
    )

    elapsed = round(time.time() - start, 1)
    logger.info(f"Модель загружена за {elapsed}с. Устройство: {_gpu_info}")
    return _converter


# ════════════════════════════════════════════════════════════════════
# Фоновый парсинг
# ════════════════════════════════════════════════════════════════════

def _do_parse(job_id: str, pdf_bytes: bytes, filename: str,
              extract_images: bool, image_scale: float):
    """Фоновая задача парсинга PDF. Выполняется в отдельном потоке."""
    try:
        _jobs[job_id]["status"] = "processing"
        start_time = time.time()

        converter = _get_converter()

        # Сохраняем во временный файл (Docling требует путь)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            result = converter.convert(tmp_path)
            md_content = result.document.export_to_markdown()
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        # Подсчёт заглушек
        placeholder_count = md_content.count("<!-- image -->")
        formula_not_decoded = md_content.count("<!-- formula-not-decoded -->")

        # Заменяем заглушки на [IMAGE_N]
        img_idx = 0
        new_lines = []
        for line in md_content.split("\n"):
            if "<!-- image -->" in line:
                img_idx += 1
                line = line.replace("<!-- image -->", f"[IMAGE_{img_idx}]")
            new_lines.append(line)
        md_content = "\n".join(new_lines)

        # Подсчёт формул
        display = len(re.findall(r"\$\$(.+?)\$\$", md_content, re.DOTALL))
        inline = len(re.findall(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", md_content))
        formula_count = display + inline

        # Извлечение изображений
        images_data = []
        if extract_images:
            images_data = _extract_images(result, image_scale)

        elapsed = round(time.time() - start_time, 1)
        logger.info(
            f"Парсинг завершён [{job_id[:8]}]: {len(md_content):,} символов, "
            f"{formula_count} формул, {elapsed}с"
        )

        _jobs[job_id]["result"] = {
            "markdown": md_content,
            "image_placeholders": placeholder_count,
            "formula_count": formula_count,
            "formula_not_decoded": formula_not_decoded,
            "images": images_data,
            "parse_time_sec": elapsed,
            "error": None,
        }
        _jobs[job_id]["status"] = "done"

    except Exception as e:
        logger.exception(f"Ошибка парсинга [{job_id[:8]}]: {e}")
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(e)


# ════════════════════════════════════════════════════════════════════
# Endpoints
# ════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    """Проверка доступности сервера."""
    return {
        "status": "ok",
        "gpu": _gpu_info or "не инициализировано",
        "converter_loaded": _converter is not None,
        "active_jobs": sum(
            1 for j in _jobs.values()
            if j["status"] in ("pending", "processing")
        ),
    }


@app.post("/parse")
async def parse_pdf(
    file: UploadFile = File(..., description="PDF файл для парсинга"),
    extract_images: bool = Form(False),
    image_scale: float = Form(2.0),
):
    """Начать парсинг PDF в фоне. Возвращает job_id для polling.

    Polling pattern: POST /parse → GET /status/{job_id} → GET /result/{job_id}
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Файл должен быть PDF")

    pdf_bytes = await file.read()
    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Пустой файл")

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {
        "status": "pending",
        "filename": file.filename,
        "created_at": time.time(),
        "result": None,
        "error": None,
    }

    logger.info(
        f"Новая задача [{job_id[:8]}]: {file.filename} ({len(pdf_bytes):,} байт)"
    )

    # Запускаем парсинг в фоне
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        _do_parse,
        job_id, pdf_bytes, file.filename, extract_images, image_scale,
    )

    return {"job_id": job_id, "status": "pending"}


@app.get("/status/{job_id}")
async def job_status(job_id: str):
    """Проверить статус задачи парсинга."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    job = _jobs[job_id]
    resp = {
        "job_id": job_id,
        "status": job["status"],
    }
    if job["status"] == "error":
        resp["error"] = job["error"]
    if job["status"] == "done":
        resp["parse_time_sec"] = job["result"]["parse_time_sec"]

    return resp


@app.get("/result/{job_id}")
async def job_result(job_id: str):
    """Получить результат завершённой задачи."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    job = _jobs[job_id]
    if job["status"] != "done":
        raise HTTPException(
            status_code=400,
            detail=f"Задача ещё не завершена (статус: {job['status']})",
        )

    return job["result"]


@app.post("/preload")
async def preload_model():
    """Предзагрузка модели Docling (ленивая инициализация).

    Вызывать ПЕРЕД первым /parse.
    """
    try:
        converter = await asyncio.to_thread(_get_converter)
        return {
            "status": "ok",
            "gpu": _gpu_info,
            "converter_loaded": converter is not None,
            "message": "Модель загружена и готова к работе",
        }
    except Exception as e:
        logger.exception(f"Ошибка предзагрузки модели: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "converter_loaded": False},
        )


@app.get("/models_info")
async def models_info():
    """Информация о загруженных моделях."""
    return {
        "converter_loaded": _converter is not None,
        "gpu": _gpu_info or "не инициализировано",
        "total_jobs": len(_jobs),
    }


# ════════════════════════════════════════════════════════════════════
# Извлечение изображений
# ════════════════════════════════════════════════════════════════════

def _extract_images(conv_result, scale: float) -> list[dict]:
    """Извлечь изображения из результата Docling и вернуть как base64."""
    import base64
    from PIL import Image

    images = []
    pictures = getattr(conv_result, "pictures", []) or []

    for pic_idx, pic in enumerate(pictures, start=1):
        try:
            img = None
            if hasattr(pic, "get_image"):
                try:
                    img = pic.get_image(conv_result, scale=scale)
                except Exception:
                    pass
            if img is None and hasattr(pic, "image") and pic.image is not None:
                img = pic.image

            if img is not None:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                images.append({
                    "filename": f"image_{pic_idx:03d}.png",
                    "data": b64,
                })
        except Exception as e:
            logger.warning(f"Ошибка при извлечении изображения {pic_idx}: {e}")

    return images


# ════════════════════════════════════════════════════════════════════
# Запуск
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Docling Parser API Server (v2.0, polling)")
    print("  http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)