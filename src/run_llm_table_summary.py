"""Оркестратор LLM-сводок для oversized таблиц.

Обходит таблицы из extracted/extracted_tables/, превышающие chunk_size,
и отправляет их в LLM для генерации текстового описания.
Результат сохраняется в extracted/table_summaries/ отдельными файлами.

Threading + progress_callback + cancel_event — стандартный паттерн проекта.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from src.llm_clients import call_text_api, FatalAPIError
from src.check_chunk_sizes import find_book_extracted_table_dirs

logger = logging.getLogger(__name__)


# ── Промпты по умолчанию ───────────────────────────────────────────────
DEFAULT_SYSTEM_PROMPT = (
    "Ты — технический редактор. Опиши таблицу двумя абзацами.\n"
    "Правила:\n"
    "1. Описание должно начинаться с заголовка/названия таблицы "
    "(как указано перед таблицей, например «Таблица 10. Химический состав...»)\n"
    "2. Первый абзац — структура: название таблицы, столбцы, группы строк\n"
    "3. Второй абзац — ключевые значения, диапазоны, единицы измерения\n"
    "4. Сохрани ВСЕ технические термины\n"
    "5. НЕ добавляй вводных слов (\"Вот описание\", \"В таблице представлено\" и т.д.)\n"
    "6. Только описание — сразу по сути\n"
    "7. Строки с # — это breadcrumbs (путь по главам) для понимания контекста, "
    "описывать их НЕ нужно"
)

DEFAULT_USER_PROMPT_TEMPLATE = (
    "Ниже представлен файл с таблицей. "
    "Заголовки (#) — breadcrumbs для контекста, перед таблицей — её название.\n\n"
    "{table}"
)


def find_oversized_tables(
    base_dir: Path,
    chunk_size: int = 1200,
    book_names: Optional[list[str]] = None,
) -> list[dict]:
    """Найти таблицы, превышающие chunk_size и не имеющие LLM-сводки.

    Проверяет наличие файла в extracted/table_summaries/ с тем же именем.

    Returns:
        Список словарей: {"book": str, "file": Path, "chars": int}
    """
    table_dirs = find_book_extracted_table_dirs(base_dir)

    if book_names:
        table_dirs = [(bn, d) for bn, d in table_dirs if bn in book_names]

    results: list[dict] = []

    for book_name, tables_dir in table_dirs:
        summaries_dir = tables_dir.parent / "table_summaries"

        for md_file in sorted(tables_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue

            chars = len(text)

            # Превышает порог и ещё нет LLM-сводки в table_summaries/
            summary_file = summaries_dir / md_file.name
            if chars > chunk_size and not summary_file.exists():
                results.append({
                    "book": book_name,
                    "file": md_file,
                    "chars": chars,
                })

    return results


def run_llm_table_summary(
    base_dir: str | Path,
    chunk_size: int = 1200,
    model_name: str = "glm-4.6v",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    temperature: float = 0.1,
    book_names: Optional[list[str]] = None,
    recreate: bool = False,
    cancel_event: Optional[threading.Event] = None,
    progress_callback=None,
) -> dict:
    """Запустить генерацию LLM-сводок для oversized таблиц.

    Результаты сохраняются в extracted/table_summaries/ отдельными файлами.
    Для oversized таблиц LLM-сводка перезапишет структурную сводку (если есть).

    Args:
        base_dir: Папка с книгами (data/books).
        chunk_size: Порог размера (символы).
        model_name: Имя модели из реестра.
        system_prompt: Системный промпт для LLM.
        user_prompt_template: Шаблон пользовательского промпта с {table}.
        temperature: Температура генерации.
        book_names: Список книг. None = все.
        recreate: Если True — пересоздать LLM-сводки.
        cancel_event: Событие отмены.
        progress_callback: Функция (current, total, current_file, stats).

    Returns:
        {
            "processed": int,
            "skipped": int,
            "errors": int,
            "total_tokens": int,
            "cancelled": bool,
            "elapsed_sec": float,
        }
    """
    import threading  # noqa — нужен для type hint cancel_event

    start_time = time.time()
    base_path = Path(base_dir)

    # При recreate — удалить только LLM-сводки oversized таблиц
    if recreate:
        _recreate_llm_summaries(base_path, chunk_size, book_names)

    # Найти oversized таблицы без LLM-сводки
    oversized = find_oversized_tables(base_path, chunk_size, book_names)
    total = len(oversized)

    if total == 0:
        _notify(progress_callback, 0, 0, "нет oversized таблиц", {
            "processed": 0, "skipped": 0, "errors": 0, "total_tokens": 0,
        })
        return _result(0, 0, 0, 0, False, start_time)

    logger.info("Найдено %d oversized таблиц без LLM-сводки", total)

    processed = 0
    errors = 0
    total_tokens = 0
    created_dirs: set[str] = set()

    for i, item in enumerate(oversized):
        # Проверка отмены
        if cancel_event and cancel_event.is_set():
            _notify(progress_callback, i, total, "отменено", {
                "processed": processed, "skipped": 0,
                "errors": errors, "total_tokens": total_tokens,
            })
            return _result(processed, 0, errors, total_tokens, True, start_time)

        md_file = item["file"]
        book_name = item["book"]
        file_label = f"{book_name}/{md_file.name}"

        _notify(progress_callback, i, total, file_label, {
            "processed": processed, "skipped": 0,
            "errors": errors, "total_tokens": total_tokens,
        })

        try:
            # Прочитать файл целиком (заголовки + caption + таблица)
            text = md_file.read_text(encoding="utf-8")

            if not text.strip():
                logger.warning("Пустой файл %s — пропуск", md_file)
                errors += 1
                continue

            # Сформировать промпт с полным текстом файла
            user_prompt = user_prompt_template.replace("{table}", text)

            # Вызвать LLM (без max_tokens — reasoning-моделям нужен запас на размышления)
            result = call_text_api(
                model_name=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
            )

            summary = result["content"]
            total_tokens += result.get("total_tokens", 0)

            # Проверка: LLM вернул пустой ответ — не писать файл
            if not summary.strip():
                logger.warning(
                    "LLM вернул пустой ответ для %s — пропуск", file_label
                )
                errors += 1
                continue

            # Сохранить в extracted/table_summaries/
            summaries_dir = base_path / book_name / "extracted" / "table_summaries"
            if book_name not in created_dirs:
                summaries_dir.mkdir(parents=True, exist_ok=True)
                created_dirs.add(book_name)

            summary_file = summaries_dir / md_file.name
            summary_file.write_text(summary, encoding="utf-8")
            processed += 1

            logger.info(
                "LLM-сводка для %s: %d символов, %d токенов → %s",
                file_label, len(summary), result.get("total_tokens", 0),
                summary_file,
            )

        except FatalAPIError as e:
            logger.error("Фатальная ошибка API: %s", e)
            errors += 1
            # Остановить пайплайн при фатальной ошибке
            _notify(progress_callback, i + 1, total, f"фатальная ошибка: {e}", {
                "processed": processed, "skipped": 0,
                "errors": errors, "total_tokens": total_tokens,
            })
            return _result(processed, 0, errors, total_tokens, False, start_time)

        except Exception as e:
            logger.error("Ошибка обработки %s: %s", md_file, e)
            errors += 1

    # Финальное уведомление
    _notify(progress_callback, total, total, "готово", {
        "processed": processed, "skipped": 0,
        "errors": errors, "total_tokens": total_tokens,
    })

    return _result(processed, 0, errors, total_tokens, False, start_time)


# ── Хелперы ────────────────────────────────────────────────────────────

def _notify(progress_callback, current, total, current_file, stats):
    if progress_callback:
        progress_callback(current, total, current_file, stats)


def _recreate_llm_summaries(
    base_path: Path,
    chunk_size: int,
    book_names: Optional[list[str]] = None,
) -> None:
    """При recreate=False — ничего не делает. При True — удаляет LLM-сводки
    oversized таблиц из table_summaries/, чтобы пересоздать."""
    table_dirs = find_book_extracted_table_dirs(base_path)
    if book_names:
        table_dirs = [(bn, d) for bn, d in table_dirs if bn in book_names]

    for book_name, tables_dir in table_dirs:
        summaries_dir = tables_dir.parent / "table_summaries"
        if not summaries_dir.is_dir():
            continue

        for md_file in sorted(tables_dir.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
            except OSError:
                continue
            if len(text) > chunk_size:
                summary_file = summaries_dir / md_file.name
                if summary_file.exists():
                    summary_file.unlink()
                    logger.info("Удалена LLM-сводка: %s", summary_file)


def _result(processed, skipped, errors, total_tokens, cancelled, start_time):
    return {
        "processed": processed,
        "skipped": skipped,
        "errors": errors,
        "total_tokens": total_tokens,
        "cancelled": cancelled,
        "elapsed_sec": round(time.time() - start_time, 1),
    }