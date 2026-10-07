"""Оркестратор очистки артефактов парсинга.

Обходит структуру: base_dir/название_книги/parsed/*.md
Для каждого файла удаляет выбранные артефакты (in-place).

Поддерживает:
- Выбор конкретных книг (book_names)
- Выбор типов артефактов (enabled)
- Отмену через threading.Event
- Обратный вызов прогресса
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from src.config import setup_logging
from src.clean_table_headers import process_file, count_artifacts, ALL_ARTIFACT_KEYS

logger = logging.getLogger(__name__)

MD_EXTS = (".md",)


def _find_md_files(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Найти все .md в подпапках */parsed/.

    Args:
        base_dir: Корневая папка с книгами.
        book_names: Список книг для обработки. None = все книги.

    Returns:
        Список кортежей: (book_name, md_path)
    """
    results: list[tuple[str, Path]] = []

    if not base_dir.exists():
        return results

    for book_dir in sorted(base_dir.iterdir()):
        if not book_dir.is_dir():
            continue

        # Фильтр по выбранным книгам
        if book_names and book_dir.name not in book_names:
            continue

        parsed_dir = book_dir / "parsed"
        if not parsed_dir.is_dir():
            logger.debug("Пропуск %s: нет parsed/", book_dir.name)
            continue

        for md_file in sorted(parsed_dir.glob("*")):
            if md_file.is_file() and md_file.suffix.lower() in MD_EXTS:
                results.append((book_dir.name, md_file))

    return results


def count_files_with_artifacts(
    base_dir: Path,
    book_names: list[str] | None = None,
    enabled: dict[str, bool] | None = None,
) -> dict:
    """Подсчитать файлы с артефактами по типам.

    Быстрая проверка — без модификации файлов.

    Args:
        base_dir: Корневая папка с книгами.
        book_names: Список книг. None = все.
        enabled: Типы артефактов. None = все.

    Returns:
        dict: {
            "files_with_artifacts": int,
            "total_files": int,
            "artifact_counts": {тип: общее_количество},
            "total_artifacts": int,
        }
    """
    files_with_artifacts = 0
    total_files = 0
    artifact_counts: dict[str, int] = {}

    for _, md_path in _find_md_files(base_dir, book_names):
        try:
            content = md_path.read_text(encoding="utf-8")
            counts = count_artifacts(content, enabled)
            total_files += 1

            if counts["total"] > 0:
                files_with_artifacts += 1
                for key, val in counts.items():
                    if key == "total":
                        continue
                    artifact_counts[key] = artifact_counts.get(key, 0) + val
        except Exception:
            pass

    return {
        "files_with_artifacts": files_with_artifacts,
        "total_files": total_files,
        "artifact_counts": artifact_counts,
        "total_artifacts": sum(artifact_counts.values()),
    }


# ══════════════════════════════════════════════════════════════════════
# Обратная совместимость
# ══════════════════════════════════════════════════════════════════════

def count_files_with_table_headers(
    base_dir: Path,
    book_names: list[str] | None = None,
) -> int:
    """Посчитать файлы с любыми артефактами.

    .. deprecated:: Используйте count_files_with_artifacts().
    """
    result = count_files_with_artifacts(base_dir, book_names, enabled=None)
    return result["files_with_artifacts"]


# ══════════════════════════════════════════════════════════════════════
# Публичный интерфейс
# ══════════════════════════════════════════════════════════════════════

def run_clean_table_headers(
    base_dir: str = "data/books",
    book_names: list[str] | None = None,
    enabled: dict[str, bool] | None = None,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable | None = None,
    log_level: str = "INFO",
) -> dict:
    """Запустить очистку артефактов парсинга.

    Обходит: base_dir/*/parsed/*.md (или только выбранные книги).
    Модифицирует файлы in-place.

    Args:
        base_dir:          Корневая папка с книгами.
        book_names:        Список книг для обработки. None = все.
        enabled:           Словарь {тип: True/False}. None = все типы.
        cancel_event:      Событие отмены.
        progress_callback: Функция(current, total, current_file, stats).
        log_level:         Уровень логирования.

    Returns:
        dict: total, modified, unchanged, errors, total_replacements,
              elapsed_sec, cancelled.
    """
    setup_logging(log_level)

    logger.info("=== Запуск очистки артефактов парсинга ===")
    logger.info("Базовая папка: %s", base_dir)
    if book_names:
        logger.info("Выбранные книги: %s", ", ".join(book_names))
    if enabled:
        active = [k for k, v in enabled.items() if v]
        logger.info("Активные артефакты: %s", ", ".join(active))

    base = Path(base_dir)
    files = _find_md_files(base, book_names)
    total = len(files)

    if total == 0:
        logger.info("Нет .md файлов для обработки в %s", base_dir)
        return {
            "total": 0, "modified": 0, "unchanged": 0, "errors": 0,
            "total_replacements": 0, "elapsed_sec": 0.0, "cancelled": False,
        }

    logger.info("К обработке: %d .md файлов", total)

    t0 = time.perf_counter()
    modified = 0
    unchanged = 0
    errors = 0
    total_replacements = 0
    cancelled = False

    for i, (book_name, md_path) in enumerate(files):
        # Проверка отмены
        if cancel_event and cancel_event.is_set():
            logger.info("Очистка отменена на файле %s", md_path.name)
            cancelled = True
            break

        relative_name = f"{book_name}/parsed/{md_path.name}"

        logger.info("[%d/%d] %s", i + 1, total, relative_name)

        result = process_file(md_path, enabled=enabled)

        if result["error"]:
            errors += 1
            logger.error("❌ Ошибка %s: %s", relative_name, result["error"])
        elif result["modified"]:
            modified += 1
            total_replacements += result["replacements"]
            logger.info(
                "✅ %s: %d замен", relative_name, result["replacements"],
            )
        else:
            unchanged += 1

        # Обратный вызов прогресса
        if progress_callback:
            progress_callback(i + 1, total, relative_name, {
                "modified": modified,
                "unchanged": unchanged,
                "errors": errors,
                "total_replacements": total_replacements,
            })

    elapsed = round(time.perf_counter() - t0, 2)
    logger.info(
        "=== Очистка завершена: %d изменено, %d без изменений, "
        "%d ошибок, %d замен%s, %.2fс ===",
        modified, unchanged, errors, total_replacements,
        ", ОТМЕНЕНО" if cancelled else "",
        elapsed,
    )

    return {
        "total": total,
        "modified": modified,
        "unchanged": unchanged,
        "errors": errors,
        "total_replacements": total_replacements,
        "elapsed_sec": elapsed,
        "cancelled": cancelled,
    }