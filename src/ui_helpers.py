"""Переиспользуемые UI-компоненты для Streamlit страниц.

Универсальный компонент — используется всеми страницами.
Содержит только UI-элементы, без бизнес-логики.

Компоненты:
- render_source_selector() — выбор источника (Local/Yandex)
- render_path_inputs() — поля ввода путей с кэшированием в session_state
- render_run_button() — кнопка запуска с spinner и обратной связью
"""

from __future__ import annotations

from typing import Callable, Any

import streamlit as st


def render_source_selector(
    key_prefix: str = "default",
    label: str = "Источник данных",
) -> tuple[str, str]:
    """Радио-кнопки выбора источника и назначения.

    Returns:
        Кортеж (source_type, dest_type): 'local' или 'yandex'
    """
    source_type = st.radio(
        label,
        options=["local", "yandex"],
        format_func=lambda x: "📁 Локальная папка" if x == "local" else "☁️ Яндекс.Диск",
        horizontal=True,
        key=f"{key_prefix}_source_type",
    )
    dest_type = st.radio(
        "Куда сохранять",
        options=["local", "yandex"],
        format_func=lambda x: "📁 Локальная папка" if x == "local" else "☁️ Яндекс.Диск",
        horizontal=True,
        key=f"{key_prefix}_dest_type",
    )
    return source_type, dest_type


def render_path_inputs(
    source_type: str,
    dest_type: str,
    key_prefix: str = "default",
    source_label: str = "Папка с исходными файлами",
    dest_label: str = "Папка для результатов",
    source_placeholder: str = "",
    dest_placeholder: str = "",
) -> tuple[str, str]:
    """Поля ввода путей для источника и назначения.

    Значения сохраняются в session_state для сохранения между перерисовками.

    Returns:
        Кортеж (source_dir, dest_dir)
    """
    if source_type == "local":
        source_dir = st.text_input(
            source_label,
            value=st.session_state.get(f"{key_prefix}_local_source_dir", ""),
            key=f"{key_prefix}_local_source_dir",
            placeholder=source_placeholder or "Напр.: data/raw",
        )
    else:
        source_dir = st.text_input(
            source_label + " (Яндекс.Диск)",
            value=st.session_state.get(f"{key_prefix}_yandex_source_dir", ""),
            key=f"{key_prefix}_yandex_source_dir",
            placeholder=source_placeholder or "Напр.: Books/Scan/pages",
        )

    if dest_type == "local":
        dest_dir = st.text_input(
            dest_label,
            value=st.session_state.get(f"{key_prefix}_local_dest_dir", ""),
            key=f"{key_prefix}_local_dest_dir",
            placeholder=dest_placeholder or "Напр.: data/processed",
        )
    else:
        dest_dir = st.text_input(
            dest_label + " (Яндекс.Диск)",
            value=st.session_state.get(f"{key_prefix}_yandex_dest_dir", ""),
            key=f"{key_prefix}_yandex_dest_dir",
            placeholder=dest_placeholder or "Напр.: Books/Scan/split",
        )

    return source_dir, dest_dir


def render_run_button(
    label: str,
    run_fn: Callable[..., dict],
    args: dict[str, Any] | None = None,
    key_prefix: str = "default",
) -> dict | None:
    """Кнопка запуска обработки с spinner и обратной связью.

    Args:
        label: Текст кнопки
        run_fn: Функция-оркестратор, возвращает dict с результатами
        args: Аргументы для run_fn
        key_prefix: Префикс для ключей session_state

    Returns:
        dict с результатами или None (если кнопка не нажата)
    """
    if st.button(label, key=f"{key_prefix}_run_btn", use_container_width=True):
        if args is None:
            args = {}
        with st.spinner("⏳ Обработка..."):
            try:
                result = run_fn(**args)
                _dest = result.get("dest_dir", "")
                _dest_msg = f"\n📁 Результаты: `{_dest}`" if _dest else ""
                _errors = result.get("errors", 0)
                if _errors > 0:
                    st.warning(
                        f"⚠️ Завершено с ошибками. Успешно: {result.get('success', '?')}/{result.get('total', '?')}, "
                        f"Ошибок: {_errors}. "
                        f"Время: {result.get('elapsed_sec', '?')}с{_dest_msg}"
                    )
                else:
                    st.success(
                        f"✅ Готово! Обработано: {result.get('success', '?')} файлов, "
                        f"{result.get('total_pages', '')} страниц. "
                        f"Время: {result.get('elapsed_sec', '?')}с{_dest_msg}"
                    )
                return result
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
                return {"error": str(e)}
    return None