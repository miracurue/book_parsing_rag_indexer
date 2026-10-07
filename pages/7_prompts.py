"""Страница управления промптами.

Создание, просмотр, редактирование и удаление промптов.
Промпты сохраняются в data/prompts.json и доступны для выбора
на других страницах через src.prompts.list_prompt_names().
"""

import streamlit as st

from src.prompts import (
    load_prompts,
    add_prompt,
    update_prompt,
    delete_prompt,
    get_prompt_by_id,
)

st.set_page_config(page_title="Промпты", page_icon="📝", layout="wide")
st.title("📝 Управление промптами")
st.markdown(
    "Создавайте и управляйте промптами для использования в скриптах RAG-пайплайна. "
    "На других страницах промпты можно выбрать по названию из списка."
)

# ── Вкладки ───────────────────────────────────────────────────────────
tab_all, tab_new, tab_edit = st.tabs([
    "📋 Все промпты",
    "➕ Новый промпт",
    "✏️ Редактировать",
])


# ══════════════════════════════════════════════════════════════════════
# Вкладка: Все промпты
# ══════════════════════════════════════════════════════════════════════
with tab_all:
    prompts = load_prompts()

    if not prompts:
        st.info(
            "📭 Промптов пока нет. Перейдите на вкладку **«➕ Новый промпт»**, "
            "чтобы создать первый."
        )
    else:
        st.caption(f"Всего промптов: **{len(prompts)}**")

        for p in prompts:
            with st.container(border=True):
                col_header, col_btn_edit, col_btn_del = st.columns([5, 1, 1])

                with col_header:
                    st.subheader(p["name"])
                    if p.get("description"):
                        st.caption(p["description"])

                with col_btn_edit:
                    if st.button(
                        "✏️",
                        key=f"edit_btn_{p['id']}",
                        help="Редактировать",
                    ):
                        st.session_state["edit_prompt_id"] = p["id"]
                        st.info("📌 Перейдите на вкладку **«✏️ Редактировать»** — промпт предвыбран.")

                with col_btn_del:
                    if st.button(
                        "🗑️",
                        key=f"del_btn_{p['id']}",
                        help="Удалить",
                    ):
                        st.session_state["delete_prompt_id"] = p["id"]

                # Текст промпта (свёрнутый)
                with st.expander("📄 Текст промпта"):
                    st.text(p["text"])

                # Метаинформация
                st.caption(
                    f"Создан: {p.get('created_at', '?')}  |  "
                    f"Обновлён: {p.get('updated_at', '?')}"
                )

    # ── Подтверждение удаления ──────────────────────────────────────
    if "delete_prompt_id" in st.session_state:
        _del_id = st.session_state["delete_prompt_id"]
        _del_prompt = get_prompt_by_id(_del_id)
        if _del_prompt:
            st.warning(f"⚠️ Удалить промпт **«{_del_prompt['name']}»**?")
            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("✅ Да, удалить", key="confirm_delete_btn"):
                    try:
                        delete_prompt(_del_id)
                        st.success(f"🗑️ Промпт «{_del_prompt['name']}» удалён.")
                    except ValueError as e:
                        st.error(f"❌ {e}")
                    del st.session_state["delete_prompt_id"]
                    st.rerun()
            with col_cancel:
                if st.button("❌ Отмена", key="cancel_delete_btn"):
                    del st.session_state["delete_prompt_id"]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════
# Вкладка: Новый промпт
# ══════════════════════════════════════════════════════════════════════
with tab_new:
    st.subheader("➕ Создание нового промпта")

    with st.form("new_prompt_form", clear_on_submit=True):
        new_name = st.text_input(
            "Название *",
            key="prompt_new_name",
            placeholder="Напр.: RAG-поиск по документам",
            help="Уникальное название для выбора промпта на других страницах.",
        )
        new_description = st.text_input(
            "Описание",
            key="prompt_new_description",
            placeholder="Краткое описание: для чего этот промпт",
        )
        new_text = st.text_area(
            "Текст промпта *",
            key="prompt_new_text",
            placeholder="Введите текст промпта...",
            height=200,
        )

        st.caption("* — обязательные поля")

        submitted = st.form_submit_button("💾 Сохранить промпт", type="primary")
        if submitted:
            if not new_name.strip() or not new_text.strip():
                st.error("❌ Название и текст промпта обязательны.")
            else:
                try:
                    created = add_prompt(new_name, new_description, new_text)
                    st.success(f"✅ Промпт «{created['name']}» сохранён!")
                except ValueError as e:
                    st.error(f"❌ {e}")


# ══════════════════════════════════════════════════════════════════════
# Вкладка: Редактирование
# ══════════════════════════════════════════════════════════════════════
with tab_edit:
    st.subheader("✏️ Редактирование промпта")

    # Выбор промпта для редактирования
    prompts_list = load_prompts()

    if not prompts_list:
        st.info("📭 Нет промптов для редактирования.")
    else:
        # Если пришли по кнопке «✏️» из списка — предвыбрать
        _preselect_idx = 0
        if "edit_prompt_id" in st.session_state:
            _edit_id = st.session_state["edit_prompt_id"]
            for i, p in enumerate(prompts_list):
                if p["id"] == _edit_id:
                    _preselect_idx = i
                    break

        prompt_options = [p["name"] for p in prompts_list]
        selected_name = st.selectbox(
            "Выберите промпт",
            options=prompt_options,
            index=_preselect_idx,
            key="edit_prompt_select",
        )

        # Получить данные выбранного промпта
        selected_prompt = None
        for p in prompts_list:
            if p["name"] == selected_name:
                selected_prompt = p
                break

        if selected_prompt:
            _sid = selected_prompt["id"]

            # Поля с текущими значениями (используем отдельные ключи)
            edit_name = st.text_input(
                "Название *",
                value=selected_prompt["name"],
                key=f"edit_name_{_sid}",
            )
            edit_description = st.text_input(
                "Описание",
                value=selected_prompt.get("description", ""),
                key=f"edit_desc_{_sid}",
            )
            edit_text = st.text_area(
                "Текст промпта *",
                value=selected_prompt["text"],
                key=f"edit_text_{_sid}",
                height=200,
            )

            col_save, col_cancel = st.columns(2)

            with col_save:
                if st.button("💾 Сохранить изменения", key="save_edit_btn", type="primary"):
                    if not edit_name.strip() or not edit_text.strip():
                        st.error("❌ Название и текст промпта обязательны.")
                    else:
                        try:
                            updated = update_prompt(
                                _sid,
                                name=edit_name,
                                description=edit_description,
                                text=edit_text,
                            )
                            st.success(f"✅ Промпт «{updated['name']}» обновлён!")
                            # Очистить ID редактирования
                            if "edit_prompt_id" in st.session_state:
                                del st.session_state["edit_prompt_id"]
                            st.rerun()
                        except ValueError as e:
                            st.error(f"❌ {e}")

            with col_cancel:
                if st.button("❌ Отмена", key="cancel_edit_btn"):
                    if "edit_prompt_id" in st.session_state:
                        del st.session_state["edit_prompt_id"]
                    st.rerun()

            # Метаинформация
            st.caption(
                f"Создан: {selected_prompt.get('created_at', '?')}  |  "
                f"Обновлён: {selected_prompt.get('updated_at', '?')}"
            )