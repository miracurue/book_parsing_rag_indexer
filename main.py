"""Главная страница — Dashboard / статус конвейера подготовки данных для RAG."""

import streamlit as st

from src.config import get_secret

st.set_page_config(
    page_title="Book Parsing RAG Indexer",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Book Parsing RAG Indexer — Dashboard")
st.markdown("---")

st.markdown("""
### Конвейер подготовки данных для RAG: парсинг книг и статей → векторная БД Qdrant

Используйте меню слева для навигации по страницам:

| Страница | Описание |
|----------|----------|
| 📂 **Препроцессинг** | Разделение PDF, конвертация, OCR |
| 🤖 **Парсинг** | AI-парсинг страниц через VLM |
| 🔧 **Постпроцессинг** | Нумерация, объединение по главам, чанкирование |
| 🧠 **Векторизация** | Создание эмбеддингов (dense + sparse), индексация в Qdrant |
| 📝 **Промпты** | Управление промптами |
| 📰 **Статьи** | Парсинг PDF-статей через Docling |
| 📥 **Загрузчик arXiv** | Поиск и скачивание статей с arXiv.org |
""")

# ── Qdrant Dashboard ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗄️ Qdrant Vector DB")

qdrant_host = get_secret("QDRANT_HOST", "localhost")
qdrant_port = get_secret("QDRANT_PORT", "6333")
qdrant_url = f"http://{qdrant_host}:{qdrant_port}"

# Проверка доступности Qdrant
try:
    import urllib.request
    resp = urllib.request.urlopen(f"{qdrant_url}/healthz", timeout=3)
    if resp.status == 200:
        st.success(f"🟢 Qdrant запущен (`{qdrant_host}:{qdrant_port}`)")
        st.link_button(
            "🔗 Открыть Qdrant Dashboard",
            url=f"{qdrant_url}/dashboard",
            use_container_width=True,
        )
    else:
        st.warning("🟡 Qdrant ответил с ошибкой")
except Exception:
    st.warning(
        f"🔴 Qdrant не доступен (`{qdrant_host}:{qdrant_port}`).\n\n"
        "Запустите: `docker compose up -d`"
    )

st.markdown("---")
st.info("💡 Начните с **📂 Препроцессинг** — загрузите и обработайте PDF файлы.")
