"""
Поиск статей на arXiv.org.
Чистые функции — без Streamlit, без побочных эффектов.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import arxiv

from src.config import get_secret

logger = logging.getLogger(__name__)

# ── Настройки retry ────────────────────────────────────────────
_MAX_OUTER_RETRIES = 3
_OUTER_BACKOFF_DELAYS = [5.0, 15.0, 30.0]


def _get_arxiv_user_agent() -> str:
    """Формирует User-Agent для arXiv API с email из конфига."""
    email = get_secret("ARXIV_EMAIL", "")
    if email:
        return f"book_parsing_rag_indexer/1.0 (mailto:{email})"
    return "book_parsing_rag_indexer/1.0"


def _patch_client_ua(client: arxiv.Client) -> None:
    """Обновляет User-Agent в сессии arxiv.Client."""
    ua = _get_arxiv_user_agent()
    client._session.headers.update({"User-Agent": ua})
    logger.debug("arXiv User-Agent: %s", ua)


@dataclass
class ArxivArticle:
    """Унифицированная статья с arXiv."""
    arxiv_id: str
    title: str
    authors: list[str] = field(default_factory=list)
    published: str = ""
    updated: str = ""
    categories: list[str] = field(default_factory=list)
    abstract: str = ""
    pdf_url: str = ""
    entry_id: str = ""

    def safe_filename(self) -> str:
        """Имя файла из arxiv_id (2401.01234 → 2401.01234.pdf)."""
        return f"{self.arxiv_id.replace('/', '_')}.pdf"


def search_articles(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> list[ArxivArticle]:
    """
    Поиск статей на arXiv.

    Parameters
    ----------
    query : str
        Поисковый запрос в формате arXiv API.
        Примеры: ``ti:rag AND abs:medicine``, ``all:briquetting``, ``cat:cs.AI``.
    max_results : int
        Максимум результатов.
    sort_by : str
        Поле сортировки: "relevance", "lastUpdatedDate", "submittedDate".
    sort_order : str
        Порядок: "ascending", "descending".

    Returns
    -------
    list[ArxivArticle]

    Raises
    ------
    arxiv.HTTPError
        Если все попытки исчерпаны (429/5xx).
    """
    # Маппинг сортировки
    sort_map = {
        "relevance": arxiv.SortCriterion.Relevance,
        "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
        "submittedDate": arxiv.SortCriterion.SubmittedDate,
    }
    order_map = {
        "ascending": arxiv.SortOrder.Ascending,
        "descending": arxiv.SortOrder.Descending,
    }

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=sort_map.get(sort_by, arxiv.SortCriterion.Relevance),
        sort_order=order_map.get(sort_order, arxiv.SortOrder.Descending),
    )

    # Внешний retry с exponential backoff поверх встроенного Client retry
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_OUTER_RETRIES):
        try:
            client = arxiv.Client(
                page_size=min(max_results, 50),
                delay_seconds=5.0,
                num_retries=5,
            )
            _patch_client_ua(client)

            articles: list[ArxivArticle] = []
            for result in client.results(search):
                arxiv_id = result.get_short_id()
                article = ArxivArticle(
                    arxiv_id=arxiv_id,
                    title=result.title.replace("\n", " ").strip(),
                    authors=[a.name for a in result.authors],
                    published=str(result.published.date()) if result.published else "",
                    updated=str(result.updated.date()) if result.updated else "",
                    categories=result.categories,
                    abstract=result.summary.replace("\n", " ").strip(),
                    pdf_url=result.pdf_url,
                    entry_id=result.entry_id,
                )
                articles.append(article)

            logger.info("Поиск '%s': найдено %d статей", query, len(articles))
            return articles

        except arxiv.HTTPError as e:
            last_exc = e
            delay = _OUTER_BACKOFF_DELAYS[min(attempt, len(_OUTER_BACKOFF_DELAYS) - 1)]
            logger.warning(
                "arXiv HTTP-ошибка (попытка %d/%d): %s. Повтор через %.0f сек.",
                attempt + 1, _MAX_OUTER_RETRIES, e, delay,
            )
            if attempt < _MAX_OUTER_RETRIES - 1:
                time.sleep(delay)

        except arxiv.UnexpectedEmptyPageError as e:
            last_exc = e
            logger.warning("arXiv: пустой ответ (попытка %d/%d): %s", attempt + 1, _MAX_OUTER_RETRIES, e)
            if attempt < _MAX_OUTER_RETRIES - 1:
                time.sleep(5.0)

    # Все попытки исчерпаны
    raise last_exc  # type: ignore[misc]
