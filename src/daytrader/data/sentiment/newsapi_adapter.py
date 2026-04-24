"""NewsAPI.org adapter with optional VADER scoring.

Free tier: 1000 requests/day, 24h article lookback limit (developer plan).
Register at ``https://newsapi.org`` and store ``NEWSAPI_KEY``.

If ``vaderSentiment`` is installed, headlines are scored inline; otherwise
events come back with ``sentiment_score=None`` for downstream scoring.
"""

from __future__ import annotations

import time
from datetime import datetime

import httpx

from ...core.types.common import utcnow
from ..adapters.base import AdapterHealth
from .base import NewsEvent, SentimentAdapter, score_text_vader

_BASE_URL = "https://newsapi.org/v2"


class NewsAPIAdapter(SentimentAdapter):
    """NewsAPI.org sentiment adapter."""

    def __init__(self, api_key: str, timeout: float = 10.0) -> None:
        if not api_key:
            raise ValueError("NewsAPIAdapter requires an API key")
        self._api_key = api_key
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "newsapi"

    async def fetch_news(
        self,
        query: str,
        start: datetime,
        end: datetime,
        *,
        limit: int = 100,
    ) -> list[NewsEvent]:
        params = {
            "q": query,
            "from": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "to": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "sortBy": "publishedAt",
            "pageSize": min(100, max(1, limit)),
            "language": "en",
            "apiKey": self._api_key,
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{_BASE_URL}/everything", params=params)
            resp.raise_for_status()
            payload = resp.json()

        if payload.get("status") != "ok":
            return []

        events: list[NewsEvent] = []
        for article in payload.get("articles", [])[:limit]:
            title = article.get("title") or ""
            snippet = article.get("description") or ""
            published_at = article.get("publishedAt")
            try:
                ts = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
            except (TypeError, ValueError):
                continue

            # Score title + snippet if VADER is available.
            text_for_scoring = f"{title}. {snippet}".strip()
            score = score_text_vader(text_for_scoring) if text_for_scoring else None

            events.append(NewsEvent(
                timestamp=ts,
                source=(article.get("source") or {}).get("name", "unknown"),
                title=title,
                url=article.get("url"),
                snippet=snippet or None,
                sentiment_score=score,
                topic=query,
            ))
        return events

    async def health(self) -> AdapterHealth:
        try:
            t0 = time.monotonic()
            params = {
                "q": "market",
                "pageSize": 1,
                "language": "en",
                "apiKey": self._api_key,
            }
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{_BASE_URL}/everything", params=params)
                resp.raise_for_status()
            elapsed = (time.monotonic() - t0) * 1000
            return AdapterHealth(
                status="ok",
                latency_ms=round(elapsed, 1),
                last_successful_call=utcnow(),
            )
        except Exception as exc:  # noqa: BLE001
            return AdapterHealth(status="down", error=str(exc))
