"""SentimentAdapter ABC — news, social, event streams with optional scoring.

A sentiment adapter returns a list of ``NewsEvent`` records for a given
query and time window. Events carry a title, source, timestamp, URL, and
an optional normalized score in ``[-1, +1]``. Adapters that don't score
inline leave ``sentiment_score=None``; downstream features can score on
demand.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

from ..adapters.base import AdapterHealth


@dataclass(frozen=True, slots=True)
class NewsEvent:
    """One scored (or scoreable) news/event record."""

    timestamp: datetime
    source: str
    title: str
    url: str | None = None
    snippet: str | None = None
    sentiment_score: float | None = None  # [-1, +1] or None
    topic: str | None = None


class SentimentAdapter(ABC):
    """Abstract base for news/social/event data adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique adapter identifier, e.g. ``'newsapi'``."""

    @abstractmethod
    async def fetch_news(
        self,
        query: str,
        start: datetime,
        end: datetime,
        *,
        limit: int = 100,
    ) -> list[NewsEvent]:
        """Fetch news events matching the query in the given window."""

    @abstractmethod
    async def health(self) -> AdapterHealth:
        """Probe operational health."""


class SentimentAdapterRegistry:
    """Registry of available sentiment adapters."""

    _adapters: ClassVar[dict[str, SentimentAdapter]] = {}

    @classmethod
    def register(cls, adapter: SentimentAdapter) -> None:
        cls._adapters[adapter.name] = adapter

    @classmethod
    def get(cls, name: str) -> SentimentAdapter:
        if name not in cls._adapters:
            raise KeyError(
                f"Sentiment adapter {name!r} not registered. "
                f"Available: {sorted(cls._adapters)}"
            )
        return cls._adapters[name]

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._adapters)

    @classmethod
    def auto_register(cls) -> None:
        """Register built-in sentiment adapters if their API keys are present."""
        if "newsapi" not in cls._adapters:
            try:
                from ...core.settings import get_settings

                key = get_settings().newsapi_key.get_secret_value()
                if key:
                    from .newsapi_adapter import NewsAPIAdapter

                    cls.register(NewsAPIAdapter(api_key=key))
            except Exception:
                pass

    @classmethod
    def clear(cls) -> None:
        cls._adapters.clear()


def score_text_vader(text: str) -> float | None:
    """Score a string in ``[-1, +1]`` using VADER, if available.

    Returns ``None`` if VADER isn't installed — sentiment adapters then
    return unscored events and callers can run a model of their choice.
    """
    try:
        from vaderSentiment.vaderSentiment import (  # noqa: F401
            SentimentIntensityAnalyzer,
        )
    except ImportError:
        return None
    analyzer = _get_vader()
    if analyzer is None:
        return None
    score = analyzer.polarity_scores(text)
    return float(score.get("compound", 0.0))


_vader_instance: Any = None


def _get_vader() -> Any:
    """Lazy-singleton VADER analyzer (initialization is non-trivial)."""
    global _vader_instance
    if _vader_instance is not None:
        return _vader_instance
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _vader_instance = SentimentIntensityAnalyzer()
        return _vader_instance
    except Exception:
        return None
