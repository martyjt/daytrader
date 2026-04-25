"""Feature hydration — fetch a discovered feature's current value at bar time.

The Exploration Agent measures *offline* whether a candidate feature
(FRED series, news sentiment, another asset's price) carries lift over
the price-only baseline. To trade on that signal live, the trading loop
needs the feature's *current* value injected into
``AlgorithmContext.features`` before each bar.

This module owns that responsibility. ``hydrate_feature`` takes a
``DiscoveryModel`` and a "now" timestamp, dispatches to the right
adapter by ``candidate_source``, and returns a single float — or
``None`` if the upstream source is unreachable, returns no data, or
isn't configured. The trading loop treats ``None`` as "feature
missing" and the algorithm decides how to behave (typically: stay
flat).

Three sources are supported, matching what the Exploration Agent
produces in ``exploration_agent._gather_candidates``:

* ``fred`` — pulls the latest observation from the FRED macro adapter
  for the series id stored in ``discovery.meta["series_id"]``.
* ``sentiment`` — fetches recent news for the query stored in
  ``discovery.meta["query"]`` and returns the mean VADER score.
* ``cross_asset`` — fetches the latest close for the symbol stored in
  ``discovery.meta["symbol"]`` from the OHLCV adapter registry.

A small in-memory TTL cache (default 60s) avoids hammering upstream
APIs when multiple personas share the same discovery — important for
FRED whose free tier is rate-limited.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# How long a hydrated value stays valid in cache. Macro and sentiment
# series move on cadences of hours-to-days, so a minute is plenty;
# cross-asset close prices update bar-by-bar but we accept the cache
# horizon as good enough for the discovery promotion use case.
_DEFAULT_TTL_SECONDS = 60.0


@dataclass(frozen=True)
class _CacheKey:
    source: str
    key: str  # series_id / query / symbol


@dataclass
class _CacheEntry:
    value: float | None
    expires_at: float  # monotonic seconds


class FeatureHydrator:
    """Resolves discovered features to current numeric values.

    One instance lives per process; the trading loop reuses it across
    cycles and personas via ``get_feature_hydrator()``.
    """

    def __init__(self, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> None:
        self._ttl = ttl_seconds
        self._cache: dict[_CacheKey, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def hydrate(
        self,
        discovery: Any,
        *,
        now: datetime | None = None,
    ) -> float | None:
        """Resolve ``discovery``'s feature to a float at ``now``.

        Returns ``None`` when the source is unconfigured, the upstream
        call fails, or the response is empty. Callers must handle
        ``None`` — it is the "feature missing" signal, not an error.
        """
        source = (discovery.candidate_source or "").lower()
        meta = dict(discovery.meta or {})

        try:
            if source == "fred":
                return await self._hydrate_fred(meta, now=now)
            if source == "sentiment":
                return await self._hydrate_sentiment(meta, now=now)
            if source == "cross_asset":
                return await self._hydrate_cross_asset(meta, now=now)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Feature hydration failed for source=%s name=%s",
                source, discovery.candidate_name,
            )
            return None

        logger.warning(
            "Unknown candidate_source %r on discovery %s — cannot hydrate",
            source, getattr(discovery, "id", "<no id>"),
        )
        return None

    # ------------------------------------------------------------------
    # Source dispatch
    # ------------------------------------------------------------------

    async def _hydrate_fred(
        self, meta: dict[str, Any], *, now: datetime | None
    ) -> float | None:
        series_id = meta.get("series_id")
        if not series_id:
            return None

        cached = self._cache_get(_CacheKey("fred", str(series_id)))
        if cached is not None:
            return cached

        from ..data.macro.base import MacroAdapterRegistry

        if "fred" not in MacroAdapterRegistry.available():
            return self._cache_put(_CacheKey("fred", str(series_id)), None)

        adapter = MacroAdapterRegistry.get("fred")
        end = now or datetime.now(timezone.utc)
        # FRED daily series — pull a 30-day window so we always get the
        # most recent observation even for weekly/monthly cadences.
        start = end - timedelta(days=30)
        df = await adapter.fetch_series(str(series_id), start, end)
        value: float | None = None
        if not df.is_empty():
            last = df["value"].to_list()[-1]
            value = float(last) if last is not None else None
        return self._cache_put(_CacheKey("fred", str(series_id)), value)

    async def _hydrate_sentiment(
        self, meta: dict[str, Any], *, now: datetime | None
    ) -> float | None:
        query = meta.get("query")
        if not query:
            return None

        cached = self._cache_get(_CacheKey("sentiment", str(query)))
        if cached is not None:
            return cached

        from ..data.sentiment.base import SentimentAdapterRegistry

        if "newsapi" not in SentimentAdapterRegistry.available():
            return self._cache_put(_CacheKey("sentiment", str(query)), None)

        adapter = SentimentAdapterRegistry.get("newsapi")
        end = now or datetime.now(timezone.utc)
        # NewsAPI free tier limits historical depth; 24h is enough for a
        # rolling sentiment read.
        start = end - timedelta(days=1)
        events = await adapter.fetch_news(str(query), start, end, limit=100)
        scores = [
            float(e.sentiment_score) for e in events
            if e.sentiment_score is not None
        ]
        value = sum(scores) / len(scores) if scores else None
        return self._cache_put(_CacheKey("sentiment", str(query)), value)

    async def _hydrate_cross_asset(
        self, meta: dict[str, Any], *, now: datetime | None
    ) -> float | None:
        symbol_raw = meta.get("symbol")
        if not symbol_raw:
            return None

        cached = self._cache_get(_CacheKey("cross_asset", str(symbol_raw)))
        if cached is not None:
            return cached

        from ..core.types.bars import Timeframe
        from ..core.types.symbols import AssetClass, Symbol
        from ..data.adapters.registry import AdapterRegistry

        AdapterRegistry.auto_register()
        try:
            symbol = Symbol.parse(str(symbol_raw))
        except Exception:  # noqa: BLE001
            logger.warning(
                "cross_asset symbol %r could not be parsed", symbol_raw,
            )
            return self._cache_put(
                _CacheKey("cross_asset", str(symbol_raw)), None
            )

        adapter_name = (
            "binance_public"
            if symbol.asset_class == AssetClass.CRYPTO
            and "binance_public" in AdapterRegistry.available()
            else "yfinance"
        )
        try:
            adapter = AdapterRegistry.get(adapter_name)
        except KeyError:
            return self._cache_put(
                _CacheKey("cross_asset", str(symbol_raw)), None
            )

        timeframe = Timeframe(meta.get("timeframe", "1d"))
        end = now or datetime.now(timezone.utc)
        start = end - timedelta(days=10)
        df = await adapter.fetch_ohlcv(symbol, timeframe, start, end)
        value: float | None = None
        if not df.is_empty():
            last = df["close"].to_list()[-1]
            value = float(last) if last is not None else None
        return self._cache_put(_CacheKey("cross_asset", str(symbol_raw)), value)

    # ------------------------------------------------------------------
    # Cache plumbing
    # ------------------------------------------------------------------

    def _cache_get(self, key: _CacheKey) -> float | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.expires_at < time.monotonic():
            self._cache.pop(key, None)
            return None
        return entry.value

    def _cache_put(self, key: _CacheKey, value: float | None) -> float | None:
        self._cache[key] = _CacheEntry(
            value=value, expires_at=time.monotonic() + self._ttl,
        )
        return value

    def clear(self) -> None:
        """Drop all cached values. Used by tests."""
        self._cache.clear()


_singleton: FeatureHydrator | None = None


def get_feature_hydrator() -> FeatureHydrator:
    """Return the process-wide ``FeatureHydrator`` instance."""
    global _singleton  # noqa: PLW0603
    if _singleton is None:
        _singleton = FeatureHydrator()
    return _singleton


def reset_feature_hydrator() -> None:
    """Reset the singleton — for tests only."""
    global _singleton  # noqa: PLW0603
    _singleton = None
