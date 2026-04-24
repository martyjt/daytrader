"""Exploration Agent — hunts for predictive features against live data.

One ``scan()`` run:

1. Loads a target OHLCV window for (symbol, timeframe, range).
2. Builds the prediction target:
     * Classification: 1 if next-bar return > 0 else 0.
     * Regression: next-bar log return.
3. Builds a **baseline feature matrix** from price-derived features
   only (log returns, rolling vol, RSI, ATR proxy). These are
   intentionally modest — if a candidate feature adds lift on top of
   these, it's providing genuinely new information.
4. Gathers **candidate features** from the registered adapters:
     * FRED macro series (curated list + any user-specified IDs)
     * NewsAPI sentiment: rolling mean of VADER scores per query term
5. For each candidate, aligns it to the target's time index
   (forward-fill), then runs the feature-lift test.
6. Applies Benjamini-Hochberg FDR correction across all candidates.
7. Writes one ``DiscoveryModel`` row per candidate (accepted or not)
   so the UI can display and filter them.

This is **v1**: single-threaded, one symbol per scan, synchronous
lift tests. Parallelism and ticker fan-out come later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np
import polars as pl

from ..core.context import tenant_scope
from ..core.types.bars import Timeframe
from ..core.types.symbols import Symbol
from ..storage.database import get_session
from ..storage.models import DiscoveryModel
from .feature_lift import benjamini_hochberg, feature_lift, q_values


@dataclass
class ExplorationConfig:
    """How aggressive a scan to run."""

    n_folds: int = 5
    min_train: int = 60
    fdr_alpha: float = 0.1
    task: str = "classification"  # "classification" | "regression"
    sentiment_window_bars: int = 5  # rolling mean window for sentiment
    include_fred: bool = True
    fred_series_ids: list[str] | None = None  # None = use curated default
    sentiment_queries: list[str] = field(default_factory=list)


@dataclass
class CandidateFeature:
    """One candidate column aligned to the target series."""

    name: str          # human-readable label, e.g. "fred:DGS10"
    source: str        # "fred" | "sentiment" | "cross_asset"
    values: np.ndarray # same length as target, aligned
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResult:
    """Summary of one scan run."""

    target_symbol: str
    target_timeframe: str
    n_bars: int
    n_candidates: int
    n_significant: int
    written_rows: list[UUID] = field(default_factory=list)


class ExplorationAgent:
    """Proposes candidate features and scores them against a target series."""

    def __init__(self, config: ExplorationConfig | None = None) -> None:
        self._config = config or ExplorationConfig()

    async def scan(
        self,
        *,
        tenant_id: UUID,
        symbol_str: str,
        timeframe_str: str,
        start: datetime,
        end: datetime,
    ) -> ScanResult:
        symbol = Symbol.parse(symbol_str)
        timeframe = Timeframe(timeframe_str)

        # ---- 1. Target OHLCV -------------------------------------------
        price_df = await self._fetch_prices(symbol, timeframe, start, end)
        if price_df.is_empty() or len(price_df) < (self._config.min_train + 10):
            return ScanResult(
                target_symbol=symbol.key,
                target_timeframe=timeframe.value,
                n_bars=len(price_df),
                n_candidates=0,
                n_significant=0,
            )

        closes = price_df["close"].to_numpy().astype(float)
        highs = price_df["high"].to_numpy().astype(float)
        lows = price_df["low"].to_numpy().astype(float)
        volumes = price_df["volume"].to_numpy().astype(float)
        timestamps = list(price_df["timestamp"].to_list())

        # ---- 2. Target vector ------------------------------------------
        y = self._build_target(closes)

        # ---- 3. Baseline feature matrix --------------------------------
        X_base = self._build_baseline(closes, highs, lows, volumes)

        # Drop initial bars where baseline features have NaN from lookback.
        # Use the first fully-valid row as the start.
        valid_from = self._first_valid_row(X_base)
        if valid_from + self._config.min_train >= len(y):
            return ScanResult(
                target_symbol=symbol.key,
                target_timeframe=timeframe.value,
                n_bars=len(closes),
                n_candidates=0,
                n_significant=0,
            )

        # ---- 4. Gather candidates --------------------------------------
        candidates = await self._gather_candidates(timestamps, start, end)

        # ---- 5. Align + lift test each candidate -----------------------
        results: list[tuple[CandidateFeature, Any]] = []
        p_values: list[float | None] = []
        for cand in candidates:
            if cand.values.shape != y.shape:
                continue
            try:
                lift = feature_lift(
                    y=y[valid_from:],
                    X_base=X_base[valid_from:],
                    X_cand=cand.values[valid_from:],
                    task=self._config.task,
                    n_folds=self._config.n_folds,
                    min_train=self._config.min_train,
                )
            except Exception as exc:  # noqa: BLE001
                # Record a skip row with p=None so BH ignores it.
                results.append((cand, _FailedLift(error=str(exc))))
                p_values.append(None)
                continue
            results.append((cand, lift))
            p_values.append(lift.p_value)

        # ---- 6. Multi-test correction ----------------------------------
        sig_flags = benjamini_hochberg(p_values, alpha=self._config.fdr_alpha)
        qs = q_values(p_values)

        # ---- 7. Persist ------------------------------------------------
        written: list[UUID] = []
        n_significant = 0
        async with get_session() as session:
            with tenant_scope(tenant_id):
                for (cand, lift), is_sig, q in zip(results, sig_flags, qs):
                    if isinstance(lift, _FailedLift):
                        row = DiscoveryModel(
                            tenant_id=tenant_id,
                            candidate_name=cand.name,
                            candidate_source=cand.source,
                            target_symbol=symbol.key,
                            target_timeframe=timeframe.value,
                            baseline_metric=0.0,
                            candidate_metric=0.0,
                            lift=0.0,
                            p_value=None,
                            q_value=None,
                            significant=False,
                            n_folds=0,
                            status="error",
                            meta={"error": lift.error, **cand.metadata},
                        )
                    else:
                        # "Significant" requires both lift positive (candidate is
                        # better than baseline) and FDR-accepted.
                        significant = bool(is_sig and lift.lift > 0)
                        if significant:
                            n_significant += 1
                        row = DiscoveryModel(
                            tenant_id=tenant_id,
                            candidate_name=cand.name,
                            candidate_source=cand.source,
                            target_symbol=symbol.key,
                            target_timeframe=timeframe.value,
                            baseline_metric=lift.baseline_metric,
                            candidate_metric=lift.candidate_metric,
                            lift=lift.lift,
                            p_value=lift.p_value,
                            q_value=q,
                            significant=significant,
                            n_folds=lift.n_folds,
                            status="new",
                            meta={"task": lift.task, **cand.metadata},
                        )
                    session.add(row)
                    await session.flush()
                    written.append(row.id)
                await session.commit()

        return ScanResult(
            target_symbol=symbol.key,
            target_timeframe=timeframe.value,
            n_bars=len(closes),
            n_candidates=len(candidates),
            n_significant=n_significant,
            written_rows=written,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _fetch_prices(
        self, symbol: Symbol, timeframe: Timeframe, start: datetime, end: datetime,
    ) -> pl.DataFrame:
        # Reuse the backtest engine's cached fetch so we share Parquet cache.
        from ..backtest.engine import _fetch_ohlcv_cached
        from ..data.adapters.registry import AdapterRegistry

        AdapterRegistry.auto_register()
        # Prefer binance_public for crypto (proper 4h support, free); fall back
        # to yfinance otherwise.
        from ..core.types.symbols import AssetClass

        adapter_name = (
            "binance_public"
            if symbol.asset_class == AssetClass.CRYPTO
            and "binance_public" in AdapterRegistry.available()
            else "yfinance"
        )
        adapter = AdapterRegistry.get(adapter_name)
        return await _fetch_ohlcv_cached(adapter, symbol, timeframe, start, end)

    def _build_target(self, closes: np.ndarray) -> np.ndarray:
        """Build the prediction target, aligned to ``closes``."""
        # Next-bar log return. Last bar has no target — fill with 0 (it gets
        # dropped at the first_valid_row stage anyway).
        log_ret = np.zeros(len(closes), dtype=float)
        valid = np.where(closes > 0)[0]
        if len(valid) < 2:
            return log_ret
        ratio = np.full(len(closes), np.nan)
        ratio[:-1] = closes[1:] / closes[:-1]
        log_ret_full = np.log(np.where(ratio > 0, ratio, np.nan))
        if self._config.task == "classification":
            # Direction: 1 if next return > 0, else 0. NaN stays as 0 for
            # XGBoost (it'll learn to ignore those in test folds anyway).
            return (log_ret_full > 0).astype(float)
        # Regression: log return. Replace NaN with 0.
        return np.nan_to_num(log_ret_full, nan=0.0)

    def _build_baseline(
        self, closes: np.ndarray, highs: np.ndarray,
        lows: np.ndarray, volumes: np.ndarray,
    ) -> np.ndarray:
        """Modest price-derived feature matrix: returns, volatility, RSI, HL range."""
        from ..algorithms.indicators import rsi

        n = len(closes)

        log_r1 = _safe_log_ratio(closes, 1)
        log_r5 = _safe_log_ratio(closes, 5)
        log_r20 = _safe_log_ratio(closes, 20)

        # Rolling std of 1-bar returns
        vol_20 = _rolling_std(log_r1, 20)

        # RSI-14
        rsi_14 = rsi(closes, 14)

        # HL range / close
        hl_rng = np.full(n, np.nan)
        hl_rng[closes > 0] = (highs - lows)[closes > 0] / closes[closes > 0]

        # Volume z-score (20-bar)
        vol_mean = _rolling_mean(volumes, 20)
        vol_std = _rolling_std(volumes, 20)
        vol_z = np.full(n, np.nan)
        safe = (~np.isnan(vol_std)) & (vol_std > 0)
        vol_z[safe] = (volumes[safe] - vol_mean[safe]) / vol_std[safe]

        return np.column_stack([log_r1, log_r5, log_r20, vol_20, rsi_14, hl_rng, vol_z])

    @staticmethod
    def _first_valid_row(X: np.ndarray) -> int:
        """Index of the first row where all columns are non-NaN."""
        valid_mask = ~np.isnan(X).any(axis=1)
        idx = np.argmax(valid_mask)
        return int(idx) if valid_mask[idx] else len(X)

    async def _gather_candidates(
        self,
        timestamps: list[Any],
        start: datetime,
        end: datetime,
    ) -> list[CandidateFeature]:
        """Collect candidate columns from each registered adapter."""
        out: list[CandidateFeature] = []
        if self._config.include_fred:
            out.extend(await self._gather_fred(timestamps, start, end))
        if self._config.sentiment_queries:
            out.extend(await self._gather_sentiment(timestamps, start, end))
        return out

    async def _gather_fred(
        self, timestamps: list[Any], start: datetime, end: datetime,
    ) -> list[CandidateFeature]:
        from ..data.macro.base import MacroAdapterRegistry
        from ..data.macro.fred_adapter import CURATED_SERIES

        if "fred" not in MacroAdapterRegistry.available():
            return []
        adapter = MacroAdapterRegistry.get("fred")

        ids = self._config.fred_series_ids or [s.series_id for s in CURATED_SERIES]
        out: list[CandidateFeature] = []
        for sid in ids:
            try:
                df = await adapter.fetch_series(sid, start, end)
            except Exception:  # noqa: BLE001 — skip unavailable series
                continue
            if df.is_empty():
                continue
            aligned = _align_to_timestamps(
                df["timestamp"].to_list(), df["value"].to_numpy().astype(float),
                timestamps,
            )
            out.append(CandidateFeature(
                name=f"fred:{sid}",
                source="fred",
                values=aligned,
                metadata={"series_id": sid},
            ))
        return out

    async def _gather_sentiment(
        self, timestamps: list[Any], start: datetime, end: datetime,
    ) -> list[CandidateFeature]:
        from ..data.sentiment.base import SentimentAdapterRegistry

        if "newsapi" not in SentimentAdapterRegistry.available():
            return []
        adapter = SentimentAdapterRegistry.get("newsapi")

        out: list[CandidateFeature] = []
        for query in self._config.sentiment_queries:
            try:
                events = await adapter.fetch_news(query, start, end, limit=100)
            except Exception:  # noqa: BLE001
                continue
            if not events:
                continue
            # Aggregate to the target's cadence: rolling mean sentiment.
            scores = [e.sentiment_score for e in events if e.sentiment_score is not None]
            if not scores:
                continue
            # Bucket by day (or whatever cadence the target has) then
            # forward-fill.
            bucketed = _bucket_sentiment(events, timestamps)
            if bucketed is None:
                continue
            # Rolling mean to smooth.
            smoothed = _rolling_mean(bucketed, self._config.sentiment_window_bars)
            out.append(CandidateFeature(
                name=f"sentiment:{query}",
                source="sentiment",
                values=smoothed,
                metadata={"query": query, "n_events": len(events)},
            ))
        return out


# ----------------------------------------------------------------------
# Standalone helpers
# ----------------------------------------------------------------------


@dataclass
class _FailedLift:
    error: str


def _safe_log_ratio(x: np.ndarray, lag: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if lag >= len(x):
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = x[lag:] / x[:-lag]
        out[lag:] = np.log(np.where(ratio > 0, ratio, np.nan))
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < window:
        return out
    for i in range(window - 1, len(x)):
        slice_ = x[i - window + 1 : i + 1]
        valid = ~np.isnan(slice_)
        if valid.any():
            out[i] = float(np.mean(slice_[valid]))
    return out


def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < window:
        return out
    for i in range(window - 1, len(x)):
        slice_ = x[i - window + 1 : i + 1]
        valid = ~np.isnan(slice_)
        if valid.sum() >= 2:
            out[i] = float(np.std(slice_[valid], ddof=0))
    return out


def _align_to_timestamps(
    src_ts: list[Any], src_vals: np.ndarray, target_ts: list[Any],
) -> np.ndarray:
    """Forward-fill a sparse source series onto a dense target index.

    Each target timestamp gets the most recent source value at or before
    its own time. Missing values at the start stay NaN — XGBoost handles
    NaN natively.
    """
    out = np.full(len(target_ts), np.nan)
    if not src_ts or len(src_vals) == 0:
        return out
    # Ensure sortable comparables.
    src_pairs = sorted(zip(src_ts, src_vals), key=lambda p: p[0])
    j = 0
    last = np.nan
    for i, t in enumerate(target_ts):
        while j < len(src_pairs) and src_pairs[j][0] <= t:
            last = float(src_pairs[j][1])
            j += 1
        out[i] = last
    return out


def _bucket_sentiment(
    events: list[Any], target_ts: list[Any],
) -> np.ndarray | None:
    """Average sentiment scores into the same cadence as ``target_ts``."""
    if not events or not target_ts:
        return None
    import bisect

    target_ts_sorted = list(target_ts)
    buckets: list[list[float]] = [[] for _ in target_ts_sorted]
    for ev in events:
        if ev.sentiment_score is None:
            continue
        idx = bisect.bisect_right(target_ts_sorted, ev.timestamp) - 1
        if 0 <= idx < len(buckets):
            buckets[idx].append(float(ev.sentiment_score))
    out = np.full(len(target_ts_sorted), np.nan)
    for i, bucket in enumerate(buckets):
        if bucket:
            out[i] = float(np.mean(bucket))
    return out
