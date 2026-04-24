"""Regime detection service — for the shell-header Regime Badge.

Runs the ``RegimeHMMAlgorithm`` on a recent window of OHLCV for a
chosen "pulse" symbol (defaults to BTC-USD daily) and returns
``{bull, bear, sideways}`` probabilities. Cached for 5 minutes so
page navigation doesn't re-fit the HMM.

Design notes:

* The badge is informational — it's a broad-market pulse, not a
  persona-specific regime. One model fit per pulse symbol.
* Cache lives in-process. A restart triggers a fresh fit on next read.
* If the HMM library or adapters are unavailable the service returns
  ``RegimeSnapshot(status="unavailable")`` so the widget hides gracefully.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


_CACHE_TTL_SECONDS = 300.0  # 5 minutes


@dataclass
class RegimeSnapshot:
    """One broad-market pulse reading."""

    status: str                           # "ok" | "unavailable" | "warming_up"
    regime: str = "unknown"               # "bull" | "bear" | "sideways" | "unknown"
    probabilities: dict[str, float] = field(default_factory=dict)
    pulse_symbol: str = ""
    pulse_timeframe: str = ""
    bars_analyzed: int = 0
    as_of: datetime | None = None
    message: str = ""


# ---- in-process cache -----------------------------------------------------

_cache: dict[tuple[str, str], tuple[float, RegimeSnapshot]] = {}
_cache_lock = asyncio.Lock()


async def get_current_regime(
    *,
    symbol_str: str = "BTC-USD",
    timeframe_str: str = "1d",
    lookback_days: int = 180,
    force_refresh: bool = False,
) -> RegimeSnapshot:
    """Return the most recent broad-market regime reading.

    Args:
        symbol_str: pulse symbol whose regime stands in for "the market".
        timeframe_str: timeframe for HMM input bars.
        lookback_days: how much history to fit on.
        force_refresh: bypass the cache.
    """
    key = (symbol_str, timeframe_str)
    now = time.monotonic()

    if not force_refresh:
        cached = _cache.get(key)
        if cached is not None and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return cached[1]

    async with _cache_lock:
        # Re-check under lock (cheap hit avoids duplicate fits across concurrent pages).
        if not force_refresh:
            cached = _cache.get(key)
            if cached is not None and (now - cached[0]) < _CACHE_TTL_SECONDS:
                return cached[1]

        snapshot = await _compute_regime(symbol_str, timeframe_str, lookback_days)
        _cache[key] = (time.monotonic(), snapshot)
        return snapshot


_last_fired_regime: dict[tuple[str, str], str] = {}


async def _compute_regime(
    symbol_str: str, timeframe_str: str, lookback_days: int,
) -> RegimeSnapshot:
    try:
        from ..algorithms.builtin.regime_hmm import _build_hmm_features
        from hmmlearn.hmm import GaussianHMM
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        return RegimeSnapshot(
            status="unavailable",
            message=f"HMM dependencies missing: {exc}",
            pulse_symbol=symbol_str,
            pulse_timeframe=timeframe_str,
        )

    try:
        import numpy as np

        from ..backtest.engine import _fetch_ohlcv_cached
        from ..core.types.bars import Timeframe
        from ..core.types.symbols import AssetClass, Symbol
        from ..data.adapters.registry import AdapterRegistry
    except ImportError as exc:
        return RegimeSnapshot(
            status="unavailable",
            message=f"Adapter deps missing: {exc}",
            pulse_symbol=symbol_str,
            pulse_timeframe=timeframe_str,
        )

    AdapterRegistry.auto_register()
    symbol = Symbol.parse(symbol_str)
    timeframe = Timeframe(timeframe_str)
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    adapter_name = (
        "binance_public"
        if symbol.asset_class == AssetClass.CRYPTO
        and "binance_public" in AdapterRegistry.available()
        else "yfinance"
    )
    try:
        adapter = AdapterRegistry.get(adapter_name)
        df = await _fetch_ohlcv_cached(adapter, symbol, timeframe, start, end)
    except Exception as exc:  # noqa: BLE001
        return RegimeSnapshot(
            status="unavailable",
            message=f"Data fetch failed: {exc}",
            pulse_symbol=symbol_str,
            pulse_timeframe=timeframe_str,
        )

    if df.is_empty() or len(df) < 60:
        return RegimeSnapshot(
            status="warming_up",
            message=f"Only {len(df)} bars available; need at least 60.",
            pulse_symbol=symbol_str,
            pulse_timeframe=timeframe_str,
            bars_analyzed=len(df),
        )

    # Fit HMM in a thread so the event loop keeps serving other requests.
    import numpy as np

    closes = df["close"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    volumes = df["volume"].to_numpy().astype(float)

    def _fit_and_predict() -> tuple[str, dict[str, float]]:
        features = _build_hmm_features(closes, highs, lows, volumes)
        valid_mask = ~np.isnan(features).any(axis=1)
        valid = features[valid_mask]
        if len(valid) < 30:
            raise RuntimeError(f"Not enough valid feature rows: {len(valid)}")

        scaler = StandardScaler()
        scaled = scaler.fit_transform(valid)
        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(scaled)

        # Label states by their mean log return: lowest = bear, highest = bull.
        return_means = model.means_[:, 0]
        sorted_states = np.argsort(return_means)
        labels: dict[int, str] = {
            int(sorted_states[0]): "bear",
            int(sorted_states[1]): "sideways",
            int(sorted_states[2]): "bull",
        }

        probs = model.predict_proba(scaled)[-1]
        label_probs = {
            labels[i]: float(probs[i]) for i in range(len(probs))
        }
        top_regime = max(label_probs, key=label_probs.get)
        return top_regime, label_probs

    try:
        top_regime, label_probs = await asyncio.to_thread(_fit_and_predict)
    except Exception as exc:  # noqa: BLE001
        return RegimeSnapshot(
            status="unavailable",
            message=f"HMM fit failed: {exc}",
            pulse_symbol=symbol_str,
            pulse_timeframe=timeframe_str,
        )

    snapshot = RegimeSnapshot(
        status="ok",
        regime=top_regime,
        probabilities=label_probs,
        pulse_symbol=symbol_str,
        pulse_timeframe=timeframe_str,
        bars_analyzed=len(df),
        as_of=datetime.utcnow(),
    )

    # Fire an alert on first observation or whenever the top regime changes.
    key = (symbol_str, timeframe_str)
    prev = _last_fired_regime.get(key)
    if prev != top_regime:
        _last_fired_regime[key] = top_regime
        if prev is not None:
            try:
                from .alerts import alerts as _alerts

                top_pct = int(round(label_probs.get(top_regime, 0.0) * 100))
                _alerts().add(
                    level="warning" if top_regime == "bear" else "info",
                    title=f"Regime change: {prev.upper()} → {top_regime.upper()}",
                    body=(
                        f"Broad-market pulse on {symbol_str} {timeframe_str} "
                        f"shifted from {prev} to {top_regime} ({top_pct}% "
                        f"probability). Review algos with mismatched "
                        f"suitable_regimes."
                    ),
                    source="regime",
                    data={"from": prev, "to": top_regime, "probabilities": label_probs},
                )
            except Exception:
                pass

    return snapshot


def invalidate_regime_cache() -> None:
    """Drop cached regime snapshots — useful after env changes."""
    _cache.clear()
