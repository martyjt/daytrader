"""Feature drift monitor — covariate-shift detection for the baseline pipeline.

The Exploration Agent's baseline feature matrix (log returns, RSI,
volatility, HL range, volume z-score) is assumed stationary when it
evaluates candidate-feature lift. If those baseline distributions drift
over time — a new volatility regime, a regime shift in volume behavior
— *all* of the agent's lift tests become suspect.

This module compares the **reference window** (older half of the
lookback) to the **current window** (newer half) on each baseline
feature using Population Stability Index (PSI). PSI is the de-facto
production drift metric in credit scoring / banking and has
well-understood severity thresholds:

    PSI < 0.1        stable
    0.1 ≤ PSI < 0.25 moderate drift
    PSI ≥ 0.25       significant drift

Output: one ``DriftReport`` per feature with PSI value + severity label,
suitable for rendering in the Discoveries tab or surfacing as a
warning badge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np


@dataclass(frozen=True)
class DriftReport:
    """One feature's drift report."""

    feature: str
    psi: float
    severity: str       # "stable" | "moderate" | "significant" | "insufficient_data"
    reference_mean: float
    reference_std: float
    current_mean: float
    current_std: float
    n_reference: int
    n_current: int


@dataclass
class DriftScanResult:
    """Aggregate drift scan result."""

    target_symbol: str
    target_timeframe: str
    reference_start: datetime | None = None
    reference_end: datetime | None = None
    current_start: datetime | None = None
    current_end: datetime | None = None
    reports: list[DriftReport] = field(default_factory=list)

    @property
    def overall_severity(self) -> str:
        """Worst-case severity across all features."""
        order = {"insufficient_data": -1, "stable": 0, "moderate": 1, "significant": 2}
        if not self.reports:
            return "insufficient_data"
        return max(self.reports, key=lambda r: order.get(r.severity, -1)).severity


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_buckets: int = 10,
    min_prob: float = 1e-4,
) -> float:
    """Population Stability Index between two distributions.

    Buckets the reference into ``n_buckets`` quantile-based bins, maps
    the current distribution into the same bins, and sums
    ``(p_cur - p_ref) * log(p_cur / p_ref)`` over bins.

    Zero-count bins are floored at ``min_prob`` to avoid ``log(0)``.
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) < n_buckets or len(cur) < n_buckets:
        return float("nan")

    # Quantile edges from reference.
    qs = np.linspace(0, 1, n_buckets + 1)
    edges = np.unique(np.quantile(ref, qs))
    if len(edges) < 3:
        return 0.0  # essentially constant reference — no meaningful PSI

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_p = np.maximum(ref_counts / len(ref), min_prob)
    cur_p = np.maximum(cur_counts / len(cur), min_prob)

    psi = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    return psi


def classify_psi(psi: float) -> str:
    if not np.isfinite(psi):
        return "insufficient_data"
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"


async def scan_drift(
    *,
    symbol_str: str = "BTC-USD",
    timeframe_str: str = "1d",
    lookback_days: int = 240,
) -> DriftScanResult:
    """Scan baseline-feature drift on a single symbol.

    The lookback is split 50/50: older half = reference, newer half =
    current. Each of the 7 baseline features gets a ``DriftReport``.
    """
    from ..backtest.engine import _fetch_ohlcv_cached
    from ..core.types.bars import Timeframe
    from ..core.types.symbols import AssetClass, Symbol
    from ..data.adapters.registry import AdapterRegistry
    from .exploration_agent import ExplorationAgent

    symbol = Symbol.parse(symbol_str)
    timeframe = Timeframe(timeframe_str)
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)

    AdapterRegistry.auto_register()
    adapter_name = (
        "binance_public"
        if symbol.asset_class == AssetClass.CRYPTO
        and "binance_public" in AdapterRegistry.available()
        else "yfinance"
    )
    adapter = AdapterRegistry.get(adapter_name)
    df = await _fetch_ohlcv_cached(adapter, symbol, timeframe, start, end)

    result = DriftScanResult(
        target_symbol=symbol.key,
        target_timeframe=timeframe.value,
    )
    if df.is_empty() or len(df) < 40:
        return result

    closes = df["close"].to_numpy().astype(float)
    highs = df["high"].to_numpy().astype(float)
    lows = df["low"].to_numpy().astype(float)
    volumes = df["volume"].to_numpy().astype(float)
    timestamps = df["timestamp"].to_list()

    # Build the same baseline as the Exploration Agent so drift alerts
    # speak to what that agent actually trains on.
    agent = ExplorationAgent()
    X_base = agent._build_baseline(closes, highs, lows, volumes)

    feature_names = [
        "log_r1", "log_r5", "log_r20",
        "vol_20", "rsi_14", "hl_range", "volume_z",
    ]

    half = len(closes) // 2
    reports: list[DriftReport] = []
    for i, name in enumerate(feature_names):
        ref = X_base[:half, i]
        cur = X_base[half:, i]
        psi = compute_psi(ref, cur)
        severity = classify_psi(psi)
        ref_valid = ref[~np.isnan(ref)]
        cur_valid = cur[~np.isnan(cur)]
        reports.append(DriftReport(
            feature=name,
            psi=psi if np.isfinite(psi) else 0.0,
            severity=severity,
            reference_mean=float(np.mean(ref_valid)) if len(ref_valid) else 0.0,
            reference_std=float(np.std(ref_valid, ddof=0)) if len(ref_valid) else 0.0,
            current_mean=float(np.mean(cur_valid)) if len(cur_valid) else 0.0,
            current_std=float(np.std(cur_valid, ddof=0)) if len(cur_valid) else 0.0,
            n_reference=len(ref_valid),
            n_current=len(cur_valid),
        ))

    result.reports = reports
    result.reference_start = timestamps[0] if timestamps else None
    result.reference_end = timestamps[half] if len(timestamps) > half else None
    result.current_start = timestamps[half] if len(timestamps) > half else None
    result.current_end = timestamps[-1] if timestamps else None

    # Fire an alert if any feature shows significant drift.
    sig_features = [r.feature for r in reports if r.severity == "significant"]
    if sig_features:
        try:
            from ..ui.alerts import alerts as _alerts

            _alerts().add(
                level="warning",
                title=f"Feature drift: {len(sig_features)} features significant",
                body=(
                    f"PSI > 0.25 on {', '.join(sig_features[:5])}"
                    + (f" (+{len(sig_features) - 5} more)" if len(sig_features) > 5 else "")
                    + f" — baseline distributions on {symbol_str} {timeframe_str} "
                    "have shifted. Exploration Agent lift tests may be stale."
                ),
                source="drift",
                data={
                    "symbol": symbol_str,
                    "timeframe": timeframe_str,
                    "significant_features": sig_features,
                },
            )
        except Exception:
            pass
    return result
