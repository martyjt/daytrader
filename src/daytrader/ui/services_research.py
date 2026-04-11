"""Research Lab service functions.

Orchestrates multi-backtest comparison, parameter sweep, feature
attribution extraction, and walk-forward stability metrics.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from .services import run_backtest_service, run_walk_forward_service


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SweepResult:
    """Result of a hyperparameter sweep."""

    algo_id: str
    param_names: list[str]
    grid_points: list[dict[str, Any]]
    results: list[Any]  # list of BacktestResult
    best_index: int
    best_params: dict[str, Any]
    best_kpis: dict[str, float]


# ---------------------------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------------------------


async def run_comparison_service(
    *,
    algo_ids: list[str],
    symbol_str: str,
    timeframe_str: str,
    start_str: str,
    end_str: str,
    capital: float = 10_000.0,
    venue: str = "binance_spot",
    risk_enabled: bool = False,
) -> list[tuple[str, Any]]:
    """Run backtests for multiple algorithms and return (algo_id, result) pairs."""
    results: list[tuple[str, Any]] = []
    for algo_id in algo_ids:
        result = await run_backtest_service(
            algo_id=algo_id,
            symbol_str=symbol_str,
            timeframe_str=timeframe_str,
            start_str=start_str,
            end_str=end_str,
            capital=capital,
            venue=venue,
            risk_enabled=risk_enabled,
        )
        results.append((algo_id, result))
    return results


# ---------------------------------------------------------------------------
# Hyperparameter Sweep
# ---------------------------------------------------------------------------

MAX_SWEEP_GRID = 200


def expand_param_grid(param_ranges: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand parameter ranges into a list of parameter dicts.

    ``param_ranges`` maps param names to either:
    - ``{"sweep": False, "value": X}`` — fixed value
    - ``{"sweep": True, "min": a, "max": b, "step": s}`` — range to expand
    """
    names: list[str] = []
    value_lists: list[list[Any]] = []

    for name, spec in param_ranges.items():
        names.append(name)
        if not spec.get("sweep", False):
            value_lists.append([spec["value"]])
        else:
            vals = []
            v = spec["min"]
            step = spec.get("step", 1)
            param_type = spec.get("type", "float")
            while v <= spec["max"]:
                vals.append(int(v) if param_type == "int" else round(v, 6))
                v += step
            if not vals:
                vals = [spec["min"]]
            value_lists.append(vals)

    points = []
    for combo in itertools.product(*value_lists):
        points.append(dict(zip(names, combo)))
    return points


async def run_sweep_service(
    *,
    algo_id: str,
    symbol_str: str,
    timeframe_str: str,
    start_str: str,
    end_str: str,
    capital: float = 10_000.0,
    venue: str = "binance_spot",
    risk_enabled: bool = False,
    param_grid: list[dict[str, Any]],
    rank_by: str = "sharpe_ratio",
    on_progress: Callable[[int, int], None] | None = None,
) -> SweepResult:
    """Run backtests across a parameter grid and identify the best config."""
    results: list[Any] = []
    total = len(param_grid)

    for i, params in enumerate(param_grid):
        result = await run_backtest_service(
            algo_id=algo_id,
            symbol_str=symbol_str,
            timeframe_str=timeframe_str,
            start_str=start_str,
            end_str=end_str,
            capital=capital,
            venue=venue,
            risk_enabled=risk_enabled,
            algo_params=params,
        )
        results.append(result)
        if on_progress:
            on_progress(i + 1, total)

    # Find best by rank metric
    best_idx = 0
    best_val = float("-inf")
    for i, r in enumerate(results):
        val = r.kpis.get(rank_by, 0.0)
        # For max_drawdown, less negative is better
        if rank_by == "max_drawdown_pct":
            val = -abs(val)
        if val > best_val:
            best_val = val
            best_idx = i

    swept_names = list(param_grid[0].keys()) if param_grid else []

    return SweepResult(
        algo_id=algo_id,
        param_names=swept_names,
        grid_points=param_grid,
        results=results,
        best_index=best_idx,
        best_params=param_grid[best_idx] if param_grid else {},
        best_kpis=results[best_idx].kpis if results else {},
    )


# ---------------------------------------------------------------------------
# Feature Attribution
# ---------------------------------------------------------------------------


def extract_feature_importance(debug_logs: list[dict[str, Any]]) -> dict[str, float]:
    """Average feature importances from ML algorithm debug logs.

    XGBoost logs feature importance per bar via ctx.log("xgboost_prediction", **importance).
    DL algorithms log up_probability similarly.
    """
    # Known feature importance keys from XGBoost
    importance_keys: set[str] = set()
    counts: dict[str, float] = {}

    for log in debug_logs:
        # XGBoost pattern: message == "xgboost_prediction", keys are feature names
        for key, val in log.items():
            if key in ("message", "bar", "timestamp", "up_probability"):
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                importance_keys.add(key)
                counts[key] = counts.get(key, 0.0) + val

    if not importance_keys:
        return {}

    n_entries = sum(1 for log in debug_logs if any(k in log for k in importance_keys))
    if n_entries == 0:
        return {}

    return {k: round(v / n_entries, 6) for k, v in sorted(counts.items(), key=lambda x: -x[1])}


# ---------------------------------------------------------------------------
# Walk-Forward Stability
# ---------------------------------------------------------------------------


def compute_stability_metrics(wf_result: Any) -> dict[str, float]:
    """Compute stability-specific metrics from a WalkForwardResult."""
    import numpy as np

    sharpes = wf_result.per_fold_oos_sharpes
    if not sharpes:
        return {
            "oos_sharpe": 0.0,
            "sharpe_std": 0.0,
            "degradation_pct": 0.0,
            "consistency_pct": 0.0,
            "worst_fold_sharpe": 0.0,
            "best_fold_sharpe": 0.0,
        }

    oos_sharpes = np.array(sharpes, dtype=float)
    is_sharpes = np.array(
        [f.train_result.kpis.get("sharpe_ratio", 0.0) for f in wf_result.folds],
        dtype=float,
    )

    avg_is = float(np.mean(is_sharpes)) if len(is_sharpes) > 0 else 0.0
    avg_oos = float(np.mean(oos_sharpes))

    degradation = 0.0
    if avg_is != 0:
        degradation = (avg_is - avg_oos) / abs(avg_is) * 100

    consistency = float(np.sum(oos_sharpes > 0) / len(oos_sharpes) * 100)

    return {
        "oos_sharpe": round(wf_result.aggregate_oos_sharpe, 2),
        "sharpe_std": round(float(np.std(oos_sharpes)), 2),
        "degradation_pct": round(degradation, 1),
        "consistency_pct": round(consistency, 1),
        "worst_fold_sharpe": round(float(np.min(oos_sharpes)), 2),
        "best_fold_sharpe": round(float(np.max(oos_sharpes)), 2),
    }
