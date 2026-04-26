"""Promotion gates — the Ritual's quantitative checkpoints.

Before a strategy can be promoted from backtest → walk-forward → paper
→ live, it must pass the gates defined in ``config/default.yaml``
under ``ritual.gates``.

Each gate produces a ``GateResult`` with per-check pass/fail detail
and an overall verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..backtest.engine import BacktestResult
    from ..backtest.walk_forward import WalkForwardResult


@dataclass(frozen=True)
class GateCheck:
    """Result of one gate check."""

    gate_name: str
    metric_name: str
    required_value: float
    actual_value: float
    passed: bool
    description: str = ""


@dataclass(frozen=True)
class GateResult:
    """Aggregate gate evaluation result."""

    stage: str
    checks: tuple[GateCheck, ...]
    overall_pass: bool
    timestamp: datetime

    @property
    def failed_checks(self) -> list[GateCheck]:
        return [c for c in self.checks if not c.passed]


def _utcnow() -> datetime:
    return datetime.now(UTC)


class GateEvaluator:
    """Evaluate promotion gates from config.

    Usage::

        evaluator = GateEvaluator()
        result = evaluator.evaluate_backtest(backtest_result)
        if result.overall_pass:
            print("Backtest gates passed!")
    """

    def __init__(self, gate_config: dict[str, Any] | None = None) -> None:
        if gate_config is not None:
            self._config = gate_config
        else:
            try:
                from .settings import get_yaml_config

                cfg = get_yaml_config()
                self._config = cfg.get("ritual", "gates", default={}) or {}
            except Exception:
                self._config = {}

    def evaluate_backtest(self, result: BacktestResult) -> GateResult:
        """Check backtest gates: min_trades and min_sharpe."""
        bt_config = self._config.get("backtest", {})
        min_trades = bt_config.get("min_trades", 30)
        min_sharpe = bt_config.get("min_sharpe", 0.5)

        actual_trades = result.kpis.get("num_trades", 0)
        actual_sharpe = result.kpis.get("sharpe_ratio", 0.0)

        checks = [
            GateCheck(
                gate_name="backtest_min_trades",
                metric_name="num_trades",
                required_value=min_trades,
                actual_value=actual_trades,
                passed=actual_trades >= min_trades,
                description=f"Backtest must have at least {min_trades} trades",
            ),
            GateCheck(
                gate_name="backtest_min_sharpe",
                metric_name="sharpe_ratio",
                required_value=min_sharpe,
                actual_value=actual_sharpe,
                passed=actual_sharpe >= min_sharpe,
                description=f"Backtest Sharpe ratio must be >= {min_sharpe}",
            ),
        ]

        return GateResult(
            stage="backtest",
            checks=tuple(checks),
            overall_pass=all(c.passed for c in checks),
            timestamp=_utcnow(),
        )

    def evaluate_walk_forward(self, result: WalkForwardResult) -> GateResult:
        """Check walk-forward gates: min_oos_sharpe."""
        wf_config = self._config.get("walk_forward", {})
        min_oos_sharpe = wf_config.get("min_oos_sharpe", 0.3)

        actual_sharpe = result.aggregate_oos_sharpe

        checks = [
            GateCheck(
                gate_name="walk_forward_min_oos_sharpe",
                metric_name="aggregate_oos_sharpe",
                required_value=min_oos_sharpe,
                actual_value=actual_sharpe,
                passed=actual_sharpe >= min_oos_sharpe,
                description=f"Walk-forward OOS Sharpe must be >= {min_oos_sharpe}",
            ),
        ]

        return GateResult(
            stage="walk_forward",
            checks=tuple(checks),
            overall_pass=all(c.passed for c in checks),
            timestamp=_utcnow(),
        )

    def evaluate_paper(self, days_active: int, num_trades: int) -> GateResult:
        """Check paper trading gates: min_days and min_trades."""
        paper_config = self._config.get("paper", {})
        min_days = paper_config.get("min_days", 7)
        min_trades = paper_config.get("min_trades", 10)

        checks = [
            GateCheck(
                gate_name="paper_min_days",
                metric_name="days_active",
                required_value=min_days,
                actual_value=days_active,
                passed=days_active >= min_days,
                description=f"Paper trading must run for at least {min_days} days",
            ),
            GateCheck(
                gate_name="paper_min_trades",
                metric_name="num_trades",
                required_value=min_trades,
                actual_value=num_trades,
                passed=num_trades >= min_trades,
                description=f"Paper trading must have at least {min_trades} trades",
            ),
        ]

        return GateResult(
            stage="paper",
            checks=tuple(checks),
            overall_pass=all(c.passed for c in checks),
            timestamp=_utcnow(),
        )
