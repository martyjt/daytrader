"""mlflow experiment tracking — optional, never crashes the main flow.

Logs backtest and walk-forward runs to mlflow for experiment comparison,
parameter search, and artifact versioning.

Fails silently if the mlflow server is unreachable or the library is
not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .engine import BacktestResult
    from .walk_forward import WalkForwardResult
    from ..core.gates import GateResult

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """Thin mlflow wrapper for experiment tracking.

    Usage::

        tracker = ExperimentTracker()
        run_id = tracker.log_backtest(
            result=backtest_result,
            algorithm_id="xgboost_trend",
            symbol="BTC-USD",
            timeframe="1d",
            venue="binance_spot",
        )
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        experiment_name: str = "daytrader",
    ) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name

    def _resolve_uri(self) -> str:
        if self._tracking_uri:
            return self._tracking_uri
        try:
            from ..core.settings import get_settings
            return get_settings().mlflow_tracking_uri
        except Exception:
            return "http://localhost:5000"

    def log_backtest(
        self,
        *,
        result: BacktestResult,
        algorithm_id: str,
        symbol: str,
        timeframe: str,
        venue: str,
        gate_result: GateResult | None = None,
    ) -> str | None:
        """Log a backtest run. Returns the mlflow run_id or None on failure."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self._resolve_uri())
            mlflow.set_experiment(self._experiment_name)

            with mlflow.start_run(run_name=f"backtest_{algorithm_id}_{symbol}"):
                mlflow.log_params({
                    "algorithm": algorithm_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "venue": venue,
                    "initial_capital": result.initial_capital,
                })

                mlflow.log_metrics({
                    "sharpe_ratio": result.kpis.get("sharpe_ratio", 0),
                    "total_return_pct": result.kpis.get("total_return_pct", 0),
                    "net_return_pct": result.kpis.get("net_return_pct", 0),
                    "max_drawdown_pct": result.kpis.get("max_drawdown_pct", 0),
                    "num_trades": result.kpis.get("num_trades", 0),
                    "win_rate_pct": result.kpis.get("win_rate_pct", 0),
                    "fee_drag_pct": result.kpis.get("fee_drag_pct", 0),
                    "final_equity": result.final_equity,
                    "total_fees_paid": result.total_fees_paid,
                })

                if gate_result:
                    mlflow.log_params({
                        "gate_passed": str(gate_result.overall_pass),
                        "gate_stage": gate_result.stage,
                    })

                return mlflow.active_run().info.run_id

        except Exception as exc:
            logger.debug("mlflow logging failed: %s", exc)
            return None

    def log_walk_forward(
        self,
        *,
        result: WalkForwardResult,
        algorithm_id: str,
        symbol: str,
        timeframe: str,
        venue: str,
        gate_result: GateResult | None = None,
    ) -> str | None:
        """Log a walk-forward run with per-fold metrics."""
        try:
            import mlflow

            mlflow.set_tracking_uri(self._resolve_uri())
            mlflow.set_experiment(self._experiment_name)

            with mlflow.start_run(run_name=f"walkforward_{algorithm_id}_{symbol}"):
                mlflow.log_params({
                    "algorithm": algorithm_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "venue": venue,
                    "n_folds": result.config.n_folds,
                    "anchored": str(result.config.anchored),
                })

                mlflow.log_metrics({
                    "oos_sharpe": result.aggregate_oos_sharpe,
                    "oos_return_pct": result.aggregate_oos_return_pct,
                    "oos_max_drawdown_pct": result.aggregate_oos_max_drawdown_pct,
                    "total_bars": result.total_bars,
                })

                for fold in result.folds:
                    mlflow.log_metric(
                        f"fold_{fold.fold_index}_oos_sharpe",
                        fold.oos_sharpe,
                    )
                    mlflow.log_metric(
                        f"fold_{fold.fold_index}_oos_return_pct",
                        fold.oos_return_pct,
                    )

                if gate_result:
                    mlflow.log_params({
                        "gate_passed": str(gate_result.overall_pass),
                        "gate_stage": gate_result.stage,
                    })

                return mlflow.active_run().info.run_id

        except Exception as exc:
            logger.debug("mlflow logging failed: %s", exc)
            return None
