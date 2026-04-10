"""Tests for mlflow experiment tracking (mocked — no real server)."""

from unittest.mock import MagicMock, patch

from daytrader.backtest.tracking import ExperimentTracker


def _mock_backtest_result():
    result = MagicMock()
    result.kpis = {
        "sharpe_ratio": 0.8,
        "total_return_pct": 15.5,
        "net_return_pct": 14.2,
        "max_drawdown_pct": -5.3,
        "num_trades": 42,
        "win_rate_pct": 55.0,
        "fee_drag_pct": 1.3,
    }
    result.initial_capital = 10000.0
    result.final_equity = 11550.0
    result.total_fees_paid = 130.0
    return result


def _mock_wf_result():
    result = MagicMock()
    result.aggregate_oos_sharpe = 0.45
    result.aggregate_oos_return_pct = 8.2
    result.aggregate_oos_max_drawdown_pct = -4.1
    result.total_bars = 500
    result.config.n_folds = 5
    result.config.anchored = True

    fold1 = MagicMock()
    fold1.fold_index = 0
    fold1.oos_sharpe = 0.5
    fold1.oos_return_pct = 3.2

    fold2 = MagicMock()
    fold2.fold_index = 1
    fold2.oos_sharpe = 0.4
    fold2.oos_return_pct = 5.0

    result.folds = [fold1, fold2]
    return result


def _mock_gate_result():
    gate = MagicMock()
    gate.overall_pass = True
    gate.stage = "backtest"
    return gate


@patch("daytrader.backtest.tracking.mlflow", create=True)
def test_log_backtest_calls_mlflow(mock_mlflow):
    """log_backtest should call mlflow with correct params and metrics."""
    # Patch the import inside the method
    import sys
    mock_module = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-123"
    mock_module.active_run.return_value = mock_run
    mock_module.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_module.start_run.return_value.__exit__ = MagicMock(return_value=False)

    sys.modules["mlflow"] = mock_module
    try:
        tracker = ExperimentTracker(tracking_uri="http://test:5000")
        run_id = tracker.log_backtest(
            result=_mock_backtest_result(),
            algorithm_id="xgboost_trend",
            symbol="BTC-USD",
            timeframe="1d",
            venue="binance_spot",
        )

        mock_module.set_tracking_uri.assert_called_once_with("http://test:5000")
        mock_module.set_experiment.assert_called_once_with("daytrader")
        mock_module.log_params.assert_called()
        mock_module.log_metrics.assert_called()

        # Verify params contain expected keys
        params_call = mock_module.log_params.call_args[0][0]
        assert params_call["algorithm"] == "xgboost_trend"
        assert params_call["symbol"] == "BTC-USD"

        # Verify metrics contain expected keys
        metrics_call = mock_module.log_metrics.call_args[0][0]
        assert metrics_call["sharpe_ratio"] == 0.8
        assert metrics_call["final_equity"] == 11550.0

        assert run_id == "test-run-123"
    finally:
        sys.modules.pop("mlflow", None)


@patch("daytrader.backtest.tracking.mlflow", create=True)
def test_log_walk_forward_logs_per_fold(mock_mlflow):
    """log_walk_forward should log per-fold OOS metrics."""
    import sys
    mock_module = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "wf-run-456"
    mock_module.active_run.return_value = mock_run
    mock_module.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_module.start_run.return_value.__exit__ = MagicMock(return_value=False)

    sys.modules["mlflow"] = mock_module
    try:
        tracker = ExperimentTracker(tracking_uri="http://test:5000")
        run_id = tracker.log_walk_forward(
            result=_mock_wf_result(),
            algorithm_id="xgboost_trend",
            symbol="BTC-USD",
            timeframe="1d",
            venue="binance_spot",
        )

        # Should log per-fold metrics
        log_metric_calls = mock_module.log_metric.call_args_list
        fold_keys = [call[0][0] for call in log_metric_calls]
        assert "fold_0_oos_sharpe" in fold_keys
        assert "fold_1_oos_sharpe" in fold_keys

        assert run_id == "wf-run-456"
    finally:
        sys.modules.pop("mlflow", None)


def test_log_backtest_returns_none_on_failure():
    """log_backtest should return None when mlflow is unavailable."""
    import sys
    # Remove mlflow from sys.modules to force ImportError
    original = sys.modules.pop("mlflow", None)
    try:
        tracker = ExperimentTracker(tracking_uri="http://nonexistent:5000")
        run_id = tracker.log_backtest(
            result=_mock_backtest_result(),
            algorithm_id="test",
            symbol="TEST-USD",
            timeframe="1d",
            venue="custom",
        )
        assert run_id is None
    finally:
        if original is not None:
            sys.modules["mlflow"] = original


def test_log_backtest_with_gate_result():
    """Gate result should be logged as params when provided."""
    import sys
    mock_module = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "gate-run-789"
    mock_module.active_run.return_value = mock_run
    mock_module.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
    mock_module.start_run.return_value.__exit__ = MagicMock(return_value=False)

    sys.modules["mlflow"] = mock_module
    try:
        tracker = ExperimentTracker(tracking_uri="http://test:5000")
        tracker.log_backtest(
            result=_mock_backtest_result(),
            algorithm_id="test",
            symbol="TEST-USD",
            timeframe="1d",
            venue="custom",
            gate_result=_mock_gate_result(),
        )

        # Second log_params call should include gate info
        all_params_calls = mock_module.log_params.call_args_list
        assert len(all_params_calls) == 2  # main params + gate params
        gate_params = all_params_calls[1][0][0]
        assert gate_params["gate_passed"] == "True"
        assert gate_params["gate_stage"] == "backtest"
    finally:
        sys.modules.pop("mlflow", None)
