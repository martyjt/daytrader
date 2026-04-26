"""Tests for promotion gates."""

from daytrader.core.gates import GateEvaluator


def _mock_backtest_result(sharpe: float = 0.6, num_trades: int = 40):
    """Create a mock BacktestResult-like object with kpis dict."""

    class _MockResult:
        def __init__(self, kpis):
            self.kpis = kpis

    return _MockResult(kpis={"sharpe_ratio": sharpe, "num_trades": num_trades})


def _mock_wf_result(oos_sharpe: float = 0.4):
    """Create a mock WalkForwardResult-like object."""

    class _MockResult:
        def __init__(self, aggregate_oos_sharpe):
            self.aggregate_oos_sharpe = aggregate_oos_sharpe

    return _MockResult(aggregate_oos_sharpe=oos_sharpe)


# ---------------------------------------------------------------------------
# Backtest gates
# ---------------------------------------------------------------------------


def test_backtest_gates_pass():
    evaluator = GateEvaluator(gate_config={
        "backtest": {"min_trades": 30, "min_sharpe": 0.5},
    })
    result = evaluator.evaluate_backtest(_mock_backtest_result(sharpe=0.6, num_trades=40))
    assert result.overall_pass
    assert result.stage == "backtest"
    assert len(result.checks) == 2
    assert all(c.passed for c in result.checks)


def test_backtest_sharpe_fails():
    evaluator = GateEvaluator(gate_config={
        "backtest": {"min_trades": 30, "min_sharpe": 0.5},
    })
    result = evaluator.evaluate_backtest(_mock_backtest_result(sharpe=0.3, num_trades=40))
    assert not result.overall_pass
    assert len(result.failed_checks) == 1
    assert result.failed_checks[0].gate_name == "backtest_min_sharpe"


def test_backtest_trades_fails():
    evaluator = GateEvaluator(gate_config={
        "backtest": {"min_trades": 30, "min_sharpe": 0.5},
    })
    result = evaluator.evaluate_backtest(_mock_backtest_result(sharpe=0.6, num_trades=10))
    assert not result.overall_pass
    assert len(result.failed_checks) == 1
    assert result.failed_checks[0].gate_name == "backtest_min_trades"


def test_backtest_both_fail():
    evaluator = GateEvaluator(gate_config={
        "backtest": {"min_trades": 30, "min_sharpe": 0.5},
    })
    result = evaluator.evaluate_backtest(_mock_backtest_result(sharpe=0.2, num_trades=5))
    assert not result.overall_pass
    assert len(result.failed_checks) == 2


# ---------------------------------------------------------------------------
# Walk-forward gates
# ---------------------------------------------------------------------------


def test_walk_forward_gate_pass():
    evaluator = GateEvaluator(gate_config={
        "walk_forward": {"min_oos_sharpe": 0.3},
    })
    result = evaluator.evaluate_walk_forward(_mock_wf_result(oos_sharpe=0.4))
    assert result.overall_pass
    assert result.stage == "walk_forward"


def test_walk_forward_gate_fail():
    evaluator = GateEvaluator(gate_config={
        "walk_forward": {"min_oos_sharpe": 0.3},
    })
    result = evaluator.evaluate_walk_forward(_mock_wf_result(oos_sharpe=0.1))
    assert not result.overall_pass
    assert len(result.failed_checks) == 1


# ---------------------------------------------------------------------------
# Paper gates
# ---------------------------------------------------------------------------


def test_paper_gates_pass():
    evaluator = GateEvaluator(gate_config={
        "paper": {"min_days": 7, "min_trades": 10},
    })
    result = evaluator.evaluate_paper(days_active=8, num_trades=15)
    assert result.overall_pass
    assert result.stage == "paper"


def test_paper_days_fail():
    evaluator = GateEvaluator(gate_config={
        "paper": {"min_days": 7, "min_trades": 10},
    })
    result = evaluator.evaluate_paper(days_active=5, num_trades=15)
    assert not result.overall_pass
    assert result.failed_checks[0].gate_name == "paper_min_days"


def test_paper_trades_fail():
    evaluator = GateEvaluator(gate_config={
        "paper": {"min_days": 7, "min_trades": 10},
    })
    result = evaluator.evaluate_paper(days_active=10, num_trades=3)
    assert not result.overall_pass
    assert result.failed_checks[0].gate_name == "paper_min_trades"


# ---------------------------------------------------------------------------
# GateResult properties
# ---------------------------------------------------------------------------


def test_failed_checks_empty_when_all_pass():
    evaluator = GateEvaluator(gate_config={
        "backtest": {"min_trades": 1, "min_sharpe": 0.0},
    })
    result = evaluator.evaluate_backtest(_mock_backtest_result(sharpe=0.5, num_trades=10))
    assert result.failed_checks == []


def test_custom_thresholds():
    """Custom thresholds from config should be used."""
    evaluator = GateEvaluator(gate_config={
        "backtest": {"min_trades": 100, "min_sharpe": 2.0},
    })
    result = evaluator.evaluate_backtest(_mock_backtest_result(sharpe=1.5, num_trades=50))
    assert not result.overall_pass
    assert len(result.failed_checks) == 2


def test_default_config():
    """Evaluator with no config should use defaults."""
    evaluator = GateEvaluator(gate_config={})
    result = evaluator.evaluate_backtest(_mock_backtest_result(sharpe=0.6, num_trades=40))
    assert result.overall_pass
