"""Tests for DAG combinator functions."""

from daytrader.algorithms.dag.combinators import (
    majority_vote,
    threshold_filter,
    unanimous,
    weighted_average,
)
from daytrader.core.types.signals import Signal


def _sig(score: float, confidence: float = 0.8) -> Signal:
    return Signal.new(
        symbol_key="crypto:TEST/USD",
        score=score,
        confidence=confidence,
        source="test",
    )


# ---------------------------------------------------------------------------
# weighted_average
# ---------------------------------------------------------------------------


def test_weighted_average_basic():
    signals = [_sig(0.8), _sig(0.4)]
    result = weighted_average(signals, [1.0, 1.0])
    assert result is not None
    score, _conf = result
    assert abs(score - 0.6) < 0.01  # (0.8 + 0.4) / 2


def test_weighted_average_with_weights():
    signals = [_sig(1.0), _sig(0.0)]
    result = weighted_average(signals, [3.0, 1.0])
    assert result is not None
    score, _ = result
    assert abs(score - 0.75) < 0.01  # 3/4 * 1.0 + 1/4 * 0.0


def test_weighted_average_all_none():
    result = weighted_average([None, None], [1.0, 1.0])
    assert result is None


def test_weighted_average_partial_none():
    signals = [_sig(0.6), None]
    result = weighted_average(signals, [1.0, 1.0])
    assert result is not None
    score, _ = result
    assert abs(score - 0.6) < 0.01


# ---------------------------------------------------------------------------
# majority_vote
# ---------------------------------------------------------------------------


def test_majority_vote_bullish():
    signals = [_sig(0.8), _sig(0.6), _sig(-0.3)]
    result = majority_vote(signals, [1.0, 1.0, 1.0])
    assert result is not None
    score, conf = result
    assert score > 0
    assert conf > 0.5


def test_majority_vote_bearish():
    signals = [_sig(-0.7), _sig(-0.5), _sig(0.2)]
    result = majority_vote(signals, [1.0, 1.0, 1.0])
    assert result is not None
    score, _ = result
    assert score < 0


def test_majority_vote_no_consensus():
    signals = [_sig(0.5), _sig(-0.5)]
    # 50/50 with min_agreement=0.6 should fail
    result = majority_vote(signals, [1.0, 1.0], params={"min_agreement": 0.6})
    assert result is None


def test_majority_vote_all_none():
    result = majority_vote([None, None], [1.0, 1.0])
    assert result is None


# ---------------------------------------------------------------------------
# threshold_filter
# ---------------------------------------------------------------------------


def test_threshold_filter_passes():
    source = _sig(0.7)
    gate = _sig(0.8)  # above threshold
    result = threshold_filter([source, gate], [1.0, 1.0], params={"threshold": 0.5})
    assert result is not None
    score, _ = result
    assert abs(score - 0.7) < 0.01


def test_threshold_filter_blocks():
    source = _sig(0.7)
    gate = _sig(0.3)  # below threshold
    result = threshold_filter([source, gate], [1.0, 1.0], params={"threshold": 0.5})
    assert result is None


def test_threshold_filter_none_gate():
    result = threshold_filter([_sig(0.7), None], [1.0, 1.0], params={"threshold": 0.5})
    assert result is None


# ---------------------------------------------------------------------------
# unanimous
# ---------------------------------------------------------------------------


def test_unanimous_all_bullish():
    signals = [_sig(0.6), _sig(0.8), _sig(0.5)]
    result = unanimous(signals, [1.0, 1.0, 1.0])
    assert result is not None
    score, _ = result
    assert score > 0


def test_unanimous_all_bearish():
    signals = [_sig(-0.6), _sig(-0.8)]
    result = unanimous(signals, [1.0, 1.0])
    assert result is not None
    score, _ = result
    assert score < 0


def test_unanimous_disagreement():
    signals = [_sig(0.6), _sig(-0.3)]
    result = unanimous(signals, [1.0, 1.0])
    assert result is None


def test_unanimous_with_none():
    """None signals are filtered out, remaining must agree."""
    signals = [_sig(0.6), None, _sig(0.8)]
    result = unanimous(signals, [1.0, 1.0, 1.0])
    assert result is not None
    score, _ = result
    assert score > 0
