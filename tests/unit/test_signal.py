import pytest

from daytrader.core.types.signals import Signal, SignalContribution


def test_signal_valid():
    s = Signal.new(
        symbol_key="crypto:BTC/USDT",
        score=0.7,
        confidence=0.9,
        source="ema_cross",
        reason="fast crossed above slow",
    )
    assert s.score == 0.7
    assert s.confidence == 0.9
    assert s.source == "ema_cross"


def test_signal_score_upper_bound():
    with pytest.raises(ValueError):
        Signal.new(symbol_key="x", score=1.5, source="test")


def test_signal_score_lower_bound():
    with pytest.raises(ValueError):
        Signal.new(symbol_key="x", score=-2.0, source="test")


def test_signal_confidence_bounds():
    with pytest.raises(ValueError):
        Signal.new(symbol_key="x", score=0.5, confidence=1.2, source="test")


def test_signal_with_attribution():
    attr = SignalContribution(
        node_id="rsi_14",
        node_type="algorithm",
        score=0.6,
        confidence=0.8,
        reason="oversold",
        features_snapshot={"rsi_14": 28.3},
    )
    s = Signal.new(
        symbol_key="crypto:BTC/USDT",
        score=0.6,
        source="rsi_14",
        attribution=attr,
    )
    assert s.attribution is attr
    assert s.attribution.features_snapshot["rsi_14"] == 28.3
