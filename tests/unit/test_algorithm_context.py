from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import numpy as np
import pytest

from daytrader.core.context import AlgorithmContext
from daytrader.core.types.bars import Bar, Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _make_context() -> tuple[AlgorithmContext, list]:
    emitted: list = []
    logs: list = []

    bar = Bar(
        timestamp=datetime.now(timezone.utc),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100.5"),
        volume=Decimal("1000"),
    )
    ctx = AlgorithmContext(
        tenant_id=uuid4(),
        persona_id=uuid4(),
        algorithm_id="test_algo",
        symbol=Symbol("BTC", "USDT", AssetClass.CRYPTO, "binance"),
        timeframe=Timeframe.H1,
        now=datetime.now(timezone.utc),
        bar=bar,
        history_arrays={"close": np.array([99.0, 100.0, 100.5])},
        features={"rsi_14": 55.3, "ema_fast": 100.2},
        params={"period": 14},
        emit_fn=emitted.append,
        log_fn=lambda msg, fields: logs.append((msg, fields)),
    )
    return ctx, emitted


def test_history_access():
    ctx, _ = _make_context()
    h = ctx.history(2, "close")
    assert len(h) == 2
    assert h[-1] == 100.5


def test_history_exceeds_available():
    ctx, _ = _make_context()
    with pytest.raises(ValueError):
        ctx.history(10, "close")


def test_history_unknown_field():
    ctx, _ = _make_context()
    with pytest.raises(KeyError):
        ctx.history(1, "volume")


def test_feature_access():
    ctx, _ = _make_context()
    assert ctx.feature("rsi_14") == 55.3


def test_feature_missing_raises():
    ctx, _ = _make_context()
    with pytest.raises(KeyError):
        ctx.feature("unknown")


def test_feature_missing_with_default():
    ctx, _ = _make_context()
    assert ctx.feature("unknown", default=0.0) == 0.0


def test_has_feature():
    ctx, _ = _make_context()
    assert ctx.has_feature("rsi_14")
    assert not ctx.has_feature("nope")


def test_param_access():
    ctx, _ = _make_context()
    assert ctx.param("period") == 14


def test_param_missing_raises():
    ctx, _ = _make_context()
    with pytest.raises(KeyError):
        ctx.param("missing")


def test_param_missing_with_default():
    ctx, _ = _make_context()
    assert ctx.param("missing", default=42) == 42


def test_emit_signal():
    ctx, emitted = _make_context()
    sig = ctx.emit(score=0.8, confidence=0.9, reason="bullish cross")
    assert len(emitted) == 1
    assert emitted[0] is sig
    assert sig.score == 0.8
    assert sig.confidence == 0.9
    assert sig.source == "test_algo"
    assert sig.attribution is not None
    assert sig.attribution.node_id == "test_algo"
    assert sig.attribution.node_type == "algorithm"
    assert sig.attribution.features_snapshot["rsi_14"] == 55.3


def test_emit_score_out_of_bounds():
    ctx, _ = _make_context()
    with pytest.raises(ValueError):
        ctx.emit(score=1.5)
