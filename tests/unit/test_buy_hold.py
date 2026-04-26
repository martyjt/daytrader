"""Tests for the Buy & Hold built-in algorithm."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import numpy as np

from daytrader.algorithms.builtin.buy_hold import BuyHoldAlgorithm
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.bars import Bar, Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _make_ctx() -> tuple[AlgorithmContext, list]:
    emitted: list = []
    return (
        AlgorithmContext(
            tenant_id=uuid4(),
            persona_id=uuid4(),
            algorithm_id="buy_hold",
            symbol=Symbol("BTC", "USD", AssetClass.CRYPTO),
            timeframe=Timeframe.D1,
            now=datetime.now(UTC),
            bar=Bar(
                timestamp=datetime.now(UTC),
                open=Decimal("100"),
                high=Decimal("101"),
                low=Decimal("99"),
                close=Decimal("100"),
                volume=Decimal("1000"),
            ),
            history_arrays={"close": np.array([100.0])},
            features={},
            params={},
            emit_fn=emitted.append,
            log_fn=lambda msg, fields: None,
        ),
        emitted,
    )


def test_manifest():
    algo = BuyHoldAlgorithm()
    m = algo.manifest
    assert m.id == "buy_hold"
    assert m.name == "Buy & Hold"
    assert "crypto" in m.asset_classes
    assert "equities" in m.asset_classes


def test_emits_max_long_signal():
    algo = BuyHoldAlgorithm()
    ctx, emitted = _make_ctx()
    result = algo.on_bar(ctx)

    assert result is not None
    assert result.score == 1.0
    assert result.confidence == 1.0
    assert len(emitted) == 1
    assert emitted[0] is result


def test_warmup_is_zero():
    algo = BuyHoldAlgorithm()
    assert algo.warmup_bars() == 0


def test_signal_has_attribution():
    algo = BuyHoldAlgorithm()
    ctx, _ = _make_ctx()
    sig = algo.on_bar(ctx)

    assert sig.attribution is not None
    assert sig.attribution.node_id == "buy_hold"
    assert sig.attribution.node_type == "algorithm"
