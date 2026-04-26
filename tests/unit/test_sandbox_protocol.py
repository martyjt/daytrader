"""Codec round-trip + framing tests for the plugin-sandbox protocol.

The protocol is the trust boundary between the parent process and a
worker subprocess — these tests are defensive: every dataclass we ship
across the pipe must reconstruct bit-identically, and every malformed
frame must raise ``ProtocolError`` instead of returning garbage.
"""

from __future__ import annotations

import io
import json
import struct
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import numpy as np
import pytest

from daytrader.algorithms.sandbox import protocol
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.bars import Bar, Timeframe
from daytrader.core.types.signals import Signal, SignalContribution
from daytrader.core.types.symbols import AssetClass, Symbol

# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def test_round_trip_one_frame():
    buf = io.BytesIO()
    protocol.write_frame(buf, {"hello": "world", "n": 42})
    buf.seek(0)
    obj = protocol.read_frame(buf)
    assert obj == {"hello": "world", "n": 42}


def test_clean_eof_returns_none():
    obj = protocol.read_frame(io.BytesIO())
    assert obj is None


def test_truncated_header_raises():
    buf = io.BytesIO(b"\x00\x00")  # only 2 of 4 bytes
    with pytest.raises(protocol.ProtocolError):
        protocol.read_frame(buf)


def test_truncated_body_raises():
    body_len = struct.pack(">I", 100)
    buf = io.BytesIO(body_len + b"short")
    with pytest.raises(protocol.ProtocolError):
        protocol.read_frame(buf)


def test_zero_length_frame_rejected():
    buf = io.BytesIO(struct.pack(">I", 0))
    with pytest.raises(protocol.ProtocolError):
        protocol.read_frame(buf)


def test_oversized_frame_rejected():
    buf = io.BytesIO(struct.pack(">I", protocol.MAX_FRAME_SIZE + 1))
    with pytest.raises(protocol.ProtocolError):
        protocol.read_frame(buf)


def test_non_object_top_level_rejected():
    payload = json.dumps([1, 2, 3]).encode("utf-8")
    buf = io.BytesIO(struct.pack(">I", len(payload)) + payload)
    with pytest.raises(protocol.ProtocolError):
        protocol.read_frame(buf)


def test_invalid_utf8_rejected():
    payload = b"\xff\xfe\xfd"
    buf = io.BytesIO(struct.pack(">I", len(payload)) + payload)
    with pytest.raises(protocol.ProtocolError):
        protocol.read_frame(buf)


# ---------------------------------------------------------------------------
# Numpy arrays
# ---------------------------------------------------------------------------


def test_array_round_trip_float64():
    arr = np.array([1.5, 2.5, 3.5], dtype="float64")
    encoded = protocol._encode_array(arr)
    decoded = protocol._decode_array(encoded)
    assert decoded.dtype == arr.dtype
    assert decoded.shape == arr.shape
    np.testing.assert_array_equal(decoded, arr)


def test_array_round_trip_2d():
    arr = np.arange(12, dtype="int32").reshape(3, 4)
    decoded = protocol._decode_array(protocol._encode_array(arr))
    assert decoded.shape == (3, 4)
    np.testing.assert_array_equal(decoded, arr)


def test_array_object_dtype_rejected():
    arr = np.array([1, "two", 3.0], dtype=object)
    with pytest.raises(protocol.ProtocolError):
        protocol._encode_array(arr)


def test_array_decode_bad_dtype_rejected():
    payload = {"dtype": "object", "shape": [3], "data": ""}
    with pytest.raises(protocol.ProtocolError):
        protocol._decode_array(payload)


def test_array_decode_size_mismatch_rejected():
    payload = protocol._encode_array(np.arange(6, dtype="int32"))
    payload["shape"] = [4]  # claim 4 elems but bytes encode 6
    with pytest.raises(protocol.ProtocolError):
        protocol._decode_array(payload)


# ---------------------------------------------------------------------------
# AlgorithmContext round-trip
# ---------------------------------------------------------------------------


def _build_ctx() -> AlgorithmContext:
    closes = np.array([100.0, 101.0, 102.0], dtype="float64")
    return AlgorithmContext(
        tenant_id=uuid4(),
        persona_id=uuid4(),
        algorithm_id="my_algo",
        symbol=Symbol("BTC", "USDT", AssetClass.CRYPTO, "binance"),
        timeframe=Timeframe.D1,
        now=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
        bar=Bar(
            timestamp=datetime(2026, 4, 25, tzinfo=UTC),
            open=Decimal("100.5"),
            high=Decimal("103.5"),
            low=Decimal("99.5"),
            close=Decimal("102.5"),
            volume=Decimal("1234.5"),
        ),
        history_arrays={"close": closes, "volume": closes},
        features={"rsi_14": 0.42, "macd": -0.05},
        params={"window": 20, "threshold": 0.3, "long_only": True, "label": "x"},
        emit_fn=lambda s: None,
        log_fn=lambda m, f: None,
    )


def test_context_round_trip_preserves_all_fields():
    ctx = _build_ctx()
    captured: list[Signal] = []
    payload = protocol.serialize_context(ctx)
    rebuilt = protocol.deserialize_context(
        payload, emit_fn=captured.append, log_fn=lambda m, f: None,
    )
    assert rebuilt.tenant_id == ctx.tenant_id
    assert rebuilt.persona_id == ctx.persona_id
    assert rebuilt.algorithm_id == ctx.algorithm_id
    assert rebuilt.symbol == ctx.symbol
    assert rebuilt.timeframe == ctx.timeframe
    assert rebuilt.now == ctx.now
    assert rebuilt.bar.open == ctx.bar.open  # Decimals via str
    assert rebuilt.bar.close == ctx.bar.close
    np.testing.assert_array_equal(
        rebuilt.history_arrays["close"], ctx.history_arrays["close"]
    )
    assert rebuilt.features == ctx.features
    assert rebuilt.params == ctx.params


def test_context_with_no_features_or_params():
    ctx = AlgorithmContext(
        tenant_id=uuid4(), persona_id=uuid4(), algorithm_id="x",
        symbol=Symbol("AAPL", "USD", AssetClass.EQUITIES),
        timeframe=Timeframe.M5,
        now=datetime(2026, 1, 1, tzinfo=UTC),
        bar=Bar(
            timestamp=datetime(2026, 1, 1, tzinfo=UTC),
            open=Decimal("100"), high=Decimal("100"), low=Decimal("100"),
            close=Decimal("100"), volume=Decimal("0"),
        ),
        history_arrays={},
        features={},
        params={},
        emit_fn=lambda s: None, log_fn=lambda m, f: None,
    )
    payload = protocol.serialize_context(ctx)
    rebuilt = protocol.deserialize_context(
        payload, emit_fn=lambda s: None, log_fn=lambda m, f: None,
    )
    assert rebuilt.features == {}
    assert rebuilt.params == {}


def test_context_with_unsupported_param_type_rejected():
    ctx = _build_ctx()
    ctx.params["bad"] = object()
    with pytest.raises(protocol.ProtocolError):
        protocol.serialize_context(ctx)


# ---------------------------------------------------------------------------
# Signal + SignalContribution round-trip
# ---------------------------------------------------------------------------


def test_signal_round_trip():
    sig = Signal.new(
        symbol_key="BTC-USDT",
        score=0.75,
        confidence=0.9,
        source="my_algo",
        reason="trend up",
        metadata={"window": 20, "tag": "x", "items": [1, 2, 3]},
    )
    payload = protocol.serialize_signal(sig)
    rebuilt = protocol.deserialize_signal(payload)
    assert rebuilt.id == sig.id
    assert rebuilt.score == sig.score
    assert rebuilt.confidence == sig.confidence
    assert rebuilt.source == sig.source
    assert rebuilt.reason == sig.reason
    assert rebuilt.metadata == sig.metadata


def test_signal_with_attribution_tree():
    inner = SignalContribution(
        node_id="leaf",
        node_type="algorithm",
        score=0.5,
        confidence=0.8,
        weight=0.3,
        reason="leaf",
        features_snapshot={"rsi": 0.4},
    )
    outer = SignalContribution(
        node_id="root",
        node_type="combinator",
        score=0.5,
        confidence=0.8,
        weight=None,
        reason="vote",
        children=(inner,),
    )
    sig = Signal.new(
        symbol_key="BTC-USDT", score=0.5, confidence=0.8,
        source="dag:test", attribution=outer,
    )
    payload = protocol.serialize_signal(sig)
    rebuilt = protocol.deserialize_signal(payload)
    assert rebuilt.attribution is not None
    assert rebuilt.attribution.node_id == "root"
    assert len(rebuilt.attribution.children) == 1
    assert rebuilt.attribution.children[0].node_id == "leaf"
    assert rebuilt.attribution.children[0].weight == 0.3


def test_signal_invalid_score_raises_via_dataclass_check():
    payload = {
        "id": str(uuid4()),
        "timestamp": "2026-04-25T00:00:00+00:00",
        "symbol_key": "X",
        "score": 1.5,  # out of range; Signal.__post_init__ rejects
        "confidence": 0.5,
        "source": "x",
        "reason": "",
        "attribution": None,
        "metadata": {},
    }
    with pytest.raises(ValueError):
        protocol.deserialize_signal(payload)


def test_signal_metadata_with_nested_dict_round_trip():
    sig = Signal.new(
        symbol_key="X", score=0.0, confidence=1.0, source="s",
        metadata={"k": {"a": 1, "b": "hi"}, "n": 42},
    )
    rebuilt = protocol.deserialize_signal(protocol.serialize_signal(sig))
    assert rebuilt.metadata == sig.metadata
