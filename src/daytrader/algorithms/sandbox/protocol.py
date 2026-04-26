"""Length-prefixed JSON IPC protocol for plugin workers.

Frames are a 4-byte big-endian unsigned length prefix followed by a JSON
UTF-8 payload. Numpy arrays travel as ``{"__nd__": True, "dtype": "...",
"shape": [...], "data": "<base64>"}`` so we keep dtype/shape exactly.

The parent process never trusts a worker's payload as code. Deserialization
goes through strict factories that validate types and ranges before
reconstructing dataclasses. No pickle, no eval, no reflection — that's
the whole point of doing this in stdlib JSON instead of pickle.
"""

from __future__ import annotations

import base64
import io
import json
import struct
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import IO, Any
from uuid import UUID

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.bars import Bar, Timeframe
from ...core.types.signals import Signal, SignalContribution
from ...core.types.symbols import AssetClass, Symbol

# 16 MiB. A 200-bar warmup of 5 OHLCV float64 arrays is ~8 KiB; even 1m bars
# with a year of lookback fits well under this. A frame larger than this is
# almost certainly corrupt or hostile — we refuse it rather than allocate.
MAX_FRAME_SIZE = 16 * 1024 * 1024


class ProtocolError(Exception):
    """Raised when a frame violates the protocol (bad shape, bad type, oversize)."""


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def _read_exactly(stream: IO[bytes], n: int) -> bytes | None:
    """Read exactly ``n`` bytes or return None on clean EOF.

    Returns None only when zero bytes were available (clean stream close).
    A short read after some bytes were consumed is a protocol violation.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if not buf:
                return None
            raise ProtocolError(
                f"Truncated frame: wanted {n} bytes, got {len(buf)} before EOF"
            )
        buf.extend(chunk)
    return bytes(buf)


def read_frame(stream: IO[bytes]) -> dict[str, Any] | None:
    """Read one frame from ``stream``. Returns None on clean EOF."""
    header = _read_exactly(stream, 4)
    if header is None:
        return None
    (length,) = struct.unpack(">I", header)
    if length == 0:
        raise ProtocolError("Zero-length frame")
    if length > MAX_FRAME_SIZE:
        raise ProtocolError(f"Frame too large: {length} > {MAX_FRAME_SIZE}")
    body = _read_exactly(stream, length)
    if body is None or len(body) != length:
        raise ProtocolError(f"Truncated frame body: wanted {length}")
    try:
        obj = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"Frame is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError(f"Frame must be a JSON object, got {type(obj).__name__}")
    return obj


def write_frame(stream: IO[bytes], obj: dict[str, Any]) -> None:
    """Write one frame to ``stream`` and flush."""
    body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_FRAME_SIZE:
        raise ProtocolError(f"Frame too large to send: {len(body)} > {MAX_FRAME_SIZE}")
    stream.write(struct.pack(">I", len(body)))
    stream.write(body)
    stream.flush()


# ---------------------------------------------------------------------------
# Numpy arrays
# ---------------------------------------------------------------------------

# Allowlist of dtypes a worker is allowed to return. Reject object/void/string —
# anything that could carry pickle bytes.
_ALLOWED_DTYPES = frozenset(
    [
        "bool",
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
    ]
)


def _encode_array(arr: np.ndarray) -> dict[str, Any]:
    if arr.dtype.name not in _ALLOWED_DTYPES:
        # Cast plain Python-numeric history to float64. Anything else (object,
        # bytes, datetime64) we reject — it shouldn't appear in history_arrays
        # which is documented as numeric OHLCV.
        if arr.dtype.kind in "iuf":
            arr = arr.astype("float64")
        else:
            raise ProtocolError(f"Cannot serialize array of dtype {arr.dtype.name!r}")
    return {
        "__nd__": True,
        "dtype": arr.dtype.name,
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def _decode_array(payload: dict[str, Any]) -> np.ndarray:
    dtype = payload.get("dtype")
    shape = payload.get("shape")
    data = payload.get("data")
    if not isinstance(dtype, str) or dtype not in _ALLOWED_DTYPES:
        raise ProtocolError(f"Invalid array dtype {dtype!r}")
    if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
        raise ProtocolError(f"Invalid array shape {shape!r}")
    if not isinstance(data, str):
        raise ProtocolError("Array data must be a base64 string")
    raw = base64.b64decode(data, validate=True)
    arr = np.frombuffer(raw, dtype=np.dtype(dtype))
    expected = 1
    for d in shape:
        expected *= d
    if arr.size != expected:
        raise ProtocolError(
            f"Array size mismatch: bytes give {arr.size} elems, shape implies {expected}"
        )
    # ``frombuffer`` returns a read-only view; reshape may fail on 0-d.
    return arr.reshape(shape) if shape else arr.copy()


# ---------------------------------------------------------------------------
# Primitive helpers (strict)
# ---------------------------------------------------------------------------


def _as_str(v: Any, field: str) -> str:
    if not isinstance(v, str):
        raise ProtocolError(f"{field} must be string, got {type(v).__name__}")
    return v


def _as_float(v: Any, field: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ProtocolError(f"{field} must be number, got {type(v).__name__}")
    return float(v)


def _as_dict(v: Any, field: str) -> dict[str, Any]:
    if not isinstance(v, dict):
        raise ProtocolError(f"{field} must be object, got {type(v).__name__}")
    return v


def _as_list(v: Any, field: str) -> list[Any]:
    if not isinstance(v, list):
        raise ProtocolError(f"{field} must be array, got {type(v).__name__}")
    return v


# ---------------------------------------------------------------------------
# Domain types — Symbol, Timeframe, Bar
# ---------------------------------------------------------------------------


def _encode_symbol(sym: Symbol) -> dict[str, Any]:
    return {
        "base": sym.base,
        "quote": sym.quote,
        "asset_class": sym.asset_class.value,
        "venue": sym.venue,
    }


def _decode_symbol(payload: dict[str, Any]) -> Symbol:
    ac = _as_str(payload.get("asset_class"), "symbol.asset_class")
    try:
        asset_class = AssetClass(ac)
    except ValueError as exc:
        raise ProtocolError(f"Unknown asset_class {ac!r}") from exc
    venue = payload.get("venue")
    if venue is not None and not isinstance(venue, str):
        raise ProtocolError("symbol.venue must be string or null")
    return Symbol(
        base=_as_str(payload.get("base"), "symbol.base"),
        quote=_as_str(payload.get("quote"), "symbol.quote"),
        asset_class=asset_class,
        venue=venue,
    )


def _encode_bar(bar: Bar) -> dict[str, Any]:
    return {
        "timestamp": bar.timestamp.isoformat(),
        "open": str(bar.open),
        "high": str(bar.high),
        "low": str(bar.low),
        "close": str(bar.close),
        "volume": str(bar.volume),
    }


def _decode_bar(payload: dict[str, Any]) -> Bar:
    return Bar(
        timestamp=_decode_datetime(payload.get("timestamp"), "bar.timestamp"),
        open=Decimal(_as_str(payload.get("open"), "bar.open")),
        high=Decimal(_as_str(payload.get("high"), "bar.high")),
        low=Decimal(_as_str(payload.get("low"), "bar.low")),
        close=Decimal(_as_str(payload.get("close"), "bar.close")),
        volume=Decimal(_as_str(payload.get("volume"), "bar.volume")),
    )


def _decode_datetime(raw: Any, field: str) -> datetime:
    s = _as_str(raw, field)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as exc:
        raise ProtocolError(f"{field} is not a valid ISO datetime: {s!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _decode_uuid(raw: Any, field: str) -> UUID:
    s = _as_str(raw, field)
    try:
        return UUID(s)
    except ValueError as exc:
        raise ProtocolError(f"{field} is not a valid UUID") from exc


# ---------------------------------------------------------------------------
# AlgorithmContext
# ---------------------------------------------------------------------------


def serialize_context(ctx: AlgorithmContext) -> dict[str, Any]:
    """Serialize an AlgorithmContext to a JSON-safe dict.

    The ``emit_fn`` and ``log_fn`` callables are NOT serialized — the worker
    reconstructs them as local list-appenders and the captured signals/logs
    travel back in the response payload.
    """
    return {
        "tenant_id": str(ctx.tenant_id),
        "persona_id": str(ctx.persona_id),
        "algorithm_id": ctx.algorithm_id,
        "symbol": _encode_symbol(ctx.symbol),
        "timeframe": ctx.timeframe.value,
        "now": ctx.now.isoformat(),
        "bar": _encode_bar(ctx.bar),
        "history_arrays": {k: _encode_array(v) for k, v in ctx.history_arrays.items()},
        "features": _encode_features(ctx.features),
        "params": _encode_params(ctx.params),
    }


def deserialize_context(
    payload: dict[str, Any],
    *,
    emit_fn: Callable[[Signal], None],
    log_fn: Callable[[str, dict[str, Any]], None],
) -> AlgorithmContext:
    """Reconstruct an AlgorithmContext from a serialized payload.

    Caller supplies the side-effect callbacks — typically list-append closures
    on the worker side, or a thin shim on the parent side that forwards to
    the real trading-loop callbacks.
    """
    history = _as_dict(payload.get("history_arrays"), "history_arrays")
    return AlgorithmContext(
        tenant_id=_decode_uuid(payload.get("tenant_id"), "tenant_id"),
        persona_id=_decode_uuid(payload.get("persona_id"), "persona_id"),
        algorithm_id=_as_str(payload.get("algorithm_id"), "algorithm_id"),
        symbol=_decode_symbol(_as_dict(payload.get("symbol"), "symbol")),
        timeframe=Timeframe(_as_str(payload.get("timeframe"), "timeframe")),
        now=_decode_datetime(payload.get("now"), "now"),
        bar=_decode_bar(_as_dict(payload.get("bar"), "bar")),
        history_arrays={
            _as_str(k, "history_arrays.key"): _decode_array(_as_dict(v, "history_arrays.value"))
            for k, v in history.items()
        },
        features=_decode_features(_as_dict(payload.get("features"), "features")),
        params=_decode_params(_as_dict(payload.get("params"), "params")),
        emit_fn=emit_fn,
        log_fn=log_fn,
    )


def _encode_features(features: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in features.items():
        if not isinstance(k, str):
            raise ProtocolError(f"feature key must be string, got {type(k).__name__}")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ProtocolError(f"feature {k!r} must be numeric, got {type(v).__name__}")
        out[k] = float(v)
    return out


def _decode_features(payload: dict[str, Any]) -> dict[str, float]:
    return {k: _as_float(v, f"features.{k}") for k, v in payload.items()}


# Params are user-controlled and can be int/float/bool/str (see AlgorithmParam).
_PARAM_PRIMITIVES = (bool, int, float, str)


def _encode_params(params: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in params.items():
        if not isinstance(k, str):
            raise ProtocolError("param key must be string")
        if v is None or isinstance(v, _PARAM_PRIMITIVES):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            # Allow flat lists of primitives. Reject nested structures —
            # AlgorithmParam declares these flat anyway.
            for x in v:
                if x is not None and not isinstance(x, _PARAM_PRIMITIVES):
                    raise ProtocolError(
                        f"param {k!r} list contains non-primitive {type(x).__name__}"
                    )
            out[k] = list(v)
        else:
            raise ProtocolError(
                f"param {k!r} has unsupported type {type(v).__name__}"
            )
    return out


def _decode_params(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in payload.items():
        if not isinstance(k, str):
            raise ProtocolError("param key must be string")
        if v is None or isinstance(v, _PARAM_PRIMITIVES):
            out[k] = v
        elif isinstance(v, list):
            out[k] = [x for x in v]
        else:
            raise ProtocolError(
                f"param {k!r} decoded as unsupported type {type(v).__name__}"
            )
    return out


# ---------------------------------------------------------------------------
# Signal / SignalContribution
# ---------------------------------------------------------------------------


def serialize_contribution(sc: SignalContribution) -> dict[str, Any]:
    return {
        "node_id": sc.node_id,
        "node_type": sc.node_type,
        "score": sc.score,
        "confidence": sc.confidence,
        "weight": sc.weight,
        "reason": sc.reason,
        "features_snapshot": _encode_features(sc.features_snapshot),
        "children": [serialize_contribution(c) for c in sc.children],
    }


def deserialize_contribution(payload: dict[str, Any]) -> SignalContribution:
    weight = payload.get("weight")
    if weight is not None and (isinstance(weight, bool) or not isinstance(weight, (int, float))):
        raise ProtocolError("contribution.weight must be number or null")
    children_raw = _as_list(payload.get("children", []), "contribution.children")
    children = tuple(
        deserialize_contribution(_as_dict(c, "contribution.child")) for c in children_raw
    )
    return SignalContribution(
        node_id=_as_str(payload.get("node_id"), "contribution.node_id"),
        node_type=_as_str(payload.get("node_type"), "contribution.node_type"),
        score=_as_float(payload.get("score"), "contribution.score"),
        confidence=_as_float(payload.get("confidence"), "contribution.confidence"),
        weight=float(weight) if weight is not None else None,
        reason=_as_str(payload.get("reason", ""), "contribution.reason"),
        features_snapshot=_decode_features(
            _as_dict(payload.get("features_snapshot", {}), "contribution.features_snapshot")
        ),
        children=children,
    )


def serialize_signal(sig: Signal) -> dict[str, Any]:
    return {
        "id": str(sig.id),
        "timestamp": sig.timestamp.isoformat(),
        "symbol_key": sig.symbol_key,
        "score": sig.score,
        "confidence": sig.confidence,
        "source": sig.source,
        "reason": sig.reason,
        "attribution": serialize_contribution(sig.attribution) if sig.attribution else None,
        "metadata": _encode_metadata(sig.metadata),
    }


def deserialize_signal(payload: dict[str, Any]) -> Signal:
    """Reconstruct a Signal from a worker-supplied payload.

    Range-checks score and confidence — if the worker sent garbage, the
    Signal dataclass __post_init__ will reject it, which is the same
    validation built-in algos go through.
    """
    attr_raw = payload.get("attribution")
    attribution = (
        deserialize_contribution(_as_dict(attr_raw, "attribution"))
        if attr_raw is not None
        else None
    )
    return Signal(
        id=_decode_uuid(payload.get("id"), "signal.id"),
        timestamp=_decode_datetime(payload.get("timestamp"), "signal.timestamp"),
        symbol_key=_as_str(payload.get("symbol_key"), "signal.symbol_key"),
        score=_as_float(payload.get("score"), "signal.score"),
        confidence=_as_float(payload.get("confidence"), "signal.confidence"),
        source=_as_str(payload.get("source"), "signal.source"),
        reason=_as_str(payload.get("reason", ""), "signal.reason"),
        attribution=attribution,
        metadata=_decode_metadata(_as_dict(payload.get("metadata", {}), "signal.metadata")),
    )


# Signal metadata is user-controlled; allow JSON-native primitives only. No
# nested structures with arbitrary depth — bound it at the first level.
_METADATA_PRIMITIVES = (bool, int, float, str)


def _encode_metadata(md: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in md.items():
        if not isinstance(k, str):
            raise ProtocolError("metadata key must be string")
        if v is None or isinstance(v, _METADATA_PRIMITIVES):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            for x in v:
                if x is not None and not isinstance(x, _METADATA_PRIMITIVES):
                    raise ProtocolError(
                        f"metadata {k!r} list contains non-primitive"
                    )
            out[k] = list(v)
        elif isinstance(v, dict):
            inner: dict[str, Any] = {}
            for ik, iv in v.items():
                if not isinstance(ik, str):
                    raise ProtocolError("nested metadata key must be string")
                if iv is None or isinstance(iv, _METADATA_PRIMITIVES):
                    inner[ik] = iv
                else:
                    raise ProtocolError(
                        f"metadata {k!r}.{ik!r} has unsupported type "
                        f"{type(iv).__name__}"
                    )
            out[k] = inner
        else:
            raise ProtocolError(
                f"metadata {k!r} has unsupported type {type(v).__name__}"
            )
    return out


def _decode_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return _encode_metadata(payload)  # symmetric — same validation rules


# ---------------------------------------------------------------------------
# Convenience for tests / in-memory round-trips
# ---------------------------------------------------------------------------


@dataclass
class _BytesStream:
    """Minimal IO[bytes] facade backed by a BytesIO. Used in unit tests."""

    buf: io.BytesIO

    def read(self, n: int) -> bytes:
        return self.buf.read(n)

    def write(self, data: bytes) -> int:
        return self.buf.write(data)

    def flush(self) -> None:
        pass
