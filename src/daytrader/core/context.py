"""Tenant context and the AlgorithmContext sandbox.

These are the two isolation primitives of the whole system:

* **tenant_scope** — every DB query, feature read, and persona action
  runs inside a context var. The storage layer refuses to operate
  without one. Cross-tenant leaks become structurally impossible.

* **AlgorithmContext** — the narrow sandbox handed to each algorithm
  plugin on each bar. Algos can read features and params, emit
  signals, and log. They cannot touch the DB, broker, filesystem, or
  other algos. This is what makes algorithm composition safe by
  construction.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import numpy as np

from .types.bars import Bar, Timeframe
from .types.signals import Signal, SignalContribution
from .types.symbols import Symbol

# ---------------------------------------------------------------------------
# Tenant context
# ---------------------------------------------------------------------------

_current_tenant: contextvars.ContextVar[UUID | None] = contextvars.ContextVar(
    "daytrader_current_tenant", default=None
)


def current_tenant() -> UUID:
    """Return the tenant in scope, raising if none is set."""
    tid = _current_tenant.get()
    if tid is None:
        raise RuntimeError(
            "No tenant in context. Wrap the call in tenant_scope(tenant_id) "
            "or ensure the auth middleware ran."
        )
    return tid


def try_current_tenant() -> UUID | None:
    """Return the tenant in scope or ``None``. Non-raising variant."""
    return _current_tenant.get()


@contextmanager
def tenant_scope(tenant_id: UUID) -> Iterator[None]:
    """Set the current tenant for the duration of the block (nestable)."""
    token = _current_tenant.set(tenant_id)
    try:
        yield
    finally:
        _current_tenant.reset(token)


# ---------------------------------------------------------------------------
# AlgorithmContext
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AlgorithmContext:
    """The sandbox every algorithm receives on each bar.

    Deliberately narrow surface. An algorithm can:

    * read the current bar, recent history, and precomputed features
    * read its declared parameters
    * emit a ``Signal`` with optional reason + metadata
    * emit a structured log line

    It cannot:

    * touch the database, broker, or filesystem
    * read other algorithms' state
    * import anything from ``daytrader.storage`` or ``daytrader.execution``

    This is enforced by convention in Phase 0 and by a test-time
    import linter in Phase 1.
    """

    tenant_id: UUID
    persona_id: UUID
    algorithm_id: str
    symbol: Symbol
    timeframe: Timeframe
    now: datetime
    bar: Bar
    history_arrays: dict[str, np.ndarray]
    features: dict[str, float]
    params: dict[str, Any]
    emit_fn: Callable[[Signal], None]
    log_fn: Callable[[str, dict[str, Any]], None]

    # ---- history ---------------------------------------------------------

    def history(self, n: int, field_name: str = "close") -> np.ndarray:
        """Return the last ``n`` values of a historical field (e.g. ``close``)."""
        arr = self.history_arrays.get(field_name)
        if arr is None:
            raise KeyError(
                f"No history for field {field_name!r}. "
                f"Available: {sorted(self.history_arrays)}"
            )
        if n > len(arr):
            raise ValueError(
                f"Requested {n} bars of {field_name!r} but only {len(arr)} "
                "are available. Increase the strategy warmup window."
            )
        return arr[-n:]

    # ---- features --------------------------------------------------------

    def feature(self, name: str, default: Any = ...) -> Any:
        """Read a precomputed feature. Raises ``KeyError`` if missing and no default."""
        if name in self.features:
            return self.features[name]
        if default is ...:
            raise KeyError(
                f"Feature {name!r} not available. Did you declare it in the "
                f"algorithm manifest? Available: {sorted(self.features)}"
            )
        return default

    def has_feature(self, name: str) -> bool:
        return name in self.features

    # ---- params ----------------------------------------------------------

    def param(self, name: str, default: Any = ...) -> Any:
        """Read a declared algorithm parameter."""
        if name in self.params:
            return self.params[name]
        if default is ...:
            raise KeyError(f"Param {name!r} not set and no default provided")
        return default

    # ---- emission --------------------------------------------------------

    def emit(
        self,
        score: float,
        *,
        confidence: float = 1.0,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Signal:
        """Emit a trading signal. ``score`` in ``[-1, 1]``, ``confidence`` in ``[0, 1]``."""
        attribution = SignalContribution(
            node_id=self.algorithm_id,
            node_type="algorithm",
            score=score,
            confidence=confidence,
            reason=reason,
            features_snapshot=dict(self.features),
        )
        signal = Signal.new(
            symbol_key=self.symbol.key,
            score=score,
            confidence=confidence,
            source=self.algorithm_id,
            reason=reason,
            attribution=attribution,
            metadata=metadata or {},
        )
        self.emit_fn(signal)
        return signal

    def log(self, message: str, **fields: Any) -> None:
        """Structured debug log captured per-run for the explainability view."""
        self.log_fn(message, fields)
