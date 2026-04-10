"""Trading signals and their attribution trees.

The attribution tree is the single feature that earns user trust:
every emitted signal can be expanded to show exactly which algorithm
or combinator contributed, with what score, confidence, weight, and
feature snapshot at the moment of emission.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class SignalContribution:
    """One node's contribution to a (possibly composed) signal."""

    node_id: str
    node_type: str  # "algorithm" | "combinator" | "risk_filter" | "ml_model"
    score: float
    confidence: float
    weight: float | None = None
    reason: str = ""
    features_snapshot: dict[str, float] = field(default_factory=dict)
    children: tuple["SignalContribution", ...] = ()


@dataclass(frozen=True, slots=True)
class Signal:
    """An emitted trading intent.

    ``score``       — direction/magnitude in ``[-1.0, 1.0]``. Negative shorts, positive longs.
    ``confidence``  — how sure the emitter is, in ``[0.0, 1.0]``. Orthogonal to score.
    """

    id: UUID
    timestamp: datetime
    symbol_key: str
    score: float
    confidence: float
    source: str
    reason: str = ""
    attribution: SignalContribution | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(f"Signal score must be in [-1, 1], got {self.score}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Signal confidence must be in [0, 1], got {self.confidence}")

    @classmethod
    def new(
        cls,
        *,
        symbol_key: str,
        score: float,
        source: str,
        confidence: float = 1.0,
        reason: str = "",
        attribution: SignalContribution | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "Signal":
        return cls(
            id=uuid4(),
            timestamp=datetime.now(timezone.utc),
            symbol_key=symbol_key,
            score=score,
            confidence=confidence,
            source=source,
            reason=reason,
            attribution=attribution,
            metadata=metadata or {},
        )
