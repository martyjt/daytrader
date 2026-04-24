"""Cross-persona correlation monitor — risk layer 3 overfit/coupling detector.

Two personas are supposed to be independent. If their signal streams or
equity curves correlate too strongly, at least one of three things is
true:

1. They secretly trade the same features (unintentional coupling).
2. They're overfit to the same market quirk (a correlation mirage).
3. They hold the same position because the market's only doing one thing.

The monitor computes pairwise Pearson correlation of recent signal
scores across personas. A correlation ``>= warn_threshold`` surfaces a
warning; ``>= breach_threshold`` logs a risk-layer-3 event. The monitor
is diagnostic — it does NOT auto-halt trading, because high correlation
in a strong trend is benign.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

import numpy as np


@dataclass(frozen=True)
class PersonaPair:
    """One pairwise correlation result."""

    persona_a: str
    persona_b: str
    persona_a_id: UUID
    persona_b_id: UUID
    correlation: float
    n_shared_buckets: int
    severity: str  # "ok" | "warn" | "breach"


@dataclass
class CorrelationReport:
    """Aggregate report of pairwise correlations."""

    window_start: datetime | None = None
    window_end: datetime | None = None
    bucket_seconds: int = 3600
    pairs: list[PersonaPair] = field(default_factory=list)
    warn_threshold: float = 0.7
    breach_threshold: float = 0.9

    @property
    def worst(self) -> PersonaPair | None:
        """Pair with the highest absolute correlation (largest coupling)."""
        if not self.pairs:
            return None
        return max(self.pairs, key=lambda p: abs(p.correlation))

    @property
    def overall_severity(self) -> str:
        worst = self.worst
        if worst is None:
            return "ok"
        return worst.severity


def classify_correlation(
    abs_corr: float,
    *,
    warn: float = 0.7,
    breach: float = 0.9,
) -> str:
    """Map absolute correlation to ``ok | warn | breach``."""
    if abs_corr >= breach:
        return "breach"
    if abs_corr >= warn:
        return "warn"
    return "ok"


def _bucketize_scores(
    timestamps: list[datetime],
    scores: list[float],
    start: datetime,
    end: datetime,
    bucket_seconds: int,
) -> np.ndarray:
    """Average scores into fixed-width time buckets between ``start`` and ``end``.

    Empty buckets are filled with NaN; ``_pairwise_pearson`` treats them
    as missing.
    """
    total = max(1, int((end - start).total_seconds() // bucket_seconds))
    buckets: list[list[float]] = [[] for _ in range(total)]
    for ts, score in zip(timestamps, scores):
        if ts < start or ts >= end:
            continue
        idx = int((ts - start).total_seconds() // bucket_seconds)
        if 0 <= idx < total:
            buckets[idx].append(float(score))
    out = np.full(total, np.nan)
    for i, b in enumerate(buckets):
        if b:
            out[i] = float(np.mean(b))
    return out


def _pairwise_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """Pearson correlation on the intersection of non-NaN buckets.

    Returns ``(corr, n_shared)``. ``corr`` is 0.0 if no variance or <2 shared.
    """
    mask = ~(np.isnan(a) | np.isnan(b))
    n = int(mask.sum())
    if n < 2:
        return 0.0, n
    xa = a[mask]
    xb = b[mask]
    if np.std(xa) == 0 or np.std(xb) == 0:
        return 0.0, n
    corr = float(np.corrcoef(xa, xb)[0, 1])
    if not np.isfinite(corr):
        return 0.0, n
    return corr, n


async def scan_persona_correlations(
    *,
    tenant_id: UUID,
    lookback_hours: int = 72,
    bucket_seconds: int = 3600,
    warn_threshold: float = 0.7,
    breach_threshold: float = 0.9,
) -> CorrelationReport:
    """Compute pairwise signal-score correlations between all personas.

    Queries the SignalModel table directly (not through the broader
    repository) because we need raw per-timestamp reads across personas.
    """
    from sqlalchemy import select

    from ..core.context import tenant_scope
    from ..core.types.common import utcnow
    from ..storage.database import get_session
    from ..storage.models import PersonaModel, SignalModel

    end = utcnow()
    start = end - timedelta(hours=lookback_hours)

    report = CorrelationReport(
        window_start=start,
        window_end=end,
        bucket_seconds=bucket_seconds,
        warn_threshold=warn_threshold,
        breach_threshold=breach_threshold,
    )

    async with get_session() as session:
        with tenant_scope(tenant_id):
            persona_rows = (await session.execute(
                select(PersonaModel).where(PersonaModel.tenant_id == tenant_id)
            )).scalars().all()
            if len(persona_rows) < 2:
                return report

            signal_rows = (await session.execute(
                select(SignalModel)
                .where(SignalModel.tenant_id == tenant_id)
                .where(SignalModel.created_at >= start)
                .order_by(SignalModel.created_at.asc())
            )).scalars().all()

    # Bucketize each persona's scores.
    per_persona: dict[UUID, tuple[list[datetime], list[float]]] = {
        p.id: ([], []) for p in persona_rows
    }
    for row in signal_rows:
        pid = row.persona_id
        if pid in per_persona:
            per_persona[pid][0].append(row.created_at)
            per_persona[pid][1].append(float(row.score))

    bucketed: dict[UUID, np.ndarray] = {
        p.id: _bucketize_scores(
            per_persona[p.id][0], per_persona[p.id][1],
            start, end, bucket_seconds,
        )
        for p in persona_rows
    }

    # Pairwise Pearson, emit one row per (a, b) with a.id < b.id.
    personas_by_id = {p.id: p for p in persona_rows}
    ids = sorted(personas_by_id.keys(), key=lambda i: str(i))
    for i, a_id in enumerate(ids):
        for b_id in ids[i + 1:]:
            corr, n_shared = _pairwise_pearson(bucketed[a_id], bucketed[b_id])
            severity = classify_correlation(
                abs(corr), warn=warn_threshold, breach=breach_threshold,
            )
            report.pairs.append(PersonaPair(
                persona_a=personas_by_id[a_id].name,
                persona_b=personas_by_id[b_id].name,
                persona_a_id=a_id,
                persona_b_id=b_id,
                correlation=corr,
                n_shared_buckets=n_shared,
                severity=severity,
            ))
    return report
