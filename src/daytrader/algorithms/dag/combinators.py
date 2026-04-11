"""Built-in combinator functions for DAG composition.

Each combinator is a pure function that merges child signals into
a single composite signal. They operate on already-emitted signals,
not on raw bar data.
"""

from __future__ import annotations

from typing import Any

from ...core.types.signals import Signal


def weighted_average(
    signals: list[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Weighted average of child scores and confidences.

    Returns ``(score, confidence)`` or ``None`` if no valid signals.
    """
    normalize = (params or {}).get("normalize_weights", True)

    valid_scores: list[float] = []
    valid_confidences: list[float] = []
    valid_weights: list[float] = []

    for sig, w in zip(signals, weights):
        if sig is not None:
            valid_scores.append(sig.score)
            valid_confidences.append(sig.confidence)
            valid_weights.append(w)

    if not valid_scores:
        return None

    total_w = sum(valid_weights)
    if total_w == 0:
        return None

    if normalize:
        norm_weights = [w / total_w for w in valid_weights]
    else:
        norm_weights = valid_weights

    score = sum(s * w for s, w in zip(valid_scores, norm_weights))
    confidence = sum(c * w for c, w in zip(valid_confidences, norm_weights))

    score = max(-1.0, min(1.0, score))
    confidence = max(0.0, min(1.0, confidence))
    return score, confidence


def majority_vote(
    signals: list[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Majority vote: signal direction = sign of weighted majority.

    Returns ``(score, confidence)`` or ``None`` if no consensus
    meets the minimum agreement threshold.
    """
    min_agreement = (params or {}).get("min_agreement", 0.5)

    valid = [(sig, w) for sig, w in zip(signals, weights) if sig is not None]
    if not valid:
        return None

    bullish_weight = sum(w for sig, w in valid if sig.score > 0)
    bearish_weight = sum(w for sig, w in valid if sig.score < 0)
    total_weight = sum(w for _, w in valid)

    if total_weight == 0:
        return None

    bull_ratio = bullish_weight / total_weight
    bear_ratio = bearish_weight / total_weight

    if bull_ratio >= min_agreement:
        avg_score = sum(sig.score * w for sig, w in valid if sig.score > 0) / bullish_weight if bullish_weight > 0 else 0.5
        return max(0.0, min(1.0, avg_score)), bull_ratio
    elif bear_ratio >= min_agreement:
        avg_score = sum(sig.score * w for sig, w in valid if sig.score < 0) / bearish_weight if bearish_weight > 0 else -0.5
        return max(-1.0, min(0.0, avg_score)), bear_ratio

    return None  # No consensus


def threshold_filter(
    signals: list[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Pass first child's signal only if second child's score exceeds threshold.

    Expects exactly 2 children: index 0 = signal source, index 1 = gate.
    """
    threshold = (params or {}).get("threshold", 0.5)
    gate_index = (params or {}).get("gate_index", 1)
    signal_index = 1 - gate_index  # the other one

    if len(signals) < 2:
        return None

    gate_sig = signals[gate_index]
    source_sig = signals[signal_index]

    if gate_sig is None or source_sig is None:
        return None

    if abs(gate_sig.score) >= threshold:
        return source_sig.score, source_sig.confidence

    return None  # Gate blocked


def unanimous(
    signals: list[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Emit only if ALL children agree on direction.

    All must be positive or all must be negative.
    """
    valid = [sig for sig in signals if sig is not None]
    if not valid:
        return None

    all_bullish = all(s.score > 0 for s in valid)
    all_bearish = all(s.score < 0 for s in valid)

    if not (all_bullish or all_bearish):
        return None

    avg_score = sum(s.score for s in valid) / len(valid)
    avg_confidence = sum(s.confidence for s in valid) / len(valid)
    return max(-1.0, min(1.0, avg_score)), max(0.0, min(1.0, avg_confidence))


COMBINATORS = {
    "weighted_average": weighted_average,
    "majority_vote": majority_vote,
    "threshold_filter": threshold_filter,
    "unanimous": unanimous,
}
