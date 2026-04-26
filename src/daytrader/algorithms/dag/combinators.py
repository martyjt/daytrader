"""Built-in combinator functions for DAG composition.

Each combinator merges child signals into a single composite signal.
They operate on already-emitted signals, not on raw bar data.

Most combinators are stateless pure functions; rolling variants
accept an optional ``state`` dict that persists across bars so they
can remember recent signals within a window.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ...core.types.signals import Signal


def weighted_average(
    signals: Sequence[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Weighted average of child scores and confidences.

    Returns ``(score, confidence)`` or ``None`` if no valid signals.
    """
    normalize = (params or {}).get("normalize_weights", True)

    valid_scores: list[float] = []
    valid_confidences: list[float] = []
    valid_weights: list[float] = []

    for sig, w in zip(signals, weights, strict=False):
        if sig is not None:
            valid_scores.append(sig.score)
            valid_confidences.append(sig.confidence)
            valid_weights.append(w)

    if not valid_scores:
        return None

    total_w = sum(valid_weights)
    if total_w == 0:
        return None

    norm_weights = [w / total_w for w in valid_weights] if normalize else valid_weights

    score = sum(s * w for s, w in zip(valid_scores, norm_weights, strict=False))
    confidence = sum(c * w for c, w in zip(valid_confidences, norm_weights, strict=False))

    score = max(-1.0, min(1.0, score))
    confidence = max(0.0, min(1.0, confidence))
    return score, confidence


def majority_vote(
    signals: Sequence[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Majority vote: signal direction = sign of weighted majority.

    Returns ``(score, confidence)`` or ``None`` if no consensus
    meets the minimum agreement threshold.
    """
    min_agreement = (params or {}).get("min_agreement", 0.5)

    valid = [(sig, w) for sig, w in zip(signals, weights, strict=False) if sig is not None]
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
    signals: Sequence[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
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
    signals: Sequence[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
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


def _latest_signals_in_window(
    signals: Sequence[Signal | None],
    state: dict[str, Any] | None,
    window: int,
) -> list[tuple[float, float] | None]:
    """Update state with the current bar's signals and return the most
    recent non-None (score, confidence) tuple per slot, within the
    rolling window.

    State layout: ``state["history"]`` is a list of per-bar entries;
    each entry is a list of (score, confidence) tuples or None per slot.
    """
    if state is None:
        # No state = can only see this bar. Degenerate to the given signals.
        return [(s.score, s.confidence) if s else None for s in signals]

    history: list[list[tuple[float, float] | None]] = state.setdefault(
        "history", []
    )
    current = [(s.score, s.confidence) if s else None for s in signals]
    history.append(current)
    if len(history) > window:
        del history[: len(history) - window]

    # Find the most recent non-None per slot
    n_slots = len(signals)
    latest: list[tuple[float, float] | None] = [None] * n_slots
    for bar_entry in history:
        for i in range(min(n_slots, len(bar_entry))):
            if bar_entry[i] is not None:
                latest[i] = bar_entry[i]
    return latest


def rolling_unanimous(
    signals: Sequence[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Emit when at least ``min_fired`` children have fired within the
    last ``window_bars`` and every one of those remembered signals
    agrees on direction.

    Unlike regular ``unanimous``, which only sees signals that fire on
    the exact same bar (near-impossible with event-driven algos), this
    version keeps a per-child memory so a MACD crossover 3 bars ago can
    still "count" as recent agreement with an ADX signal firing now.

    Children that have not fired within the window are ignored (they
    don't vote either way). This means a silent child never blocks the
    consensus — you don't need every algorithm to speak, just enough.

    Params:
        window_bars: int (default 5)
        min_fired: int (default 2) — minimum number of children that
            must have fired within the window for the vote to count.
    """
    window = int((params or {}).get("window_bars", 5))
    min_fired = int((params or {}).get("min_fired", 2))
    latest = _latest_signals_in_window(signals, state, window)

    active = [s for s in latest if s is not None]
    if len(active) < min_fired:
        return None

    scores = [s[0] for s in active]
    confs = [s[1] for s in active]

    all_bullish = all(v > 0 for v in scores)
    all_bearish = all(v < 0 for v in scores)
    if not (all_bullish or all_bearish):
        return None

    avg_score = sum(scores) / len(scores)
    avg_conf = sum(confs) / len(confs)
    return max(-1.0, min(1.0, avg_score)), max(0.0, min(1.0, avg_conf))


def rolling_majority_vote(
    signals: Sequence[Signal | None],
    weights: list[float],
    *,
    params: dict[str, Any] | None = None,
    state: dict[str, Any] | None = None,
) -> tuple[float, float] | None:
    """Weighted majority of each child's most-recent signal within the
    last ``window_bars``. Children that have not fired within the window
    do not vote.

    Fires when the bullish or bearish weight exceeds
    ``min_agreement`` (default 0.5) of the voting weight.

    Default window: 5 bars.
    """
    window = int((params or {}).get("window_bars", 5))
    min_agreement = float((params or {}).get("min_agreement", 0.5))
    latest = _latest_signals_in_window(signals, state, window)

    # Each voter is (score, confidence, weight); score can be pos/neg
    voters: list[tuple[float, float, float]] = []
    for i in range(len(latest)):
        item = latest[i]
        if item is None or i >= len(weights):
            continue
        score, conf = item
        voters.append((score, conf, weights[i]))

    if not voters:
        return None

    total_w = sum(w for _, _, w in voters)
    if total_w == 0:
        return None

    bull_w = sum(w for sc, _, w in voters if sc > 0)
    bear_w = sum(w for sc, _, w in voters if sc < 0)

    if bull_w > 0 and bull_w / total_w >= min_agreement:
        avg = sum(sc * w for sc, _, w in voters if sc > 0) / bull_w
        return max(0.0, min(1.0, avg)), bull_w / total_w
    if bear_w > 0 and bear_w / total_w >= min_agreement:
        avg = sum(sc * w for sc, _, w in voters if sc < 0) / bear_w
        return max(-1.0, min(0.0, avg)), bear_w / total_w
    return None


COMBINATORS = {
    "weighted_average": weighted_average,
    "majority_vote": majority_vote,
    "threshold_filter": threshold_filter,
    "unanimous": unanimous,
    "rolling_unanimous": rolling_unanimous,
    "rolling_majority_vote": rolling_majority_vote,
}
