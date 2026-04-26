"""Feature-lift test: does adding candidate X to the feature set improve prediction?

Given:
    * A target vector ``y`` (e.g. next-bar log return, or up/down direction).
    * A baseline feature matrix ``X_base``.
    * One candidate column ``X_cand`` (the new feature to evaluate).

The test fits a shallow gradient-boosted tree on ``X_base`` alone, then
on ``[X_base, X_cand]``, under **time-series cross-validation** (no peek
into the future). It reports the lift in OOS metric and a paired t-test
p-value across folds.

Design notes:

* We use XGBoost because it's already a dependency and handles missing
  values (``NaN``) natively — important when candidate features have
  different cadences than the target (weekly macro vs daily prices).
* Time-series CV splits chronologically. We use an **expanding-window**
  strategy: fold k trains on ``[0, k*fold_size)`` and tests on the next
  slice. This mirrors the walk-forward harness used in backtesting.
* Metric: AUC for classification (direction up/down), RMSE for
  regression (log returns). Lift is reported as a signed float where
  higher is always better.
* Paired t-test: compares per-fold metric between baseline and
  candidate. Null hypothesis: candidate has zero effect.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LiftResult:
    """Result of one feature-lift test."""

    baseline_metric: float
    candidate_metric: float
    lift: float
    p_value: float | None
    n_folds: int
    task: str  # "classification" | "regression"


def time_series_splits(
    n_samples: int, n_folds: int = 5, min_train: int = 60,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window time-series splits.

    Returns ``n_folds`` (train_idx, test_idx) pairs. Fold k tests on a
    chronologically-ordered slice that never overlaps training.
    """
    n_folds = max(2, n_folds)
    fold_size = max(1, (n_samples - min_train) // n_folds)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        train_end = min_train + k * fold_size
        test_end = train_end + fold_size
        if train_end >= n_samples or test_end > n_samples:
            break
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        splits.append((train_idx, test_idx))
    return splits


def _fit_score_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Return **negative RMSE** so "higher is better" (matches AUC convention)."""
    import xgboost as xgb  # lazy import

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        verbosity=0,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((pred - y_test) ** 2)))
    return -rmse


def _fit_score_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Return AUC. Falls back to accuracy if one class dominates."""
    import xgboost as xgb  # lazy import
    from sklearn.metrics import roc_auc_score

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        verbosity=0,
        objective="binary:logistic",
        eval_metric="logloss",
    )
    model.fit(X_train, y_train.astype(int))
    if len(np.unique(y_test)) < 2:
        # AUC undefined — use accuracy.
        preds = model.predict(X_test)
        return float((preds == y_test.astype(int)).mean())
    proba = model.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test.astype(int), proba))


def paired_t_pvalue(baseline: list[float], candidate: list[float]) -> float | None:
    """Two-sided paired t-test p-value for ``candidate - baseline``.

    Returns None if there's insufficient data or zero variance.
    """
    diffs = np.array(candidate) - np.array(baseline)
    n = len(diffs)
    if n < 2:
        return None
    mean = float(np.mean(diffs))
    std = float(np.std(diffs, ddof=1))
    if std == 0:
        return 0.0 if mean != 0 else 1.0
    t = mean / (std / math.sqrt(n))
    # Welch-style DF = n-1; approximate two-sided p via survival function.
    try:
        from scipy.stats import t as student_t

        return float(2 * (1 - student_t.cdf(abs(t), df=n - 1)))
    except ImportError:
        # Fallback: normal approximation (fine for n >= 30, rough for small n).
        z = abs(t)
        # Two-sided Gaussian tail
        p = math.erfc(z / math.sqrt(2))
        return float(min(1.0, max(0.0, p)))


def feature_lift(
    y: np.ndarray,
    X_base: np.ndarray,
    X_cand: np.ndarray,
    *,
    task: str = "classification",
    n_folds: int = 5,
    min_train: int = 60,
) -> LiftResult:
    """Evaluate whether adding ``X_cand`` improves prediction of ``y``.

    ``X_cand`` must be a 1D array of length ``len(y)`` aligned to the
    same time index as ``X_base``'s rows (NaNs allowed — XGBoost handles
    missingness natively).
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    if X_base.ndim != 2:
        raise ValueError(f"X_base must be 2D, got shape {X_base.shape}")
    if len(X_cand) != len(y) or len(X_base) != len(y):
        raise ValueError("X_base, X_cand, and y must have equal length")

    X_full = np.column_stack([X_base, X_cand.reshape(-1, 1)])
    splits = time_series_splits(len(y), n_folds=n_folds, min_train=min_train)
    if not splits:
        raise ValueError(
            f"Not enough samples ({len(y)}) for {n_folds} folds with min_train={min_train}"
        )

    score_fn = (
        _fit_score_regression if task == "regression" else _fit_score_classification
    )

    base_scores: list[float] = []
    cand_scores: list[float] = []
    for train_idx, test_idx in splits:
        base_scores.append(score_fn(
            X_base[train_idx], y[train_idx],
            X_base[test_idx], y[test_idx],
        ))
        cand_scores.append(score_fn(
            X_full[train_idx], y[train_idx],
            X_full[test_idx], y[test_idx],
        ))

    baseline = float(np.mean(base_scores))
    candidate = float(np.mean(cand_scores))
    return LiftResult(
        baseline_metric=baseline,
        candidate_metric=candidate,
        lift=candidate - baseline,
        p_value=paired_t_pvalue(base_scores, cand_scores),
        n_folds=len(splits),
        task=task,
    )


def benjamini_hochberg(p_values: Sequence[float | None], alpha: float = 0.1) -> list[bool]:
    """Benjamini-Hochberg FDR correction.

    Returns a list of booleans (same length as input): True for p-values
    accepted as discoveries at false discovery rate ``alpha``. None entries
    are never accepted.

    Implements the standard step-up procedure:
        sort p-values; find largest i s.t. p(i) <= i/m * alpha; accept
        all i' <= i.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None]
    if not indexed:
        return [False] * m
    indexed.sort(key=lambda t: t[1])

    rejected = [False] * m
    threshold_idx = -1
    for rank, (_orig_idx, p) in enumerate(indexed, start=1):
        if p <= (rank / m) * alpha:
            threshold_idx = rank

    if threshold_idx > 0:
        for rank, (orig_idx, _p) in enumerate(indexed, start=1):
            if rank <= threshold_idx:
                rejected[orig_idx] = True
    return rejected


def q_values(p_values: Sequence[float | None]) -> list[float | None]:
    """Compute BH-adjusted q-values (step-up monotone correction).

    For each p(i) in ascending order the q-value is ``min(p(j) * m/j)`` for
    j >= i. None inputs produce None outputs.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(
        [(i, p) for i, p in enumerate(p_values) if p is not None],
        key=lambda t: t[1],
    )
    q_sorted: list[float] = []
    running_min = 1.0
    # Walk backwards so running_min accumulates ``min`` from the tail.
    for rank in range(len(indexed), 0, -1):
        p = indexed[rank - 1][1]
        q = min(running_min, p * m / rank)
        running_min = q
        q_sorted.insert(0, q)

    out: list[float | None] = [None] * m
    for q, (orig_idx, _p) in zip(q_sorted, indexed, strict=False):
        out[orig_idx] = float(min(1.0, max(0.0, q)))
    return out
