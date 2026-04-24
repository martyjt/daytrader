"""Unit tests for ``daytrader.research.feature_lift``.

Locks the statistical invariants we rely on:

* ``benjamini_hochberg`` is monotone and controls FDR.
* ``q_values`` are in ``[0, 1]`` and monotone w.r.t. sorted p-values.
* ``paired_t_pvalue`` returns small p for clear effects, large for noise.
* ``time_series_splits`` never leaks future bars into training.
"""

from __future__ import annotations

import numpy as np
import pytest

from daytrader.research.feature_lift import (
    benjamini_hochberg,
    paired_t_pvalue,
    q_values,
    time_series_splits,
)


class TestTimeSeriesSplits:
    def test_no_future_leakage(self) -> None:
        splits = time_series_splits(200, n_folds=5, min_train=40)
        for train_idx, test_idx in splits:
            assert train_idx.max() < test_idx.min(), (
                "train fold must end strictly before test fold starts"
            )

    def test_fold_count_respects_n_folds(self) -> None:
        splits = time_series_splits(200, n_folds=5, min_train=40)
        assert len(splits) == 5

    def test_returns_empty_when_not_enough_samples(self) -> None:
        # min_train=60 plus at least one fold of size >=1 needs n > 60.
        splits = time_series_splits(50, n_folds=5, min_train=60)
        assert splits == []


class TestBenjaminiHochberg:
    def test_accepts_significant_pvalues(self) -> None:
        flags = benjamini_hochberg([0.001, 0.01, 0.5], alpha=0.1)
        assert flags[0] is True
        assert flags[1] is True
        assert flags[2] is False

    def test_handles_none_as_never_accepted(self) -> None:
        flags = benjamini_hochberg([0.001, None, 0.01], alpha=0.1)
        assert flags == [True, False, True]

    def test_empty_input(self) -> None:
        assert benjamini_hochberg([], alpha=0.1) == []

    def test_stricter_alpha_rejects_more(self) -> None:
        ps = [0.001, 0.02, 0.04, 0.2]
        loose = benjamini_hochberg(ps, alpha=0.1)
        strict = benjamini_hochberg(ps, alpha=0.01)
        assert sum(strict) <= sum(loose)


class TestQValues:
    def test_q_values_in_unit_interval(self) -> None:
        ps = [0.001, 0.01, 0.05, 0.2, 0.5]
        qs = q_values(ps)
        for q in qs:
            assert q is not None
            assert 0.0 <= q <= 1.0

    def test_q_values_monotone_on_sorted_ps(self) -> None:
        ps = [0.001, 0.01, 0.05, 0.2, 0.5]
        qs = q_values(ps)
        # Sorted ps should produce non-decreasing q-values (BH step-up).
        for i in range(1, len(qs)):
            assert qs[i] >= qs[i - 1] - 1e-9

    def test_none_preserved(self) -> None:
        qs = q_values([0.01, None, 0.5])
        assert qs[1] is None


class TestPairedTPvalue:
    def test_small_p_for_strong_effect(self) -> None:
        # Candidate consistently better than baseline by 0.1.
        baseline = [0.5, 0.55, 0.52, 0.58, 0.50]
        candidate = [b + 0.10 for b in baseline]
        p = paired_t_pvalue(baseline, candidate)
        assert p is not None
        assert p < 0.01

    def test_high_p_for_zero_effect(self) -> None:
        values = [0.5, 0.55, 0.52, 0.58, 0.50]
        p = paired_t_pvalue(values, values)
        assert p == 1.0

    def test_returns_none_for_too_few_samples(self) -> None:
        assert paired_t_pvalue([0.5], [0.6]) is None


class TestFeatureLiftIntegration:
    """End-to-end: synthetic data with a hidden predictor should be detected."""

    def test_hidden_signal_wins_over_noise(self) -> None:
        pytest.importorskip("xgboost")
        pytest.importorskip("sklearn")
        from daytrader.research.feature_lift import feature_lift

        rng = np.random.default_rng(42)
        n = 250
        X_base = rng.standard_normal((n, 5))
        hidden = rng.standard_normal(n)
        logits = 2.0 * hidden + 0.5 * rng.standard_normal(n)
        y = (logits > 0).astype(float)

        predictive = feature_lift(
            y=y, X_base=X_base, X_cand=hidden,
            task="classification", n_folds=5, min_train=80,
        )
        noise = feature_lift(
            y=y, X_base=X_base, X_cand=rng.standard_normal(n),
            task="classification", n_folds=5, min_train=80,
        )

        assert predictive.lift > 0.1, (
            f"hidden predictor should lift AUC; got {predictive.lift:+.3f}"
        )
        assert noise.lift < predictive.lift
