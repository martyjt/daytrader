"""Unit tests for ``daytrader.research.drift_monitor``.

PSI and its severity classifier are industry-standard; these tests lock
the canonical threshold behaviour so someone can't silently widen the
bands in the future.
"""

from __future__ import annotations

import numpy as np

from daytrader.research.drift_monitor import classify_psi, compute_psi


def _reference_rng() -> np.random.Generator:
    return np.random.default_rng(42)


class TestComputePsi:
    def test_stable_distributions(self) -> None:
        rng = _reference_rng()
        ref = rng.standard_normal(1000)
        cur = rng.standard_normal(1000)
        psi = compute_psi(ref, cur)
        assert psi < 0.1, f"same-distribution PSI should be < 0.1; got {psi:.4f}"

    def test_significant_mean_shift(self) -> None:
        rng = _reference_rng()
        ref = rng.standard_normal(1000)
        cur = rng.standard_normal(1000) + 1.0
        psi = compute_psi(ref, cur)
        assert psi >= 0.25, f"1σ mean shift should be significant; got {psi:.4f}"

    def test_significant_variance_change(self) -> None:
        rng = _reference_rng()
        ref = rng.standard_normal(1000)
        cur = rng.standard_normal(1000) * 2.0
        psi = compute_psi(ref, cur)
        assert psi >= 0.25, f"2× variance should be significant; got {psi:.4f}"

    def test_nan_resilience(self) -> None:
        rng = _reference_rng()
        ref = rng.standard_normal(1000)
        cur = rng.standard_normal(1000)
        cur[:100] = np.nan  # partial NaN shouldn't blow up
        psi = compute_psi(ref, cur)
        assert np.isfinite(psi)

    def test_insufficient_data_returns_nan(self) -> None:
        psi = compute_psi(np.array([1.0, 2.0]), np.array([1.0]))
        assert np.isnan(psi)


class TestClassifyPsi:
    def test_classification_thresholds(self) -> None:
        assert classify_psi(0.0) == "stable"
        assert classify_psi(0.05) == "stable"
        assert classify_psi(0.1) == "moderate"
        assert classify_psi(0.24) == "moderate"
        assert classify_psi(0.25) == "significant"
        assert classify_psi(1.0) == "significant"

    def test_nan_is_insufficient_data(self) -> None:
        assert classify_psi(float("nan")) == "insufficient_data"
