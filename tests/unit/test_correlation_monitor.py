"""Unit tests for the cross-persona correlation monitor's numeric core."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from daytrader.risk.correlation_monitor import (
    _bucketize_scores,
    _pairwise_pearson,
    classify_correlation,
)


class TestBucketize:
    def test_assigns_scores_to_correct_bucket(self) -> None:
        start = datetime(2024, 1, 1)
        end = start + timedelta(hours=3)
        timestamps = [
            start + timedelta(minutes=30),   # bucket 0
            start + timedelta(minutes=90),   # bucket 1
            start + timedelta(minutes=150),  # bucket 2
        ]
        scores = [0.5, -0.5, 0.25]
        out = _bucketize_scores(timestamps, scores, start, end, bucket_seconds=3600)
        assert out.shape == (3,)
        assert out[0] == pytest.approx(0.5)
        assert out[1] == pytest.approx(-0.5)
        assert out[2] == pytest.approx(0.25)

    def test_empty_bucket_is_nan(self) -> None:
        start = datetime(2024, 1, 1)
        end = start + timedelta(hours=3)
        timestamps = [start + timedelta(minutes=30)]  # only bucket 0 filled
        scores = [0.5]
        out = _bucketize_scores(timestamps, scores, start, end, bucket_seconds=3600)
        assert np.isnan(out[1])
        assert np.isnan(out[2])

    def test_averages_multiple_scores_in_same_bucket(self) -> None:
        start = datetime(2024, 1, 1)
        end = start + timedelta(hours=1)
        timestamps = [
            start + timedelta(minutes=10),
            start + timedelta(minutes=20),
        ]
        scores = [0.4, 0.6]
        out = _bucketize_scores(timestamps, scores, start, end, bucket_seconds=3600)
        assert out[0] == pytest.approx(0.5)


class TestPairwisePearson:
    def test_identical_series_correlate_perfectly(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        corr, n = _pairwise_pearson(x, x.copy())
        assert corr == pytest.approx(1.0)
        assert n == 5

    def test_anti_correlated(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -x
        corr, n = _pairwise_pearson(x, y)
        assert corr == pytest.approx(-1.0)
        assert n == 5

    def test_nan_alignment(self) -> None:
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        corr, n = _pairwise_pearson(x, y)
        # Only three positions (0, 1, 4) are non-NaN in both.
        assert n == 3
        assert -1.0 <= corr <= 1.0

    def test_insufficient_overlap_returns_zero(self) -> None:
        x = np.array([np.nan, np.nan, np.nan, 4.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        corr, n = _pairwise_pearson(x, y)
        assert n < 2
        assert corr == 0.0

    def test_constant_series_returns_zero(self) -> None:
        x = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        corr, _n = _pairwise_pearson(x, y)
        assert corr == 0.0


class TestClassifyCorrelation:
    def test_thresholds(self) -> None:
        assert classify_correlation(0.0) == "ok"
        assert classify_correlation(0.69) == "ok"
        assert classify_correlation(0.70) == "warn"
        assert classify_correlation(0.89) == "warn"
        assert classify_correlation(0.90) == "breach"
        assert classify_correlation(1.00) == "breach"

    def test_custom_thresholds(self) -> None:
        assert classify_correlation(0.5, warn=0.4, breach=0.6) == "warn"
        assert classify_correlation(0.7, warn=0.4, breach=0.6) == "breach"
