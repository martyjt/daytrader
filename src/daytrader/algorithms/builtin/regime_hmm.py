"""Regime-Switching Hidden Markov Model.

Detects market regimes (bull/bear/sideways) from returns, volatility,
and volume features using a Gaussian HMM.  Emits directional signals
based on the most probable current regime.

Features:
    - Log return
    - Short-term volatility (10-bar rolling std of log returns)
    - Long-term volatility (30-bar rolling std of log returns)
    - Volume ratio — current / 20-bar mean
    - Trend strength — linear regression slope of last 20 closes
    - High-low range ratio — (H-L) / C

Supports both walk-forward (explicit ``train()`` calls per fold) and
standalone backtest (auto-train on initial history).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal


_FEATURE_NAMES = [
    "log_return",
    "volatility_10",
    "volatility_30",
    "volume_ratio",
    "trend_strength",
    "hl_range_pct",
]


def _rolling_std_returns(closes: np.ndarray, period: int) -> np.ndarray:
    """Rolling std of log returns over *period* bars."""
    out = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return out
    log_ret = np.diff(np.log(np.maximum(closes, 1e-10)))
    for i in range(period, len(log_ret)):
        out[i + 1] = np.std(log_ret[i - period + 1 : i + 1])
    return out


def _trend_strength(closes: np.ndarray, period: int = 20) -> np.ndarray:
    """Normalised linear-regression slope over a rolling window."""
    out = np.full(len(closes), np.nan)
    if len(closes) < period:
        return out
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1 : i + 1]
        slope = np.sum((x - x_mean) * (window - window.mean())) / x_var
        # Normalise by price level
        mid = window.mean()
        out[i] = slope / mid if mid > 0 else 0.0
    return out


def _build_hmm_features(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Build feature matrix (n_bars, 6) from raw OHLCV arrays."""
    n = len(closes)
    safe_closes = np.maximum(closes, 1e-10)
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.diff(np.log(safe_closes))

    # Volume ratio: current / 20-bar rolling mean
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        mean_vol = np.mean(volumes[i - 20 : i])
        vol_ratio[i] = volumes[i] / mean_vol if mean_vol > 0 else 1.0

    features = np.column_stack([
        log_ret,
        _rolling_std_returns(closes, 10),
        _rolling_std_returns(closes, 30),
        vol_ratio,
        _trend_strength(closes, 20),
        np.where(closes > 0, (highs - lows) / closes, 0.0),
    ])
    return features


class RegimeHMMAlgorithm(Algorithm):
    """ML regime detector using a Gaussian Hidden Markov Model.

    In walk-forward mode, ``train()`` is called before each test fold.
    In standalone backtest mode with ``auto_train=True``, the model
    auto-trains on available history after warmup bars accumulate.
    """

    def __init__(
        self,
        *,
        n_regimes: int = 3,
        lookback: int = 60,
        retrain_interval: int = 20,
        covariance_type: str = "full",
        n_iter: int = 100,
        score_threshold: float = 0.6,
        auto_train: bool = True,
    ) -> None:
        self._n_regimes = n_regimes
        self._lookback = lookback
        self._retrain_interval = retrain_interval
        self._covariance_type = covariance_type
        self._n_iter = n_iter
        self._score_threshold = score_threshold
        self._auto_train = auto_train
        self._model: GaussianHMM | None = None
        self._scaler: StandardScaler | None = None
        self._is_trained: bool = False
        self._regime_labels: dict[int, str] = {}
        self._bars_since_train: int = 0

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="regime_hmm",
            name="Regime-Switching HMM",
            version="1.0.0",
            description=(
                "ML: Hidden Markov Model detecting market regimes "
                "(bull/bear/sideways) from returns, volatility, and "
                "volume features. Emits directional signals based on "
                "the most probable current regime."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam("n_regimes", "int", self._n_regimes, min=2, max=6, description="Number of HMM states"),
                AlgorithmParam("lookback", "int", self._lookback, min=30, max=500, description="Training lookback bars"),
                AlgorithmParam("retrain_interval", "int", self._retrain_interval, min=5, max=100, description="Bars between auto-retrain"),
                AlgorithmParam("score_threshold", "float", self._score_threshold, min=0.5, max=0.95, step=0.05, description="Min state probability to emit signal"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._lookback

    def train(self, data: pl.DataFrame) -> None:
        """Train the HMM from OHLCV data.

        Called by ``WalkForwardEngine`` before each test fold.
        """
        closes = data["close"].to_numpy().astype(float)
        highs = data["high"].to_numpy().astype(float)
        lows = data["low"].to_numpy().astype(float)
        volumes = data["volume"].to_numpy().astype(float)

        self._fit(closes, highs, lows, volumes)

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        if not self._is_trained:
            if self._auto_train:
                self._auto_train_from_context(ctx)
            if not self._is_trained:
                return None

        self._bars_since_train += 1

        # Periodic retrain in auto mode
        if (
            self._auto_train
            and self._bars_since_train >= self._retrain_interval
        ):
            self._auto_retrain_from_context(ctx)

        closes = ctx.history_arrays.get("close")
        highs = ctx.history_arrays.get("high")
        lows = ctx.history_arrays.get("low")
        volumes = ctx.history_arrays.get("volume")

        if closes is None or len(closes) < self._lookback:
            return None

        # Build features for recent window and predict regime
        features = _build_hmm_features(
            closes[-self._lookback :],
            highs[-self._lookback :],
            lows[-self._lookback :],
            volumes[-self._lookback :],
        )

        # Remove NaN warmup rows
        valid_mask = ~np.isnan(features).any(axis=1)
        valid_features = features[valid_mask]
        if len(valid_features) < 5:
            return None

        scaled = self._scaler.transform(valid_features)
        state_probs = self._model.predict_proba(scaled)
        current_probs = state_probs[-1]  # last bar
        predicted_state = int(np.argmax(current_probs))
        max_prob = float(current_probs[predicted_state])

        regime_label = self._regime_labels.get(predicted_state, "unknown")

        prob_map = {
            self._regime_labels.get(s, f"state_{s}"): float(current_probs[s])
            for s in range(self._n_regimes)
        }
        ctx.log("regime_hmm", regime=regime_label, max_probability=max_prob, **prob_map)

        if max_prob < self._score_threshold:
            return None

        score = self._regime_to_score(regime_label, max_prob)
        if score == 0.0:
            return None

        return ctx.emit(
            score=score,
            confidence=max_prob,
            reason=f"HMM regime={regime_label} (P={max_prob:.3f})",
            metadata={
                "regime": regime_label,
                "state_probabilities": prob_map,
                "transition_matrix": self._model.transmat_.tolist(),
                "regime_means": {
                    self._regime_labels.get(s, f"state_{s}"): self._model.means_[s].tolist()
                    for s in range(self._n_regimes)
                },
            },
        )

    # ----- internal -------------------------------------------------------

    def _fit(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> None:
        """Fit the HMM on raw OHLCV arrays."""
        features = _build_hmm_features(closes, highs, lows, volumes)
        valid_mask = ~np.isnan(features).any(axis=1)
        valid_features = features[valid_mask]

        if len(valid_features) < 30:
            return

        scaler = StandardScaler()
        scaled = scaler.fit_transform(valid_features)

        model = GaussianHMM(
            n_components=self._n_regimes,
            covariance_type=self._covariance_type,
            n_iter=self._n_iter,
            random_state=42,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model.fit(scaled)
            except Exception:
                return  # convergence failure — stay untrained

        self._model = model
        self._scaler = scaler
        self._is_trained = True
        self._bars_since_train = 0
        self._assign_regime_labels()

    def _assign_regime_labels(self) -> None:
        """Map arbitrary HMM state indices to semantic regime labels.

        States are sorted by their mean log_return feature
        (index 0 in the feature matrix): highest → bull,
        lowest → bear, middle → sideways.
        """
        means = self._model.means_  # (n_regimes, n_features)
        # log_return is feature index 0
        return_means = means[:, 0]
        sorted_states = np.argsort(return_means)

        labels: dict[int, str] = {}
        n = self._n_regimes
        if n == 2:
            labels[int(sorted_states[0])] = "bear"
            labels[int(sorted_states[1])] = "bull"
        elif n == 3:
            labels[int(sorted_states[0])] = "bear"
            labels[int(sorted_states[1])] = "sideways"
            labels[int(sorted_states[2])] = "bull"
        else:
            labels[int(sorted_states[0])] = "bear"
            labels[int(sorted_states[-1])] = "bull"
            for i in range(1, n - 1):
                labels[int(sorted_states[i])] = "sideways"
        self._regime_labels = labels

    def _regime_to_score(self, regime: str, probability: float) -> float:
        """Convert regime label + probability to a trading score."""
        if regime == "bull":
            return 0.5 + (probability - 0.5) * 1.0
        elif regime == "bear":
            return -(0.5 + (probability - 0.5) * 1.0)
        return 0.0  # sideways — no signal

    def _auto_train_from_context(self, ctx: AlgorithmContext) -> None:
        """Auto-train on the first half of available history."""
        closes = ctx.history_arrays.get("close")
        highs = ctx.history_arrays.get("high")
        lows = ctx.history_arrays.get("low")
        volumes = ctx.history_arrays.get("volume")

        if closes is None or len(closes) < self._lookback * 2:
            return

        half = len(closes) // 2
        self._fit(closes[:half], highs[:half], lows[:half], volumes[:half])

    def _auto_retrain_from_context(self, ctx: AlgorithmContext) -> None:
        """Retrain on recent lookback bars."""
        closes = ctx.history_arrays.get("close")
        highs = ctx.history_arrays.get("high")
        lows = ctx.history_arrays.get("low")
        volumes = ctx.history_arrays.get("volume")

        if closes is None or len(closes) < self._lookback:
            return

        self._fit(
            closes[-self._lookback :],
            highs[-self._lookback :],
            lows[-self._lookback :],
            volumes[-self._lookback :],
        )
