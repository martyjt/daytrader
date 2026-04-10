"""XGBoost Trend Classifier — first ML algorithm.

Predicts next-N-bar price direction using a gradient-boosted tree
trained on 7 technical features computed from raw OHLCV arrays.
Supports both walk-forward (explicit ``train()`` calls per fold)
and standalone backtest (auto-train on initial history).

Features:
    - Rolling returns (5, 10, 20 periods)
    - RSI(14)
    - Volatility(20) — rolling std of returns
    - Volume ratio — current / 20-bar mean
    - High-low range ratio — (H-L) / C

Labels:
    Binary classification: 1 if close[i + horizon] > close[i], else 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..indicators import rsi as _compute_rsi
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal


_FEATURE_NAMES = [
    "returns_5",
    "returns_10",
    "returns_20",
    "rsi_14",
    "volatility_20",
    "volume_ratio",
    "hl_range_ratio",
]


def _rolling_return(closes: np.ndarray, period: int) -> np.ndarray:
    """Compute rolling return over ``period`` bars."""
    out = np.full(len(closes), np.nan)
    out[period:] = (closes[period:] - closes[:-period]) / closes[:-period]
    return out


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI from close prices using Wilder smoothing."""
    return _compute_rsi(closes, period)


def _rolling_volatility(closes: np.ndarray, period: int = 20) -> np.ndarray:
    """Rolling standard deviation of returns."""
    out = np.full(len(closes), np.nan)
    if len(closes) < period + 1:
        return out
    returns = np.diff(closes) / closes[:-1]
    for i in range(period, len(returns)):
        out[i + 1] = np.std(returns[i - period + 1 : i + 1])
    return out


def _build_feature_matrix(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Build feature matrix (n_bars, 7) from raw OHLCV arrays."""
    n = len(closes)
    features = np.column_stack([
        _rolling_return(closes, 5),
        _rolling_return(closes, 10),
        _rolling_return(closes, 20),
        _rsi(closes, 14),
        _rolling_volatility(closes, 20),
        # Volume ratio: current / 20-bar rolling mean
        np.concatenate([
            np.full(20, np.nan),
            volumes[20:] / np.array([
                np.mean(volumes[i - 20:i]) if np.mean(volumes[i - 20:i]) > 0 else 1.0
                for i in range(20, n)
            ]),
        ]),
        # High-low range ratio
        np.where(closes > 0, (highs - lows) / closes, 0.0),
    ])
    return features


class XGBoostTrendAlgorithm(Algorithm):
    """ML trend classifier using XGBoost.

    In walk-forward mode, ``train()`` is called before each test fold.
    In standalone backtest mode with ``auto_train=True``, the model
    auto-trains on available history after warmup bars accumulate.
    """

    def __init__(
        self,
        *,
        lookback: int = 50,
        forecast_horizon: int = 5,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        score_threshold: float = 0.55,
        auto_train: bool = True,
    ) -> None:
        self._lookback = lookback
        self._forecast_horizon = forecast_horizon
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._score_threshold = score_threshold
        self._auto_train = auto_train
        self._model: xgb.XGBClassifier | None = None
        self._is_trained: bool = False

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="xgboost_trend",
            name="XGBoost Trend",
            version="1.0.0",
            description=(
                "ML: next-bar direction classifier trained on 7 TA "
                "features (rolling returns, RSI, volatility, volume, "
                "range). Emits signals when prediction confidence "
                "exceeds threshold."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam("lookback", "int", self._lookback, min=20, max=200, description="History lookback bars"),
                AlgorithmParam("forecast_horizon", "int", self._forecast_horizon, min=1, max=20, description="Bars ahead to predict"),
                AlgorithmParam("n_estimators", "int", self._n_estimators, min=10, max=500, description="XGBoost trees"),
                AlgorithmParam("score_threshold", "float", self._score_threshold, min=0.5, max=0.9, step=0.05, description="Min probability to emit signal"),
            ],
            author="Daytrader built-in",
        )

    def warmup_bars(self) -> int:
        return self._lookback

    def train(self, data: pl.DataFrame) -> None:
        """Train the XGBoost model from OHLCV data.

        Called by ``WalkForwardEngine`` before each test fold.
        """
        closes = data["close"].to_numpy().astype(float)
        highs = data["high"].to_numpy().astype(float)
        lows = data["low"].to_numpy().astype(float)
        volumes = data["volume"].to_numpy().astype(float)

        X, y = self._build_features_and_labels(closes, highs, lows, volumes)
        if len(X) < 30:
            return

        self._model = xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X, y)
        self._is_trained = True

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        if not self._is_trained:
            if self._auto_train:
                self._auto_train_from_context(ctx)
            if not self._is_trained:
                return None

        features = self._extract_features_from_context(ctx)
        if features is None:
            return None

        proba = self._model.predict_proba(features.reshape(1, -1))[0]
        up_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Log feature importance for explainability
        importance = dict(zip(_FEATURE_NAMES, self._model.feature_importances_))
        ctx.log("xgboost_prediction", up_probability=up_prob, **importance)

        # Not confident enough — skip
        if abs(up_prob - 0.5) < (self._score_threshold - 0.5):
            return None

        # Convert probability to score: [0,1] → [-1,1]
        score = (up_prob - 0.5) * 2
        score = max(-1.0, min(1.0, score))

        return ctx.emit(
            score=score,
            confidence=abs(up_prob - 0.5) * 2,
            reason=f"XGBoost P(up)={up_prob:.3f}",
            metadata={"feature_importance": importance},
        )

    # ----- internal -------------------------------------------------------

    def _build_features_and_labels(
        self,
        closes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build feature matrix X and label vector y."""
        features = _build_feature_matrix(closes, highs, lows, volumes)
        n = len(closes)
        horizon = self._forecast_horizon

        # Labels: 1 if close[i + horizon] > close[i], else 0
        labels = np.zeros(n)
        for i in range(n - horizon):
            labels[i] = 1.0 if closes[i + horizon] > closes[i] else 0.0

        # Only use rows where all features are valid and label is defined
        valid = ~np.isnan(features).any(axis=1)
        valid[n - horizon:] = False  # no label for last `horizon` bars

        X = features[valid]
        y = labels[valid]
        return X, y

    def _extract_features_from_context(
        self, ctx: AlgorithmContext
    ) -> np.ndarray | None:
        """Extract feature vector for the current bar from history arrays."""
        closes = ctx.history_arrays.get("close")
        highs = ctx.history_arrays.get("high")
        lows = ctx.history_arrays.get("low")
        volumes = ctx.history_arrays.get("volume")

        if closes is None or len(closes) < self._lookback:
            return None

        features = _build_feature_matrix(closes, highs, lows, volumes)
        row = features[-1]
        if np.isnan(row).any():
            return None
        return row

    def _auto_train_from_context(self, ctx: AlgorithmContext) -> None:
        """Auto-train on the first half of available history."""
        closes = ctx.history_arrays.get("close")
        highs = ctx.history_arrays.get("high")
        lows = ctx.history_arrays.get("low")
        volumes = ctx.history_arrays.get("volume")

        if closes is None or len(closes) < self._lookback * 2:
            return

        # Train on first half
        half = len(closes) // 2
        X, y = self._build_features_and_labels(
            closes[:half], highs[:half], lows[:half], volumes[:half],
        )
        if len(X) < 30:
            return

        self._model = xgb.XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            eval_metric="logloss",
            verbosity=0,
        )
        self._model.fit(X, y)
        self._is_trained = True
