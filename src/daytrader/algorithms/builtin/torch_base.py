"""Shared base class for PyTorch deep learning algorithms.

Provides:
- 13-feature pipeline reusing indicators.py functions
- Sequence windowing for temporal models
- Training loop with early stopping and StandardScaler
- Auto-train from AlgorithmContext
- Model checkpoint save/load
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm
from ..indicators import atr as _compute_atr
from ..indicators import ema as _compute_ema
from ..indicators import rsi as _compute_rsi
from ..indicators import stochastic as _compute_stochastic

_DL_FEATURE_NAMES = [
    "log_return",
    "returns_5",
    "returns_10",
    "returns_20",
    "rsi_14",
    "volatility_10",
    "volatility_20",
    "volume_ratio",
    "hl_range_ratio",
    "atr_ratio",
    "ema_diff_10",
    "ema_diff_30",
    "stochastic_14",
]

N_FEATURES = len(_DL_FEATURE_NAMES)


def _rolling_return(closes: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(closes), np.nan)
    out[period:] = (closes[period:] - closes[:-period]) / np.where(
        closes[:-period] != 0, closes[:-period], 1.0
    )
    return out


def _rolling_volatility(log_returns: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(log_returns), np.nan)
    if len(log_returns) < period + 1:
        return out
    for i in range(period, len(log_returns)):
        out[i] = np.std(log_returns[i - period + 1 : i + 1])
    return out


def build_dl_feature_matrix(
    closes: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
) -> np.ndarray:
    """Build (n_bars, 13) feature matrix for DL models."""
    n = len(closes)

    # Log returns
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(
        np.where(closes[:-1] > 0, closes[1:] / closes[:-1], 1.0)
    )

    # Rolling returns
    ret5 = _rolling_return(closes, 5)
    ret10 = _rolling_return(closes, 10)
    ret20 = _rolling_return(closes, 20)

    # RSI normalized to [0, 1]
    rsi_raw = _compute_rsi(closes, 14)
    rsi_norm = rsi_raw / 100.0

    # Volatility of log returns
    vol10 = _rolling_volatility(log_ret, 10)
    vol20 = _rolling_volatility(log_ret, 20)

    # Volume ratio
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        mean_vol = np.mean(volumes[i - 20 : i])
        vol_ratio[i] = volumes[i] / mean_vol if mean_vol > 0 else 1.0

    # High-low range ratio
    hl_ratio = np.where(closes > 0, (highs - lows) / closes, 0.0)

    # ATR ratio
    atr_vals = _compute_atr(highs, lows, closes, 14)
    atr_ratio = np.where(closes > 0, atr_vals / closes, 0.0)

    # EMA differences (normalized by close)
    ema10 = _compute_ema(closes, 10)
    ema30 = _compute_ema(closes, 30)
    ema_diff_10 = np.where(closes > 0, (closes - ema10) / closes, 0.0)
    ema_diff_30 = np.where(closes > 0, (closes - ema30) / closes, 0.0)

    # Stochastic
    stoch = _compute_stochastic(closes, 14)

    return np.column_stack([
        log_ret, ret5, ret10, ret20,
        rsi_norm, vol10, vol20, vol_ratio,
        hl_ratio, atr_ratio, ema_diff_10, ema_diff_30,
        stoch,
    ])


class TorchBaseAlgorithm(Algorithm):
    """Base class for PyTorch-based trading algorithms.

    Subclasses must implement ``_create_model()`` and ``manifest``.
    Everything else (feature engineering, training, inference, checkpointing)
    is handled here.
    """

    def __init__(
        self,
        *,
        lookback: int = 60,
        sequence_length: int = 20,
        forecast_horizon: int = 5,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        score_threshold: float = 0.55,
        auto_train: bool = True,
    ) -> None:
        self._lookback = lookback
        self._sequence_length = sequence_length
        self._forecast_horizon = forecast_horizon
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._patience = patience
        self._score_threshold = score_threshold
        self._auto_train = auto_train

        self._model: nn.Module | None = None
        self._scaler: StandardScaler | None = None
        self._is_trained: bool = False
        self._device = torch.device("cpu")

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Return the PyTorch model. Called at the start of training."""

    def warmup_bars(self) -> int:
        return self._lookback

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: pl.DataFrame) -> None:
        """Train from a Polars OHLCV DataFrame (walk-forward compatible)."""
        closes = data["close"].to_numpy().astype(float)
        opens = data["open"].to_numpy().astype(float)
        highs = data["high"].to_numpy().astype(float)
        lows = data["low"].to_numpy().astype(float)
        volumes = data["volume"].to_numpy().astype(float)

        features = build_dl_feature_matrix(closes, opens, highs, lows, volumes)
        labels = self._build_labels(closes)
        X, y = self._build_sequences(features, labels)

        if X is None or y is None or len(X) < 30:
            return

        self._train_model(X, y)

    def _build_labels(self, closes: np.ndarray) -> np.ndarray:
        """Binary labels: 1 if close[i + horizon] > close[i], else 0."""
        n = len(closes)
        labels = np.full(n, np.nan)
        h = self._forecast_horizon
        for i in range(n - h):
            labels[i] = 1.0 if closes[i + h] > closes[i] else 0.0
        return labels

    def _build_sequences(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Convert (n_bars, n_features) into (n_seq, seq_len, n_features) windows.

        Only includes windows where all features are valid (no NaN).
        If labels are provided, the label for each sequence is the label of
        the last bar in the window.
        """
        n = len(features)
        seq_len = self._sequence_length
        if n < seq_len:
            return None, None

        sequences = []
        seq_labels = []

        for i in range(seq_len, n + 1):
            window = features[i - seq_len : i]
            if np.isnan(window).any():
                continue
            if labels is not None:
                label_idx = i - 1  # label of last bar in window
                if np.isnan(labels[label_idx]):
                    continue
                seq_labels.append(labels[label_idx])
            sequences.append(window)

        if not sequences:
            return None, None

        X = np.array(sequences, dtype=np.float32)
        y = np.array(seq_labels, dtype=np.float32) if seq_labels else None
        return X, y

    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Full training loop with early stopping."""
        torch.manual_seed(42)

        # Fit scaler on training features (flatten sequences for fitting)
        self._scaler = StandardScaler()
        n_seq, seq_len, n_feat = X.shape
        X_flat = X.reshape(-1, n_feat)
        self._scaler.fit(X_flat)
        X_scaled = self._scaler.transform(X_flat).reshape(n_seq, seq_len, n_feat)

        # Chronological train/val split (80/20)
        split = int(len(X_scaled) * 0.8)
        if split < 10:
            split = len(X_scaled)  # skip validation if too few samples

        X_train = torch.tensor(X_scaled[:split], dtype=torch.float32)
        y_train = torch.tensor(y[:split], dtype=torch.float32)

        has_val = split < len(X_scaled)
        if has_val:
            X_val = torch.tensor(X_scaled[split:], dtype=torch.float32)
            y_val = torch.tensor(y[split:], dtype=torch.float32)

        # Create model
        self._model = self._create_model().to(self._device)
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._learning_rate,
            weight_decay=1e-5,
        )
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for _epoch in range(self._epochs):
            self._model.train()

            # Mini-batch training
            indices = torch.randperm(len(X_train))
            epoch_loss = 0.0

            for start in range(0, len(X_train), self._batch_size):
                batch_idx = indices[start : start + self._batch_size]
                xb = X_train[batch_idx].to(self._device)
                yb = y_train[batch_idx].to(self._device)

                optimizer.zero_grad()
                logits = self._model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation and early stopping
            if has_val:
                self._model.eval()
                with torch.no_grad():
                    val_logits = self._model(X_val.to(self._device))
                    val_loss = criterion(val_logits, y_val.to(self._device)).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.clone() for k, v in self._model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self._patience:
                        break

        # Restore best weights
        if best_state is not None:
            self._model.load_state_dict(best_state)

        self._model.eval()
        self._is_trained = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        if not self._is_trained:
            if self._auto_train:
                self._auto_train_from_context(ctx)
            if not self._is_trained:
                return None

        seq = self._extract_latest_sequence(ctx)
        if seq is None:
            return None

        # _is_trained ⇒ _model is set
        assert self._model is not None
        self._model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self._device)
            logit = self._model(input_tensor)
            up_prob = torch.sigmoid(logit).item()

        ctx.log(
            f"{self.manifest.id}_prediction",
            up_probability=up_prob,
        )

        # Not confident enough
        if abs(up_prob - 0.5) < (self._score_threshold - 0.5):
            return None

        score = (up_prob - 0.5) * 2
        score = max(-1.0, min(1.0, score))

        return ctx.emit(
            score=score,
            confidence=abs(up_prob - 0.5) * 2,
            reason=f"{self.manifest.name} P(up)={up_prob:.3f}",
            metadata={"up_probability": up_prob},
        )

    def _extract_latest_sequence(self, ctx: AlgorithmContext) -> np.ndarray | None:
        """Extract the most recent (seq_len, n_features) window from history."""
        closes = ctx.history_arrays["close"]
        opens = ctx.history_arrays["open"]
        highs = ctx.history_arrays["high"]
        lows = ctx.history_arrays["low"]
        volumes = ctx.history_arrays["volume"]

        if len(closes) < self._lookback:
            return None

        features = build_dl_feature_matrix(closes, opens, highs, lows, volumes)

        # Take the last seq_len rows
        if len(features) < self._sequence_length:
            return None

        window = features[-self._sequence_length :]
        if np.isnan(window).any():
            return None

        # Scale
        if self._scaler is not None:
            window = self._scaler.transform(window)

        return window.astype(np.float32)

    def _auto_train_from_context(self, ctx: AlgorithmContext) -> None:
        """Auto-train on the first half of available history."""
        closes = ctx.history_arrays["close"]
        opens = ctx.history_arrays["open"]
        highs = ctx.history_arrays["high"]
        lows = ctx.history_arrays["low"]
        volumes = ctx.history_arrays["volume"]

        if len(closes) < self._lookback * 2:
            return

        half = len(closes) // 2
        features = build_dl_feature_matrix(
            closes[:half], opens[:half], highs[:half], lows[:half], volumes[:half],
        )
        labels = self._build_labels(closes[:half])
        X, y = self._build_sequences(features, labels)

        if X is None or y is None or len(X) < 30:
            return

        self._train_model(X, y)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> dict[str, Any]:
        """Serialize model + scaler state for persistence."""
        if not self._is_trained or self._model is None or self._scaler is None:
            raise RuntimeError("Cannot save checkpoint: model not trained")
        return {
            "model_state_dict": {
                k: v.cpu().tolist() for k, v in self._model.state_dict().items()
            },
            "scaler_mean": self._scaler.mean_.tolist(),
            "scaler_scale": self._scaler.scale_.tolist(),
            "manifest_id": self.manifest.id,
            "sequence_length": self._sequence_length,
        }

    def load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore model and scaler from a checkpoint dict."""
        self._model = self._create_model().to(self._device)
        state_dict = {
            k: torch.tensor(v) for k, v in checkpoint["model_state_dict"].items()
        }
        self._model.load_state_dict(state_dict)
        self._model.eval()

        self._scaler = StandardScaler()
        self._scaler.mean_ = np.array(checkpoint["scaler_mean"])
        self._scaler.scale_ = np.array(checkpoint["scaler_scale"])
        self._is_trained = True
