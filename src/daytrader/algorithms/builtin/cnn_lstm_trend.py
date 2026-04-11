"""CNN-LSTM Trend Classifier — hybrid deep learning model.

Two 1D convolutional layers extract local patterns from 13 TA features,
then an LSTM captures temporal dependencies. Predicts next-bar price
direction. Supports walk-forward and auto-train modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base import AlgorithmManifest, AlgorithmParam
from .torch_base import N_FEATURES, TorchBaseAlgorithm


class _CNNLSTMTrendModel(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        cnn_channels: int = 16,
        kernel_size: int = 3,
        hidden_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = x.transpose(1, 2)           # (batch, n_features, seq_len)
        x = self.conv_block(x)          # (batch, cnn_channels, seq_len)
        x = x.transpose(1, 2)           # (batch, seq_len, cnn_channels)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]           # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)


class CNNLSTMTrendAlgorithm(TorchBaseAlgorithm):
    """CNN-LSTM hybrid for next-bar direction prediction."""

    def __init__(
        self,
        *,
        lookback: int = 60,
        sequence_length: int = 20,
        forecast_horizon: int = 5,
        cnn_channels: int = 16,
        kernel_size: int = 3,
        hidden_size: int = 32,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        score_threshold: float = 0.55,
        auto_train: bool = True,
    ) -> None:
        super().__init__(
            lookback=lookback,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            score_threshold=score_threshold,
            auto_train=auto_train,
        )
        self._cnn_channels = cnn_channels
        self._kernel_size = kernel_size

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="cnn_lstm_trend",
            name="CNN-LSTM Trend",
            version="1.0.0",
            description=(
                "DL: 1D convolutional layers extract local patterns from 13 TA "
                "features, then an LSTM captures temporal dependencies. Combines "
                "pattern recognition with sequence modeling."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam("lookback", "int", 60, min=30, max=500, description="History lookback bars"),
                AlgorithmParam("sequence_length", "int", 20, min=5, max=60, description="Input window length"),
                AlgorithmParam("forecast_horizon", "int", 5, min=1, max=20, description="Bars ahead to predict"),
                AlgorithmParam("cnn_channels", "int", 16, min=8, max=64, step=8, description="Conv filter channels"),
                AlgorithmParam("hidden_size", "int", 32, min=8, max=128, step=8, description="LSTM hidden units"),
                AlgorithmParam("kernel_size", "int", 3, min=2, max=7, description="Conv kernel size"),
                AlgorithmParam("epochs", "int", 50, min=10, max=200, description="Training epochs"),
                AlgorithmParam("score_threshold", "float", 0.55, min=0.5, max=0.9, step=0.05, description="Min probability to emit signal"),
            ],
            author="Daytrader built-in",
        )

    def _create_model(self) -> nn.Module:
        return _CNNLSTMTrendModel(
            n_features=N_FEATURES,
            cnn_channels=self._cnn_channels,
            kernel_size=self._kernel_size,
            hidden_size=self._hidden_size,
            dropout=self._dropout,
        )
