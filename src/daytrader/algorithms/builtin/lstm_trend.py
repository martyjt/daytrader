"""LSTM Trend Classifier — deep learning sequence model.

Two-layer LSTM over a sliding window of 13 TA features, predicting
next-bar price direction. Supports walk-forward training and
auto-train modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base import AlgorithmManifest, AlgorithmParam
from .torch_base import N_FEATURES, TorchBaseAlgorithm


class _LSTMTrendModel(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        return self.head(last_hidden).squeeze(-1)


class LSTMTrendAlgorithm(TorchBaseAlgorithm):
    """LSTM sequence model for next-bar direction prediction."""

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="lstm_trend",
            name="LSTM Trend",
            version="1.0.0",
            description=(
                "DL: LSTM sequence model predicting next-bar direction from "
                "13 TA features over a sliding window. Captures temporal "
                "dependencies in price action and volume patterns."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam("lookback", "int", 60, min=30, max=500, description="History lookback bars"),
                AlgorithmParam("sequence_length", "int", 20, min=5, max=60, description="LSTM input window length"),
                AlgorithmParam("forecast_horizon", "int", 5, min=1, max=20, description="Bars ahead to predict"),
                AlgorithmParam("hidden_size", "int", 32, min=8, max=128, step=8, description="LSTM hidden units"),
                AlgorithmParam("num_layers", "int", 2, min=1, max=4, description="LSTM layers"),
                AlgorithmParam("epochs", "int", 50, min=10, max=200, description="Training epochs"),
                AlgorithmParam("score_threshold", "float", 0.55, min=0.5, max=0.9, step=0.05, description="Min probability to emit signal"),
            ],
            author="Daytrader built-in",
        )

    def _create_model(self) -> nn.Module:
        return _LSTMTrendModel(
            n_features=N_FEATURES,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dropout=self._dropout,
        )
