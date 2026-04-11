"""Transformer Trend Classifier — attention-based deep learning model.

Self-attention encoder over a sliding window of 13 TA features, with
learnable positional encoding and mean pooling. Predicts next-bar
price direction. Supports walk-forward and auto-train modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base import AlgorithmManifest, AlgorithmParam
from .torch_base import N_FEATURES, TorchBaseAlgorithm


class _TransformerTrendModel(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 20,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, seq_len, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        x = self.input_proj(x) + self.pos_encoding[:, : x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)  # mean pool over sequence
        return self.head(x).squeeze(-1)


class TransformerTrendAlgorithm(TorchBaseAlgorithm):
    """Transformer encoder for next-bar direction prediction."""

    def __init__(
        self,
        *,
        lookback: int = 60,
        sequence_length: int = 20,
        forecast_horizon: int = 5,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 2,
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
            hidden_size=d_model,
            num_layers=num_encoder_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            score_threshold=score_threshold,
            auto_train=auto_train,
        )
        self._d_model = d_model
        self._nhead = nhead
        self._num_encoder_layers = num_encoder_layers

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="transformer_trend",
            name="Transformer Trend",
            version="1.0.0",
            description=(
                "DL: Transformer encoder predicting next-bar direction via "
                "self-attention over 13 TA features. Captures non-local "
                "temporal patterns without recurrence."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam("lookback", "int", 60, min=30, max=500, description="History lookback bars"),
                AlgorithmParam("sequence_length", "int", 20, min=5, max=60, description="Attention window length"),
                AlgorithmParam("forecast_horizon", "int", 5, min=1, max=20, description="Bars ahead to predict"),
                AlgorithmParam("d_model", "int", 32, min=16, max=128, step=8, description="Transformer model dimension"),
                AlgorithmParam("nhead", "int", 4, min=1, max=8, description="Attention heads"),
                AlgorithmParam("num_encoder_layers", "int", 2, min=1, max=4, description="Encoder layers"),
                AlgorithmParam("epochs", "int", 50, min=10, max=200, description="Training epochs"),
                AlgorithmParam("score_threshold", "float", 0.55, min=0.5, max=0.9, step=0.05, description="Min probability to emit signal"),
            ],
            author="Daytrader built-in",
        )

    def _create_model(self) -> nn.Module:
        return _TransformerTrendModel(
            n_features=N_FEATURES,
            d_model=self._d_model,
            nhead=self._nhead,
            num_encoder_layers=self._num_encoder_layers,
            dropout=self._dropout,
            seq_len=self._sequence_length,
        )
