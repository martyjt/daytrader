"""Gymnasium env wrapping OHLCV bars as an RL trading environment.

Observation: sliding window of the Phase-5 ``build_dl_feature_matrix``
(13 columns) — so RL agents consume exactly the same features as the
LSTM / Transformer / CNN-LSTM supervised models. Sharing the feature
surface means we can compare supervised vs RL head-to-head.

Action: discrete {0=short, 1=flat, 2=long} by default. Agents that
prefer continuous sizing can swap in the Box variant via
``continuous=True``.

Reward: any ``RewardShaper`` from ``reward_shapers.py``. Default is raw
position PnL minus turnover costs.

Episode: a single pass through the provided ``pl.DataFrame``. Reset
restarts from the first bar after warmup. ``terminated=True`` at the
last bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

try:  # Lazy / optional — gymnasium may not be installed.
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _GYM_AVAILABLE = False

from ..builtin.torch_base import build_dl_feature_matrix
from .reward_shapers import RewardContext, RewardShaper, get_shaper


_DEFAULT_WINDOW = 20
_DEFAULT_FEATURE_DIM = 13  # matches build_dl_feature_matrix


def _require_gym() -> None:
    if not _GYM_AVAILABLE:
        raise ImportError(
            "gymnasium is not installed — install the 'rl' extra: "
            "pip install daytrader[rl]"
        )


if _GYM_AVAILABLE:

    class BacktestTradingEnv(gym.Env):
        """OHLCV → gym env. See module docstring for design."""

        metadata = {"render_modes": []}

        def __init__(
            self,
            data: pl.DataFrame,
            *,
            window: int = _DEFAULT_WINDOW,
            continuous: bool = False,
            initial_capital: float = 10_000.0,
            fee_bps: float = 5.0,
            reward_shaper: RewardShaper | str = "pnl",
        ) -> None:
            super().__init__()
            if len(data) <= window + 2:
                raise ValueError(
                    f"Need at least {window + 3} bars for a one-step env; got {len(data)}"
                )

            self._window = window
            self._continuous = continuous
            self._initial_capital = float(initial_capital)
            self._fee_bps = float(fee_bps)
            self._shaper = (
                get_shaper(reward_shaper) if isinstance(reward_shaper, str)
                else reward_shaper
            )

            closes = data["close"].to_numpy().astype(float)
            opens = data["open"].to_numpy().astype(float)
            highs = data["high"].to_numpy().astype(float)
            lows = data["low"].to_numpy().astype(float)
            volumes = data["volume"].to_numpy().astype(float)

            self._closes = closes
            features = build_dl_feature_matrix(closes, opens, highs, lows, volumes)
            # Replace NaN with 0 so the agent doesn't explode on warmup bars.
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            self._features = features.astype(np.float32)

            # Spaces
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(window, self._features.shape[1]),
                dtype=np.float32,
            )
            if continuous:
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
                )
            else:
                self.action_space = spaces.Discrete(3)

            # Mutable state.
            self._idx = window
            self._position = 0.0
            self._equity = self._initial_capital
            self._peak_equity = self._initial_capital

        # ------------------------------------------------------------------
        # gym.Env interface
        # ------------------------------------------------------------------

        def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            super().reset(seed=seed)
            self._idx = self._window
            self._position = 0.0
            self._equity = self._initial_capital
            self._peak_equity = self._initial_capital
            if hasattr(self._shaper, "reset") and callable(self._shaper.reset):  # type: ignore[attr-defined]
                self._shaper.reset()  # type: ignore[attr-defined]
            return self._observation(), {}

        def step(
            self, action: int | np.ndarray,
        ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
            new_position = self._action_to_position(action)
            prev_price = float(self._closes[self._idx])
            next_idx = self._idx + 1
            next_price = float(self._closes[next_idx]) if next_idx < len(self._closes) else prev_price

            bar_return = (next_price / prev_price) - 1.0 if prev_price > 0 else 0.0

            # Update equity before reward so drawdown penalty sees the new value.
            position_pnl = self._position * bar_return
            self._equity *= (1.0 + position_pnl)
            self._peak_equity = max(self._peak_equity, self._equity)

            ctx = RewardContext(
                prev_price=prev_price,
                next_price=next_price,
                prev_position=self._position,
                new_position=new_position,
                equity=self._equity,
                peak_equity=self._peak_equity,
                fee_bps=self._fee_bps,
                bar_return=bar_return,
            )
            reward = float(self._shaper(ctx))

            # Debit fees from equity on turnover so the equity curve is
            # net of costs (matches how the reward is calculated).
            turnover = abs(new_position - self._position)
            self._equity *= (1.0 - turnover * (self._fee_bps / 10000.0))

            self._position = new_position
            self._idx = next_idx
            terminated = self._idx + 1 >= len(self._closes)
            truncated = False
            info = {
                "equity": self._equity,
                "position": self._position,
            }
            return self._observation(), reward, terminated, truncated, info

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _action_to_position(self, action: int | np.ndarray) -> float:
            if self._continuous:
                arr = np.asarray(action, dtype=np.float32).flatten()
                return float(np.clip(arr[0], -1.0, 1.0))
            # Discrete: 0 short, 1 flat, 2 long
            mapping = {0: -1.0, 1: 0.0, 2: 1.0}
            return mapping.get(int(action), 0.0)

        def _observation(self) -> np.ndarray:
            start = self._idx - self._window
            return self._features[start : self._idx].copy()


def make_env(
    data: pl.DataFrame,
    *,
    window: int = _DEFAULT_WINDOW,
    continuous: bool = False,
    initial_capital: float = 10_000.0,
    fee_bps: float = 5.0,
    reward_shaper: RewardShaper | str = "pnl",
) -> Any:
    """Factory helper that raises a friendly error if gymnasium is missing."""
    _require_gym()
    return BacktestTradingEnv(
        data=data,
        window=window,
        continuous=continuous,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        reward_shaper=reward_shaper,
    )
