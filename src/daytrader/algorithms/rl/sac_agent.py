"""SAC trading agent — continuous position sizing via Soft Actor-Critic.

Same interface as ``PPOAgent`` but outputs a continuous position in
``[-1, +1]`` instead of discrete long/flat/short. Useful when:

* You want graduated exposure — 0.3 long instead of "all-in long".
* The reward shaper benefits from a continuous action space
  (e.g. Sharpe reward discourages flipping between extremes).

SAC is off-policy and generally more sample-efficient than PPO for
continuous control, at the cost of a larger replay buffer in RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..builtin.torch_base import build_dl_feature_matrix
from .gym_env import _DEFAULT_WINDOW, _GYM_AVAILABLE, make_env


def _require_sac():
    try:
        from stable_baselines3 import SAC  # type: ignore
        return SAC
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is not installed — install the 'rl' extra: "
            "pip install daytrader[rl]"
        ) from exc


class SACAgent(Algorithm):
    """SAC-based trading agent with continuous position output."""

    def __init__(
        self,
        *,
        window: int = _DEFAULT_WINDOW,
        total_timesteps: int = 20_000,
        learning_rate: float = 3e-4,
        buffer_size: int = 50_000,
        batch_size: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        reward_shaper: str = "sharpe",
        min_position_magnitude: float = 0.1,
        auto_train: bool = True,
    ) -> None:
        self._window = window
        self._total_timesteps = total_timesteps
        self._learning_rate = learning_rate
        self._buffer_size = buffer_size
        self._batch_size = batch_size
        self._gamma = gamma
        self._tau = tau
        self._reward_shaper = reward_shaper
        self._min_position_magnitude = min_position_magnitude
        self._auto_train = auto_train

        self._model: Any = None
        self._last_position: float = 0.0

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="sac_agent",
            name="SAC Agent (RL, continuous)",
            version="1.0.0",
            description=(
                "Reinforcement-learning agent trained with Soft Actor-Critic. "
                "Outputs a continuous position in [-1, +1], letting the policy "
                "pick graduated exposure. Sample-efficient on long backtests."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    "window", "int", self._window, min=5, max=120,
                    description="Observation window (bars of features per step)",
                ),
                AlgorithmParam(
                    "total_timesteps", "int", self._total_timesteps,
                    min=1_000, max=1_000_000,
                    description="Training timesteps per train() call",
                ),
                AlgorithmParam(
                    "learning_rate", "float", self._learning_rate,
                    min=1e-5, max=1e-2, step=1e-4,
                    description="SAC Adam learning rate",
                ),
                AlgorithmParam(
                    "buffer_size", "int", self._buffer_size,
                    min=1_000, max=1_000_000,
                    description="Replay buffer size",
                ),
                AlgorithmParam(
                    "gamma", "float", self._gamma, min=0.5, max=0.999, step=0.01,
                    description="Discount factor",
                ),
                AlgorithmParam(
                    "reward_shaper", "str", self._reward_shaper,
                    choices=["pnl", "sharpe", "drawdown_penalized", "turnover_penalized"],
                    description="Reward function (sharpe recommended for continuous)",
                ),
                AlgorithmParam(
                    "min_position_magnitude", "float", self._min_position_magnitude,
                    min=0.0, max=0.5, step=0.05,
                    description="Absolute position below this emits no signal (flat)",
                ),
            ],
            author="Daytrader built-in (RL)",
        )

    def warmup_bars(self) -> int:
        return max(self._window, 30)

    def train(self, data: pl.DataFrame) -> None:
        if not _GYM_AVAILABLE:
            return
        SAC = _require_sac()
        if len(data) <= self._window + 20:
            return
        env = make_env(
            data=data,
            window=self._window,
            continuous=True,  # SAC always continuous
            reward_shaper=self._reward_shaper,
        )
        self._model = SAC(
            "MlpPolicy",
            env,
            learning_rate=self._learning_rate,
            buffer_size=self._buffer_size,
            batch_size=self._batch_size,
            gamma=self._gamma,
            tau=self._tau,
            verbose=0,
        )
        self._model.learn(total_timesteps=self._total_timesteps)

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        if self._model is None:
            return None

        closes = ctx.history_arrays["close"]
        if len(closes) < self._window + 1:
            return None

        opens = ctx.history_arrays["open"]
        highs = ctx.history_arrays["high"]
        lows = ctx.history_arrays["low"]
        volumes = ctx.history_arrays["volume"]

        features = build_dl_feature_matrix(closes, opens, highs, lows, volumes)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        obs = features[-self._window :].astype(np.float32)

        action, _ = self._model.predict(obs, deterministic=True)
        arr = np.asarray(action, dtype=np.float32).flatten()
        position = float(np.clip(arr[0], -1.0, 1.0))

        if abs(position) < self._min_position_magnitude:
            self._last_position = 0.0
            return None
        if abs(position - self._last_position) < 1e-3:
            return None  # dedupe identical sizing
        self._last_position = position

        return ctx.emit(
            score=position,
            confidence=min(1.0, abs(position) + 0.2),
            reason=f"SAC policy → position {position:+.3f}",
        )

    def save_checkpoint(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("SACAgent.save_checkpoint called before train()")
        self._model.save(str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        SAC = _require_sac()
        self._model = SAC.load(str(path))
