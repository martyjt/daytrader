"""PPO trading agent built on Stable-Baselines3.

Implements the standard ``Algorithm`` ABC so the rest of the platform —
the DAG composer, walk-forward engine, Strategy Lab, paper/live loop —
treats it like any other algorithm.

Design:

* ``train(data)`` wraps the provided OHLCV frame as a ``BacktestTradingEnv``
  and runs ``PPO.learn(total_timesteps=...)``. When the walk-forward
  engine calls ``train`` before each fold, the agent retrains cleanly.
* ``on_bar(ctx)`` loads the most recent window of features, calls
  ``policy.predict``, and emits a Signal in ``[-1, +1]``.
* Model state can be saved/loaded via ``save_checkpoint`` /
  ``load_checkpoint`` — walk-forward can snapshot fold-specific
  policies for later inspection.

Install ``pip install daytrader[rl]`` to enable. Without SB3 installed
the agent class still imports but raises a friendly error on first use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from .gym_env import _GYM_AVAILABLE, _DEFAULT_WINDOW, make_env
from ..builtin.torch_base import build_dl_feature_matrix


def _require_sb3():
    try:
        from stable_baselines3 import PPO  # type: ignore
        return PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is not installed — install the 'rl' extra: "
            "pip install daytrader[rl]"
        ) from exc


class PPOAgent(Algorithm):
    """PPO-based trading agent. Discrete {short/flat/long} by default."""

    def __init__(
        self,
        *,
        window: int = _DEFAULT_WINDOW,
        total_timesteps: int = 20_000,
        learning_rate: float = 3e-4,
        n_steps: int = 512,
        gamma: float = 0.99,
        reward_shaper: str = "pnl",
        continuous: bool = False,
        score_threshold: float = 0.55,
        auto_train: bool = True,
    ) -> None:
        self._window = window
        self._total_timesteps = total_timesteps
        self._learning_rate = learning_rate
        self._n_steps = n_steps
        self._gamma = gamma
        self._reward_shaper = reward_shaper
        self._continuous = continuous
        self._score_threshold = score_threshold
        self._auto_train = auto_train

        self._model: Any = None  # sb3.PPO instance once trained
        self._last_position: float = 0.0

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="ppo_agent",
            name="PPO Agent (RL)",
            version="1.0.0",
            description=(
                "Reinforcement-learning trading agent trained with PPO. "
                "Observation: 20-bar window of 13-feature DL matrix. "
                "Action: short / flat / long. Reward: shaper-configurable."
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
                    description="PPO Adam learning rate",
                ),
                AlgorithmParam(
                    "gamma", "float", self._gamma, min=0.5, max=0.999, step=0.01,
                    description="Discount factor",
                ),
                AlgorithmParam(
                    "reward_shaper", "str", self._reward_shaper,
                    choices=["pnl", "sharpe", "drawdown_penalized", "turnover_penalized"],
                    description="Reward function used during training",
                ),
                AlgorithmParam(
                    "score_threshold", "float", self._score_threshold,
                    min=0.5, max=0.9, step=0.01,
                    description="Min action-probability to emit a directional signal",
                ),
            ],
            author="Daytrader built-in (RL)",
        )

    def warmup_bars(self) -> int:
        return max(self._window, 30)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: pl.DataFrame) -> None:
        """Train the PPO policy on the given OHLCV window."""
        if not _GYM_AVAILABLE:
            return  # gracefully no-op if gymnasium isn't installed
        PPO = _require_sb3()

        if len(data) <= self._window + 20:
            return  # not enough data to make an env

        env = make_env(
            data=data,
            window=self._window,
            continuous=self._continuous,
            reward_shaper=self._reward_shaper,
        )
        self._model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self._learning_rate,
            n_steps=self._n_steps,
            gamma=self._gamma,
            verbose=0,
        )
        self._model.learn(total_timesteps=self._total_timesteps)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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

        # SB3 expects either a batched or unbatched observation — be explicit.
        action, _ = self._model.predict(obs, deterministic=True)

        position = self._action_to_position(action)
        if position == 0.0:
            self._last_position = 0.0
            return None  # flat — no signal

        score = float(np.clip(position, -1.0, 1.0))
        if score == self._last_position:
            return None  # don't spam identical signals
        self._last_position = score

        return ctx.emit(
            score=score,
            confidence=min(1.0, abs(score) + 0.2),
            reason=f"PPO policy → position {score:+.2f}",
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("PPOAgent.save_checkpoint called before train()")
        self._model.save(str(path))

    def load_checkpoint(self, path: str | Path) -> None:
        PPO = _require_sb3()
        self._model = PPO.load(str(path))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _action_to_position(self, action: Any) -> float:
        if self._continuous:
            arr = np.asarray(action, dtype=np.float32).flatten()
            return float(np.clip(arr[0], -1.0, 1.0))
        mapping = {0: -1.0, 1: 0.0, 2: 1.0}
        try:
            return mapping.get(int(np.asarray(action).flatten()[0]), 0.0)
        except (ValueError, TypeError):
            return 0.0
