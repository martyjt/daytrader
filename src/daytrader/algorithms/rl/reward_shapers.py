"""Reward shapers for the trading gym env.

A reward shaper is a callable that maps one step of price/position/fee
context to a scalar reward. Different shapers encode different trading
objectives — raw PnL, risk-adjusted (Sharpe-like), turnover-penalized,
drawdown-penalized. Swap shapers to train agents against different
goals without touching env wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RewardContext:
    """Per-step context handed to every shaper.

    All fields are floats so shapers don't need numpy or decimal
    conversions. Positions are unit quantities (-1 short, 0 flat, +1 long
    for discrete actions; any value in [-1, +1] for continuous).
    """

    prev_price: float
    next_price: float
    prev_position: float
    new_position: float
    equity: float
    peak_equity: float
    fee_bps: float = 5.0  # round-trip fee in basis points when flipping position
    bar_return: float = 0.0  # derived convenience: (next_price/prev_price) - 1


class RewardShaper(Protocol):
    """Anything callable with ``(ctx) -> float``."""

    def __call__(self, ctx: RewardContext) -> float:
        ...


def pnl_reward(ctx: RewardContext) -> float:
    """Raw per-step position PnL minus trading cost on position changes.

    Positive when a long position captures an up-move or a short position
    captures a down-move. Penalized for the bps cost of flipping.
    """
    position_pnl = ctx.prev_position * ctx.bar_return
    turnover = abs(ctx.new_position - ctx.prev_position)
    cost = turnover * (ctx.fee_bps / 10000.0)
    return float(position_pnl - cost)


@dataclass
class SharpeReward:
    """Rolling differential Sharpe reward.

    Maintains an exponentially-weighted mean and variance of per-step
    returns; reward = incremental improvement in the running Sharpe
    approximation (Moody & Saffell's differential Sharpe ratio). Useful
    when the agent should prefer smooth equity curves to spiky ones.
    """

    eta: float = 0.01  # EW decay

    def __post_init__(self) -> None:
        self._mean: float = 0.0
        self._var: float = 0.0

    def reset(self) -> None:
        self._mean = 0.0
        self._var = 0.0

    def __call__(self, ctx: RewardContext) -> float:
        r = ctx.prev_position * ctx.bar_return - abs(
            ctx.new_position - ctx.prev_position
        ) * (ctx.fee_bps / 10000.0)
        # Moody & Saffell differential Sharpe
        delta_mean = r - self._mean
        delta_var = r * r - self._var
        denom = (self._var - self._mean ** 2) ** 1.5
        if denom <= 1e-9:
            self._mean += self.eta * delta_mean
            self._var += self.eta * delta_var
            return float(r)
        d = (self._var * delta_mean - 0.5 * self._mean * delta_var) / denom
        self._mean += self.eta * delta_mean
        self._var += self.eta * delta_var
        return float(d)


@dataclass
class DrawdownPenalizedReward:
    """PnL reward with an explicit drawdown penalty."""

    drawdown_lambda: float = 2.0

    def __call__(self, ctx: RewardContext) -> float:
        base = pnl_reward(ctx)
        if ctx.peak_equity <= 0:
            return float(base)
        drawdown = max(0.0, (ctx.peak_equity - ctx.equity) / ctx.peak_equity)
        return float(base - self.drawdown_lambda * drawdown)


@dataclass
class TurnoverPenalizedReward:
    """PnL minus an extra penalty scaling with position turnover magnitude."""

    turnover_lambda: float = 0.001  # on top of fee_bps

    def __call__(self, ctx: RewardContext) -> float:
        base = pnl_reward(ctx)
        turnover = abs(ctx.new_position - ctx.prev_position)
        return float(base - self.turnover_lambda * turnover)


REWARD_SHAPERS: dict[str, RewardShaper] = {
    "pnl": pnl_reward,
    "sharpe": SharpeReward(),
    "drawdown_penalized": DrawdownPenalizedReward(),
    "turnover_penalized": TurnoverPenalizedReward(),
}


def get_shaper(name: str) -> RewardShaper:
    """Resolve a reward shaper by name."""
    if name not in REWARD_SHAPERS:
        raise KeyError(
            f"Reward shaper {name!r} not found. "
            f"Available: {sorted(REWARD_SHAPERS)}"
        )
    return REWARD_SHAPERS[name]
