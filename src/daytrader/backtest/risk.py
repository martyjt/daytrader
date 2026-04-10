"""Risk layers for the backtest engine.

Layer 1 — per-trade: ATR-based stop-loss and take-profit.
Layer 2 — per-persona: daily loss limit, max hold bars.

Risk parameters are loaded from ``config/default.yaml`` under the
``risk:`` key but can be overridden per-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np


@dataclass(frozen=True)
class RiskConfig:
    """Risk parameters for a backtest run."""

    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 4.0
    max_hold_bars: int = 500
    daily_loss_limit_pct: float = 5.0
    atr_period: int = 14
    enabled: bool = True

    @classmethod
    def from_yaml(cls) -> RiskConfig:
        """Load risk config from the default YAML config file."""
        try:
            from ..core.settings import get_yaml_config

            cfg = get_yaml_config()
            return cls(
                stop_loss_atr_mult=cfg.get(
                    "risk", "per_trade", "default_stop_loss_atr_mult", default=2.0
                ),
                take_profit_atr_mult=cfg.get(
                    "risk", "per_trade", "default_take_profit_atr_mult", default=4.0
                ),
                max_hold_bars=cfg.get(
                    "risk", "per_trade", "max_hold_bars", default=500
                ),
                daily_loss_limit_pct=cfg.get(
                    "risk", "per_persona", "default_daily_loss_limit_pct", default=5.0
                ),
            )
        except Exception:
            return cls()

    @classmethod
    def disabled(cls) -> RiskConfig:
        """Return a config with risk checks turned off."""
        return cls(enabled=False)


def compute_atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> float:
    """Compute the current ATR value from numpy arrays.

    Uses the standard Wilder smoothing: ATR is the exponential moving
    average of true range over ``period`` bars.

    Requires at least ``period + 1`` bars of data.  Returns 0.0 if
    insufficient data.
    """
    n = len(closes)
    if n < period + 1:
        return 0.0

    # True range for each bar (starting from index 1)
    tr = np.empty(n - 1)
    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i - 1] = max(hl, hc, lc)

    # Wilder smoothing: start with SMA, then EMA
    atr = float(np.mean(tr[:period]))
    alpha = 1.0 / period
    for i in range(period, len(tr)):
        atr = atr * (1 - alpha) + tr[i] * alpha

    return atr


def check_stop_loss(
    current_low: float,
    entry_price: float,
    atr: float,
    multiplier: float,
) -> bool:
    """Return True if the stop-loss level is breached (long position)."""
    if atr <= 0 or multiplier <= 0:
        return False
    stop_price = entry_price - atr * multiplier
    return current_low <= stop_price


def check_take_profit(
    current_high: float,
    entry_price: float,
    atr: float,
    multiplier: float,
) -> bool:
    """Return True if the take-profit level is breached (long position)."""
    if atr <= 0 or multiplier <= 0:
        return False
    tp_price = entry_price + atr * multiplier
    return current_high >= tp_price


def stop_loss_price(entry_price: float, atr: float, multiplier: float) -> float:
    """Compute the stop-loss exit price."""
    return entry_price - atr * multiplier


def take_profit_price(entry_price: float, atr: float, multiplier: float) -> float:
    """Compute the take-profit exit price."""
    return entry_price + atr * multiplier


class DailyPnLTracker:
    """Tracks intraday P&L and enforces a daily loss limit.

    Call ``update()`` on every bar.  It returns True if trading should
    be halted for the rest of the current day.  The halt resets when a
    new calendar day begins.
    """

    def __init__(self, initial_equity: float, daily_loss_limit_pct: float) -> None:
        self._daily_loss_limit_pct = daily_loss_limit_pct
        self._day_start_equity = initial_equity
        self._current_day: date | None = None
        self._halted = False

    @property
    def is_halted(self) -> bool:
        return self._halted

    def update(self, current_equity: float, bar_timestamp: datetime) -> bool:
        """Update tracker. Returns True if trading should be halted."""
        bar_date = bar_timestamp.date() if isinstance(bar_timestamp, datetime) else bar_timestamp

        # New day — reset
        if self._current_day is None or bar_date != self._current_day:
            self._current_day = bar_date
            self._day_start_equity = current_equity
            self._halted = False

        if self._halted:
            return True

        # Check daily loss
        if self._day_start_equity > 0:
            day_loss_pct = (
                (self._day_start_equity - current_equity) / self._day_start_equity * 100
            )
            if day_loss_pct >= self._daily_loss_limit_pct:
                self._halted = True

        return self._halted
