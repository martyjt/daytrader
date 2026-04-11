"""Global risk layer 3 — aggregate drawdown and data staleness monitoring.

Layer 1 (per-trade ATR stop-loss/take-profit) and Layer 2 (per-persona
daily loss limit) live in ``backtest/risk.py``.  This module adds the
global layer that spans all personas.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..core.types.common import utcnow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalRiskConfig:
    """Global risk parameters from ``config/default.yaml`` ``risk.global``."""

    kill_switch_enabled: bool = True
    max_drawdown_pct: float = 20.0
    data_staleness_seconds: int = 120

    @classmethod
    def from_yaml(cls) -> GlobalRiskConfig:
        """Load config from the default YAML file."""
        try:
            from ..core.settings import get_yaml_config

            cfg = get_yaml_config()
            return cls(
                kill_switch_enabled=cfg.get(
                    "risk", "global", "kill_switch_enabled", default=True
                ),
                max_drawdown_pct=cfg.get(
                    "risk", "global", "max_drawdown_pct", default=20.0
                ),
                data_staleness_seconds=cfg.get(
                    "risk", "global", "data_staleness_seconds", default=120
                ),
            )
        except Exception:
            return cls()


class GlobalRiskMonitor:
    """Layer 3: global risk across all personas.

    Tracks peak aggregate equity and detects when drawdown exceeds the
    configured threshold.  Also monitors the freshness of market data
    per symbol.

    Usage::

        monitor = GlobalRiskMonitor(config)
        if await monitor.check_drawdown(total_equity):
            # Drawdown breached — activate kill switch
            ...
        stale = await monitor.check_staleness()
    """

    def __init__(
        self,
        config: GlobalRiskConfig,
        on_breach: Callable[[str, dict], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._on_breach = on_breach
        self._peak_equity: float = 0.0
        self._last_data_timestamps: dict[str, datetime] = {}
        self._breached = False

    @property
    def config(self) -> GlobalRiskConfig:
        return self._config

    @property
    def is_breached(self) -> bool:
        return self._breached

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown percentage from peak (0.0 if no peak yet)."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._last_equity) / self._peak_equity * 100)

    async def check_drawdown(self, aggregate_equity: float) -> bool:
        """Update peak equity and check if max drawdown is exceeded.

        Returns True if breached.
        """
        self._last_equity = aggregate_equity

        if aggregate_equity > self._peak_equity:
            self._peak_equity = aggregate_equity

        if self._peak_equity <= 0:
            return False

        drawdown_pct = (self._peak_equity - aggregate_equity) / self._peak_equity * 100
        if drawdown_pct >= self._config.max_drawdown_pct:
            self._breached = True
            logger.warning(
                "Global drawdown breach: %.1f%% (limit %.1f%%)",
                drawdown_pct,
                self._config.max_drawdown_pct,
            )
            if self._on_breach:
                await self._on_breach(
                    "max_drawdown",
                    {
                        "drawdown_pct": drawdown_pct,
                        "peak_equity": self._peak_equity,
                        "current_equity": aggregate_equity,
                    },
                )
            return True
        return False

    def record_data_timestamp(self, symbol_key: str, timestamp: datetime) -> None:
        """Record the latest data timestamp for a symbol."""
        self._last_data_timestamps[symbol_key] = timestamp

    async def check_staleness(self) -> list[str]:
        """Return list of symbols with stale data (exceeding threshold)."""
        if not self._last_data_timestamps:
            return []

        stale: list[str] = []
        now = utcnow()
        threshold = timedelta(seconds=self._config.data_staleness_seconds)

        for symbol, ts in self._last_data_timestamps.items():
            if now - ts > threshold:
                stale.append(symbol)

        if stale:
            logger.warning("Stale data for symbols: %s", stale)
            if self._on_breach:
                await self._on_breach("data_staleness", {"stale_symbols": stale})

        return stale

    def reset(self) -> None:
        """Reset breach state (e.g. after manual resolution)."""
        self._breached = False
        self._peak_equity = 0.0
        self._last_data_timestamps.clear()
        self._last_equity = 0.0

    # Initialise the tracking attribute
    _last_equity: float = 0.0
