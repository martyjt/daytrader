"""OHLCV bars and timeframes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum

_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14_400,
    "1d": 86_400,
    "1w": 604_800,
}


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def seconds(self) -> int:
        return _SECONDS[self.value]


@dataclass(frozen=True, slots=True)
class Bar:
    """A single OHLCV bar. Prices are Decimal to avoid float drift."""

    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    @property
    def typical(self) -> Decimal:
        return (self.high + self.low + self.close) / Decimal(3)

    @property
    def range(self) -> Decimal:
        return self.high - self.low
