"""Realistic transaction cost model.

Fees are the #1 reason >99% of retail traders lose money. A backtest
that ignores spread, slippage, and venue-specific rates produces
dangerously optimistic results.

This module provides:

* ``FeeSchedule`` — a venue's fee structure (maker/taker, spread, slippage)
* ``FeeModel`` — computes per-trade and round-trip costs
* ``VENUE_PROFILES`` — real fee schedules for major venues (2026 rates)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeeSchedule:
    """Fee structure for a specific venue and tier.

    All rates are in **basis points** (1 bps = 0.01%).
    """

    venue: str
    maker_bps: float
    taker_bps: float
    spread_bps: float          # estimated average bid-ask spread
    slippage_model: str        # "fixed" | "volatility_scaled" | "volume_scaled"
    slippage_base_bps: float   # base slippage before scaling
    funding_rate_daily_bps: float = 0.0  # cost of holding leveraged positions
    min_trade_fee: float = 0.0           # minimum fee per trade (in $)

    @classmethod
    def from_flat_bps(cls, bps: float) -> FeeSchedule:
        """Create a flat fee schedule (backward compat with ``commission_bps``)."""
        return cls(
            venue="custom",
            maker_bps=bps,
            taker_bps=bps,
            spread_bps=0.0,
            slippage_model="fixed",
            slippage_base_bps=0.0,
            min_trade_fee=0.0,
        )

    @property
    def total_one_side_bps(self) -> float:
        """Estimated total one-side cost (taker + spread/2 + slippage base)."""
        return self.taker_bps + self.spread_bps / 2 + self.slippage_base_bps

    @property
    def total_round_trip_bps(self) -> float:
        """Estimated total round-trip cost (buy + sell)."""
        return self.total_one_side_bps * 2


class FeeModel:
    """Computes realistic transaction costs per trade.

    Usage::

        model = FeeModel(VENUE_PROFILES["binance_spot"])
        cost = model.trade_cost(10_000)           # $10k taker order
        cost = model.trade_cost(10_000, maker=True)  # limit order
        rt = model.round_trip_cost(10_000)        # buy + sell
    """

    def __init__(self, schedule: FeeSchedule) -> None:
        self.schedule = schedule

    def trade_cost(
        self,
        trade_value: float,
        *,
        maker: bool = False,
        volatility_pct: float = 0.0,
        volume_ratio: float = 0.0,
    ) -> float:
        """Total cost in dollars for a single side (buy or sell).

        Args:
            trade_value: dollar value of the trade
            maker: True if this is a limit order adding liquidity
            volatility_pct: current ATR as % of price (for slippage scaling)
            volume_ratio: order size / average bar volume (for impact scaling)
        """
        bps_to_frac = 1 / 10_000
        s = self.schedule

        # Commission
        rate = s.maker_bps if maker else s.taker_bps
        commission = trade_value * rate * bps_to_frac

        # Spread (half per side — you pay half the spread on entry, half on exit)
        spread = trade_value * s.spread_bps * bps_to_frac / 2

        # Slippage
        slippage = self._slippage(trade_value, volatility_pct, volume_ratio)

        total = commission + spread + slippage
        return max(total, s.min_trade_fee)

    def round_trip_cost(self, trade_value: float, **kwargs) -> float:
        """Total cost for buy + sell at the same trade value."""
        return self.trade_cost(trade_value, **kwargs) * 2

    def effective_round_trip_bps(self, trade_value: float = 10_000, **kwargs) -> float:
        """Round-trip cost expressed in basis points."""
        rt = self.round_trip_cost(trade_value, **kwargs)
        return rt / trade_value * 10_000

    def _slippage(
        self, trade_value: float, volatility_pct: float, volume_ratio: float
    ) -> float:
        s = self.schedule
        base = trade_value * s.slippage_base_bps / 10_000

        if s.slippage_model == "volatility_scaled" and volatility_pct > 0:
            # Higher volatility = wider effective spread = more slippage
            scale = 1 + volatility_pct / 2  # 2% vol → 2x slippage
            return base * scale
        elif s.slippage_model == "volume_scaled" and volume_ratio > 0:
            # Large orders relative to volume create market impact
            scale = 1 + math.sqrt(volume_ratio)
            return base * scale
        return base


# ---------------------------------------------------------------------------
# Pre-built venue profiles (2026 retail rates, standard tier)
# ---------------------------------------------------------------------------

VENUE_PROFILES: dict[str, FeeSchedule] = {
    "binance_spot": FeeSchedule(
        venue="Binance Spot",
        maker_bps=10,
        taker_bps=10,
        spread_bps=2,
        slippage_model="volatility_scaled",
        slippage_base_bps=3,
    ),
    "binance_futures": FeeSchedule(
        venue="Binance Futures",
        maker_bps=2,
        taker_bps=5,
        spread_bps=1,
        slippage_model="volatility_scaled",
        slippage_base_bps=2,
        funding_rate_daily_bps=9,  # ~0.03% per 8h × 3
    ),
    "binance_vip1": FeeSchedule(
        venue="Binance VIP 1",
        maker_bps=9,
        taker_bps=9,
        spread_bps=2,
        slippage_model="volatility_scaled",
        slippage_base_bps=3,
    ),
    "coinbase": FeeSchedule(
        venue="Coinbase",
        maker_bps=60,
        taker_bps=80,
        spread_bps=5,
        slippage_model="fixed",
        slippage_base_bps=5,
    ),
    "coinbase_advanced": FeeSchedule(
        venue="Coinbase Advanced",
        maker_bps=40,
        taker_bps=60,
        spread_bps=3,
        slippage_model="fixed",
        slippage_base_bps=3,
    ),
    "alpaca": FeeSchedule(
        venue="Alpaca (US Equities)",
        maker_bps=0,
        taker_bps=0,
        spread_bps=3,
        slippage_model="volatility_scaled",
        slippage_base_bps=2,
    ),
    "interactive_brokers": FeeSchedule(
        venue="Interactive Brokers",
        maker_bps=1,
        taker_bps=2,
        spread_bps=1,
        slippage_model="volume_scaled",
        slippage_base_bps=1,
        min_trade_fee=1.0,
    ),
    "kraken": FeeSchedule(
        venue="Kraken",
        maker_bps=16,
        taker_bps=26,
        spread_bps=3,
        slippage_model="fixed",
        slippage_base_bps=4,
    ),
    "zero_fees": FeeSchedule(
        venue="Zero Fees (unrealistic)",
        maker_bps=0,
        taker_bps=0,
        spread_bps=0,
        slippage_model="fixed",
        slippage_base_bps=0,
    ),
}
