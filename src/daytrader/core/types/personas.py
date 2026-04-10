"""Persona — a named bot identity with portfolio, strategy, risk profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID


class PersonaMode(StrEnum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    PAUSED = "paused"


class RiskProfile(StrEnum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


@dataclass
class Persona:
    """A named bot identity owned by a tenant.

    Personas run strategies against a portfolio in one of three modes:
    backtest (historical), paper (live data, fake money), or live
    (real broker). Every persona has an independent equity curve, risk
    profile, and activity journal.
    """

    id: UUID
    tenant_id: UUID
    name: str
    mode: PersonaMode
    asset_class: str
    base_currency: str
    initial_capital: Decimal
    current_equity: Decimal
    risk_profile: RiskProfile
    created_at: datetime
    updated_at: datetime
    strategy_id: UUID | None = None
    universe: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return self.mode == PersonaMode.LIVE

    @property
    def is_running(self) -> bool:
        return self.mode in (PersonaMode.PAPER, PersonaMode.LIVE)

    @property
    def unrealized_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return float((self.current_equity - self.initial_capital) / self.initial_capital * 100)
