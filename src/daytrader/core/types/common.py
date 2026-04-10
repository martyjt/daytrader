"""Shared scalar types and helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import NewType
from uuid import UUID

# Strongly-typed ID aliases. mypy enforces these; runtime is just UUID.
TenantId = NewType("TenantId", UUID)
UserId = NewType("UserId", UUID)
PersonaId = NewType("PersonaId", UUID)
StrategyId = NewType("StrategyId", UUID)
AlgorithmId = NewType("AlgorithmId", str)

# Money arithmetic must be Decimal — never float.
Money = Decimal


def utcnow() -> datetime:
    """Timezone-aware UTC now. Use this, never ``datetime.utcnow()``."""
    return datetime.now(timezone.utc)
