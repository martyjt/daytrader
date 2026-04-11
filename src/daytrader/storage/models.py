"""SQLAlchemy 2.0 ORM models.

Every entity model uses ``TenantMixin`` for row-level tenant isolation.
Column types are cross-dialect (``Uuid`` maps to native UUID on Postgres,
CHAR(32) on SQLite) so unit tests can run against an in-memory SQLite.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    JSON,
    Numeric,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.orm import Mapped, declared_attr, mapped_column

from .database import Base


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class TenantMixin:
    """Adds ``tenant_id`` FK to every entity table."""

    @declared_attr
    @classmethod
    def tenant_id(cls) -> Mapped[UUID]:
        return mapped_column(Uuid, ForeignKey("tenants.id"), index=True)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


class TenantModel(Base):
    __tablename__ = "tenants"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserModel(TenantMixin, Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255))
    display_name: Mapped[str | None] = mapped_column(String(255), default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ---------------------------------------------------------------------------
# Strategy & Persona
# ---------------------------------------------------------------------------


class StrategyModel(TenantMixin, Base):
    __tablename__ = "strategies"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text, default="")
    config: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class PersonaModel(TenantMixin, Base):
    __tablename__ = "personas"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(255))
    mode: Mapped[str] = mapped_column(String(50), default="paper")
    asset_class: Mapped[str] = mapped_column(String(50), default="crypto")
    base_currency: Mapped[str] = mapped_column(String(10), default="USDT")
    initial_capital: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    current_equity: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    risk_profile: Mapped[str] = mapped_column(String(50), default="balanced")
    strategy_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("strategies.id"), default=None
    )
    universe: Mapped[list] = mapped_column(JSON, default=list)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


# ---------------------------------------------------------------------------
# Trading records
# ---------------------------------------------------------------------------


class SignalModel(TenantMixin, Base):
    __tablename__ = "signals"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    persona_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("personas.id"), index=True
    )
    symbol_key: Mapped[str] = mapped_column(String(100), index=True)
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(255))
    reason: Mapped[str] = mapped_column(Text, default="")
    attribution: Mapped[dict | None] = mapped_column(JSON, default=None)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class PositionModel(TenantMixin, Base):
    __tablename__ = "positions"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    persona_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("personas.id"), index=True
    )
    symbol_key: Mapped[str] = mapped_column(String(100), index=True)
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    avg_entry_price: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    realized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), default=Decimal(0)
    )
    unrealized_pnl: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), default=Decimal(0)
    )
    stop_loss: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), default=None)
    take_profit: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), default=None
    )
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    meta: Mapped[dict] = mapped_column(JSON, default=dict)


class OrderModel(TenantMixin, Base):
    __tablename__ = "orders"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    persona_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("personas.id"), index=True
    )
    symbol_key: Mapped[str] = mapped_column(String(100))
    side: Mapped[str] = mapped_column(String(10))
    order_type: Mapped[str] = mapped_column("type", String(20))
    quantity: Mapped[Decimal] = mapped_column(Numeric(20, 8))
    price: Mapped[Decimal | None] = mapped_column(Numeric(20, 8), default=None)
    stop_price: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), default=None
    )
    status: Mapped[str] = mapped_column(String(30), default="pending")
    filled_quantity: Mapped[Decimal] = mapped_column(
        Numeric(20, 8), default=Decimal(0)
    )
    avg_fill_price: Mapped[Decimal | None] = mapped_column(
        Numeric(20, 8), default=None
    )
    reason: Mapped[str] = mapped_column(Text, default="")
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


class JournalEntryModel(TenantMixin, Base):
    __tablename__ = "journal_entries"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    persona_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("personas.id"), index=True, default=None
    )
    event_type: Mapped[str] = mapped_column(String(50), index=True)
    severity: Mapped[str] = mapped_column(String(20), default="info")
    summary: Mapped[str] = mapped_column(String(500))
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


class BrokerCredentialModel(TenantMixin, Base):
    __tablename__ = "broker_credentials"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    broker_name: Mapped[str] = mapped_column(String(100))
    credential_data: Mapped[str] = mapped_column(Text)  # Fernet-encrypted JSON
    is_testnet: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
