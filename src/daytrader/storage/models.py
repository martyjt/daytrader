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
    Integer,
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
    background_workers_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserModel(TenantMixin, Base):
    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(255), default=None)
    password_hash: Mapped[str] = mapped_column(String(255), default="")
    role: Mapped[str] = mapped_column(String(20), default="member")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class UserInviteModel(Base):
    __tablename__ = "user_invites"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    tenant_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("tenants.id"), default=None, index=True
    )
    role: Mapped[str] = mapped_column(String(20), default="member")
    invited_by: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id"), default=None
    )
    used_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
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


class DiscoveryModel(TenantMixin, Base):
    """Exploration Agent output: one row per candidate feature tested.

    The agent writes every tested candidate (significant or not); the UI
    filters by ``significant`` + ``q_value`` for display. Storing the
    full set lets us audit negative results and tune the multiple-
    comparisons threshold after the fact.
    """

    __tablename__ = "discoveries"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    candidate_name: Mapped[str] = mapped_column(String(200), index=True)
    candidate_source: Mapped[str] = mapped_column(String(50), index=True)
    # e.g. "fred", "sentiment", "cross_asset"
    target_symbol: Mapped[str] = mapped_column(String(100), index=True)
    target_timeframe: Mapped[str] = mapped_column(String(10))

    baseline_metric: Mapped[float] = mapped_column(Float)
    candidate_metric: Mapped[float] = mapped_column(Float)
    lift: Mapped[float] = mapped_column(Float, index=True)
    p_value: Mapped[float | None] = mapped_column(Float, default=None)
    q_value: Mapped[float | None] = mapped_column(Float, default=None)
    significant: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    n_folds: Mapped[int] = mapped_column(Integer, default=5)

    status: Mapped[str] = mapped_column(String(30), default="new", index=True)
    # "new" | "promoted" | "dismissed"
    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class SymbolUniverseModel(TenantMixin, Base):
    """Named symbol universe reusable across Portfolio + Shadow workflows.

    A universe is just a list of symbols with a human name. Persisting
    them lets users curate watchlists ("Top 5 crypto", "Mega-cap tech",
    "My FX basket") and reuse them from any multi-symbol feature.
    """

    __tablename__ = "symbol_universes"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(120), index=True)
    symbols: Mapped[list] = mapped_column(JSON, default=list)
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class StrategyConfigModel(TenantMixin, Base):
    """Saved strategy configuration — algo + params + market bindings.

    Parallel to PersonaModel but lighter-weight: a persona binds a
    strategy to a capital pool and venue; a ``StrategyConfigModel`` is
    just the reusable recipe (algo, symbol, timeframe, params). Users
    can promote a saved config to any persona.
    """

    __tablename__ = "strategy_configs"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(120), index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    algo_id: Mapped[str] = mapped_column(String(100), index=True)
    symbol: Mapped[str] = mapped_column(String(100))
    timeframe: Mapped[str] = mapped_column(String(10))
    venue: Mapped[str] = mapped_column(String(50), default="binance_spot")
    algo_params: Mapped[dict] = mapped_column(JSON, default=dict)
    tags: Mapped[list] = mapped_column(JSON, default=list)
    # Set when this config was created by promoting a Discovery row.
    # Tells the trading loop to hydrate the discovered feature into
    # ``ctx.features`` each bar before invoking the algorithm.
    source_discovery_id: Mapped[UUID | None] = mapped_column(
        Uuid, ForeignKey("discoveries.id"), nullable=True, default=None, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ShadowRunModel(TenantMixin, Base):
    """Shadow tournament result — one row per candidate per run.

    Persists the history of shadow competitions so promotion decisions
    have an audit trail: which candidate beat the primary, on what
    symbol/timeframe/window, by how much, and with what walk-forward
    stability.
    """

    __tablename__ = "shadow_runs"

    id: Mapped[UUID] = mapped_column(Uuid, primary_key=True, default=uuid4)
    tournament_id: Mapped[UUID] = mapped_column(Uuid, index=True)
    primary_algo_id: Mapped[str] = mapped_column(String(100), index=True)
    candidate_algo_id: Mapped[str] = mapped_column(String(100), index=True)
    target_symbol: Mapped[str] = mapped_column(String(100), index=True)
    target_timeframe: Mapped[str] = mapped_column(String(10))
    window_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    window_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    sharpe: Mapped[float] = mapped_column(Float, default=0.0)
    net_return_pct: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)
    num_trades: Mapped[int] = mapped_column(Integer, default=0)
    stability_score: Mapped[float] = mapped_column(Float, default=0.0)
    # Fraction of walk-forward folds where the candidate beat the primary.

    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    beat_primary: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    promotion_status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    # "pending" | "promoted" | "dismissed"

    meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


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
