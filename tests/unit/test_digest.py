"""Tests for the Phase 13 daily digest module.

Covers:
* :func:`build_digest` counts signals/fills/breaches inside the window
  and ignores rows outside it.
* :func:`format_digest` renders the canonical one-paragraph string.
* :class:`DailyDigestWorker` schedules to the next 08:00 local boundary,
  hands the digest to the active notifier, and dedupes per date.
* :class:`DailyDigestSupervisor` only spins up workers when
  ``daily_digest_enabled`` and ``background_workers_enabled`` are both
  on.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import UUID

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker

from daytrader.core.events.base import EventType
from daytrader.core.settings import get_settings
from daytrader.digest import (
    DailyDigestSupervisor,
    DailyDigestWorker,
    build_digest,
    format_digest,
)
from daytrader.digest.worker import parse_local_time
from daytrader.storage.models import (
    JournalEntryModel,
    PersonaModel,
    SignalModel,
    TenantModel,
)

TENANT_A = UUID("00000000-0000-0000-0000-0000000000a1")
TENANT_B = UUID("00000000-0000-0000-0000-0000000000a2")


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def seeded(session_factory):
    async with session_factory() as s:
        s.add(TenantModel(id=TENANT_A, name="alpha"))
        s.add(TenantModel(id=TENANT_B, name="bravo"))
        await s.commit()
    return session_factory


def _patch_get_session(monkeypatch, factory, *modules: str) -> None:
    @asynccontextmanager
    async def _get():
        async with factory() as s:
            try:
                yield s
            except Exception:
                await s.rollback()
                raise

    for mod in modules:
        monkeypatch.setattr(f"{mod}.get_session", _get)


def _persona(tenant_id: UUID, *, mode: str = "paper", equity: str = "1000") -> PersonaModel:
    return PersonaModel(
        tenant_id=tenant_id,
        name=f"p-{mode}",
        mode=mode,
        asset_class="crypto",
        base_currency="USDT",
        initial_capital=Decimal(equity),
        current_equity=Decimal(equity),
        risk_profile="balanced",
    )


# ---------------------------------------------------------------------------
# build_digest + format_digest
# ---------------------------------------------------------------------------


async def test_build_digest_counts_window_only(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.digest.service")

    window_end = datetime(2026, 4, 26, 8, 0, 0)
    inside = window_end - timedelta(hours=12)
    outside = window_end - timedelta(hours=48)

    async with seeded() as s:
        p_a = _persona(TENANT_A, mode="paper", equity="1500")
        p_b = _persona(TENANT_A, mode="backtest", equity="500")
        p_other = _persona(TENANT_B, mode="paper", equity="0")
        s.add_all([p_a, p_b, p_other])
        await s.commit()

        s.add(
            SignalModel(
                tenant_id=TENANT_A,
                persona_id=p_a.id,
                symbol_key="BTCUSDT@binance:1d",
                score=0.5,
                confidence=0.5,
                source="x",
                created_at=inside,
            )
        )
        s.add(
            SignalModel(
                tenant_id=TENANT_A,
                persona_id=p_a.id,
                symbol_key="BTCUSDT@binance:1d",
                score=0.5,
                confidence=0.5,
                source="x",
                created_at=outside,
            )
        )
        s.add(
            JournalEntryModel(
                tenant_id=TENANT_A,
                persona_id=p_a.id,
                event_type=EventType.ORDER_FILLED.value,
                summary="filled",
                created_at=inside,
            )
        )
        s.add(
            JournalEntryModel(
                tenant_id=TENANT_A,
                persona_id=p_a.id,
                event_type=EventType.RISK_BREACH.value,
                summary="breach",
                severity="critical",
                created_at=inside,
            )
        )
        # Other tenant's row must not leak in.
        s.add(
            SignalModel(
                tenant_id=TENANT_B,
                persona_id=p_other.id,
                symbol_key="ETHUSDT@binance:1d",
                score=0.5,
                confidence=0.5,
                source="x",
                created_at=inside,
            )
        )
        await s.commit()

    summary = await build_digest(TENANT_A, window_end=window_end)
    assert summary.tenant_id == TENANT_A
    assert summary.persona_count == 2
    assert summary.active_persona_count == 1  # only paper counts
    assert summary.total_equity == Decimal(2000)
    assert summary.signal_count == 1
    assert summary.fill_count == 1
    assert summary.risk_breach_count == 1


async def test_build_digest_zero_when_empty(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.digest.service")
    summary = await build_digest(
        TENANT_A, window_end=datetime(2026, 4, 26, 8, 0, 0)
    )
    assert summary.persona_count == 0
    assert summary.signal_count == 0
    assert summary.fill_count == 0
    assert summary.total_equity == Decimal(0)


def test_format_digest_no_personas():
    summary = type(
        "S",
        (),
        dict(
            tenant_id=TENANT_A,
            window_start=datetime(2026, 4, 25, 8, 0, 0),
            window_end=datetime(2026, 4, 26, 8, 0, 0),
            persona_count=0,
            active_persona_count=0,
            signal_count=0,
            fill_count=0,
            risk_breach_count=0,
            total_equity=Decimal(0),
        ),
    )()
    text = format_digest(summary)
    assert "no personas yet" in text
    assert "2026-04-26" in text


def test_format_digest_quiet_day():
    from daytrader.digest.service import DigestSummary

    summary = DigestSummary(
        tenant_id=TENANT_A,
        window_start=datetime(2026, 4, 25, 8, 0, 0),
        window_end=datetime(2026, 4, 26, 8, 0, 0),
        persona_count=2,
        active_persona_count=1,
        signal_count=0,
        fill_count=0,
        risk_breach_count=0,
        total_equity=Decimal("1234.56"),
    )
    text = format_digest(summary)
    assert "Quiet day" in text
    assert "1/2 personas active" in text
    assert "$1,234.56" in text


def test_format_digest_active_day():
    from daytrader.digest.service import DigestSummary

    summary = DigestSummary(
        tenant_id=TENANT_A,
        window_start=datetime(2026, 4, 25, 8, 0, 0),
        window_end=datetime(2026, 4, 26, 8, 0, 0),
        persona_count=3,
        active_persona_count=3,
        signal_count=42,
        fill_count=5,
        risk_breach_count=1,
        total_equity=Decimal(10000),
    )
    text = format_digest(summary)
    assert "42 signal(s)" in text
    assert "5 fill(s)" in text
    assert "1 risk breach(es)" in text


# ---------------------------------------------------------------------------
# DailyDigestWorker
# ---------------------------------------------------------------------------


def test_worker_seconds_until_next_run_today():
    """If we're before the cutoff, the next run is today."""
    fake_now = datetime(2026, 4, 26, 7, 0, 0)
    w = DailyDigestWorker(
        tenant_id=TENANT_A, hour_local=8, minute_local=0, clock=lambda: fake_now
    )
    assert w.seconds_until_next_run() == pytest.approx(3600.0)


def test_worker_seconds_until_next_run_tomorrow():
    """If we're past the cutoff, the next run rolls to tomorrow."""
    fake_now = datetime(2026, 4, 26, 9, 0, 0)
    w = DailyDigestWorker(
        tenant_id=TENANT_A, hour_local=8, minute_local=0, clock=lambda: fake_now
    )
    expected = (24 - 1) * 3600.0
    assert w.seconds_until_next_run() == pytest.approx(expected)


def test_worker_seconds_until_next_run_at_exact_boundary():
    """Exactly at 08:00 schedules tomorrow, not now."""
    fake_now = datetime(2026, 4, 26, 8, 0, 0)
    w = DailyDigestWorker(
        tenant_id=TENANT_A, hour_local=8, minute_local=0, clock=lambda: fake_now
    )
    assert w.seconds_until_next_run() == pytest.approx(86400.0)


async def test_worker_run_once_calls_notifier(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.digest.service")

    captured: list[tuple[UUID, str, str | None]] = []

    async def _notify(tid, message, *, dedupe_key=None):
        captured.append((tid, message, dedupe_key))

    fake_now = datetime(2026, 4, 26, 8, 0, 0)
    w = DailyDigestWorker(
        tenant_id=TENANT_A,
        clock=lambda: fake_now,
        notify=_notify,
    )
    text = await w.run_once()
    assert len(captured) == 1
    tid, message, dedupe_key = captured[0]
    assert tid == TENANT_A
    assert message == text
    assert dedupe_key == "digest:2026-04-26"


def test_parse_local_time_default():
    assert parse_local_time(None) == (8, 0)
    assert parse_local_time("") == (8, 0)


def test_parse_local_time_valid():
    assert parse_local_time("06:30") == (6, 30)
    assert parse_local_time("23:00") == (23, 0)


def test_parse_local_time_invalid_falls_back():
    assert parse_local_time("not-a-time") == (8, 0)


# ---------------------------------------------------------------------------
# DailyDigestSupervisor
# ---------------------------------------------------------------------------


async def test_supervisor_returns_empty_when_disabled(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.digest.supervisor")
    settings = get_settings()
    monkeypatch.setattr(settings, "daily_digest_enabled", False)

    sup = DailyDigestSupervisor()
    assert await sup._active_tenants() == set()


async def test_supervisor_lists_enabled_tenants(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.digest.supervisor")
    settings = get_settings()
    monkeypatch.setattr(settings, "daily_digest_enabled", True)

    sup = DailyDigestSupervisor()
    active = await sup._active_tenants()
    # Both seeded tenants default to background_workers_enabled=True.
    assert TENANT_A in active
    assert TENANT_B in active


async def test_supervisor_skips_tenants_with_workers_disabled(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.digest.supervisor")
    settings = get_settings()
    monkeypatch.setattr(settings, "daily_digest_enabled", True)

    async with seeded() as s:
        row = await s.get(TenantModel, TENANT_B)
        assert row is not None
        row.background_workers_enabled = False
        await s.commit()

    sup = DailyDigestSupervisor()
    active = await sup._active_tenants()
    assert active == {TENANT_A}


def test_supervisor_make_worker_uses_settings(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "daily_digest_local_time", "06:30")
    sup = DailyDigestSupervisor()
    worker = sup._make_worker(TENANT_A)
    assert isinstance(worker, DailyDigestWorker)
    assert worker.tenant_id == TENANT_A
    fake_now = datetime(2026, 4, 26, 5, 30, 0)
    assert worker.seconds_until_next_run(fake_now) == pytest.approx(3600.0)
