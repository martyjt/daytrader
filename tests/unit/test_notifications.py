"""Tests for the Phase 11 notifications module."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from daytrader.core import audit
from daytrader.core.crypto import reset_codec_cache
from daytrader.notifications import (
    NoopNotifier,
    SlackNotifier,
    ThrottledNotifier,
    WebhookError,
    clear_webhook_url,
    get_active_notifier,
    has_webhook,
    notify_active,
    resolve_webhook_url,
    save_webhook_url,
    send_test_message,
    set_active_notifier,
)
from daytrader.notifications.service import _validate_webhook_url
from daytrader.storage.models import AuditLogModel, TenantModel

TENANT_A = UUID("00000000-0000-0000-0000-000000000001")
TENANT_B = UUID("00000000-0000-0000-0000-000000000002")


@pytest.fixture(autouse=True)
def _force_codec(monkeypatch):
    """Use a deterministic Fernet key so encrypt/decrypt round-trips work."""
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("APP_ENCRYPTION_KEY", key)

    from daytrader.core import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    reset_codec_cache()
    yield
    settings_mod.get_settings.cache_clear()
    reset_codec_cache()


@pytest_asyncio.fixture
async def session_factory(engine):
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def seeded(session_factory):
    """Seed two tenants so service-layer queries hit real rows."""
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


def _silence_audit(monkeypatch) -> None:
    """Stop audit.record() from hitting an unrelated DB session in the unit tests."""
    monkeypatch.setattr(audit, "current_tenant_id", lambda: None)
    monkeypatch.setattr(audit, "current_user_id", lambda: None)


@pytest.fixture
def _clean_active_notifier():
    set_active_notifier(None)
    yield
    set_active_notifier(None)


# ---------------------------------------------------------------------------
# ThrottledNotifier
# ---------------------------------------------------------------------------


class _RecordingNotifier(NoopNotifier):
    def __init__(self):
        self.calls: list[tuple[UUID, str, str | None]] = []

    async def notify(self, tenant_id, message, *, dedupe_key=None):
        self.calls.append((tenant_id, message, dedupe_key))


async def test_throttle_passes_first_call_through():
    inner = _RecordingNotifier()
    n = ThrottledNotifier(inner, window_seconds=300, clock=lambda: 0.0)
    await n.notify(TENANT_A, "hi", dedupe_key="k")
    assert inner.calls == [(TENANT_A, "hi", "k")]


async def test_throttle_dedupes_within_window():
    inner = _RecordingNotifier()
    times = iter([0.0, 10.0, 60.0])  # all inside 300s window
    n = ThrottledNotifier(inner, window_seconds=300, clock=lambda: next(times))
    await n.notify(TENANT_A, "first", dedupe_key="k")
    await n.notify(TENANT_A, "second", dedupe_key="k")
    await n.notify(TENANT_A, "third", dedupe_key="k")
    assert len(inner.calls) == 1
    assert inner.calls[0][1] == "first"


async def test_throttle_releases_after_window():
    inner = _RecordingNotifier()
    times = iter([0.0, 301.0])
    n = ThrottledNotifier(inner, window_seconds=300, clock=lambda: next(times))
    await n.notify(TENANT_A, "a", dedupe_key="k")
    await n.notify(TENANT_A, "b", dedupe_key="k")
    assert [c[1] for c in inner.calls] == ["a", "b"]


async def test_throttle_keys_per_tenant():
    inner = _RecordingNotifier()
    times = iter([0.0, 10.0])
    n = ThrottledNotifier(inner, window_seconds=300, clock=lambda: next(times))
    await n.notify(TENANT_A, "a", dedupe_key="same")
    await n.notify(TENANT_B, "b", dedupe_key="same")
    assert len(inner.calls) == 2


async def test_throttle_keys_per_dedupe_key():
    inner = _RecordingNotifier()
    times = iter([0.0, 10.0])
    n = ThrottledNotifier(inner, window_seconds=300, clock=lambda: next(times))
    await n.notify(TENANT_A, "a", dedupe_key="k1")
    await n.notify(TENANT_A, "b", dedupe_key="k2")
    assert len(inner.calls) == 2


async def test_throttle_passes_through_when_no_dedupe_key():
    inner = _RecordingNotifier()
    n = ThrottledNotifier(inner, window_seconds=300, clock=lambda: 0.0)
    await n.notify(TENANT_A, "1")
    await n.notify(TENANT_A, "2")
    await n.notify(TENANT_A, "3")
    assert len(inner.calls) == 3


# ---------------------------------------------------------------------------
# SlackNotifier
# ---------------------------------------------------------------------------


async def test_slack_posts_to_resolved_url():
    captured: list[tuple[str, str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append((str(request.url), request.read().decode()))
        return httpx.Response(200)

    transport = httpx.MockTransport(_handler)

    async def _resolver(_tid):
        return "https://hooks.example/abc"

    n = SlackNotifier(
        url_resolver=_resolver,
        client_factory=lambda: httpx.AsyncClient(transport=transport, timeout=5.0),
    )
    await n.notify(TENANT_A, "boom")

    assert len(captured) == 1
    url, body = captured[0]
    assert url == "https://hooks.example/abc"
    assert "boom" in body
    assert "text" in body


async def test_slack_silent_when_no_url():
    posted: list[str] = []

    def _handler(request):
        posted.append(str(request.url))
        return httpx.Response(200)

    transport = httpx.MockTransport(_handler)

    async def _resolver(_tid):
        return None

    n = SlackNotifier(
        url_resolver=_resolver,
        client_factory=lambda: httpx.AsyncClient(transport=transport, timeout=5.0),
    )
    await n.notify(TENANT_A, "boom")
    assert posted == []


async def test_slack_swallows_http_errors(caplog):
    def _handler(_request):
        return httpx.Response(500, text="server error")

    transport = httpx.MockTransport(_handler)

    async def _resolver(_tid):
        return "https://hooks.example/abc"

    n = SlackNotifier(
        url_resolver=_resolver,
        client_factory=lambda: httpx.AsyncClient(transport=transport, timeout=5.0),
    )
    with caplog.at_level(logging.WARNING, logger="daytrader.notifications.slack"):
        await n.notify(TENANT_A, "boom")
    assert any("HTTP 500" in r.message for r in caplog.records)


async def test_slack_swallows_resolver_failures(caplog):
    async def _resolver(_tid):
        raise RuntimeError("boom")

    n = SlackNotifier(url_resolver=_resolver)
    with caplog.at_level(logging.WARNING, logger="daytrader.notifications.slack"):
        await n.notify(TENANT_A, "msg")
    assert any("Failed to resolve webhook URL" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# resolve_webhook_url + has_webhook + save/clear
# ---------------------------------------------------------------------------


async def test_resolve_returns_none_when_unset(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.notifications.service")
    assert await resolve_webhook_url(TENANT_A) is None
    assert await has_webhook(TENANT_A) is False


async def test_resolve_returns_none_for_missing_tenant(seeded, monkeypatch):
    _patch_get_session(monkeypatch, seeded, "daytrader.notifications.service")
    assert await resolve_webhook_url(uuid4()) is None


async def test_save_then_resolve_roundtrip(seeded, monkeypatch):
    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.notifications.service",
        "daytrader.core.audit",
    )
    _silence_audit(monkeypatch)

    url = "https://hooks.slack.com/services/T0/B0/abc"
    await save_webhook_url(TENANT_A, url)

    assert await has_webhook(TENANT_A) is True
    assert await resolve_webhook_url(TENANT_A) == url

    # Ciphertext on disk is not the plaintext URL
    async with seeded() as s:
        row = (
            await s.execute(select(TenantModel).where(TenantModel.id == TENANT_A))
        ).scalar_one()
    assert row.notification_webhook_url is not None
    assert url not in row.notification_webhook_url


async def test_save_writes_audit_row(seeded, monkeypatch):
    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.notifications.service",
        "daytrader.core.audit",
    )
    _silence_audit(monkeypatch)

    await save_webhook_url(TENANT_A, "https://hooks.example/x")

    async with seeded() as s:
        rows = (
            await s.execute(
                select(AuditLogModel).where(
                    AuditLogModel.action == "notification.webhook.save"
                )
            )
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].tenant_id == TENANT_A
    assert rows[0].resource_type == "tenant"
    assert rows[0].resource_id == str(TENANT_A)


async def test_clear_resets_field_and_audits(seeded, monkeypatch):
    _patch_get_session(
        monkeypatch,
        seeded,
        "daytrader.notifications.service",
        "daytrader.core.audit",
    )
    _silence_audit(monkeypatch)

    await save_webhook_url(TENANT_A, "https://hooks.example/x")
    await clear_webhook_url(TENANT_A)

    assert await has_webhook(TENANT_A) is False
    assert await resolve_webhook_url(TENANT_A) is None

    async with seeded() as s:
        rows = (
            await s.execute(
                select(AuditLogModel).where(
                    AuditLogModel.action == "notification.webhook.clear"
                )
            )
        ).scalars().all()
    assert len(rows) == 1


async def test_save_rejects_non_https():
    with pytest.raises(WebhookError, match="https"):
        _validate_webhook_url("http://hooks.example/x")


async def test_save_rejects_empty():
    with pytest.raises(WebhookError, match="empty"):
        _validate_webhook_url("")


async def test_resolve_returns_none_when_codec_fails(seeded, monkeypatch, caplog):
    _patch_get_session(monkeypatch, seeded, "daytrader.notifications.service")
    # Stuff garbage into the field directly so decrypt fails.
    async with seeded() as s:
        row = (
            await s.execute(select(TenantModel).where(TenantModel.id == TENANT_A))
        ).scalar_one()
        row.notification_webhook_url = "not-a-fernet-token"
        await s.commit()

    with caplog.at_level(logging.WARNING, logger="daytrader.notifications.service"):
        assert await resolve_webhook_url(TENANT_A) is None
    assert any("Failed to decrypt" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# send_test_message (the admin UI's "Test" button)
# ---------------------------------------------------------------------------


async def test_send_test_message_posts():
    received: list[tuple[str, str]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        received.append((str(request.url), request.read().decode()))
        return httpx.Response(200)

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, timeout=5.0) as client:
        await send_test_message(
            "https://hooks.example/x",
            message="hello",
            client=client,
        )
    assert received[0][0] == "https://hooks.example/x"
    assert "hello" in received[0][1]


async def test_send_test_message_raises_on_http_error():
    def _handler(_request):
        return httpx.Response(403, text="forbidden")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, timeout=5.0) as client:
        with pytest.raises(WebhookError, match="HTTP 403"):
            await send_test_message("https://hooks.example/x", client=client)


async def test_send_test_message_raises_on_validation_error():
    with pytest.raises(WebhookError):
        await send_test_message("")


async def test_send_test_message_raises_on_transport_error():
    def _handler(_request):
        raise httpx.ConnectError("nope")

    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport, timeout=5.0) as client:
        with pytest.raises(WebhookError, match="reach"):
            await send_test_message("https://hooks.example/x", client=client)


# ---------------------------------------------------------------------------
# notify_active singleton
# ---------------------------------------------------------------------------


async def test_notify_active_no_op_when_unset(_clean_active_notifier):
    assert get_active_notifier() is None
    # Should not raise.
    await notify_active(TENANT_A, "msg")


async def test_notify_active_calls_installed(_clean_active_notifier):
    rec = _RecordingNotifier()
    set_active_notifier(rec)
    await notify_active(TENANT_A, "msg", dedupe_key="k")
    assert rec.calls == [(TENANT_A, "msg", "k")]


async def test_notify_active_swallows_exceptions(_clean_active_notifier, caplog):
    class _Boom(NoopNotifier):
        async def notify(self, *a, **kw):
            raise RuntimeError("boom")

    set_active_notifier(_Boom())
    with caplog.at_level(logging.WARNING, logger="daytrader.notifications.service"):
        await notify_active(TENANT_A, "msg")
    assert any("Notifier failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# SandboxedAlgorithm._record_error wiring
# ---------------------------------------------------------------------------


async def test_record_error_pushes_notification(_clean_active_notifier, monkeypatch):
    """SandboxedAlgorithm._record_error fires the active notifier with a
    plugin-error dedupe key."""
    from daytrader.algorithms.sandbox.adapter import SandboxedAlgorithm

    rec = _RecordingNotifier()
    set_active_notifier(rec)

    # Stub out the DB write side of _record_error; we only care about the notifier.
    async def _noop_record(**kw):
        return None

    monkeypatch.setattr(
        "daytrader.algorithms.sandbox.installer.record_plugin_error", _noop_record
    )

    # Build a minimally-initialised adapter — _record_error doesn't touch
    # anything besides _tenant_id, _algo_id, and the notifier.
    adapter = SandboxedAlgorithm.__new__(SandboxedAlgorithm)
    adapter._tenant_id = TENANT_A
    adapter._algo_id = "my_algo"

    await adapter._record_error("KeyError: 'foo'")

    assert len(rec.calls) == 1
    tid, msg, dedupe = rec.calls[0]
    assert tid == TENANT_A
    assert "my_algo" in msg
    assert "KeyError" in msg
    assert dedupe == "plugin_error:my_algo"
