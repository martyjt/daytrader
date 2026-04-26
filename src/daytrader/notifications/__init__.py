"""Tenant notifications (Phase 11 — productionization).

A small abstraction that lets the rest of the system fire human-readable
alerts at a tenant without caring how they're delivered. Slack incoming
webhooks are the only channel today; email or Discord can be swapped in
behind the same :class:`Notifier` ABC without touching callers.

Public surface:

* :class:`Notifier` — the ABC. ``await notifier.notify(tenant_id, msg)``.
* :class:`SlackNotifier` — posts to the per-tenant webhook URL stored on
  ``TenantModel.notification_webhook_url`` (encrypted at rest).
* :class:`NoopNotifier` — silent, used by tests and when the active
  notifier is intentionally unset.
* :class:`ThrottledNotifier` — wraps another notifier and suppresses
  duplicate ``(tenant_id, dedupe_key)`` notifications within a window.
* :func:`set_active_notifier` / :func:`get_active_notifier` —
  process-global singleton, installed at startup.
"""

from .base import Notifier
from .noop import NoopNotifier
from .service import (
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
from .slack import SlackNotifier
from .throttle import ThrottledNotifier

__all__ = [
    "NoopNotifier",
    "Notifier",
    "SlackNotifier",
    "ThrottledNotifier",
    "WebhookError",
    "clear_webhook_url",
    "get_active_notifier",
    "has_webhook",
    "notify_active",
    "resolve_webhook_url",
    "save_webhook_url",
    "send_test_message",
    "set_active_notifier",
]
