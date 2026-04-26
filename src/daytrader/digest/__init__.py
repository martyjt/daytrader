"""Daily tenant digest (Phase 13 — productionization).

A small module that compiles "what happened yesterday" for each tenant
and posts the summary through the same :class:`Notifier` that Phase 11
uses for plugin-error alerts. Goal: a friend who logs in once a week
still gets a 1-line nudge every morning telling them whether anything
happened on their personas while they were away.

Public surface:

* :func:`build_digest` — pure-ish: tenant_id + a window → digest text.
* :class:`DailyDigestWorker` — per-tenant worker; sleeps until the next
  08:00 local, then sends and reschedules.
* :class:`DailyDigestSupervisor` — :class:`BackgroundSupervisor` that
  keeps one worker per active tenant alive.
"""

from .service import DigestSummary, build_digest, format_digest
from .supervisor import DailyDigestSupervisor
from .worker import DailyDigestWorker

__all__ = [
    "DailyDigestSupervisor",
    "DailyDigestWorker",
    "DigestSummary",
    "build_digest",
    "format_digest",
]
