"""Onboarding-state helpers for the /welcome page.

A tenant is "fresh" when they have no personas, no broker credentials,
and no notification webhook. The home page redirects fresh tenants to
``/welcome`` and the welcome page uses these checks to decide which
checklist steps are already done.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, select

from ..notifications import has_webhook
from ..storage.database import get_session
from ..storage.models import BrokerCredentialModel, PersonaModel


@dataclass(frozen=True, slots=True)
class OnboardingStatus:
    """Per-step state for the /welcome checklist."""

    has_broker_creds: bool
    has_persona: bool
    has_webhook: bool

    @property
    def is_fresh(self) -> bool:
        """A tenant is "fresh" when nothing on the checklist is done."""
        return not (self.has_broker_creds or self.has_persona or self.has_webhook)

    @property
    def is_complete(self) -> bool:
        return self.has_broker_creds and self.has_persona and self.has_webhook


async def get_onboarding_status(tenant_id: UUID) -> OnboardingStatus:
    """Snapshot the three onboarding signals for a tenant."""
    async with get_session() as session:
        persona_count = await session.scalar(
            select(func.count())
            .select_from(PersonaModel)
            .where(PersonaModel.tenant_id == tenant_id)
        ) or 0
        broker_count = await session.scalar(
            select(func.count())
            .select_from(BrokerCredentialModel)
            .where(BrokerCredentialModel.tenant_id == tenant_id)
        ) or 0
    return OnboardingStatus(
        has_broker_creds=int(broker_count) > 0,
        has_persona=int(persona_count) > 0,
        has_webhook=await has_webhook(tenant_id),
    )
