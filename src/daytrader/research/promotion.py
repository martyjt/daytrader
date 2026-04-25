"""Discovery → StrategyConfig promotion.

When a user clicks "promote" on a significant Discovery, this module
turns the row into a tradeable artifact:

1. Loads the ``DiscoveryModel`` (tenant-scoped — no cross-tenant
   promotion possible).
2. Refuses to re-promote rows already marked ``"promoted"``.
3. Creates a new ``StrategyConfigModel`` bound to the
   ``feature_threshold`` baseline algorithm, with the discovered
   feature name plumbed through ``algo_params``.
4. Sets ``source_discovery_id`` so the trading loop knows to hydrate
   the feature each bar.
5. Flips ``discovery.status`` to ``"promoted"`` and stores the new
   strategy id in ``discovery.meta`` for traceability.

The result is a single ``StrategyConfigModel`` ready to be attached
to a persona via the existing ``persona.meta.strategy_config_id``
mechanism — promotion does **not** auto-bind to a persona; that's
left to the user.

The promoted threshold defaults are conservative:

* ``upper_threshold = 0`` for sentiment / cross_asset (positive value
  → long).
* For FRED series we still default to 0 — the user is expected to
  open the resulting StrategyConfig and tune. The point of promotion
  is artifact creation, not perfect tuning.

Threshold suggestions could one day come from the discovery's
distributional stats, but that's a separate refinement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from ..core.context import tenant_scope
from ..storage.database import get_session
from ..storage.models import DiscoveryModel, StrategyConfigModel
from ..storage.repository import TenantRepository

logger = logging.getLogger(__name__)


class PromotionError(Exception):
    """Raised when a Discovery cannot be promoted."""


@dataclass(frozen=True)
class PromotionResult:
    """Outcome of a successful ``promote_discovery`` call."""

    discovery_id: UUID
    strategy_config_id: UUID
    strategy_name: str
    algo_id: str
    feature_name: str


_BASELINE_ALGO = "feature_threshold"


async def promote_discovery(
    *,
    tenant_id: UUID,
    discovery_id: UUID,
) -> PromotionResult:
    """Promote a Discovery row into a tradeable ``StrategyConfigModel``.

    Tenant-scoped: the ``tenant_id`` argument is the authoritative
    boundary; if the discovery exists under a different tenant the
    repository's row-level filter returns ``None`` and we raise
    ``PromotionError`` rather than leaking it.

    Idempotency: re-promoting an already-promoted discovery raises
    ``PromotionError``. The user must dismiss-then-re-promote if they
    want a fresh artifact.
    """
    async with get_session() as session:
        with tenant_scope(tenant_id):
            disc_repo = TenantRepository(session, DiscoveryModel)
            discovery = await disc_repo.get(discovery_id)
            if discovery is None:
                raise PromotionError(
                    f"Discovery {discovery_id} not found in this tenant"
                )
            if discovery.status == "promoted":
                raise PromotionError(
                    f"Discovery {discovery_id} is already promoted "
                    f"(strategy_config_id={discovery.meta.get('strategy_config_id')})"
                )
            if not discovery.significant:
                # Allowed, but worth a warning — promoting a non-significant
                # discovery means trading on noise.
                logger.warning(
                    "Promoting non-significant discovery %s (lift=%.4f)",
                    discovery.id, discovery.lift,
                )

            algo_params = _build_algo_params(discovery)
            strategy_name = _strategy_name_for(discovery)

            strategy_repo = TenantRepository(session, StrategyConfigModel)
            strategy = await strategy_repo.create(
                name=strategy_name,
                description=(
                    f"Auto-generated from Discovery {discovery.id}. "
                    f"Source: {discovery.candidate_source}, "
                    f"feature: {discovery.candidate_name}, "
                    f"offline lift: {discovery.lift:+.4f}"
                ),
                algo_id=_BASELINE_ALGO,
                symbol=discovery.target_symbol,
                timeframe=discovery.target_timeframe,
                venue=_venue_for(discovery.target_symbol),
                algo_params=algo_params,
                tags=["discovery", discovery.candidate_source],
                source_discovery_id=discovery.id,
            )

            new_meta = dict(discovery.meta or {})
            new_meta["strategy_config_id"] = str(strategy.id)
            await disc_repo.update(
                discovery.id,
                status="promoted",
                meta=new_meta,
            )
            await session.commit()

            return PromotionResult(
                discovery_id=discovery.id,
                strategy_config_id=strategy.id,
                strategy_name=strategy_name,
                algo_id=_BASELINE_ALGO,
                feature_name=discovery.candidate_name,
            )


def _build_algo_params(discovery: DiscoveryModel) -> dict:
    """Assemble the algo_params dict for a feature_threshold config.

    Defaults are deliberately conservative — the user opens the new
    strategy config and tunes thresholds. The important piece is
    ``feature_name``, which links the config to the hydrated value.
    """
    direction = "both" if discovery.candidate_source == "cross_asset" else "long_only"
    return {
        "feature_name": discovery.candidate_name,
        "upper_threshold": 0.0,
        "lower_threshold": 0.0,
        "direction": direction,
    }


def _strategy_name_for(discovery: DiscoveryModel) -> str:
    """Build a human-readable name for the new StrategyConfig."""
    return f"Discovery: {discovery.candidate_name} ({discovery.target_symbol})"


def _venue_for(symbol: str) -> str:
    """Default venue when promoting — best-effort by symbol shape.

    Crypto pairs (``BTC/USDT``-style) → binance_spot; everything else
    → alpaca paper. Users can edit later.
    """
    if "/" in symbol:
        return "binance_spot"
    return "alpaca"
