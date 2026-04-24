"""Personas page — create, manage, and compare bot identities.

This is the user's primary entry point. Every persona is an isolated
bot with its own portfolio, strategy binding, risk profile, and mode
(backtest / paper / live).
"""

from __future__ import annotations

from decimal import Decimal

from nicegui import ui

from ..services import create_persona, delete_persona, list_personas
from ..shell import page_layout, persona_card


@ui.page("/personas")
async def personas_page() -> None:
    page_layout("Personas")

    container = ui.row().classes("w-full gap-4 flex-wrap")

    async def refresh() -> None:
        container.clear()
        personas = await list_personas()
        with container:
            if not personas:
                with ui.column().classes("w-full items-center q-pa-xl"):
                    ui.icon("person_add", size="64px").classes("text-grey-6")
                    ui.label("No personas yet").classes(
                        "text-h6 text-grey-6 q-pb-xs"
                    )
                    ui.label("Click the button above to create one.").classes(
                        "text-body2 text-grey-7"
                    )
            else:
                for p in personas:
                    persona_card(p, on_delete=refresh, on_refresh=refresh)

    async def open_create_dialog() -> None:
        # Fetch saved strategies so the create dialog can bind to one.
        from ..services import list_strategies

        try:
            saved = await list_strategies()
        except Exception:
            saved = []
        strategy_options = {"": "— (no strategy)"} | {
            str(s.id): f"{s.name} · {s.symbol} {s.timeframe}"
            for s in saved
        }

        with ui.dialog() as dlg, ui.card().classes("w-[420px]"):
            ui.label("Create Persona").classes("text-h6 q-pb-sm")

            name = ui.input(
                "Name", placeholder="e.g. BTC Paper Tester", validation={
                    "Required": lambda v: bool(v and v.strip()),
                }
            )
            asset_class = ui.select(
                ["crypto", "equities", "forex", "commodities"],
                value="crypto",
                label="Asset Class",
            )
            base_currency = ui.select(
                ["USDT", "USD", "EUR", "BTC"],
                value="USDT",
                label="Base Currency",
            )
            capital = ui.number(
                "Initial Capital ($)", value=10000, min=1, format="%.2f"
            )
            risk = ui.select(
                ["conservative", "balanced", "aggressive"],
                value="balanced",
                label="Risk Profile",
            )

            # Optional binding to a saved strategy recipe. When set, the
            # persona's meta records the strategy id so downstream tools
            # (live loop, Strategy Lab quicklink) can pull the recipe.
            strategy_select = ui.select(
                options=strategy_options,
                value="",
                label="Saved strategy (optional)",
            )
            if saved:
                ui.label(
                    "Binding a strategy records the recipe on the persona. "
                    "Change later from the Strategies page."
                ).classes("text-caption text-grey-6")

            ui.separator().classes("q-my-sm")

            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=dlg.close).props("flat")

                async def _do_create() -> None:
                    if not name.value or not name.value.strip():
                        ui.notify("Name is required", type="negative")
                        return

                    strategy_id = strategy_select.value or None
                    meta: dict = {}
                    chosen = None
                    if strategy_id and saved:
                        from uuid import UUID

                        try:
                            uid = UUID(strategy_id)
                            chosen = next((s for s in saved if s.id == uid), None)
                        except Exception:
                            chosen = None
                        if chosen:
                            meta = {
                                "strategy_config_id": str(chosen.id),
                                "strategy_name": chosen.name,
                                "algo_id": chosen.algo_id,
                                "symbol": chosen.symbol,
                                "timeframe": chosen.timeframe,
                                "venue": chosen.venue,
                            }

                    # Use the richer service that supports meta.
                    from ...core.context import tenant_scope
                    from ...core.settings import get_settings
                    from ...storage.database import get_session
                    from ...storage.models import PersonaModel

                    tid = get_settings().default_tenant_id
                    async with get_session() as session:
                        with tenant_scope(tid):
                            row = PersonaModel(
                                tenant_id=tid,
                                name=name.value.strip(),
                                mode="paper",
                                asset_class=asset_class.value,
                                base_currency=base_currency.value,
                                initial_capital=Decimal(str(capital.value)),
                                current_equity=Decimal(str(capital.value)),
                                risk_profile=risk.value,
                                meta=meta,
                            )
                            session.add(row)
                            await session.commit()

                    dlg.close()
                    if chosen:
                        ui.notify(
                            f"Persona '{name.value}' created — bound to "
                            f"strategy '{chosen.name}'",
                            type="positive",
                        )
                    else:
                        ui.notify(
                            f"Persona '{name.value}' created",
                            type="positive",
                        )
                    await refresh()

                ui.button("Create", on_click=_do_create).props(
                    "color=primary unelevated"
                )

        dlg.open()

    # ---- Header row + create button --------------------------------------
    with ui.row().classes("w-full justify-between items-center q-pb-md"):
        ui.label("Your Personas").classes("text-h5")
        ui.button(
            "+ New Persona", icon="add", on_click=open_create_dialog
        ).props("color=primary unelevated")

    await refresh()
