"""Risk Center — 'Am I safe right now?'

Global kill switch, per-persona risk indicators, drawdown tracking.
Phase 0: kill switch + placeholder risk metrics.
"""

from __future__ import annotations

from nicegui import ui

from ..services import list_personas
from ..shell import page_layout, stat_card


@ui.page("/risk")
async def risk_center_page() -> None:
    if not page_layout("Risk Center"):
        return

    # ---- Kill switch banner ----------------------------------------------
    with ui.card().classes("w-full").style("background-color: #2d1111"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.column().classes("gap-0"):
                ui.label("Emergency Kill Switch").classes(
                    "text-h6 text-white"
                )
                ui.label(
                    "Immediately halt all live and paper trading across "
                    "every persona."
                ).classes("text-body2 text-grey-5")

            async def _on_kill() -> None:
                from ..services import kill_all_trading

                try:
                    count = await kill_all_trading(reason="manual")
                    ui.notify(
                        f"KILL SWITCH ACTIVATED — {count} persona(s) halted",
                        type="warning",
                        position="top",
                        timeout=5000,
                    )
                    ui.navigate.to("/risk")  # Refresh the page
                except RuntimeError:
                    ui.notify(
                        "Kill switch not ready",
                        type="negative",
                        position="top",
                    )

            ui.button("KILL ALL", color="negative", on_click=_on_kill).props(
                "size=lg unelevated"
            )

    # ---- Live broker balances --------------------------------------------
    ui.label("Live Broker Balances").classes("text-h6 q-pt-lg q-pb-sm")
    balances_container = ui.row().classes("w-full gap-4 q-mb-md")

    async def _refresh_balances() -> None:
        from ..services_brokers import live_broker_balances

        balances_container.clear()
        with balances_container:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="sm")
                ui.label("Fetching balances…").classes("text-caption text-grey-5")

        rows = await live_broker_balances()
        balances_container.clear()
        with balances_container:
            if not rows:
                ui.label(
                    "No broker credentials saved. Add API keys on the "
                    "Broker Keys page to see live balances here."
                ).classes("text-caption text-grey-6")
                return
            for row in rows:
                with ui.card().classes("q-pa-md min-w-[200px]"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon(
                            "account_balance_wallet",
                            color="positive" if row.ok else "negative",
                            size="sm",
                        )
                        ui.label(row.broker_name).classes(
                            "text-caption text-grey-5"
                        )
                        ui.badge(
                            "testnet" if row.is_testnet else "live",
                            color="warning" if row.is_testnet else "negative",
                        ).props("outline")
                    if row.ok:
                        ui.label(f"${float(row.balance):,.2f}").classes(
                            "text-h5 text-weight-bold q-pt-xs"
                        )
                    else:
                        ui.label(row.error or "error").classes(
                            "text-body2 text-negative q-pt-xs"
                        )

    ui.timer(0.5, _refresh_balances, once=True)

    # ---- Global risk metrics (placeholder) --------------------------------
    ui.label("Global Risk Metrics").classes("text-h6 q-pt-lg q-pb-sm")

    with ui.row().classes("w-full gap-4"):
        stat_card(
            "Max Drawdown", "0.0%", icon="trending_down", color="positive"
        )
        stat_card("Open Positions", "0", icon="swap_vert")
        stat_card("Daily P&L", "$0.00", icon="today")
        stat_card("Data Staleness", "OK", icon="wifi", color="positive")

    # ---- Per-persona risk status ------------------------------------------
    ui.separator().classes("q-my-md")
    ui.label("Per-Persona Risk Status").classes("text-h6 q-pb-sm")

    personas = await list_personas()
    if not personas:
        ui.label(
            "No personas running. Create one and start paper trading "
            "to see risk indicators here."
        ).classes("text-body2 text-grey-6")
    else:
        columns = [
            {"name": "name", "label": "Persona", "field": "name"},
            {"name": "mode", "label": "Mode", "field": "mode"},
            {"name": "equity", "label": "Equity", "field": "equity"},
            {"name": "risk", "label": "Risk Status", "field": "risk"},
        ]
        rows = [
            {
                "name": p.name,
                "mode": p.mode,
                "equity": f"${float(p.current_equity):,.2f}",
                "risk": "OK",
            }
            for p in personas
        ]
        ui.aggrid(
            {
                "columnDefs": columns,
                "rowData": rows,
                "domLayout": "autoHeight",
            }
        ).classes("w-full")

    # ---- Cross-persona correlation monitor --------------------------------
    ui.separator().classes("q-my-md")
    ui.label("Cross-Persona Signal Correlation").classes("text-h6 q-pb-xs")
    ui.label(
        "Supposedly independent personas should NOT have highly "
        "correlated signals. High correlation means hidden coupling or "
        "overfitting to the same feature. Warn: |ρ| ≥ 0.7 · "
        "Breach: |ρ| ≥ 0.9."
    ).classes("text-caption text-grey-6 q-pb-sm")

    corr_container = ui.column().classes("w-full")

    async def _run_correlation_scan() -> None:
        corr_container.clear()
        with corr_container:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="md")
                ui.label(
                    "Scanning cross-persona signal correlation (72h window)..."
                ).classes("text-grey-5")
        try:
            from ..services import run_correlation_scan_service

            report = await run_correlation_scan_service()
        except Exception as exc:  # noqa: BLE001
            corr_container.clear()
            with corr_container:
                ui.label(f"Correlation scan failed: {exc}").classes(
                    "text-negative"
                )
            return

        corr_container.clear()
        with corr_container:
            if not report.pairs:
                ui.label(
                    "Need at least 2 personas with recent signal activity. "
                    "Come back after running paper trading for a few hours."
                ).classes("text-grey-6 text-caption")
                return

            color = {"ok": "positive", "warn": "warning", "breach": "negative"}
            icon = {"ok": "check_circle", "warn": "warning", "breach": "error"}
            overall = report.overall_severity
            with ui.row().classes("items-center gap-2 q-pb-xs"):
                ui.icon(icon[overall], color=color[overall])
                ui.label(f"Overall: {overall.upper()}").classes(
                    f"text-body1 text-weight-medium text-{color[overall]}"
                )

            cols = [
                {"name": "a", "label": "Persona A", "field": "a"},
                {"name": "b", "label": "Persona B", "field": "b"},
                {"name": "corr", "label": "ρ", "field": "corr"},
                {"name": "abs", "label": "|ρ|", "field": "abs", "sortable": True},
                {"name": "n", "label": "Buckets", "field": "n"},
                {"name": "sev", "label": "Severity", "field": "sev"},
            ]
            rows = [
                {
                    "a": p.persona_a,
                    "b": p.persona_b,
                    "corr": f"{p.correlation:+.3f}",
                    "abs": f"{abs(p.correlation):.3f}",
                    "n": p.n_shared_buckets,
                    "sev": p.severity,
                }
                for p in report.pairs
            ]
            ui.aggrid({
                "columnDefs": cols,
                "rowData": rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"sortable": True, "resizable": True},
            }).classes("w-full")

    ui.button(
        "Scan Correlations",
        icon="hub",
        on_click=_run_correlation_scan,
    ).props("color=secondary outline size=sm").classes("q-mb-sm")

    # ---- Risk rules reference ---------------------------------------------
    ui.separator().classes("q-my-md")
    ui.label("Risk Rules (from config/default.yaml)").classes(
        "text-h6 q-pb-sm"
    )
    with ui.card().classes("w-full q-pa-md"):
        rules = [
            ("Per-trade stop-loss", "2.0× ATR"),
            ("Per-trade take-profit", "4.0× ATR"),
            ("Max hold duration", "500 bars"),
            ("Daily loss limit", "5.0% per persona"),
            ("Max open positions", "10 per persona"),
            ("Global max drawdown", "20.0%"),
            ("Data staleness alert", "120 seconds"),
        ]
        for rule, val in rules:
            with ui.row().classes("items-center justify-between w-full"):
                ui.label(rule).classes("text-body2")
                ui.badge(val, color="primary").props("outline")
