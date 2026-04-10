"""Strategy Lab — 'Build an algo, prove it, promote it.'

The Ritual lives here:
    Idea → Configure → Backtest → Walk-forward → Paper → Promote → Live

Phase 1: venue-specific fee modeling with gross vs net return breakdown,
fee drag tracking, and trade frequency warnings.
"""

from __future__ import annotations

from nicegui import ui

from ..shell import page_layout


@ui.page("/strategy-lab")
async def strategy_lab_page() -> None:
    page_layout("Strategy Lab")

    # ---- Ritual banner ---------------------------------------------------
    with ui.card().classes("w-full bg-dark q-mb-md"):
        ui.label(
            "The Ritual:  Idea → Configure → Backtest → Walk-forward "
            "→ Paper → Promote → Live"
        ).classes("text-subtitle2 text-grey-5")
        with ui.row().classes("gap-2 q-pt-xs"):
            for step, active in [
                ("Configure", True),
                ("Backtest", False),
                ("Walk-forward", False),
                ("Paper", False),
                ("Promote", False),
            ]:
                color = "primary" if active else "grey-8"
                ui.badge(step, color=color).props(
                    "outline" if not active else ""
                )

    # ---- Algorithm + venue lists from registries -------------------------
    from ...algorithms.registry import AlgorithmRegistry
    from ...backtest.fees import VENUE_PROFILES

    algo_ids = AlgorithmRegistry.available()
    algo_labels = {
        aid: AlgorithmRegistry.get(aid).manifest.name for aid in algo_ids
    }
    venue_labels = {k: v.venue for k, v in VENUE_PROFILES.items()}

    # ---- Configuration form ----------------------------------------------
    with ui.card().classes("w-full"):
        ui.label("Configure").classes("text-h6 q-pb-sm")

        with ui.row().classes("w-full gap-4 items-end"):
            algo = ui.select(
                options={aid: algo_labels[aid] for aid in algo_ids},
                value=algo_ids[0] if algo_ids else "",
                label="Algorithm",
            ).classes("min-w-[200px]")
            symbol = ui.input(
                "Symbol", value="BTC-USD", placeholder="e.g. BTC-USD, AAPL"
            )
            timeframe = ui.select(
                ["1d", "1h", "15m", "5m", "1w"], value="1d", label="Timeframe"
            )

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            start_date = ui.input("Start Date", value="2024-01-01")
            end_date = ui.input("End Date", value="2024-12-31")
            capital = ui.number("Capital ($)", value=10000, min=1)
            venue = ui.select(
                options=venue_labels,
                value="binance_spot",
                label="Venue / Fee Profile",
            ).classes("min-w-[200px]")

        # Show selected venue's fee breakdown
        venue_info = ui.label("").classes("text-caption text-grey-6 q-pt-xs")

        def _update_venue_info():
            v = venue.value
            if v and v in VENUE_PROFILES:
                s = VENUE_PROFILES[v]
                venue_info.text = (
                    f"Fees: {s.maker_bps}/{s.taker_bps} bps maker/taker · "
                    f"Spread: ~{s.spread_bps} bps · "
                    f"Slippage: ~{s.slippage_base_bps} bps ({s.slippage_model}) · "
                    f"Est. round-trip: ~{s.total_round_trip_bps:.0f} bps"
                )

        venue.on_value_change(lambda _: _update_venue_info())
        _update_venue_info()

    # ---- Results area ----------------------------------------------------
    results = ui.column().classes("w-full q-pt-md")

    async def run_backtest() -> None:
        results.clear()
        with results:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="lg")
                ui.label(
                    f"Fetching data and backtesting {symbol.value}..."
                ).classes("text-grey-5")

        try:
            from ..services import run_backtest_service

            result = await run_backtest_service(
                algo_id=algo.value,
                symbol_str=symbol.value,
                timeframe_str=timeframe.value,
                start_str=start_date.value,
                end_str=end_date.value,
                capital=float(capital.value or 10000),
                venue=venue.value,
            )
        except Exception as exc:
            results.clear()
            with results:
                ui.icon("error", size="lg", color="negative")
                ui.label(f"Backtest failed: {exc}").classes(
                    "text-negative text-body1"
                )
            return

        results.clear()
        with results:
            kpis = result.kpis

            # ---- Return comparison: gross vs net -------------------------
            ui.label("Backtest Results").classes("text-h6 q-pb-xs")
            ui.label(
                f"Venue: {result.venue} · "
                f"Fees paid: ${result.total_fees_paid:,.2f} · "
                f"Fee drag: {kpis.get('fee_drag_pct', 0):.2f}%"
            ).classes("text-body2 text-grey-5 q-pb-sm")

            # ---- KPI cards -----------------------------------------------
            net_ret = kpis.get("net_return_pct", 0)
            gross_ret = kpis.get("gross_return_pct", 0)

            with ui.row().classes("gap-3 q-pb-md flex-wrap"):
                _kpi_card(
                    "Gross Return",
                    f"{gross_ret:+.2f}%",
                    color="grey-5",
                )
                _kpi_card(
                    "Net Return",
                    f"{net_ret:+.2f}%",
                    color="positive" if net_ret >= 0 else "negative",
                )
                _kpi_card(
                    "Fee Drag",
                    f"-{kpis.get('fee_drag_pct', 0):.2f}%",
                    color="warning" if kpis.get("fee_drag_pct", 0) > 1 else "grey-5",
                )
                _kpi_card("Sharpe", f"{kpis.get('sharpe_ratio', 0):.2f}")
                _kpi_card(
                    "Max Drawdown",
                    f"{kpis.get('max_drawdown_pct', 0):.2f}%",
                    color="negative"
                    if kpis.get("max_drawdown_pct", 0) < -5
                    else "warning",
                )
                _kpi_card("Trades", str(kpis.get("num_trades", 0)))
                _kpi_card(
                    "Final Equity",
                    f"${result.final_equity:,.2f}",
                    color="positive"
                    if result.final_equity >= result.initial_capital
                    else "negative",
                )

            # ---- Trade frequency warning ---------------------------------
            atpd = kpis.get("avg_trades_per_day", 0)
            if atpd > 2:
                with ui.card().classes("w-full q-pa-sm").style(
                    "background-color: #3d2e00"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("warning", color="warning")
                        ui.label(
                            f"High trade frequency: {atpd:.1f} trades/day. "
                            f"At {result.venue} fee rates, this strategy needs "
                            f"significant daily alpha to overcome fee drag."
                        ).classes("text-body2 text-warning")

            # ---- Equity curve (ECharts) -----------------------------------
            eq = result.equity_curve
            if eq:
                ui.echart(
                    {
                        "backgroundColor": "transparent",
                        "tooltip": {
                            "trigger": "axis",
                            "formatter": "{b}<br/>Equity: ${c}",
                        },
                        "xAxis": {
                            "type": "category",
                            "data": list(range(len(eq))),
                            "name": "Bar",
                            "axisLabel": {"color": "#999"},
                        },
                        "yAxis": {
                            "type": "value",
                            "name": "Equity ($)",
                            "axisLabel": {"color": "#999"},
                        },
                        "series": [
                            {
                                "data": [round(v, 2) for v in eq],
                                "type": "line",
                                "smooth": True,
                                "showSymbol": False,
                                "areaStyle": {"opacity": 0.15},
                                "lineStyle": {"color": "#5c7cfa"},
                                "itemStyle": {"color": "#5c7cfa"},
                            }
                        ],
                        "grid": {
                            "left": "10%",
                            "right": "4%",
                            "top": "10%",
                            "bottom": "15%",
                        },
                    }
                ).classes("w-full").style("height: 420px")

            # ---- Summary line --------------------------------------------
            n_signals = len(result.signals)
            n_trades = kpis.get("num_trades", 0)
            ui.label(
                f"{n_signals} signals · {n_trades} trades · "
                f"{len(eq)} bars · ${result.total_fees_paid:,.2f} in fees"
            ).classes("text-caption text-grey-6 q-pt-sm")

    with ui.row().classes("q-pt-md"):
        ui.button(
            "Run Backtest", icon="play_arrow", on_click=run_backtest
        ).props("color=primary unelevated size=lg")


def _kpi_card(label: str, value: str, *, color: str = "primary") -> None:
    with ui.card().classes("q-pa-sm min-w-[110px]"):
        ui.label(label).classes("text-caption text-grey-6")
        ui.label(value).classes(f"text-h6 text-weight-bold text-{color}")
