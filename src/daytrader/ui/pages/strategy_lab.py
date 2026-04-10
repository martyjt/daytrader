"""Strategy Lab — 'Build an algo, prove it, promote it.'

The Ritual lives here:
    Idea → Configure → Backtest → Walk-forward → Paper → Promote → Live

Phase 0: form-based algo selection, symbol/timeframe/dates, Run
Backtest with real yfinance data + real backtest engine, real KPIs
and equity curve.
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

    # ---- Algorithm list from registry ------------------------------------
    from ...algorithms.registry import AlgorithmRegistry

    algo_ids = AlgorithmRegistry.available()
    algo_labels = {
        aid: AlgorithmRegistry.get(aid).manifest.name for aid in algo_ids
    }

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

            # ---- KPI cards -----------------------------------------------
            ui.label("Backtest Results").classes("text-h6 q-pb-sm")

            with ui.row().classes("gap-4 q-pb-md flex-wrap"):
                _kpi_card(
                    "Total Return",
                    f"{kpis.get('total_return_pct', 0):+.2f}%",
                    color="positive"
                    if kpis.get("total_return_pct", 0) >= 0
                    else "negative",
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
                    "Win Rate",
                    f"{kpis.get('win_rate_pct', 0):.1f}%",
                )
                _kpi_card(
                    "Final Equity",
                    f"${result.final_equity:,.2f}",
                    color="positive"
                    if result.final_equity >= result.initial_capital
                    else "negative",
                )

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

            # ---- Signals summary -----------------------------------------
            n_signals = len(result.signals)
            n_trades = kpis.get("num_trades", 0)
            ui.label(
                f"{n_signals} signals emitted · {n_trades} trades executed · "
                f"{len(eq)} bars processed"
            ).classes("text-caption text-grey-6 q-pt-sm")

    with ui.row().classes("q-pt-md"):
        ui.button(
            "Run Backtest", icon="play_arrow", on_click=run_backtest
        ).props("color=primary unelevated size=lg")


def _kpi_card(label: str, value: str, *, color: str = "primary") -> None:
    with ui.card().classes("q-pa-sm min-w-[120px]"):
        ui.label(label).classes("text-caption text-grey-6")
        ui.label(value).classes(f"text-h6 text-weight-bold text-{color}")
