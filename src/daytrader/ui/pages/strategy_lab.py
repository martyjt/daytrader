"""Strategy Lab — 'Build an algo, prove it, promote it.'

The Ritual lives here:
    Idea -> Configure -> Backtest -> Walk-forward -> Paper -> Promote -> Live

Phase 1: walk-forward testing, promotion gates, risk layers,
venue-specific fee modeling with gross vs net return breakdown.
"""

from __future__ import annotations

from nicegui import ui

from ..components.param_form import render_param_form
from ..shell import page_layout


@ui.page("/strategy-lab")
async def strategy_lab_page(strategy: str | None = None) -> None:
    page_layout("Strategy Lab")

    # ---- shared state -------------------------------------------------------
    _state: dict = {
        "backtest_result": None,
        "wf_result": None,
        "gate_results": [],
        "ritual_step": "configure",
    }
    param_values: dict = {}

    # ---- Optional: load saved strategy from ?strategy=<uuid> ----------------
    preload: dict = {}
    if strategy:
        try:
            from uuid import UUID

            from sqlalchemy import select

            from ...core.context import tenant_scope
            from ...core.settings import get_settings
            from ...storage.database import get_session
            from ...storage.models import StrategyConfigModel

            tid = get_settings().default_tenant_id
            async with get_session() as session:
                with tenant_scope(tid):
                    row = (await session.execute(
                        select(StrategyConfigModel)
                        .where(StrategyConfigModel.id == UUID(strategy))
                        .where(StrategyConfigModel.tenant_id == tid)
                    )).scalar_one_or_none()
                    if row:
                        preload = {
                            "name": row.name,
                            "algo_id": row.algo_id,
                            "symbol": row.symbol,
                            "timeframe": row.timeframe,
                            "venue": row.venue,
                            "algo_params": dict(row.algo_params or {}),
                        }
                        param_values.update(preload["algo_params"])
        except Exception:
            preload = {}

    # ---- Ritual banner ------------------------------------------------------
    ritual_badges: dict[str, ui.badge] = {}

    with ui.card().classes("w-full bg-dark q-mb-md"):
        ui.label(
            "The Ritual:  Idea -> Configure -> Backtest -> Walk-forward "
            "-> Paper -> Promote -> Live"
        ).classes("text-subtitle2 text-grey-5")
        with ui.row().classes("gap-2 q-pt-xs"):
            for step in ["Configure", "Backtest", "Walk-forward", "Paper", "Promote"]:
                active = step == "Configure"
                color = "primary" if active else "grey-8"
                b = ui.badge(step, color=color).props(
                    "outline" if not active else ""
                )
                ritual_badges[step.lower()] = b

    def _update_ritual_step(step: str) -> None:
        _state["ritual_step"] = step
        step_order = ["configure", "backtest", "walk-forward", "paper", "promote"]
        current_idx = step_order.index(step) if step in step_order else 0
        for i, s in enumerate(step_order):
            badge = ritual_badges.get(s)
            if badge:
                if i <= current_idx:
                    badge.props(remove="outline")
                    badge._props["color"] = "primary"
                else:
                    badge.props("outline")
                    badge._props["color"] = "grey-8"
                badge.update()

    # ---- Algorithm + venue lists from registries ----------------------------
    from ...algorithms.registry import AlgorithmRegistry
    from ...backtest.fees import VENUE_PROFILES

    algo_ids = AlgorithmRegistry.available()
    algo_labels = {
        aid: AlgorithmRegistry.get(aid).manifest.name for aid in algo_ids
    }
    venue_labels = {k: v.venue for k, v in VENUE_PROFILES.items()}

    # ---- Configuration form -------------------------------------------------
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center"):
            ui.label("Configure").classes("text-h6")
            if preload:
                ui.space()
                ui.badge(f"Loaded: {preload['name']}", color="positive").props("outline")

        default_algo = (
            preload.get("algo_id") if preload.get("algo_id") in algo_ids
            else (algo_ids[0] if algo_ids else "")
        )

        with ui.row().classes("w-full gap-4 items-end"):
            algo = ui.select(
                options={aid: algo_labels[aid] for aid in algo_ids},
                value=default_algo,
                label="Algorithm",
            ).classes("min-w-[200px]")
            symbol = ui.input(
                "Symbol",
                value=preload.get("symbol", "BTC-USD"),
                placeholder="e.g. BTC-USD, AAPL",
            )
            timeframe = ui.select(
                ["1d", "1h", "15m", "5m", "1w"],
                value=preload.get("timeframe", "1d"),
                label="Timeframe",
            ).classes("min-w-[140px]")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            start_date = ui.input("Start Date", value="2024-01-01")
            end_date = ui.input("End Date", value="2024-12-31")
            capital = ui.number("Capital ($)", value=10000, min=1)
            venue = ui.select(
                options=venue_labels,
                value=preload.get("venue", "binance_spot"),
                label="Venue / Fee Profile",
            ).classes("min-w-[200px]")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            risk_toggle = ui.switch("Enable Risk Layers (SL/TP)", value=False)

        # ---- Regime suitability warning --------------------------------------
        regime_warning = ui.row().classes("w-full q-pt-xs")

        async def _refresh_regime_warning() -> None:
            regime_warning.clear()
            algo_id = algo.value
            if not algo_id:
                return
            try:
                from ...algorithms.registry import AlgorithmRegistry

                suitable = AlgorithmRegistry.get(algo_id).manifest.suitable_regimes
            except Exception:
                suitable = None
            if not suitable:
                return  # algo is regime-agnostic
            try:
                from ..services_regime import get_current_regime

                snap = await get_current_regime()
            except Exception:
                return
            if snap.status != "ok":
                return
            if snap.regime in suitable:
                return
            top_pct = int(round(snap.probabilities.get(snap.regime, 0.0) * 100))
            with regime_warning:
                with ui.card().classes("w-full q-pa-sm").style(
                    "background-color: #3d2e00; border-left: 4px solid #f76707"
                ):
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon("warning", color="warning")
                        ui.label(
                            f"Regime mismatch: market looks {snap.regime.upper()} "
                            f"({top_pct}%) but this algorithm is designed for "
                            f"{', '.join(r.upper() for r in suitable)}. "
                            f"Expect reduced edge."
                        ).classes("text-body2 text-warning")

        # ---- Dynamic parameter form -----------------------------------------
        param_container = ui.column().classes("w-full q-pt-xs")

        def _on_algo_change(_):
            param_values.clear()
            render_param_form(algo.value, param_container, param_values)
            ui.timer(0.1, _refresh_regime_warning, once=True)

        algo.on_value_change(_on_algo_change)
        # Render initial form
        if algo.value:
            render_param_form(algo.value, param_container, param_values)
            ui.timer(0.3, _refresh_regime_warning, once=True)

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

    # ---- Results area -------------------------------------------------------
    results = ui.column().classes("w-full q-pt-md")

    # ---- Walk-forward results area ------------------------------------------
    wf_results = ui.column().classes("w-full")

    # ---- Gates area ---------------------------------------------------------
    gates_area = ui.column().classes("w-full")

    # ---- Backtest handler ---------------------------------------------------
    async def run_backtest() -> None:
        _state["backtest_result"] = None
        _state["wf_result"] = None
        _state["gate_results"] = []
        wf_results.clear()
        gates_area.clear()
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
                risk_enabled=risk_toggle.value,
                algo_params=dict(param_values) if param_values else None,
            )
        except Exception as exc:
            results.clear()
            with results:
                ui.icon("error", size="lg", color="negative")
                ui.label(f"Backtest failed: {exc}").classes(
                    "text-negative text-body1"
                )
            return

        _state["backtest_result"] = result
        _update_ritual_step("backtest")
        _render_backtest_results(result)

    def _render_backtest_results(result) -> None:
        results.clear()
        with results:
            kpis = result.kpis

            # ---- Return comparison: gross vs net ----------------------------
            ui.label("Backtest Results").classes("text-h6 q-pb-xs")
            ui.label(
                f"Venue: {result.venue} · "
                f"Fees paid: ${result.total_fees_paid:,.2f} · "
                f"Fee drag: {kpis.get('fee_drag_pct', 0):.2f}%"
            ).classes("text-body2 text-grey-5 q-pb-sm")

            # ---- Data-sufficiency warnings ---------------------------------
            for warning in getattr(result, "warnings", []) or []:
                with ui.card().classes("w-full q-pa-sm q-mb-sm").style(
                    "background-color: #3a2a1a; border-left: 4px solid #f76707"
                ):
                    with ui.row().classes("items-start gap-2 no-wrap"):
                        ui.icon("warning", color="warning", size="md")
                        ui.label(warning).classes("text-body2 text-grey-3")

            # ---- KPI cards --------------------------------------------------
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

            # ---- Risk events summary ----------------------------------------
            if result.risk_events:
                sl_count = sum(1 for e in result.risk_events if e["type"] == "stop_loss")
                tp_count = sum(1 for e in result.risk_events if e["type"] == "take_profit")
                with ui.row().classes("gap-2 q-pb-sm"):
                    if sl_count:
                        ui.badge(f"Stop-Loss: {sl_count}x", color="negative")
                    if tp_count:
                        ui.badge(f"Take-Profit: {tp_count}x", color="positive")

            # ---- Trade frequency warning ------------------------------------
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

            # ---- Equity curve (ECharts) -------------------------------------
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

            # ---- Summary line -----------------------------------------------
            n_signals = len(result.signals)
            n_trades = kpis.get("num_trades", 0)
            ui.label(
                f"{n_signals} signals · {n_trades} trades · "
                f"{len(eq)} bars · ${result.total_fees_paid:,.2f} in fees"
            ).classes("text-caption text-grey-6 q-pt-sm")

    # ---- Walk-forward handler -----------------------------------------------
    async def run_walk_forward() -> None:
        if _state["backtest_result"] is None:
            ui.notify("Run a backtest first", type="warning")
            return

        wf_results.clear()
        with wf_results:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="lg")
                ui.label("Running walk-forward analysis (5 folds)...").classes(
                    "text-grey-5"
                )

        try:
            from ..services import run_walk_forward_service

            wf = await run_walk_forward_service(
                algo_id=algo.value,
                symbol_str=symbol.value,
                timeframe_str=timeframe.value,
                start_str=start_date.value,
                end_str=end_date.value,
                capital=float(capital.value or 10000),
                venue=venue.value,
                risk_enabled=risk_toggle.value,
                algo_params=dict(param_values) if param_values else None,
            )
        except Exception as exc:
            wf_results.clear()
            with wf_results:
                ui.icon("error", size="lg", color="negative")
                ui.label(f"Walk-forward failed: {exc}").classes(
                    "text-negative text-body1"
                )
            return

        _state["wf_result"] = wf
        _update_ritual_step("walk-forward")
        _render_wf_results(wf)
        _evaluate_and_render_gates()

    def _render_wf_results(wf) -> None:
        wf_results.clear()
        with wf_results:
            ui.separator().classes("q-my-md")
            ui.label("Walk-Forward Results").classes("text-h6 q-pb-xs")

            # Aggregate OOS metrics
            with ui.row().classes("gap-3 q-pb-md flex-wrap"):
                _kpi_card(
                    "OOS Sharpe",
                    f"{wf.aggregate_oos_sharpe:.2f}",
                    color="positive" if wf.aggregate_oos_sharpe >= 0.3 else "warning",
                )
                _kpi_card(
                    "OOS Return",
                    f"{wf.aggregate_oos_return_pct:+.2f}%",
                    color="positive" if wf.aggregate_oos_return_pct >= 0 else "negative",
                )
                _kpi_card(
                    "OOS Max DD",
                    f"{wf.aggregate_oos_max_drawdown_pct:.2f}%",
                    color="negative"
                    if wf.aggregate_oos_max_drawdown_pct < -5
                    else "warning",
                )
                _kpi_card("Folds", str(len(wf.folds)))

            # Per-fold table
            columns = [
                {"name": "fold", "label": "Fold", "field": "fold"},
                {"name": "train", "label": "Train Period", "field": "train"},
                {"name": "test", "label": "Test Period", "field": "test"},
                {"name": "oos_sharpe", "label": "OOS Sharpe", "field": "oos_sharpe"},
                {"name": "oos_return", "label": "OOS Return", "field": "oos_return"},
            ]
            rows = [
                {
                    "fold": f.fold_index + 1,
                    "train": f"{str(f.train_start)[:10]} -> {str(f.train_end)[:10]}",
                    "test": f"{str(f.test_start)[:10]} -> {str(f.test_end)[:10]}",
                    "oos_sharpe": f"{f.oos_sharpe:.2f}",
                    "oos_return": f"{f.oos_return_pct:+.2f}%",
                }
                for f in wf.folds
            ]
            ui.aggrid(
                {
                    "columnDefs": columns,
                    "rowData": rows,
                    "domLayout": "autoHeight",
                }
            ).classes("w-full")

            # OOS equity curve
            if wf.oos_equity_curve:
                ui.echart(
                    {
                        "backgroundColor": "transparent",
                        "tooltip": {"trigger": "axis"},
                        "xAxis": {
                            "type": "category",
                            "data": list(range(len(wf.oos_equity_curve))),
                            "name": "OOS Bar",
                            "axisLabel": {"color": "#999"},
                        },
                        "yAxis": {
                            "type": "value",
                            "name": "OOS Equity ($)",
                            "axisLabel": {"color": "#999"},
                        },
                        "series": [
                            {
                                "data": [round(v, 2) for v in wf.oos_equity_curve],
                                "type": "line",
                                "smooth": True,
                                "showSymbol": False,
                                "areaStyle": {"opacity": 0.15},
                                "lineStyle": {"color": "#22b8cf"},
                                "itemStyle": {"color": "#22b8cf"},
                            }
                        ],
                        "grid": {
                            "left": "10%",
                            "right": "4%",
                            "top": "10%",
                            "bottom": "15%",
                        },
                    }
                ).classes("w-full").style("height: 300px")

    # ---- Gate evaluation ----------------------------------------------------
    def _evaluate_and_render_gates() -> None:
        from ..services import evaluate_gates_service

        gate_results = evaluate_gates_service(
            backtest_result=_state["backtest_result"],
            walk_forward_result=_state["wf_result"],
        )
        _state["gate_results"] = gate_results
        _render_gates(gate_results)

    def _render_gates(gate_results) -> None:
        gates_area.clear()
        with gates_area:
            ui.separator().classes("q-my-md")
            ui.label("Promotion Gates").classes("text-h6 q-pb-xs")

            all_pass = all(g.overall_pass for g in gate_results)

            with ui.row().classes("gap-3 q-pb-md flex-wrap"):
                for gate_result in gate_results:
                    for check in gate_result.checks:
                        color = "positive" if check.passed else "negative"
                        icon = "check_circle" if check.passed else "cancel"
                        with ui.card().classes("q-pa-sm min-w-[140px]"):
                            with ui.row().classes("items-center gap-1"):
                                ui.icon(icon, color=color, size="sm")
                                ui.label(check.gate_name.replace("_", " ").title()).classes(
                                    "text-caption text-grey-6"
                                )
                            ui.label(
                                f"{check.actual_value:.2f} / {check.required_value:.1f}"
                            ).classes(f"text-body1 text-weight-bold text-{color}")

            # Overall verdict
            if all_pass:
                with ui.card().classes("w-full q-pa-sm").style(
                    "background-color: #1a3a1a"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("verified", color="positive", size="md")
                        ui.label(
                            "All gates passed! This strategy is eligible "
                            "for promotion to paper trading."
                        ).classes("text-body1 text-positive")
            else:
                failed = []
                for g in gate_results:
                    failed.extend(g.failed_checks)
                with ui.card().classes("w-full q-pa-sm").style(
                    "background-color: #3a1a1a"
                ):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("block", color="negative", size="md")
                        ui.label(
                            f"{len(failed)} gate(s) failed. Strategy does not "
                            "qualify for promotion yet."
                        ).classes("text-body1 text-negative")

            # Promote button (only when all pass)
            if all_pass:
                async def _promote() -> None:
                    from ..services import promote_to_paper

                    bt = _state["backtest_result"]
                    if bt:
                        ui.notify(
                            "Strategy promoted to paper trading!",
                            type="positive",
                            position="top",
                        )
                        _update_ritual_step("paper")

                ui.button(
                    "Promote to Paper Trading",
                    icon="rocket_launch",
                    on_click=_promote,
                ).props("color=positive unelevated size=lg").classes("q-mt-sm")

    # ---- Action buttons -----------------------------------------------------
    with ui.row().classes("q-pt-md gap-2"):
        ui.button(
            "Run Backtest", icon="play_arrow", on_click=run_backtest
        ).props("color=primary unelevated size=lg")

        ui.button(
            "Run Walk-Forward", icon="trending_up", on_click=run_walk_forward
        ).props("color=secondary unelevated size=lg")


def _kpi_card(label: str, value: str, *, color: str = "primary") -> None:
    with ui.card().classes("q-pa-sm min-w-[110px]"):
        ui.label(label).classes("text-caption text-grey-6")
        ui.label(value).classes(f"text-h6 text-weight-bold text-{color}")
