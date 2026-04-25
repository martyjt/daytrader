"""Research Lab — advanced research tooling for algorithm analysis.

Five tabs:
    1. Model Comparison — run multiple algos side-by-side
    2. Hyperparameter Sweep — grid search over algo params
    3. Feature Attribution — visualize ML feature importance
    4. Walk-Forward Stability — analyze OOS consistency
    5. Discoveries — Exploration Agent output (candidate features ranked by OOS lift)
"""

from __future__ import annotations

from nicegui import ui

from ..shell import page_layout

RESEARCH_PALETTE = ["#5c7cfa", "#22b8cf", "#f76707", "#51cf66", "#cc5de8"]

RANK_OPTIONS = {
    "sharpe_ratio": "Sharpe Ratio",
    "net_return_pct": "Net Return %",
    "win_rate_pct": "Win Rate %",
    "max_drawdown_pct": "Max Drawdown %",
    "profit_factor": "Profit Factor",
}


@ui.page("/research-lab")
async def research_lab_page() -> None:
    page_layout("Research Lab")

    from ...algorithms.registry import AlgorithmRegistry
    from ...backtest.fees import VENUE_PROFILES

    algo_ids = AlgorithmRegistry.available()
    algo_labels = {
        aid: AlgorithmRegistry.get(aid).manifest.name for aid in algo_ids
    }
    venue_labels = {k: v.venue for k, v in VENUE_PROFILES.items()}

    with ui.tabs().classes("w-full") as tabs:
        tab_compare = ui.tab("Model Comparison", icon="compare_arrows")
        tab_sweep = ui.tab("Parameter Sweep", icon="tune")
        tab_attrib = ui.tab("Feature Attribution", icon="insights")
        tab_stability = ui.tab("WF Stability", icon="analytics")
        tab_discover = ui.tab("Discoveries", icon="travel_explore")
        tab_shadow = ui.tab("Shadow", icon="emoji_events")
        tab_portfolio = ui.tab("Portfolio", icon="pie_chart")

    with ui.tab_panels(tabs, value=tab_compare).classes("w-full"):

        # ================================================================
        # TAB 1: Model Comparison
        # ================================================================
        with ui.tab_panel(tab_compare):
            _build_comparison_tab(algo_ids, algo_labels, venue_labels)

        # ================================================================
        # TAB 2: Hyperparameter Sweep
        # ================================================================
        with ui.tab_panel(tab_sweep):
            _build_sweep_tab(algo_ids, algo_labels, venue_labels)

        # ================================================================
        # TAB 3: Feature Attribution
        # ================================================================
        with ui.tab_panel(tab_attrib):
            _build_attribution_tab(algo_ids, algo_labels, venue_labels)

        # ================================================================
        # TAB 4: Walk-Forward Stability
        # ================================================================
        with ui.tab_panel(tab_stability):
            _build_stability_tab(algo_ids, algo_labels, venue_labels)

        # ================================================================
        # TAB 5: Discoveries (Exploration Agent output)
        # ================================================================
        with ui.tab_panel(tab_discover):
            _build_discoveries_tab()

        # ================================================================
        # TAB 6: Shadow Tournament
        # ================================================================
        with ui.tab_panel(tab_shadow):
            _build_shadow_tab(algo_ids, algo_labels, venue_labels)

        # ================================================================
        # TAB 7: Portfolio backtest across a symbol universe
        # ================================================================
        with ui.tab_panel(tab_portfolio):
            _build_portfolio_tab(algo_ids, algo_labels, venue_labels)


# ========================================================================
# Tab 1: Model Comparison
# ========================================================================


def _build_comparison_tab(
    algo_ids: list[str],
    algo_labels: dict[str, str],
    venue_labels: dict[str, str],
) -> None:
    with ui.card().classes("w-full"):
        ui.label("Compare Algorithms").classes("text-h6 q-pb-sm")

        with ui.row().classes("w-full gap-4 items-end"):
            algos_select = ui.select(
                options={aid: algo_labels[aid] for aid in algo_ids},
                value=[],
                label="Algorithms (2-5)",
                multiple=True,
            ).classes("min-w-[300px]")
            rank_by = ui.select(
                RANK_OPTIONS, value="sharpe_ratio", label="Rank by"
            ).classes("min-w-[160px]")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            symbol = ui.input("Symbol", value="BTC-USD")
            timeframe = ui.select(["1d", "1h", "15m", "5m", "1w"], value="1d", label="Timeframe")
            start_date = ui.input("Start Date", value="2024-01-01")
            end_date = ui.input("End Date", value="2024-12-31")
            capital = ui.number("Capital ($)", value=10000, min=1)
            venue = ui.select(options=venue_labels, value="binance_spot", label="Venue").classes("min-w-[160px]")

    results_area = ui.column().classes("w-full q-pt-md")

    async def run_comparison() -> None:
        selected = algos_select.value or []
        if len(selected) < 2:
            ui.notify("Select at least 2 algorithms", type="warning")
            return
        if len(selected) > 5:
            ui.notify("Maximum 5 algorithms", type="warning")
            return

        results_area.clear()
        with results_area:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="lg")
                ui.label(f"Running {len(selected)} backtests...").classes("text-grey-5")

        try:
            from ..services_research import run_comparison_service
            results = await run_comparison_service(
                algo_ids=selected,
                symbol_str=symbol.value,
                timeframe_str=timeframe.value,
                start_str=start_date.value,
                end_str=end_date.value,
                capital=float(capital.value or 10000),
                venue=venue.value,
            )
        except Exception as exc:
            results_area.clear()
            with results_area:
                ui.label(f"Comparison failed: {exc}").classes("text-negative")
            return

        _render_comparison_results(results_area, results, algo_labels, rank_by.value)

    ui.button("Run Comparison", icon="compare_arrows", on_click=run_comparison).props(
        "color=primary unelevated size=lg"
    ).classes("q-mt-sm")


def _render_comparison_results(
    area: ui.element,
    results: list[tuple[str, object]],
    algo_labels: dict[str, str],
    rank_key: str,
) -> None:
    area.clear()
    with area:
        ui.label("Comparison Results").classes("text-h6 q-pb-xs")

        # Rank results
        def _sort_val(item):
            val = item[1].kpis.get(rank_key, 0.0)
            if rank_key == "max_drawdown_pct":
                return val  # less negative is better (higher)
            return -val  # descending for most metrics

        ranked = sorted(results, key=_sort_val)

        # Ranking badges
        with ui.row().classes("gap-2 q-pb-md flex-wrap"):
            for i, (aid, r) in enumerate(ranked):
                val = r.kpis.get(rank_key, 0.0)
                name = algo_labels.get(aid, aid)
                ui.badge(
                    f"#{i+1} {name} ({RANK_OPTIONS.get(rank_key, rank_key)}: {val:.2f})",
                    color=RESEARCH_PALETTE[i % len(RESEARCH_PALETTE)],
                )

        # KPI comparison table
        columns = [
            {"name": "algo", "label": "Algorithm", "field": "algo"},
            {"name": "net_return", "label": "Net Return %", "field": "net_return"},
            {"name": "sharpe", "label": "Sharpe", "field": "sharpe"},
            {"name": "max_dd", "label": "Max DD %", "field": "max_dd"},
            {"name": "win_rate", "label": "Win Rate %", "field": "win_rate"},
            {"name": "profit_factor", "label": "Profit Factor", "field": "profit_factor"},
            {"name": "trades", "label": "Trades", "field": "trades"},
            {"name": "fees", "label": "Fees $", "field": "fees"},
        ]
        rows = []
        for aid, r in ranked:
            k = r.kpis
            rows.append({
                "algo": algo_labels.get(aid, aid),
                "net_return": f"{k.get('net_return_pct', 0):+.2f}",
                "sharpe": f"{k.get('sharpe_ratio', 0):.2f}",
                "max_dd": f"{k.get('max_drawdown_pct', 0):.2f}",
                "win_rate": f"{k.get('win_rate_pct', 0):.1f}",
                "profit_factor": f"{k.get('profit_factor', 0):.2f}",
                "trades": str(k.get("num_trades", 0)),
                "fees": f"{r.total_fees_paid:,.2f}",
            })
        ui.aggrid({
            "columnDefs": columns,
            "rowData": rows,
            "domLayout": "autoHeight",
        }).classes("w-full q-pb-md")

        # Overlaid equity curves
        series = []
        for i, (aid, r) in enumerate(results):
            if r.equity_curve:
                series.append({
                    "name": algo_labels.get(aid, aid),
                    "data": [round(v, 2) for v in r.equity_curve],
                    "type": "line",
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {"color": RESEARCH_PALETTE[i % len(RESEARCH_PALETTE)]},
                    "itemStyle": {"color": RESEARCH_PALETTE[i % len(RESEARCH_PALETTE)]},
                })

        if series:
            max_len = max(len(s["data"]) for s in series)
            ui.echart({
                "backgroundColor": "transparent",
                "tooltip": {"trigger": "axis"},
                "legend": {"data": [s["name"] for s in series], "textStyle": {"color": "#999"}},
                "xAxis": {
                    "type": "category",
                    "data": list(range(max_len)),
                    "name": "Bar",
                    "axisLabel": {"color": "#999"},
                },
                "yAxis": {
                    "type": "value",
                    "name": "Equity ($)",
                    "axisLabel": {"color": "#999"},
                },
                "series": series,
                "grid": {"left": "10%", "right": "4%", "top": "15%", "bottom": "15%"},
            }).classes("w-full").style("height: 420px")


# ========================================================================
# Tab 2: Hyperparameter Sweep
# ========================================================================


def _build_sweep_tab(
    algo_ids: list[str],
    algo_labels: dict[str, str],
    venue_labels: dict[str, str],
) -> None:
    sweep_config: dict = {}

    with ui.card().classes("w-full"):
        ui.label("Hyperparameter Sweep").classes("text-h6 q-pb-sm")

        with ui.row().classes("w-full gap-4 items-end"):
            algo = ui.select(
                options={aid: algo_labels[aid] for aid in algo_ids},
                value=algo_ids[0] if algo_ids else "",
                label="Algorithm",
            ).classes("min-w-[200px]")
            rank_by = ui.select(RANK_OPTIONS, value="sharpe_ratio", label="Rank by").classes("min-w-[160px]")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            symbol = ui.input("Symbol", value="BTC-USD")
            timeframe = ui.select(["1d", "1h", "15m", "5m", "1w"], value="1d", label="Timeframe")
            start_date = ui.input("Start Date", value="2024-01-01")
            end_date = ui.input("End Date", value="2024-12-31")
            capital = ui.number("Capital ($)", value=10000, min=1)
            venue = ui.select(options=venue_labels, value="binance_spot", label="Venue").classes("min-w-[160px]")

        param_container = ui.column().classes("w-full q-pt-xs")

        from ..components.sweep_form import render_sweep_form

        def _on_algo_change(_):
            render_sweep_form(algo.value, param_container, sweep_config)

        algo.on_value_change(_on_algo_change)
        if algo.value:
            render_sweep_form(algo.value, param_container, sweep_config)

    grid_info = ui.label("").classes("text-caption text-grey-6")
    progress = ui.linear_progress(value=0, show_value=False).classes("w-full")
    progress.visible = False

    results_area = ui.column().classes("w-full q-pt-md")

    async def run_sweep() -> None:
        from ..services_research import expand_param_grid, run_sweep_service, MAX_SWEEP_GRID

        grid = expand_param_grid(sweep_config)
        if len(grid) == 0:
            ui.notify("No parameter combinations to test", type="warning")
            return
        if len(grid) > MAX_SWEEP_GRID:
            ui.notify(f"Grid too large ({len(grid)} points, max {MAX_SWEEP_GRID}). Narrow your ranges.", type="negative")
            return

        grid_info.text = f"Running {len(grid)} backtests..."
        progress.visible = True
        progress.value = 0
        results_area.clear()

        def _on_progress(done: int, total: int) -> None:
            progress.value = done / total

        try:
            result = await run_sweep_service(
                algo_id=algo.value,
                symbol_str=symbol.value,
                timeframe_str=timeframe.value,
                start_str=start_date.value,
                end_str=end_date.value,
                capital=float(capital.value or 10000),
                venue=venue.value,
                param_grid=grid,
                rank_by=rank_by.value,
                on_progress=_on_progress,
            )
        except Exception as exc:
            progress.visible = False
            results_area.clear()
            with results_area:
                ui.label(f"Sweep failed: {exc}").classes("text-negative")
            return

        progress.visible = False
        grid_info.text = f"Completed {len(grid)} backtests"
        _render_sweep_results(results_area, result, rank_by.value)

    ui.button("Run Sweep", icon="tune", on_click=run_sweep).props(
        "color=primary unelevated size=lg"
    ).classes("q-mt-sm")


def _render_sweep_results(area: ui.element, result: object, rank_key: str) -> None:
    area.clear()
    with area:
        ui.label("Sweep Results").classes("text-h6 q-pb-xs")

        # Best configuration card
        with ui.card().classes("w-full q-pa-sm q-mb-md").style("background-color: #1a3a1a"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("star", color="positive", size="md")
                ui.label("Best Configuration").classes("text-body1 text-positive")
            with ui.row().classes("gap-3 q-pt-xs flex-wrap"):
                for k, v in result.best_params.items():
                    ui.badge(f"{k}: {v}", color="primary")
            with ui.row().classes("gap-3 q-pt-xs flex-wrap"):
                for k, v in result.best_kpis.items():
                    if k in ("sharpe_ratio", "net_return_pct", "max_drawdown_pct", "win_rate_pct"):
                        _kpi_card(k.replace("_", " ").title(), f"{v:.2f}")

        # Determine swept params for visualization
        swept = [n for n, spec in zip(result.param_names, result.grid_points[0].items()) if True]
        swept_params = []
        seen_vals: dict[str, set] = {}
        for pt in result.grid_points:
            for k, v in pt.items():
                seen_vals.setdefault(k, set()).add(v)
        swept_params = [k for k, vals in seen_vals.items() if len(vals) > 1]

        if len(swept_params) >= 2:
            _render_heatmap(result, swept_params[0], swept_params[1], rank_key)
        elif len(swept_params) == 1:
            _render_line_sweep(result, swept_params[0], rank_key)

        # Full grid table
        if result.grid_points:
            cols = [{"name": k, "label": k.replace("_", " ").title(), "field": k} for k in result.grid_points[0]]
            cols.append({"name": "sharpe", "label": "Sharpe", "field": "sharpe"})
            cols.append({"name": "return", "label": "Net Return %", "field": "return"})
            rows = []
            for i, pt in enumerate(result.grid_points):
                row = dict(pt)
                row["sharpe"] = f"{result.results[i].kpis.get('sharpe_ratio', 0):.2f}"
                row["return"] = f"{result.results[i].kpis.get('net_return_pct', 0):+.2f}"
                rows.append(row)

            ui.aggrid({
                "columnDefs": cols,
                "rowData": rows,
                "domLayout": "autoHeight",
            }).classes("w-full q-pt-md")


def _render_heatmap(result: object, param_x: str, param_y: str, rank_key: str) -> None:
    x_vals = sorted(set(pt[param_x] for pt in result.grid_points))
    y_vals = sorted(set(pt[param_y] for pt in result.grid_points))
    x_map = {v: i for i, v in enumerate(x_vals)}
    y_map = {v: i for i, v in enumerate(y_vals)}

    data = []
    for i, pt in enumerate(result.grid_points):
        xi = x_map.get(pt[param_x])
        yi = y_map.get(pt[param_y])
        val = result.results[i].kpis.get(rank_key, 0.0)
        if xi is not None and yi is not None:
            data.append([xi, yi, round(val, 2)])

    all_vals = [d[2] for d in data]
    min_val = min(all_vals) if all_vals else 0
    max_val = max(all_vals) if all_vals else 1

    ui.echart({
        "backgroundColor": "transparent",
        "tooltip": {"position": "top"},
        "xAxis": {
            "type": "category",
            "data": [str(v) for v in x_vals],
            "name": param_x.replace("_", " ").title(),
            "axisLabel": {"color": "#999"},
        },
        "yAxis": {
            "type": "category",
            "data": [str(v) for v in y_vals],
            "name": param_y.replace("_", " ").title(),
            "axisLabel": {"color": "#999"},
        },
        "visualMap": {
            "min": min_val, "max": max_val,
            "calculable": True, "orient": "horizontal",
            "left": "center", "bottom": "0%",
            "inRange": {"color": ["#3b1f2b", "#5c7cfa", "#51cf66"]},
            "textStyle": {"color": "#999"},
        },
        "series": [{
            "type": "heatmap",
            "data": data,
            "label": {"show": True, "color": "#fff", "fontSize": 10},
        }],
        "grid": {"left": "15%", "right": "4%", "top": "5%", "bottom": "20%"},
    }).classes("w-full").style("height: 400px")


def _render_line_sweep(result: object, param_name: str, rank_key: str) -> None:
    points = []
    for i, pt in enumerate(result.grid_points):
        points.append((pt[param_name], result.results[i].kpis.get(rank_key, 0.0)))
    points.sort(key=lambda x: x[0])

    ui.echart({
        "backgroundColor": "transparent",
        "tooltip": {"trigger": "axis"},
        "xAxis": {
            "type": "category",
            "data": [str(p[0]) for p in points],
            "name": param_name.replace("_", " ").title(),
            "axisLabel": {"color": "#999"},
        },
        "yAxis": {
            "type": "value",
            "name": RANK_OPTIONS.get(rank_key, rank_key),
            "axisLabel": {"color": "#999"},
        },
        "series": [{
            "data": [round(p[1], 2) for p in points],
            "type": "line",
            "smooth": True,
            "showSymbol": True,
            "lineStyle": {"color": "#5c7cfa"},
            "itemStyle": {"color": "#5c7cfa"},
            "areaStyle": {"opacity": 0.1},
        }],
        "grid": {"left": "10%", "right": "4%", "top": "10%", "bottom": "15%"},
    }).classes("w-full").style("height: 350px")


# ========================================================================
# Tab 3: Feature Attribution
# ========================================================================


def _build_attribution_tab(
    algo_ids: list[str],
    algo_labels: dict[str, str],
    venue_labels: dict[str, str],
) -> None:
    ml_algos = [aid for aid in algo_ids if any(
        kw in aid for kw in ("xgboost", "lstm", "transformer", "cnn_lstm", "regime")
    )]

    with ui.card().classes("w-full"):
        ui.label("Feature Attribution").classes("text-h6 q-pb-sm")

        if not ml_algos:
            ui.label("No ML algorithms available.").classes("text-grey-5")
            return

        with ui.row().classes("w-full gap-4 items-end"):
            algo = ui.select(
                options={aid: algo_labels.get(aid, aid) for aid in ml_algos},
                value=ml_algos[0],
                label="ML Algorithm",
            ).classes("min-w-[200px]")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            symbol = ui.input("Symbol", value="BTC-USD")
            timeframe = ui.select(["1d", "1h", "15m", "5m", "1w"], value="1d", label="Timeframe")
            start_date = ui.input("Start Date", value="2024-01-01")
            end_date = ui.input("End Date", value="2024-12-31")
            capital = ui.number("Capital ($)", value=10000, min=1)
            venue = ui.select(options=venue_labels, value="binance_spot", label="Venue").classes("min-w-[160px]")

    results_area = ui.column().classes("w-full q-pt-md")

    async def run_attribution() -> None:
        results_area.clear()
        with results_area:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="lg")
                ui.label("Running backtest for feature analysis...").classes("text-grey-5")

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
            results_area.clear()
            with results_area:
                ui.label(f"Analysis failed: {exc}").classes("text-negative")
            return

        from ..services_research import extract_feature_importance
        importance = extract_feature_importance(result.debug_logs)

        results_area.clear()
        with results_area:
            if not importance:
                with ui.card().classes("w-full q-pa-md"):
                    ui.icon("info", color="grey-5", size="md")
                    ui.label(
                        "This algorithm does not produce feature importance data. "
                        "XGBoost Trend provides per-prediction feature importance."
                    ).classes("text-grey-5")
                return

            ui.label("Feature Importance (averaged across predictions)").classes("text-h6 q-pb-xs")

            names = list(importance.keys())
            values = list(importance.values())

            ui.echart({
                "backgroundColor": "transparent",
                "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                "xAxis": {
                    "type": "value",
                    "name": "Importance",
                    "axisLabel": {"color": "#999"},
                },
                "yAxis": {
                    "type": "category",
                    "data": names[::-1],
                    "axisLabel": {"color": "#999"},
                },
                "series": [{
                    "type": "bar",
                    "data": values[::-1],
                    "itemStyle": {"color": "#5c7cfa"},
                }],
                "grid": {"left": "25%", "right": "8%", "top": "5%", "bottom": "10%"},
            }).classes("w-full").style("height: 400px")

    ui.button("Run Analysis", icon="insights", on_click=run_attribution).props(
        "color=primary unelevated size=lg"
    ).classes("q-mt-sm")


# ========================================================================
# Tab 4: Walk-Forward Stability
# ========================================================================


def _build_stability_tab(
    algo_ids: list[str],
    algo_labels: dict[str, str],
    venue_labels: dict[str, str],
) -> None:
    with ui.card().classes("w-full"):
        ui.label("Walk-Forward Stability Analysis").classes("text-h6 q-pb-sm")

        with ui.row().classes("w-full gap-4 items-end"):
            algo = ui.select(
                options={aid: algo_labels[aid] for aid in algo_ids},
                value=algo_ids[0] if algo_ids else "",
                label="Algorithm",
            ).classes("min-w-[200px]")
            n_folds = ui.slider(min=3, max=10, value=5, step=1).classes("min-w-[200px]")
            ui.label("Folds:").bind_text_from(n_folds, "value", backward=lambda v: f"Folds: {int(v)}")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            symbol = ui.input("Symbol", value="BTC-USD")
            timeframe = ui.select(["1d", "1h", "15m", "5m", "1w"], value="1d", label="Timeframe")
            start_date = ui.input("Start Date", value="2024-01-01")
            end_date = ui.input("End Date", value="2024-12-31")
            capital = ui.number("Capital ($)", value=10000, min=1)
            venue = ui.select(options=venue_labels, value="binance_spot", label="Venue").classes("min-w-[160px]")

    results_area = ui.column().classes("w-full q-pt-md")

    async def run_stability() -> None:
        results_area.clear()
        with results_area:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="lg")
                ui.label(f"Running walk-forward ({int(n_folds.value)} folds)...").classes("text-grey-5")

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
                n_folds=int(n_folds.value),
            )
        except Exception as exc:
            results_area.clear()
            with results_area:
                ui.label(f"Stability analysis failed: {exc}").classes("text-negative")
            return

        from ..services_research import compute_stability_metrics
        metrics = compute_stability_metrics(wf)
        _render_stability_results(results_area, wf, metrics)

    ui.button("Run Stability Analysis", icon="analytics", on_click=run_stability).props(
        "color=primary unelevated size=lg"
    ).classes("q-mt-sm")


def _render_stability_results(area: ui.element, wf: object, metrics: dict) -> None:
    area.clear()
    with area:
        ui.label("Stability Results").classes("text-h6 q-pb-xs")

        # KPI cards
        with ui.row().classes("gap-3 q-pb-md flex-wrap"):
            _kpi_card("OOS Sharpe", f"{metrics['oos_sharpe']:.2f}",
                      color="positive" if metrics["oos_sharpe"] >= 0.3 else "warning")
            _kpi_card("Sharpe Std Dev", f"{metrics['sharpe_std']:.2f}",
                      color="positive" if metrics["sharpe_std"] < 0.5 else "warning")
            _kpi_card("IS->OOS Degradation", f"{metrics['degradation_pct']:.1f}%",
                      color="warning" if metrics["degradation_pct"] > 30 else "positive")
            _kpi_card("Consistency", f"{metrics['consistency_pct']:.0f}%",
                      color="positive" if metrics["consistency_pct"] >= 60 else "negative")
            _kpi_card("Worst Fold", f"{metrics['worst_fold_sharpe']:.2f}",
                      color="negative" if metrics["worst_fold_sharpe"] < 0 else "positive")

        # Per-fold OOS equity curves overlaid
        fold_series = []
        for i, fold in enumerate(wf.folds):
            eq = fold.test_result.equity_curve
            if eq:
                fold_series.append({
                    "name": f"Fold {i+1}",
                    "data": [round(v, 2) for v in eq],
                    "type": "line",
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {"color": RESEARCH_PALETTE[i % len(RESEARCH_PALETTE)]},
                    "itemStyle": {"color": RESEARCH_PALETTE[i % len(RESEARCH_PALETTE)]},
                })

        if fold_series:
            max_len = max(len(s["data"]) for s in fold_series)
            ui.echart({
                "backgroundColor": "transparent",
                "tooltip": {"trigger": "axis"},
                "legend": {"data": [s["name"] for s in fold_series], "textStyle": {"color": "#999"}},
                "xAxis": {
                    "type": "category",
                    "data": list(range(max_len)),
                    "name": "OOS Bar",
                    "axisLabel": {"color": "#999"},
                },
                "yAxis": {"type": "value", "name": "Equity ($)", "axisLabel": {"color": "#999"}},
                "series": fold_series,
                "grid": {"left": "10%", "right": "4%", "top": "15%", "bottom": "15%"},
            }).classes("w-full").style("height: 350px")

        # Sharpe stability bar chart
        oos_sharpes = [f.oos_sharpe for f in wf.folds]
        is_sharpes = [f.train_result.kpis.get("sharpe_ratio", 0.0) for f in wf.folds]
        fold_labels = [f"Fold {i+1}" for i in range(len(wf.folds))]

        ui.echart({
            "backgroundColor": "transparent",
            "tooltip": {"trigger": "axis"},
            "legend": {"data": ["In-Sample Sharpe", "OOS Sharpe"], "textStyle": {"color": "#999"}},
            "xAxis": {
                "type": "category",
                "data": fold_labels,
                "axisLabel": {"color": "#999"},
            },
            "yAxis": {"type": "value", "name": "Sharpe Ratio", "axisLabel": {"color": "#999"}},
            "series": [
                {
                    "name": "In-Sample Sharpe",
                    "type": "bar",
                    "data": [round(v, 2) for v in is_sharpes],
                    "itemStyle": {"color": "#495057"},
                },
                {
                    "name": "OOS Sharpe",
                    "type": "bar",
                    "data": [round(v, 2) for v in oos_sharpes],
                    "itemStyle": {"color": "#5c7cfa"},
                },
            ],
            "grid": {"left": "10%", "right": "4%", "top": "15%", "bottom": "10%"},
        }).classes("w-full").style("height: 300px")


# ========================================================================
# Shared helpers
# ========================================================================


# ========================================================================
# Tab 6: Shadow Tournament
# ========================================================================


def _build_shadow_tab(
    algo_ids: list[str],
    algo_labels: dict[str, str],
    venue_labels: dict[str, str],
) -> None:
    """Candidate algorithms race a primary on the same data + walk-forward.

    Winners get flagged for review; results persist in shadow_runs so the
    promotion decision has an audit trail.
    """
    ui.label(
        "Pick a primary (the incumbent) and N candidates. Each runs "
        "walk-forward on the same symbol/window; candidates that beat the "
        "primary on aggregate Sharpe AND ≥50% fold stability become "
        "eligible for promotion. Mode: single-symbol, per-symbol universe "
        "(one tournament per symbol), or basket universe (mean across symbols)."
    ).classes("text-caption text-grey-5 q-pb-sm")

    # State that the universe picker mutates and the run handler reads.
    universe_state = {"id": None, "name": "", "symbols": []}

    with ui.card().classes("w-full"):
        ui.label("Tournament config").classes("text-subtitle1 q-pb-sm")

        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            default_primary = algo_ids[0] if algo_ids else ""
            primary_in = ui.select(
                options={aid: algo_labels[aid] for aid in algo_ids},
                value=default_primary,
                label="Primary (incumbent)",
            ).classes("min-w-[220px]")
            preferred = [
                aid for aid in ("ema_crossover", "macd_signal", "rsi_mean_reversion")
                if aid in algo_ids and aid != default_primary
            ][:3]
            candidates_in = ui.select(
                options={aid: algo_labels[aid] for aid in algo_ids},
                value=preferred,
                label="Candidates",
                multiple=True,
            ).props("use-chips").classes("min-w-[420px] flex-grow")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            mode_in = ui.toggle(
                {"single": "Single", "per_symbol": "Per-Symbol", "basket": "Basket"},
                value="single",
            ).props("color=primary")
            symbol_in = ui.input("Symbol", value="BTC-USD").classes("min-w-[140px]")
            tf_in = ui.select(
                ["1d", "4h", "1h"], value="1d", label="Timeframe",
            ).classes("min-w-[100px]")
            start_in = ui.input("Start Date", value="2024-01-01")
            end_in = ui.input("End Date", value="2024-12-31")
            capital_in = ui.number("Capital", value=10_000.0, min=100.0)
            venue_in = ui.select(
                options=venue_labels,
                value="binance_spot",
                label="Venue",
            ).classes("min-w-[160px]")
            folds_in = ui.number("WF Folds", value=5, min=3, max=10)

        # Universe picker — only relevant for per-symbol/basket modes.
        universe_row = ui.row().classes("w-full gap-2 items-end q-pt-xs")
        universe_status = ui.label("").classes("text-caption text-grey-6")

        async def _refresh_universe_picker() -> None:
            universe_row.clear()
            try:
                from ..services import list_universes

                universes = await list_universes()
            except Exception:
                universes = []
            with universe_row:
                if not universes:
                    ui.label(
                        "No saved universes — Single mode only. Build one on "
                        "the Universes page to enable Per-Symbol / Basket modes."
                    ).classes("text-caption text-grey-6")
                    return
                picker = ui.select(
                    options={
                        str(u.id): f"{u.name} ({len(u.symbols)})"
                        for u in universes
                    },
                    label="Universe (for Per-Symbol / Basket)",
                ).classes("min-w-[260px]")

                def _apply_universe(_=None) -> None:
                    if not picker.value:
                        return
                    from uuid import UUID as _UUID

                    uid = _UUID(picker.value)
                    u = next((x for x in universes if x.id == uid), None)
                    if not u:
                        return
                    universe_state["id"] = u.id
                    universe_state["name"] = u.name
                    universe_state["symbols"] = list(u.symbols)
                    universe_status.text = (
                        f"Loaded universe: {u.name} ({len(u.symbols)} symbols)"
                    )

                picker.on_value_change(_apply_universe)

    result_area = ui.column().classes("w-full q-pt-md")
    history_area = ui.column().classes("w-full q-pt-md")

    async def _refresh_history() -> None:
        history_area.clear()
        try:
            from ..services import list_shadow_tournaments_service

            rows = await list_shadow_tournaments_service(limit=20)
        except Exception as exc:  # noqa: BLE001
            with history_area:
                ui.label(f"Failed to load history: {exc}").classes("text-negative")
            return

        with history_area:
            ui.separator().classes("q-my-md")
            ui.label("Recent tournaments").classes("text-subtitle1 q-pb-xs")
            if not rows:
                ui.label("No tournaments yet.").classes("text-caption text-grey-6")
                return
            cols = [
                {"name": "when", "label": "When", "field": "when"},
                {"name": "symbol", "label": "Symbol", "field": "symbol"},
                {"name": "tf", "label": "Timeframe", "field": "tf"},
                {"name": "primary", "label": "Primary", "field": "primary"},
                {"name": "n_cand", "label": "# Candidates", "field": "n_cand"},
                {"name": "n_wins", "label": "# Winners", "field": "n_wins"},
            ]
            table_rows = [
                {
                    "when": t["created_at"].strftime("%Y-%m-%d %H:%M") if t["created_at"] else "",
                    "symbol": t["symbol"],
                    "tf": t["timeframe"],
                    "primary": t["primary"],
                    "n_cand": t["n_candidates"],
                    "n_wins": t["n_winners"],
                }
                for t in rows
            ]
            ui.aggrid({
                "columnDefs": cols,
                "rowData": table_rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"sortable": True, "resizable": True},
            }).classes("w-full")

    def _render_single_report(report, *, header_prefix: str = "") -> None:
        """Render one ShadowTournamentReport into the *current* UI context.

        Caller wraps this in a card / expansion / column. Used by all three
        modes — single-symbol writes one of these; per-symbol writes one
        per universe symbol; basket writes one synthetic aggregate.
        """
        header = (
            f"{header_prefix}Tournament on {report.target_symbol} "
            f"{report.target_timeframe} · "
            f"{len(report.candidates)} algorithm(s) · "
            f"{len(report.winners)} winner(s)"
        )
        with ui.row().classes("w-full items-center"):
            ui.label(header).classes("text-body1")
            ui.space()
            _render_export_button(report, "tournament")

        cols = [
            {"name": "name", "label": "Algorithm", "field": "name"},
            {"name": "sharpe", "label": "OOS Sharpe", "field": "sharpe", "sortable": True},
            {"name": "ret", "label": "OOS Return %", "field": "ret"},
            {"name": "dd", "label": "Max DD %", "field": "dd"},
            {"name": "trades", "label": "# Trades", "field": "trades"},
            {"name": "stab", "label": "Stability", "field": "stab"},
            {"name": "flag", "label": "Verdict", "field": "flag"},
        ]
        table_rows = []
        for c in report.ranked():
            flag = "PRIMARY" if c.is_primary else ("WINNER" if c.beat_primary else "—")
            table_rows.append({
                "name": c.algo_name + (" *" if c.is_primary else ""),
                "sharpe": f"{c.sharpe:+.2f}",
                "ret": f"{c.net_return_pct:+.2f}",
                "dd": f"{c.max_drawdown_pct:+.2f}",
                "trades": c.num_trades,
                "stab": f"{int(c.stability_score * 100)}%",
                "flag": flag,
            })
        ui.aggrid({
            "columnDefs": cols,
            "rowData": table_rows,
            "domLayout": "autoHeight",
            "defaultColDef": {"sortable": True, "resizable": True},
        }).classes("w-full")

        action_rows = [c for c in report.ranked() if not c.is_primary and not c.error]
        if action_rows:
            winners = [c for c in action_rows if c.beat_primary]
            with ui.row().classes("w-full items-center q-pt-md"):
                ui.label("Actions").classes("text-subtitle2")
                ui.space()
                if winners:
                    async def _promote_all_winners() -> None:
                        from ..services import update_shadow_status_service

                        ok = 0
                        for c in winners:
                            try:
                                await update_shadow_status_service(
                                    report.tournament_id, c.algo_id, "promoted",
                                )
                                ok += 1
                            except Exception:  # noqa: BLE001
                                continue
                        ui.notify(
                            f"Promoted {ok}/{len(winners)} winners",
                            type="positive",
                        )

                    ui.button(
                        f"Promote all {len(winners)} winner(s)",
                        icon="north",
                        on_click=_promote_all_winners,
                    ).props("color=positive outline dense")

            for c in action_rows:
                _render_shadow_action_row(report.tournament_id, c)

        curves_series = []
        for i, c in enumerate(report.ranked()):
            if not c.oos_equity_curve:
                continue
            curves_series.append({
                "name": c.algo_name + (" *" if c.is_primary else ""),
                "type": "line",
                "data": [round(v, 2) for v in c.oos_equity_curve],
                "smooth": True,
                "showSymbol": False,
                "lineStyle": {
                    "color": RESEARCH_PALETTE[i % len(RESEARCH_PALETTE)],
                    "width": 3 if c.is_primary else 1.5,
                },
            })
        if curves_series:
            max_len = max(len(s["data"]) for s in curves_series)
            ui.echart({
                "backgroundColor": "transparent",
                "tooltip": {"trigger": "axis"},
                "legend": {"textStyle": {"color": "#bbb"}},
                "xAxis": {
                    "type": "category",
                    "data": list(range(max_len)),
                    "name": "OOS Bar",
                    "axisLabel": {"color": "#999"},
                },
                "yAxis": {"type": "value", "axisLabel": {"color": "#999"}},
                "series": curves_series,
                "grid": {"left": "8%", "right": "4%", "top": "12%", "bottom": "10%"},
            }).classes("w-full").style("height: 360px")

    async def _run_tournament() -> None:
        result_area.clear()
        if not primary_in.value or not candidates_in.value:
            with result_area:
                ui.label("Pick a primary and at least one candidate.").classes(
                    "text-grey-5"
                )
            return

        mode = mode_in.value or "single"
        if mode != "single" and not universe_state["symbols"]:
            with result_area:
                ui.label(
                    f"Mode '{mode}' needs a loaded universe — pick one above first."
                ).classes("text-warning")
            return

        symbols = (
            [symbol_in.value] if mode == "single" else universe_state["symbols"]
        )
        with result_area:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="lg")
                ui.label(
                    f"Running {mode} tournament: {primary_in.value} vs "
                    f"{len(candidates_in.value)} candidate(s) on "
                    f"{len(symbols)} symbol(s)..."
                ).classes("text-grey-5")

        try:
            if mode == "single":
                from ..services import run_shadow_tournament_service

                report = await run_shadow_tournament_service(
                    primary_algo_id=primary_in.value,
                    candidate_algo_ids=list(candidates_in.value),
                    symbol_str=symbol_in.value,
                    timeframe_str=tf_in.value,
                    start_str=start_in.value,
                    end_str=end_in.value,
                    initial_capital=float(capital_in.value or 10_000.0),
                    venue=venue_in.value,
                    n_folds=int(folds_in.value or 5),
                )
                result_area.clear()
                with result_area:
                    _render_single_report(report)
            elif mode == "per_symbol":
                from ..services import run_per_symbol_universe_tournament_service

                bundle = await run_per_symbol_universe_tournament_service(
                    primary_algo_id=primary_in.value,
                    candidate_algo_ids=list(candidates_in.value),
                    symbols=symbols,
                    universe_name=universe_state["name"],
                    timeframe_str=tf_in.value,
                    start_str=start_in.value,
                    end_str=end_in.value,
                    initial_capital=float(capital_in.value or 10_000.0),
                    venue=venue_in.value,
                    n_folds=int(folds_in.value or 5),
                )
                result_area.clear()
                with result_area:
                    _render_per_symbol_bundle(bundle)
            else:  # basket
                from ..services import run_basket_tournament_service

                report = await run_basket_tournament_service(
                    primary_algo_id=primary_in.value,
                    candidate_algo_ids=list(candidates_in.value),
                    symbols=symbols,
                    universe_name=universe_state["name"],
                    timeframe_str=tf_in.value,
                    start_str=start_in.value,
                    end_str=end_in.value,
                    initial_capital=float(capital_in.value or 10_000.0),
                    venue=venue_in.value,
                    n_folds=int(folds_in.value or 5),
                )
                result_area.clear()
                with result_area:
                    ui.label(
                        f"Basket: {universe_state['name']} "
                        f"({len(symbols)} symbols, mean across symbols)"
                    ).classes("text-caption text-grey-5 q-pb-xs")
                    _render_single_report(
                        report,
                        header_prefix=f"BASKET ({len(symbols)} symbols) — ",
                    )
        except Exception as exc:  # noqa: BLE001
            result_area.clear()
            with result_area:
                ui.icon("error", color="negative")
                ui.label(f"Tournament failed: {exc}").classes("text-negative")
            return

        await _refresh_history()

    def _render_per_symbol_bundle(bundle) -> None:
        """Top-level wins-per-algo summary, then one expansion per symbol."""
        win_counts = bundle.win_counts()
        names = bundle.algo_names()
        n_syms = len(bundle.symbols)

        ui.label(
            f"Universe: {bundle.universe_name} ({n_syms} symbols)"
        ).classes("text-subtitle1 q-pb-xs")

        if win_counts:
            with ui.row().classes("w-full gap-3 q-pb-md"):
                for aid, n in sorted(
                    win_counts.items(), key=lambda kv: -kv[1]
                ):
                    label = names.get(aid, aid)
                    with ui.card().classes("q-pa-sm"):
                        ui.label(label).classes("text-caption")
                        ui.label(f"{n} / {n_syms}").classes(
                            "text-h6 text-positive"
                        )
                        ui.label("symbols won").classes("text-caption text-grey-6")
        else:
            ui.label(
                "No candidate beat the primary on any symbol."
            ).classes("text-caption text-grey-6 q-pb-sm")

        for report in bundle.per_symbol_reports:
            with ui.expansion(
                f"{report.target_symbol} — {len(report.winners)} winner(s)",
                icon="show_chart",
                value=(len(report.winners) > 0),  # auto-open ones with winners
            ).classes("w-full"):
                with ui.card().classes("w-full"):
                    _render_single_report(report)

    ui.button(
        "Run Tournament", icon="emoji_events", on_click=_run_tournament,
    ).props("color=primary unelevated size=lg").classes("q-mt-sm")

    ui.timer(0.2, _refresh_history, once=True)
    ui.timer(0.2, _refresh_universe_picker, once=True)


# ========================================================================
# Tab 7: Portfolio (multi-symbol backtest)
# ========================================================================


def _build_portfolio_tab(
    algo_ids: list[str],
    algo_labels: dict[str, str],
    venue_labels: dict[str, str],
) -> None:
    """Backtest one algorithm across a universe of symbols with 1/N capital."""
    ui.label(
        "Run the same algorithm against a universe of symbols with equal "
        "1/N capital allocation. Returns per-symbol and portfolio-level "
        "KPIs plus an aggregate equity curve."
    ).classes("text-caption text-grey-5 q-pb-sm")

    with ui.card().classes("w-full"):
        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            algo_in = ui.select(
                options={aid: algo_labels[aid] for aid in algo_ids},
                value=algo_ids[0] if algo_ids else "",
                label="Algorithm",
            ).classes("min-w-[220px]")
            symbols_in = ui.input(
                "Symbols (comma-separated)",
                value="BTC-USD, ETH-USD, AAPL",
            ).classes("min-w-[360px] flex-grow")

        # Saved universe picker — one-click populates the Symbols field.
        universe_row = ui.row().classes("w-full gap-2 items-end q-pt-xs")

        async def _refresh_universe_picker() -> None:
            universe_row.clear()
            try:
                from ..services import list_universes

                universes = await list_universes()
            except Exception:
                universes = []
            with universe_row:
                if not universes:
                    ui.label(
                        "No saved universes. Build one on the Universes page "
                        "to populate symbols in one click."
                    ).classes("text-caption text-grey-6")
                    return
                picker = ui.select(
                    options={str(u.id): f"{u.name} ({len(u.symbols)})" for u in universes},
                    label="Load universe",
                ).classes("min-w-[220px]")

                def _apply_universe(_=None) -> None:
                    if not picker.value:
                        return
                    from uuid import UUID as _UUID

                    uid = _UUID(picker.value)
                    u = next((x for x in universes if x.id == uid), None)
                    if not u:
                        return
                    symbols_in.value = ", ".join(u.symbols)
                    ui.notify(
                        f"Loaded universe: {u.name} ({len(u.symbols)} symbols)",
                        type="info",
                    )

                picker.on_value_change(_apply_universe)

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            tf_in = ui.select(
                ["1d", "4h", "1h"], value="1d", label="Timeframe",
            ).classes("min-w-[100px]")
            start_in = ui.input("Start Date", value="2024-01-01")
            end_in = ui.input("End Date", value="2024-12-31")
            capital_in = ui.number(
                "Total Capital", value=10_000.0, min=100.0,
            )
            venue_in = ui.select(
                options=venue_labels,
                value="binance_spot",
                label="Venue",
            ).classes("min-w-[160px]")

    result_area = ui.column().classes("w-full q-pt-md")
    ui.timer(0.2, _refresh_universe_picker, once=True)

    async def run_portfolio() -> None:
        result_area.clear()
        symbols = [s.strip() for s in (symbols_in.value or "").split(",") if s.strip()]
        if not symbols:
            with result_area:
                ui.label("Enter at least one symbol.").classes("text-grey-5")
            return

        with result_area:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="lg")
                ui.label(
                    f"Backtesting {algo_in.value} on {len(symbols)} symbols..."
                ).classes("text-grey-5")

        try:
            from ..services import run_portfolio_backtest_service

            report = await run_portfolio_backtest_service(
                algo_id=algo_in.value,
                symbols=symbols,
                timeframe_str=tf_in.value,
                start_str=start_in.value,
                end_str=end_in.value,
                total_capital=float(capital_in.value or 10_000.0),
                venue=venue_in.value,
            )
        except Exception as exc:  # noqa: BLE001
            result_area.clear()
            with result_area:
                ui.icon("error", color="negative")
                ui.label(f"Portfolio run failed: {exc}").classes("text-negative")
            return

        result_area.clear()
        with result_area:
            with ui.row().classes("w-full items-center"):
                ui.label("Portfolio Results").classes("text-h6")
                ui.space()
                _render_export_button(report, "portfolio")
            with ui.row().classes("gap-3 flex-wrap q-pb-sm"):
                _kpi_card(
                    "Total Return",
                    f"{report.portfolio_return_pct:+.2f}%",
                    color="positive" if report.portfolio_return_pct >= 0 else "negative",
                )
                _kpi_card(
                    "Portfolio Sharpe",
                    f"{report.portfolio_sharpe:.2f}",
                    color="positive" if report.portfolio_sharpe >= 0.3 else "warning",
                )
                _kpi_card(
                    "Max Drawdown",
                    f"{report.portfolio_max_drawdown_pct:.2f}%",
                    color="negative" if report.portfolio_max_drawdown_pct > 10 else "warning",
                )
                _kpi_card(
                    "Final Equity",
                    f"${report.final_total_equity:,.0f}",
                )

            # Per-symbol table
            cols = [
                {"name": "symbol", "label": "Symbol", "field": "symbol"},
                {"name": "sharpe", "label": "Sharpe", "field": "sharpe", "sortable": True},
                {"name": "ret", "label": "Return %", "field": "ret"},
                {"name": "dd", "label": "Max DD %", "field": "dd"},
                {"name": "trades", "label": "# Trades", "field": "trades"},
                {"name": "final", "label": "Final $", "field": "final"},
                {"name": "status", "label": "Status", "field": "status"},
            ]
            table_rows = []
            for s in report.symbols:
                table_rows.append({
                    "symbol": s.symbol,
                    "sharpe": f"{s.sharpe:+.2f}",
                    "ret": f"{s.net_return_pct:+.2f}",
                    "dd": f"{s.max_drawdown_pct:+.2f}",
                    "trades": s.num_trades,
                    "final": f"${s.final_equity:,.0f}",
                    "status": "error" if s.error else "ok",
                })
            ui.aggrid({
                "columnDefs": cols,
                "rowData": table_rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"sortable": True, "resizable": True},
            }).classes("w-full")

            best, worst = report.best_symbol, report.worst_symbol
            if best and worst and best is not worst:
                with ui.row().classes("gap-4 q-pt-sm"):
                    ui.badge(
                        f"Best: {best.symbol} (Sharpe {best.sharpe:+.2f})",
                        color="positive",
                    )
                    ui.badge(
                        f"Worst: {worst.symbol} (Sharpe {worst.sharpe:+.2f})",
                        color="negative",
                    )

            # Equity curves: portfolio + each symbol (thin)
            curves = []
            if report.portfolio_equity_curve:
                curves.append({
                    "name": "Portfolio (sum)",
                    "type": "line",
                    "data": [round(v, 2) for v in report.portfolio_equity_curve],
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {"color": "#5c7cfa", "width": 3},
                })
            for i, s in enumerate(report.symbols):
                if not s.equity_curve:
                    continue
                curves.append({
                    "name": s.symbol,
                    "type": "line",
                    "data": [round(v, 2) for v in s.equity_curve],
                    "smooth": True,
                    "showSymbol": False,
                    "lineStyle": {
                        "color": RESEARCH_PALETTE[i % len(RESEARCH_PALETTE)],
                        "width": 1,
                    },
                })
            if curves:
                max_len = max(len(c["data"]) for c in curves)
                ui.echart({
                    "backgroundColor": "transparent",
                    "tooltip": {"trigger": "axis"},
                    "legend": {"textStyle": {"color": "#bbb"}},
                    "xAxis": {
                        "type": "category",
                        "data": list(range(max_len)),
                        "axisLabel": {"color": "#999"},
                    },
                    "yAxis": {
                        "type": "value",
                        "axisLabel": {"color": "#999"},
                    },
                    "series": curves,
                    "grid": {"left": "8%", "right": "4%", "top": "12%", "bottom": "10%"},
                }).classes("w-full").style("height: 360px")

    ui.button(
        "Run Portfolio Backtest",
        icon="pie_chart",
        on_click=run_portfolio,
    ).props("color=primary unelevated size=lg").classes("q-mt-sm")


def _render_export_button(report, kind: str) -> None:
    """Button that opens a dialog with the report rendered as Markdown."""

    def _open_export() -> None:
        try:
            from ..reports import portfolio_to_markdown, tournament_to_markdown

            if kind == "tournament":
                md = tournament_to_markdown(report)
                title = "Tournament report — Markdown"
            elif kind == "portfolio":
                md = portfolio_to_markdown(report)
                title = "Portfolio report — Markdown"
            else:
                md = f"(unknown kind: {kind})"
                title = "Report"
        except Exception as exc:  # noqa: BLE001
            md = f"Export failed: {exc}"
            title = "Export error"

        with ui.dialog() as dialog, ui.card().classes("min-w-[720px] max-w-[1000px]"):
            with ui.row().classes("w-full items-center justify-between q-pb-sm"):
                ui.label(title).classes("text-h6")
                ui.button(icon="close", on_click=dialog.close).props("flat dense")
            ui.markdown(md).classes("w-full")
            with ui.expansion("Copyable source", icon="content_copy").classes("w-full q-pt-sm"):
                textarea = ui.textarea(value=md).classes("w-full").props("readonly rows=20")

                def _copy() -> None:
                    try:
                        ui.run_javascript(
                            f"navigator.clipboard.writeText({md!r})"
                        )
                        ui.notify("Report copied to clipboard", type="positive")
                    except Exception:  # noqa: BLE001
                        ui.notify("Copy failed — select from the box", type="warning")

                ui.button("Copy to clipboard", icon="content_copy", on_click=_copy).props(
                    "color=primary flat"
                )
        dialog.open()

    ui.button("Export", icon="file_download", on_click=_open_export).props(
        "flat dense"
    ).tooltip("Copy this result as Markdown")


def _render_shadow_action_row(tournament_id, candidate) -> None:
    """One-line promote/dismiss control for a shadow candidate.

    Shown beneath the tournament result table. Updates promotion_status
    on the ``shadow_runs`` row(s) and notifies the user via an alert.
    """
    status_text = ui.label("")

    def _set_status_text(status: str) -> None:
        status_text.text = f"Status: {status}"
        color = {
            "promoted": "positive",
            "dismissed": "negative",
            "pending": "grey-5",
        }.get(status, "grey-5")
        status_text.classes(replace=f"text-caption text-{color}")

    async def _set_status(new_status: str) -> None:
        try:
            from ..services import update_shadow_status_service

            await update_shadow_status_service(
                tournament_id, candidate.algo_id, new_status,
            )
            try:
                _set_status_text(new_status)
                ui.notify(
                    f"{candidate.algo_name} → {new_status}",
                    type="positive" if new_status == "promoted" else "warning",
                )
            except RuntimeError:
                return
        except Exception as exc:  # noqa: BLE001
            try:
                ui.notify(f"Action failed: {exc}", type="negative")
            except RuntimeError:
                return

    badge_color = "positive" if candidate.beat_primary else "grey-8"
    with ui.card().classes("w-full q-pa-sm q-mb-xs"):
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.badge(
                "WINNER" if candidate.beat_primary else "CANDIDATE",
                color=badge_color,
            ).props("outline" if not candidate.beat_primary else "")
            ui.label(candidate.algo_name).classes("text-body2 text-weight-medium")
            ui.label(
                f"Sharpe {candidate.sharpe:+.2f} · Ret {candidate.net_return_pct:+.2f}% "
                f"· Stab {int(candidate.stability_score*100)}%"
            ).classes("text-caption text-grey-5")
            ui.space()
            _set_status_text("pending")
            ui.button(
                "Promote",
                icon="north",
                on_click=lambda _e=None: _set_status("promoted"),
            ).props(
                f"color=positive {'' if candidate.beat_primary else 'flat'} dense"
            )
            ui.button(
                "Dismiss",
                icon="south",
                on_click=lambda _e=None: _set_status("dismissed"),
            ).props("color=negative flat dense")


def _kpi_card(label: str, value: str, *, color: str = "primary") -> None:
    with ui.card().classes("q-pa-sm min-w-[110px]"):
        ui.label(label).classes("text-caption text-grey-6")
        ui.label(value).classes(f"text-h6 text-weight-bold text-{color}")


# ========================================================================
# Tab 5: Discoveries (Exploration Agent)
# ========================================================================


def _build_discoveries_tab() -> None:
    """Run the Exploration Agent and list candidate features ranked by OOS lift.

    Every candidate seeded from FRED macro series + NewsAPI sentiment queries
    is scored against a price-derived baseline. Benjamini-Hochberg FDR
    correction keeps the significant-discovery list honest across many tests.
    """
    state: dict = {"last_result": None}

    ui.label(
        "The Exploration Agent tests whether candidate features (macro "
        "series, news sentiment, cross-asset correlates) add predictive "
        "lift over a price-derived baseline. Significant candidates pass "
        "Benjamini-Hochberg FDR correction and can be promoted into your "
        "algorithms."
    ).classes("text-caption text-grey-5 q-pb-sm")

    with ui.card().classes("w-full"):
        ui.label("Run a scan").classes("text-subtitle1 q-pb-sm")
        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            symbol_in = ui.input("Target Symbol", value="BTC-USD").classes("min-w-[150px]")
            tf_in = ui.select(
                ["1d", "4h", "1h"], value="1d", label="Timeframe",
            ).classes("min-w-[100px]")
            start_in = ui.input("Start Date", value="2023-01-01")
            end_in = ui.input("End Date", value="2024-12-31")
            task_in = ui.select(
                {"classification": "Direction (up/down)", "regression": "Log return"},
                value="classification", label="Task",
            ).classes("min-w-[180px]")
            alpha_in = ui.number("FDR alpha", value=0.10, min=0.01, max=0.5, step=0.01)

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            include_fred_in = ui.switch("Include FRED macro", value=True)
            sentiment_in = ui.input(
                "Sentiment queries (comma-separated)",
                value="",
                placeholder="e.g. bitcoin, ethereum",
            ).classes("flex-grow")

    scan_status = ui.column().classes("w-full q-pt-sm")
    results_area = ui.column().classes("w-full q-pt-md")

    async def run_scan() -> None:
        scan_status.clear()
        results_area.clear()
        with scan_status:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="lg")
                ui.label(
                    f"Running exploration scan on {symbol_in.value} "
                    f"{tf_in.value} — this may take a minute..."
                ).classes("text-grey-5")
        try:
            from ..services import run_exploration_service

            queries = [
                q.strip() for q in (sentiment_in.value or "").split(",") if q.strip()
            ]
            result = await run_exploration_service(
                symbol_str=symbol_in.value,
                timeframe_str=tf_in.value,
                start_str=start_in.value,
                end_str=end_in.value,
                task=task_in.value,
                fdr_alpha=float(alpha_in.value or 0.1),
                include_fred=include_fred_in.value,
                sentiment_queries=queries,
            )
        except Exception as exc:  # noqa: BLE001
            scan_status.clear()
            with scan_status:
                ui.icon("error", size="lg", color="negative")
                ui.label(f"Scan failed: {exc}").classes("text-negative")
            return

        state["last_result"] = result
        scan_status.clear()
        with scan_status:
            ui.icon("check_circle", color="positive")
            ui.label(
                f"Scan complete — {result.n_candidates} candidates tested, "
                f"{result.n_significant} passed FDR correction."
            ).classes("text-positive")
        await _refresh_discoveries_list(results_area, target_symbol=result.target_symbol)

    ui.button(
        "Run Scan", icon="search", on_click=run_scan,
    ).props("color=primary unelevated size=lg").classes("q-mt-sm")

    # ---- Feature drift monitor ----------------------------------------
    ui.separator().classes("q-my-md")
    ui.label("Baseline feature drift").classes("text-subtitle1 q-pb-xs")
    ui.label(
        "Population Stability Index on the Exploration Agent's baseline "
        "features. Significant drift means the older lift tests are "
        "comparing against a distribution that no longer matches the "
        "current market — rerun a scan before acting on those results."
    ).classes("text-caption text-grey-6 q-pb-xs")

    drift_container = ui.column().classes("w-full")

    async def run_drift_scan() -> None:
        drift_container.clear()
        with drift_container:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="md")
                ui.label("Scanning baseline feature drift...").classes("text-grey-5")
        try:
            from ...research.drift_monitor import scan_drift

            drift = await scan_drift(
                symbol_str=symbol_in.value,
                timeframe_str=tf_in.value,
            )
        except Exception as exc:  # noqa: BLE001
            drift_container.clear()
            with drift_container:
                ui.label(f"Drift scan failed: {exc}").classes("text-negative")
            return

        drift_container.clear()
        if not drift.reports:
            with drift_container:
                ui.label(
                    "Not enough bars to scan drift (need at least 40 on the "
                    "selected symbol / timeframe)."
                ).classes("text-warning")
            return

        color = {
            "stable": "positive",
            "moderate": "warning",
            "significant": "negative",
            "insufficient_data": "grey-5",
        }
        icon = {
            "stable": "check_circle",
            "moderate": "warning",
            "significant": "error",
            "insufficient_data": "help",
        }

        with drift_container:
            overall = drift.overall_severity
            with ui.row().classes("items-center gap-2 q-pb-xs"):
                ui.icon(icon.get(overall, "help"), color=color.get(overall, "grey-5"))
                ui.label(f"Overall: {overall.upper()}").classes(
                    f"text-body1 text-weight-medium text-{color.get(overall, 'grey-5')}"
                )

            cols = [
                {"name": "feature", "label": "Feature", "field": "feature"},
                {"name": "psi", "label": "PSI", "field": "psi", "sortable": True},
                {"name": "severity", "label": "Severity", "field": "severity"},
                {"name": "ref_mean", "label": "Ref mean", "field": "ref_mean"},
                {"name": "cur_mean", "label": "Cur mean", "field": "cur_mean"},
                {"name": "ref_std", "label": "Ref std", "field": "ref_std"},
                {"name": "cur_std", "label": "Cur std", "field": "cur_std"},
            ]
            rows = [
                {
                    "feature": r.feature,
                    "psi": f"{r.psi:.4f}",
                    "severity": r.severity,
                    "ref_mean": f"{r.reference_mean:.4f}",
                    "cur_mean": f"{r.current_mean:.4f}",
                    "ref_std": f"{r.reference_std:.4f}",
                    "cur_std": f"{r.current_std:.4f}",
                }
                for r in drift.reports
            ]
            ui.aggrid({
                "columnDefs": cols,
                "rowData": rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"sortable": True, "resizable": True},
            }).classes("w-full")

    ui.button(
        "Scan drift", icon="speed", on_click=run_drift_scan,
    ).props("color=secondary outline size=sm").classes("q-mt-xs q-mb-sm")

    ui.separator().classes("q-my-md")
    ui.label("Recent discoveries").classes("text-subtitle1 q-pb-xs")

    discoveries_container = ui.column().classes("w-full")

    async def initial_load() -> None:
        await _refresh_discoveries_list(discoveries_container)

    ui.timer(0.1, initial_load, once=True)


async def _refresh_discoveries_list(container, *, target_symbol: str | None = None) -> None:
    from ..services import list_discoveries

    container.clear()
    try:
        rows = await list_discoveries(target_symbol=target_symbol, limit=200)
    except Exception as exc:  # noqa: BLE001
        with container:
            ui.label(f"Failed to load discoveries: {exc}").classes("text-negative")
        return

    if not rows:
        with container:
            ui.label(
                "No discoveries yet — run a scan above to seed the list."
            ).classes("text-grey-6 text-caption")
        return

    # AG Grid table of all candidates
    columns = [
        {"name": "created", "label": "When", "field": "created", "align": "left"},
        {"name": "candidate", "label": "Candidate", "field": "candidate", "align": "left"},
        {"name": "source", "label": "Source", "field": "source", "align": "left"},
        {"name": "target", "label": "Target", "field": "target", "align": "left"},
        {"name": "lift", "label": "Lift", "field": "lift", "align": "right", "sortable": True},
        {"name": "p_value", "label": "p", "field": "p_value", "align": "right"},
        {"name": "q_value", "label": "q (FDR)", "field": "q_value", "align": "right"},
        {"name": "significant", "label": "Sig?", "field": "significant", "align": "center"},
        {"name": "status", "label": "Status", "field": "status", "align": "left"},
    ]

    def _fmt(x: float | None, digits: int = 4) -> str:
        if x is None:
            return "—"
        return f"{x:.{digits}f}"

    table_rows = []
    for r in rows:
        table_rows.append({
            "id": str(r.id),
            "created": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
            "candidate": r.candidate_name,
            "source": r.candidate_source,
            "target": f"{r.target_symbol} @ {r.target_timeframe}",
            "lift": _fmt(r.lift, 4),
            "p_value": _fmt(r.p_value, 4),
            "q_value": _fmt(r.q_value, 4),
            "significant": "✓" if r.significant else "",
            "status": r.status,
        })

    with container:
        ui.aggrid({
            "columnDefs": columns,
            "rowData": table_rows,
            "domLayout": "autoHeight",
            "defaultColDef": {"sortable": True, "resizable": True},
        }).classes("w-full")

        # Action rows for significant + pending discoveries: promote or dismiss.
        actionable = [r for r in rows if r.significant and r.status == "new"]
        if actionable:
            ui.label(
                "Significant discoveries awaiting review"
            ).classes("text-subtitle2 q-pt-md")
            for r in actionable[:10]:
                _render_discovery_action_row(r, container)


def _render_discovery_action_row(discovery, outer_container) -> None:
    """Promote/dismiss buttons for one significant discovery."""
    status_text = ui.label(f"Status: {discovery.status}").classes(
        "text-caption text-grey-5"
    )

    async def _set_status(new_status: str) -> None:
        from ..services import update_discovery_status

        try:
            await update_discovery_status(discovery.id, new_status)
            status_text.text = f"Status: {new_status}"
            color = (
                "positive" if new_status == "promoted"
                else "negative" if new_status == "dismissed"
                else "grey-5"
            )
            status_text.classes(replace=f"text-caption text-{color}")
            ui.notify(
                f"{discovery.candidate_name} → {new_status}",
                type="positive" if new_status == "promoted" else "warning",
            )
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Action failed: {exc}", type="negative")

    with ui.card().classes("w-full q-pa-sm q-mb-xs"):
        with ui.row().classes("w-full items-center gap-3 no-wrap"):
            ui.badge(discovery.candidate_source, color="primary").props("outline")
            ui.label(discovery.candidate_name).classes("text-body2 text-weight-medium")
            ui.label(
                f"lift {discovery.lift:+.3f} · q "
                f"{f'{discovery.q_value:.3f}' if discovery.q_value is not None else '—'}"
            ).classes("text-caption text-grey-5")
            ui.space()
            status_text
            ui.button(
                "Promote", icon="north",
                on_click=lambda _e=None: _set_status("promoted"),
            ).props("color=positive dense flat")
            ui.button(
                "Dismiss", icon="south",
                on_click=lambda _e=None: _set_status("dismissed"),
            ).props("color=negative dense flat")
