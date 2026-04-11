"""Research Lab — advanced research tooling for algorithm analysis.

Four tabs:
    1. Model Comparison — run multiple algos side-by-side
    2. Hyperparameter Sweep — grid search over algo params
    3. Feature Attribution — visualize ML feature importance
    4. Walk-Forward Stability — analyze OOS consistency
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


def _kpi_card(label: str, value: str, *, color: str = "primary") -> None:
    with ui.card().classes("q-pa-sm min-w-[110px]"):
        ui.label(label).classes("text-caption text-grey-6")
        ui.label(value).classes(f"text-h6 text-weight-bold text-{color}")
