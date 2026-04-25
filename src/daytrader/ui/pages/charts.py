"""Charts Workbench — see what the algorithms actually do.

Purpose-built page for visually inspecting algorithm behavior on real
price data. Users pick one or more algorithms from the registry, a
symbol, a timeframe, and a date range; the page runs each algorithm
bar-by-bar and renders:

* A candlestick + volume primary pane with signal markers and any
  "price-overlay" indicator traces (EMA lines, Bollinger bands, the
  Ichimoku cloud, etc.)
* One sub-pane per algorithm that exposes "own-panel" traces via
  ``Algorithm.visualize()`` — MACD histogram, RSI line with thresholds,
  etc. Algorithms that don't override ``visualize()`` fall back to a
  score / confidence line in their pane.
* A per-algorithm **agreement ribbon** at the bottom: colored strips
  (green=long / red=short / gray=flat) side-by-side so the user can
  instantly see where algorithms concur or diverge.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ..components.dag_render import DagRenderNode, dag_to_mermaid
from ..shell import page_layout


# Bounded color palette for per-algorithm fallback "own" traces.
_FALLBACK_COLORS = [
    "#5c7cfa", "#22b8cf", "#f76707", "#40c057",
    "#e64980", "#845ef7", "#fab005", "#15aabf",
]


@ui.page("/charts")
async def charts_page() -> None:
    if not page_layout("Charts Workbench"):
        return

    from ...algorithms.registry import AlgorithmRegistry

    algo_ids = AlgorithmRegistry.available()
    algo_labels = {
        aid: AlgorithmRegistry.get(aid).manifest.name for aid in algo_ids
    }

    # ---- Shared state ------------------------------------------------------
    state: dict[str, Any] = {"last_result": None}

    # ---- Controls ----------------------------------------------------------
    with ui.card().classes("w-full"):
        ui.label("Configure").classes("text-h6 q-pb-sm")

        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            symbol = ui.input(
                "Symbol", value="BTC-USD", placeholder="e.g. BTC-USD, AAPL",
            )
            timeframe = ui.select(
                ["1d", "4h", "1h", "15m", "5m", "1w"],
                value="1d",
                label="Timeframe",
            ).classes("min-w-[120px]")
            start_date = ui.input("Start Date", value="2024-01-01")
            end_date = ui.input("End Date", value="2024-12-31")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            # Default to a handful of visually-distinct algorithms if they
            # exist in the registry.
            preferred = [
                aid for aid in ("ema_crossover", "rsi_mean_reversion", "macd_signal")
                if aid in algo_ids
            ]
            algo_picker = ui.select(
                options={aid: algo_labels[aid] for aid in algo_ids},
                value=preferred or (algo_ids[:3] if algo_ids else []),
                label="Algorithms",
                multiple=True,
            ).props("use-chips").classes("min-w-[420px] flex-grow")

    # ---- Render area -------------------------------------------------------
    render_area = ui.column().classes("w-full q-pt-md")

    async def on_load() -> None:
        render_area.clear()
        if not symbol.value or not algo_picker.value:
            with render_area:
                ui.label("Pick a symbol and at least one algorithm.").classes(
                    "text-grey-5"
                )
            return

        with render_area:
            with ui.row().classes("items-center gap-2"):
                ui.spinner(size="lg")
                ui.label(
                    f"Fetching {symbol.value} and running "
                    f"{len(algo_picker.value)} algorithm(s)…"
                ).classes("text-grey-5")

        try:
            from ..services_charts import run_charts_service

            result = await run_charts_service(
                symbol_str=symbol.value,
                timeframe_str=timeframe.value,
                start_str=start_date.value,
                end_str=end_date.value,
                algo_ids=list(algo_picker.value),
            )
        except Exception as exc:  # noqa: BLE001
            render_area.clear()
            with render_area:
                ui.icon("error", size="lg", color="negative")
                ui.label(f"Chart run failed: {exc}").classes(
                    "text-negative text-body1"
                )
            return

        state["last_result"] = result

        if not result.timestamps:
            render_area.clear()
            with render_area:
                ui.icon("info", size="lg", color="warning")
                ui.label(
                    f"No OHLCV data for {symbol.value} on {timeframe.value} "
                    f"in the requested range."
                ).classes("text-warning text-body1")
            return

        render_area.clear()
        with render_area:
            _render_summary(result)
            _render_chart(result)
            _render_dag_compositions(result)
            _render_agreement_legend(result)

    ui.button(
        "Load", icon="play_arrow", on_click=on_load,
    ).props("color=primary unelevated size=lg").classes("q-mt-sm")


def _render_summary(result) -> None:
    n_bars = len(result.timestamps)
    n_algos = len(result.algorithms)
    total_signals = sum(len(a.signals) for a in result.algorithms)
    ui.label(
        f"{result.symbol} · {result.timeframe} · {n_bars} bars · "
        f"{n_algos} algorithm(s) · {total_signals} signals emitted"
    ).classes("text-caption text-grey-6 q-pb-sm")

    for algo in result.algorithms:
        if algo.error:
            with ui.card().classes("w-full q-pa-sm q-mb-xs").style(
                "background-color: #3a1a1a"
            ):
                ui.label(f"{algo.algo_name}: {algo.error}").classes(
                    "text-caption text-negative"
                )


def _render_chart(result) -> None:
    """Build a single ECharts option with stacked grids and render it."""
    option = _build_echarts_option(result)
    n_panes = 2 + len(result.algorithms) + 1  # price + volume + N algos + ribbon
    # Give each pane breathing room; ribbon stays compact.
    height_px = 180 + 80 + 160 * len(result.algorithms) + 80 + 40
    ui.echart(option).classes("w-full").style(f"height: {height_px}px")


def _render_dag_compositions(result) -> None:
    """For every DAG-typed algo in the run, render its topology with latest scores.

    Uses mermaid for the graph layout — each node is color-coded by its
    latest score direction and carries a score/confidence badge.
    """
    dag_runs = [a for a in result.algorithms if a.dag_nodes]
    if not dag_runs:
        return

    ui.separator().classes("q-my-md")
    ui.label("DAG composition attribution").classes("text-subtitle1 q-pb-xs")
    ui.label(
        "Each DAG node labelled with its latest emitted score "
        "(green = long, red = short, grey = no contribution this run). "
        "Edges show the combinator feed direction."
    ).classes("text-caption text-grey-6 q-pb-sm")

    for algo in dag_runs:
        with ui.card().classes("w-full q-mb-sm"):
            ui.label(algo.algo_name).classes("text-subtitle2 q-pb-xs")
            render_nodes = [
                DagRenderNode(
                    node_id=n.node_id,
                    node_type=n.node_type,
                    label=n.label,
                    parents=list(n.parents),
                    latest_score=n.latest_score,
                )
                for n in algo.dag_nodes
            ]
            ui.mermaid(dag_to_mermaid(render_nodes)).classes("w-full")

            # Companion table below the diagram with the same data in tabular form.
            cols = [
                {"name": "node", "label": "Node", "field": "node"},
                {"name": "type", "label": "Type", "field": "type"},
                {"name": "score", "label": "Score", "field": "score"},
                {"name": "conf", "label": "Confidence", "field": "conf"},
                {"name": "weight", "label": "Weight", "field": "weight"},
            ]
            rows = [
                {
                    "node": n.node_id,
                    "type": n.node_type,
                    "score": f"{n.latest_score:+.3f}" if n.latest_score is not None else "—",
                    "conf": f"{n.latest_confidence:.2f}" if n.latest_confidence is not None else "—",
                    "weight": f"{n.weight:.2f}",
                }
                for n in algo.dag_nodes
            ]
            ui.aggrid({
                "columnDefs": cols,
                "rowData": rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"resizable": True},
            }).classes("w-full q-pt-xs")


def _render_agreement_legend(result) -> None:
    ui.separator().classes("q-my-md")
    ui.label("Agreement ribbon legend").classes("text-subtitle2 q-pb-xs")
    with ui.row().classes("gap-4 items-center"):
        with ui.row().classes("items-center gap-1"):
            ui.element("div").style(
                "width: 14px; height: 14px; background:#40c057; border-radius:2px"
            )
            ui.label("Long").classes("text-caption text-grey-5")
        with ui.row().classes("items-center gap-1"):
            ui.element("div").style(
                "width: 14px; height: 14px; background:#6c757d; border-radius:2px"
            )
            ui.label("Flat").classes("text-caption text-grey-5")
        with ui.row().classes("items-center gap-1"):
            ui.element("div").style(
                "width: 14px; height: 14px; background:#fa5252; border-radius:2px"
            )
            ui.label("Short").classes("text-caption text-grey-5")
    ui.label(
        "Each row is one algorithm. The ribbon makes algorithm agreement "
        "(all-green or all-red columns) and divergence (mixed columns) "
        "visible at a glance."
    ).classes("text-caption text-grey-6 q-pt-xs")


# ---------------------------------------------------------------------------
# ECharts option builder
# ---------------------------------------------------------------------------


def _build_echarts_option(result) -> dict[str, Any]:
    """Compose the ECharts option: grids, axes, and per-pane series."""
    x_labels = [_fmt_timestamp(t) for t in result.timestamps]
    n_algos = len(result.algorithms)

    # --- Vertical layout --------------------------------------------------
    # Distribute heights as percentages of the chart container. The layout
    # adapts to the algorithm count: each algo pane gets a fixed share, the
    # price pane absorbs whatever's left.
    margins = {"left": "8%", "right": "4%"}
    algo_pane_pct = 12.0
    volume_pct = 7.0
    ribbon_pct = 6.0
    zoom_pct = 5.0
    reserved = volume_pct + ribbon_pct + zoom_pct + algo_pane_pct * n_algos
    price_pct = max(25.0, 100.0 - reserved - 4.0)

    grids: list[dict[str, Any]] = []
    x_axes: list[dict[str, Any]] = []
    y_axes: list[dict[str, Any]] = []

    cursor = 2.0  # top padding %
    # Grid 0: price
    grids.append({**margins, "top": f"{cursor}%", "height": f"{price_pct}%"})
    x_axes.append(_x_axis(x_labels, grid_index=0, show_labels=False))
    y_axes.append({
        "gridIndex": 0, "scale": True,
        "axisLabel": {"color": "#999"},
        "splitLine": {"lineStyle": {"color": "#2a2b3a"}},
    })
    cursor += price_pct + 1.0

    # Grid 1: volume
    grids.append({**margins, "top": f"{cursor}%", "height": f"{volume_pct}%"})
    x_axes.append(_x_axis(x_labels, grid_index=1, show_labels=False))
    y_axes.append({
        "gridIndex": 1, "axisLabel": {"color": "#666", "fontSize": 9},
        "splitNumber": 2,
        "splitLine": {"show": False},
    })
    cursor += volume_pct + 1.0

    # Grids 2..2+n-1: per-algorithm own panes
    for i in range(n_algos):
        grid_idx = 2 + i
        grids.append({**margins, "top": f"{cursor}%", "height": f"{algo_pane_pct}%"})
        x_axes.append(_x_axis(x_labels, grid_index=grid_idx, show_labels=False))
        algo = result.algorithms[i]
        y_bounds = _pane_y_bounds(algo)
        y_axis_cfg: dict[str, Any] = {
            "gridIndex": grid_idx,
            "axisLabel": {"color": "#999", "fontSize": 10},
            "splitLine": {"lineStyle": {"color": "#2a2b3a"}},
            "name": algo.algo_name,
            "nameLocation": "middle",
            "nameGap": 44,
            "nameTextStyle": {"color": "#999", "fontSize": 10},
        }
        if y_bounds is not None:
            y_axis_cfg["min"] = y_bounds[0]
            y_axis_cfg["max"] = y_bounds[1]
        y_axes.append(y_axis_cfg)
        cursor += algo_pane_pct + 1.0

    # Grid last: agreement ribbon
    ribbon_grid_index = 2 + n_algos
    grids.append({**margins, "top": f"{cursor}%", "height": f"{ribbon_pct}%"})
    x_axes.append(_x_axis(x_labels, grid_index=ribbon_grid_index, show_labels=True))
    ribbon_categories = [a.algo_name for a in result.algorithms] or ["(no algos)"]
    y_axes.append({
        "gridIndex": ribbon_grid_index,
        "type": "category",
        "data": ribbon_categories,
        "axisLabel": {"color": "#bbb", "fontSize": 10},
        "splitLine": {"show": False},
    })

    # --- Series -----------------------------------------------------------
    series: list[dict[str, Any]] = []

    # Candlestick
    candles = [
        [result.opens[i], result.closes[i], result.lows[i], result.highs[i]]
        for i in range(len(result.timestamps))
    ]
    series.append({
        "name": result.symbol,
        "type": "candlestick",
        "xAxisIndex": 0,
        "yAxisIndex": 0,
        "data": candles,
        "itemStyle": {
            "color": "#40c057",          # up body
            "color0": "#fa5252",         # down body
            "borderColor": "#2f9e44",
            "borderColor0": "#c92a2a",
        },
    })

    # Price-overlay traces from all algorithms
    for algo_idx, algo in enumerate(result.algorithms):
        for trace in algo.traces:
            if trace.panel == "price":
                series.extend(_trace_to_series(trace, grid_x_index=0, grid_y_index=0))

    # Buy / sell markers on the price pane (one scatter per algo)
    for algo_idx, algo in enumerate(result.algorithms):
        buy_points, sell_points = _extract_markers(algo, result)
        color = _FALLBACK_COLORS[algo_idx % len(_FALLBACK_COLORS)]
        if buy_points:
            series.append({
                "name": f"{algo.algo_name} BUY",
                "type": "scatter",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": buy_points,
                "symbol": "triangle",
                "symbolSize": 10,
                "itemStyle": {"color": "#40c057", "borderColor": color, "borderWidth": 1},
                "tooltip": {"formatter": f"{algo.algo_name} BUY @ {{c1}}"},
            })
        if sell_points:
            series.append({
                "name": f"{algo.algo_name} SELL",
                "type": "scatter",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "data": sell_points,
                "symbol": "triangle",
                "symbolRotate": 180,
                "symbolSize": 10,
                "itemStyle": {"color": "#fa5252", "borderColor": color, "borderWidth": 1},
                "tooltip": {"formatter": f"{algo.algo_name} SELL @ {{c1}}"},
            })

    # Volume bars (colored by up/down close)
    vol_data = [
        {
            "value": result.volumes[i],
            "itemStyle": {
                "color": "#40c05744" if result.closes[i] >= result.opens[i] else "#fa525244",
            },
        }
        for i in range(len(result.timestamps))
    ]
    series.append({
        "name": "Volume",
        "type": "bar",
        "xAxisIndex": 1,
        "yAxisIndex": 1,
        "data": vol_data,
    })

    # Per-algo own-panel traces (or score fallback)
    for i, algo in enumerate(result.algorithms):
        grid_idx = 2 + i
        own_traces = [t for t in algo.traces if t.panel == "own"]
        if own_traces:
            for trace in own_traces:
                series.extend(_trace_to_series(trace, grid_x_index=grid_idx, grid_y_index=grid_idx))
        else:
            # Fallback: plot score series as a step line in the own pane.
            series.append({
                "name": f"{algo.algo_name} score",
                "type": "line",
                "xAxisIndex": grid_idx,
                "yAxisIndex": grid_idx,
                "data": algo.scores,
                "step": "middle",
                "showSymbol": False,
                "lineStyle": {"color": _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)], "width": 1.5},
                "areaStyle": {"opacity": 0.12, "color": _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)]},
            })

    # Agreement ribbon via colored scatter markers (simpler than heatmap,
    # no visualMap required — each point carries its own color).
    ribbon_colors = {1: "#40c057", -1: "#fa5252", 0: "#6c757d"}
    ribbon_data: list[dict[str, Any]] = []
    for algo_idx, algo in enumerate(result.algorithms):
        for bar_idx, direction in enumerate(algo.directions):
            ribbon_data.append({
                "value": [bar_idx, algo_idx],
                "itemStyle": {"color": ribbon_colors.get(int(direction), "#6c757d")},
            })
    if ribbon_data:
        series.append({
            "name": "Agreement",
            "type": "scatter",
            "xAxisIndex": ribbon_grid_index,
            "yAxisIndex": ribbon_grid_index,
            "data": ribbon_data,
            "symbol": "rect",
            "symbolSize": [6, 18],
            "large": True,
            "largeThreshold": 200,
        })

    # --- DataZoom (synchronized) ------------------------------------------
    x_axis_indices = list(range(len(x_axes)))
    data_zoom = [
        {"type": "inside", "xAxisIndex": x_axis_indices},
        {
            "type": "slider",
            "xAxisIndex": x_axis_indices,
            "bottom": "1%",
            "height": f"{zoom_pct - 2}%",
            "backgroundColor": "#1a1b2e",
            "fillerColor": "#5c7cfa22",
            "borderColor": "#2a2b3a",
            "textStyle": {"color": "#999"},
        },
    ]

    option: dict[str, Any] = {
        "backgroundColor": "transparent",
        "animation": False,
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "cross", "link": [{"xAxisIndex": "all"}]},
            "backgroundColor": "rgba(26,27,46,0.9)",
            "borderColor": "#2a2b3a",
            "textStyle": {"color": "#eee"},
        },
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "legend": {
            "top": "0%",
            "textStyle": {"color": "#bbb", "fontSize": 10},
            "itemHeight": 8,
        },
        "grid": grids,
        "xAxis": x_axes,
        "yAxis": y_axes,
        "series": series,
        "dataZoom": data_zoom,
    }
    return option


def _x_axis(x_labels: list[str], *, grid_index: int, show_labels: bool) -> dict[str, Any]:
    return {
        "gridIndex": grid_index,
        "type": "category",
        "data": x_labels,
        "boundaryGap": True,
        "axisLine": {"lineStyle": {"color": "#2a2b3a"}},
        "axisTick": {"show": show_labels},
        "axisLabel": {
            "show": show_labels,
            "color": "#999",
            "fontSize": 10,
        },
        "splitLine": {"show": False},
    }


def _fmt_timestamp(t: Any) -> str:
    """Best-effort ISO-ish formatting for chart x-axis labels."""
    if hasattr(t, "isoformat"):
        return t.isoformat()
    return str(t)


def _pane_y_bounds(algo) -> tuple[float, float] | None:
    """Infer y-axis bounds for the algorithm's pane from its traces."""
    bounds: list[tuple[float, float]] = []
    for tr in algo.traces:
        if tr.panel == "own" and tr.y_min is not None and tr.y_max is not None:
            bounds.append((tr.y_min, tr.y_max))
    if not bounds:
        return None
    return (min(b[0] for b in bounds), max(b[1] for b in bounds))


def _trace_to_series(trace, *, grid_x_index: int, grid_y_index: int) -> list[dict[str, Any]]:
    """Convert a PlotTrace into one or more ECharts series entries."""
    base = {
        "xAxisIndex": grid_x_index,
        "yAxisIndex": grid_y_index,
        "name": trace.name,
        "showSymbol": False,
    }

    # Reference lines (thresholds/bands) attach via markLine on whatever
    # series we emit for this trace — avoids spawning an empty-data series
    # just to host them, which breaks layout in some ECharts configs.
    mark_line = None
    if trace.reference_lines:
        mark_line = {
            "silent": True,
            "symbol": "none",
            "data": [
                {
                    "yAxis": y,
                    "name": label,
                    "lineStyle": {"color": col, "type": "dashed", "width": 1},
                    "label": {"color": col, "fontSize": 9, "formatter": label},
                }
                for label, y, col in trace.reference_lines
            ],
        }

    out: list[dict[str, Any]] = []

    if trace.kind == "line":
        line_style: dict[str, Any] = {"color": trace.color, "width": 1.5}
        if trace.dash == "dashed":
            line_style["type"] = "dashed"
        elif trace.dash == "dotted":
            line_style["type"] = "dotted"
        series = {
            **base,
            "type": "line",
            "data": trace.data,
            "lineStyle": line_style,
            "itemStyle": {"color": trace.color},
            "connectNulls": False,
            "sampling": "lttb",
        }
        if mark_line:
            series["markLine"] = mark_line
            mark_line = None
        out.append(series)
    elif trace.kind == "histogram":
        bar_data = [
            {
                "value": v,
                "itemStyle": {"color": trace.color if v is not None and v >= 0 else "#fa525288"},
            }
            if v is not None else {"value": None}
            for v in trace.data
        ]
        series = {**base, "type": "bar", "data": bar_data, "large": True}
        if mark_line:
            series["markLine"] = mark_line
            mark_line = None
        out.append(series)
    elif trace.kind == "band":
        # Render as two semi-transparent lines. Simpler than the stacked
        # invisible-line trick and doesn't break when data has gaps.
        uppers = [row[0] if row else None for row in trace.data]
        lowers = [row[1] if row else None for row in trace.data]
        upper_series = {
            **base,
            "name": f"{trace.name} upper",
            "type": "line",
            "data": uppers,
            "lineStyle": {"color": trace.color, "width": 1, "opacity": trace.opacity + 0.4},
            "itemStyle": {"color": trace.color},
            "connectNulls": False,
        }
        lower_series = {
            **base,
            "name": f"{trace.name} lower",
            "type": "line",
            "data": lowers,
            "lineStyle": {"color": trace.color, "width": 1, "opacity": trace.opacity + 0.4},
            "itemStyle": {"color": trace.color},
            "connectNulls": False,
        }
        out.append(upper_series)
        out.append(lower_series)
    elif trace.kind == "threshold":
        value = trace.data[0] if trace.data else 0
        # Render the threshold itself as the mark line; no data series.
        out.append({
            **base,
            "type": "line",
            "data": [None] * len(trace.data) if len(trace.data) > 1 else [],
            "markLine": {
                "silent": True,
                "symbol": "none",
                "lineStyle": {"color": trace.color, "type": "dashed"},
                "data": [{"yAxis": value, "name": trace.name}],
            },
        })
        mark_line = None  # already consumed above
    else:
        # background/unknown — skip
        pass

    # If we still have unclaimed reference lines (e.g. nothing in 'out' to
    # attach to), drop them rather than emit an empty-data placeholder.
    return out


def _extract_markers(algo, result) -> tuple[list[list[Any]], list[list[Any]]]:
    """Split signal emissions into buy / sell marker datasets.

    Returns two lists of ``[bar_index, price]`` points.
    """
    buys: list[list[Any]] = []
    sells: list[list[Any]] = []
    for bar_idx, sig in algo.signals:
        if bar_idx >= len(result.closes):
            continue
        price = result.closes[bar_idx]
        point = [bar_idx, price]
        if sig.score > 0:
            buys.append(point)
        elif sig.score < 0:
            sells.append(point)
    return buys, sells
