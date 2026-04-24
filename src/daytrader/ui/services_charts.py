"""Service layer for the Charts Workbench page.

Responsible for:

* Fetching OHLCV bars for a (symbol, timeframe, range) via the standard
  adapter registry, with read-through caching (same cache the backtest
  engine uses — no redundant downloads when the user switches between
  Charts and Strategy Lab on the same range).
* For each selected algorithm, calling ``visualize()`` once to collect
  indicator traces, and replaying ``on_bar`` bar-by-bar to capture the
  signal history (score / confidence / direction per bar).

The service is deliberately simple — no risk filters, no fee modelling,
no position tracking. The Charts page is about *what the algorithm
computes*, not "did it make money".
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from ..algorithms.base import Algorithm
from ..algorithms.registry import AlgorithmRegistry
from ..core.context import AlgorithmContext
from ..core.types.bars import Bar, Timeframe
from ..core.types.signals import Signal
from ..core.types.symbols import Symbol
from ..core.types.visualize import PlotTrace, VisualizeContext


@dataclass
class DAGNodeSnapshot:
    """One DAG node's latest state for rendering the composition widget."""

    node_id: str
    node_type: str  # "algorithm" | "combinator"
    label: str
    latest_score: float | None = None
    latest_confidence: float | None = None
    parents: list[str] = field(default_factory=list)  # upstream node ids
    weight: float = 1.0


@dataclass
class AlgorithmChartRun:
    """Everything the Charts page needs to render one algorithm."""

    algo_id: str
    algo_name: str
    traces: list[PlotTrace]
    # Per-bar signal state, aligned to the OHLCV index:
    #   scores[i]       — emitted signal score (or None if no emission)
    #   confidences[i]  — emitted signal confidence (or None)
    #   directions[i]   — -1, 0, +1 long/flat/short held position sense
    #                     (derived: last emission's sign until a new one flips)
    scores: list[float | None]
    confidences: list[float | None]
    directions: list[int]
    signals: list[tuple[int, Signal]] = field(default_factory=list)
    dag_nodes: list[DAGNodeSnapshot] = field(default_factory=list)
    error: str | None = None


@dataclass
class ChartsRunResult:
    """Combined output of a Charts page run."""

    symbol: str
    timeframe: str
    timestamps: list[Any]
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[float]
    algorithms: list[AlgorithmChartRun]


async def run_charts_service(
    *,
    symbol_str: str,
    timeframe_str: str,
    start_str: str,
    end_str: str,
    algo_ids: list[str],
    algo_params: dict[str, dict[str, Any]] | None = None,
) -> ChartsRunResult:
    """Fetch bars, run each selected algorithm, and return chart-ready data.

    ``algo_params`` maps ``algo_id -> params dict``. Missing entries use
    the algorithm's manifest defaults.
    """
    from ..backtest.engine import _fetch_ohlcv_cached
    from ..data.adapters.registry import AdapterRegistry

    symbol = Symbol.parse(symbol_str)
    timeframe = Timeframe(timeframe_str)
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")

    AdapterRegistry.auto_register()
    adapter = AdapterRegistry.get("yfinance")
    data = await _fetch_ohlcv_cached(adapter, symbol, timeframe, start, end)

    if data.is_empty():
        return ChartsRunResult(
            symbol=symbol_str,
            timeframe=timeframe_str,
            timestamps=[],
            opens=[], highs=[], lows=[], closes=[], volumes=[],
            algorithms=[],
        )

    # Run the CPU-bound replay off the asyncio thread so the NiceGUI
    # websocket heartbeat stays responsive on long windows.
    return await asyncio.to_thread(
        _run_sync,
        data, symbol, timeframe, algo_ids, algo_params or {},
    )


def _run_sync(
    data: pl.DataFrame,
    symbol: Symbol,
    timeframe: Timeframe,
    algo_ids: list[str],
    algo_params: dict[str, dict[str, Any]],
) -> ChartsRunResult:
    opens = data["open"].to_numpy().astype(float)
    highs = data["high"].to_numpy().astype(float)
    lows = data["low"].to_numpy().astype(float)
    closes = data["close"].to_numpy().astype(float)
    volumes = data["volume"].to_numpy().astype(float)
    timestamps = data["timestamp"].to_list()

    runs: list[AlgorithmChartRun] = []
    for algo_id in algo_ids:
        try:
            template = AlgorithmRegistry.get(algo_id)
        except KeyError as exc:
            runs.append(AlgorithmChartRun(
                algo_id=algo_id,
                algo_name=algo_id,
                traces=[],
                scores=[None] * len(closes),
                confidences=[None] * len(closes),
                directions=[0] * len(closes),
                error=str(exc),
            ))
            continue

        algo = copy.deepcopy(template)
        params = algo.manifest.param_defaults()
        params.update(algo_params.get(algo_id, {}))

        try:
            traces = algo.visualize(VisualizeContext(
                opens=opens, highs=highs, lows=lows,
                closes=closes, volumes=volumes,
                params=params,
            )) or []
        except Exception as exc:  # noqa: BLE001 — visualize is untrusted plugin code
            traces = []
            err = f"visualize() failed: {exc}"
        else:
            err = None

        scores, confidences, directions, signals = _replay_on_bar(
            algo, symbol, timeframe, opens, highs, lows, closes, volumes,
            timestamps, params,
        )

        # If this is a CompositeAlgorithm (DAG), snapshot its nodes for the
        # right-rail composition widget. Latest score/confidence come from
        # the final emitted signal's attribution tree.
        dag_nodes: list[DAGNodeSnapshot] = []
        try:
            from ..algorithms.dag.composite import CompositeAlgorithm

            if isinstance(algo, CompositeAlgorithm):
                dag_nodes = _snapshot_dag_nodes(algo, signals)
        except Exception:
            dag_nodes = []

        runs.append(AlgorithmChartRun(
            algo_id=algo_id,
            algo_name=algo.manifest.name,
            traces=list(traces),
            scores=scores,
            confidences=confidences,
            directions=directions,
            signals=signals,
            dag_nodes=dag_nodes,
            error=err,
        ))

    return ChartsRunResult(
        symbol=symbol.key,
        timeframe=timeframe.value,
        timestamps=timestamps,
        opens=opens.tolist(),
        highs=highs.tolist(),
        lows=lows.tolist(),
        closes=closes.tolist(),
        volumes=volumes.tolist(),
        algorithms=runs,
    )


def _snapshot_dag_nodes(
    composite, signals: list[tuple[int, Signal]],
) -> list[DAGNodeSnapshot]:
    """Produce a list of ``DAGNodeSnapshot`` for rendering a composition widget.

    Node ``latest_score`` and ``latest_confidence`` come from the final
    signal's attribution tree (flattened). Nodes that never contributed
    show ``None``.
    """
    dag = composite._dag  # noqa: SLF001 — internal but stable

    # Flatten the most recent attribution tree into a dict by node_id.
    latest_by_node: dict[str, tuple[float, float]] = {}
    if signals:
        _, last_sig = signals[-1]
        root = last_sig.attribution
        if root is not None:
            _flatten_attribution(root, latest_by_node)

    # Build parent map from edges (source -> feeds into -> target).
    # For display, each node lists its upstream inputs (parents = sources feeding it).
    parents_map: dict[str, list[str]] = {n.node_id: [] for n in dag.nodes}
    for edge in dag.edges:
        parents_map.setdefault(edge.target_id, []).append(edge.source_id)

    out: list[DAGNodeSnapshot] = []
    for node in dag.nodes:
        label = node.node_id
        if node.node_type == "algorithm" and node.algorithm_id:
            label = f"{node.node_id}\n({node.algorithm_id})"
        elif node.node_type == "combinator" and node.combinator_type:
            label = f"{node.node_id}\n[{node.combinator_type}]"
        score_conf = latest_by_node.get(node.node_id)
        out.append(DAGNodeSnapshot(
            node_id=node.node_id,
            node_type=node.node_type,
            label=label,
            latest_score=score_conf[0] if score_conf else None,
            latest_confidence=score_conf[1] if score_conf else None,
            parents=parents_map.get(node.node_id, []),
            weight=node.weight,
        ))
    return out


def _flatten_attribution(
    contribution, out: dict[str, tuple[float, float]],
) -> None:
    """Walk the attribution tree and collect (score, confidence) per node_id."""
    try:
        out[contribution.node_id] = (
            float(contribution.score),
            float(contribution.confidence),
        )
    except AttributeError:
        return
    for child in getattr(contribution, "children", ()) or ():
        _flatten_attribution(child, out)


def _replay_on_bar(
    algorithm: Algorithm,
    symbol: Symbol,
    timeframe: Timeframe,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    timestamps: list[Any],
    params: dict[str, Any],
) -> tuple[list[float | None], list[float | None], list[int], list[tuple[int, Signal]]]:
    """Run ``algorithm.on_bar`` over every bar and collect per-bar state."""
    n = len(closes)
    warmup = algorithm.warmup_bars()
    scores: list[float | None] = [None] * n
    confidences: list[float | None] = [None] * n
    directions: list[int] = [0] * n
    signals: list[tuple[int, Signal]] = []

    tenant_id = uuid4()
    persona_id = uuid4()

    current_direction = 0  # -1 short, 0 flat, +1 long — sticky between emissions
    for i in range(n):
        if i < warmup:
            directions[i] = current_direction
            continue

        bar = Bar(
            timestamp=timestamps[i],
            open=Decimal(str(opens[i])),
            high=Decimal(str(highs[i])),
            low=Decimal(str(lows[i])),
            close=Decimal(str(closes[i])),
            volume=Decimal(str(volumes[i])),
        )
        emitted: list[Signal] = []
        ctx = AlgorithmContext(
            tenant_id=tenant_id,
            persona_id=persona_id,
            algorithm_id=algorithm.manifest.id,
            symbol=symbol,
            timeframe=timeframe,
            now=timestamps[i],
            bar=bar,
            history_arrays={
                "close": closes[: i + 1],
                "open": opens[: i + 1],
                "high": highs[: i + 1],
                "low": lows[: i + 1],
                "volume": volumes[: i + 1],
            },
            features={},
            params=params,
            emit_fn=emitted.append,
            log_fn=lambda _msg, _fields: None,
        )
        try:
            algorithm.on_bar(ctx)
        except Exception:  # noqa: BLE001 — replay shouldn't propagate plugin errors
            directions[i] = current_direction
            continue

        if emitted:
            sig = emitted[0]
            scores[i] = float(sig.score)
            confidences[i] = float(sig.confidence)
            signals.append((i, sig))
            if sig.score > 0.05:
                current_direction = 1
            elif sig.score < -0.05:
                current_direction = -1
            # Scores in the neutral band don't flip direction.

        directions[i] = current_direction

    return scores, confidences, directions, signals
