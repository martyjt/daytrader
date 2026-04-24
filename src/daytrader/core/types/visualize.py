"""Chart traces produced by ``Algorithm.visualize()``.

An algorithm can describe its internal state — the lines, bands, and
histograms that make up its "native view" — by returning a list of
``PlotTrace`` objects. The Charts Workbench renders these alongside the
price pane so users can see exactly what the algorithm computes.

Algorithms that don't override ``visualize()`` get an empty list; the
Charts page falls back to plotting signal score / confidence over time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

# Where the trace renders:
#   - "price"  — overlay on the main candlestick chart (e.g. EMA, BB bands)
#   - "own"    — the algorithm's own pane below the price chart
#   - "volume" — overlay on the volume sub-chart
PlotPanel = Literal["price", "own", "volume"]

# Trace rendering style:
#   - "line"       — continuous line series
#   - "band"       — shaded region between two lines (data is list of [upper, lower] pairs)
#   - "histogram"  — vertical bars (positive/negative auto-colored)
#   - "threshold"  — horizontal reference line at a constant value (data = [y])
#   - "background" — regime shading; data is a list of ints encoding regime id per bar
TraceKind = Literal["line", "band", "histogram", "threshold", "background"]


@dataclass(frozen=True)
class PlotTrace:
    """One chart trace, aligned to the OHLCV bar index.

    For ``kind="line"``, ``data`` is ``list[float|None]`` with length == number of bars.
    For ``kind="band"``, ``data`` is ``list[[float|None, float|None]]``.
    For ``kind="histogram"``, ``data`` is ``list[float|None]``.
    For ``kind="threshold"``, ``data`` is a single-element list ``[y_value]``.
    For ``kind="background"``, ``data`` is ``list[int|None]`` encoding regime id.

    ``None`` values in per-bar arrays render as gaps.
    """

    name: str
    kind: TraceKind
    data: list[Any]
    panel: PlotPanel = "own"
    color: str = "#5c7cfa"
    dash: str | None = None  # None | "dashed" | "dotted"
    opacity: float = 1.0
    # For "own" panels, optional y-axis hints (e.g. RSI sets 0..100).
    y_min: float | None = None
    y_max: float | None = None
    # Optional constant reference lines drawn in the same "own" pane
    # (e.g. RSI 30/70). Each entry is (label, y_value, color).
    reference_lines: tuple[tuple[str, float, str], ...] = ()


@dataclass
class VisualizeContext:
    """Inputs passed to ``Algorithm.visualize()``.

    All arrays are the same length (the full window being charted) and
    aligned element-wise. Params mirror what the algo would receive via
    ``ctx.param`` on each bar.
    """

    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.closes)


def nan_array_to_jsonable(arr: np.ndarray) -> list[float | None]:
    """Convert a numpy array to a JSON-safe list with NaN -> None.

    Exists because ECharts (and JSON) have no native NaN representation
    but happily render ``null`` as a gap.
    """
    out: list[float | None] = []
    for v in arr:
        if v != v:  # NaN check
            out.append(None)
        else:
            out.append(float(v))
    return out
