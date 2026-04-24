"""Algorithm ABC and manifest.

Every algorithm (built-in or user plugin) subclasses ``Algorithm`` and
implements ``on_bar(ctx)``. The ``AlgorithmContext`` sandbox ensures
algorithms can only read features and emit signals — no DB, broker, or
global state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..core.context import AlgorithmContext
from ..core.types.signals import Signal
from ..core.types.visualize import PlotTrace, VisualizeContext


@dataclass(frozen=True)
class AlgorithmParam:
    """Typed parameter declaration for auto-generated UI forms."""

    name: str
    type: str = "float"  # "int" | "float" | "bool" | "str"
    default: Any = 0
    min: float | None = None
    max: float | None = None
    step: float | None = None
    description: str = ""
    choices: list[Any] | None = None


@dataclass(frozen=True)
class AlgorithmManifest:
    """Metadata describing an algorithm plugin.

    ``suitable_regimes`` lets algorithms declare where they shine:
    a trend-follower sets ``["bull", "bear"]``, a mean-reverter sets
    ``["sideways"]``. The Strategy Lab warns when you run an algorithm
    outside its suitable regimes and the live loop can optionally gate
    execution on regime match (driven by the Regime Badge service).
    Default ``None`` means "no opinion" — the algorithm is expected to
    handle all regimes.
    """

    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    asset_classes: list[str] = field(
        default_factory=lambda: ["crypto", "equities"]
    )
    timeframes: list[str] = field(default_factory=lambda: ["1d"])
    params: list[AlgorithmParam] = field(default_factory=list)
    author: str = ""
    # Empty / None = agnostic. Non-empty subset of {"bull","bear","sideways"}.
    suitable_regimes: list[str] | None = None

    def param_defaults(self) -> dict[str, Any]:
        """Return a dict of parameter names to their default values."""
        return {p.name: p.default for p in self.params}


class Algorithm(ABC):
    """Base class for all trading algorithms.

    Subclasses must implement:

    * ``manifest`` — metadata (id, name, supported assets/timeframes)
    * ``on_bar(ctx)`` — process one bar, optionally emit a signal

    The runtime creates one instance per persona per run. Instance state
    (e.g. tracking whether you've entered a position) is therefore
    scoped to a single run and reset automatically.
    """

    @property
    @abstractmethod
    def manifest(self) -> AlgorithmManifest:
        """Algorithm metadata."""

    @abstractmethod
    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        """Process one bar. Return ``ctx.emit(...)`` or ``None``."""

    def warmup_bars(self) -> int:
        """Minimum history bars needed before the algorithm can emit.

        Override if your algo needs lookback (e.g. 200 for a 200-period
        moving average).
        """
        return 0

    def visualize(self, vctx: VisualizeContext) -> list[PlotTrace]:
        """Return chart traces describing this algorithm's native view.

        Default: no traces. The Charts Workbench falls back to plotting
        the signal score / confidence time-series for algos that don't
        override this.

        Override to expose your indicator internals — MACD histogram,
        RSI line, Bollinger bands, etc. Traces align element-wise with
        the OHLCV arrays passed in ``vctx``.
        """
        return []
