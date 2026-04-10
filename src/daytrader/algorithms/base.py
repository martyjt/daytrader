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


@dataclass(frozen=True)
class AlgorithmManifest:
    """Metadata describing an algorithm plugin."""

    id: str
    name: str
    version: str = "0.1.0"
    description: str = ""
    asset_classes: list[str] = field(
        default_factory=lambda: ["crypto", "equities"]
    )
    timeframes: list[str] = field(default_factory=lambda: ["1d"])
    params: dict[str, Any] = field(default_factory=dict)
    author: str = ""


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
