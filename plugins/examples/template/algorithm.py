"""Plugin template — copy, rename, fill in.

Workflow:
1. Copy ``plugins/examples/template/`` to ``plugins/<your_id>/``.
2. Edit ``manifest.yaml`` — change ``id``, ``name``, ``params``.
3. Replace the body of ``on_bar`` with your strategy logic.
4. Restart the app (or upload via Plugins → Upload).

The skeleton below emits a neutral/no signal on every bar. As-is it
won't trade — that's intentional. Replace the body with your own
indicator + threshold logic.
"""

from __future__ import annotations

from daytrader.algorithms.base import (
    Algorithm,
    AlgorithmManifest,
    AlgorithmParam,
)
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.signals import Signal


class ExampleTemplate(Algorithm):
    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="example_template",
            name="Example — Template (fork me)",
            version="0.1.0",
            description="Skeleton plugin. Replace this with your own strategy.",
            asset_classes=["crypto", "equities"],
            timeframes=["1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    name="example_threshold",
                    type="float",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    description="Replace me.",
                ),
            ],
            author="Your name here",
        )

    def warmup_bars(self) -> int:
        # Override if your strategy needs N bars of history before its
        # first signal — e.g. a 50-bar SMA needs 50.
        return 0

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        # Replace the body of this method.
        #
        # threshold = float(ctx.param("example_threshold", 0.5))
        # closes = ctx.history(20, "close")
        # if closes[-1] > closes[-20] * (1.0 + threshold):
        #     return ctx.emit(score=1.0, confidence=0.7, reason="momentum")
        return None
