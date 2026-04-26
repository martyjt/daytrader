"""Feature-threshold algorithm — generic baseline for promoted discoveries.

The Exploration Agent finds candidate features that *predict* next-bar
returns; ``feature_threshold`` is the simplest possible algorithm that
can *trade* on one. It reads a single named feature from
``ctx.features`` and emits long / short / flat based on whether the
feature crosses configurable thresholds.

This algorithm exists so promoting a Discovery has a meaningful
landing spot without writing a bespoke subclass per discovery. The
trading loop hydrates the discovered feature into ``ctx.features``
before each bar (see ``research.feature_hydration``); this algorithm
is what turns it into orders.

Params:

* ``feature_name`` — the key in ``ctx.features`` to read (e.g.
  ``"fred:DGS10"``). Required; no default.
* ``upper_threshold`` — emit ``+1`` (long) when feature > this.
* ``lower_threshold`` — emit ``-1`` (short) when feature < this.
  When ``upper`` and ``lower`` are both 0 (the default) the algorithm
  emits long whenever the value is non-zero positive — a sane
  starting point for normalized scores like sentiment.
* ``direction`` — ``"long_only"`` (default), ``"short_only"``, or
  ``"both"``. Filters the emitted side; missing-feature is always
  flat.
"""

from __future__ import annotations

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam


class FeatureThresholdAlgorithm(Algorithm):
    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="feature_threshold",
            name="Feature Threshold",
            version="1.0.0",
            description=(
                "Reads one named feature from ctx.features and emits "
                "long/short/flat by threshold crossing. Used as the "
                "baseline algorithm for promoted Discovery rows."
            ),
            asset_classes=["crypto", "equities", "forex", "commodities"],
            timeframes=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
            params=[
                AlgorithmParam(
                    name="feature_name",
                    type="str",
                    default="",
                    description=(
                        "Key in ctx.features (e.g. 'fred:DGS10'). Required."
                    ),
                ),
                AlgorithmParam(
                    name="upper_threshold",
                    type="float",
                    default=0.0,
                    description="Emit long when feature > this.",
                ),
                AlgorithmParam(
                    name="lower_threshold",
                    type="float",
                    default=0.0,
                    description="Emit short when feature < this.",
                ),
                AlgorithmParam(
                    name="direction",
                    type="str",
                    default="long_only",
                    choices=["long_only", "short_only", "both"],
                    description="Restrict emitted side.",
                ),
            ],
            author="Daytrader built-in",
        )

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        feature_name = ctx.param("feature_name", "") or ""
        if not feature_name:
            ctx.log("feature_threshold: no feature_name configured")
            return None

        value = ctx.feature(feature_name, default=None)
        if value is None:
            # Missing feature → stay flat. The trading loop logs an
            # explicit warning once when hydration returns None, so
            # we don't double-log here.
            return None

        upper = float(ctx.param("upper_threshold", 0.0))
        lower = float(ctx.param("lower_threshold", 0.0))
        direction = str(ctx.param("direction", "long_only")).lower()

        score = 0.0
        if value > upper and direction in ("long_only", "both"):
            score = 1.0
        elif value < lower and direction in ("short_only", "both"):
            score = -1.0

        if score == 0.0:
            return None

        return ctx.emit(
            score=score,
            confidence=1.0,
            reason=(
                f"{feature_name}={value:.4f} "
                f"crossed {'upper' if score > 0 else 'lower'} "
                f"threshold ({upper if score > 0 else lower:.4f})"
            ),
            metadata={"feature_name": feature_name, "feature_value": float(value)},
        )
