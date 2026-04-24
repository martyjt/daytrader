"""Thompson-sampling bandit allocator — routes capital between child algos.

Wraps N child algorithms. On each bar every child emits (or doesn't);
the allocator picks ONE arm via Thompson sampling over Gaussian
posteriors fit to recent per-arm rewards, and forwards that arm's
signal as its own output.

Rewards are the per-bar PnL that results from the *previous* selected
arm's signal — credit-assigned one step later. Arms that haven't emitted
signals recently drift back toward the prior so they get re-explored.

Useful as a safe, explainable way to combine heterogeneous strategies
without hard-coding weights: the allocator learns which child works best
in the current regime and shifts capital automatically. Shows up as a
single meta-algorithm in the registry, so it fits anywhere the existing
Algorithm ABC goes (DAG, Strategy Lab, walk-forward, live).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ...core.context import AlgorithmContext
from ...core.types.signals import Signal
from ..base import Algorithm, AlgorithmManifest, AlgorithmParam


@dataclass
class _ArmStats:
    """Running Gaussian posterior for one arm's reward."""

    # Normal-Gamma conjugate prior (we just track running mean/var here;
    # the "prior drift" decay replaces proper Bayesian update for speed).
    mean: float = 0.0
    var: float = 1.0
    n: float = 1.0  # pseudo-count (non-integer to allow decay)

    def sample(self, rng: np.random.Generator) -> float:
        std = math.sqrt(max(1e-9, self.var / max(1.0, self.n)))
        return float(rng.normal(self.mean, std))

    def update(self, reward: float, lr: float = 0.05) -> None:
        # Running EW update; ``lr`` is the learning rate toward the prior when
        # reward is observed.
        self.n = self.n * (1 - lr) + 1.0 * lr  # keeps pseudo-count bounded
        delta = reward - self.mean
        self.mean += lr * delta
        # Welford-style variance update (EW)
        self.var = (1 - lr) * self.var + lr * (delta ** 2)

    def decay(self, factor: float) -> None:
        """Shrink toward prior for arms that haven't been selected recently."""
        self.mean *= factor
        self.var = self.var * factor + (1 - factor) * 1.0  # regress to unit var
        self.n = max(1.0, self.n * factor)


class BanditAllocator(Algorithm):
    """Thompson-sampling bandit over child algorithms."""

    def __init__(
        self,
        children: list[Algorithm] | None = None,
        *,
        learning_rate: float = 0.1,
        decay: float = 0.99,
        seed: int = 0,
    ) -> None:
        self._children: list[Algorithm] = list(children) if children else []
        self._lr = learning_rate
        self._decay = decay
        self._rng = np.random.default_rng(seed)
        self._stats: list[_ArmStats] = [_ArmStats() for _ in self._children]
        self._last_arm: int | None = None
        self._last_price: float | None = None
        self._last_action_sign: int = 0  # -1, 0, +1 for short/flat/long

    @property
    def manifest(self) -> AlgorithmManifest:
        return AlgorithmManifest(
            id="bandit_allocator",
            name="Bandit Allocator (RL)",
            version="1.0.0",
            description=(
                "Thompson-sampling bandit that routes each bar to one of "
                "N child algorithms based on their recent reward. Wraps "
                "a heterogeneous strategy library into a single learned "
                "meta-policy. Children are injected at construction time."
            ),
            asset_classes=["crypto", "equities"],
            timeframes=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
            params=[
                AlgorithmParam(
                    "learning_rate", "float", self._lr,
                    min=0.01, max=0.9, step=0.01,
                    description="EW update rate for arm posteriors",
                ),
                AlgorithmParam(
                    "decay", "float", self._decay,
                    min=0.9, max=1.0, step=0.005,
                    description="Per-bar drift of unused arms back to the prior",
                ),
            ],
            author="Daytrader built-in (RL)",
        )

    def set_children(self, children: list[Algorithm]) -> None:
        """Inject child algorithms (e.g. after loading from a DAG YAML)."""
        self._children = list(children)
        self._stats = [_ArmStats() for _ in self._children]
        self._last_arm = None
        self._last_price = None
        self._last_action_sign = 0

    def warmup_bars(self) -> int:
        if not self._children:
            return 0
        return max(a.warmup_bars() for a in self._children)

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        if not self._children:
            return None

        # --- 1. Credit-assign last arm's choice with this bar's return ---
        cur_price = float(ctx.bar.close)
        if self._last_arm is not None and self._last_price is not None and self._last_price > 0:
            bar_return = (cur_price / self._last_price) - 1.0
            reward = self._last_action_sign * bar_return
            self._stats[self._last_arm].update(reward, lr=self._lr)

        # --- 2. Decay all arms slightly so unused arms regress to prior ---
        for s in self._stats:
            s.decay(self._decay)

        # --- 3. Sample a new arm via Thompson sampling ---
        samples = [stats.sample(self._rng) for stats in self._stats]
        chosen = int(np.argmax(samples))

        # --- 4. Run only the chosen child and forward its signal ---
        chosen_signal = self._children[chosen].on_bar(ctx)

        self._last_arm = chosen
        self._last_price = cur_price
        if chosen_signal is None:
            self._last_action_sign = 0
            return None
        self._last_action_sign = 1 if chosen_signal.score > 0 else -1 if chosen_signal.score < 0 else 0

        # Re-emit the child signal from the allocator's source id so the
        # journal shows who made the call.
        return ctx.emit(
            score=chosen_signal.score,
            confidence=chosen_signal.confidence,
            reason=(
                f"arm[{chosen}]={self._children[chosen].manifest.name}: "
                f"{chosen_signal.reason}"
            ),
            metadata={
                "bandit_arm": chosen,
                "bandit_arm_name": self._children[chosen].manifest.name,
            },
        )
