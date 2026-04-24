"""CompositeAlgorithm — wraps a DAG and implements the Algorithm ABC.

The BacktestEngine and WalkForwardEngine see this as a single
Algorithm. On each bar it evaluates all leaf algorithm nodes,
walks combinators in topological order, and returns the root signal.
"""

from __future__ import annotations

import copy
from typing import Any

from ..base import Algorithm, AlgorithmManifest, AlgorithmParam
from ..registry import AlgorithmRegistry
from .combinators import COMBINATORS
from .types import DAGDefinition, DAGNode
from .validation import topological_order, validate
from ...core.context import AlgorithmContext
from ...core.types.signals import Signal, SignalContribution


class CompositeAlgorithm(Algorithm):
    """Wraps a DAGDefinition and implements the Algorithm ABC.

    On each bar:
    1. Evaluate leaf nodes (algorithm nodes) by calling their on_bar()
    2. Walk up the DAG in topological order
    3. At each combinator, merge child signals
    4. Return the root combinator's output Signal
    """

    def __init__(self, dag: DAGDefinition) -> None:
        errors = validate(dag)
        if errors:
            raise ValueError(
                f"Invalid DAG {dag.id!r}: {'; '.join(errors)}"
            )
        self._dag = dag
        self._topo_order = topological_order(dag)
        self._algorithms: dict[str, Algorithm] = {}
        # Per-combinator-node mutable state (used by rolling combinators
        # that need to remember signals across bars).
        self._combinator_state: dict[str, dict[str, Any]] = {}
        self._instantiate_algorithms()

    @property
    def manifest(self) -> AlgorithmManifest:
        # Expose every child algorithm's params at the top level so users
        # can tune them from Strategy Lab (or any other UI that reads
        # ``manifest.params``). Names are prefixed with the node id to
        # keep them unique when multiple nodes share the same algorithm.
        composed_params: list[AlgorithmParam] = []
        for node in self._dag.nodes:
            if node.node_type != "algorithm" or not node.algorithm_id:
                continue
            child = self._algorithms.get(node.node_id)
            if child is None:
                continue
            child_name = child.manifest.name
            for p in child.manifest.params:
                # Current override takes precedence over manifest default
                override = node.params.get(p.name, p.default)
                composed_params.append(
                    AlgorithmParam(
                        name=f"{node.node_id}__{p.name}",
                        type=p.type,
                        default=override,
                        min=p.min,
                        max=p.max,
                        step=p.step,
                        choices=p.choices,
                        description=f"[{child_name}] {p.description or p.name}",
                    )
                )

        return AlgorithmManifest(
            id=f"dag:{self._dag.id}",
            name=self._dag.name,
            version=self._dag.version,
            description=self._dag.description,
            params=composed_params,
            asset_classes=["crypto", "equities"],
            timeframes=["1m", "5m", "15m", "1h", "4h", "1d"],
            author="DAG Composer",
        )

    def warmup_bars(self) -> int:
        if not self._algorithms:
            return 0
        return max(a.warmup_bars() for a in self._algorithms.values())

    def train(self, data: Any) -> None:
        """Delegate training to child algorithms that support it."""
        for algo in self._algorithms.values():
            if hasattr(algo, "train") and callable(algo.train):
                # Only call train if it's not the base Algorithm stub
                try:
                    algo.train(data)
                except TypeError:
                    pass

    def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
        node_signals: dict[str, Signal | None] = {}

        for node_id in self._topo_order:
            node = self._dag.get_node(node_id)

            if node.node_type == "algorithm":
                node_ctx = self._make_node_context(ctx, node)
                node_signals[node_id] = self._algorithms[node_id].on_bar(node_ctx)
            else:
                # Combinator / risk_filter
                child_ids = self._dag.children_of(node_id)
                child_signals = [node_signals.get(cid) for cid in child_ids]
                child_weights = [
                    self._dag.get_node(cid).weight for cid in child_ids
                ]
                node_signals[node_id] = self._apply_combinator(
                    node, child_signals, child_weights, ctx,
                )

        root_signal = node_signals.get(self._dag.root_node_id)
        if root_signal is None:
            return None

        # Build attribution tree covering every DAG node.
        attribution = self._build_attribution(self._dag.root_node_id, node_signals)

        # Emit directly so the full attribution tree is preserved on the Signal.
        # (``ctx.emit`` wraps a fresh single-node tree that would discard ours.)
        signal = Signal.new(
            symbol_key=ctx.symbol.key,
            score=root_signal.score,
            confidence=root_signal.confidence,
            source=f"dag:{self._dag.id}",
            reason=f"DAG:{self._dag.id} → {root_signal.reason}",
            attribution=attribution,
            metadata={
                "dag_id": self._dag.id,
                "dag_attribution_root": attribution.node_id if attribution else None,
            },
        )
        ctx.emit_fn(signal)
        return signal

    # ----- internal -------------------------------------------------------

    def _instantiate_algorithms(self) -> None:
        """Create algorithm instances for each algorithm node."""
        for node in self._dag.nodes:
            if node.node_type == "algorithm" and node.algorithm_id:
                # Deep-copy the registry algorithm so each node has its own state
                template = AlgorithmRegistry.get(node.algorithm_id)
                self._algorithms[node.node_id] = copy.deepcopy(template)

    def _make_node_context(
        self, parent_ctx: AlgorithmContext, node: DAGNode,
    ) -> AlgorithmContext:
        """Create a per-node AlgorithmContext with isolated params.

        Param precedence (lowest → highest):
        1. Child algorithm's manifest defaults
        2. DAG-time per-node overrides (``node.params``)
        3. Runtime overrides from the parent context, carrying names
           prefixed with ``{node_id}__`` (lets users tune child params
           from Strategy Lab without editing the saved YAML).
        """
        algo = self._algorithms[node.node_id]
        merged_params = dict(algo.manifest.param_defaults())
        merged_params.update(node.params)

        # Pull runtime overrides for this specific node, stripping the
        # {node_id}__ prefix. Silently ignores params addressed to other
        # nodes.
        prefix = f"{node.node_id}__"
        for k, v in (parent_ctx.params or {}).items():
            if isinstance(k, str) and k.startswith(prefix):
                merged_params[k[len(prefix):]] = v

        # Capture signals locally instead of broadcasting to engine
        captured: list[Signal] = []

        return AlgorithmContext(
            tenant_id=parent_ctx.tenant_id,
            persona_id=parent_ctx.persona_id,
            algorithm_id=node.node_id,
            symbol=parent_ctx.symbol,
            timeframe=parent_ctx.timeframe,
            now=parent_ctx.now,
            bar=parent_ctx.bar,
            history_arrays=parent_ctx.history_arrays,
            features=parent_ctx.features,
            params=merged_params,
            emit_fn=captured.append,
            log_fn=parent_ctx.log_fn,
        )

    def _apply_combinator(
        self,
        node: DAGNode,
        child_signals: list[Signal | None],
        child_weights: list[float],
        ctx: AlgorithmContext,
    ) -> Signal | None:
        """Apply a combinator function and wrap result as a Signal."""
        combinator_fn = COMBINATORS.get(node.combinator_type or "")
        if combinator_fn is None:
            return None

        node_state = self._combinator_state.setdefault(node.node_id, {})
        result = combinator_fn(
            child_signals,
            child_weights,
            params=node.params,
            state=node_state,
        )
        if result is None:
            return None

        score, confidence = result
        return Signal.new(
            symbol_key=ctx.symbol.key,
            score=score,
            confidence=confidence,
            source=node.node_id,
            reason=f"{node.combinator_type}({len([s for s in child_signals if s])} signals)",
        )

    def _build_attribution(
        self,
        node_id: str,
        node_signals: dict[str, Signal | None],
    ) -> SignalContribution | None:
        """Recursively build the attribution tree."""
        sig = node_signals.get(node_id)
        if sig is None:
            return None

        node = self._dag.get_node(node_id)
        child_ids = self._dag.children_of(node_id)
        children = tuple(
            c for cid in child_ids
            if (c := self._build_attribution(cid, node_signals)) is not None
        )

        return SignalContribution(
            node_id=node_id,
            node_type=node.node_type,
            score=sig.score,
            confidence=sig.confidence,
            weight=node.weight,
            reason=sig.reason,
            children=children,
        )
