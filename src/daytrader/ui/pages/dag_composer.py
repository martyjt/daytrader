"""DAG Composer — tier-2 visual strategy composition page.

Provides a node/edge editor for composing multiple algorithms into
a single strategy DAG. Uses LiteGraph.js for the canvas with a
Python-side data model for validation and execution.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from nicegui import app, ui

from pathlib import Path

from ..components.dag_render import DagRenderNode, dag_to_mermaid
from ..shell import page_layout
from ...algorithms.dag.combinators import COMBINATORS
from ...algorithms.dag.composite import CompositeAlgorithm
from ...algorithms.dag.serialization import dag_from_yaml, dag_to_yaml, save_dag, load_dag
from ...algorithms.dag.types import DAGDefinition, DAGEdge, DAGNode
from ...algorithms.dag.validation import validate
from ...algorithms.registry import AlgorithmRegistry

_DAGS_DIR = Path(__file__).resolve().parents[4] / "data" / "dags"


def _list_saved_dags() -> list[str]:
    """Return names of saved DAG files (without extension)."""
    if not _DAGS_DIR.exists():
        return []
    return sorted(p.stem for p in _DAGS_DIR.glob("*.yaml"))

# ---------------------------------------------------------------------------
# LiteGraph.js bridge
# ---------------------------------------------------------------------------

_LITEGRAPH_CDN = "https://cdn.jsdelivr.net/npm/litegraph.js@0.7.18/build/litegraph.min.js"
_LITEGRAPH_CSS = "https://cdn.jsdelivr.net/npm/litegraph.js@0.7.18/css/litegraph.css"

_BRIDGE_JS = """
// --- Daytrader DAG Composer bridge ---
window._dt_graph = null;
window._dt_canvas = null;
window._dt_node_map = {};  // node_id -> litegraph node id

function dtInitCanvas(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    // Size the canvas element to match its container
    const cvs = container.querySelector('canvas');
    const rect = container.getBoundingClientRect();
    cvs.width = rect.width || 1000;
    cvs.height = rect.height || 600;

    const graph = new LGraph();
    const canvas = new LGraphCanvas(cvs, graph);
    canvas.background_color = '#141522';
    canvas.default_link_color = '#5c7cfa';
    canvas.highquality_render = true;
    canvas.render_connections_border = false;
    canvas.connections_width = 3;

    // Resize canvas when window resizes
    window.addEventListener('resize', function() {
        const r = container.getBoundingClientRect();
        cvs.width = r.width || 1000;
        cvs.height = r.height || 600;
        graph.setDirtyCanvas(true);
    });

    // Register custom node types
    function DaytraderAlgoNode() {
        this.addOutput("signal", "signal");
        this.title = "Algorithm";
        this.color = "#2a3a5c";
        this.bgcolor = "#1a2744";
        this.properties = { node_id: "", algorithm_id: "", params: {} };
        this.size = [200, 60];
    }
    DaytraderAlgoNode.title = "Algorithm";
    LiteGraph.registerNodeType("daytrader/algorithm", DaytraderAlgoNode);

    function DaytraderCombinatorNode() {
        this.addInput("signal_0", "signal");
        this.addInput("signal_1", "signal");
        this.addInput("signal_2", "signal");
        this.addInput("signal_3", "signal");
        this.addOutput("signal", "signal");
        this.title = "Combinator";
        this.color = "#3a2a5c";
        this.bgcolor = "#241a44";
        this.properties = { node_id: "", combinator_type: "", params: {} };
        this.size = [220, 140];
    }
    DaytraderCombinatorNode.title = "Combinator";
    LiteGraph.registerNodeType("daytrader/combinator", DaytraderCombinatorNode);

    // Event callbacks
    graph.onNodeSelected = function(node) {
        if (node && node.properties.node_id) {
            emitEvent('node_selected', { node_id: node.properties.node_id });
        }
    };

    graph.onNodeRemoved = function(node) {
        if (node && node.properties.node_id) {
            emitEvent('node_removed', { node_id: node.properties.node_id });
        }
    };

    graph.onAfterChange = function() {
        dtSyncToBackend();
    };

    window._dt_graph = graph;
    window._dt_canvas = canvas;
    graph.start();
}

function dtAddAlgoNode(nodeId, title, x, y) {
    const node = LiteGraph.createNode("daytrader/algorithm");
    node.title = title;
    node.pos = [x, y];
    node.properties.node_id = nodeId;
    node.properties.algorithm_id = title;
    window._dt_graph.add(node);
    window._dt_node_map[nodeId] = node.id;
    return node.id;
}

function dtAddCombinatorNode(nodeId, title, x, y, numInputs) {
    const node = LiteGraph.createNode("daytrader/combinator");
    node.title = title;
    node.pos = [x, y];
    node.properties.node_id = nodeId;
    node.properties.combinator_type = title;
    // Node type already has 4 inputs; add more only if needed
    var totalInputs = Math.max(numInputs, 4);
    for (let i = 4; i < totalInputs; i++) {
        node.addInput("signal_" + i, "signal");
    }
    node.size = [220, 60 + totalInputs * 22];
    window._dt_graph.add(node);
    window._dt_node_map[nodeId] = node.id;
    return node.id;
}

function dtAddConnection(sourceId, targetId, targetSlot) {
    const srcLgId = window._dt_node_map[sourceId];
    const tgtLgId = window._dt_node_map[targetId];
    if (srcLgId === undefined || tgtLgId === undefined) return;
    const srcNode = window._dt_graph.getNodeById(srcLgId);
    const tgtNode = window._dt_graph.getNodeById(tgtLgId);
    if (srcNode && tgtNode) {
        srcNode.connect(0, tgtNode, targetSlot || 0);
    }
}

function dtClearGraph() {
    if (window._dt_graph) {
        window._dt_graph.clear();
        window._dt_node_map = {};
    }
}

function dtSyncToBackend() {
    // Serialize current graph state and send to Python
    if (!window._dt_graph) return;
    const nodes = [];
    const allNodes = window._dt_graph._nodes;
    for (const n of allNodes) {
        nodes.push({
            node_id: n.properties.node_id,
            title: n.title,
            x: n.pos[0],
            y: n.pos[1],
            type: n.type,
        });
    }
    // Read connections from links
    const edges = [];
    const links = window._dt_graph.links;
    if (links) {
        for (const linkId in links) {
            const link = links[linkId];
            if (!link) continue;
            const srcNode = window._dt_graph.getNodeById(link.origin_id);
            const tgtNode = window._dt_graph.getNodeById(link.target_id);
            if (srcNode && tgtNode) {
                edges.push({
                    source_id: srcNode.properties.node_id,
                    target_id: tgtNode.properties.node_id,
                    target_slot: link.target_slot || 0,
                });
            }
        }
    }
}

function emitEvent(name, data) {
    // Bridge to NiceGUI via custom event on the canvas container
    const el = document.getElementById('dag-canvas-container');
    if (el) {
        el.dispatchEvent(new CustomEvent('dag-event', {
            detail: JSON.stringify({ event: name, ...data })
        }));
    }
}
"""


@ui.page("/dag-composer")
async def dag_composer_page() -> None:
    if not page_layout("DAG Composer"):
        return

    # ----- State ----------------------------------------------------------
    _state: dict[str, Any] = {
        "nodes": [],        # list of DAGNode-like dicts
        "edges": [],        # list of DAGEdge-like dicts
        "selected": None,   # selected node_id
        "counter": 0,       # unique node ID counter
        "dag_yaml": "",     # last saved YAML
        "dag_name": "",     # current DAG name (for save/load)
    }

    def _next_id(prefix: str) -> str:
        _state["counter"] += 1
        return f"{prefix}_{_state['counter']}"

    # ----- Inject LiteGraph -----------------------------------------------
    ui.add_head_html(f'<script src="{_LITEGRAPH_CDN}"></script>')
    ui.add_head_html(f'<link rel="stylesheet" href="{_LITEGRAPH_CSS}">')
    ui.add_head_html(f"<script>{_BRIDGE_JS}</script>")

    # ----- Build DAG from state -------------------------------------------
    def _build_dag() -> DAGDefinition:
        nodes = []
        for n in _state["nodes"]:
            nodes.append(DAGNode(
                node_id=n["node_id"],
                node_type=n["node_type"],
                algorithm_id=n.get("algorithm_id"),
                combinator_type=n.get("combinator_type"),
                params=n.get("params", {}),
                position=(n.get("x", 0), n.get("y", 0)),
                weight=n.get("weight", 1.0),
            ))
        edges = []
        for e in _state["edges"]:
            edges.append(DAGEdge(
                source_id=e["source_id"],
                target_id=e["target_id"],
            ))
        # Root = last combinator added (or only combinator)
        combinators = [n for n in nodes if n.node_type == "combinator"]
        root_id = combinators[-1].node_id if combinators else None

        dag_name = _state["dag_name"] or "Untitled DAG"
        dag_id = dag_name.lower().replace(" ", "_").replace("-", "_")
        return DAGDefinition(
            id=dag_id,
            name=dag_name,
            nodes=nodes,
            edges=edges,
            root_node_id=root_id,
        )

    # ----- Handlers -------------------------------------------------------
    def _add_algorithm(algo_id: str | None) -> None:
        if not algo_id:
            return
        nid = _next_id(algo_id.replace("-", "_"))
        algo_count = len([n for n in _state["nodes"] if n["node_type"] == "algorithm"])
        x = 50
        y = 30 + algo_count * 100
        _state["nodes"].append({
            "node_id": nid,
            "node_type": "algorithm",
            "algorithm_id": algo_id,
            "params": {},
            "x": x, "y": y, "weight": 1.0,
        })
        try:
            algo = AlgorithmRegistry.get(algo_id)
            title = algo.manifest.name
        except KeyError:
            title = algo_id
        ui.run_javascript(f'dtAddAlgoNode("{nid}", "{title}", {x}, {y})')
        _update_node_list()

    def _add_combinator(ctype: str | None) -> None:
        if not ctype:
            return
        nid = _next_id(ctype)
        x = 400
        y = 80
        n_inputs = len([n for n in _state["nodes"] if n["node_type"] == "algorithm"])
        _state["nodes"].append({
            "node_id": nid,
            "node_type": "combinator",
            "combinator_type": ctype,
            "params": {},
            "x": x, "y": y, "weight": 1.0,
        })
        ui.run_javascript(
            f'dtAddCombinatorNode("{nid}", "{ctype}", {x}, {y}, {max(n_inputs, 4)})'
        )
        _update_node_list()

    # Status tracking for the "sync indicator" shown in the toolbar.
    # ``paused`` is set during load/clear operations so the periodic
    # auto-sync doesn't race against in-flight JS commands.
    _sync_status: dict[str, Any] = {
        "nodes": 0, "edges": 0, "error": None, "paused": False,
    }

    _SYNC_JS = """
(function() {
    if (!window._dt_graph) return "null";
    var nodes = [];
    for (var n of window._dt_graph._nodes) {
        nodes.push({
            node_id: n.properties.node_id || "",
            title: n.title,
            type: n.type,
            pos: [n.pos[0], n.pos[1]]
        });
    }
    var edges = [];
    var links = window._dt_graph.links;
    if (links) {
        for (var id in links) {
            var l = links[id];
            if (!l) continue;
            var src = window._dt_graph.getNodeById(l.origin_id);
            var tgt = window._dt_graph.getNodeById(l.target_id);
            if (src && tgt && src.properties.node_id && tgt.properties.node_id) {
                edges.push({
                    source_id: src.properties.node_id,
                    target_id: tgt.properties.node_id,
                    target_slot: l.target_slot || 0
                });
            }
        }
    }
    return JSON.stringify({nodes: nodes, edges: edges});
})()
"""

    async def _sync_from_canvas(*, silent: bool = True) -> bool:
        """Pull the current graph state from LiteGraph into Python ``_state``.

        LiteGraph is the source of truth for user drag-and-drop actions,
        but Python-side handlers build the DAG from ``_state``. Without
        this sync, manually wired connections would be invisible to
        validate/save/send-to-strategy-lab.

        Returns True on success, False on failure. Errors are captured
        in ``_sync_status["error"]`` and surfaced via the status label.

        Safe against transient canvas states: if the sync observes
        fewer nodes than Python has (e.g. while JS add/connect commands
        are still in-flight after a Load), it leaves ``_state`` alone
        rather than wiping it.
        """
        if _sync_status.get("paused"):
            return False
        try:
            result = await ui.run_javascript(_SYNC_JS, timeout=2.0)
        except Exception as exc:
            _sync_status["error"] = f"JS error: {exc}"
            if not silent:
                ui.notify(f"Sync failed: {exc}", type="negative")
            _refresh_sync_label()
            return False

        if not result or result == "null":
            _sync_status["error"] = "canvas not initialized"
            _refresh_sync_label()
            return False

        try:
            data = json.loads(result)
        except Exception as exc:
            _sync_status["error"] = f"parse error: {exc}"
            _refresh_sync_label()
            return False

        canvas_nodes = data.get("nodes", [])
        canvas_edges = data.get("edges", [])

        # Defensive: if the canvas reports fewer nodes than Python
        # knows about, we're almost certainly observing an in-flight
        # state (e.g. Load in progress) — don't wipe anything.
        if len(canvas_nodes) < len(_state["nodes"]):
            _sync_status["nodes"] = len(_state["nodes"])
            _sync_status["edges"] = len(_state["edges"])
            _sync_status["error"] = None
            _refresh_sync_label()
            return True

        # Update positions of existing nodes from canvas (user may have
        # dragged them). Don't create nodes here — they're created via
        # the dropdown handlers which own the Python-side metadata.
        pos_map = {
            n["node_id"]: n["pos"] for n in canvas_nodes if n.get("node_id")
        }
        for state_node in _state["nodes"]:
            if state_node["node_id"] in pos_map:
                x, y = pos_map[state_node["node_id"]]
                state_node["x"] = x
                state_node["y"] = y

        # Replace edges entirely with whatever the canvas shows.
        _state["edges"] = [
            {
                "source_id": e["source_id"],
                "target_id": e["target_id"],
            }
            for e in canvas_edges
            if e.get("source_id") and e.get("target_id")
        ]

        _sync_status["nodes"] = len(_state["nodes"])
        _sync_status["edges"] = len(_state["edges"])
        _sync_status["error"] = None
        _refresh_sync_label()
        _refresh_diagram()
        return True

    def _refresh_sync_label() -> None:
        """Update the live status label (called after every sync)."""
        if "label" not in _sync_status:
            return
        label = _sync_status["label"]
        if _sync_status.get("error"):
            label.text = f"⚠ {_sync_status['error']}"
            label.classes(replace="text-caption text-orange")
        else:
            label.text = (
                f"nodes: {_sync_status['nodes']} · "
                f"edges: {_sync_status['edges']}"
            )
            label.classes(replace="text-caption text-grey-5")

    async def _manual_sync() -> None:
        """User-triggered sync from the Refresh button."""
        ok = await _sync_from_canvas(silent=False)
        if ok:
            ui.notify(
                f"Synced: {_sync_status['nodes']} nodes, {_sync_status['edges']} edges",
                type="positive",
            )

    def _auto_connect() -> None:
        """Auto-connect all algorithm nodes to the first combinator."""
        combinator = None
        for n in _state["nodes"]:
            if n["node_type"] == "combinator":
                combinator = n
                break
        if not combinator:
            ui.notify("Add a combinator first", type="warning")
            return

        _state["edges"].clear()
        slot = 0
        for n in _state["nodes"]:
            if n["node_type"] == "algorithm":
                _state["edges"].append({
                    "source_id": n["node_id"],
                    "target_id": combinator["node_id"],
                })
                ui.run_javascript(
                    f'dtAddConnection("{n["node_id"]}", "{combinator["node_id"]}", {slot})'
                )
                slot += 1
        ui.notify(f"Connected {slot} algorithms to {combinator['combinator_type']}", type="positive")

    async def _validate_dag() -> None:
        await _sync_from_canvas()
        dag = _build_dag()
        errors = validate(dag)
        if errors:
            for e in errors:
                ui.notify(e, type="negative")
        else:
            ui.notify("DAG is valid!", type="positive")

    async def _send_to_strategy_lab() -> None:
        await _sync_from_canvas()
        dag = _build_dag()
        errors = validate(dag)
        if errors:
            for e in errors:
                ui.notify(e, type="negative")
            return
        try:
            composite = CompositeAlgorithm(dag)
            AlgorithmRegistry.register(composite)
            ui.notify(
                f"Registered {composite.manifest.id} — redirecting to Strategy Lab",
                type="positive",
            )
            ui.navigate.to("/strategy-lab")
        except Exception as exc:
            ui.notify(str(exc), type="negative")

    def _clear_dag() -> None:
        _sync_status["paused"] = True
        try:
            _state["nodes"].clear()
            _state["edges"].clear()
            _state["selected"] = None
            ui.run_javascript("dtClearGraph()")
            _update_node_list()
            ui.notify("Canvas cleared", type="info")
        finally:
            _sync_status["paused"] = False

    async def _export_yaml() -> None:
        await _sync_from_canvas()
        dag = _build_dag()
        errors = validate(dag)
        if errors:
            for e in errors:
                ui.notify(e, type="negative")
            return
        yaml_str = dag_to_yaml(dag)
        _state["dag_yaml"] = yaml_str
        yaml_display.set_value(yaml_str)
        yaml_dialog.open()

    def _import_yaml() -> None:
        yaml_input.set_value("")
        import_dialog.open()

    def _do_import() -> None:
        yaml_str = yaml_input.value
        if not yaml_str.strip():
            ui.notify("YAML is empty", type="warning")
            return
        try:
            dag = dag_from_yaml(yaml_str)
        except Exception as exc:
            ui.notify(f"Parse error: {exc}", type="negative")
            return

        # Rebuild state from imported DAG
        _state["nodes"].clear()
        _state["edges"].clear()
        ui.run_javascript("dtClearGraph()")

        for node in dag.nodes:
            nd = {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "algorithm_id": node.algorithm_id,
                "combinator_type": node.combinator_type,
                "params": dict(node.params),
                "x": node.position[0], "y": node.position[1],
                "weight": node.weight,
            }
            _state["nodes"].append(nd)
            if node.node_type == "algorithm":
                try:
                    algo = AlgorithmRegistry.get(node.algorithm_id or "")
                    title = algo.manifest.name
                except KeyError:
                    title = node.algorithm_id or "Unknown"
                ui.run_javascript(
                    f'dtAddAlgoNode("{node.node_id}", "{title}", '
                    f'{node.position[0]}, {node.position[1]})'
                )
            else:
                n_inputs = len(dag.children_of(node.node_id))
                ui.run_javascript(
                    f'dtAddCombinatorNode("{node.node_id}", "{node.combinator_type}", '
                    f'{node.position[0]}, {node.position[1]}, {max(n_inputs, 2)})'
                )

        for i, edge in enumerate(dag.edges):
            _state["edges"].append({
                "source_id": edge.source_id,
                "target_id": edge.target_id,
            })
            ui.run_javascript(
                f'dtAddConnection("{edge.source_id}", "{edge.target_id}", {i})'
            )

        _update_node_list()
        import_dialog.close()
        ui.notify(f"Imported DAG: {dag.name}", type="positive")

    async def _save_dag() -> None:
        name = save_name_input.value.strip()
        if not name:
            ui.notify("Enter a name for this DAG", type="warning")
            return
        _state["dag_name"] = name
        await _sync_from_canvas()
        dag = _build_dag()
        errors = validate(dag)
        if errors:
            for e in errors:
                ui.notify(e, type="negative")
            return
        try:
            filename = name.lower().replace(" ", "_").replace("-", "_")
            save_dag(dag, _DAGS_DIR / f"{filename}.yaml")
            # Also register as a composite algorithm so it shows up in
            # Strategy Lab / Research Lab / plugin list immediately.
            composite = CompositeAlgorithm(dag)
            AlgorithmRegistry.register(composite)
            ui.notify(
                f"Saved '{name}' — available in Strategy Lab as '{composite.manifest.id}'",
                type="positive",
            )
            save_dialog.close()
            load_select.set_options(_list_saved_dags())
        except Exception as exc:
            ui.notify(f"Save failed: {exc}", type="negative")

    def _open_save_dialog() -> None:
        save_name_input.set_value(_state["dag_name"])
        save_dialog.open()

    def _load_saved_dag(name: str | None) -> None:
        if not name:
            return
        path = _DAGS_DIR / f"{name}.yaml"
        if not path.exists():
            ui.notify(f"DAG file not found: {name}", type="negative")
            return
        try:
            dag = load_dag(path)
        except Exception as exc:
            ui.notify(f"Load failed: {exc}", type="negative")
            return

        # Pause the auto-sync so in-flight JS commands can't race with
        # the periodic canvas read and wipe the edges we're about to add.
        _sync_status["paused"] = True

        # Clear and rebuild
        _state["nodes"].clear()
        _state["edges"].clear()
        _state["dag_name"] = dag.name
        ui.run_javascript("dtClearGraph()")

        for node in dag.nodes:
            nd = {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "algorithm_id": node.algorithm_id,
                "combinator_type": node.combinator_type,
                "params": dict(node.params),
                "x": node.position[0], "y": node.position[1],
                "weight": node.weight,
            }
            _state["nodes"].append(nd)
            if node.node_type == "algorithm":
                try:
                    algo = AlgorithmRegistry.get(node.algorithm_id or "")
                    title = algo.manifest.name
                except KeyError:
                    title = node.algorithm_id or "Unknown"
                ui.run_javascript(
                    f'dtAddAlgoNode("{node.node_id}", "{title}", '
                    f'{node.position[0]}, {node.position[1]})'
                )
            else:
                n_inputs = len(dag.children_of(node.node_id))
                ui.run_javascript(
                    f'dtAddCombinatorNode("{node.node_id}", "{node.combinator_type}", '
                    f'{node.position[0]}, {node.position[1]}, {max(n_inputs, 4)})'
                )

        for i, edge in enumerate(dag.edges):
            _state["edges"].append({
                "source_id": edge.source_id,
                "target_id": edge.target_id,
            })
            ui.run_javascript(
                f'dtAddConnection("{edge.source_id}", "{edge.target_id}", {i})'
            )

        # Update counter to avoid ID collisions
        _state["counter"] = len(dag.nodes) + 1
        _update_node_list()
        load_select.set_value(None)

        # Give the browser a moment to execute the queued JS commands
        # before re-enabling auto-sync (prevents race with canvas draw).
        async def _resume_sync_later() -> None:
            await asyncio.sleep(1.5)
            _sync_status["paused"] = False

        asyncio.create_task(_resume_sync_later())

        ui.notify(f"Loaded '{dag.name}'", type="positive")

    def _remove_selected() -> None:
        sel = _state["selected"]
        if not sel:
            ui.notify("No node selected", type="warning")
            return
        _state["nodes"] = [n for n in _state["nodes"] if n["node_id"] != sel]
        _state["edges"] = [
            e for e in _state["edges"]
            if e["source_id"] != sel and e["target_id"] != sel
        ]
        _state["selected"] = None
        # Rebuild canvas
        ui.run_javascript("dtClearGraph()")
        _rebuild_canvas()
        _update_node_list()

    def _rebuild_canvas() -> None:
        """Rebuild the JS canvas from Python state."""
        for n in _state["nodes"]:
            if n["node_type"] == "algorithm":
                try:
                    algo = AlgorithmRegistry.get(n.get("algorithm_id", ""))
                    title = algo.manifest.name
                except KeyError:
                    title = n.get("algorithm_id", "?")
                ui.run_javascript(
                    f'dtAddAlgoNode("{n["node_id"]}", "{title}", {n["x"]}, {n["y"]})'
                )
            else:
                n_inputs = len([
                    e for e in _state["edges"] if e["target_id"] == n["node_id"]
                ])
                ui.run_javascript(
                    f'dtAddCombinatorNode("{n["node_id"]}", "{n.get("combinator_type", "?")}", '
                    f'{n["x"]}, {n["y"]}, {max(n_inputs, 2)})'
                )
        for i, e in enumerate(_state["edges"]):
            ui.run_javascript(
                f'dtAddConnection("{e["source_id"]}", "{e["target_id"]}", {i})'
            )

    def _update_node_list() -> None:
        node_list_container.clear()
        with node_list_container:
            for n in _state["nodes"]:
                ntype = n["node_type"]
                label = n.get("algorithm_id") or n.get("combinator_type") or "?"
                icon = "functions" if ntype == "algorithm" else "merge_type"
                color = "blue" if ntype == "algorithm" else "purple"
                with ui.row().classes("items-center gap-1 w-full"):
                    ui.icon(icon, size="xs", color=color)
                    btn = ui.button(
                        f"{label} ({n['node_id']})",
                        on_click=lambda nid=n["node_id"]: _select_node(nid),
                    ).props("flat dense no-caps").classes("text-xs")
        _refresh_diagram()

    def _select_node(node_id: str) -> None:
        _state["selected"] = node_id
        _render_properties()

    def _render_properties() -> None:
        props_container.clear()
        nid = _state["selected"]
        if not nid:
            return
        node = None
        for n in _state["nodes"]:
            if n["node_id"] == nid:
                node = n
                break
        if not node:
            return

        with props_container:
            ui.label(node["node_id"]).classes("text-subtitle2 text-weight-bold")
            ui.label(f"Type: {node['node_type']}").classes("text-caption text-grey-5")

            if node["node_type"] == "algorithm" and node.get("algorithm_id"):
                try:
                    algo = AlgorithmRegistry.get(node["algorithm_id"])
                    manifest = algo.manifest
                    ui.separator()
                    ui.label("Parameters").classes("text-caption text-grey-5 q-mt-sm")
                    for param in manifest.params:
                        current_val = node.get("params", {}).get(
                            param.name, param.default
                        )
                        if param.type == "bool":
                            sw = ui.switch(param.name, value=bool(current_val))
                            sw.on_value_change(
                                lambda e, p=param.name, nd=node: nd.setdefault("params", {}).__setitem__(p, e.value)
                            )
                        elif param.type == "int":
                            inp = ui.number(
                                param.name, value=int(current_val),
                                min=param.min, max=param.max, step=1,
                            ).classes("w-full")
                            inp.on_value_change(
                                lambda e, p=param.name, nd=node: nd.setdefault("params", {}).__setitem__(p, int(e.value)) if e.value is not None else None
                            )
                        elif param.type == "float":
                            inp = ui.number(
                                param.name, value=float(current_val),
                                min=param.min, max=param.max,
                                step=param.step or 0.01,
                            ).classes("w-full")
                            inp.on_value_change(
                                lambda e, p=param.name, nd=node: nd.setdefault("params", {}).__setitem__(p, float(e.value)) if e.value is not None else None
                            )
                except KeyError:
                    ui.label("Algorithm not found in registry").classes("text-negative")

            elif node["node_type"] == "combinator":
                ctype = node.get("combinator_type", "")
                ui.separator()
                ui.label("Combinator Params").classes("text-caption text-grey-5 q-mt-sm")

                # Rolling variants take window_bars + min_fired / min_agreement
                if ctype in ("rolling_unanimous", "rolling_majority_vote"):
                    window_val = int(node.get("params", {}).get("window_bars", 5))
                    w_inp = ui.number(
                        "Window bars", value=window_val,
                        min=1, max=200, step=1,
                    ).classes("w-full")
                    w_inp.on_value_change(
                        lambda e, nd=node: nd.setdefault("params", {}).__setitem__("window_bars", int(e.value)) if e.value is not None else None
                    )

                    if ctype == "rolling_unanimous":
                        min_fired_val = int(node.get("params", {}).get("min_fired", 2))
                        mf_inp = ui.number(
                            "Min fired", value=min_fired_val,
                            min=1, max=10, step=1,
                        ).classes("w-full")
                        mf_inp.on_value_change(
                            lambda e, nd=node: nd.setdefault("params", {}).__setitem__("min_fired", int(e.value)) if e.value is not None else None
                        )
                    else:  # rolling_majority_vote
                        min_agree_val = float(node.get("params", {}).get("min_agreement", 0.5))
                        ma_inp = ui.number(
                            "Min agreement", value=min_agree_val,
                            min=0.0, max=1.0, step=0.1,
                        ).classes("w-full")
                        ma_inp.on_value_change(
                            lambda e, nd=node: nd.setdefault("params", {}).__setitem__("min_agreement", float(e.value)) if e.value is not None else None
                        )

                elif ctype == "majority_vote":
                    min_agree_val = float(node.get("params", {}).get("min_agreement", 0.5))
                    ui.number(
                        "Min agreement", value=min_agree_val,
                        min=0.0, max=1.0, step=0.1,
                    ).classes("w-full").on_value_change(
                        lambda e, nd=node: nd.setdefault("params", {}).__setitem__("min_agreement", float(e.value)) if e.value is not None else None
                    )
                else:
                    ui.label("No params for this combinator").classes(
                        "text-caption text-grey-6"
                    )

            ui.separator()
            with ui.row().classes("q-mt-sm"):
                ui.number(
                    "Weight", value=node.get("weight", 1.0),
                    min=0.0, max=10.0, step=0.1,
                ).classes("w-full").on_value_change(
                    lambda e, nd=node: nd.__setitem__("weight", float(e.value)) if e.value is not None else None
                )

    # ----- Page layout ----------------------------------------------------

    # Toolbar
    with ui.row().classes("w-full items-center gap-2 q-pa-sm").style(
        "background-color: #1a1b2e; border-bottom: 1px solid #333"
    ):
        def _on_algo_select(e) -> None:
            if e.value:
                _add_algorithm(e.value)
                algo_select.set_value(None)

        # DAG composition with sandboxed plugins is deferred to a future
        # phase (deepcopy semantics for subprocess-backed adapters need
        # design work). Show only algorithms that are safe to compose:
        # built-ins and any other plain in-process algorithms.
        from ...algorithms.sandbox import SandboxedAlgorithm

        _composable_ids = [
            aid for aid in AlgorithmRegistry.available()
            if not isinstance(AlgorithmRegistry.get(aid), SandboxedAlgorithm)
        ]
        algo_select = ui.select(
            _composable_ids,
            label="Add Algorithm",
            on_change=_on_algo_select,
        ).classes("w-48")

        def _on_combinator_select(e) -> None:
            if e.value:
                _add_combinator(e.value)
                combinator_select.set_value(None)

        combinator_select = ui.select(
            list(COMBINATORS.keys()),
            label="Add Combinator",
            on_change=_on_combinator_select,
        ).classes("w-48")

        ui.button("Auto-Connect", icon="link", on_click=_auto_connect).props("flat dense")
        ui.button("Validate", icon="check_circle", on_click=_validate_dag).props("flat dense")
        ui.button("Remove Selected", icon="delete", on_click=_remove_selected).props("flat dense color=negative")
        ui.separator().props("vertical")
        ui.button("Save", icon="save", on_click=_open_save_dialog).props("flat dense")
        ui.button(icon="refresh", on_click=_manual_sync).props(
            "flat dense"
        ).tooltip("Force-sync canvas state from LiteGraph to Python")

        # Live status indicator
        _sync_status["label"] = ui.label("nodes: 0 · edges: 0").classes(
            "text-caption text-grey-5 q-ml-sm"
        )

        def _on_load_select(e) -> None:
            if e.value:
                _load_saved_dag(e.value)

        load_select = ui.select(
            _list_saved_dags(),
            label="Load DAG",
            on_change=_on_load_select,
        ).classes("w-40")

        ui.space()
        ui.button("Export YAML", icon="download", on_click=_export_yaml).props("flat dense")
        ui.button("Import YAML", icon="upload", on_click=_import_yaml).props("flat dense")
        ui.button("Clear", icon="clear_all", on_click=_clear_dag).props("flat dense")
        ui.button(
            "Send to Strategy Lab", icon="send",
            on_click=_send_to_strategy_lab,
        ).props("dense color=primary")

    # Main content: canvas + side panels
    with ui.row().classes("w-full gap-0").style(
        "height: calc(100vh - 120px); min-height: 500px"
    ):
        # Left: node list
        with ui.column().classes("q-pa-sm gap-1").style(
            "width: 200px; background-color: #141522; overflow-y: auto; flex-shrink: 0"
        ):
            ui.label("NODES").classes("text-overline text-grey-7")
            node_list_container = ui.column().classes("gap-1 w-full")

        # Center: LiteGraph canvas
        with ui.column().classes("flex-grow").style("overflow: hidden"):
            canvas_container = ui.html(
                '<div id="dag-canvas-container" style="width: 100%; height: 100%; position: relative">'
                '<canvas style="width: 100%; height: 100%"></canvas>'
                '</div>'
            ).classes("w-full").style("height: 100%")

        # Right: properties panel
        with ui.column().classes("q-pa-sm gap-2").style(
            "width: 260px; background-color: #141522; overflow-y: auto; flex-shrink: 0"
        ):
            ui.label("PROPERTIES").classes("text-overline text-grey-7")
            props_container = ui.column().classes("gap-2 w-full")

    # Diagram preview — same Mermaid renderer the Charts Workbench uses,
    # so the user can see the structure of the DAG they're wiring without
    # leaving the composer. Refreshed on every state mutation and on each
    # auto-sync tick (covers drag-and-drop connections).
    with ui.expansion("Diagram preview", icon="account_tree", value=True).classes(
        "w-full"
    ).style("background-color: #1a1b2e; border-top: 1px solid #333"):
        with ui.card().classes("w-full"):
            mermaid_widget = ui.mermaid("flowchart LR").classes("w-full")

    def _refresh_diagram() -> None:
        """Rebuild the Mermaid diagram from the current builder state."""
        render_nodes: list[DagRenderNode] = []
        for n in _state["nodes"]:
            if n["node_type"] == "algorithm":
                aid = n.get("algorithm_id") or "?"
                try:
                    label = AlgorithmRegistry.get(aid).manifest.name
                except KeyError:
                    label = aid
            else:
                label = n.get("combinator_type") or "?"
            parents = [
                e["source_id"] for e in _state["edges"]
                if e["target_id"] == n["node_id"]
            ]
            render_nodes.append(DagRenderNode(
                node_id=n["node_id"],
                node_type=n["node_type"],
                label=label,
                parents=parents,
            ))
        mermaid_widget.set_content(dag_to_mermaid(render_nodes))

    # Initialize canvas after DOM is ready
    ui.timer(0.5, lambda: ui.run_javascript('dtInitCanvas("dag-canvas-container")'), once=True)

    # Periodic auto-sync: pull graph state from LiteGraph every 2s so
    # drag-and-drop connections stay in Python state without the user
    # needing to remember to refresh manually. Skipped while `paused`.
    ui.timer(2.0, _sync_from_canvas)

    # ----- Dialogs --------------------------------------------------------

    with ui.dialog() as yaml_dialog, ui.card().classes("w-[600px]"):
        ui.label("DAG YAML").classes("text-h6")
        yaml_display = ui.textarea().classes("w-full font-mono").props("readonly rows=20")
        ui.button("Close", on_click=yaml_dialog.close).props("flat")

    with ui.dialog() as import_dialog, ui.card().classes("w-[600px]"):
        ui.label("Import DAG from YAML").classes("text-h6")
        yaml_input = ui.textarea(placeholder="Paste YAML here...").classes("w-full font-mono").props("rows=20")
        with ui.row():
            ui.button("Import", on_click=_do_import).props("color=primary")
            ui.button("Cancel", on_click=import_dialog.close).props("flat")

    with ui.dialog() as save_dialog, ui.card().classes("w-[400px]"):
        ui.label("Save DAG").classes("text-h6")
        save_name_input = ui.input(
            "DAG Name",
            placeholder="e.g. MACD + ADX + RSI",
        ).classes("w-full")
        with ui.row():
            ui.button("Save", on_click=_save_dag).props("color=primary")
            ui.button("Cancel", on_click=save_dialog.close).props("flat")
