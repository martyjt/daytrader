"""Shared DAG visualization helpers.

The Charts Workbench (read-side, with runtime scores) and the DAG
Composer (build-side, structure only) both render the same Mermaid
flowchart. This module owns the rendering so the two pages stay in
visual sync.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class DagRenderNode:
    node_id: str
    node_type: str  # "algorithm" | "combinator"
    label: str
    parents: list[str] = field(default_factory=list)
    latest_score: float | None = None


def dag_to_mermaid(nodes: Sequence[DagRenderNode]) -> str:
    """Render a DAG as a Mermaid ``flowchart LR`` with per-node color coding.

    Nodes with ``latest_score`` set color by direction (long/short/flat);
    nodes without scores all render as flat — useful for the build-side
    view where the DAG hasn't been run yet.
    """
    lines: list[str] = ["flowchart LR"]
    for n in nodes:
        score = n.latest_score
        if score is None:
            cls = "flat"
            score_txt = "—"
        elif score > 0.05:
            cls = "long"
            score_txt = f"+{score:.2f}"
        elif score < -0.05:
            cls = "short"
            score_txt = f"{score:.2f}"
        else:
            cls = "flat"
            score_txt = f"{score:+.2f}"
        label = n.label.replace('"', "'").replace("\n", "<br/>")
        shape_open, shape_close = (
            ("((", "))") if n.node_type == "combinator" else ("[", "]")
        )
        lines.append(
            f'  {n.node_id}{shape_open}"{label}<br/><b>{score_txt}</b>"{shape_close}:::{cls}'
        )

    for n in nodes:
        for p in n.parents:
            lines.append(f"  {p} --> {n.node_id}")

    lines.append("  classDef long fill:#1d4024,stroke:#40c057,color:#e8f5e9;")
    lines.append("  classDef short fill:#4a1818,stroke:#fa5252,color:#fdecea;")
    lines.append("  classDef flat fill:#2a2b3a,stroke:#868e96,color:#e4e4e4;")
    return "\n".join(lines)
