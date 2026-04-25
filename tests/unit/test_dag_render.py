"""Tests for the shared DAG mermaid renderer."""

from daytrader.ui.components.dag_render import DagRenderNode, dag_to_mermaid


def test_empty_dag_still_emits_flowchart_header():
    out = dag_to_mermaid([])
    assert out.startswith("flowchart LR")
    # ClassDefs always present so styling holds even when nodes get added later
    assert "classDef long" in out
    assert "classDef short" in out
    assert "classDef flat" in out


def test_build_side_dag_without_scores_is_all_flat():
    """Composer-style: no run yet, so no scores."""
    nodes = [
        DagRenderNode("ema_1", "algorithm", label="EMA Crossover"),
        DagRenderNode("rsi_1", "algorithm", label="RSI Mean Reversion"),
        DagRenderNode(
            "vote_0", "combinator", label="majority_vote",
            parents=["ema_1", "rsi_1"],
        ),
    ]
    out = dag_to_mermaid(nodes)
    # All nodes referenced and edges present
    assert "ema_1" in out
    assert "rsi_1" in out
    assert "vote_0" in out
    assert "ema_1 --> vote_0" in out
    assert "rsi_1 --> vote_0" in out
    # No scores ⇒ everyone is :::flat with em-dash placeholder
    assert ":::flat" in out
    assert "—" in out


def test_runtime_dag_colors_by_score_direction():
    nodes = [
        DagRenderNode("a", "algorithm", label="A", latest_score=0.42),
        DagRenderNode("b", "algorithm", label="B", latest_score=-0.31),
        DagRenderNode("c", "algorithm", label="C", latest_score=0.0),
        DagRenderNode("d", "algorithm", label="D", latest_score=None),
    ]
    out = dag_to_mermaid(nodes)
    assert "a[" in out and ":::long" in out
    assert "b[" in out and ":::short" in out
    # 0.0 and None both render as flat
    assert out.count(":::flat") == 2


def test_combinator_uses_double_paren_shape():
    nodes = [
        DagRenderNode("c", "combinator", label="majority_vote"),
    ]
    out = dag_to_mermaid(nodes)
    # Mermaid `((label))` shape distinguishes combinators visually
    assert "c((" in out
    assert "))" in out


def test_label_quotes_and_newlines_are_escaped():
    nodes = [
        DagRenderNode("x", "algorithm", label='He said "hi"\nthen left'),
    ]
    out = dag_to_mermaid(nodes)
    # Double quotes flipped to singles, newline → <br/>
    assert '"hi"' not in out
    assert "'hi'" in out
    assert "<br/>" in out
