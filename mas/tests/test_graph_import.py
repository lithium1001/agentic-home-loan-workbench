"""Smoke test for the extracted graph module (graph.py).

Guards the cell-13 -> graph.py extraction: the compiled graph must import cleanly,
register every agent as a node, and produce a well-formed initial state via
new_state(). Does NOT call the LLM (no network), so it is safe to run offline.
"""


def test_graph_compiles_with_all_agents():
    from graph import AGENTS, graph

    nodes = set(graph.get_graph().nodes.keys())
    # Each agent contributes a setup node and an _llm node.
    for name in AGENTS:
        assert name in nodes, f"missing setup node for {name}"
        assert f"{name}_llm" in nodes, f"missing llm node for {name}"
    # Core control-flow nodes are present.
    for core in ("orchestrator", "tools", "hitl_review", "replan"):
        assert core in nodes, f"missing core node {core}"


def test_new_state_shape():
    from graph import CopilotState, new_state

    st = new_state("APP0001", "Check LTV eligibility")
    assert set(st.keys()) == set(CopilotState.__annotations__.keys())
    assert st["applicant_id"] == "APP0001"
    assert len(st["messages"]) == 1
    assert st["messages"][0].content == "Check LTV eligibility"
