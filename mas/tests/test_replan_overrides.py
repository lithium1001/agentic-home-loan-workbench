"""Replan must not silently drop the RM's requested change.

Observed 2026-08-04: the RM rejected a draft with "change rate to 1.2%" and the
console showed `[Replan] kind=redraft route_to=document_drafting overrides={}`.
The letter came back with the OLD rate, looking exactly like the model had ignored
the instruction. It had not: the reply failed `json.loads`, and the fail-safe
rewrote the decision to a plain redraft — the same defect class as the 2026-07-22
orchestrator routing-JSON bug, in the node that never received that fix.

Two silent-degradation paths are pinned here:
  * a decision that is CORRECT but wrapped in prose / a fence must still be read;
  * a decision that genuinely cannot be honoured must SAY so, not just look like
    the model declining to act.

`replan_node` calls an LLM, so these tests drive the parsing/validation logic
through a stubbed `chat` — no network.
"""
import graph


def _run(monkeypatch, reply, feedback="change rate to 1.2%"):
    """Drive replan_node with a canned model reply; return its state update."""
    monkeypatch.setattr(graph, "chat", lambda *a, **k: reply)
    state = graph.new_state("APP0001", "")
    state["stage"] = "IPA"
    state["hitl_feedback"] = feedback
    state["payload"] = {"figures": {"monthly_repayment": 4490}}
    return graph.replan_node(state)


CLEAN = ('{"kind":"recompute","route_to":"property_analysis",'
         '"overrides":{"interest_rate_pct":1.2},"message":""}')


def test_clean_json_recomputes(monkeypatch):
    out = _run(monkeypatch, CLEAN)
    assert out["replan_kind"] == "recompute"
    assert out["overrides"] == {"interest_rate_pct": 1.2}


def test_json_wrapped_in_prose_is_still_honoured(monkeypatch):
    """The reported failure: a correct decision that json.loads() could not read."""
    out = _run(monkeypatch, "Sure, here is my decision:\n" + CLEAN + "\nHope that helps!")
    assert out["replan_kind"] == "recompute"
    assert out["overrides"] == {"interest_rate_pct": 1.2}


def test_json_in_a_fenced_block_is_still_honoured(monkeypatch):
    out = _run(monkeypatch, "```json\n" + CLEAN + "\n```")
    assert out["replan_kind"] == "recompute"
    assert out["overrides"] == {"interest_rate_pct": 1.2}


def test_percent_string_is_coerced_to_a_number(monkeypatch):
    """The prompt asks for 3.2 not "3.2%", but a string would otherwise reach
    calculate_loan and fail far from the cause."""
    out = _run(monkeypatch, CLEAN.replace('"interest_rate_pct":1.2',
                                          '"interest_rate_pct":"1.2%"'))
    assert out["overrides"] == {"interest_rate_pct": 1.2}


def test_unparseable_reply_still_falls_back_to_redraft(monkeypatch):
    """Fail SAFE, not fail silent — the numbers must never be recomputed from a
    decision nobody could read."""
    out = _run(monkeypatch, "I think we should probably adjust the rate a bit.")
    assert out["replan_kind"] == "redraft"
    assert not out.get("overrides")


def test_unparseable_reply_is_logged(monkeypatch, capsys):
    """Swallowing the raw text is what made this look like a model failure and
    left nothing to diagnose from."""
    _run(monkeypatch, "no json here at all")
    assert "did not parse" in capsys.readouterr().out


def test_invented_override_key_is_reported(monkeypatch, capsys):
    """`tenure` is not adjustable. Dropping it is right; dropping it in silence is
    what makes the RM think the system ignored them."""
    out = _run(monkeypatch,
               '{"kind":"recompute","route_to":"property_analysis",'
               '"overrides":{"tenure":10},"message":""}',
               feedback="make the tenure 10 years")
    assert out["replan_kind"] == "redraft"          # nothing usable survived
    out_text = capsys.readouterr().out
    assert "non-adjustable" in out_text
    assert "will NOT be applied" in out_text


def test_recompute_reaches_the_producing_agent(monkeypatch):
    """The override has to travel in state/payload — the transcript is trimmed, so
    a change carried only in conversation would be lost (see _agent_segment)."""
    out = _run(monkeypatch, CLEAN)
    assert out["route"] == "property_analysis"
    assert out["payload"]["overrides"] == {"interest_rate_pct": 1.2}