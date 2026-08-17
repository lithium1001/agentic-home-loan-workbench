"""Guards that a reject→replan REDRAFT re-opens the HITL gate.

Regression ("entered replan but no letter comes out"): after the RM rejects a
draft and asks for a wording change, replan routes replan_kind='redraft' →
document_drafting, and overwrites state['route'] with 'document_drafting'. That
makes _is_full_flow read False, so the drafting router used to fall through to
END — the redrafted letter then leaked out as a plain answer with no PDF and no
Approve/Revise gate. _route_drafting_llm must treat a redraft (and a recompute)
as gate-bound. No LLM / network.
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END

from graph import _route_drafting_llm, _is_redraft


def _st(**kw):
    base = {"messages": [AIMessage(content="letter body")], "replan_kind": "",
            "route": "document_drafting", "intent": "", "stage": "IPA"}
    base.update(kw)
    return base


def test_redraft_routes_to_gate():
    st = _st(replan_kind="redraft")
    assert _is_redraft(st) is True
    assert _route_drafting_llm(st) == "hitl_review"


def test_recompute_still_routes_to_gate():
    st = _st(replan_kind="recompute", route="property_analysis")
    assert _route_drafting_llm(st) == "hitl_review"


def test_full_flow_routes_to_gate():
    st = _st(replan_kind="", route="full_ipa", stage="IPA")
    assert _route_drafting_llm(st) == "hitl_review"


def test_standalone_drafting_ends_without_gate():
    # A bare re-render of an already-cleared payload (no replan, not a full flow)
    # is NOT force-gated — it stops at END as before.
    st = _st(replan_kind="", route="document_drafting")
    assert _route_drafting_llm(st) == END


def test_tool_calls_take_priority():
    msg = AIMessage(content="", tool_calls=[{"name": "draft_letter", "args": {}, "id": "1"}])
    st = _st(replan_kind="redraft", messages=[msg])
    assert _route_drafting_llm(st) == "tools"
