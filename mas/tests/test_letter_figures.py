"""Regression tests for the letter's Indicative Terms figures.

Guards two defects, both from 2026-07-10 (see ``eval/evidence/``):

1. The drafting agent passed ``monthly_repayment`` to ``draft_letter`` as a
   free-form float and typed 2,299 where the calculator said 4,490 — a
   plausible-looking number, on bank letterhead, that no reviewer would catch.
2. The first fix sourced the figures from ``payload["cleared_figures"]``, written
   only by the compliance node. A standalone "draft the letter" request never
   routes through compliance, so the tool found nothing and the agent wrote its
   refusal *as the letter body* — which shipped as a bank-letterhead PDF.

The fix: no figure is ever an argument the LLM can fill. ``draft_letter`` reads
graph state via ``InjectedState``, preferring figures an upstream ``calculate_loan``
already produced and otherwise re-deriving them from the case's CSV row. It refuses
only when neither exists.
"""
import io
import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from graph import _collect_tool_results, tool_node
from utils import letter_store
from utils.calculator import calculate_loan
from utils.letter_pdf import BANK_REF, build_letter_pdf
from utils.tools import _figures_for, draft_letter

_AID = "APP0001"
_HALLUCINATED = 2299     # what the broken letter printed
_TRUE_INSTALMENT = 4490  # what calculate_loan says


class _S(TypedDict):
    messages: Annotated[list, add_messages]
    applicant_id: str
    payload: dict
    overrides: dict


def _render(applicant_id=_AID, payload=None, overrides=None, stage="IPA") -> dict:
    """Drive draft_letter the way production does — through a compiled graph, so
    ToolNode performs the InjectedState injection (a bare tool_node.invoke() cannot)."""
    call = AIMessage(content="", tool_calls=[
        {"name": "draft_letter", "args": {"stage": stage}, "id": "call1"}])
    g = StateGraph(_S)
    g.add_node("tools", tool_node)
    g.add_edge(START, "tools")
    g.add_edge("tools", END)
    out = g.compile().invoke({
        "messages": [call],
        "applicant_id": applicant_id,
        "payload": payload or {},
        "overrides": overrides or {},
    })
    return json.loads(out["messages"][-1].content)


def test_llm_cannot_supply_any_figure():
    """The tool schema exposes `stage` and nothing else.

    This is the whole guarantee. A prompt telling the model not to re-derive numbers
    is not a constraint — the old docstring said exactly that, and the model
    re-derived anyway. Deleting the parameter is the constraint.
    """
    args = set(draft_letter.args)
    assert args == {"stage"}, f"draft_letter must expose only `stage`, got {sorted(args)}"
    for forbidden in ("monthly_repayment", "loan_amount", "property_price", "tenure_years",
                      "interest_rate_pct", "applicant_id", "state"):
        assert forbidden not in args, f"`{forbidden}` is reachable by the LLM again"


def test_standalone_drafting_still_gets_real_figures():
    """Defect 2: a drafting turn with an empty payload must still print the truth.

    This is the exact state that produced a letterhead PDF whose body read 'the case has
    not yet passed compliance validation'.
    """
    res = _render(payload={})
    assert res["figures"], "standalone drafting produced no figures — the letter would be a refusal"
    assert res["figures"]["monthly_repayment"] == _TRUE_INSTALMENT
    _body, _recipient, facts = letter_store.get_body(_AID, "IPA")
    assert facts["monthly_repayment"] == _TRUE_INSTALMENT


def test_printed_instalment_is_the_calculator_value():
    """Defect 1: the instalment in the PDF is the calculator's, never 2,299."""
    res = _render(payload={})
    assert res["figures"]["monthly_repayment"] != _HALLUCINATED, "the 2026-07-10 hallucination is back"

    truth = _figures_for(_AID)
    _body, recipient, facts = letter_store.get_body(_AID, "IPA")
    assert facts["loan_amount"] == truth["eligible_loan"]
    assert facts["property_price"] == truth["property_price"]
    assert facts["tenure_years"] == truth["loan_tenure_years"]
    assert facts["interest_rate_pct"] == truth["interest_rate_pct"]
    assert facts["monthly_repayment"] == truth["monthly_repayment"]
    # identity comes from state, never chosen by the model
    assert recipient["name"] == "Cassidy Flores"


def test_upstream_calculation_wins_over_the_csv_fallback():
    """In a full flow the agents' own calculate_loan result is authoritative; the
    CSV re-derivation is only a fallback."""
    upstream = {"eligible_loan": 999.0, "property_price": 1.0, "loan_tenure_years": 5,
                "interest_rate_pct": 2.0, "monthly_repayment": 17.5}
    res = _render(payload={"figures": upstream})
    assert res["figures"]["monthly_repayment"] == 17.5
    assert res["figures"]["loan_amount"] == 999.0


def test_rm_rate_override_reprices_the_letter():
    """reject -> replan(recompute) -> the letter must show the repriced instalment."""
    base = _render(payload={})["figures"]["monthly_repayment"]
    bumped = _render(payload={}, overrides={"interest_rate_pct": 3.2})["figures"]
    assert bumped["interest_rate_pct"] == 3.2
    assert bumped["monthly_repayment"] > base


def test_uncleared_case_refuses_rather_than_inventing():
    """No calculation and no usable loan row => no figures, and an explicit
    instruction not to write a letter."""
    res = _render(applicant_id="NOPE", payload={})
    assert res["figures"] == {}
    status = res["status"].lower()
    assert "invent" in status and "no letter" in status


def test_prose_context_is_supplied_so_the_body_need_not_derive():
    """The broken letter's body hand-computed 'LTV approximately 68.3%' instead of
    the cleared 75% cap. The tool now hands those values back to quote."""
    ctx = _render(payload={})["context"]
    truth = _figures_for(_AID)
    assert ctx["ltv_limit_pct"] == truth["ltv_limit_pct"] == 75.0
    assert ctx["tdsr_pct"] == truth["tdsr_pct"]
    assert "monthly_repayment_stress" in ctx


def test_compliance_freezes_the_calculator_output():
    """`_collect_tool_results` is what the graph harvests into payload['figures'];
    it must surface the calculator dict verbatim."""
    truth = calculate_loan(
        borrowers=[{"age": 38, "monthly_income": 11250.0, "nationality": "Singapore Citizen"}],
        property_type="Private", n_outstanding_loans=0, n_props_owned=1,
        interest_rate_pct=1.31, monthly_car_loan=0.0, monthly_other=0.0,
        cash_cpf_available=626780.42)
    msgs = [ToolMessage(content=json.dumps(truth), name="calculate_loan", tool_call_id="t")]
    got = _collect_tool_results(msgs)["calculate_loan"]
    assert got["monthly_repayment"] == truth["monthly_repayment"]


def test_calculator_echoes_its_rate_input():
    """`interest_rate_pct` is an input, not a result; the letter needs it, so the
    calculator echoes it. Without this the rate could only come from the LLM."""
    assert _figures_for(_AID)["interest_rate_pct"] == 1.31


# ── what the rendered PDF actually shows ────────────────────────────────────
# The renderer owns the header, so anything the header already carries must not
# also appear in the body — the agent keeps restating it despite the prompt.

def _render_pdf(body: str) -> str:
    """Build a real PDF from `body` and return its extracted page text."""
    pypdf = pytest.importorskip("pypdf")
    truth = _figures_for(_AID)
    pdf, _name = build_letter_pdf(
        _AID, "IPA", body, draft=True,
        recipient={"name": "Cassidy Flores", "nric": "*********",
                   "property_detail": "Canninghill Piers, #40-14, Singapore 393328"},
        facts={"loan_amount": truth["eligible_loan"],
               "property_price": truth["property_price"],
               "tenure_years": truth["loan_tenure_years"],
               "interest_rate_pct": truth["interest_rate_pct"],
               "monthly_repayment": truth["monthly_repayment"]},
    )
    return pypdf.PdfReader(io.BytesIO(pdf)).pages[0].extract_text()


def test_applicant_id_appears_only_in_the_reference_line():
    """The address block is the customer's own details; the internal case number
    belongs in `Ref:` and nowhere else."""
    text = _render_pdf("We are pleased to inform you that your loan is approved in-principle.")
    assert text.count(_AID) == 1, f"{_AID} printed {text.count(_AID)}x; expected only the Ref: line"
    assert f"Ref: {BANK_REF}/IPA/{_AID}" in text
    assert "Applicant ID:" not in text


def test_body_metadata_lines_are_stripped():
    """The agent restates the header under the RE: heading, with a stale date
    ('20 May 2026', the application date) contradicting the letter date. The
    renderer drops any line whose header it already prints."""
    body = ("Date: 20 May 2026 Applicant: Cassidy Flores Applicant ID: APP0001\n\n"
            "We are pleased to inform you that your application has been approved.\n\n"
            "To proceed toward a formal Letter of Offer, please submit your documents.\n\n"
            "DRAFT — pending RM review.")
    text = _render_pdf(body)
    assert "20 May 2026" not in text, "the agent's stale date line survived into the PDF"
    assert "Applicant: Cassidy Flores" not in text
    # ...while the real prose survives, including a sentence that opens with "To "
    assert "pleased to inform" in text
    assert "To proceed toward" in text
