"""Guards the `interest_savings` agent tool — the COMPARE track's numeric backing.

Background (2026-08-04). The Package Comparison panel computed its figures through
`utils.calculator.interest_savings` via `POST /api/compare/savings`, but that function
was never registered as an agent tool. The panel's action chips ask the assistant the
same question the panel just answered, and with no tool able to answer it the model
did the arithmetic in its head: it quoted the *monthly instalment* difference
(~S$3/month) as the saving, where the real answer is ~S$11.9k of interest. Answer
happened to reach the right verdict for invented reasons.

These tests lock in the fix:
  1. the tool exists, is reachable by the agent that the COMPARE track routes to, and
  2. it returns exactly what the panel's endpoint returns (one code path, no drift), and
  3. the shortcut the model reached for is demonstrably not the answer.

Pure calculator/registry layer — no network, no LLM. Run from mas/:
    py -m pytest -q tests/test_interest_savings_tool.py
"""
import pytest

from graph import _AGENT_TOOLS
from utils.calculator import interest_savings
from utils.tools import TOOLS_BY_NAME

# The panel's default inputs (CMP_FIELDS in server/static/app.js).
_PANEL = dict(
    outstanding_loan=1_200_000,
    current_rate_pct=2.0,
    remaining_months=360,
    convert_after_months=3,
    rate_a_pct=1.55,
    rate_b_pct=1.50,
)


def test_tool_is_registered():
    """Without this the agent has no way to answer the panel's own question."""
    assert "interest_savings" in TOOLS_BY_NAME


def test_reprice_agent_can_call_it():
    """The COMPARE track routes to reprice_retention (see skills/orchestrator).
    A tool missing from the allowlist cannot be called, however good the prompt is."""
    assert "interest_savings" in _AGENT_TOOLS["reprice_retention"]


def test_tool_matches_the_panel_figures():
    """The tool and the panel must be one code path: an assistant answer that
    contradicts the figures on screen beside it is worse than no answer."""
    direct = interest_savings(**_PANEL)
    via_tool = TOOLS_BY_NAME["interest_savings"].invoke(dict(_PANEL))
    assert via_tool == direct

    savings = {s["id"]: s["savings"] for s in via_tool["scenarios"]}
    # Values shown by the panel at its defaults, verified on screen 2026-08-04.
    assert savings[1] == pytest.approx(11_906, abs=1)
    assert savings[2] == pytest.approx(11_711, abs=1)


def test_omitting_convert_after_months_changes_scenario_1_too():
    """Why the assistant contradicted the panel on screen (2026-08-04).

    The panel showed Scenario 1 = S$11,906 while the chat answer said S$10,608.93
    for the same loan. Both are this function; the answer had simply left
    `convert_after_months` out, and it is NOT only scenario 2's start date — it also
    sets the comparison window (default N + 24), so dropping it shortened the window
    from 27 months to 24 and quietly understated scenario 1 by ~S$1,297.

    Pinned because the divergence is invisible from either figure alone: each is a
    correct amortisation over its own window, so nothing looks wrong until the two
    are read side by side, which is exactly what the RM does.
    """
    full = dict(_PANEL, rate_b_pct=1.40)
    with_k = interest_savings(**full)
    without_k = interest_savings(**{**full, "convert_after_months": 0})

    assert with_k["inputs"]["horizon_months"] == 27
    assert without_k["inputs"]["horizon_months"] == 24

    s1_with = next(s for s in with_k["scenarios"] if s["id"] == 1)["savings"]
    s1_without = next(s for s in without_k["scenarios"] if s["id"] == 1)["savings"]
    assert s1_with == pytest.approx(11_906, abs=1)      # the panel's figure
    assert s1_without == pytest.approx(10_608.93, abs=1)  # the chat's figure
    assert s1_with - s1_without == pytest.approx(1_297, abs=5)


def test_tool_docstring_warns_about_the_window():
    """The tool description is the only thing the model reads at call time, so the
    coupling between convert_after_months and the window has to be stated there."""
    doc = TOOLS_BY_NAME["interest_savings"].description
    assert "convert_after_months" in doc
    assert "window" in doc.lower()


def test_converting_now_beats_waiting_at_these_inputs():
    """The verdict the panel gives, and the one the assistant must reproduce."""
    s = {x["id"]: x["savings"] for x in interest_savings(**_PANEL)["scenarios"]}
    assert s[1] > s[2]
    # …but only just. The model narrated this as a large gap; it is ~S$195, which
    # is the kind of detail an RM would repeat to a customer.
    assert s[1] - s[2] == pytest.approx(195, abs=15)


def test_instalment_difference_is_not_the_saving():
    """Reproduces the actual hallucination: quoting the monthly-repayment delta.

    The model said waiting for 1.50% saves "an additional S$3 per month … roughly
    S$972 in total". That is wrong twice over: the instalment delta is really
    S$28.85/month, and the instalment delta is the wrong quantity anyway — the
    saving is interest on a falling balance. This test pins the gap between the
    shortcut and the truth so the prompt guardrail has a reason to exist.
    """
    def instalment(rate_pct):
        """Standard amortisation payment — the figure the model was comparing."""
        r = rate_pct / 100 / 12
        n = _PANEL["remaining_months"]
        p = _PANEL["outstanding_loan"]
        return p * r / (1 - (1 + r) ** -n)

    monthly_delta = abs(instalment(_PANEL["rate_a_pct"]) - instalment(_PANEL["rate_b_pct"]))
    assert monthly_delta < 40, "the instalment gap really is small — that is the trap"

    # The model multiplied a small monthly delta out to ~S$972. The true difference
    # between the two scenarios is a different quantity entirely.
    s = {x["id"]: x["savings"] for x in interest_savings(**_PANEL)["scenarios"]}
    assert s[1] > 10_000, "scenario savings are four figures, not hundreds"
    assert monthly_delta * _PANEL["remaining_months"] < s[1], (
        "instalment delta over the tenure must not be mistaken for the saving"
    )


def test_missing_inputs_return_empty_not_a_guess():
    """A tool that invents a number when under-specified would defeat the point."""
    for bad in ({"outstanding_loan": 0}, {"remaining_months": 0}, {"current_rate_pct": None}):
        out = interest_savings(**{**_PANEL, **bad})
        assert out["scenarios"] == []


def test_earlier_conversion_saves_more_at_the_same_rate():
    """The tool's core claim, and what the assistant is asked to explain: interest is
    charged on a balance that falls, so the same rate cut is worth more earlier."""
    early = interest_savings(**{**_PANEL, "convert_after_months": 3, "rate_b_pct": 1.55})
    late = interest_savings(**{**_PANEL, "convert_after_months": 24, "rate_b_pct": 1.55,
                               "horizon_months": 60})
    early_s2 = next(s for s in early["scenarios"] if s["id"] == 2)["savings"]
    late_s2 = next(s for s in late["scenarios"] if s["id"] == 2)["savings"]
    # Same rate, same horizon length would favour the earlier switch; compare per
    # month of post-switch exposure to keep the two windows comparable.
    assert early_s2 / max(1, 27 - 3) > late_s2 / max(1, 60 - 24)
