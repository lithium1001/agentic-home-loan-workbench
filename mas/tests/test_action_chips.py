"""Guards the Assistant panel's action chips, and the COMPARE track's in particular.

The Package Comparison track is client-only: it has no server-side stage record, so
it is deliberately absent from STAGE_KEYS (which drives case progress, documents and
the HITL gate). Its chips therefore have to be attached to the case payload
separately — easy to lose in a refactor, and losing them leaves the panel telling
the RM to "Pick an action below" with nothing below. No LLM / network.
"""

import re
from pathlib import Path

from server.case_service import ACTION_CHIPS, STAGE_KEYS, _next_action_chips

_APP_JS = Path(__file__).resolve().parent.parent / "server" / "static" / "app.js"


def _cmp_field_defaults():
    """The Package Comparison panel's default input values, read out of app.js's
    CMP_FIELDS. Returns {field id: default value}."""
    src = _APP_JS.read_text(encoding="utf-8")
    block = re.search(r"const CMP_FIELDS = \[(.*?)\n\];", src, re.S)
    assert block, "CMP_FIELDS not found in app.js — the panel was restructured"
    return {
        m.group("id"): float(m.group("val"))
        for m in re.finditer(
            r"id:\s*'(?P<id>\w+)'.*?value:\s*(?P<val>[\d.]+)", block.group(1)
        )
    }


def test_compare_is_not_a_server_side_stage():
    """If COMPARE ever joins STAGE_KEYS it acquires progress/docs/HITL semantics it
    has no data for — the separate wiring is the point, not an oversight."""
    assert "COMPARE" not in STAGE_KEYS


def test_compare_has_chips():
    assert ACTION_CHIPS.get("COMPARE"), "the Package Comparison panel would be empty"


def test_compare_chips_have_no_primary():
    """A what-if tool has no next-best action, so nothing should be highlighted."""
    chips = _next_action_chips("COMPARE", "none", [])
    assert chips
    assert not any(c["primary"] for c in chips)


def test_compare_chips_keep_authored_order():
    """_next_action_chips sorts the recommended chip to the front; with none
    recommended the authored order (read → pressure-test → customer-facing) must
    survive."""
    labels = [c["label"] for c in _next_action_chips("COMPARE", "none", [])]
    assert labels == ACTION_CHIPS["COMPARE"]


def test_compare_chips_are_self_contained():
    """The assistant cannot see the comparison panel's inputs — the panel posts to
    the calculator API, not into the chat history. A chip that says "my two
    scenarios" gets "I don't have any scenarios in our current conversation to
    compare" (observed live), so the numeric chip must carry its own loan terms."""
    numeric = [c for c in ACTION_CHIPS["COMPARE"] if "S$" in c]
    assert numeric, "no chip states its own loan terms"
    for chip in numeric:
        assert "%" in chip and "months" in chip


def test_served_compare_chips_match_the_panel_defaults():
    """These chips are only the FALLBACK: app.js's compareChips() overrides them with
    the RM's live inputs as soon as the panel holds a real loan. They are what an RM
    sees before touching anything, so they must state the panel's own default terms —
    otherwise the fallback quotes a loan the panel is not showing. If CMP_FIELDS
    changes, this test fails and the chip text has to follow."""
    d = _cmp_field_defaults()
    scenario = ACTION_CHIPS["COMPARE"][0]

    # S$1.2m written the way the chip words it, from CMP_FIELDS' 1200000.
    assert f"S${d['cmpLoan'] / 1_000_000:.1f}m".replace(".0m", "m") in scenario
    assert f"{d['cmpRate']:g}%" in scenario
    assert f"{d['cmpTenure']:g} months" in scenario
    assert f"{d['cmpRateA']:g}%" in scenario
    assert f"{d['cmpRateB']:g}%" in scenario
    assert f"{d['cmpAfter']:g} months" in scenario


def test_compare_chips_never_state_a_computed_saving():
    """The chips carry the panel's INPUTS only. Pasting the computed savings in would
    leave the assistant restating arithmetic it never ran, breaking the tool-call
    audit trail that makes its figures trustworthy."""
    for chip in ACTION_CHIPS["COMPARE"]:
        assert not re.search(r"sav(es|ings) (of|:)\s*S\$", chip, re.I)


def test_every_stage_with_chips_builds():
    """Every chip list must survive _next_action_chips without raising, for both an
    active and an inactive stage."""
    for stage in ACTION_CHIPS:
        for status in ("active", "none", "done", "pending"):
            out = _next_action_chips(stage, status, [])
            assert len(out) == len(ACTION_CHIPS[stage])
            assert all(isinstance(c["label"], str) and c["label"] for c in out)
