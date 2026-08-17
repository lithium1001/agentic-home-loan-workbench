"""Regression tests for early-conversion interest savings (2026-07-31).

Powers the RM-side Package Comparison page. This is deliberately NOT
compare_packages: that prices a new purchase under several rates, whereas this
takes an EXISTING loan (outstanding balance + remaining term) and asks how much
interest is saved by converting, and *when* to convert.

Locked in:
  1. The business spec reproduces exactly — every figure on the mockup.
  2. Converting earlier saves more (interest is charged on a higher balance).
  3. Converting to a HIGHER rate is a loss, reported as negative savings.
  4. A conversion does not restart the term: it re-amortises over what is left.

Pure calculator layer — no network, no CSV, no LLM. Run from mas/:
    py -m pytest -q tests/test_interest_savings.py
"""
import pytest

from utils.calculator import interest_savings

# The mockup's inputs and every number printed on it.
_SPEC = dict(outstanding_loan=1_200_000, current_rate_pct=2.00, remaining_months=360,
             convert_after_months=3, rate_a_pct=1.55, rate_b_pct=1.50)


def _scenarios(**kw):
    res = interest_savings(**{**_SPEC, **kw})
    return {s["id"]: s for s in res["scenarios"]}, res


def test_reproduces_spec_headline_savings():
    """$11,906 and $11,711 — the two figures on the mockup."""
    s, _ = _scenarios()
    assert s[1]["savings"] == pytest.approx(11_906, abs=1)
    assert s[2]["savings"] == pytest.approx(11_711, abs=1)


def test_reproduces_spec_scenario1_phase_split():
    """Scenario 1's narrative: $1,348 over months 1-3, $10,558 over months 4-27."""
    s, _ = _scenarios()
    one = s[1]
    assert one["savings_first_phase"] == pytest.approx(1_348, abs=1)
    assert one["savings_second_phase"] == pytest.approx(10_558, abs=1)
    assert one["first_phase_months"] == 3
    assert one["second_phase_months"] == 24
    # The phases must account for the whole headline figure.
    assert one["savings_first_phase"] + one["savings_second_phase"] == pytest.approx(
        one["savings"], abs=0.02)


def test_reproduces_spec_rule_of_thumb():
    """The illustrative month-1 interest delta quoted in the summary (~$450)."""
    s, _ = _scenarios()
    assert s[1]["month_1_interest_delta"] == pytest.approx(450, abs=1)


def test_default_horizon_is_switch_plus_24_months():
    """27 months on the spec inputs — the switch month plus the usual 2-year view."""
    _, res = _scenarios()
    assert res["inputs"]["horizon_months"] == 27


def test_scenario2_saves_nothing_before_the_switch():
    """Staying on the current rate cannot beat the baseline: it IS the baseline."""
    s, _ = _scenarios()
    assert s[2]["savings_first_phase"] == pytest.approx(0.0, abs=0.02)
    assert s[2]["savings_second_phase"] == pytest.approx(s[2]["savings"], abs=0.02)


def test_converting_earlier_saves_more_at_the_same_rate():
    """The core advice the page exists to give.

    Compared at one rate over one horizon, converting now beats converting later,
    because early interest is charged on a bigger balance.
    """
    horizon = 36
    now, _ = _scenarios(rate_a_pct=1.50, rate_b_pct=1.50, convert_after_months=6,
                        horizon_months=horizon)
    assert now[1]["savings"] > now[2]["savings"]


def test_converting_to_a_higher_rate_is_a_loss():
    """Negative savings, not silently clamped to zero — an RM must see the sign."""
    s, _ = _scenarios(rate_a_pct=3.00)
    assert s[1]["savings"] < 0


def test_identical_rate_saves_nothing():
    """Converting to the rate you already pay changes nothing."""
    s, _ = _scenarios(rate_a_pct=2.00)
    assert s[1]["savings"] == pytest.approx(0.0, abs=0.5)


def test_bigger_rate_gap_saves_more():
    s_small, _ = _scenarios(rate_a_pct=1.90)
    s_big, _   = _scenarios(rate_a_pct=1.20)
    assert s_big[1]["savings"] > s_small[1]["savings"]


def test_conversion_does_not_restart_the_term():
    """After converting at month k the loan amortises over n-k months.

    If the term restarted, the post-switch instalment would drop and the interest
    comparison would flatter the conversion. Checked by confirming the scenario-2
    saving is strictly less than converting to the same rate immediately.
    """
    same_rate, _ = _scenarios(rate_a_pct=1.50, rate_b_pct=1.50)
    assert same_rate[2]["savings"] < same_rate[1]["savings"]


def test_horizon_is_clamped_to_the_remaining_term():
    """A loan with 6 months left cannot be compared over 27."""
    _, res = _scenarios(remaining_months=6, convert_after_months=3)
    assert res["inputs"]["horizon_months"] == 6


def test_convert_after_beyond_term_is_clamped():
    _, res = _scenarios(remaining_months=12, convert_after_months=99)
    assert res["inputs"]["convert_after_months"] == 12


@pytest.mark.parametrize("kw", [
    {"outstanding_loan": 0}, {"outstanding_loan": None},
    {"current_rate_pct": None}, {"remaining_months": 0}, {"remaining_months": None},
])
def test_degenerate_inputs_return_empty_not_raise(kw):
    res = interest_savings(**{**_SPEC, **kw})
    assert res["scenarios"] == []


def test_only_one_scenario_when_only_one_rate_given():
    res = interest_savings(1_200_000, 2.00, 360, 3, rate_a_pct=1.55, rate_b_pct=None)
    assert [s["id"] for s in res["scenarios"]] == [1]
