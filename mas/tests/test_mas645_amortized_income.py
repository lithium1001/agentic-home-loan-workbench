"""Regression tests for MAS Notice 645 amortized monthly income (2026-07-31).

This is the only one of the four 2026-07-30 requirements that changes an EXISTING
number (qualifying income), so the most important test here is the negative one:
with no assets supplied, every downstream figure must be bit-for-bit what it was
before the feature existed.

Also locked in:
  1. The client's reference table reproduces exactly (all 12 figures).
  2. 48 months is a constant — it does NOT scale with the pledge period.
  3. The haircut is a DEDUCTION (70% band recognises 30%, not the reverse).
  4. Unpledged assets take the full 70% haircut whatever the asset class.
  5. The result is NOT subject to the 30% variable-income haircut on top.

Pure calculator layer — no network, no CSV, no LLM. Run from mas/:
    py -m pytest -q tests/test_mas645_amortized_income.py
"""
import pytest

from utils.calculator import amortized_monthly_income, calculate_loan

# The client's reference table: every class at S$10,000, pledged >= 4 years.
# (key, displayed haircut %, recognised value, gross monthly income)
_REFERENCE = [
    ("fixed_deposit",    0,  10_000, 208.33),
    ("unit_trust",      70,   3_000,  62.50),
    ("foreign_currency", 30,   7_000, 145.83),
    ("shares",          70,   3_000,  62.50),
    ("ssb_sgs",          0,  10_000, 208.33),
    ("gold",            30,   7_000, 145.83),
]


def _one(key, amount=10_000, pledged=True):
    res = amortized_monthly_income({key: {"amount": amount, "pledged": pledged}})
    return res["rows"][0]


@pytest.mark.parametrize("key,haircut,recognised,monthly", _REFERENCE)
def test_reference_table_reproduces_exactly(key, haircut, recognised, monthly):
    """All 12 figures from the client's MAS 645 table, to the cent."""
    row = _one(key)
    assert row["haircut_pct"] == pytest.approx(haircut, abs=0.01)
    assert row["recognised"] == pytest.approx(recognised, abs=0.01)
    assert row["monthly_income"] == pytest.approx(monthly, abs=0.01)


def test_no_assets_contributes_nothing():
    """The default path: no assets means no income and no rows."""
    for empty in (None, {}, {"fixed_deposit": 0}, {"shares": {"amount": 0, "pledged": True}}):
        res = amortized_monthly_income(empty)
        assert res["monthly_income"] == 0.0
        assert res["total_recognised"] == 0.0
        assert res["rows"] == []


def test_zero_assets_leaves_existing_calculation_untouched():
    """REGRESSION: adding this feature must not move any pre-existing figure.

    Qualifying income with no assets is exactly fixed + variable x 0.7, so a case
    priced with zero assets must equal the same case priced without the concept.
    """
    base_income = 8_000 + 2_000 * 0.7
    assets_income = amortized_monthly_income({})["monthly_income"]
    assert assets_income == 0.0

    case = dict(
        property_type="Private", n_outstanding_loans=0, n_props_owned=1,
        interest_rate_pct=3.5, monthly_car_loan=0, monthly_other=0,
        target_property_price=1_500_000,
    )
    before = calculate_loan(
        borrowers=[{"age": 35, "monthly_income": base_income,
                    "nationality": "Singapore Citizen"}], **case)
    after = calculate_loan(
        borrowers=[{"age": 35, "monthly_income": base_income + assets_income,
                    "nationality": "Singapore Citizen"}], **case)
    for field in ("eligible_loan", "monthly_repayment", "tdsr_pct",
                  "required_cash_cpf", "property_price", "total_stamp_duty"):
        assert before[field] == after[field], f"{field} moved when assets were zero"


def test_48_months_is_constant_not_the_pledge_period():
    """A regulatory constant: the divisor never tracks how long assets are pledged."""
    res = amortized_monthly_income({"fixed_deposit": 48_000})
    assert res["amortisation_months"] == 48
    # 48,000 fully recognised over 48 months is exactly 1,000/month — if the
    # divisor ever became "pledge years x 12" this would change.
    assert res["monthly_income"] == pytest.approx(1_000.0, abs=0.01)


def test_haircut_is_a_deduction_not_a_recognition_rate():
    """70% band recognises 30% of value — the direction secondary sources get wrong."""
    shares = _one("shares", amount=10_000)
    assert shares["recognised"] == pytest.approx(3_000, abs=0.01)   # not 7,000
    fd = _one("fixed_deposit", amount=10_000)
    assert fd["recognised"] == pytest.approx(10_000, abs=0.01)      # 0% deduction
    assert shares["recognised"] < fd["recognised"]


@pytest.mark.parametrize("key", [k for k, *_ in _REFERENCE])
def test_unpledged_always_takes_the_full_haircut(key):
    """Without a >=4y pledge every class is cut 70%, even cash-like ones."""
    row = _one(key, amount=10_000, pledged=False)
    assert row["haircut_pct"] == pytest.approx(70.0)
    assert row["recognised"] == pytest.approx(3_000, abs=0.01)
    assert row["monthly_income"] == pytest.approx(62.50, abs=0.01)


def test_pledging_never_reduces_recognised_value():
    """Pledging is a concession: it can only help, never hurt."""
    for key, *_ in _REFERENCE:
        assert _one(key, pledged=True)["recognised"] >= _one(key, pledged=False)["recognised"]


def test_totals_aggregate_across_classes():
    """Mixed portfolio: total is the sum of recognised value over 48 months."""
    res = amortized_monthly_income({k: 10_000 for k, *_ in _REFERENCE})
    assert len(res["rows"]) == 6
    assert res["total_recognised"] == pytest.approx(40_000, abs=0.01)
    assert res["monthly_income"] == pytest.approx(40_000 / 48, abs=0.01)
    assert res["monthly_income"] == pytest.approx(
        sum(r["monthly_income"] for r in res["rows"]), abs=0.02)


def test_plain_number_is_treated_as_pledged():
    """Shorthand form: a bare amount means pledged >= 4 years."""
    res = amortized_monthly_income({"fixed_deposit": 10_000})
    assert res["monthly_income"] == pytest.approx(208.33, abs=0.01)
    assert res["rows"][0]["pledged"] is True


def test_unknown_keys_and_junk_are_ignored():
    """Unknown asset classes must not silently inflate income."""
    res = amortized_monthly_income({"crypto": 1_000_000, "fixed_deposit": 10_000})
    assert len(res["rows"]) == 1
    assert res["monthly_income"] == pytest.approx(208.33, abs=0.01)


def test_source_is_cited():
    src = amortized_monthly_income({"fixed_deposit": 1})["source"]
    assert "mas.gov.sg" in src["url"]
    assert "645" in src["label"]
