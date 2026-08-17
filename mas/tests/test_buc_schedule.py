"""Regression tests for the BUC progressive payment schedule (2026-07-31).

Two things must stay true, and they pull in different directions:

  1. It reproduces the business spec table (manager's 2026-07-30 screenshot) —
     the funding split and the rising instalment column, to the cent.
  2. It follows the STATUTORY schedule (Housing Developers Rules, prescribed S&P
     agreement, clause 5), which sums to 100% of the purchase price. The spec
     table only went as far as 60% of price, so the statutory tail (car park 5%,
     TOP 25%, CSC 15%) is present here and deliberately goes BEYOND the spec.

The bridge between them is the final instalment: once the whole loan is drawn,
the BUC table's monthly figure must equal the ordinary amortising instalment for
the same loan — i.e. this feature and the A2 schedule cannot disagree.

Pure calculator layer — no network, no CSV, no LLM. Run from mas/:
    py -m pytest -q tests/test_buc_schedule.py
"""
import pytest

from utils.calculator import (
    _monthly_pmt,
    amortization_schedule,
    buc_progressive_schedule,
)

_PRICE, _RATE, _TENURE = 1_600_000, 1.80, 30


def _rows(**kw):
    return buc_progressive_schedule(_PRICE, _RATE, _TENURE, **kw)["rows"]


def _by_stage(rows, needle):
    return next(r for r in rows if needle.lower() in r["stage"].lower())


def test_reproduces_spec_monthly_column():
    """The rising instalment column matches the manager's screenshot exactly.

    Screenshot (1.6M @ 1.80%, 30y, LTV 75%): the loan starts drawing at Foundation
    (split 50/50 with the last of the cash), and the instalment climbs as each
    later stage is drawn.
    """
    rows = _rows(ltv_pct=75.0)
    expected = [
        ("Foundation",                     287.76),
        ("Reinforced Concrete Framework",   863.28),
        ("Partition Walls",                1151.04),
        ("Roofing / Ceiling",              1438.79),
        ("Doors, Windows",                 1726.55),
    ]
    for needle, monthly in expected:
        assert _by_stage(rows, needle)["monthly_repayment"] == pytest.approx(monthly, abs=0.01)


def test_spec_funding_split_and_zero_instalments():
    """Cash/CPF is spent first; nothing is repayable until the loan starts drawing."""
    rows = _rows(ltv_pct=75.0)
    # Everything up to and including the S&P/fees is pure cash, hence no instalment.
    for needle in ("Booking Fee", "Sign S&P", "Stamp Duty", "Legal fees", "Valuation fees"):
        row = _by_stage(rows, needle)
        assert row["loan"] == 0
        assert row["monthly_repayment"] == 0.0
    # Foundation straddles the cash/loan crossover: 80k cash + 80k loan on 160k.
    foundation = _by_stage(rows, "Foundation")
    assert foundation["cash_cpf"] == pytest.approx(80_000, abs=1)
    assert foundation["loan"] == pytest.approx(80_000, abs=1)
    assert foundation["funding"] == "50% Cash/CPF 50% Loan"


def test_stage_percentages_sum_to_statutory_100():
    """The statutory schedule covers the whole price — the spec table's 60% was partial."""
    res = buc_progressive_schedule(_PRICE, _RATE, _TENURE)
    assert res["totals"]["pct_of_price_total"] == pytest.approx(100.0, abs=0.01)
    # Every price-linked stage's amount sums to exactly the purchase price. The
    # S&P row DISPLAYS the statutory 20% (inclusive of the booking fee) but is
    # only charged the 15% balance, so there is no double-count.
    price_rows = [r for r in res["rows"] if r["pct_of_price"] is not None]
    assert sum(r["amount_payable"] for r in price_rows) == pytest.approx(_PRICE, abs=1)
    snp = _by_stage(res["rows"], "Sign S&P")
    assert snp["pct_of_price"] == pytest.approx(20.0)                    # statutory label
    assert snp["amount_payable"] == pytest.approx(_PRICE * 0.15, abs=1)  # balance charged


def test_statutory_tail_is_present():
    """Car park, TOP and CSC — the 40% the business spec omitted."""
    rows = _rows()
    assert _by_stage(rows, "Car Park")["pct_of_price"] == pytest.approx(5.0)
    assert _by_stage(rows, "Temporary Occupation Permit")["pct_of_price"] == pytest.approx(25.0)
    assert _by_stage(rows, "Certificate of Statutory Completion")["pct_of_price"] == pytest.approx(15.0)


def test_final_instalment_equals_plain_amortising_instalment():
    """Once fully drawn, BUC and the ordinary schedule must agree.

    This is the anti-drift guard between the two customer-facing tables: a
    borrower must never see one monthly figure on the BUC card and a different
    one on the repayment schedule for the same loan.
    """
    res = buc_progressive_schedule(_PRICE, _RATE, _TENURE, ltv_pct=75.0)
    loan = res["totals"]["loan_total"]
    assert loan == pytest.approx(_PRICE * 0.75, abs=1)

    plain = amortization_schedule(loan, _RATE, _TENURE, max_rows=1)[0]["instalment"]
    assert res["totals"]["final_monthly_repayment"] == pytest.approx(plain, abs=0.02)
    assert res["totals"]["final_monthly_repayment"] == pytest.approx(
        _monthly_pmt(loan, _RATE, _TENURE), abs=0.02)


def test_income_capped_loan_caps_the_drawdown():
    """When income binds before LTV, BUC must draw only the eligible loan.

    Otherwise a borrower whose card says they can borrow 921,629 would see a BUC
    table climbing to the instalment for a 1.2M loan — the exact cross-table
    contradiction this feature must not produce.
    """
    eligible = 921_629
    res = buc_progressive_schedule(_PRICE, _RATE, _TENURE, ltv_pct=75.0,
                                   eligible_loan=eligible)
    assert res["totals"]["loan_total"] == pytest.approx(eligible, abs=1)
    assert res["totals"]["loan_capped_by"] == "Income (TDSR/MSR)"
    assert res["totals"]["final_monthly_repayment"] == pytest.approx(
        _monthly_pmt(eligible, _RATE, _TENURE), abs=0.02)
    # The extra shortfall is cash the buyer must find on top of the LTV downpayment.
    assert res["totals"]["cash_cpf_total"] > _PRICE * 0.25


def test_ltv_capped_case_is_labelled_as_such():
    res = buc_progressive_schedule(_PRICE, _RATE, _TENURE, ltv_pct=75.0,
                                   eligible_loan=_PRICE)   # income not binding
    assert res["totals"]["loan_capped_by"] == "Loan-to-value limit"
    assert res["totals"]["loan_total"] == pytest.approx(_PRICE * 0.75, abs=1)


def test_loan_total_respects_ltv():
    """Total drawn equals the LTV share of price; cash covers the rest (plus fees)."""
    for ltv in (55.0, 70.0, 75.0):
        res = buc_progressive_schedule(_PRICE, _RATE, _TENURE, ltv_pct=ltv)
        assert res["totals"]["loan_total"] == pytest.approx(_PRICE * ltv / 100, abs=1)


def test_cumulative_loan_is_monotonic_and_matches_rows():
    """Drawdown only ever increases, and the instalment tracks it."""
    rows = _rows(ltv_pct=75.0)
    running = 0.0
    prev_monthly = -1.0
    for r in rows:
        running += r["loan"]
        assert r["cumulative_loan"] == pytest.approx(running, abs=0.02)
        assert r["monthly_repayment"] >= prev_monthly - 0.01   # never goes down
        prev_monthly = r["monthly_repayment"]


def test_bsd_agrees_with_the_calculator_not_a_flat_percentage():
    """BSD comes from _calc_bsd, so it matches the duty shown elsewhere in the app."""
    from utils.calculator import _calc_bsd
    row = _by_stage(_rows(), "Stamp Duty")
    assert row["amount_payable"] == pytest.approx(_calc_bsd(_PRICE), abs=0.01)
    assert row["amount_payable"] == pytest.approx(49_600, abs=1)   # spec screenshot
    assert row["loan"] == 0        # duty is never financed by the loan


def test_fees_are_fixed_sums_not_price_percentages():
    """Legal/valuation are flat fees: doubling the price must not change them."""
    a = buc_progressive_schedule(1_600_000, _RATE, _TENURE)["rows"]
    b = buc_progressive_schedule(3_200_000, _RATE, _TENURE)["rows"]
    for needle in ("Legal fees", "Valuation fees"):
        assert _by_stage(a, needle)["amount_payable"] == _by_stage(b, needle)["amount_payable"]
    assert _by_stage(a, "Legal fees")["amount_payable"] == 3_000
    assert _by_stage(a, "Valuation fees")["amount_payable"] == 500
    # ...but they are overridable for a case with different quoted fees.
    c = buc_progressive_schedule(_PRICE, _RATE, _TENURE, legal_fees=4_200, valuation_fees=700)["rows"]
    assert _by_stage(c, "Legal fees")["amount_payable"] == 4_200
    assert _by_stage(c, "Valuation fees")["amount_payable"] == 700


def test_source_is_cited():
    """The stage table is statutory, so the citation travels with the data."""
    src = buc_progressive_schedule(_PRICE, _RATE, _TENURE)["source"]
    assert "sso.agc.gov.sg" in src["url"]
    assert "Housing Developers Rules" in src["label"]


@pytest.mark.parametrize("price,rate,tenure", [
    (0, 1.8, 30), (None, 1.8, 30), (1_600_000, None, 30),
    (1_600_000, 1.8, 0), (1_600_000, 1.8, None),
])
def test_degenerate_inputs_return_empty_not_raise(price, rate, tenure):
    res = buc_progressive_schedule(price, rate, tenure)
    assert res["rows"] == []
    assert res["source"]["url"]          # citation still available for the panel
