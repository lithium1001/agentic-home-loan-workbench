"""Regression tests for the month-by-month amortisation schedule (2026-07-30).

The schedule is a customer-facing table, so the risk it carries is the same one
that produced the hallucinated-instalment evidence: a figure on screen that
disagrees with the calculator that priced the case. These tests lock in:

  1. The manager's spec table reproduces to the cent (1.2M @ 1.80% over 30y).
  2. The schedule's instalment IS calculate_loan's monthly_repayment — the table
     can never drift from the KPI card above it.
  3. The identities hold on every row: interest accrues on the opening balance,
     opening − principal = closing, and instalment = interest + principal.
  4. The loan closes at exactly 0.00 after n months (no floating-point residue).
  5. max_rows truncates without changing the arithmetic.
  6. Degenerate inputs return [] instead of raising (it renders as an optional panel).

Pure calculator layer — no network, no CSV, no LLM. Run from mas/:
    py -m pytest -q tests/test_amortization_schedule.py
"""
import pytest

from utils.calculator import amortization_schedule, calculate_loan


# The exact case in the manager's 2026-07-30 requirement screenshot.
_SPEC_LOAN, _SPEC_RATE, _SPEC_TENURE = 1_200_000, 1.80, 30

# Rows 1-5 as printed in that screenshot: (beginning, instalment, interest, principal, ending)
_SPEC_ROWS = [
    (1_200_000.00, 4316.38, 1800.00, 2516.38, 1_197_483.62),
    (1_197_483.62, 4316.38, 1796.23, 2520.16, 1_194_963.46),
    (1_194_963.46, 4316.38, 1792.45, 2523.94, 1_192_439.52),
    (1_192_439.52, 4316.38, 1788.66, 2527.72, 1_189_911.80),
    (1_189_911.80, 4316.38, 1784.87, 2531.51, 1_187_380.29),
]


def test_matches_manager_spec_table_to_the_cent():
    """The five rows in the requirement screenshot reproduce exactly."""
    rows = amortization_schedule(_SPEC_LOAN, _SPEC_RATE, _SPEC_TENURE, max_rows=5)
    assert len(rows) == 5
    for row, (begin, inst, interest, principal, end) in zip(rows, _SPEC_ROWS):
        assert row["beginning_balance"] == pytest.approx(begin, abs=0.01)
        assert row["instalment"] == pytest.approx(inst, abs=0.01)
        assert row["interest_paid"] == pytest.approx(interest, abs=0.01)
        assert row["principal_paid"] == pytest.approx(principal, abs=0.01)
        assert row["ending_balance"] == pytest.approx(end, abs=0.01)


def test_instalment_agrees_with_calculate_loan():
    """The table's instalment is the SAME number the result card shows.

    This is the anti-drift guard: the schedule must never present a monthly
    figure the case's own calculation did not produce.
    """
    calc = calculate_loan(
        borrowers=[{"age": 35, "monthly_income": 12000, "nationality": "Singapore Citizen"}],
        property_type="Private",
        n_outstanding_loans=0,
        n_props_owned=1,
        interest_rate_pct=3.5,
        monthly_car_loan=0,
        monthly_other=0,
        target_property_price=1_500_000,
    )
    rows = amortization_schedule(
        calc["eligible_loan"], calc["interest_rate_pct"], calc["loan_tenure_years"], max_rows=1)
    # calculate_loan rounds the repayment to the dollar for display; the schedule
    # keeps cents, so they agree to within that rounding.
    assert rows[0]["instalment"] == pytest.approx(calc["monthly_repayment"], abs=1.0)


def test_row_identities_hold_throughout():
    """interest = opening × r; instalment = interest + principal; opening − principal = closing.

    Tolerance is 1 cent, not 0: balances are carried at full precision and each
    column is rounded independently for display, so two rounded columns can sit a
    cent from the third. Anything larger would be a genuine error in the schedule.
    """
    rate, tenure = 2.4, 25
    rows = amortization_schedule(900_000, rate, tenure)
    r_monthly = (rate / 100) / 12
    for row in rows:
        assert row["interest_paid"] == pytest.approx(row["beginning_balance"] * r_monthly, abs=0.01)
        assert row["instalment"] == pytest.approx(
            row["interest_paid"] + row["principal_paid"], abs=0.011)
        assert row["ending_balance"] == pytest.approx(
            row["beginning_balance"] - row["principal_paid"], abs=0.011)


def test_balance_chains_and_closes_at_zero():
    """Each row opens where the previous closed, and the loan ends at exactly 0."""
    tenure = 30
    rows = amortization_schedule(_SPEC_LOAN, _SPEC_RATE, tenure)
    assert len(rows) == tenure * 12
    for prev, nxt in zip(rows, rows[1:]):
        assert nxt["beginning_balance"] == pytest.approx(prev["ending_balance"], abs=0.01)
    assert rows[-1]["ending_balance"] == 0.0


def test_total_principal_repaid_equals_the_loan():
    """Principal columns sum to the original loan — nothing created or lost."""
    rows = amortization_schedule(_SPEC_LOAN, _SPEC_RATE, _SPEC_TENURE)
    assert sum(r["principal_paid"] for r in rows) == pytest.approx(_SPEC_LOAN, abs=0.05)


def test_max_rows_truncates_without_changing_math():
    """A truncated view holds the same rows as the full schedule's prefix."""
    full = amortization_schedule(_SPEC_LOAN, _SPEC_RATE, _SPEC_TENURE)
    head = amortization_schedule(_SPEC_LOAN, _SPEC_RATE, _SPEC_TENURE, max_rows=12)
    assert len(head) == 12
    assert head == full[:12]


@pytest.mark.parametrize("loan,rate,tenure", [
    (0, 3.5, 30),          # no loan
    (None, 3.5, 30),       # missing loan
    (500_000, None, 30),   # missing rate
    (500_000, 3.5, 0),     # no tenure
    (500_000, 3.5, None),  # missing tenure
])
def test_degenerate_inputs_return_empty_not_raise(loan, rate, tenure):
    """Renders as an optional panel, so bad inputs must degrade quietly."""
    assert amortization_schedule(loan, rate, tenure) == []


def test_zero_interest_loan_is_straight_line():
    """A 0% loan repays principal evenly and still closes at zero."""
    rows = amortization_schedule(120_000, 0.0, 10)
    assert all(r["interest_paid"] == 0.0 for r in rows)
    assert rows[0]["principal_paid"] == pytest.approx(1000.0, abs=0.01)
    assert rows[-1]["ending_balance"] == 0.0
