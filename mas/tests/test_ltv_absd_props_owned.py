"""`n_props_owned` convention regression — LTV + ABSD tiers.

Bug found 2026-07-13. `calculate_loan` follows the reference implementation
(`redundant/mortgage_planner.py`), where the property count **includes the
property being bought**:

    prop_count = st.number_input("Property Count (incl. this)", 1, 10, 1)
    ltv = 0.75 if prop_count == 1 else (0.45 if prop_count == 2 else 0.35)

The calculator's tier table was correct. The DATA was not: the generator sampled
`props_owned` as properties *already* owned (0 for a first-time buyer) and wrote
that straight into `fact_loan_applications.no_sg_properties_owned`, while every
consumer read it as the calculator's "incl. this" count. Off by one, and it broke
in both directions:

  * first-time buyer (stored 0): LTV fell through to 35% (should be 75%), and
    `idx = n - 1 = -1` made Python's negative indexing silently return the LAST
    ABSD tier — a citizen who owes nothing was charged the 30% third-property
    rate ($537,600 on a $1.79M flat).
  * second-property buyer (stored 1): LTV read as 75% (should be 45%) and ABSD as
    0% (should be 20%) — i.e. over-lent and under-taxed.

Fix (2026-07-13): the CSV and the generator now store the count INCLUDING this
purchase, so every consumer reads the column directly with no conversion. The
calculation logic is untouched. The clamps below are belt-and-braces so an
out-of-convention value can never again silently pick the harshest ABSD tier.
"""

import pandas as pd
import pytest

from utils.calculator import _calc_absd, _calc_bsd, _calc_ltv, calculate_loan
from utils.config import DATA_DIR

CITIZEN = [{"age": 38, "monthly_income": 11_250, "nationality": "Singapore Citizen"}]


# ── BSD: the reference planner's constants are WRONG; IRAS is the authority ──
# reference code/mortgage_planner_.py hardcodes cumulative constants per bracket,
# and from $1.5M up they are inflated by exactly $25,000 (it uses 69,600 where the
# rate chain gives 44,600, and 144,600 where it gives 119,600). The table is
# discontinuous at $1.5M as a result — the same price yields two different duties
# depending on which branch you enter. calculator._calc_bsd derives from the rate
# chain and matches IRAS, so it is correct and must NOT be "fixed" to match the
# reference. These cases pin it.
#   IRAS residential BSD (from 15 Feb 2023): first 180k @1%, next 180k @2%,
#   next 640k @3%, 1.0-1.5M @4%, 1.5-3.0M @5%, above 3M @6%.
@pytest.mark.parametrize(
    "price, expected",
    [
        (150_000, 1_500),        # within the first 1% band
        (360_000, 5_400),        # exactly on the 2% -> 3% boundary
        (1_000_000, 24_600),     # exactly on the 3% -> 4% boundary
        (1_500_000, 44_600),     # the boundary the reference gets wrong (says 69,600)
        (1_792_000, 59_200),     # APP0001's property
        (3_000_000, 119_600),    # reference says 144,600
        (4_000_000, 179_600),
    ],
)
def test_bsd_matches_iras_not_the_reference_constants(price, expected):
    assert _calc_bsd(price) == pytest.approx(expected)


def test_bsd_is_continuous_at_every_bracket_boundary():
    """A progressive tax table must not jump at a boundary.

    This is the property that proves the reference's constants are wrong: at
    $1.5M its two branches disagree by $25,000. Ours must never do that.
    """
    for boundary in (180_000, 360_000, 1_000_000, 1_500_000, 3_000_000):
        below = _calc_bsd(boundary)
        above = _calc_bsd(boundary + 0.01)
        assert above - below < 1.0, f"BSD jumps at ${boundary:,}"


# ── Tier tables: the count INCLUDES this purchase, so 1 = first-time buyer ──
@pytest.mark.parametrize("n_props, expected", [(1, 0.75), (2, 0.45), (3, 0.35), (4, 0.35)])
def test_ltv_tiers_match_the_reference_implementation(n_props, expected):
    """Mirrors mortgage_planner.py:61 exactly."""
    assert _calc_ltv(n_props) == expected


@pytest.mark.parametrize("n_props, rate", [(1, 0.00), (2, 0.20), (3, 0.30), (4, 0.30)])
def test_absd_citizen_tiers(n_props, rate):
    price = 1_000_000
    assert _calc_absd(price, CITIZEN, n_props_owned=n_props) == pytest.approx(price * rate)


def test_absd_first_property_citizen_is_zero():
    """A first-time citizen pays no ABSD. Regression: idx=-1 used to charge 30%."""
    assert _calc_absd(1_792_000, CITIZEN, n_props_owned=1) == 0.0


def test_absd_out_of_convention_count_cannot_pick_the_harshest_tier():
    """Fail-safe: a stray 0 must not wrap to [-1] and bill the 30% rate.

    The original failure mode was silent and expensive — it produced a plausible
    number that would have been printed into a customer's IPA letter.
    """
    assert _calc_absd(1_000_000, CITIZEN, n_props_owned=0) < 0.20 * 1_000_000


# ── End-to-end ─────────────────────────────────────────────────────────────
def test_first_time_buyer_gets_75_ltv_and_zero_absd():
    """APP0001's real figures: 38yo citizen, $1.792M private, first property."""
    r = calculate_loan(
        borrowers=CITIZEN,
        property_type="Private",
        n_outstanding_loans=0,
        n_props_owned=1,               # incl. this purchase
        interest_rate_pct=1.31,
        monthly_car_loan=0,
        monthly_other=0,
        target_property_price=1_792_000,
    )
    assert r["ltv_limit_pct"] == 75.0
    assert r["additional_bsd"] == 0.0              # was $537,600 before the fix
    # Income is what binds here, not LTV — the 75% cap would allow $1,344,000.
    # The point is that the wrong 35% cap is no longer the thing that binds.
    assert r["binding_constraint"] == "Income (TDSR)"


def test_second_property_gets_45_ltv_and_20pct_absd():
    r = calculate_loan(
        borrowers=CITIZEN,
        property_type="Private",
        n_outstanding_loans=1,
        n_props_owned=2,
        interest_rate_pct=1.31,
        monthly_car_loan=0,
        monthly_other=0,
        target_property_price=1_000_000,
    )
    assert r["ltv_limit_pct"] == 45.0
    assert r["additional_bsd"] == pytest.approx(200_000.0)
    assert r["eligible_loan"] == 450_000.0         # LTV-bound at 45%


# ── The data itself must now be in the calculator's convention ─────────────
def test_csv_stores_the_count_including_this_purchase():
    """No applicant may carry 0 — the minimum is 1 (the property being bought).

    This is the guard that keeps the data and the calculator from drifting apart
    again. If someone regenerates the CSV with the old convention, this fails.
    """
    df = pd.read_csv(f"{DATA_DIR}/fact_loan_applications.csv")
    assert df["no_sg_properties_owned"].min() >= 1, (
        "no_sg_properties_owned must INCLUDE the property being bought "
        "(1 = first-time buyer). A 0 means the CSV reverted to the "
        "'already owned' convention and every LTV/ABSD figure is wrong."
    )


def test_ipa_and_lo_lanes_agree_on_props_owned():
    """The two lanes used to disagree: IPA passed the raw column, LO coerced 0->1.

    Same case, two different LTVs depending on how the orchestrator routed. With
    one convention in the data, both read the column directly and must match.
    """
    from graph import _build_lo_basis
    from utils.tools import store

    loan = store.get_loan_application("APP0001")
    lo_basis = _build_lo_basis("APP0001")
    assert int(loan["no_sg_properties_owned"]) == lo_basis["n_props_owned"] == 1


# ── Missing borrower fields (2026-08-04) ─────────────────────────────────
# A live reprice turn died with "KeyError: 'nationality' — while calling the LLM
# endpoint". Two defects: the calculator indexed b["nationality"] directly (the
# compare_packages path takes borrower dicts STRAIGHT from the model, which had
# omitted it), and the error was then blamed on the provider. Nothing about it
# involved the network.
def test_absd_tolerates_a_missing_nationality():
    """Must not raise — and must fail SAFE, at the harshest rate, so an omission
    can never under-charge ABSD."""
    from utils.calculator import _calc_absd

    assert _calc_absd(1_000_000, [{"age": 38}], 2) == 600_000.0        # Foreigner tier
    # An explicit nationality is still honoured.
    assert _calc_absd(1_000_000, [{"nationality": "Singapore Citizen"}], 2) == 200_000.0


def test_compare_packages_runs_without_a_nationality():
    """The exact shape of the failing turn: two packages, borrower has no
    nationality key at all."""
    from utils.calculator import compare_packages

    out = compare_packages(
        packages=[{"label": "ours", "interest_rate_pct": 1.3, "rate_type": "floating"},
                  {"label": "DBS", "interest_rate_pct": 1.55, "rate_type": "floating"}],
        borrowers=[{"age": 38, "monthly_income": 9000}],
        property_type="Private", n_outstanding_loans=1, n_props_owned=2,
        monthly_car_loan=0, monthly_other=0, target_property_price=1_200_000)

    assert "error" not in out
    assert len(out["packages"]) == 2


def test_tool_layer_defaults_nationality_but_not_the_numbers():
    """The tool boundary fills nationality (it only picks an ABSD tier) and refuses
    to invent age / income (they would land in a TDSR the RM acts on)."""
    from utils.tools import _normalise_borrowers

    got = _normalise_borrowers([{"age": 38, "monthly_income": 9000}])
    assert got[0]["nationality"] == "Singapore Citizen"
    # An explicit value, and the citizenship alias, both survive untouched.
    assert _normalise_borrowers([{"nationality": "Foreigner"}])[0]["nationality"] == "Foreigner"
    assert _normalise_borrowers(
        [{"citizenship": "Permanent Resident"}])[0]["nationality"] == "Permanent Resident"


def test_missing_age_reports_the_field_by_name():
    """A named ValueError the agent can act on, not a bare KeyError."""
    import pytest

    from utils.calculator import calculate_loan

    with pytest.raises(ValueError, match="age is missing"):
        calculate_loan(borrowers=[{"monthly_income": 9000}], property_type="Private",
                       n_outstanding_loans=1, n_props_owned=1, interest_rate_pct=1.3,
                       monthly_car_loan=0, monthly_other=0,
                       target_property_price=1_200_000)
