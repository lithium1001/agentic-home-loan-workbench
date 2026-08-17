"""Regression tests for the fixed-vs-floating package comparison.

These lock in the behaviour confirmed/fixed on 2026-06-19 so a future change to
calculator.py can't silently break it:

  1. total_interest = total repaid − principal (NOT total repaid), never negative.
  2. The same interest rate fed to a single calculate_loan call and to the
     matching package in compare_packages produces the SAME monthly repayment
     (so the IPA single-case figure and the compare-table floating column agree).
  3. At realistic Jun-2026 rates, floating (~1.3%) is cheaper than fixed (1.40%).

Pure calculator layer — no network, no CSV, no LLM. Run from mas/:
    py -m pytest -q tests/test_compare_packages.py
"""
import pytest

from utils.calculator import calculate_loan, compare_packages

# A representative single-borrower private case used across the tests.
_BORROWERS = [{"age": 35, "monthly_income": 12000, "nationality": "Singapore Citizen"}]
_CASE = dict(
    borrowers=_BORROWERS,
    property_type="Private",
    n_outstanding_loans=0,
    n_props_owned=1,
    monthly_car_loan=0,
    monthly_other=0,
    target_property_price=1_500_000,
)

FLOATING_RATE = 1.28   # case's SORA-pegged rate (would come from get_loan_application)
FIXED_RATE = 1.40      # catalog fixed package (from list_loan_packages)


def _compare():
    return compare_packages(
        packages=[
            {"label": "SORA Floating", "interest_rate_pct": FLOATING_RATE, "rate_type": "Floating"},
            {"label": "2-Year Fixed", "interest_rate_pct": FIXED_RATE, "rate_type": "Fixed"},
        ],
        **_CASE,
    )


def test_total_interest_is_repaid_minus_principal():
    """total_interest must subtract the principal — the 2026-06-19 bugfix.

    Before the fix it returned monthly × months (total repaid), which over-stated
    the cost of credit by the entire principal.
    """
    for p in _compare()["packages"]:
        total_repaid = p["monthly_repayment"] * p["loan_tenure_years"] * 12
        expected_interest = total_repaid - p["eligible_loan"]
        assert p["total_interest"] == pytest.approx(expected_interest, abs=1.0)


def test_total_interest_far_below_principal_at_low_rates():
    """Sanity bound: at ~1.3% over ~30y, interest is a fraction of principal.

    Guards against a regression back to total-repaid (which is ALWAYS > principal).
    """
    for p in _compare()["packages"]:
        assert 0 < p["total_interest"] < p["eligible_loan"]


def test_single_case_rate_matches_compare_floating_column():
    """The floating monthly repayment must equal a standalone calculate_loan at the
    same rate — this is what keeps the IPA single-case figure and the compare-table
    floating column from disagreeing in a demo."""
    single = calculate_loan(
        borrowers=_BORROWERS,
        property_type="Private",
        n_outstanding_loans=0,
        n_props_owned=1,
        interest_rate_pct=FLOATING_RATE,
        monthly_car_loan=0,
        monthly_other=0,
        target_property_price=1_500_000,
    )
    floating = next(p for p in _compare()["packages"] if p["rate_type"] == "Floating")
    assert floating["monthly_repayment"] == single["monthly_repayment"]


def test_floating_cheaper_than_fixed():
    """Realistic Jun-2026 market: floating (~1.3%) beats fixed (1.40%) on monthly cost.

    The baseline is the floating package, so the fixed package's delta_vs_baseline
    for monthly_repayment should be POSITIVE (fixed costs more).
    """
    pkgs = {p["rate_type"]: p for p in _compare()["packages"]}
    assert pkgs["Floating"]["monthly_repayment"] < pkgs["Fixed"]["monthly_repayment"]
    assert pkgs["Fixed"]["delta_vs_baseline"]["monthly_repayment"] > 0


def test_needs_at_least_two_packages():
    """compare_packages guards against a single-package (nothing to compare) call."""
    out = compare_packages(
        packages=[{"label": "Solo", "interest_rate_pct": 1.4}],
        **_CASE,
    )
    assert "error" in out


# ── Malformed model input must not crash the turn (2026-08-04 sweep) ─────
# `packages` and `borrowers` arrive STRAIGHT from the model — compare_packages is
# the only tool taking two nested arrays, which is why it is where these land. The
# @tool wrappers are called by ToolNode, which does NOT catch, so an exception here
# ends the turn and the RM sees a stack-trace fragment instead of an answer.
def _base():
    return dict(borrowers=[{"age": 38, "monthly_income": 9000}],
                property_type="Private", n_outstanding_loans=1, n_props_owned=1,
                monthly_car_loan=0, monthly_other=0, target_property_price=1_200_000)


def test_package_without_a_rate_is_named_not_crashed():
    """Was KeyError: 'interest_rate_pct'. The rate is never defaulted — inventing one
    would price a package the bank never quoted."""
    from utils.calculator import compare_packages

    out = compare_packages(packages=[{"label": "ours"},
                                     {"label": "DBS", "interest_rate_pct": 1.55}],
                           **_base())
    assert "ours" in out["error"]
    assert "interest_rate_pct" in out["error"]


def test_unlabelled_package_is_identified_by_position():
    from utils.calculator import compare_packages

    out = compare_packages(packages=[{}, {"interest_rate_pct": 1.5}], **_base())
    assert "package #1" in out["error"]


def test_a_package_that_is_not_even_a_dict_is_handled():
    from utils.calculator import compare_packages

    out = compare_packages(packages=["oops", {"interest_rate_pct": 1.5}], **_base())
    assert "package #1" in out["error"]


def test_tool_wrapper_returns_bad_input_as_a_result():
    """ToolNode does not catch, so the wrapper must convert a malformed-argument
    exception into {"error": ...} the agent can read and retry from."""
    from utils.tools import TOOLS_BY_NAME

    kw = _base()
    kw["borrowers"] = [{}]                       # no age, no income
    kw["packages"] = [{"interest_rate_pct": 1.3}, {"interest_rate_pct": 1.5}]

    out = TOOLS_BY_NAME["compare_packages"].invoke(kw)
    assert "age is missing" in out["error"]


def test_the_happy_path_is_untouched():
    from utils.tools import TOOLS_BY_NAME

    kw = _base()
    kw["packages"] = [{"label": "ours", "interest_rate_pct": 1.3},
                      {"label": "DBS", "interest_rate_pct": 1.55}]

    out = TOOLS_BY_NAME["compare_packages"].invoke(kw)
    assert "error" not in out
    assert [p["label"] for p in out["packages"]] == ["ours", "DBS"]


def test_wrapper_does_not_swallow_unexpected_errors():
    """Narrow by design: only the malformed-argument family is reported to the model.
    An uncharacterised bug must stay loud rather than look like RM input error."""
    from utils.tools import _tool_errors_as_result

    @_tool_errors_as_result
    def _boom():
        raise RuntimeError("something genuinely broken")

    import pytest
    with pytest.raises(RuntimeError):
        _boom()


def test_tool_logging_survives_a_gbk_console(monkeypatch):
    """The guard prints a diagnostic; on this repo's GBK console a non-CP936 glyph
    raises UnicodeEncodeError, which would escape the very handler meant to rescue
    the turn."""
    from utils.tools import _tool_errors_as_result

    def _explode(*_a, **_k):
        raise UnicodeEncodeError("gbk", "x", 0, 1, "illegal multibyte sequence")

    monkeypatch.setattr("builtins.print", _explode)

    @_tool_errors_as_result
    def _bad():
        raise ValueError("age is missing (None).")

    assert "age is missing" in _bad()["error"]
