"""MAS-compliant loan calculator — pure Python, no LLM, no DB.

All MAS mortgage arithmetic lives here. Agents must call this via the
``calculate_loan`` tool rather than doing their own arithmetic, so numbers
are always consistent.

Helper functions (prefixed ``_calc_`` / ``_pv_`` / ``_monthly_``) are
internal; :func:`calculate_loan` is the single public entry point exposed
as a tool. This module has zero dependencies on the rest of the package,
which makes it the natural unit-test target (``tests/test_calculator.py``).
"""

import math  # noqa: F401  (kept for parity with notebook cell; safe to drop later)
import re


def _to_float(value, field: str = "value") -> float:
    """Coerce an LLM-supplied numeric arg to float, tolerating common junk.

    The tool schema says type=number, but models still emit strings like
    "16T", "16k", "16,000", "$16000", "16 000". Strip currency/space/commas
    and expand a trailing k/K (thousand) or T/t (used colloquially for
    "thousand" in SG, e.g. "16T" = 16,000). Raises a clear ValueError if the
    value still isn't numeric, so the tool returns a useful message instead of
    a raw stack trace.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        raise ValueError(f"{field} is missing (None).")
    s = str(value).strip()
    if not s:
        raise ValueError(f"{field} is empty.")
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    mult = 1.0
    m = re.fullmatch(r"(?i)([0-9]*\.?[0-9]+)([kt])?", s)
    if m:
        num, suffix = m.group(1), m.group(2)
        if suffix:                       # k/K or t/T → thousands
            mult = 1_000.0
        try:
            return float(num) * mult
        except ValueError:
            pass
    raise ValueError(f"{field}={value!r} is not a valid number.")


def _calc_waa(borrowers: list[dict]) -> float:
    """Income-weighted average age across all borrowers."""
    total = sum(b["monthly_income"] for b in borrowers)
    if total == 0:
        return 0.0
    return sum(b["age"] * b["monthly_income"] for b in borrowers) / total

def _calc_tenure(waa: float, prop_type: str) -> int:
    """Max loan tenure: lower of property-type cap and age cap (retirement at 65)."""
    type_cap = 35 if prop_type == "Private" else 30  # Private: 35yr, HDB: 30yr
    age_cap  = int(65 - waa)
    return max(1, min(type_cap, age_cap))

def _calc_ltv(n_props_owned: int) -> float:
    """LTV limit by property count (MAS fixed tiers, INCLUDING this purchase).

    n_props_owned == 1 is a first-time buyer. Callers reading the CSV column
    `no_sg_properties_owned` (which counts properties ALREADY owned) must add 1
    first — see graph._props_owned_incl_purchase.
    """
    if n_props_owned <= 1:
        return 0.75
    elif n_props_owned == 2:
        return 0.45
    else:
        return 0.35

def _pv_annuity(pmt: float, r_monthly: float, n: int) -> float:
    """Present value of an annuity — used to back-calculate max loan from a payment cap."""
    if r_monthly == 0:
        return pmt * n
    return pmt * ((1 + r_monthly)**n - 1) / (r_monthly * (1 + r_monthly)**n)

def _monthly_pmt(loan: float, r_annual_pct: float, tenure_yrs: int) -> float:
    """Standard mortgage payment formula."""
    r = (r_annual_pct / 100) / 12
    n = tenure_yrs * 12
    if r == 0:
        return loan / n
    return loan * r * (1 + r)**n / ((1 + r)**n - 1)

def _calc_bsd(price: float) -> float:
    """Buyer's Stamp Duty — progressive brackets effective Budget 2024."""
    brackets = [
        (180_000,   0.01),
        (180_000,   0.02),
        (640_000,   0.03),
        (500_000,   0.04),
        (1_500_000, 0.05),
    ]
    bsd, rem = 0.0, price
    for limit, rate in brackets:
        if rem <= 0:
            break
        taxable = min(rem, limit)
        bsd    += taxable * rate
        rem    -= taxable
    if rem > 0:
        bsd += rem * 0.06   # 6% on amounts above $3M
    return round(bsd, 2)

# ABSD rates indexed by [n_props_owned - 1], capped at index 2.
_ABSD_TABLE = {
    "Singapore Citizen":  [0.00, 0.20, 0.30],
    "Permanent Resident": [0.05, 0.30, 0.35],
    "Foreigner":          [0.60, 0.60, 0.60],
}

def _calc_absd(price: float, borrowers: list[dict], n_props_owned: int) -> float:
    """ABSD: worst-case nationality among all borrowers drives the rate.

    n_props_owned INCLUDES this purchase, so 1 = first property. The index is
    clamped to [0, 2]: without the lower clamp a count of 0 became idx=-1, and
    Python's negative indexing silently returned the LAST (harshest) tier — a
    first-time citizen was charged 30% ABSD instead of 0%. Clamping fails safe.
    """
    idx  = min(max(n_props_owned - 1, 0), 2)
    # `.get(...)` on a MISSING key, not b["nationality"]: on the compare_packages /
    # reprice path the borrower dicts come straight from the model, and a reply that
    # omits `nationality` used to raise KeyError mid-run — surfaced to the RM as
    # "KeyError: 'nationality' while calling the LLM endpoint", which reads like a
    # provider fault and hides that the input was simply incomplete. The default is
    # the HARSHEST tier (Foreigner, 60%), so an omission can never under-charge ABSD.
    rate = max(_ABSD_TABLE.get(b.get("nationality") or "", [0.60, 0.60, 0.60])[idx]
               for b in borrowers)
    return round(price * rate, 2)


def calculate_loan(
    borrowers:             list[dict],
    property_type:         str,
    n_outstanding_loans:   int,
    n_props_owned:         int,
    interest_rate_pct:     float,
    monthly_car_loan:      float,
    monthly_other:         float,
    cash_cpf_available:    float | None = None,
    target_property_price: float | None = None,
) -> dict:
    """MAS-compliant loan calculator — the single source of truth for all loan numbers.

    Two modes (supply exactly one of the last two args):
      forward mode  — cash_cpf_available given    → compute max property price & loan
      reverse mode  — target_property_price given → compute required cash+CPF & loan

    borrowers: list of {"age": int, "monthly_income": float, "nationality": str}
      nationality ∈ {"Singapore Citizen", "Permanent Resident", "Foreigner"}
      monthly_income = QUALIFYING income = fixed + variable * 0.7 (the 30% MAS
        haircut on variable income is applied UPSTREAM, where fixed/variable are
        still split — see _build_ltv_fusion. Pass the already-haircut figure here.)

    n_outstanding_loans: kept for API compatibility; not used in LTV (see _calc_ltv)
    n_props_owned: properties owned INCLUDING this purchase — drives both LTV and ABSD
    interest_rate_pct: market rate in % p.a.; stress test is always fixed at 4%

    The returned dict includes `calculation_steps`: an ordered, human-readable
    trace (label / formula-with-numbers / value) of every derivation step, so a
    human auditor can follow exactly how each figure was produced.
    """
    # The LLM sometimes emits numeric tool-call args as strings (e.g. "8000",
    # "16T", "$16,000") despite the schema saying type=number. Coerce all numeric
    # inputs up front via _to_float so every downstream calc gets real floats.
    # .get() rather than [] so a borrower dict MISSING a field reaches _to_float,
    # which raises "age is missing (None)" — a message the agent can act on. A bare
    # KeyError here escapes as an unhandled exception and reaches the RM as
    # "KeyError: 'age' while calling the LLM endpoint", blaming the provider for what
    # is really an incomplete tool-call argument.
    borrowers = [
        {**b, "age": _to_float(b.get("age"), "age"),
              "monthly_income": _to_float(b.get("monthly_income"), "monthly_income")}
        for b in borrowers
    ]
    interest_rate_pct = _to_float(interest_rate_pct, "interest_rate_pct")
    monthly_car_loan  = _to_float(monthly_car_loan, "monthly_car_loan")
    monthly_other     = _to_float(monthly_other, "monthly_other")
    if cash_cpf_available is not None:
        cash_cpf_available = _to_float(cash_cpf_available, "cash_cpf_available")
    if target_property_price is not None:
        target_property_price = _to_float(target_property_price, "target_property_price")
    # Exactly one mode input is required. If the caller (the LLM) supplies neither,
    # reverse mode would hit `None * ltv` → TypeError; fail with a clear, actionable
    # message instead so the tool layer can surface it back to the agent to retry.
    if cash_cpf_available is None and target_property_price is None:
        raise ValueError(
            "calculate_loan needs exactly one of `cash_cpf_available` (forward mode) "
            "or `target_property_price` (reverse mode); both were missing. "
            "Pass the property price/valuation as `target_property_price`, or the "
            "available cash+CPF as `cash_cpf_available`.")

    # steps[] accumulates an audit trail; each entry pairs the plugged-in formula
    # with its resulting value so a reviewer never has to reconstruct the math.
    steps: list[dict] = []
    def _step(label, formula, value):
        steps.append({"step": label, "formula": formula, "value": value})

    waa          = _calc_waa(borrowers)
    tenure       = _calc_tenure(waa, property_type)
    ltv          = _calc_ltv(n_props_owned)
    total_income = sum(b["monthly_income"] for b in borrowers)
    stress_rate  = 0.04          # MAS fixed stress test: always 4%, regardless of market rate
    n_months     = tenure * 12
    mr_stress    = stress_rate / 12
    non_mtg_debt = monthly_car_loan + monthly_other

    _step(
        "Income-weighted average age (WAA)",
        " + ".join(f"{b['age']}×{b['monthly_income']:,.0f}" for b in borrowers)
            + f" ÷ {total_income:,.0f}",
        round(waa, 1),
    )
    _step(
        "Loan tenure",
        f"min(type_cap={35 if property_type=='Private' else 30}, age_cap=65−{int(waa)}={int(65-waa)}) "
        f"[{property_type}]",
        tenure,
    )
    _step(
        "LTV limit",
        f"property #{n_props_owned} owned → MAS tier",
        f"{ltv*100:.0f}%",
    )
    _step(
        "Total qualifying monthly income",
        " + ".join(f"{b['monthly_income']:,.0f}" for b in borrowers),
        round(total_income, 2),
    )
    _step(
        "Non-mortgage monthly debt",
        f"car {monthly_car_loan:,.0f} + other {monthly_other:,.0f}",
        round(non_mtg_debt, 2),
    )

    # ── TDSR gate: max allowable monthly mortgage payment ─────────────────────
    # TDSR cap: total debt service ≤ 55% of gross income, non-mortgage debt deducted first.
    tdsr_pmt_cap = total_income * 0.55 - non_mtg_debt
    _step(
        "TDSR monthly payment cap",
        f"{total_income:,.0f} × 55% − {non_mtg_debt:,.0f}",
        round(tdsr_pmt_cap, 2),
    )
    # HDB additionally subject to MSR: mortgage ≤ 30% of gross income.
    if property_type == "HDB":
        msr_cap = total_income * 0.30
        tdsr_pmt_cap = min(tdsr_pmt_cap, msr_cap)
        _step(
            "MSR cap (HDB only) — binding payment cap = min(TDSR, MSR)",
            f"min({round(total_income*0.55 - non_mtg_debt, 2):,.0f}, "
            f"MSR {total_income:,.0f}×30%={msr_cap:,.0f})",
            round(tdsr_pmt_cap, 2),
        )
    # Back-calculate max loan from the payment cap at the stress rate.
    max_loan_tdsr = _pv_annuity(max(0.0, tdsr_pmt_cap), mr_stress, n_months)
    _step(
        "Max loan from income (PV of payment cap @ 4% stress over tenure)",
        f"PV(pmt={max(0.0, tdsr_pmt_cap):,.0f}, r={stress_rate*100:.0f}%/12, n={n_months})",
        round(max_loan_tdsr, 0),
    )

    # ── LTV gate + binding constraint selection ───────────────────────────────
    if cash_cpf_available is not None:
        # Forward mode: given cash+CPF, find max property price.
        max_prop_ltv  = cash_cpf_available / (1 - ltv) if ltv < 1 else 0.0
        max_loan_ltv  = max_prop_ltv * ltv
        _step(
            "Max property price from down payment (LTV)",
            f"cash+CPF {cash_cpf_available:,.0f} ÷ (1 − {ltv:.2f})",
            round(max_prop_ltv, 0),
        )
        _step(
            "Max loan from down payment (LTV)",
            f"{max_prop_ltv:,.0f} × {ltv:.2f}",
            round(max_loan_ltv, 0),
        )

        if max_loan_tdsr <= max_loan_ltv:
            eligible_loan = max_loan_tdsr
            prop_price    = cash_cpf_available + eligible_loan
            binding       = "Income (TDSR)"
        else:
            eligible_loan = max_loan_ltv
            prop_price    = max_prop_ltv
            binding       = "Down payment (LTV)"

        req_cash_cpf = prop_price - eligible_loan
        _step(
            "Binding constraint (forward) — eligible loan = min(income, LTV)",
            f"min(TDSR {max_loan_tdsr:,.0f}, LTV {max_loan_ltv:,.0f}) → {binding}",
            round(max(0.0, eligible_loan), 0),
        )

    else:
        # Reverse mode: given target property price, find required cash+CPF.
        prop_price   = target_property_price
        max_loan_ltv = prop_price * ltv
        _step(
            "Max loan from LTV on target price",
            f"target {prop_price:,.0f} × {ltv:.2f}",
            round(max_loan_ltv, 0),
        )

        if max_loan_ltv <= max_loan_tdsr:
            eligible_loan = max_loan_ltv
            binding       = "Down payment (LTV)"
        else:
            eligible_loan = max_loan_tdsr
            binding       = "Income (TDSR)"

        req_cash_cpf = prop_price - eligible_loan
        _step(
            "Binding constraint (reverse) — eligible loan = min(LTV, income)",
            f"min(LTV {max_loan_ltv:,.0f}, TDSR {max_loan_tdsr:,.0f}) → {binding}",
            round(max(0.0, eligible_loan), 0),
        )

    eligible_loan = max(0.0, eligible_loan)
    prop_price    = max(0.0, prop_price)
    _step(
        "Required cash + CPF (property price − eligible loan)",
        f"{prop_price:,.0f} − {eligible_loan:,.0f}",
        round(max(0.0, req_cash_cpf), 0),
    )

    # ── Repayment at market rate (what the borrower actually pays) ────────────
    pmt_actual  = _monthly_pmt(eligible_loan, interest_rate_pct, tenure)
    pmt_stress  = _monthly_pmt(eligible_loan, stress_rate * 100, tenure)
    tdsr_actual = (non_mtg_debt + pmt_actual) / total_income if total_income > 0 else None
    _step(
        "Monthly repayment @ market rate",
        f"amortise(loan={eligible_loan:,.0f}, rate={interest_rate_pct}%, tenure={tenure}y)",
        round(pmt_actual, 0),
    )
    _step(
        "Monthly repayment @ 4% stress",
        f"amortise(loan={eligible_loan:,.0f}, rate={stress_rate*100:.0f}%, tenure={tenure}y)",
        round(pmt_stress, 0),
    )
    if tdsr_actual is not None:
        _step(
            "Actual TDSR (all debt service ÷ income)",
            f"({non_mtg_debt:,.0f} + {pmt_actual:,.0f}) ÷ {total_income:,.0f}",
            f"{tdsr_actual*100:.2f}%",
        )

    bsd  = _calc_bsd(prop_price)
    absd = _calc_absd(prop_price, borrowers, n_props_owned)
    _absd_idx = min(max(n_props_owned - 1, 0), 2)   # same clamp as _calc_absd
    # Same missing-key tolerance as _calc_absd above — this line only re-derives the
    # rate for the audit trail, so it must not be the one that raises.
    _absd_rate = max(_ABSD_TABLE.get(b.get("nationality") or "", [0.60, 0.60, 0.60])[_absd_idx]
                     for b in borrowers)
    _step(
        "Buyer's Stamp Duty (progressive brackets)",
        f"BSD on {prop_price:,.0f}",
        bsd,
    )
    _step(
        "Additional Buyer's Stamp Duty",
        f"{prop_price:,.0f} × {_absd_rate*100:.0f}% (worst-case nationality, property #{n_props_owned})",
        absd,
    )

    return {
        # inputs echoed back for transparency
        "waa_years":                round(waa, 1),
        "loan_tenure_years":        tenure,
        "ltv_limit_pct":            round(ltv * 100, 1),
        # Echoed so consumers (e.g. the letter's terms table) never have to source
        # the rate from anywhere but the calculation it actually priced.
        "interest_rate_pct":        interest_rate_pct,
        "stress_rate_pct":          stress_rate * 100,
        "total_monthly_income":     round(total_income, 2),
        "non_mortgage_debt_pm":     round(non_mtg_debt, 2),
        # results
        "eligible_loan":            round(eligible_loan, 0),
        "property_price":           round(prop_price, 0),
        "required_cash_cpf":        round(req_cash_cpf, 0),
        "binding_constraint":       binding,
        "monthly_repayment":        round(pmt_actual, 0),
        "monthly_repayment_stress": round(pmt_stress, 0),
        "tdsr_pct":                 round(tdsr_actual * 100, 2) if tdsr_actual else None,
        "buyer_stamp_duty":         bsd,
        "additional_bsd":           absd,
        "total_stamp_duty":         round(bsd + absd, 2),
        # full derivation trace for human audit
        "calculation_steps":        steps,
    }


def _total_interest(
    monthly_repayment: float | None,
    tenure_years: int,
    eligible_loan: float | None,
) -> float | None:
    """Total interest over the full tenure = total repaid − principal.

    Level monthly repayment × number of months gives the total amount repaid;
    subtracting the principal (eligible_loan) leaves the interest — the standard
    "cost of credit" figure RMs quote when comparing packages (fixed vs floating,
    own vs competitor). Floating is indicative only — its rate can reprice — so
    for a floating package this is a snapshot at the current rate, held flat for
    the whole tenure. Never negative.
    """
    if monthly_repayment is None or eligible_loan is None:
        return None
    total_repaid = monthly_repayment * tenure_years * 12
    return round(max(0.0, total_repaid - eligible_loan), 0)


def amortization_schedule(
    eligible_loan:     float | None,
    interest_rate_pct: float | None,
    tenure_years:      int | None,
    max_rows:          int | None = None,
) -> list[dict]:
    """Month-by-month repayment schedule for a level-instalment (annuity) loan.

    Returns one row per month: opening balance, the (constant) instalment, how
    much of it is interest, how much repays principal, and the closing balance.
    This is the standard amortisation identity — the instalment never changes,
    but as the balance falls the interest share shrinks and the principal share
    grows — so it re-derives nothing: the instalment comes from the same
    :func:`_monthly_pmt` the rest of this module prices with, which is what keeps
    the table's figures identical to the ``monthly_repayment`` shown on the
    result card.

    Interest each month accrues on the CURRENT outstanding balance at the market
    rate (rate/100/12), never the 4% stress rate: this table shows what the
    borrower actually pays, whereas the stress rate is only a qualifying gate.

    The final row is forced to close at exactly zero and its principal is taken
    as the whole remaining balance, so accumulated floating-point cents can never
    leave a phantom balance behind on a 360-row table.

    Balances are carried at full precision and rounded only for display, which is
    the convention the business spec was written against. Displayed columns can
    therefore be a cent apart from each other on a given row (see the note in the
    body); the underlying schedule is exact.

    ``max_rows`` truncates the returned list (the UI shows the first year and
    expands on demand); it never changes the arithmetic of the rows returned.
    Returns [] when the loan, rate or tenure is missing/zero rather than raising,
    since callers render this as an optional panel.
    """
    if not eligible_loan or eligible_loan <= 0 or not tenure_years or tenure_years <= 0:
        return []
    if interest_rate_pct is None:
        return []

    n_months  = int(tenure_years) * 12
    r_monthly = (_to_float(interest_rate_pct, "interest_rate_pct") / 100) / 12
    balance   = _to_float(eligible_loan, "eligible_loan")
    instalment = _monthly_pmt(balance, interest_rate_pct, int(tenure_years))

    # Balances are carried at FULL precision and rounded only for display. The
    # alternative (carrying rounded cents, as a bank statement does) makes every
    # row add up exactly as printed, but it compounds each month's rounding into
    # the balance and drifts a cent or two away from the schedule the business
    # spec was written against — so this follows the spec. The visible cost is
    # that a row's printed interest + principal can sit a cent off the printed
    # instalment; that is display rounding, not a discrepancy in what is owed.
    rows: list[dict] = []
    limit = n_months if max_rows is None else min(n_months, max(0, int(max_rows)))
    for month in range(1, limit + 1):
        opening  = balance
        interest = opening * r_monthly
        if month == n_months:
            # Close the loan exactly: the last instalment settles whatever is
            # left, absorbing the rounding drift of every prior month.
            principal = opening
            payment   = opening + interest
            closing   = 0.0
        else:
            principal = instalment - interest
            payment   = instalment
            closing   = opening - principal
        rows.append({
            "month":             month,
            "beginning_balance": round(opening, 2),
            "instalment":        round(payment, 2),
            "interest_paid":     round(interest, 2),
            "principal_paid":    round(principal, 2),
            "ending_balance":    round(max(0.0, closing), 2),
        })
        balance = closing

    return rows


# ── MAS Notice 645: eligible financial assets as an income stream ────────────
# A borrower's liquid assets can be converted into a monthly "income stream" for
# TDSR purposes: apply a haircut, then amortise the recognised value over 48
# months (MAS Notice 645, eligible financial assets / income streams).
#   https://www.mas.gov.sg/regulation/notices/notice-645
#
# 48 is a REGULATORY CONSTANT, not a function of how long the assets are pledged:
# a 4-year pledge and a 2-year pledge are both amortised over 48 months. The
# pledge period only decides which haircut band applies (a pledge of at least 4
# years to the lending bank is what earns the concessionary bands below).
#
# The haircut is a DEDUCTION: 70% means 30% of the value is recognised. The bands
# below are ordered by how stable the asset's value is, and reproduce the client's
# reference table exactly (verified against all 12 of its figures).
_MAS645_AMORTISATION_MONTHS = 48

_MAS645_ASSET_CLASSES: list[dict] = [
    # key                     label                          pledged  unpledged
    {"key": "fixed_deposit",  "label": "Fixed Deposit",              "pledged_haircut": 0.00},
    {"key": "ssb_sgs",        "label": "Singapore Savings Bonds / SGS", "pledged_haircut": 0.00},
    {"key": "foreign_currency", "label": "Foreign Currency Deposit", "pledged_haircut": 0.30},
    {"key": "gold",           "label": "Gold Certificate",           "pledged_haircut": 0.30},
    {"key": "shares",         "label": "Shares",                     "pledged_haircut": 0.70},
    {"key": "unit_trust",     "label": "Unit Trust",                 "pledged_haircut": 0.70},
]
# Anything not pledged to the lending bank for at least 4 years takes the full
# 70% haircut regardless of asset class.
_MAS645_UNPLEDGED_HAIRCUT = 0.70

MAS645_SOURCE_URL   = "https://www.mas.gov.sg/regulation/notices/notice-645"
MAS645_SOURCE_LABEL = "MAS Notice 645 — eligible financial assets as income streams"


def amortized_monthly_income(assets: dict | None) -> dict:
    """Convert eligible financial assets into a recognised monthly income stream.

    ``assets`` maps an asset-class key from :data:`_MAS645_ASSET_CLASSES` to either
    a plain amount (assumed pledged for at least 4 years) or a dict
    ``{"amount": float, "pledged": bool}``. Unknown keys are ignored, and missing
    or zero amounts contribute nothing, so the default (no assets) yields exactly
    zero and leaves every downstream figure unchanged.

    Per asset class::

        recognised     = amount x (1 - haircut)
        monthly income = recognised / 48

    Returns ``{"rows": [...], "total_recognised": float, "monthly_income": float,
    "source": {...}}``. The rows carry the haircut actually applied and the
    resulting monthly figure so the UI can show the customer the full derivation
    rather than a single unexplained number.

    IMPORTANT: the result is already net of the MAS 645 haircut and must NOT have
    the 30% variable-income haircut applied on top of it. The two are different
    rules for different things; stacking them double-counts the discount.
    """
    rows: list[dict] = []
    if not assets:
        return {"rows": rows, "total_recognised": 0.0, "monthly_income": 0.0,
                "source": {"label": MAS645_SOURCE_LABEL, "url": MAS645_SOURCE_URL}}

    for spec in _MAS645_ASSET_CLASSES:
        raw = assets.get(spec["key"])
        if raw is None:
            continue
        if isinstance(raw, dict):
            amount  = _to_float(raw.get("amount") or 0, spec["key"])
            pledged = bool(raw.get("pledged", True))
        else:
            amount, pledged = _to_float(raw or 0, spec["key"]), True
        if amount <= 0:
            continue

        haircut    = spec["pledged_haircut"] if pledged else _MAS645_UNPLEDGED_HAIRCUT
        recognised = amount * (1 - haircut)
        rows.append({
            "key":            spec["key"],
            "label":          spec["label"],
            "amount":         round(amount, 2),
            "pledged":        pledged,
            "haircut_pct":    round(haircut * 100, 1),
            "recognised":     round(recognised, 2),
            "monthly_income": round(recognised / _MAS645_AMORTISATION_MONTHS, 2),
        })

    total_recognised = sum(r["recognised"] for r in rows)
    return {
        "rows":             rows,
        "total_recognised": round(total_recognised, 2),
        # Derived from the unrounded total so it cannot drift from the rows by a cent.
        "monthly_income":   round(total_recognised / _MAS645_AMORTISATION_MONTHS, 2),
        "amortisation_months": _MAS645_AMORTISATION_MONTHS,
        "source": {"label": MAS645_SOURCE_LABEL, "url": MAS645_SOURCE_URL},
    }


# ── BUC (Building Under Construction) progressive payment scheme ─────────────
# The stage percentages below are STATUTORY, not a bank product: they are the
# Payment Schedule in clause 5 of the prescribed form of Sale and Purchase
# Agreement, Schedule to the Housing Developers Rules (Housing Developers
# (Control and Licensing) Act 1965).
#   https://sso.agc.gov.sg/SL/HDCLA1965-R1?ProvIds=Sc1-
# Verbatim from that Payment Schedule:
#   item 1     20% on signing the S&P (INCLUSIVE of the booking fee)
#   item 2(a)  10% foundation work (inclusive of pile caps)
#   item 2(b)  10% reinforced concrete framework
#   item 2(c)   5% partition walls
#   item 2(d)   5% roofing
#   item 2(e)   5% door/window frames, electrical wiring, internal plastering, plumbing
#   item 2(f)   5% car park, roads and drains
#   item 3     25% on TOP or CSC (whichever the purchaser receives first)
#   item 4/5   15% final payment on completion / CSC
# Statutory item 1 is "20% ... (inclusive of the Booking Fee)". The booking fee is
# listed separately below because it is paid earlier (on exercising the option),
# and the S&P row carries the full 20% — matching how the business spec presents
# it. The two rows therefore overlap by the 5% booking fee by design: the booking
# fee row is an EARLIER instalment of the same statutory 20%, not an extra
# payment. `pct_of_price` for the S&P row is reported net of it (15%) so the
# column still sums to 100%, while the cash actually required up front is 25%.
# The statutory final 15% is disbursed through a stakeholder in several tranches
# (2% + 13% split 8%/5%, with variants depending on whether CSC precedes
# completion). That is a conveyancing detail with no effect on what the customer
# pays or when, so it is modelled here as a single 15% payment.
_BUC_STAGES: list[dict] = [
    # `pct` is what this row adds to the 100% column; `shown_pct` (when present) is
    # the statutory figure to DISPLAY. They differ only for the S&P row, which the
    # statute states as 20% inclusive of the 5% booking fee already listed above.
    {"stage": "Exercising the Option (Booking Fee)",       "pct": 5.0,  "timeframe": "On exercising OTP"},
    {"stage": "Sign S&P Agreement (Balance Downpayment)",  "pct": 15.0, "shown_pct": 20.0,
     "timeframe": "Within 8 weeks of the Option"},
    {"stage": "Foundation of Work",                        "pct": 10.0, "timeframe": "~6-9 months from launch"},
    {"stage": "Reinforced Concrete Framework",             "pct": 10.0, "timeframe": "~6-9 months later"},
    {"stage": "Partition Walls of Unit",                   "pct": 5.0,  "timeframe": "~3-6 months later"},
    {"stage": "Roofing / Ceiling of Unit",                 "pct": 5.0,  "timeframe": "~3-6 months later"},
    {"stage": "Doors, Windows, Wiring, Plumbing",          "pct": 5.0,  "timeframe": "~3-6 months later"},
    {"stage": "Car Park, Roads and Drains",                "pct": 5.0,  "timeframe": "~3-6 months later"},
    {"stage": "Temporary Occupation Permit (TOP)",         "pct": 25.0, "timeframe": "On TOP or CSC"},
    {"stage": "Certificate of Statutory Completion (CSC)", "pct": 15.0, "timeframe": "12-18 months after TOP"},
]

BUC_SOURCE_URL   = "https://sso.agc.gov.sg/SL/HDCLA1965-R1?ProvIds=Sc1-"
BUC_SOURCE_LABEL = ("Housing Developers Rules — prescribed Sale & Purchase Agreement, "
                    "clause 5 Payment Schedule")


def buc_progressive_schedule(
    purchase_price:    float | None,
    interest_rate_pct: float | None,
    tenure_years:      int | None,
    ltv_pct:           float = 75.0,
    legal_fees:        float = 3_000.0,
    valuation_fees:    float = 500.0,
    eligible_loan:     float | None = None,
) -> dict:
    """Progressive payment schedule for a property Building Under Construction.

    For a completed property the bank disburses the whole loan at once. For a BUC
    purchase the developer is paid in statutory stages as the building goes up, so
    the loan is drawn down piecemeal and the instalment RISES with each drawdown.
    Showing only a final instalment would misrepresent the first few years, which
    is the point of this table.

    Funding follows the order money is actually used: the buyer's own cash/CPF is
    spent first, and the loan starts only once that downpayment is exhausted. The
    stage where funding switches therefore falls out of the numbers rather than
    being hard-coded, and a stage straddling the switch is split and reported as
    part cash/CPF and part loan.

    ``eligible_loan`` is how much this borrower can actually borrow. It is usually
    LESS than ``ltv × price`` because income (TDSR/MSR) binds before the LTV cap
    does, and when it is passed the shortfall must be found in cash: the cash/CPF
    budget becomes ``price - eligible_loan``, so the loan tops out at exactly the
    figure on the result card and the two never disagree. Omit it and the schedule
    assumes a purely LTV-limited purchase (``(1 - ltv) × price`` of cash).

    ``monthly_repayment`` on each row is the full amortising instalment on the
    loan drawn SO FAR, over the full tenure (:func:`_monthly_pmt`, the same
    primitive that prices everything else here). Note that Singapore banks
    commonly service BUC loans on an interest-only basis until TOP; this follows
    the fuller convention of the business spec, so treat these as the post-TOP
    steady-state instalment for each drawdown level.

    Stamp duty and the legal/valuation fees are cash items that do not draw on the
    loan; BSD is computed with :func:`_calc_bsd` rather than a flat percentage, so
    it always agrees with the duty shown elsewhere in the app. The fee amounts are
    fixed sums, not percentages of price.

    Returns ``{"rows": [...], "totals": {...}, "source": {...}}``; ``{}``-ish empty
    rows when inputs are missing, since callers render this as an optional panel.
    """
    if not purchase_price or purchase_price <= 0 or not tenure_years or tenure_years <= 0:
        return {"rows": [], "totals": {}, "source": {"label": BUC_SOURCE_LABEL, "url": BUC_SOURCE_URL}}
    if interest_rate_pct is None:
        return {"rows": [], "totals": {}, "source": {"label": BUC_SOURCE_LABEL, "url": BUC_SOURCE_URL}}

    price = _to_float(purchase_price, "purchase_price")
    ltv   = _to_float(ltv_pct, "ltv_pct") / 100.0
    tenure = int(tenure_years)
    # Cap the loan at what the borrower actually qualifies for, when known; the
    # rest of the price has to come out of cash/CPF.
    max_loan = price * ltv
    if eligible_loan is not None:
        max_loan = min(max_loan, max(0.0, _to_float(eligible_loan, "eligible_loan")))
    cash_budget = price - max_loan       # the buyer's downpayment on the PRICE

    rows: list[dict] = []
    cash_left = cash_budget
    drawn     = 0.0

    def _push(stage, timeframe, pct, amount, cash, loan, note=""):
        nonlocal drawn
        drawn += loan
        # Compare against a cent, not zero: exhausting the cash budget can leave a
        # sub-cent float residue that would otherwise print "0% Cash/CPF 100% Loan".
        if loan > 0.005 and cash > 0.005:
            funding = f"{cash / amount * 100:.0f}% Cash/CPF {loan / amount * 100:.0f}% Loan"
        elif loan > 0.005:
            funding = "Loan"
        else:
            funding = "Cash/CPF"
        rows.append({
            "stage":             stage,
            "timeframe":         timeframe,
            "pct_of_price":      round(pct, 2) if pct is not None else None,
            "amount_payable":    round(amount, 2),
            "cash_cpf":          round(cash, 2),
            "loan":              round(loan, 2),
            "funding":           funding,
            "cumulative_loan":   round(drawn, 2),
            # The instalment the borrower faces once this much of the loan is out.
            "monthly_repayment": round(_monthly_pmt(drawn, interest_rate_pct, tenure), 2) if drawn > 0 else 0.0,
            "note":              note,
        })

    for i, st in enumerate(_BUC_STAGES):
        # The S&P row is due as the statutory 20% inclusive of the booking fee, so
        # the CASH it consumes is the displayed 20% less the 5% already paid — but
        # the amount payable at that moment is the 15% balance. Both come out of
        # `pct`; `shown_pct` only changes the percentage label.
        amount = price * st["pct"] / 100.0
        cash   = min(amount, cash_left)
        loan   = amount - cash
        cash_left -= cash
        _push(st["stage"], st["timeframe"], st.get("shown_pct", st["pct"]), amount, cash, loan)

        # Cash items are interleaved where they fall due, so the customer sees the
        # true order of outgoings. They never draw on the loan.
        if i == 1:      # BSD is due within 14 days of signing the S&P
            _push("Buyer's Stamp Duty (BSD)", "Within 14 days of signing S&P",
                  None, _calc_bsd(price), _calc_bsd(price), 0.0)
            _push("Legal fees", "Within 8 weeks of exercising the option",
                  None, legal_fees, legal_fees, 0.0)
            _push("Valuation fees", "Within 8 weeks of exercising the option",
                  None, valuation_fees, valuation_fees, 0.0)

    total_price_pct = sum(s["pct"] for s in _BUC_STAGES)   # real shares, sums to 100
    totals = {
        "purchase_price":     round(price, 2),
        "pct_of_price_total": round(total_price_pct, 2),
        "cash_cpf_total":     round(sum(r["cash_cpf"] for r in rows), 2),
        "loan_total":         round(sum(r["loan"] for r in rows), 2),
        "amount_total":       round(sum(r["amount_payable"] for r in rows), 2),
        "ltv_pct":            round(ltv * 100, 1),
        # What actually caps the borrowing: the LTV rule, or this borrower's income.
        "loan_capped_by":     ("Loan-to-value limit" if eligible_loan is None
                               or max_loan >= price * ltv - 0.5 else "Income (TDSR/MSR)"),
        "final_monthly_repayment": rows[-1]["monthly_repayment"] if rows else 0.0,
    }
    return {
        "rows":   rows,
        "totals": totals,
        "source": {"label": BUC_SOURCE_LABEL, "url": BUC_SOURCE_URL},
    }


def _interest_over(
    balance: float, rate_pct: float, n_remaining: int, months: int,
) -> tuple[float, float]:
    """Amortise ``balance`` at ``rate_pct`` over ``n_remaining`` months and report
    the interest charged during the first ``months`` of that, plus the balance
    left at the end of them.

    The instalment is re-derived from the balance and the remaining term, which
    is what a bank does when a loan is repriced: the term does not restart, so a
    conversion after k months amortises over ``n - k`` months, not over ``n``.
    """
    # Same annuity formula as _monthly_pmt, but in MONTHS rather than whole years:
    # a conversion partway through leaves a term like 357 months, which cannot be
    # expressed as an integer number of years.
    r = rate_pct / 100 / 12
    if not n_remaining:
        return 0.0, balance
    if r == 0:
        pmt = balance / n_remaining
    else:
        pmt = balance * r * (1 + r) ** n_remaining / ((1 + r) ** n_remaining - 1)

    total_interest = 0.0
    for _ in range(max(0, months)):
        interest = balance * r
        total_interest += interest
        balance -= (pmt - interest)
    return total_interest, balance


def interest_savings(
    outstanding_loan:      float | None,
    current_rate_pct:      float | None,
    remaining_months:      int | None,
    convert_after_months:  int = 0,
    rate_a_pct:            float | None = None,
    rate_b_pct:            float | None = None,
    horizon_months:        int | None = None,
) -> dict:
    """Interest saved by converting an EXISTING loan to a cheaper package.

    This is the reprice/retention question, and it is a different question from
    :func:`compare_packages`. That one prices a *new purchase* under several rates
    and answers "what would this borrower qualify for"; here the loan already
    exists, the inputs are an outstanding balance and a remaining term, and the
    thing that matters is *when* the switch happens. Converting early saves more
    because the balance interest is charged on is highest early on.

    Two scenarios are compared against staying put:
      Scenario 1 — convert NOW to ``rate_a_pct``.
      Scenario 2 — stay on the current rate for ``convert_after_months``, then
                   convert to ``rate_b_pct`` (a lock-in expiry, or a package that
                   only becomes available later).

    Savings are measured over ``horizon_months`` (default: ``convert_after_months``
    + 24, i.e. the switch plus the two years that usually follow a lock-in), and
    are the plain difference in interest charged versus the baseline over that
    window. Scenario 1 is additionally split into the pre-switch months and the
    months after, because those two parts have quite different explanations: the
    first is an immediate rate replacement, the second is the rate gap continuing
    to earn on a falling balance.

    Interest only, deliberately: principal repaid is money the borrower keeps
    either way, so counting it would overstate the benefit of converting.

    Returns ``{}``-ish empty scenarios when inputs are missing, since callers
    render this as a panel.
    """
    empty = {"scenarios": [], "baseline": {}, "inputs": {}}
    if not outstanding_loan or outstanding_loan <= 0:
        return empty
    if current_rate_pct is None or not remaining_months or remaining_months <= 0:
        return empty

    loan   = _to_float(outstanding_loan, "outstanding_loan")
    base   = _to_float(current_rate_pct, "current_rate_pct")
    n      = int(remaining_months)
    k      = max(0, min(int(convert_after_months or 0), n))
    horizon = int(horizon_months) if horizon_months else min(n, k + 24)
    horizon = max(1, min(horizon, n))

    # Baseline: never convert.
    base_head, base_bal_k = _interest_over(loan, base, n, k)
    base_total, _         = _interest_over(loan, base, n, horizon)

    scenarios: list[dict] = []

    if rate_a_pct is not None:
        a = _to_float(rate_a_pct, "rate_a_pct")
        a_head, _  = _interest_over(loan, a, n, k)
        a_total, _ = _interest_over(loan, a, n, horizon)
        scenarios.append({
            "id":            1,
            "label":         f"Convert now to {a:.2f}%",
            "rate_pct":      a,
            "convert_after_months": 0,
            "savings":       round(base_total - a_total, 2),
            # The two phases the summary explains separately.
            "savings_first_phase":  round(base_head - a_head, 2),
            "savings_second_phase": round((base_total - a_total) - (base_head - a_head), 2),
            "first_phase_months":   k,
            "second_phase_months":  horizon - k,
            # Illustrative single-month figure quoted in the summary; the totals
            # above come from the amortisation simulation, not from this.
            "month_1_interest_delta": round(loan * (base - a) / 100 / 12, 2),
        })

    if rate_b_pct is not None:
        b = _to_float(rate_b_pct, "rate_b_pct")
        # Stay on the current rate for k months, then convert the remaining
        # balance over the remaining term.
        b_head, b_bal = base_head, base_bal_k
        b_tail, _     = _interest_over(b_bal, b, n - k, horizon - k)
        scenarios.append({
            "id":            2,
            "label":         f"Convert in {k} month{'s' if k != 1 else ''} to {b:.2f}%",
            "rate_pct":      b,
            "convert_after_months": k,
            "savings":       round(base_total - (b_head + b_tail), 2),
            "savings_first_phase":  round(base_head - b_head, 2),      # zero by construction
            "savings_second_phase": round(base_total - (b_head + b_tail) - (base_head - b_head), 2),
            "first_phase_months":   k,
            "second_phase_months":  horizon - k,
            "month_1_interest_delta": 0.0,   # nothing changes before the switch
        })

    return {
        "scenarios": scenarios,
        "baseline": {
            "rate_pct":        base,
            "interest_over_horizon": round(base_total, 2),
        },
        "inputs": {
            "outstanding_loan":     round(loan, 2),
            "current_rate_pct":     base,
            "remaining_months":     n,
            "convert_after_months": k,
            "horizon_months":       horizon,
        },
    }


def compare_packages(
    packages:              list[dict],
    borrowers:             list[dict],
    property_type:         str,
    n_outstanding_loans:   int,
    n_props_owned:         int,
    monthly_car_loan:      float,
    monthly_other:         float,
    cash_cpf_available:    float | None = None,
    target_property_price: float | None = None,
) -> dict:
    """Price the SAME borrower case under several loan packages and return a
    side-by-side comparison with deltas vs the first (baseline) package.

    The case fields are identical to :func:`calculate_loan`'s inputs EXCEPT
    ``interest_rate_pct``, which is supplied per-package. Each package is a dict
    ``{"label": str, "interest_rate_pct": float, "rate_type"?: str}``. The first
    package is the baseline; every other package's deltas are measured against it.

    Powers two features that share this exact core (write once, reuse):
      • fixed-vs-floating compare — two of OUR own packages.
      • reprice / retention       — our package vs a competitor's quoted rate.

    Returns ``{"packages": [...], "baseline_label": str}`` where each entry holds
    the package's key figures (monthly repayment, eligible loan, total interest,
    TDSR, required cash+CPF) plus ``delta_vs_baseline`` (None for the baseline).
    The full ``calculation_steps`` audit trace for each package is kept under
    ``calc`` so the agent can still surface a per-package derivation if asked.
    """
    if len(packages) < 2:
        return {"error": "compare_packages needs at least two packages to compare."}

    results = []
    for i, pkg in enumerate(packages):
        # `packages` arrives STRAIGHT from the model, so a package missing its rate is
        # ordinary bad input, not a broken invariant. Indexing it directly raised
        # KeyError mid-loop and escaped the @tool wrapper (which does not catch), so
        # the RM saw a crash — the same shape as the 2026-08-04 KeyError:'nationality'.
        # Name the offending package so the agent knows what to resend. NOT defaulted:
        # inventing a rate would price a package the bank never quoted.
        if not isinstance(pkg, dict) or pkg.get("interest_rate_pct") is None:
            label = (pkg.get("label") if isinstance(pkg, dict) else None) or f"package #{i + 1}"
            return {"error": f"{label} has no interest_rate_pct — every package being "
                             f"compared needs its own rate."}
        rate = pkg["interest_rate_pct"]
        calc = calculate_loan(
            borrowers             = borrowers,
            property_type         = property_type,
            n_outstanding_loans   = n_outstanding_loans,
            n_props_owned         = n_props_owned,
            interest_rate_pct     = rate,
            monthly_car_loan      = monthly_car_loan,
            monthly_other         = monthly_other,
            cash_cpf_available    = cash_cpf_available,
            target_property_price = target_property_price,
        )
        results.append({
            "label":             pkg.get("label") or f"{rate}%",
            "rate_type":         pkg.get("rate_type"),
            "interest_rate_pct": rate,
            "monthly_repayment": calc["monthly_repayment"],
            "eligible_loan":     calc["eligible_loan"],
            "property_price":    calc["property_price"],
            "required_cash_cpf": calc["required_cash_cpf"],
            "tdsr_pct":          calc["tdsr_pct"],
            "total_interest":    _total_interest(calc["monthly_repayment"], calc["loan_tenure_years"], calc["eligible_loan"]),
            "loan_tenure_years": calc["loan_tenure_years"],
            "calc":              calc,
        })

    # Deltas vs the first package: positive = this package costs MORE than baseline.
    base = results[0]
    for r in results:
        if r is base:
            r["delta_vs_baseline"] = None
            continue
        def _d(key):
            a, b = r[key], base[key]
            return round(a - b, 2) if (a is not None and b is not None) else None
        r["delta_vs_baseline"] = {
            "monthly_repayment": _d("monthly_repayment"),
            "total_interest":    _d("total_interest"),
            "tdsr_pct":          _d("tdsr_pct"),
            "eligible_loan":     _d("eligible_loan"),
        }

    return {"baseline_label": base["label"], "packages": results}
