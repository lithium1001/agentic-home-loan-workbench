"""Case-data service for the guided-workflow UI.

Derives the read-only "information system" panels (case list, deal-progress dots,
loan summary card, per-stage documents checklist, action chips) from the EXISTING
DataStore + the EXISTING calculate_loan tool. No new data sources, no DB — this is
a presentation layer over csv_tables that the FastAPI endpoints serialise to JSON.

Stage model (matches the backend graph's stages):
    IPA      — In-Principle Approval (borrowing capacity, eligibility)
    LO       — Letter of Offer (package selection, offer letter)        [mockup "loo"]
    REPRICE  — Reprice / Retention (existing loan, competitor benchmark)
"""

from __future__ import annotations

from server import case_overrides
from server.case_progress import assessed_stages, completed_stages
from utils.calculator import calculate_loan
from utils.tools import store

STAGE_KEYS = ["IPA", "LO", "REPRICE"]

STAGE_META = {
    "IPA": {
        "num": "01",
        "label": "IPA Eligibility",
        "subtitle": "Verify borrowing capacity and submit the in-principle application.",
    },
    "LO": {
        "num": "02",
        "label": "Letter of Offer",
        "subtitle": "Pick a loan package, confirm pricing, and generate the offer letter.",
    },
    "REPRICE": {
        "num": "03",
        "label": "Reprice / Retention",
        "subtitle": "Review the current rate and negotiate a competitive renewal package.",
    },
}

# Static per-stage shortcuts the RM can click instead of typing a prompt. These
# mirror the example prompts from the original Gradio UI so a new RM can drive the
# full system (eligibility → full IPA → LO compare → draft → reprice) by clicking.
# The current applicant_id is injected server-side, so chips stay applicant-free.
ACTION_CHIPS = {
    "IPA": [
        "Check borrowing capacity & LTV eligibility",
        "What documents are still missing?",
        "Run a full IPA eligibility assessment end-to-end",
        "Draft the IPA letter",
    ],
    "LO": [
        "Compare a 2-year fixed vs SORA floating package",
        "What are today's rates?",
        "Draft the Letter of Offer",
        "Make the tone more formal",
    ],
    "REPRICE": [
        "A competitor offered 1.55% — compare against our package",
        "Give me a retention script for this customer",
        "Actually they quoted 1.6%, redo the comparison",
    ],
    # COMPARE is the standalone Package Comparison / interest-savings tool. It is a
    # client-only track with no server-side stage record, so it never reaches
    # _next_action_chips' status logic and nothing here is ever promoted to primary
    # — these are plain shortcuts, deliberately in the order an RM works the panel:
    # read the result, then pressure-test it, then turn it into customer-facing words.
    #
    # Each chip RESTATES the loan terms instead of saying "my two scenarios". The
    # assistant cannot see the comparison panel's inputs (the panel posts to the
    # calculator API, not into the chat's message history), so a chip that referred
    # to them got "I don't have any scenarios in our current conversation to
    # compare" — verified live. Self-contained numbers keep every chip answerable.
    #
    # These figures mirror CMP_FIELDS' defaults in static/app.js, and are the
    # FALLBACK only: once the panel is on screen, app.js's compareChips() overrides
    # this list with the RM's live inputs, so a chip never states terms the panel no
    # longer shows. Keep the two in step — if the panel's defaults change, change
    # these too, or an RM who opens the track and reads a chip before touching an
    # input is quoted stale terms.
    "COMPARE": [
        "On a S$1.2m loan at 2% with 360 months left, what do I save converting to 1.55% now versus waiting 3 months for 1.5%?",
        "At what rate gap does converting stop being worth it?",
        "Explain in plain English why converting early saves more",
        "Turn this comparison into a note I can send the customer",
    ],
}

# Next-best-action chips, in the order a stage is worked. The recommendation walks
# through three steps as the RM progresses:
#   1. docs incomplete           -> chase paperwork (_NBA_DOCS)
#   2. docs ok, not yet assessed -> run the assessment (_NBA_ASSESS)
#   3. assessed, no letter yet   -> draft the letter (_NBA_DRAFT)
# Each value is the EXACT chip label to promote. REPRICE is a single-step advisory
# lane (no assess/draft split), so it only has an assess-equivalent recommendation.
_NBA_ASSESS = {
    "IPA": "Run a full IPA eligibility assessment end-to-end",
    "LO": "Compare a 2-year fixed vs SORA floating package",
    "REPRICE": "A competitor offered 1.55% — compare against our package",
}
_NBA_DRAFT = {
    "IPA": "Draft the IPA letter",
    "LO": "Draft the Letter of Offer",
}
# The chip that chases missing paperwork, surfaced first when docs are incomplete.
_NBA_DOCS = "What documents are still missing?"


def _next_action_chips(stage: str, status: str, docs: list[dict],
                       assessed: bool = False) -> list[dict]:
    """Order the stage's chips with a recommended next step pinned first.

    Returns [{label, primary}]. The pinned chip (primary=True) is the system's
    next-best-action given the live deal progress, document state, and whether the
    RM has already run this stage's assessment: chase paperwork while docs are
    incomplete; then recommend the assessment; once assessed, recommend drafting
    the letter. A done/pending stage gets no recommendation (chips keep their
    authored order, none primary)."""
    chips = ACTION_CHIPS.get(stage, [])
    docs_complete = all(d.get("done") for d in docs) if docs else True

    recommended = None
    if status == "active":
        if not docs_complete and _NBA_DOCS in chips:
            recommended = _NBA_DOCS
        elif assessed:
            recommended = _NBA_DRAFT.get(stage)
        else:
            recommended = _NBA_ASSESS.get(stage)

    out = [{"label": c, "primary": c == recommended} for c in chips]
    # Float the recommended chip to the front so it reads as the next step.
    out.sort(key=lambda c: not c["primary"])
    return out

# Which documents each stage expects, and how to find them in the data.
# kind: "income" -> fact_income_docs.doc_type, "property" -> fact_property_docs.doc_type.
_STAGE_DOC_SPEC = {
    "IPA": [
        ("NRIC / SingPass verified", "profile", "singpass"),
        ("Payslips — last 3 months", "income", "Payslip"),
        ("Notice of Assessment (NOA)", "income", "NOA"),
        ("Option to Purchase (OTP)", "property", "OTP"),
    ],
    "LO": [
        ("Signed OTP", "property", "OTP"),
        ("CPF Withdrawal Statement", "property", "CPF Withdrawal Statement"),
    ],
    "REPRICE": [
        ("Existing loan / refinancing record", "income", "Payslip"),
        ("Option of Sale (if selling)", "property", "Option of Sale"),
    ],
}


def _qualifying_income(loan: dict) -> float:
    """Qualifying monthly income = fixed + variable*0.7 (MAS 30% haircut on
    variable income). Mirrors _build_ltv_fusion in graph.py so the summary card
    and the agents agree on the income figure."""
    fixed = loan.get("monthly_fixed_income") or 0
    var = loan.get("monthly_variable_income") or 0
    return fixed + var * 0.7


def _stage_progress(loan_app: dict, applicant_id: str = "") -> dict[str, str]:
    """Per-stage status (done / active / pending).

    Two inputs, runtime first: the RM's COMPLETED milestones this session
    (`case_progress`, set when a draft is approved at the HITL gate) override the
    CSV-derived starting state, so the dots advance as the RM actually works. The
    CSV `application_status` + `loan_type` give the seed state when no milestone
    has been recorded yet. Best-effort presentation, not a workflow engine."""
    status = (loan_app.get("application_status") or "").lower()
    loan_type = (loan_app.get("loan_type") or "").lower()

    if "refinanc" in loan_type:
        progress = {"IPA": "done", "LO": "done", "REPRICE": "active"}
    # New-purchase lanes: IPA then LO.
    elif status in ("approved",):
        progress = {"IPA": "done", "LO": "active", "REPRICE": "pending"}
    elif status in ("in review", "submitted"):
        progress = {"IPA": "active", "LO": "pending", "REPRICE": "pending"}
    else:
        # "Pending Docs" / "Rejected" / anything else -> still working IPA.
        progress = {"IPA": "active", "LO": "pending", "REPRICE": "pending"}

    # Runtime override: a stage the RM approved this session is done; the next
    # stage in the IPA -> LO -> REPRICE order becomes the new active one (unless a
    # later stage is already done/active).
    done = completed_stages(applicant_id) if applicant_id else {}
    if done.get("IPA") and progress["IPA"] != "done":
        progress["IPA"] = "done"
        if progress["LO"] == "pending":
            progress["LO"] = "active"
    if done.get("LO") and progress["LO"] != "done":
        progress["LO"] = "done"
        if progress["REPRICE"] == "pending":
            progress["REPRICE"] = "active"
    return progress


def _active_stage(progress: dict[str, str]) -> str:
    for k in STAGE_KEYS:
        if progress.get(k) == "active":
            return k
    return "IPA"


def _stage_short(progress: dict[str, str]) -> str:
    return STAGE_META[_active_stage(progress)]["label"]


# ── public API ─────────────────────────────────────────────────────────────
def list_cases() -> list[dict]:
    """All applicants for the case switcher, with their current stage."""
    out = []
    for aid in store.list_applicants():
        profile = store.get_profile(aid)
        loan_app = store.get_loan_application(aid)
        if "error" in profile:
            continue
        progress = _stage_progress(loan_app if "error" not in loan_app else {}, aid)
        out.append({
            "id": aid,
            "name": profile.get("full_name", aid),
            "stageShort": _stage_short(progress),
            "stageStatuses": progress,
        })
    return out


def _loan_summary(loan_app: dict, overrides: dict | None = None) -> dict | None:
    """Loan Summary card via the real calculate_loan tool (reverse mode: price the
    case at its declared property value). Returns None if the case has no usable
    loan application row.

    `overrides` carries the recompute inputs the RM changed this session via the
    reject→replan loop (`interest_rate_pct` / `target_property_price` /
    `n_props_owned`). They are applied on top of the CSV row so the KPI card
    re-prices at the adjusted value — the CSV itself is never mutated."""
    if not loan_app or "error" in loan_app:
        return None
    overrides = overrides or {}
    profile_age = loan_app.get("_age")  # filled by caller
    # target_property_price override wins over the declared property value.
    price = overrides.get("target_property_price") or loan_app.get("property_value_estimated")
    if not price:
        return None
    rate = float(overrides.get("interest_rate_pct") or loan_app.get("interest_rate_pct") or 3.5)
    # n_props_owned INCLUDES this purchase (1 = first-time buyer); both the CSV
    # column and an RM override use that convention, so neither needs adjusting.
    n_props = int(overrides.get("n_props_owned") or loan_app.get("no_sg_properties_owned") or 1)
    prop_raw = (loan_app.get("property_type") or "").lower()
    prop = "Private" if "private" in prop_raw or "condo" in prop_raw or "executive" in prop_raw else "HDB"
    borrowers = [{
        "age": profile_age or 35,
        "monthly_income": _qualifying_income(loan_app),
        "nationality": "Singapore Citizen",
    }]
    try:
        calc = calculate_loan(
            borrowers=borrowers,
            property_type=prop,
            n_outstanding_loans=int(loan_app.get("no_outstanding_home_loans") or 0),
            n_props_owned=n_props,
            interest_rate_pct=rate,
            monthly_car_loan=float(loan_app.get("monthly_car_loan") or 0),
            monthly_other=float(loan_app.get("monthly_other_commitments") or 0),
            target_property_price=float(price),
        )
    except Exception as e:  # never break the page on a calc edge case
        return {"error": str(e)}
    if "error" in calc:
        return calc
    ltv_pct = round(calc["eligible_loan"] / float(price) * 100) if price else None
    return {
        "property_price": calc.get("property_price"),
        "loan_amount": calc.get("eligible_loan"),
        "ltv_pct": ltv_pct,
        "tenure_years": calc.get("loan_tenure_years"),
        "monthly_repayment": calc.get("monthly_repayment"),
        "interest_rate_pct": rate or None,
        # True when an RM adjustment repriced this card away from the CSV seed —
        # the UI shows a small "adjusted" note so the RM knows why it moved.
        "adjusted": bool(overrides),
    }


def _borrower_block(profile: dict, loan_app: dict, credit: dict) -> dict:
    """Borrower persona strip for the workspace header (v3 "borrower + property
    strip"): name, age, citizenship, credit grade, and the qualifying-income /
    CPF / commitments figures. All read straight from the existing tables."""
    cbs = credit.get("cbs_credit_score") if "error" not in credit else None
    grade = credit.get("cbs_risk_grade") if "error" not in credit else None
    # The column counts properties INCLUDING this purchase, but the UI labels this
    # field "Properties owned" — so show what they already hold, i.e. minus this one.
    n_props = max(0, int(loan_app.get("no_sg_properties_owned") or 1) - 1) if loan_app else 0
    n_loans = int(loan_app.get("no_outstanding_home_loans") or 0) if loan_app else 0
    commitments = ((loan_app.get("monthly_car_loan") or 0)
                   + (loan_app.get("monthly_other_commitments") or 0)) if loan_app else 0
    initials = "".join(w[0] for w in (profile.get("full_name") or "?").split()[:2]).upper()
    return {
        "initials": initials or "?",
        "name": profile.get("full_name", ""),
        "age": profile.get("age"),
        "citizenship": profile.get("citizenship") or profile.get("residential_status") or "",
        "nric": profile.get("fake_nric_fin", ""),
        "cbs_score": cbs,
        "cbs_grade": grade,
        "verified_income": round(_qualifying_income(loan_app)) if loan_app else None,
        "cpf_oa": loan_app.get("cpf_oa_available") if loan_app else None,
        "cash": loan_app.get("cash_available") if loan_app else None,
        "n_props_owned": n_props,
        "n_outstanding_loans": n_loans,
        "monthly_commitments": commitments,
    }


def _property_block(loan_app: dict, prop_docs: list) -> dict:
    """Property facts for the strip: price, type, and (from the OTP/property doc)
    the address / floor area / tenure when available."""
    otp = next((d for d in prop_docs if d.get("doc_type") == "OTP"), None) \
        or (prop_docs[0] if prop_docs else None)
    detail = ""
    if otp:
        parts = [otp.get("property_address"),
                 f"{otp.get('floor_area_sqm')}sqm" if otp.get("floor_area_sqm") else None,
                 otp.get("tenure")]
        detail = " · ".join(str(p) for p in parts if p)
    return {
        "price": loan_app.get("property_value_estimated") if loan_app else None,
        "type": loan_app.get("property_type") if loan_app else None,
        "detail": detail,
    }


def _income_block(loan_app: dict, income_docs: list, summary: dict | None) -> dict:
    """Income & Affordability panel (TDSR basis). Surfaces the declared vs
    document-verified income figures plus the TDSR ceiling used by the calculator,
    so the RM sees how the qualifying number is built and how much room is left."""
    if not loan_app:
        return {}
    fixed = loan_app.get("monthly_fixed_income") or 0
    var = loan_app.get("monthly_variable_income") or 0
    qualifying = _qualifying_income(loan_app)
    payslips = [d for d in income_docs if d.get("doc_type") == "Payslip"]
    noa = next((d for d in income_docs if d.get("doc_type") == "NOA"), None)
    payslip_avg = round(sum(d.get("gross_income") or 0 for d in payslips) / len(payslips)) \
        if payslips else None
    commitments = ((loan_app.get("monthly_car_loan") or 0)
                   + (loan_app.get("monthly_other_commitments") or 0))
    # MAS TDSR ceiling is 55% of gross qualifying income.
    tdsr_ceiling = round(qualifying * 0.55)
    # Actual TDSR the case's repayment implies (commitments + monthly repayment).
    tdsr_pct = None
    if summary and summary.get("monthly_repayment") and qualifying:
        tdsr_pct = round((commitments + summary["monthly_repayment"]) / qualifying * 100, 1)
    return {
        "qualifying_income": round(qualifying),
        "fixed": round(fixed),
        "variable": round(var),
        "payslip_avg": payslip_avg,
        "noa_annual": (noa.get("assessed_year_income") if noa else None),
        "monthly_commitments": round(commitments),
        "tdsr_ceiling": tdsr_ceiling,
        "tdsr_pct": tdsr_pct,
    }


def _case_flags(profile: dict, loan_app: dict, credit: dict, summary: dict | None,
                documents: dict) -> list[dict]:
    """Auto-derived case flags for the right Analysis panel — no agent call, just
    read-through checks the RM would otherwise scan across tables. Each flag is
    {label, note, status} with status in ok / warn / bad."""
    flags: list[dict] = []
    # Credit eligibility.
    if "error" not in credit:
        bad = credit.get("bankruptcy_flag") or credit.get("default_flag")
        cbs = credit.get("cbs_credit_score")
        flags.append({
            "label": "Credit eligible" if not bad else "Adverse credit flag",
            "note": f"CBS {cbs} · {credit.get('cbs_risk_grade', '')}".strip(" ·"),
            "status": "bad" if bad else "ok",
        })
    # First-property / LTV cap. The column includes this purchase, so n_props == 1
    # is the first-time buyer; already_owned is what they hold going in.
    n_props = int(loan_app.get("no_sg_properties_owned") or 1) if loan_app else 1
    already_owned = max(0, n_props - 1)
    ltv = summary.get("ltv_pct") if summary else None
    flags.append({
        "label": "1st property" if n_props <= 1
                 else f"{already_owned} propert{'y' if already_owned == 1 else 'ies'} owned",
        "note": f"LTV cap {ltv}%" if ltv is not None else "",
        "status": "ok" if n_props <= 1 else "warn",
    })
    # Commitments / TDSR room.
    commitments = ((loan_app.get("monthly_car_loan") or 0)
                   + (loan_app.get("monthly_other_commitments") or 0)) if loan_app else 0
    flags.append({
        "label": "Zero commitments" if not commitments else "Existing commitments",
        "note": "Full TDSR room" if not commitments else fmt_money(commitments) + "/mo",
        "status": "ok" if not commitments else "warn",
    })
    # Documents across all stages (a quick "paperwork complete?" read).
    all_docs = [d for items in documents.values() for d in items]
    done = sum(1 for d in all_docs if d.get("done"))
    total = len(all_docs)
    flags.append({
        "label": "Docs verified" if done == total else "Docs outstanding",
        "note": f"{done} / {total} collected",
        "status": "ok" if total and done == total else "warn",
    })
    return flags


def fmt_money(n) -> str:
    try:
        return "S$" + format(round(float(n)), ",")
    except Exception:
        return "—"


def _docs_for_stage(aid: str, stage: str, income_docs: list, prop_docs: list,
                    profile: dict) -> list[dict]:
    spec = _STAGE_DOC_SPEC.get(stage, [])
    out = []
    for name, kind, key in spec:
        if kind == "profile":
            done = bool(profile.get("singpass_verified"))
        elif kind == "income":
            rows = [d for d in income_docs if (d.get("doc_type") == key)]
            done = bool(rows) and all(bool(d.get("doc_verified")) for d in rows)
        else:  # property
            rows = [d for d in prop_docs if (d.get("doc_type") == key)]
            done = bool(rows) and all(bool(d.get("doc_verified")) for d in rows)
        out.append({"name": name, "done": done})
    return out


def get_case(applicant_id: str) -> dict:
    """Full case bundle for the left rail + center header: profile, deal-progress
    dots, loan summary card, per-stage documents checklist, action chips."""
    aid = applicant_id.upper()
    profile = store.get_profile(aid)
    if "error" in profile:
        return {"error": f"{aid} not found"}
    loan_app = store.get_loan_application(aid)
    credit = store.get_bank_credit(aid)
    income_docs = store.get_income_docs(aid)
    prop_docs = store.get_property_docs(aid)
    loan_app_ok = loan_app if "error" not in loan_app else {}

    progress = _stage_progress(loan_app_ok, aid)
    assessed = assessed_stages(aid)

    # Re-price the KPI card at any recompute override the RM applied this session
    # (e.g. repriced the letter to 1.2%). Read against the active stage's lane.
    active_stage = _active_stage(progress)
    overrides = case_overrides.get_overrides(aid, active_stage)

    summary_input = dict(loan_app_ok)
    summary_input["_age"] = profile.get("age")
    summary = _loan_summary(summary_input, overrides)

    documents = {
        stage: _docs_for_stage(aid, stage, income_docs, prop_docs, profile)
        for stage in STAGE_KEYS
    }

    return {
        "id": aid,
        "name": profile.get("full_name", aid),
        "stageStatuses": progress,
        "activeStage": _active_stage(progress),
        "stages": [
            {"key": k, **STAGE_META[k], "status": progress.get(k, "pending")}
            for k in STAGE_KEYS
        ],
        "loanSummary": summary,
        "borrower": _borrower_block(profile, loan_app_ok, credit),
        "property": _property_block(loan_app_ok, prop_docs),
        "income": _income_block(loan_app_ok, income_docs, summary),
        "caseFlags": _case_flags(profile, loan_app_ok, credit, summary, documents),
        "documents": documents,
        # Which stages the RM has already assessed this session — drives the v3
        # empty→assessed state (KPI card shows "—" until the assessment runs, then
        # fills; the right-panel Assessment Result card appears once assessed).
        "assessedStages": assessed,
        "actionChips": {
            **{
                stage: _next_action_chips(stage, progress.get(stage, "pending"),
                                          documents.get(stage, []),
                                          assessed.get(stage, False))
                for stage in STAGE_KEYS
            },
            # COMPARE is a client-only track, so it is NOT in STAGE_KEYS (which drives
            # real case progress, documents and the HITL gate) — its chips are attached
            # here instead. Passed as status "none" so nothing is promoted to primary:
            # a what-if tool has no next-best action to recommend.
            "COMPARE": _next_action_chips("COMPARE", "none", []),
        },
    }
