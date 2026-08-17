"""FastAPI app — serves the Agentic Home Loan Workbench UI and drives the graph.

Run from mas/:
    py -m uvicorn server.app:app --reload
then open http://127.0.0.1:8000

This is a thin transport layer. All agent logic lives in graph.py; all case data
lives in case_service.py (over the existing DataStore). The chat endpoint streams
graph events as Server-Sent Events so the frontend can render live thinking blocks
and the HITL draft gate.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response as FAResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import Response

from server import case_service
from utils import letter_store

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Agentic Home Loan Workbench")


# ── Usage tracking ──────────────────────────────────────────────────────────
# Assigns every browser a persistent random `cid` cookie and logs page views to
# usage_data/visits.csv, so the project can report how many distinct browsers
# opened the tool. The same cid is stamped on the customer-calculator rows, which
# is what makes the two files joinable into a conversion figure.
#
# Wrapped end to end: instrumentation that can break a page load is worse than no
# instrumentation, so every failure here degrades to "no row written".
@app.middleware("http")
async def track_usage(request, call_next):
    from utils.telemetry import usage

    try:
        cid = request.cookies.get(usage.COOKIE_NAME) or ""
        is_new = not cid
        if is_new:
            cid = usage.new_cid()
        # Stash on the request so downstream endpoints (the customer calculator)
        # can stamp the same visitor id without re-parsing cookies.
        request.state.cid = cid
    except Exception:  # noqa: BLE001
        cid, is_new = "", False

    response = await call_next(request)

    try:
        # Only real page views are logged; assets and API polls would otherwise
        # write a dozen rows per visit and make the count meaningless.
        if usage.role_for(request.url.path):
            usage.record_visit(request, cid, is_new=is_new)
        if is_new and cid:
            response.set_cookie(
                usage.COOKIE_NAME, cid,
                max_age=usage.COOKIE_MAX_AGE, httponly=True, samesite="lax",
            )
    except Exception as e:  # noqa: BLE001
        print(f"  [usage] middleware skipped: {e}")

    return response


class NoCacheStaticFiles(StaticFiles):
    """Serve the frontend with no-cache headers so the browser always picks up the
    latest app.js / styles.css during development (avoids stale-JS confusion)."""

    def file_response(self, *args, **kwargs) -> Response:
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp


# ── API: read-only case data ────────────────────────────────────────────────
@app.get("/api/cases")
def api_cases():
    return {"cases": case_service.list_cases()}


@app.get("/api/case/{applicant_id}")
def api_case(applicant_id: str):
    return case_service.get_case(applicant_id)


# ── API: chat / HITL (filled in next steps) ─────────────────────────────────
class ChatRequest(BaseModel):
    applicant_id: str
    stage: str
    message: str
    role: str = "rm"   # "rm" (workbench) | "customer" (self-service portal)


class DecisionRequest(BaseModel):
    applicant_id: str
    stage: str
    feedback: str = ""


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    from server.stream import stream_chat

    return StreamingResponse(
        stream_chat(req.applicant_id, req.stage, req.message, role=req.role),
        media_type="text/event-stream",
    )


@app.post("/api/approve")
def api_approve(req: DecisionRequest):
    from server.stream import stream_resume

    return StreamingResponse(
        stream_resume(req.applicant_id, req.stage, approved=True, feedback=req.feedback),
        media_type="text/event-stream",
    )


@app.post("/api/reject")
def api_reject(req: DecisionRequest):
    from server.stream import stream_resume

    return StreamingResponse(
        stream_resume(req.applicant_id, req.stage, approved=False, feedback=req.feedback),
        media_type="text/event-stream",
    )


class ResetRequest(BaseModel):
    applicant_id: str
    stage: str


@app.post("/api/reset")
def api_reset(req: ResetRequest):
    """Clear one lane's conversation history. Plain JSON, not SSE — nothing streams.

    Exists because the session transcript only grows, so a long case can outgrow a
    small-context endpoint and lock the RM out of the chat entirely. Clears the
    conversation only; case data (progress, overrides, letters) is untouched.
    """
    from server.stream import reset_session

    return reset_session(req.applicant_id, req.stage)


# ── API: customer self-service (toC) ────────────────────────────────────────
# The interactive calculator on /customer posts here and gets deterministic
# figures straight from utils/calculator.py — no LLM in the loop, so the panel
# is instant, free, and can never hallucinate. The chat assistant is the only
# LLM surface on that page (via /api/chat with role="customer").
class CustomerCalcRequest(BaseModel):
    mode: str = "price"                     # price | downpayment | instalment | explore
    age: int
    monthly_fixed_income: float
    monthly_variable_income: float = 0.0
    nationality: str = "Singapore Citizen"
    property_type: str = "Private"          # "HDB" | "Private"
    properties_owned_now: int = 0           # EXCLUDING the one being bought
    outstanding_home_loans: int = 0
    monthly_car_loan: float = 0.0
    monthly_other: float = 0.0
    interest_rate_pct: float
    target_property_price: float | None = None   # price mode
    cash_cpf_available: float | None = None      # downpayment + instalment modes
    monthly_budget: float | None = None          # instalment mode
    # MAS Notice 645 eligible financial assets, keyed by asset class; each value
    # is {"amount": float, "pledged": bool}. Absent/empty means no assets, which
    # contributes exactly zero and leaves every other figure unchanged.
    financial_assets: dict | None = None


# Every number in every mode comes from utils/calculator.calculate_loan — the
# single source of truth. The two derived modes never introduce new formulas:
#   instalment — root-finding (bisection on property price) OVER the calculator
#                until the calculator's own monthly_repayment meets the budget;
#   explore    — two calculator calls: a forward probe with unconstrained cash
#                to read the income-capped max loan, then a reverse pricing at
#                the price where that loan exactly meets the LTV limit.
def _customer_calc(req: CustomerCalcRequest, **mode_kwargs) -> dict:
    from utils.calculator import amortized_monthly_income, calculate_loan

    # MAS Notice 645: eligible financial assets amortised over 48 months. This is
    # already net of its own haircut, so it is added AFTER the variable-income
    # haircut and must never be multiplied by 0.7 as well — the two are separate
    # rules and stacking them would double-discount the assets. Zero when the
    # customer supplies no assets, which is the default.
    assets_income = amortized_monthly_income(req.financial_assets)["monthly_income"]

    return calculate_loan(
        borrowers=[{
            "age": req.age,
            # Qualifying income = fixed + variable × 0.7 (MAS haircut) — applied
            # here, same as the RM-side fusion layer; the calculator gets the
            # already-haircut figure and does no further haircut.
            "monthly_income": (req.monthly_fixed_income
                               + req.monthly_variable_income * 0.7
                               + assets_income),
            "nationality": req.nationality,
        }],
        property_type=req.property_type,
        n_outstanding_loans=req.outstanding_home_loans,
        # The calculator counts properties INCLUDING this purchase; the form
        # asks what the customer owns NOW, hence the +1 (0 now → first home → 1).
        n_props_owned=req.properties_owned_now + 1,
        interest_rate_pct=req.interest_rate_pct,
        monthly_car_loan=req.monthly_car_loan,
        monthly_other=req.monthly_other,
        **mode_kwargs,
    )


# Rows sent to the browser up front. The full schedule is up to 420 rows (35y);
# the panel shows the first year and fetches the rest only if the customer opens
# it, so the common case ships a small payload.
_SCHEDULE_PREVIEW_ROWS = 12


def _with_schedule(res: dict, mode: str, assets: dict | None = None) -> dict:
    """Attach the repayment schedule and the BUC progressive payment schedule.

    Both are built from the result's OWN figures (loan / price / rate / tenure /
    LTV), so neither table can contradict the card above it or each other: once
    the BUC loan is fully drawn its instalment is by construction the same
    monthly repayment the schedule shows. `schedule_total_months` lets the UI say
    how many rows exist without shipping them all.

    ``assets`` is the MAS 645 derivation, echoed back so the result page can show
    the customer how their assets became income rather than presenting a total
    that silently moved.
    """
    from utils.calculator import amortization_schedule, buc_progressive_schedule

    res["mode"] = mode
    if assets and assets.get("rows"):
        res["financial_assets"] = assets
        res["amortized_monthly_income"] = assets["monthly_income"]
    tenure = res.get("loan_tenure_years")
    res["schedule"] = amortization_schedule(
        res.get("eligible_loan"), res.get("interest_rate_pct"), tenure,
        max_rows=_SCHEDULE_PREVIEW_ROWS,
    )
    res["schedule_total_months"] = int(tenure) * 12 if tenure else 0
    # Progressive payment applies to a property still being built. It is priced
    # off the property price and the case's own LTV, so the cash/loan crossover
    # lands where this borrower's downpayment actually runs out.
    res["buc"] = buc_progressive_schedule(
        res.get("property_price"), res.get("interest_rate_pct"), tenure,
        ltv_pct=res.get("ltv_limit_pct") or 75.0,
        eligible_loan=res.get("eligible_loan"),
    )
    return res


@app.post("/api/customer/calc")
def api_customer_calc(req: CustomerCalcRequest, request: Request):
    """Run the customer calculator and log what was asked.

    The "Show my results" button already posts here, so the usage row is captured
    server-side with no frontend change. It is written on EVERY exit path — the
    early validation returns are real usage too, and dropping them would hide both
    demand and the inputs people get stuck on.
    """
    from utils.telemetry import usage

    cid = getattr(request.state, "cid", "") or ""
    res = _run_customer_calc(req)
    usage.record_customer_input(req, res, cid)
    return res


def _run_customer_calc(req: CustomerCalcRequest) -> dict:
    """The calculator dispatch itself — unchanged logic, lifted out of the endpoint
    so usage logging wraps a single exit instead of eight scattered returns."""
    from utils.calculator import amortized_monthly_income

    assets = amortized_monthly_income(req.financial_assets)
    try:
        if req.mode == "price":
            if not req.target_property_price:
                return {"error": "Enter your target property price."}
            res = _customer_calc(req, target_property_price=req.target_property_price)

        elif req.mode == "downpayment":
            if not req.cash_cpf_available:
                return {"error": "Enter the cash + CPF you have set aside."}
            res = _customer_calc(req, cash_cpf_available=req.cash_cpf_available)

        elif req.mode == "explore":
            # Probe with unconstrained cash so only income (TDSR/MSR) caps the
            # loan, then price the case at the smallest property that lets the
            # LTV limit draw that full loan — the honest "maximum loan" story.
            probe = _customer_calc(req, cash_cpf_available=1e9)
            max_loan = probe.get("eligible_loan") or 0
            ltv = (probe.get("ltv_limit_pct") or 75.0) / 100.0
            if not max_loan or not ltv:
                return {"error": "Your income and commitments leave no loan headroom."}
            res = _customer_calc(req, target_property_price=max_loan / ltv)
            res["note"] = ("Maximum loan your income supports; the property budget shown "
                           "is the price at which that loan meets the LTV limit.")

        elif req.mode == "instalment":
            if not req.monthly_budget or not req.cash_cpf_available:
                return {"error": "Enter both your monthly budget and the cash + CPF set aside."}
            # Upper bound: the most this cash could buy regardless of budget.
            fwd = _customer_calc(req, cash_cpf_available=req.cash_cpf_available)
            if (fwd.get("monthly_repayment") or 0) <= req.monthly_budget:
                fwd["note"] = ("Your monthly budget is not the limit — with this downpayment "
                               "the instalment stays within it.")
                return _with_schedule(fwd, req.mode, assets)
            # Bisection on property price over the calculator itself: largest
            # price whose calculator-computed instalment fits the budget.
            lo, hi = 50_000.0, float(fwd.get("property_price") or 0)
            if hi <= lo:
                return {"error": "The cash + CPF set aside is too small to price a purchase."}
            best = None
            for _ in range(40):
                mid = (lo + hi) / 2
                trial = _customer_calc(req, target_property_price=mid)
                if (trial.get("monthly_repayment") or 0) <= req.monthly_budget:
                    best, lo = trial, mid
                else:
                    hi = mid
            if best is None:
                return {"error": "Even the smallest purchase exceeds that monthly budget — "
                                 "try a higher budget or a lower rate."}
            best["note"] = "Priced to keep the instalment within your monthly budget."
            res = best

        else:
            return {"error": f"Unknown mode '{req.mode}'."}

        return _with_schedule(res, req.mode, assets)
    except Exception as e:  # surface calculator validation errors as data, not a 500
        return {"error": f"{type(e).__name__}: {e}"}


class ScheduleRequest(BaseModel):
    """Inputs for the full repayment schedule — the three figures the result card
    already shows, echoed back by the browser so this stays a pure function of
    them (no server-side session state to drift out of sync)."""
    eligible_loan:     float
    interest_rate_pct: float
    loan_tenure_years: int


@app.post("/api/customer/schedule")
def api_customer_schedule(req: ScheduleRequest):
    """Full month-by-month schedule, fetched when the customer expands the table."""
    from utils.calculator import amortization_schedule

    try:
        rows = amortization_schedule(
            req.eligible_loan, req.interest_rate_pct, req.loan_tenure_years)
        return {"schedule": rows, "schedule_total_months": len(rows)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


class SavingsRequest(BaseModel):
    """Inputs for the RM-side Package Comparison page.

    Deliberately a plain deterministic endpoint rather than an agent tool: the RM
    types numbers and expects an instant, repeatable answer. Routing this through
    an LLM would be slower, cost money per keystroke-driven recalculation, and
    introduce a chance of the figures drifting from the calculator.
    """
    outstanding_loan:     float
    current_rate_pct:     float
    remaining_months:     int
    convert_after_months: int = 0
    rate_a_pct:           float | None = None
    rate_b_pct:           float | None = None
    horizon_months:       int | None = None


@app.post("/api/compare/savings")
def api_compare_savings(req: SavingsRequest):
    from utils.calculator import interest_savings

    try:
        return interest_savings(
            outstanding_loan=req.outstanding_loan,
            current_rate_pct=req.current_rate_pct,
            remaining_months=req.remaining_months,
            convert_after_months=req.convert_after_months,
            rate_a_pct=req.rate_a_pct,
            rate_b_pct=req.rate_b_pct,
            horizon_months=req.horizon_months,
        )
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/api/customer/packages")
def api_customer_packages():
    """Public rate catalog for the calculator's rate picker (not case data)."""
    from utils.tools import store

    return {"packages": store.list_loan_packages()}


# ── API: market rates (MAS benchmark + competitor packages) ─────────────────
# Two routes on purpose: reading the stored CSV is instant, while a live
# collect() takes ~4s (MAS API dominates). The panel loads from /api/rates and
# only hits /api/rates/refresh when the RM explicitly asks, so opening it never
# blocks and never re-hits the sources.
def _read_market_rates() -> dict:
    """Latest competitor snapshot + the SORA series, from the cumulative CSV."""
    import csv as _csv

    from utils.rates_scraper import DEFAULT_OUT, SOURCE_SITES

    path = Path(DEFAULT_OUT)
    if not path.exists():
        return {
            "sora": [],
            "competitors": [],
            "scraped_at": None,
            "sources": SOURCE_SITES,
            "empty": True,
        }

    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(_csv.DictReader(f))

    sora = sorted(
        (
            {"as_of": r["rate_as_of"], "rate": float(r["year_1"])}
            for r in rows
            if r["source"] == "mas" and r["year_1"]
        ),
        key=lambda d: d["as_of"],
    )

    # Competitor packages repeat every run; keep only the most recent batch.
    pkg = [r for r in rows if r["source"] == "dollarback"]
    latest_at = max((r["scraped_at_utc"] for r in pkg), default=None)
    competitors = [
        {
            "bank": r["bank"],
            "rate_category": r["rate_category"],
            "loan_type": r["loan_type"],
            "lock_in_years": r["lock_in_years"],
            "years": [r[f"year_{i}"] for i in range(1, 7)],
            "note": r["note"],
        }
        for r in pkg
        if r["scraped_at_utc"] == latest_at
    ]
    return {
        "sora": sora,
        "competitors": competitors,
        "bank_history": _bank_history(pkg, sora),
        "scraped_at": latest_at,
        "sora_as_of": sora[-1]["as_of"] if sora else None,
        "sora_latest": sora[-1]["rate"] if sora else None,
        "sources": SOURCE_SITES,
        "empty": not (sora or competitors),
    }


def _sora_on(sora: list[dict], day: str) -> float | None:
    """SORA level effective on `day` — the latest published on or before it.

    MAS publishes on business days only, so a package scraped on a weekend has
    no same-day benchmark; carrying the last published value forward is what
    "the SORA in force that day" means.
    """
    val = None
    for p in sora:                      # already sorted ascending by as_of
        if p["as_of"] <= day:
            val = p["rate"]
        else:
            break
    return val


def _bank_history(pkg: list[dict], sora: list[dict]) -> dict:
    """Year-1 all-in rate per package over time: {label: {category, points}}.

    Floating packages are stored as a spread over the benchmark, but a spread
    and a fixed rate are not comparable quantities. Here the spread is resolved
    to the all-in rate (SORA on that day + spread) so every line on the chart
    means the same thing: what the borrower actually pays.

    Unlike SORA there is no historical feed for competitor rates — the page
    only ever shows "now" — so this series grows one point per day the
    collector runs. Collapsed to one point per calendar day (the last
    observation wins) so re-running several times in a day cannot bend the
    line; the shape reflects rate changes, not how often we scraped.
    """
    by_series: dict[str, dict[str, tuple[str, float]]] = {}
    category: dict[str, str] = {}
    for r in pkg:
        raw = r.get("year_1", "")
        if not raw:
            continue
        try:
            rate = float(str(raw).lstrip("+"))
        except ValueError:
            continue
        day = r["scraped_at_utc"][:10]

        if r["rate_category"] == "floating":
            base = _sora_on(sora, day)
            if base is None:
                continue        # no benchmark for that day → cannot state a real rate
            rate = round(base + rate, 4)

        label = f"{r['bank']} · {r['loan_type']}"
        category[label] = r["rate_category"]
        prev = by_series.setdefault(label, {}).get(day)
        # Keep the latest observation within the day.
        if prev is None or r["scraped_at_utc"] >= prev[0]:
            by_series[label][day] = (r["scraped_at_utc"], rate)

    # Category rides along so the UI can plot fixed and floating separately —
    # they are different units and do not belong on one axis.
    return {
        label: {
            "category": category[label],
            "points": [{"day": d, "rate": v[1]} for d, v in sorted(days.items())],
        }
        for label, days in sorted(by_series.items())
    }


@app.get("/api/rates")
def api_rates():
    """Stored rates — instant, no network."""
    return _read_market_rates()


@app.post("/api/rates/refresh")
def api_rates_refresh():
    """Re-collect from both sources, then return the refreshed view.

    Source failures come back in `errors` so the UI can say a rate is stale or
    missing; no fallback figure is ever substituted.
    """
    from utils.rates_scraper import collect

    summary = collect()
    payload = _read_market_rates()
    payload["errors"] = summary["errors"]
    payload["added"] = summary["bank_rows"] + summary["sora_rows"]
    return payload


# ── API: letter PDF download ────────────────────────────────────────────────
# The most recent draft (draft=1) or released (draft=0) letter for this lane,
# built and stashed by stream.py at the HITL gate / on approval. Streamed inline
# so the browser can preview it; the '-DRAFT' suffix lives in the filename.
@app.get("/api/letter/{applicant_id}/{stage}")
def api_letter(applicant_id: str, stage: str, draft: int = 1):
    hit = letter_store.get(applicant_id, stage, bool(draft))
    if hit is None:
        return FAResponse(status_code=404, content="No letter available for this case yet.")
    pdf, filename = hit
    return FAResponse(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ── index.html with content-hashed asset versions ──────────────────────────
# The page references /app.js?v=N and /styles.css?v=N. Rewriting N to a hash of
# the current file contents on every request means the browser is *forced* to
# refetch whenever the JS/CSS actually changes (a fixed ?v= stamp would let a
# stale cached copy linger across edits — the cause of the "thinking summary
# still broken after a fix" confusion). Served by an explicit route registered
# BEFORE the static mount so it wins over the directory's index.html.
def _file_hash(name: str) -> str:
    p = STATIC_DIR / name
    return hashlib.sha1(p.read_bytes()).hexdigest()[:8] if p.is_file() else "0"


def _page_html(page: str) -> str:
    html = (STATIC_DIR / page).read_text(encoding="utf-8")
    # Stamp each asset with a hash of ITS OWN content (independent stamps), so a
    # change to only one js/css file still busts that one file's cache.
    return re.sub(
        r'/((?:\w+)\.(?:js|css))\?v=[^"\']*',
        lambda m: f'/{m.group(1)}?v={_file_hash(m.group(1))}',
        html,
    )


if STATIC_DIR.is_dir():
    def _page(name: str) -> HTMLResponse:
        resp = HTMLResponse(_page_html(name))
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp

    # Role selection landing: the entry point picks the audience. The RM
    # workbench moved to /rm (its old /index.html URL still works).
    @app.get("/", response_class=HTMLResponse)
    def landing():
        return _page("landing.html")

    @app.get("/rm", response_class=HTMLResponse)
    @app.get("/index.html", response_class=HTMLResponse)
    def index():
        return _page("index.html")

    # Customer self-service portal (toC): its own lightweight page, same visual
    # system. Kept separate from the RM SPA so the restricted lane shares no
    # case-workspace code — it can't render what it never loads.
    @app.get("/customer", response_class=HTMLResponse)
    @app.get("/customer.html", response_class=HTMLResponse)
    def customer():
        return _page("customer.html")

    # static frontend (mounted last so /api/* and the index route take precedence)
    app.mount("/", NoCacheStaticFiles(directory=str(STATIC_DIR), html=True), name="static")
