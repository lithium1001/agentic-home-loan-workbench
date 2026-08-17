"""Tool registry — the only sanctioned data-access surface for agents.

Three sections, mirroring the original notebook cell:
  A. TOOL_SCHEMAS / TOOL_SCHEMA_MAP — OpenAI function-calling descriptors.
  B. execute_tool() — the text/JSON dispatcher (legacy path used by the
     orchestrator's plain chat() routing).
  C. LangChain @tool wrappers + TOOLS_BY_NAME / ALL_TOOLS — what LangGraph
     agents actually bind via bind_tools().

The module owns a single shared ``store`` (DataStore over config.DATA_DIR);
both execute_tool() and the @tool wrappers read through it. To add a new tool,
implement the function, append a schema (A), a dispatcher entry (B) and a
@tool wrapper (C), then register it in _AGENT_TOOLS in the notebook.
"""

import json
from itertools import zip_longest

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from typing import Annotated as _Annotated, Optional as _Opt

from utils.calculator import calculate_loan, compare_packages, interest_savings
from utils.config import DATA_DIR
from utils.data import DataStore
from utils.policy_rag import search_policy as _search_policy
from utils.rates import SoraUnavailable, get_sora_3m

# Single shared data layer for the whole tool registry.
store = DataStore(DATA_DIR)


def _sora_rate_payload() -> dict:
    """Live 3M SORA + the bank's derived floating rate (= SORA + floating spread).

    The floating spread is read from the catalog's floating package so the two
    stay consistent; defaults to 0.20% if no floating package declares a spread.
    """
    try:
        sora = get_sora_3m()
    except SoraUnavailable as exc:
        # Report the failure instead of substituting a stale rate — the agent
        # must tell the user the live rate is unavailable.
        return {
            "benchmark": "3M Compounded SORA",
            "error": str(exc),
            "sora_3m_pct": None,
            "as_of": None,
            "source": "unavailable",
            "note": "Live SORA unavailable; do not quote a floating rate.",
        }
    spread = None
    for pkg in store.list_loan_packages():
        if str(pkg.get("rate_type", "")).lower() == "floating" and pkg.get("spread_pct") is not None:
            spread = float(pkg["spread_pct"])
            break
    if spread is None:
        spread = 0.20
    return {
        "benchmark": "3M Compounded SORA",
        "sora_3m_pct": sora["rate"],
        "as_of": sora["as_of"],
        "source": sora["source"],
        "floating_spread_pct": spread,
        "floating_rate_pct": round(sora["rate"] + spread, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — TOOL_SCHEMAS: OpenAI function-calling descriptors
#
# These dicts are passed verbatim to the LLM as the `tools` parameter.
# The LLM picks a tool by name and fills in the arguments; Python then
# executes the real function in execute_tool() below.
# ═══════════════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_profile",
            "description": "Fetch basic demographic profile for a borrower (name, NRIC, DOB, citizenship, education).",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string", "description": "e.g. APP0001"}
                },
                "required": ["applicant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_loan_application",
            "description": "Fetch the raw loan application inputs: income figures, property details, available funds, interest rate, and non-mortgage obligations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string", "description": "e.g. APP0001"}
                },
                "required": ["applicant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bank_credit",
            "description": "Fetch CBS credit score, risk grade, bank account details, total debt outstanding, bankruptcy/default flags, and verified income.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string", "description": "e.g. APP0001"}
                },
                "required": ["applicant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cpf_history",
            "description": "Fetch CPF contribution history (OA/SA/MA balances) for the last N months.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string", "description": "e.g. APP0001"},
                    "months": {"type": "integer", "description": "Number of recent months to return (default 6)", "default": 6}
                },
                "required": ["applicant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_income_docs",
            "description": "Fetch OCR-extracted income documents (payslips, NOA, commission statements) with verification status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string", "description": "e.g. APP0001"}
                },
                "required": ["applicant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_property_docs",
            "description": "Fetch property documents (OTP, CPF withdrawal, option-of-sale) for a borrower's application.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string", "description": "e.g. APP0001"}
                },
                "required": ["applicant_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_packages",
            "description": (
                "Compare TWO (or more) loan packages side-by-side for the SAME borrower case. "
                "Runs calculate_loan once per package (each package only differs by its interest rate / "
                "rate type) and returns a per-package result plus pairwise deltas vs the first package "
                "(monthly repayment, eligible loan, TDSR). Use this for (a) fixed-vs-floating comparison "
                "and (b) reprice/retention (our package vs a competitor's quoted rate). "
                "All the case fields are exactly the calculate_loan inputs EXCEPT interest_rate_pct, "
                "which is supplied per-package. Pull our own rates from the loan-package catalog "
                "(list_loan_packages); for a competitor, pass the rate the customer was quoted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "packages": {
                        "type": "array",
                        "description": "Two or more packages to compare for this same case. The first is the baseline that deltas are measured against (e.g. our package).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label":             {"type": "string", "description": "Human-readable package name shown in the comparison, e.g. '2-Year Fixed', 'SORA Floating', 'DBS quoted'."},
                                "interest_rate_pct": {"type": "number", "description": "This package's market interest rate in % p.a. Stress test stays fixed at 4%."},
                                "rate_type":         {"type": "string", "enum": ["Fixed", "Floating"], "description": "Optional rate type, echoed back into the comparison for context."}
                            },
                            "required": ["label", "interest_rate_pct"]
                        }
                    },
                    "borrowers": {
                        "type": "array",
                        "description": "One entry per borrower (same as calculate_loan).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "age":            {"type": "number", "description": "Borrower age in years"},
                                "monthly_income": {"type": "number", "description": "QUALIFYING monthly income = fixed + variable*0.7 (haircut already applied)."},
                                "nationality":    {"type": "string", "enum": ["Singapore Citizen", "Permanent Resident", "Foreigner"]}
                            },
                            "required": ["age", "monthly_income", "nationality"]
                        }
                    },
                    "property_type":       {"type": "string", "enum": ["Private", "HDB"], "description": "Private or HDB."},
                    "n_outstanding_loans": {"type": "integer", "description": "Number of outstanding home loans."},
                    "n_props_owned":       {"type": "integer", "description": "Properties owned INCLUDING this purchase."},
                    "monthly_car_loan":    {"type": "number", "description": "Monthly car loan repayment in SGD (0 if none)."},
                    "monthly_other":       {"type": "number", "description": "Other monthly debt obligations in SGD."},
                    "cash_cpf_available":  {"type": "number", "description": "Forward mode: total cash + CPF for down payment. Omit for reverse mode."},
                    "target_property_price": {"type": "number", "description": "Reverse mode: target property price. Omit for forward mode."}
                },
                "required": [
                    "packages", "borrowers", "property_type", "n_outstanding_loans",
                    "n_props_owned", "monthly_car_loan", "monthly_other"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_loan_packages",
            "description": (
                "List the bank's own loan-package catalog (fixed and floating products) with their "
                "indicative rates, lock-in and notes. Bank-wide rates (uniform for all borrowers). "
                "Use this to get our own packages' interest rates before calling compare_packages."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_sora_rate",
            "description": (
                "Fetch the CURRENT live 3-month compounded SORA benchmark from MAS, and derive "
                "the bank's live floating package rate (= 3M SORA + the floating package's spread). "
                "Floating-rate packages re-price every 3 months off SORA, so their indicative_rate_pct "
                "in the catalog is only a snapshot — call THIS for the up-to-date floating rate before a "
                "fixed-vs-floating comparison. Returns the SORA value, its as-of date, the spread, and the "
                "computed floating rate. The fixed package rate stays as quoted in list_loan_packages."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_loan",
            "description": (
                "MAS-compliant loan calculator. Computes eligible loan amount, max property price "
                "or required cash+CPF, monthly repayment, TDSR, LTV, loan tenure, BSD and ABSD. "
                "Also returns calculation_steps: a step-by-step derivation trace for human audit. "
                "Always call this instead of doing the arithmetic yourself. "
                "Forward mode: provide cash_cpf_available — returns max property price. "
                "Reverse mode: provide target_property_price — returns required cash+CPF."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "borrowers": {
                        "type": "array",
                        "description": "One entry per borrower.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "age":            {"type": "number", "description": "Borrower age in years"},
                                "monthly_income": {"type": "number", "description": "QUALIFYING monthly income = fixed + variable*0.7 (30% haircut on variable income already applied). Not the gross amount."},
                                "nationality":    {"type": "string", "enum": ["Singapore Citizen", "Permanent Resident", "Foreigner"]}
                            },
                            "required": ["age", "monthly_income", "nationality"]
                        }
                    },
                    "property_type": {
                        "type": "string",
                        "enum": ["Private", "HDB"],
                        "description": "Private (max 35yr tenure) or HDB (max 30yr tenure, MSR 30% applies)"
                    },
                    "n_outstanding_loans": {
                        "type": "integer",
                        "description": "Number of outstanding home loans at time of application."
                    },
                    "n_props_owned": {
                        "type": "integer",
                        "description": "Number of SG properties owned INCLUDING this new purchase (1, 2, 3+). Drives both LTV and ABSD rate."
                    },
                    "interest_rate_pct": {
                        "type": "number",
                        "description": "Market interest rate in % p.a. (e.g. 3.5). Stress test is fixed at 4%."
                    },
                    "monthly_car_loan": {
                        "type": "number",
                        "description": "Monthly car loan repayment in SGD (0 if none)."
                    },
                    "monthly_other": {
                        "type": "number",
                        "description": "Other monthly debt obligations in SGD (credit card, personal loan, etc.)."
                    },
                    "cash_cpf_available": {
                        "type": "number",
                        "description": "Forward mode: total cash + CPF OA available for down payment. Omit for reverse mode."
                    },
                    "target_property_price": {
                        "type": "number",
                        "description": "Reverse mode: target property price in SGD. Omit for forward mode."
                    }
                },
                "required": [
                    "borrowers", "property_type", "n_outstanding_loans", "n_props_owned",
                    "interest_rate_pct", "monthly_car_loan", "monthly_other"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "Retrieve the bank's product / promotion policy clauses (e.g. home-loan "
                "Terms & Conditions, sign-up gift tiers, eligibility exclusions) from the "
                "official documents. Use this to confirm product terms, promotion eligibility, "
                "gift entitlements or exclusions with the EXACT wording instead of relying on "
                "memory. Returns the most relevant clauses, each with its source document and "
                "clause title so you can cite them. Returns an empty list if no policy document "
                "matches the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look up, e.g. 'sign up gift for a S$1.2m loan' or 'is refinancing eligible'."},
                    "k": {"type": "integer", "description": "Number of clauses to return (default 4).", "default": 4}
                },
                "required": ["query"]
            }
        }
    },
]

# Lookup map used by AgentDef.tools to filter schemas by name.
TOOL_SCHEMA_MAP: dict[str, dict] = {
    t["function"]["name"]: t for t in TOOL_SCHEMAS
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — execute_tool(): the Python-side dispatcher
#
# Maps tool name → Python function, serialises the result to JSON, and returns
# it as a string to be appended as a "tool" role message.
# ═══════════════════════════════════════════════════════════════════════════════

def execute_tool(name: str, args: dict) -> str:
    fn_map = {
        "get_profile":          lambda a: store.get_profile(a["applicant_id"]),
        "get_loan_application": lambda a: store.get_loan_application(a["applicant_id"]),
        "get_bank_credit":      lambda a: store.get_bank_credit(a["applicant_id"]),
        "get_cpf_history":      lambda a: store.get_cpf_history(a["applicant_id"], a.get("months", 6)),
        "get_income_docs":      lambda a: store.get_income_docs(a["applicant_id"]),
        "get_property_docs":    lambda a: store.get_property_docs(a["applicant_id"]),
        "list_loan_packages":   lambda a: store.list_loan_packages(),
        "get_sora_rate":        lambda a: _sora_rate_payload(),
        "search_policy":        lambda a: _agentic_search_policy(a["query"], a.get("k", 4)),
        "compare_packages":     lambda a: compare_packages(
            packages              = a["packages"],
            borrowers             = _normalise_borrowers(a["borrowers"]),
            property_type         = a["property_type"],
            n_outstanding_loans   = a["n_outstanding_loans"],
            n_props_owned         = a["n_props_owned"],
            monthly_car_loan      = a["monthly_car_loan"],
            monthly_other         = a["monthly_other"],
            cash_cpf_available    = a.get("cash_cpf_available"),
            target_property_price = a.get("target_property_price"),
        ),
        "calculate_loan":       lambda a: calculate_loan(
            borrowers             = a["borrowers"],
            property_type         = a["property_type"],
            n_outstanding_loans   = a["n_outstanding_loans"],
            n_props_owned         = a["n_props_owned"],
            interest_rate_pct     = a["interest_rate_pct"],
            monthly_car_loan      = a["monthly_car_loan"],
            monthly_other         = a["monthly_other"],
            cash_cpf_available    = a.get("cash_cpf_available"),
            target_property_price = a.get("target_property_price"),
        ),
    }
    if name not in fn_map:
        result = {"error": f"Unknown tool: {name}"}
        print(f"  🔧 {name}({args})  →  ERROR: unknown tool")
        return json.dumps(result)
    try:
        result = fn_map[name](args)
        result_str = json.dumps(result, default=str, ensure_ascii=False)
        preview = result_str if len(result_str) <= 200 else result_str[:200] + "…"
        print(f"  🔧 {name}({json.dumps(args)})  →  {preview}")
        return result_str
    except Exception as exc:
        print(f"  🔧 {name}({args})  →  EXCEPTION: {exc}")
        return json.dumps({"error": str(exc)})


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — LangChain @tool wrappers
#
# These are the functions exposed to LangGraph agents via bind_tools().
# Each wrapper calls the corresponding DataStore method or calculate_loan.
# TOOLS_BY_NAME maps name → tool object; used in the notebook to build agent LLMs.
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def get_profile(applicant_id: str) -> dict:
    'Fetch basic demographic profile for a borrower (name, NRIC, DOB, citizenship, education).'
    return store.get_profile(applicant_id)

@tool
def get_loan_application(applicant_id: str) -> dict:
    'Fetch the raw loan application: income, property details, available funds, interest rate, obligations.'
    return store.get_loan_application(applicant_id)

@tool
def get_bank_credit(applicant_id: str) -> dict:
    'Fetch CBS credit score, risk grade, bank account details, total debt, bankruptcy/default flags.'
    return store.get_bank_credit(applicant_id)

@tool
def get_cpf_history(applicant_id: str, months: int = 6) -> list:
    'Fetch CPF contribution history (OA/SA/MA balances) for the last N months.'
    return store.get_cpf_history(applicant_id, months)

@tool
def get_income_docs(applicant_id: str) -> list:
    'Fetch OCR-extracted income documents (payslips, NOA, commission statements) with verification status.'
    return store.get_income_docs(applicant_id)

@tool
def get_property_docs(applicant_id: str) -> list:
    'Fetch property documents (OTP, CPF withdrawal, option-of-sale) for a borrower application.'
    return store.get_property_docs(applicant_id)

# Common alternative key names the LLM sometimes uses → canonical names.
_BORROWER_KEY_ALIASES = {
    "income": "monthly_income",
    "monthly_salary": "monthly_income",
    "salary": "monthly_income",
    "age_years": "age",
    "citizenship": "nationality",
    "citizen_status": "nationality",
}

def _normalise_borrowers(raw: list) -> list[dict]:
    out = []
    for b in raw:
        if not isinstance(b, dict):
            continue
        normed = {}
        for k, v in b.items():
            canon = _BORROWER_KEY_ALIASES.get(k, k)
            normed[canon] = v
        # `nationality` drives only the ABSD rate, and the model omits it often enough
        # on the compare/reprice path that it crashed a live turn (2026-08-04:
        # "KeyError: 'nationality'"). Default to Singapore Citizen, matching what
        # calculate_loan_tool already assumes when the profile has no citizenship —
        # so the two entry points cannot price the same borrower differently.
        # NOT defaulted here: age and monthly_income. Guessing those would put an
        # invented number into a TDSR the RM acts on; the calculator rejects them with
        # a named message instead ("age is missing (None)").
        if not normed.get("nationality"):
            normed["nationality"] = "Singapore Citizen"
        out.append(normed)
    return out


def _tool_errors_as_result(fn):
    """Turn a bad-input exception into ``{"error": ...}`` the agent can read.

    The @tool wrappers are called by LangGraph's ToolNode, which does NOT catch: an
    exception raised inside one propagates out of the graph and ends the turn, and
    the RM sees a stack-trace fragment. That is the wrong failure mode for INPUT the
    model itself supplied — "age is missing (None)" is something the agent can fix by
    re-calling with the field, but only if it is handed back as a result instead of
    thrown. (The legacy execute_tool path has had this guard since the start; the
    @tool wrappers never got it, which is how KeyError:'nationality' reached the RM
    on 2026-08-04.)

    Deliberately narrow: ValueError / KeyError / TypeError / IndexError are the
    malformed-argument family. Anything else still propagates, because a bug we have
    not characterised must stay loud rather than be reported to the model as if the
    RM had typed something wrong.
    """
    import functools

    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            detail = str(exc).strip("'\"") or type(exc).__name__
            # ASCII only, and guarded: this repo's console is GBK, where printing a
            # non-CP936 glyph raises UnicodeEncodeError — which would escape THIS
            # handler and re-crash the turn we are here to rescue. Same trap as the
            # emoji in utils/llm.py's context-limit logger.
            try:
                print(f"  [tool] {fn.__name__} bad input: {detail}")
            except Exception:  # noqa: BLE001 — console encoding must not gate the result
                pass
            return {"error": f"{detail}. Check the arguments and call the tool again "
                             f"with the missing or corrected value."}
    return _wrapped


@tool
@_tool_errors_as_result
def calculate_loan_tool(
    state: _Annotated[dict, InjectedState],
    property_type: _Opt[str] = None,
    n_outstanding_loans: _Opt[int] = None,
    n_props_owned: _Opt[int] = None,
    interest_rate_pct: _Opt[float] = None,
    monthly_car_loan: _Opt[float] = None,
    monthly_other: _Opt[float] = None,
    cash_cpf_available: _Opt[float] = None,
    target_property_price: _Opt[float] = None,
) -> dict:
    '''MAS-compliant loan calculator. Always call this instead of manual arithmetic.

    The borrower (age, income, nationality) and this case's own figures are read from
    the case record automatically — you do NOT pass them and you cannot choose them.
    Every argument here is an OPTIONAL override for a what-if: pass one only when the RM
    asks for a different scenario (e.g. interest_rate_pct=1.75 for "what if we priced it
    at 1.75%", target_property_price for a different property). Omit everything to price
    the case exactly as it stands, which is what a plain "what is the instalment"
    question wants.

    Returns monthly_repayment (contract rate), monthly_repayment_stress (4% MAS stress
    rate), tdsr_pct, eligible_loan, loan_tenure_years and the rest of the case figures.'''
    # WHY the borrower is injected rather than passed: an LLM given a free-text
    # `borrowers` list fills in a plausible-looking persona instead of the real one —
    # measured 07-30, the model passed age 35 for borrowers who were 38 and 58, which
    # moved the tenure (65 - age), the eligible loan and the instalment by hundreds of
    # dollars while every surrounding figure stayed right. Same defect class as the
    # 07-10 hallucinated instalment, and the same fix: identity comes from state, not
    # from the model. Overrides stay open because a what-if is a legitimate RM request.
    state = state or {}
    applicant_id = state.get("applicant_id", "")
    loan_app = store.get_loan_application(applicant_id)
    profile = store.get_profile(applicant_id)
    if not isinstance(loan_app, dict) or "error" in loan_app:
        return {"error": f"no loan application on file for applicant '{applicant_id}'"}

    fixed = float(loan_app.get("monthly_fixed_income") or 0)
    var = float(loan_app.get("monthly_variable_income") or 0)
    prop_raw = (loan_app.get("property_type") or "").lower()
    prop_default = "Private" if any(k in prop_raw for k in ("private", "condo", "executive")) else "HDB"

    def pick(override, fallback):
        """An override only counts when the model actually supplied one."""
        return fallback if override is None else override

    try:
        return calculate_loan(
            # Identity: from the case record, never from the model.
            borrowers=[{
                "age": (profile or {}).get("age") or 35,
                "monthly_income": fixed + var * 0.7,   # MAS 30% haircut on variable
                "nationality": (profile or {}).get("citizenship") or "Singapore Citizen",
            }],
            property_type=pick(property_type, prop_default),
            n_outstanding_loans=pick(n_outstanding_loans,
                                     int(loan_app.get("no_outstanding_home_loans") or 0)),
            # Already counts this purchase — do not add 1 (see the 07-13 correction).
            n_props_owned=pick(n_props_owned,
                               int(loan_app.get("no_sg_properties_owned") or 1) or 1),
            interest_rate_pct=pick(interest_rate_pct,
                                   float(loan_app.get("interest_rate_pct") or 3.5)),
            monthly_car_loan=pick(monthly_car_loan,
                                  float(loan_app.get("monthly_car_loan") or 0)),
            monthly_other=pick(monthly_other,
                               float(loan_app.get("monthly_other_commitments") or 0)),
            cash_cpf_available=cash_cpf_available,
            target_property_price=pick(target_property_price,
                                       loan_app.get("property_value_estimated")),
        )
    except (ValueError, TypeError, KeyError) as exc:
        # Return calculator errors as a structured result the agent can read and act
        # on rather than letting the exception bubble up through ToolNode and abort
        # the whole graph run.
        return {"error": str(exc)}

@tool
def list_loan_packages() -> list:
    """List the bank's own loan-package catalog (fixed & floating products) with
    indicative rates, lock-in and notes. Bank-wide rates, uniform for all borrowers.
    Call this to get our packages' interest rates before comparing them."""
    return store.list_loan_packages()


@tool
def get_sora_rate() -> dict:
    """Fetch the CURRENT live 3-month compounded SORA from MAS and derive the bank's
    live floating rate (= 3M SORA + floating package spread). Floating packages re-price
    every 3 months off SORA, so call this for the up-to-date floating rate before a
    fixed-vs-floating comparison. Returns sora_3m_pct, as_of, spread and floating_rate_pct."""
    return _sora_rate_payload()


# Reciprocal Rank Fusion constant (Cormack et al., 2009). 60 is the paper's value
# and the de-facto standard; it damps the top ranks so a single list cannot
# dominate the fused order on its own.
_RRF_K = 60
# Each query retrieves this many candidates before fusion. It must exceed the
# caller's k: fusing two k-length lists into k slots means every hit the rewrite
# contributes *evicts* one of the original's, so the original's rank-4/5 answers
# fall off the end. Measured: that eviction alone cost verbatim -1.8% Hit@5. A
# deeper pool lets both lists be fully represented and lets RRF, not truncation,
# decide the final order.
_FUSION_POOL = 10
# Fused candidates handed to the listwise reranker. The 07-17 miss analysis put
# every rescuable clause at fused rank 13-19 (the rest were absent from the pool
# entirely — a vocabulary problem no reranker can fix), so the pool must reach
# past 19. Two _FUSION_POOL-deep lists union to at most 20, which is exactly
# this window.
_RERANK_POOL = 20
# Whether production search_policy reranks by default. Started False; flipped on
# the measured 07-17 win (same rule the rewrite had to obey): overall Hit@5
# 83.8%→93.9%, verbatim 94.5%→100%, paraphrase 53.3%→86.7%, colloquial
# 75.0%→83.3%, 12 rescues / 1 regression. Cost: one extra LLM call per search.
_RERANK_DEFAULT = True


def _rrf_fuse(rankings: list[list[dict]], k: int) -> list[dict]:
    """Reciprocal Rank Fusion of several ranked hit-lists into one top-k.

    score(chunk) = Σ over lists of 1 / (_RRF_K + rank), rank 1-indexed.

    Rank-based, not score-based, on purpose: BM25 scores from two *different*
    queries are on incomparable scales (different IDF mass), so summing or
    normalising them is unsound — but their ranks are comparable. A clause that
    both the original and the rewritten query surface gets two contributions and
    is promoted, which is exactly the consensus signal we want.
    """
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}
    for ranked in rankings:
        for rank, hit in enumerate(ranked, start=1):
            key = hit["text"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            best.setdefault(key, hit)
    top = sorted(scores, key=lambda t: scores[t], reverse=True)[:k]
    return [best[t] for t in top]


def _record_rewrite(fired: bool, fused: bool, set_iter: bool) -> None:
    """Report one search_policy call's rewrite activity to the telemetry probe.

    Best-effort and import-local: retrieval must never break because a metrics
    sink is unavailable (e.g. a unit test importing tools without telemetry), so
    any failure is swallowed. Observation only — it reads no state and returns
    nothing to the retrieval path.
    """
    try:
        from utils.telemetry.metrics import rewrite_probe
        rewrite_probe.record(fired=fired, fused=fused, set_iter=set_iter)
    except Exception:  # noqa: BLE001 — telemetry must not affect retrieval
        pass


def _agentic_search_policy(query: str, k: int = 4, llm=None, iterate: bool = False,
                           rerank: bool | None = None,
                           _rewritten: str | None = None, _is_set: bool | None = None,
                           _rerank_order: list[int] | None = None) -> list:
    """Agentic policy retrieval: rewrite the query into T&C vocabulary, retrieve with
    BOTH the original and the rewritten query, and fuse the rankings by rank.

    **Why rewrite.** The retrieval eval breaks Hit@k down by query style, and BM25's
    word-overlap assumption fails exactly where you would predict — verbatim questions
    (which reuse the contract's own wording) retrieve at ~98%, colloquial ones at ~46%.
    "my place isn't built yet" shares zero tokens with the clause about "Building Under
    Construction". The index is fine; the query is wrong. So an LLM restates it in the
    document's register first.

    **Why fuse rather than replace.** The reference walkthrough throws the original
    query away and retrieves only with the rewrite. We cannot: verbatim queries already
    retrieve at 98%, so betting them on the rewrite has far more downside than upside.
    Searching both and fusing by rank keeps the original's evidence in play, so the
    rewrite becomes an *additional* route to the clause rather than a replacement bet.

    **`iterate=True` — the set-question path.** Some questions are answered by a whole
    enumerated list: "what deals don't work with this promo" needs clauses 1.3.1(a)
    through (e). For those, and only those, a second pass searches the document's own
    enumeration vocabulary so the sibling sub-clauses are pulled into the fusion and
    rank above unrelated noise. See utils.query_rewrite.is_set_question for why the
    decision is made on the QUESTION (one cheap, cacheable call) rather than on the
    retrieved chunks the way the reference's per-round `_needs_more_context` does.

    **`rerank` — the ordering fix.** The 07-17 miss analysis showed the remaining
    fixable failures are ORDERING failures: the right clause is in the fused pool at
    rank 6-19, but BM25's token-overlap score cannot lift it into the top k. A listwise
    LLM rerank of the top-`_RERANK_POOL` fused candidates makes the semantic judgement
    BM25 can't ("which of these ANSWERS the question"), then the top k are returned as
    usual. Fail-open: a failed rerank keeps the fused order. `rerank=None` means "use
    the module default `_RERANK_DEFAULT`".

    **The tool always returns exactly k hits, in every mode.** A set-question does not
    get a wider result window — that would let the iterate arm answer a top-3 question
    with 5 chunks and score a rigged Hit@3 against the baseline. Iteration must earn its
    keep by RANKING the right clauses into the same k, which Recall@k is the metric for.
    """
    from utils.query_rewrite import is_set_question, rerank_order, rewrite_query

    # `_rewritten` / `_is_set` let a caller supply already-computed LLM results (the
    # eval harness caches both on disk; re-deriving them per run would be slow and paid
    # for). None means "compute it here".
    rewritten = _rewritten if _rewritten is not None else rewrite_query(query, llm=llm)

    # Did the rewrite produce something usable (non-empty and different from the
    # original)? This is the signal §6.4 tracks: a "fired" rewrite is the lever,
    # a no-op is the baseline path. Recorded to the thread-local probe below once
    # the branch is decided — observation only, it changes no returned result.
    fired = bool(rewritten and rewritten.strip().lower() != query.strip().lower())

    # An injected order implies the caller (the eval harness) already decided to
    # rerank and paid for the judgement; otherwise fall back to the module default.
    want_rerank = (_RERANK_DEFAULT if rerank is None else rerank) or _rerank_order is not None
    pool_k = _RERANK_POOL if want_rerank else k

    rankings = [_search_policy(query, _FUSION_POOL)]
    if fired:
        rankings.append(_search_policy(rewritten, _FUSION_POOL))

    set_iter = False
    if iterate:
        is_set = _is_set if _is_set is not None else is_set_question(query, llm=llm)
        if is_set:
            set_iter = True
            # A third ranking, aimed at the enumeration itself. The rewrite already
            # says WHAT the list is about; this pass adds the document's structural
            # vocabulary for the list, so siblings the topical queries ranked low get
            # a second reciprocal-rank contribution and rise into the top k.
            rankings.append(_search_policy(f"{rewritten or query} exclusions applies "
                                           f"shall not apply following", _FUSION_POOL))

    # `fused` = more than one ranking went into RRF, i.e. the rewrite (or the set
    # pass) actually contributed to the final order rather than no-op'ing away.
    _record_rewrite(fired=fired, fused=len(rankings) > 1, set_iter=set_iter)
    pool = _rrf_fuse(rankings, pool_k) if len(rankings) > 1 else rankings[0][:pool_k]

    if want_rerank and len(pool) > k:
        order = _rerank_order if _rerank_order is not None \
            else rerank_order(query, [h["text"] for h in pool], llm=llm)
        if order:                            # None = rerank failed open → fused order stands
            pool = [pool[i] for i in order if 0 <= i < len(pool)]
    return pool[:k]


@tool
def search_policy(query: str, k: int = 4) -> list:
    """Retrieve the bank's product / promotion policy clauses (home-loan T&C,
    sign-up gift tiers, eligibility exclusions) from the official documents.
    Use to confirm product terms / promotion eligibility with the EXACT wording
    instead of relying on memory. Returns the most relevant clauses, each with
    its source document and clause title to cite; empty list if nothing matches."""
    return _agentic_search_policy(query, k)


def _figures_for(applicant_id: str, overrides: dict | None = None) -> dict:
    """Re-derive this case's loan figures straight from the CSV row + calculator.

    The fallback for a drafting turn that carries no figures in state — a bare
    "draft the letter" request never routes through the agents that call
    calculate_loan, and `payload` does not survive across turns. Deterministic and
    LLM-free: same inputs, same calculator, same numbers the UI's KPI card shows
    (server.case_service._priced_scenario derives them the same way). Returns {} if
    the case has no usable loan row, so the caller can refuse rather than invent.
    """
    overrides = overrides or {}
    loan_app = store.get_loan_application(applicant_id)
    profile = store.get_profile(applicant_id)
    if not isinstance(loan_app, dict) or "error" in loan_app:
        return {}
    price = overrides.get("target_property_price") or loan_app.get("property_value_estimated")
    if not price:
        return {}

    fixed = float(loan_app.get("monthly_fixed_income") or 0)
    var = float(loan_app.get("monthly_variable_income") or 0)
    prop_raw = (loan_app.get("property_type") or "").lower()
    prop = "Private" if any(k in prop_raw for k in ("private", "condo", "executive")) else "HDB"
    try:
        calc = calculate_loan(
            borrowers=[{
                "age": (profile or {}).get("age") or 35,
                "monthly_income": fixed + var * 0.7,   # MAS 30% haircut on variable
                "nationality": (profile or {}).get("citizenship") or "Singapore Citizen",
            }],
            property_type=prop,
            n_outstanding_loans=int(loan_app.get("no_outstanding_home_loans") or 0),
            n_props_owned=int(overrides.get("n_props_owned")
                              or loan_app.get("no_sg_properties_owned") or 1) or 1,
            interest_rate_pct=float(overrides.get("interest_rate_pct")
                                    or loan_app.get("interest_rate_pct") or 3.5),
            monthly_car_loan=float(loan_app.get("monthly_car_loan") or 0),
            monthly_other=float(loan_app.get("monthly_other_commitments") or 0),
            target_property_price=float(price),
        )
    except Exception:
        return {}
    if not isinstance(calc, dict) or "error" in calc:
        return {}
    return {k: v for k, v in calc.items() if k != "calculation_steps"}


@tool
def draft_letter(
    stage: str,
    state: _Annotated[dict, InjectedState],
) -> dict:
    '''Register the customer's letter for rendering. Call this ONCE, RIGHT BEFORE
    you write the letter body.

    - stage: "IPA" (In-Principle Approval) or "LO" (Letter of Offer).

    You do NOT pass any figures, and you do NOT choose the applicant. The PDF's
    "Indicative Terms" table is filled from the case's own cleared calculation, so
    it always matches the calculator exactly. This tool RETURNS those figures — use
    them, unchanged, when you write the body. Never compute a monthly instalment, an
    LTV percentage, or any other number yourself: read them off what this returns.

    Do not pass the letter text here either. After calling this tool, write the body
    as your final answer; the system renders the PDF from that answer plus the
    figures.

    '''
    from utils import letter_store

    stage_u = (stage or "").upper()
    # applicant_id and the figures come from graph state, never from the model.
    # `state` is an InjectedState argument: hidden from the tool schema the LLM
    # sees, so the LLM can neither pick the applicant nor type a number.
    state = state or {}
    applicant_id = state.get("applicant_id", "")
    payload = state.get("payload") or {}
    # Preferred: the figures an upstream agent's calculate_loan call already
    # produced this turn. Fallback: re-derive them from the CSV row (a standalone
    # drafting request never runs an agent that calls the calculator).
    cleared = payload.get("figures") or _figures_for(
        applicant_id, state.get("overrides") or payload.get("overrides"))
    profile = store.get_profile(applicant_id)
    loan_app = store.get_loan_application(applicant_id)
    prop_docs = store.get_property_docs(applicant_id)
    name = profile.get("full_name", applicant_id) if isinstance(profile, dict) else applicant_id
    nric = profile.get("fake_nric_fin", "") if isinstance(profile, dict) else ""
    # Property one-liner from the OTP doc, best-effort.
    detail = ""
    otp = next((d for d in prop_docs if isinstance(d, dict) and d.get("doc_type") == "OTP"), None) \
        if isinstance(prop_docs, list) else None
    if otp:
        parts = [otp.get("property_address"),
                 f"{otp.get('floor_area_sqm')}sqm" if otp.get("floor_area_sqm") else None,
                 otp.get("tenure")]
        detail = " · ".join(str(p) for p in parts if p)
    elif isinstance(loan_app, dict):
        detail = loan_app.get("property_type") or ""

    # Map the calculator's own keys onto the terms-table fields. Nothing is
    # recomputed here and nothing is read from the model: `cleared` IS a
    # calculate_loan result dict.
    facts = {
        "loan_amount":       cleared.get("eligible_loan"),
        "property_price":    cleared.get("property_price"),
        "tenure_years":      cleared.get("loan_tenure_years"),
        "interest_rate_pct": cleared.get("interest_rate_pct"),
        "monthly_repayment": cleared.get("monthly_repayment"),
    }
    figures = {k: v for k, v in facts.items() if v is not None}
    if not figures:
        # No calculation and no usable loan row: there is nothing truthful to print.
        return {
            "stage": stage_u,
            "figures": {},
            "status": ("no figures could be derived for this case, so no letter can be "
                       "produced. Tell the RM the case data is incomplete. Do NOT write "
                       "a letter and do NOT invent any number."),
        }

    recipient = {"name": name, "nric": nric, "property_detail": detail}
    # Register recipient + figures; the body is filled in by stream.py from the
    # agent's final answer (kept OUT of the tool call so a long markdown body can
    # never corrupt the tool-call JSON and 400 the next LLM request).
    letter_store.put_body(applicant_id, stage_u, "", recipient, facts)
    return {
        "stage": stage_u,
        "figures": figures,
        # Extra cleared values the body may quote, so the prose never derives one.
        "context": {k: cleared[k] for k in ("ltv_limit_pct", "tdsr_pct", "required_cash_cpf",
                                            "monthly_repayment_stress", "binding_constraint")
                    if k in cleared},
        "status": ("figures registered from the case's own calculation — write the letter "
                   "body now, quoting these numbers verbatim"),
    }


@tool
@_tool_errors_as_result
def compare_packages_tool(
    packages: list,
    borrowers: list,
    property_type: str,
    n_outstanding_loans: int,
    n_props_owned: int,
    monthly_car_loan: float,
    monthly_other: float,
    cash_cpf_available: _Opt[float] = None,
    target_property_price: _Opt[float] = None,
) -> dict:
    '''Compare TWO+ loan packages side-by-side for the SAME borrower case, returning
    per-package figures (monthly repayment, eligible loan, total interest, TDSR) plus
    deltas vs the first (baseline) package. Use for fixed-vs-floating and for
    reprice/retention (our package vs a competitor's quoted rate).
    Each package: {"label": <str>, "interest_rate_pct": <float>, "rate_type"?: "Fixed"|"Floating"}.
    All other fields are the calculate_loan inputs EXCEPT interest_rate_pct (per-package).
    Each borrower: {"age": <years>, "monthly_income": <qualifying SGD/month>, "nationality": <...>}.'''
    return compare_packages(
        packages=packages,
        borrowers=_normalise_borrowers(borrowers),
        property_type=property_type,
        n_outstanding_loans=n_outstanding_loans,
        n_props_owned=n_props_owned,
        monthly_car_loan=monthly_car_loan,
        monthly_other=monthly_other,
        cash_cpf_available=cash_cpf_available,
        target_property_price=target_property_price,
    )

@tool
@_tool_errors_as_result
def interest_savings_tool(
    outstanding_loan:     float,
    current_rate_pct:     float,
    remaining_months:     int,
    convert_after_months: int = 0,
    rate_a_pct:           _Opt[float] = None,
    rate_b_pct:           _Opt[float] = None,
    horizon_months:       _Opt[int] = None,
) -> dict:
    '''Interest saved by converting an EXISTING loan to a cheaper rate, comparing
    converting NOW (rate_a_pct) against converting after convert_after_months
    (rate_b_pct). Use whenever the question is about an outstanding balance and a
    remaining tenure — "what do I save converting to X now versus waiting N months
    for Y", early conversion timing, or the rate gap at which converting stops
    paying off. Runs a month-by-month amortisation simulation.
    This is NOT compare_packages: that one prices a NEW purchase for a borrower
    (eligibility, TDSR, monthly repayment); this one prices the TIMING of a switch
    on a loan that already exists, and needs no borrower or property details.
    Savings are interest only, measured over horizon_months (default:
    convert_after_months + 24). Never estimate these figures yourself: the answer
    depends on the falling balance, so a monthly-repayment difference is not the
    saving.

    ALWAYS pass convert_after_months when the question mentions waiting N months
    ("...versus waiting 3 months for 1.40%"). It is not only scenario 2's start: it
    also sets the comparison WINDOW for BOTH scenarios (default N + 24 months), so
    omitting it silently shortens the window and understates scenario 1 as well.
    Both scenarios must be compared over the same window or the two are not
    comparable.'''
    return interest_savings(
        outstanding_loan=outstanding_loan,
        current_rate_pct=current_rate_pct,
        remaining_months=remaining_months,
        convert_after_months=convert_after_months,
        rate_a_pct=rate_a_pct,
        rate_b_pct=rate_b_pct,
        horizon_months=horizon_months,
    )


# Registry: name → LangChain tool object
TOOLS_BY_NAME: dict = {
    'get_profile':          get_profile,
    'get_loan_application': get_loan_application,
    'get_bank_credit':      get_bank_credit,
    'get_cpf_history':      get_cpf_history,
    'get_income_docs':      get_income_docs,
    'get_property_docs':    get_property_docs,
    'calculate_loan':       calculate_loan_tool,
    'list_loan_packages':   list_loan_packages,
    'get_sora_rate':        get_sora_rate,
    'compare_packages':     compare_packages_tool,
    'interest_savings':     interest_savings_tool,
    'search_policy':        search_policy,
    'draft_letter':         draft_letter,
}

ALL_TOOLS = list(TOOLS_BY_NAME.values())
