"""RM Copilot — the compiled LangGraph agent graph.

Extracted from the notebook (cells 7/11/13) into an importable module so both the
notebook (demo/debug) and the FastAPI server can drive the same graph. Behaviour is
unchanged from the notebook — this is a mechanical extraction.

Public surface used by callers (notebook re-export + FastAPI server):
    graph            — the compiled StateGraph (entry point: graph.stream / graph.invoke)
    CopilotState     — the graph state TypedDict
    AGENTS           — dict[str, AgentDef] of registered agents
    chat             — text-only LLM helper (used by the orchestrator/replan nodes)
    new_state(...)   — build a fresh CopilotState for a turn
    llm, store       — shared LLM client and DataStore (re-exported from utils)

Dependency direction (one-way): utils → graph. The graph imports only from utils/
and langgraph/langchain; it never imports the notebook or the server.
"""

import json
from dataclasses import dataclass
from operator import add
from pathlib import Path
from typing import Annotated

from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from utils import config
from utils.llm import make_llm
from utils.tools import ALL_TOOLS, TOOLS_BY_NAME, store

# Shared LLM client (same configuration the notebook uses).
#
# temperature=0 since 07-30. The 3-trial full run showed 17 of 50 golden queries flipping
# between pass and fail, and 14 of those routed IDENTICALLY all three times — so the
# variance was in generation, not orchestration. Routing (`chat`, below) and the RAG
# rewrite were already pinned at 0; agent generation at 0.2 was the only sampling left,
# and it is the layer that writes the figures an RM would act on. Determinism is worth
# more than phrasing variety in a regulated flow.
#
# This does NOT fix the compliance agent re-deriving figures it was handed (GQ024/GQ025
# fail identically on all 3 trials, so that is a systematic choice, not sampling noise).
llm = make_llm(temperature=0)

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# Budget for one agent's ReAct tool loop. Not enforced per-agent directly: it scales
# the graph-wide `recursion_limit` (MAX_TOOL_ROUNDS * len(AGENTS) + 8), which is the
# only ceiling LangGraph actually applies.
#
# Raised 8 -> 12 on 2026-08-04. Per-agent context segmentation means a downstream
# agent no longer sees what an upstream one already fetched, so it re-fetches: over
# the runs after that change, tool_calls per turn rose (full_ipa 2.9 -> 9.0) and one
# full_ipa_assess reached loop_count=7 against a ceiling of 8. Hitting it does not
# degrade gracefully — LangGraph raises GraphRecursionError and the whole turn is
# lost, after the RM has already waited through it — so the headroom matters more
# than the tighter bound. Re-fetching is cheap here (tools read local CSV), which is
# why the fix is to allow the loop rather than to restore the cross-agent context.
MAX_TOOL_ROUNDS = 12


# ══════════════════════════════════════════════════════════════════════════
# Text-only LLM helper (was notebook cell 7) — used by the orchestrator/replan
# nodes for JSON routing calls.
# ══════════════════════════════════════════════════════════════════════════
def chat(messages_dicts: list[dict], temperature: float = 0.0) -> str:
    lc_messages = []
    for m in messages_dicts:
        role, content = m["role"], m.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=content))
    bound = llm.bind(temperature=temperature)
    resp = bound.invoke(lc_messages)
    return resp.content or ""


# ══════════════════════════════════════════════════════════════════════════
# Agent definitions — loaded from skills/ (was notebook cell 11)
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class AgentDef:
    name: str
    label: str
    color: str
    system: str
    tool_names: list[str]

    @property
    def bound_llm(self):
        tools = [TOOLS_BY_NAME[n] for n in self.tool_names if n in TOOLS_BY_NAME]
        return llm.bind_tools(tools) if tools else llm

    def bound_llm_excluding(self, *exclude: str):
        """Same as bound_llm but with the named tools withheld from the LLM. Used to
        hard-remove a heavy tool from a specific flow (e.g. drop compare_packages in
        the LO drafting flow, where a free-tier model tends to emit malformed JSON
        args for it) — the model can't call what it can't see."""
        drop = set(exclude)
        tools = [TOOLS_BY_NAME[n] for n in self.tool_names
                 if n in TOOLS_BY_NAME and n not in drop]
        return llm.bind_tools(tools) if tools else llm

    @property
    def tools(self) -> list:
        return [TOOLS_BY_NAME[n] for n in self.tool_names if n in TOOLS_BY_NAME]


def _load_skill(name: str) -> tuple[str, str, str]:
    cfg = json.loads((SKILLS_DIR / name / "config.json").read_text(encoding="utf-8"))
    system = (SKILLS_DIR / name / "skill.md").read_text(encoding="utf-8").strip()
    return cfg["label"], cfg["color"], system


# Tool access list per agent — controls what each agent can call.
_AGENT_TOOLS: dict[str, list[str]] = {
    "borrower_profile": [
        "get_profile", "get_loan_application", "get_bank_credit",
        "get_cpf_history", "get_income_docs", "calculate_loan",
    ],
    "property_analysis": [
        "get_loan_application", "get_property_docs", "get_cpf_history", "calculate_loan",
    ],
    "compliance_validation": [
        "get_loan_application", "get_bank_credit", "get_income_docs",
        "get_cpf_history", "get_property_docs", "calculate_loan",
    ],
    # ── Global Shared Service: document completeness / OCR confidence check ──
    "document_validation": [
        "get_income_docs", "get_property_docs",
    ],
    # ── Stage LO: loan product & pricing selection ──
    "policy_product": [
        "get_loan_application", "get_profile", "get_bank_credit", "calculate_loan",
        "list_loan_packages", "compare_packages", "search_policy",
    ],
    # ── Stage REPRICE: reprice / retention vs a competitor (standalone advisory) ──
    # Also serves the standalone COMPARE track (Package Comparison panel), whose
    # question is the same one: an existing loan, and when to switch it.
    "reprice_retention": [
        "get_loan_application", "get_profile", "get_bank_credit",
        "list_loan_packages", "compare_packages", "interest_savings",
    ],
    # ── Global Shared Service: drafts IPA letter / Letter of Offer ──
    "document_drafting": [
        "calculate_loan",
        "draft_letter",
    ],
    # ── Customer self-service lane (toC): affordability calculator + T&C search.
    # Deliberately NO data-access tools (get_profile / get_loan_application / …):
    # the model can't call what it can't see, so a customer session structurally
    # cannot read any applicant's records — the tool allowlist IS the boundary,
    # not the prompt. list_loan_packages is the public rate catalog, not case data.
    "customer_assistant": [
        "calculate_loan", "search_policy", "list_loan_packages",
    ],
}

AGENTS: dict[str, AgentDef] = {}
for _name, _tool_names in _AGENT_TOOLS.items():
    try:
        _label, _color, _system = _load_skill(_name)
        AGENTS[_name] = AgentDef(
            name=_name, label=_label, color=_color,
            system=_system, tool_names=_tool_names,
        )
    except FileNotFoundError as e:
        print(f"  [SKIP] {_name} — missing file: {e}")


# ── LogEntry (for the audit panel) ────────────────────────────────────────
@dataclass
class LogEntry:
    kind: str
    payload: dict


# ══════════════════════════════════════════════════════════════════════════
# CopilotState
# ══════════════════════════════════════════════════════════════════════════
class CopilotState(TypedDict):
    messages: Annotated[list, add_messages]
    applicant_id: str
    role: str    # 'rm' (workbench) | 'customer' (self-service portal)
    intent: str
    stage: str   # 'IPA' | 'LO' | 'REPRICE' | 'none'
    route: str   # agent node name chosen by orchestrator
    payload: dict  # A2A payload accumulator
    audit: Annotated[list, add]
    hitl_decision: str  # '' | 'approved' | 'rejected' (set by hitl_review)
    hitl_feedback: str  # RM revision notes fed back to document_drafting
    replan_kind: str   # '' | 'recompute' | 'redraft' | 'reject_unchangeable' (set by replan)
    replan_msg: str   # replan's one-line message to the RM (reject_unchangeable)
    overrides: dict  # RM-approved recompute inputs, e.g. {'interest_rate_pct': 3.2}
    current_agent: str  # agent whose ReAct tool loop is active (tool-return routing)


def new_state(applicant_id: str = "", user_text: str = "", role: str = "rm") -> CopilotState:
    """Build a fresh CopilotState for a single turn so callers don't hand-build
    the (large) initial dict. Pass the RM's message as user_text."""
    return {
        "messages": [HumanMessage(content=user_text)] if user_text else [],
        "applicant_id": applicant_id,
        "role": role or "rm",
        "intent": "",
        "stage": "",
        "route": "",
        "payload": {},
        "audit": [],
        "hitl_decision": "",
        "hitl_feedback": "",
        "replan_kind": "",
        "replan_msg": "",
        "overrides": {},
        "current_agent": "",
    }


# ══════════════════════════════════════════════════════════════════════════
# Orchestrator node
# ══════════════════════════════════════════════════════════════════════════
def _load_orc_system() -> str:
    return (SKILLS_DIR / "orchestrator" / "skill.md").read_text(encoding="utf-8").strip()


def _full_flow_entry(token: str) -> str:
    """Map a full-flow pseudo-agent token to the real node it enters at. Uses a
    substring test so the assessment variants ('full_ipa_assess'/'full_lo_assess')
    resolve to the same entry node as their drafting counterparts — the chains are
    identical up to compliance; only the post-compliance hop differs. A non-full
    token is returned unchanged (it is a real agent name or 'none')."""
    t = (token or "").lower()
    if "full_ipa" in t:
        return "borrower_profile"
    if "full_lo" in t:
        return "policy_product"
    return token


def _prior_turn_context(messages: list) -> str:
    """A compact summary of the PREVIOUS turn, so the orchestrator can resolve a
    short follow-up ("change it to 1.6%", "compare on floating") that omits the
    applicant/stage. The orchestrator only sees the latest user message, so we
    surface the prior RM message + the prior assistant reply (truncated) as
    context it can carry forward. Returns '' on the first turn."""
    humans = [m for m in messages if isinstance(m, HumanMessage)]
    if len(humans) < 2:
        return ""   # first turn — nothing prior to carry
    prev_user = humans[-2].content
    prev_reply = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            prev_reply = m.content
            break
    prev_reply = (prev_reply[:400] + "…") if len(prev_reply) > 400 else prev_reply
    parts = [f"Previous RM message: {prev_user}"]
    if prev_reply:
        parts.append(f"Previous assistant reply (excerpt): {prev_reply}")
    return "\n".join(parts)


def _parse_route(raw: str) -> dict:
    """Parse the orchestrator's JSON routing decision, tolerating stray noise
    around the object (e.g. a leading "SS"). Tries a direct parse, then the first
    balanced {...} block. On failure returns a route to 'none' with a CLEAN
    fallback answer — never the raw model text, which must not reach the RM."""
    for candidate in (raw, _first_json_object(raw)):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {"agent": "none", "stage": "none", "intent": "unknown", "applicant_id": "",
            "answer": "I couldn't parse that request. Could you rephrase what you'd like me to do?"}


def _first_json_object(s: str) -> str:
    """Return the first brace-balanced {...} substring in s, or '' if none. Keeps
    us from tripping on leading/trailing junk the model occasionally prepends."""
    start = s.find("{")
    if start < 0:
        return ""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return ""


def orchestrator_node(state: CopilotState) -> dict:
    # ── Customer lane: the route is FIXED, not an LLM decision. A customer
    # session must never be able to reach the RM agents (and their data tools),
    # so we short-circuit before any model sees the message — routing here is a
    # constraint, and "a prompt is not a constraint". The one audit entry keeps
    # the node visible in the audit-completeness KPI.
    if state.get("role") == "customer":
        note = "Customer session — fixed route to Customer Assistant (no LLM routing)."
        audit = [LogEntry("a2a", {"role": "assistant", "from": "Orchestrator",
                                  "to": "Customer Assistant", "content": note})]
        return {
            "messages": [],
            "applicant_id": "",          # a customer session is never bound to a case
            "intent": "customer_selfserve",
            "stage": "none",
            "route": "customer_assistant",
            "payload": state.get("payload", {}),
            "audit": [vars(e) for e in audit],
        }

    sys_prompt = _load_orc_system()
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    user_text = last_user.content if last_user else ""

    # Carry prior-turn context so a short follow-up keeps the same stage/applicant.
    prior = _prior_turn_context(state["messages"])
    routing_input = (f"[Conversation so far]\n{prior}\n\n[Current RM message]\n{user_text}"
                     if prior else user_text)

    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": routing_input}]
    audit_entries = [
        LogEntry("a2a", {"role": "system", "from": "System", "to": "Orchestrator", "content": sys_prompt}),
        LogEntry("a2a", {"role": "user", "from": "RM", "to": "Orchestrator", "content": user_text}),
    ]

    raw = chat(msgs, temperature=0.0)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    audit_entries.append(LogEntry("a2a", {"role": "assistant", "from": "Orchestrator", "to": "Router", "content": raw}))

    # The routing output is JSON, but the model sometimes emits leading/trailing
    # noise (e.g. a stray "SS" before the object). Carve out the first {...} block
    # before parsing so that noise doesn't break json.loads — and, crucially, is
    # never surfaced to the RM. On a genuine parse failure we fall back to a CLEAN
    # message, never the raw model text (which would leak the routing JSON into the
    # chat bubble).
    decision = _parse_route(raw)

    agent = decision.get("agent", "none")
    app_id = decision.get("applicant_id", state.get("applicant_id", ""))
    intent = decision.get("intent", "")
    stage = decision.get("stage", "none")
    print(f"[Orchestrator] agent={agent}  stage={stage}  applicant={app_id}")

    # ── Full-flow pseudo-agents: enter the chain at its first real node ──────
    # 'full_ipa'(_assess) starts at borrower_profile; 'full_lo'(_assess) starts at
    # policy_product. The literal token is preserved in `route` so the per-agent
    # routers can detect the full flow (and assess-vs-draft) and chain accordingly.
    first_node = _full_flow_entry(agent)

    extra_msgs = []
    if first_node not in AGENTS:
        answer = decision.get("answer", "I am not sure how to help with that.")
        extra_msgs = [AIMessage(content=answer)]
        audit_entries.append(LogEntry("a2a", {"role": "assistant", "from": "Orchestrator",
                                              "to": "RM", "content": answer, "is_final": True}))
        audit_entries.append(LogEntry("answer", {"text": answer}))

    return {
        "messages": extra_msgs,
        "applicant_id": app_id or state.get("applicant_id", ""),
        "intent": intent,
        "stage": stage,
        "route": agent,   # keep raw token (may be 'full_ipa'/'full_lo')
        "payload": state.get("payload", {}),
        "audit": [vars(e) for e in audit_entries],
    }


def _route_after_orchestrator(state: CopilotState) -> str:
    route = state.get("route", "none")

    # document_drafting is a renderer, not a producer: it may only be entered
    # directly when an already-cleared drafting payload exists (e.g. supplied by
    # a memory layer, or the RM said the case is already prepared). Otherwise a
    # direct draft route is redirected to the full flow so the functional agents
    # + compliance build and clear the case first. Defence-in-depth: even if the
    # orchestrator LLM emits 'document_drafting', we never skip the chain here.
    if route == "document_drafting" and not state.get("payload", {}).get("cleared"):
        route = "full_ipa" if state.get("stage") == "IPA" else "full_lo"

    entry = _full_flow_entry(route)
    return entry if entry in AGENTS else END


# ══════════════════════════════════════════════════════════════════════════
# Agent node factory
# ══════════════════════════════════════════════════════════════════════════
def make_agent_node(agent_name: str):
    def agent_node(state: CopilotState) -> dict:
        agent = AGENTS[agent_name]
        app_id = state.get("applicant_id", "")
        last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
        question = last_user.content if last_user else state.get("intent", "")
        if app_id and app_id not in question:
            question = f"[Applicant: {app_id}] {question}"

        # Attach LTV fusion context for property_analysis
        payload_update = dict(state.get("payload", {}))
        if agent_name == "property_analysis":
            ltv_ctx = payload_update.get("ltv_fusion", {})
            if ltv_ctx:
                ctx_str = json.dumps(ltv_ctx, ensure_ascii=False)
                question = question + f"\n\nBorrower context (LTV fusion):\n{ctx_str}"

        # Attach the authoritative loan basis for the LO flow. policy_product enters
        # full_lo with no upstream payload, so without this it re-assembles the
        # calculator args itself and gets a DIFFERENT quantum than IPA (wrong). Feed
        # the same reverse-mode basis IPA/the KPI card use so LO agrees with IPA.
        if agent_name == "policy_product" and app_id and _is_full_lo(state):
            basis = _build_lo_basis(app_id)
            if basis:
                payload_update["lo_basis"] = basis
                question = question + (
                    "\n\nAuthoritative case basis — use these EXACT values for any "
                    "`calculate_loan` / `compare_packages` call (only interest_rate_pct "
                    "differs per package); do NOT re-derive or change them:\n"
                    + json.dumps(basis, ensure_ascii=False))

        # Drafting picks IPA-letter vs Letter-of-Offer from the STAGE, which the
        # orchestrator already decided and put in state. Its prompt used to say
        # "infer the stage from the conversation", which was always a weaker signal
        # than the value we hold — and once each agent sees only its own segment
        # (_agent_segment) that upstream conversation is no longer there to infer
        # from. State is the authority; pass it explicitly.
        if agent_name == "document_drafting":
            _stage = (state.get("stage") or "").upper()
            if _stage in ("IPA", "LO"):
                question = question + (
                    f"\n\n[Case stage: {_stage}] Draft the "
                    + ("In-Principle Approval letter" if _stage == "IPA"
                       else "Letter of Offer")
                    + f" and pass stage=\"{_stage}\" to `draft_letter`.")

        # HITL reject loop: drafting is re-entered with the RM's revision
        # notes so it can rewrite the draft per the feedback.
        if agent_name == "document_drafting" and state.get("hitl_feedback"):
            question = question + f'\n\n[RM revision request]: {state["hitl_feedback"]}'

        # HITL replan -> recompute: the producing agent re-calls calculate_loan
        # with the RM's overridden inputs (e.g. a new market interest rate) so the
        # numbers are recomputed by the calculator, not hand-edited in the draft.
        _ov = state.get("overrides") or payload_update.get("overrides")
        if agent_name in ("policy_product", "property_analysis") and _ov:
            question = question + (
                "\n\n[RM override — recompute the case and call calculate_loan with these "
                "EXACT input values]: " + json.dumps(_ov, ensure_ascii=False))

        print(f"\n  AGENT : {agent.label}  QUERY : {question[:100]}")

        audit_entries = [
            LogEntry("a2a", {"role": "system", "from": "Orchestrator", "to": agent.label, "content": agent.system}),
            LogEntry("a2a", {"role": "user", "from": "RM", "to": agent.label, "content": question}),
            LogEntry("a2a", {"role": "routed", "from": "Orchestrator", "to": agent.label, "agent": agent_name}),
        ]
        return {
            "messages": [SystemMessage(content=agent.system), HumanMessage(content=question)],
            "payload": payload_update,
            "audit": [vars(e) for e in audit_entries],
            "current_agent": agent_name,   # tool results route back to this agent's LLM node
        }
    agent_node.__name__ = agent_name
    return agent_node


# ══════════════════════════════════════════════════════════════════════════
# LTV fusion: extract a persona summary from the borrower's tool results.
# Borrower owns persona data only; this summary is handed to property_analysis
# (via state['payload']['ltv_fusion']) so PROPERTY computes the final LTV cap.
# ══════════════════════════════════════════════════════════════════════════
def _collect_tool_results(messages: list) -> dict:
    """Return the most recent dict result per tool name from ToolMessages.

    ToolNode content is JSON text; older results may already be dicts. Walk
    forward so later calls overwrite earlier ones for the same tool name.
    """
    out: dict[str, dict] = {}
    for m in messages:
        if isinstance(m, ToolMessage):
            name = getattr(m, "name", "") or ""
            val = m.content
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:
                    continue
            if isinstance(val, dict):
                out[name] = val
            elif isinstance(val, list) and val and isinstance(val[0], dict):
                out[name] = val[0]   # e.g. cpf history: take latest record
    return out


def _build_ltv_fusion(messages: list) -> dict:
    """Persona summary for the LTV fusion hand-off. Best-effort; missing
    fields are simply omitted so property_analysis can still proceed."""
    res = _collect_tool_results(messages)
    profile = res.get("get_profile", {})
    loan = res.get("get_loan_application", {})
    cpf = res.get("get_cpf_history", {})

    fusion: dict = {}
    # Scalar persona fields (column names follow the csv_tables schema).
    for key, rec, field in [
        ("age", profile, "age"),
        ("nationality", profile, "citizenship"),
        ("residential_status", profile, "residential_status"),
        ("property_type", loan, "property_type"),
        ("property_value", loan, "property_value_estimated"),
        # n_props_owned INCLUDES this purchase (1 = first-time buyer) — the CSV
        # column is stored in that same convention, so it passes straight through.
        ("n_props_owned", loan, "no_sg_properties_owned"),
        ("n_outstanding_loans", loan, "no_outstanding_home_loans"),
        ("cpf_oa", cpf, "oa_balance_eom"),
    ]:
        if field in rec and rec.get(field) is not None:
            fusion[key] = rec[field]

    # Qualifying income = fixed + variable * 0.7 (MAS 30% haircut on variable
    # income, matching mortgage_planner.py). Applied HERE, where fixed and
    # variable are still split; calculate_loan receives the already-haircut
    # figure as monthly_income and does no further haircut.
    fixed = loan.get("monthly_fixed_income") or 0
    var = loan.get("monthly_variable_income") or 0
    if fixed or var:
        fusion["income"] = fixed + var * 0.7
    return fusion


def _build_lo_basis(app_id: str) -> dict:
    """Structured calculate_loan / compare_packages inputs for the LO flow.

    The IPA flow hands property_analysis a structured `ltv_fusion` payload, so its
    calculator inputs are correct. The full_lo flow enters at policy_product with
    NO such payload, forcing the LLM to re-assemble the calculator args from raw
    fields — which it gets wrong (mixing forward/reverse mode, dropping the declared
    property value), producing a DIFFERENT loan quantum than IPA for the same case.

    This builds the SAME authoritative basis case_service._loan_summary uses
    (reverse mode: price at the declared property value, qualifying income =
    fixed + variable*0.7), so LO and IPA agree. Best-effort: returns {} if the case
    has no usable loan-application row, and the agent falls back to self-assembly."""
    loan = store.get_loan_application(app_id)
    profile = store.get_profile(app_id)
    if not isinstance(loan, dict) or "error" in loan:
        return {}
    price = loan.get("property_value_estimated")
    if not price:
        return {}
    fixed = loan.get("monthly_fixed_income") or 0
    var = loan.get("monthly_variable_income") or 0
    prop_raw = (loan.get("property_type") or "").lower()
    prop = "Private" if any(k in prop_raw for k in ("private", "condo", "executive")) else "HDB"
    age = profile.get("age") if isinstance(profile, dict) and "error" not in profile else None
    nat = (profile.get("citizenship") if isinstance(profile, dict) else None) or "Singapore Citizen"
    return {
        "borrowers": [{
            "age": age or 35,
            "monthly_income": fixed + var * 0.7,   # qualifying income (haircut applied)
            "nationality": nat,
        }],
        "property_type": prop,
        "n_outstanding_loans": int(loan.get("no_outstanding_home_loans") or 0),
        "n_props_owned": int(loan.get("no_sg_properties_owned") or 1),
        "monthly_car_loan": float(loan.get("monthly_car_loan") or 0),
        "monthly_other": float(loan.get("monthly_other_commitments") or 0),
        # Reverse mode: price the case at the declared property value (same as IPA).
        "target_property_price": float(price),
    }


# ══════════════════════════════════════════════════════════════════════════
# Per-agent context window
# ══════════════════════════════════════════════════════════════════════════
def _agent_segment(messages: list) -> list:
    """The messages belonging to the CURRENTLY-RUNNING agent's own turn.

    `messages` uses the `add_messages` reducer, so it only ever APPENDS: in a full
    IPA chain every agent's system prompt, tool traffic and answer stays in the list
    and is re-sent on every later LLM call. Measured, that puts the drafting agent's
    call at ~14.5k input tokens, of which ~10.8k is other agents' material — and on a
    32k endpoint the second such turn simply 400s.

    Each agent node re-seeds its own `SystemMessage` + `HumanMessage` on entry (see
    make_agent_node), so that SystemMessage is a clean segment boundary: everything
    from the LAST one onward is this agent's own working context, and everything
    before it belongs to agents that have already finished. Slicing there is a
    SEMANTIC cut, not a positional one, which is what keeps every tool_call paired
    with its ToolMessage — an orphaned ToolMessage is a hard 400.

    Cross-agent hand-off is unaffected because it does not travel in the transcript:
    it goes through `payload` (ltv_fusion, lo_basis, figures), which each agent node
    appends into its own question. `_collect_tool_results` likewise still reads the
    FULL `state["messages"]`, so the letter's authoritative figures are untouched.
    """
    # Read live off the module so a test (or an A/B run) can flip it mid-process.
    if not config.TRIM_CONTEXT:
        return messages
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], SystemMessage):
            return messages[i:]
    return messages          # no boundary yet (e.g. customer lane): send as-is


def _approx_tokens(messages: list) -> int:
    """Rough token count for the console readout — chars/3.6, the usual BPE ballpark
    for English prose mixed with JSON. Deliberately not a real tokenizer: this is a
    scale readout, not a budget, and must add no dependency or cost to a call path."""
    total = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        total += len(content if isinstance(content, str) else str(content))
        for tc in (getattr(m, "tool_calls", None) or []):
            total += len(str(tc.get("args", "")))
    return int(total / 3.6)


def _log_segment(label: str, segment: list, full: list) -> None:
    """One line per LLM call showing what segmentation saved on THIS call.

    This is the readout the offline eval sees, since the eval drives graph.stream()
    directly and never reaches the server's cross-turn compaction. Printing sent vs.
    total is what turns "did trimming make the model dumber?" into a question you can
    answer: read quality off the answers, cost off this line.
    """
    try:
        sent, total = _approx_tokens(segment), _approx_tokens(full)
        if sent < total:
            print(f"  [context] {label}: sent {sent:,} tok of {total:,} "
                  f"({len(segment)}/{len(full)} messages)")
    except Exception:  # noqa: BLE001 — a telemetry line must never break a call
        pass


# ══════════════════════════════════════════════════════════════════════════
# LLM call node — each agent's bound LLM
# ══════════════════════════════════════════════════════════════════════════
def make_llm_node(agent_name: str):
    def llm_node(state: CopilotState) -> dict:
        agent = AGENTS[agent_name]
        # In the LO DRAFTING flow the package is already chosen — the agent must
        # pick the most competitive one and hand it to drafting, NOT run a package
        # comparison. compare_packages is the heaviest tool (nested arrays) and the
        # one a free-tier model most often emits malformed JSON args for (every 400
        # in eval/runs.csv came from it), so we withhold it entirely here: the model
        # can't call what it can't see. Comparison stays available in the LO ASSESS
        # flow (full_lo_assess) and REPRICE, which do want it.
        if agent_name == "policy_product" and _is_full_lo(state) and not _is_assess_only(state):
            bound = agent.bound_llm_excluding("compare_packages")
        else:
            bound = agent.bound_llm
        # Send only this agent's own segment — see _agent_segment. The full list
        # stays in state for _collect_tool_results and the audit trail.
        _segment = _agent_segment(state["messages"])
        _log_segment(agent.label, _segment, state["messages"])
        resp = bound.invoke(_segment)
        is_final = not bool(getattr(resp, "tool_calls", None))
        audit_entries = []
        if resp.content:
            audit_entries.append(LogEntry("a2a", {
                "role": "assistant", "from": agent.label, "to": "Tool",
                "content": resp.content, "is_final": is_final,
            }))
        for tc in getattr(resp, "tool_calls", []) or []:
            audit_entries.append(LogEntry("tool_call", {
                "name": tc["name"], "args": tc["args"], "call_id": tc["id"],
            }))

        out = {"messages": [resp], "audit": [vars(e) for e in audit_entries]}
        # One shared copy: several branches below write to the payload, and
        # rebuilding it per-branch would let a later branch clobber an earlier one.
        payload_update = dict(state.get("payload", {}))
        payload_dirty = False

        # Freeze the calculator's own output the moment ANY agent finishes a tool
        # loop that called it. These become the ONLY figures the letter may print:
        # `draft_letter` reads them out of state (InjectedState) instead of having
        # the LLM retype them, which is what produced the 2026-07-10 hallucinated
        # instalment (S$2,299 vs the calculator's 4,490) — see mas/eval/evidence/.
        #
        # Harvested here, NOT at the compliance node: `document_drafting` is also
        # reachable standalone (a bare "draft the letter" request never routes
        # through compliance), and a drafting turn that finds no figures cannot
        # print a letter at all.
        if is_final:
            figures = _collect_tool_results(state["messages"]).get("calculate_loan")
            if isinstance(figures, dict) and "monthly_repayment" in figures:
                payload_update["figures"] = {
                    k: v for k, v in figures.items() if k != "calculation_steps"
                }
                payload_dirty = True

        # A2A: compliance marks the payload as cleared so a later (memory-backed)
        # re-render request can legitimately enter document_drafting directly.
        if agent_name == "compliance_validation" and is_final:
            payload_update["cleared"] = True
            payload_dirty = True

        # A2A: borrower writes the LTV fusion payload on its final answer.
        if agent_name == "borrower_profile" and is_final:
            fusion = _build_ltv_fusion(state["messages"])
            if fusion:
                payload_update["ltv_fusion"] = fusion
                payload_dirty = True
                out["audit"].append(vars(LogEntry("a2a", {
                    "role": "assistant", "from": agent.label, "to": "Property Analysis Agent",
                    "content": "LTV fusion payload:\n" + json.dumps(fusion, ensure_ascii=False, indent=2),
                })))

        if payload_dirty:
            out["payload"] = payload_update
        return out
    llm_node.__name__ = f"{agent_name}_llm"
    return llm_node


# ══════════════════════════════════════════════════════════════════════════
# Helpers: full-flow detection (for chaining between stages)
# ══════════════════════════════════════════════════════════════════════════
def _is_full_ipa(state: CopilotState) -> bool:
    route = state.get("route", "")
    intent = state.get("intent", "")
    return state.get("stage") == "IPA" and "full_ipa" in (route + intent).lower()


def _is_full_lo(state: CopilotState) -> bool:
    route = state.get("route", "")
    intent = state.get("intent", "")
    return state.get("stage") == "LO" and "full_lo" in (route + intent).lower()


def _is_full_flow(state: CopilotState) -> bool:
    return _is_full_ipa(state) or _is_full_lo(state)


def _is_recompute(state: CopilotState) -> bool:
    """True while a reject→replan RECOMPUTE is re-running the producing agent.

    On recompute the replan node overwrites state['route'] with the producing
    agent name (e.g. 'property_analysis'), which makes _is_full_ipa / _is_full_lo
    read False — so the producing agent's router must ALSO treat a recompute as a
    full flow, or the case would stop after re-pricing and never re-reach drafting
    (the "letter never reappears after changing the rate" bug). replan_kind stays
    'recompute' for the rest of this resumed run, so it's a reliable signal."""
    return state.get("replan_kind", "") == "recompute"


def _is_redraft(state: CopilotState) -> bool:
    """True while a reject→replan REDRAFT re-runs document_drafting for wording.

    A redraft exists ONLY because the RM rejected a draft at the HITL gate, so the
    redrafted letter MUST go back to the gate — never to END. Like recompute, the
    route token no longer reads as a full flow by this point (replan drives it), so
    the drafting router must treat replan_kind=='redraft' as gate-bound explicitly,
    or the letter silently falls out as a plain answer with no PDF / Approve gate
    (the 'entered replan but no letter comes out' bug)."""
    return state.get("replan_kind", "") == "redraft"


def _is_assess_only(state: CopilotState) -> bool:
    """An assessment-only full flow stops at compliance with an eligibility verdict
    — it does NOT draft a letter or open the HITL gate. The orchestrator emits
    'full_ipa_assess' / 'full_lo_assess' for "just assess eligibility" requests;
    the plain 'full_ipa' / 'full_lo' tokens continue on to drafting + the gate.
    (Note: 'full_ipa_assess' still contains the 'full_ipa' substring, so the shared
    pre-draft chain — borrower → property → docval → compliance — runs identically;
    only the post-compliance hop differs.)"""
    token = (state.get("route", "") + state.get("intent", "")).lower()
    return _is_full_flow(state) and "assess" in token


def _last_msg_has_tool_calls(state: CopilotState) -> bool:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            return bool(getattr(m, "tool_calls", None))
    return False


# ══════════════════════════════════════════════════════════════════════════
# ──── HITL review gate ───────────────────────────────────────────────
# After Document Drafting produces the draft, pause the graph (interrupt) and
# hand the draft to the RM. The RM is the FINAL decision-maker: resume with
# {approved: bool, feedback: str}. Approve -> END; Reject -> back to drafting.
def hitl_review_node(state: CopilotState) -> dict:
    # The draft is the last non-empty assistant message produced by drafting.
    draft = ""
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            draft = m.content
            break
    # Pause here until the front-end resumes. On resume this node re-runs from
    # the top and interrupt(...) returns the resume value, so keep all code
    # before interrupt() side-effect free (above is read-only).
    # replan_msg (set by a prior replan reject_unchangeable) is surfaced to the
    # RM alongside the unchanged draft so they know why nothing was recomputed.
    decision = interrupt({"type": "hitl_review", "draft": draft,
                          "replan_msg": state.get("replan_msg", "")})
    approved = bool(decision.get("approved"))
    feedback = decision.get("feedback", "") or ""
    print(f"[HITL] approved={approved}  feedback={feedback[:60]!r}")
    audit = [vars(LogEntry("a2a", {
        "role": "assistant", "from": "HITL Review Gate", "to": "RM",
        "content": ("✅ RM APPROVED the draft — releasing the letter." if approved
                    else f"↩️ RM REJECTED — revise and re-draft: {feedback}"),
        "is_final": approved,
    }))]
    # Clear replan_msg once surfaced so a stale note isn't reshown on a later
    # gate visit; the new feedback drives the next replan decision.
    return {"hitl_decision": "approved" if approved else "rejected",
            "hitl_feedback": feedback, "replan_msg": "", "audit": audit}


def _route_hitl_review(state: CopilotState) -> str:
    # Approve releases the case; Reject hands the feedback to the replan
    # sub-orchestrator, which decides recompute / re-draft / refuse.
    return END if state.get("hitl_decision") == "approved" else "replan"


# ──── Replan sub-orchestrator ────────────────────────────────────────
# Invoked after a HITL reject. Reads the RM's revision feedback + the current
# case summary and classifies the request: recompute (overridden calculate_loan
# input -> re-run the producing agent), redraft (wording only -> drafting), or
# reject_unchangeable (e.g. tenure -> explain & return to the gate, draft as-is).
def _load_replan_system() -> str:
    return (SKILLS_DIR / "replan" / "skill.md").read_text(encoding="utf-8").strip()


def replan_node(state: CopilotState) -> dict:
    """The system's Disposition Engine, in the sense of MAS SAFR (Safeguards for
    Agentic Finance at Runtime, 2026).

    It sits between a proposed action — release the drafted letter — and its
    execution, and resolves that action to one of a fixed set of dispositions.
    The three `kind`s this node emits map onto SAFR's dispositions:

        reject_unchangeable  -> Deny         (value fixed by regulation; do not execute)
        recompute / redraft  -> Escalate     (send back to a worker to correct, then re-gate)
        (RM approves at gate) -> Auto-Execute (done in hitl_review, not here — letter released)

    The `kind` names are deliberately NOT renamed to SAFR's generic verbs: they name
    the concrete corrective action, which is more specific and self-documenting. The
    mapping is recorded here and in the Design Rationale rather than baked into the
    symbols, so the code never over-claims a 1:1 SAFR implementation. The other SAFR
    components live elsewhere: the deterministic calculator is the Controls Repository
    (utils/calculator.py), and the per-run audit trail is the Audit Log
    (utils/telemetry). The one SAFR component not yet implemented is Agent Identity —
    binding each tool call to the authenticated session principal (see the leakage
    check in utils/telemetry/metrics.py, scoped as future work).
    """
    sys_prompt = _load_replan_system()
    feedback = state.get("hitl_feedback", "") or ""
    # Case summary for context: the cleared payload figures + the latest draft.
    draft = ""
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
            draft = m.content
            break
    case = {"stage": state.get("stage", "none"),
            "applicant_id": state.get("applicant_id", ""),
            "cleared_payload": {k: v for k, v in (state.get("payload") or {}).items()
                                if k not in ("overrides",)}}
    user_text = ("RM revision feedback:\n" + feedback +
                 "\n\nCurrent case summary:\n" + json.dumps(case, ensure_ascii=False, default=str) +
                 "\n\nCurrent draft (excerpt):\n" + draft[:1200])
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}]
    audit = [
        vars(LogEntry("a2a", {"role": "system", "from": "System", "to": "Replan", "content": sys_prompt})),
        vars(LogEntry("a2a", {"role": "user", "from": "RM", "to": "Replan", "content": user_text})),
        vars(LogEntry("a2a", {"role": "routed", "from": "HITL Review Gate", "to": "Replan", "agent": "replan"})),
    ]
    raw = chat(msgs, temperature=0.0)
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    audit.append(vars(LogEntry("a2a", {"role": "assistant", "from": "Replan", "to": "Router", "content": raw})))
    # Tolerate noise around the JSON exactly as the orchestrator does (_parse_route):
    # these models routinely wrap the object in prose or a fence, and a bare
    # json.loads() then throws away a decision that was actually CORRECT. Observed
    # 2026-08-04: "change rate to 1.2%" came back as kind=redraft/overrides={} — not
    # because the model misjudged it, but because its reply did not parse and the
    # fallback below silently rewrote the answer. Same class as the 2026-07-22
    # orchestrator routing-JSON bug, in the node that never got the fix.
    d = None
    for candidate in (raw, _first_json_object(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            d = parsed
            break
    if d is None:
        # Fail safe: treat an unparseable decision as a plain re-draft. Log the raw
        # text — silently swallowing it is what made this failure look like the model
        # ignoring the RM, and left nothing to diagnose from.
        print(f"  [warn] replan reply did not parse; falling back to redraft. raw={raw[:300]!r}")
        d = {"kind": "redraft", "route_to": "document_drafting", "overrides": {}, "message": ""}

    kind = d.get("kind", "redraft")
    route_to = d.get("route_to", "document_drafting")
    overrides = d.get("overrides") or {}
    message = d.get("message", "") or ""
    # Only keep adjustable override keys (defence against an invented key).
    _ALLOWED = {"interest_rate_pct", "target_property_price", "n_props_owned"}
    _dropped = {k: v for k, v in overrides.items() if k not in _ALLOWED}
    overrides = {k: v for k, v in overrides.items() if k in _ALLOWED}
    # Numbers must be numeric: the prompt says so, but a model that answers "3.2%"
    # or "1.2 %" would otherwise reach calculate_loan as a string and fail there —
    # far from the cause. Coerce what is clearly a number, drop what is not.
    for k, v in list(overrides.items()):
        if isinstance(v, str):
            try:
                overrides[k] = float(v.strip().rstrip("%").replace(",", "").strip())
            except ValueError:
                print(f"  [warn] replan override {k}={v!r} is not numeric; dropped")
                del overrides[k]
    if _dropped:
        # A recompute request lost to an unrecognised key is the RM's instruction
        # being dropped on the floor; say so rather than degrading in silence.
        print(f"  [warn] replan proposed non-adjustable override(s) {_dropped} — ignored")
    if kind == "recompute" and not overrides:
        # No usable override survived -> nothing to recompute; fall back to redraft.
        print("  [warn] replan said recompute but no usable override survived; "
              "falling back to redraft (the RM's change will NOT be applied)")
        kind, route_to = "redraft", "document_drafting"
    print(f"[Replan] kind={kind}  route_to={route_to}  overrides={overrides}")

    out = {"replan_kind": kind, "replan_msg": message, "route": route_to, "audit": audit}
    if overrides:
        payload_update = dict(state.get("payload") or {})
        payload_update["overrides"] = overrides
        out["payload"] = payload_update
        out["overrides"] = overrides
    if kind == "reject_unchangeable" and message:
        out["audit"].append(vars(LogEntry("a2a", {
            "role": "assistant", "from": "Replan", "to": "RM", "content": message})))
    return out


def _route_replan(state: CopilotState) -> str:
    kind = state.get("replan_kind", "redraft")
    if kind == "recompute":
        # Re-run the producing agent (LO -> policy_product, IPA -> property_analysis).
        tgt = state.get("route", "")
        return tgt if tgt in ("policy_product", "property_analysis") else "document_drafting"
    if kind == "reject_unchangeable":
        return "hitl_review"   # draft unchanged; RM decides again
    return "document_drafting"  # redraft (wording only)


# ══════════════════════════════════════════════════════════════════════════
# Per-agent routing functions (tools loop OR A2A chain — single registration)
# ══════════════════════════════════════════════════════════════════════════
def _route_borrower_llm(state: CopilotState) -> str:
    if _last_msg_has_tool_calls(state):
        return "tools"
    if _is_full_ipa(state):
        return "property_analysis"
    return END


def _route_property_llm(state: CopilotState) -> str:
    if _last_msg_has_tool_calls(state):
        return "tools"
    # Global Shared Services: validate document completeness/validity BEFORE
    # compliance runs (design 'A2A: validate' edge into Document validation).
    # A reject→recompute re-runs this agent and must ALSO chain onward so the
    # re-priced case flows back through compliance → drafting → the gate.
    if _is_full_ipa(state) or _is_recompute(state):
        return "document_validation"
    return END


def _route_policy_llm(state: CopilotState) -> str:
    if _last_msg_has_tool_calls(state):
        return "tools"
    # Global Shared Services: document validation precedes compliance in the
    # full flow (design 'A2A: validate' edge into Document validation).
    # A reject→recompute re-runs this agent and must ALSO chain onward (see
    # _route_property_llm) so the re-priced Letter of Offer returns to the gate.
    if _is_full_lo(state) or _is_recompute(state):
        return "document_validation"
    return END


def _route_compliance_llm(state: CopilotState) -> str:
    if _last_msg_has_tool_calls(state):
        return "tools"
    # Assessment-only flow: stop here with the eligibility verdict — no letter is
    # drafted. (The RM can then ask to draft the letter as a separate step.)
    if _is_assess_only(state):
        return END
    # Global Shared Service: in a full IPA/LO flow (or a reject→recompute re-run),
    # hand the compliance-cleared payload to Document Drafting; standalone stops.
    if _is_full_flow(state) or _is_recompute(state):
        return "document_drafting"
    return END


def _route_docval_llm(state: CopilotState) -> str:
    if _last_msg_has_tool_calls(state):
        return "tools"
    # Full flow (or a reject→recompute re-run): once docs are validated, hand the
    # payload to compliance. Standalone document-completeness query: stop here.
    if _is_full_flow(state) or _is_recompute(state):
        return "compliance_validation"
    return END


def _route_reprice_llm(state: CopilotState) -> str:
    # Standalone advisory agent: loop on tool calls, otherwise stop. Reprice is
    # not part of the IPA/LO build chain, so it never hands off to another stage.
    if _last_msg_has_tool_calls(state):
        return "tools"
    return END


def _route_customer_llm(state: CopilotState) -> str:
    # Customer self-service is a terminal lane: loop on tool calls, otherwise
    # stop. It never chains into any RM stage (no drafting, no compliance, no
    # HITL gate) — the graph topology itself enforces the toC restriction.
    if _last_msg_has_tool_calls(state):
        return "tools"
    return END


def _route_drafting_llm(state: CopilotState) -> str:
    if _last_msg_has_tool_calls(state):
        return "tools"
    # Full flow, or a reject→replan re-run (recompute OR redraft): hand the letter
    # to the HITL review gate (RM makes the final call). A replan re-run always came
    # from a rejected draft, so it MUST re-gate — otherwise the redrafted letter
    # falls out as a plain answer with no PDF / Approve gate. Standalone drafting
    # (e.g. re-render an already-cleared payload) stops here.
    if _is_full_flow(state) or _is_recompute(state) or _is_redraft(state):
        return "hitl_review"
    return END


# ── Tool → back to calling agent ─────────────────────────────────────────
def _route_tools_back(state: CopilotState) -> str:
    # Each agent's setup node records itself in state['current_agent'], so tool
    # results return to that agent's LLM node explicitly. (Replaces the old
    # system-prompt text matching, which broke if two skill.md prompts collided
    # or a prompt was edited mid-session.)
    cur = state.get("current_agent", "")
    return f"{cur}_llm" if cur in AGENTS else END


# ══════════════════════════════════════════════════════════════════════════
# Build the StateGraph
# ══════════════════════════════════════════════════════════════════════════
builder = StateGraph(CopilotState)

# Orchestrator
builder.add_node("orchestrator", orchestrator_node)
builder.set_entry_point("orchestrator")
builder.add_conditional_edges(
    "orchestrator",
    _route_after_orchestrator,
    {n: n for n in AGENTS} | {END: END},
)

# Shared ToolNode
tool_node = ToolNode(ALL_TOOLS)
builder.add_node("tools", tool_node)

# Per-routing-function map for agents
_AGENT_ROUTE_FN = {
    "borrower_profile": _route_borrower_llm,
    "property_analysis": _route_property_llm,
    "compliance_validation": _route_compliance_llm,
    "document_validation": _route_docval_llm,
    "policy_product": _route_policy_llm,
    "document_drafting": _route_drafting_llm,
    "reprice_retention": _route_reprice_llm,
    "customer_assistant": _route_customer_llm,
}
_AGENT_ROUTE_TARGETS = {
    "borrower_profile": {"tools": "tools", "property_analysis": "property_analysis", END: END},
    "property_analysis": {"tools": "tools", "document_validation": "document_validation", END: END},
    "compliance_validation": {"tools": "tools", "document_drafting": "document_drafting", END: END},
    "document_validation": {"tools": "tools", "compliance_validation": "compliance_validation", END: END},
    "policy_product": {"tools": "tools", "document_validation": "document_validation", END: END},
    "document_drafting": {"tools": "tools", "hitl_review": "hitl_review", END: END},
    "reprice_retention": {"tools": "tools", END: END},
    "customer_assistant": {"tools": "tools", END: END},
}

for _aname in AGENTS:
    builder.add_node(_aname, make_agent_node(_aname))
    builder.add_node(f"{_aname}_llm", make_llm_node(_aname))
    builder.add_edge(_aname, f"{_aname}_llm")
    builder.add_conditional_edges(
        f"{_aname}_llm",
        _AGENT_ROUTE_FN[_aname],
        _AGENT_ROUTE_TARGETS[_aname],
    )

# Tools route back to calling agent
builder.add_conditional_edges(
    "tools",
    _route_tools_back,
    {f"{n}_llm": f"{n}_llm" for n in AGENTS} | {END: END},
)

# ---- HITL review gate: drafting -> hitl_review -> (approve) END / (reject) drafting ----
builder.add_node("hitl_review", hitl_review_node)
builder.add_conditional_edges(
    "hitl_review",
    _route_hitl_review,
    {"replan": "replan", END: END},
)

# ---- Replan sub-orchestrator: reject -> replan -> recompute / redraft / refuse ----
builder.add_node("replan", replan_node)
builder.add_conditional_edges(
    "replan",
    _route_replan,
    {"policy_product": "policy_product", "property_analysis": "property_analysis",
     "document_drafting": "document_drafting", "hitl_review": "hitl_review"},
)

# Checkpointer persists per-thread graph state so an interrupt (HITL gate)
# can pause across two user interactions and resume from the pause point.
graph = builder.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    print("LangGraph compiled.")
    print("Agents:", list(AGENTS.keys()))
    print("Nodes:", list(graph.get_graph().nodes.keys()))
