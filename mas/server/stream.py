"""Stream layer — drive the LangGraph graph and emit Server-Sent Events.

This is the FastAPI equivalent of the Gradio generator callback in the notebook
(cell 17 `_consume_stream` / `handle_message` / `_resume_turn`). It consumes
`graph.stream(...)` events and translates each into a small JSON SSE event the
browser renders:

    {"event": "thinking_open",  "agent": "...", "title": "..."}
    {"event": "thinking_close", "agent": "...", "title": "...", "duration": 1.2, "summary": "..."}
    {"event": "tool_call",      "name": "...", "args": {...}}     # for the activity log
    {"event": "tool_result",    "name": "...", "result": "..."}   # for the activity log
    {"event": "misroute",       "suggested_stage": "REPRICE"}     # orchestrator chose another lane
    {"event": "answer",         "text": "...", "assessment": bool} # agent's final reply bubble
    {"event": "draft",          "draft": "...", "replan_msg": "",  # HITL gate — needs Approve/Revise
                                "assessment": bool}
        # assessment=True on both = this turn ran a full flow that CALLS the calculator
        # (full_ipa / full_lo / full_*_assess). The Loan Scenario card fills only then;
        # an ad-hoc question must leave it empty. See _computes_scenario().
    {"event": "done"}                                             # turn finished, no pending gate

Per-(applicant, stage) conversation state (running message history + the active
HITL thread_id) is kept server-side in SESSIONS so a follow-up turn and an
Approve/Reject resume hit the same thread. The graph itself is unchanged.
"""

from __future__ import annotations

import json
import re
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from graph import AGENTS, MAX_TOOL_ROUNDS, graph, new_state
from langgraph.types import Command

from server import case_progress, case_overrides
from utils import config, letter_store
from utils.telemetry import RunMetrics, rewrite_probe, token_counter
from utils.tools import store

# Business-language copy for the per-agent thinking blocks (ported from the
# notebook's _THINK_COPY). (pending title, done title).
_THINK_COPY = {
    "orchestrator": ("Understanding your request…", "Request understood"),
    "borrower_profile": ("Reviewing borrower profile, income & credit…", "Borrower profile reviewed"),
    "property_analysis": ("Assessing the property & LTV limits…", "Property & LTV assessed"),
    "policy_product": ("Selecting loan products & pricing…", "Loan products & pricing selected"),
    "document_validation": ("Checking document completeness…", "Documents checked"),
    "compliance_validation": ("Running MAS compliance checks…", "Compliance checks complete"),
    "document_drafting": ("Drafting the letter…", "Letter drafted"),
    "reprice_retention": ("Comparing packages & preparing retention…", "Retention comparison ready"),
    "customer_assistant": ("Working on your question…", "Answer ready"),
    "replan": ("Replanning based on your feedback…", "Replan decided"),
    "hitl_review": ("Waiting for your review…", "Review received"),
}
_THINK_MAX_CHARS = 700


def _is_assess_route(route: str | None) -> bool:
    """True for the ASSESS-only flows (full_ipa_assess / full_lo_assess).

    Narrower than _computes_scenario: this is the stage *milestone* — the RM got a
    verdict and the next-best-action chips should advance from "assess" to "draft".
    """
    return "assess" in (route or "").lower()


def _computes_scenario(route: str | None) -> bool:
    """True for any full flow that actually runs calculate_loan — the assess-only
    flows AND the drafting flows (full_ipa / full_lo, which assess *then* draft).

    Drives the Loan Scenario card: it may only show figures the RM asked to have
    computed. An ad-hoc question (borrower_profile, a policy lookup, …) never fills
    it, because those numbers would come from the seed data, not from this turn.
    """
    return (route or "").lower() in ("full_ipa", "full_lo", "full_ipa_assess", "full_lo_assess")

# Map a UI stage to the orchestrator's expected stage token (they already match,
# but keep the indirection explicit). "none" never triggers a misroute prompt.
_UI_STAGES = {"IPA", "LO", "REPRICE"}


# ── server-side per-(applicant, stage) session store ────────────────────────
# Keyed "APP0007|LO". Holds the running message history and the active HITL
# thread_id so a draft can be approved/rejected on a later request.
SESSIONS: dict[str, dict] = {}


def _skey(applicant_id: str, stage: str) -> str:
    return f"{applicant_id.upper()}|{stage}"


def _session(applicant_id: str, stage: str) -> dict:
    key = _skey(applicant_id, stage)
    if key not in SESSIONS:
        SESSIONS[key] = {"messages": [], "thread_id": None, "pending": False}
    # Stash the identity so _drive_events can key the override store when a
    # reject→recompute turn surfaces an `overrides` dict in the graph state.
    SESSIONS[key]["applicant_id"] = applicant_id
    SESSIONS[key]["stage"] = stage.upper()
    return SESSIONS[key]


# ── History compaction ──────────────────────────────────────────────────────
# The session transcript is REPLAYED into every later turn (see stream_chat), so
# left alone it grows without bound: a measured full-IPA turn leaves ~14.5k tokens
# behind, and the second such turn on the same lane overflows a 32k endpoint before
# the RM has typed anything. Compaction keeps what a later turn can actually use
# and drops what it cannot.
#
# What is dropped is EXECUTION SCAFFOLDING, not conversation: each agent's own
# SystemMessage (its skill.md — up to ~3k tokens, and addressed to that agent
# alone), the assistant turns that merely carry tool_calls, and older ToolMessage
# payloads. None of it informs a later turn, because every agent node re-seeds its
# own system prompt and re-calls its own tools on entry (graph.make_agent_node),
# and genuine agent-to-agent hand-off travels through `payload` (ltv_fusion /
# lo_basis / figures), never through the transcript.
#
# What is KEPT:
#   - every HumanMessage — the RM's own words. Load-bearing beyond politeness: the
#     reprice flow reads the competitor rate ("DBS offered 1.55%") from what the RM
#     said in an earlier turn, and dropping it would make the agent re-ask.
#   - each turn's final assistant answer (the prose, no tool_calls).
#   - the MOST RECENT turn's tool results, so an immediate follow-up ("explain that
#     TDSR again") still has the raw figures instead of re-fetching them.
# Older tool results fall away: by then the follow-up is about the answer, and the
# agent can always re-call the tool if it needs the data again.
_TOOL_HISTORY_TURNS = 1


def _compact_history(messages: list, keep_tool_turns: int = _TOOL_HISTORY_TURNS) -> list:
    """Return `messages` reduced to what a LATER turn can use.

    Tool-call/result PAIRING is the constraint that shapes this. A ToolMessage whose
    matching assistant tool_call is gone is an orphan, and an OpenAI-compatible
    endpoint rejects the whole request with a 400 — a worse failure than the overflow
    being fixed. So a tool round is kept or dropped whole: when a ToolMessage is
    retained, the AIMessage carrying its tool_call is retained too.
    """
    # Read live off the module so a test (or an A/B run) can flip it mid-process.
    if not messages or not config.TRIM_CONTEXT:
        return messages

    # Turn boundaries = HumanMessage positions. The last `keep_tool_turns` turns keep
    # their tool traffic; everything earlier is compacted to conversation only.
    # keep_tool_turns == 0 is handled separately: `human_idx[-0]` is `human_idx[0]`,
    # which would put the cutoff at the START and retain everything — the exact
    # opposite of "keep no tool history".
    human_idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if keep_tool_turns <= 0:
        cutoff = len(messages)                       # retain no tool traffic at all
    elif len(human_idx) >= keep_tool_turns:
        cutoff = human_idx[-keep_tool_turns]
    else:
        cutoff = 0                                   # fewer turns than asked: keep all

    # Which tool_call ids survive: those answered within the retained window.
    live_ids = {
        tc.get("id")
        for m in messages[cutoff:]
        if isinstance(m, AIMessage)
        for tc in (getattr(m, "tool_calls", None) or [])
    }

    out = []
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            out.append(m)
        elif isinstance(m, ToolMessage):
            # Keep only if recent AND its originating call is being kept.
            if i >= cutoff and getattr(m, "tool_call_id", None) in live_ids:
                out.append(m)
        elif isinstance(m, AIMessage):
            calls = getattr(m, "tool_calls", None) or []
            if not calls:
                out.append(m)                       # a real answer — always keep
            elif i >= cutoff:
                out.append(m)                       # keeps ToolMessages above non-orphaned
        # SystemMessage: dropped. Every agent re-seeds its own on entry.
    return out


def _approx_tokens(messages: list) -> int:
    """Rough token count for the console line — chars/3.6, the usual BPE ballpark
    for English prose mixed with JSON. Deliberately not a real tokenizer: this is a
    scale readout ("6k vs 46k"), not a budget, and it must never add a dependency
    or cost to a request path."""
    total = 0
    for m in messages:
        content = getattr(m, "content", "") or ""
        total += len(content if isinstance(content, str) else str(content))
        for tc in (getattr(m, "tool_calls", None) or []):
            total += len(str(tc.get("args", "")))
    return int(total / 3.6)


def _log_context(before: list, after: list) -> None:
    """One line per finished turn: what this lane will replay into the NEXT turn.

    Printed because the saving is otherwise invisible — the RM sees the same answer
    either way, and the cost only shows up later as a context-overflow 400. Having
    the number on the console is what makes "did trimming make the model dumber?"
    an answerable question: run the same queries and read quality off the answers,
    cost off this line.
    """
    try:
        kept, raw = _approx_tokens(after), _approx_tokens(before)
        print(f"  [context] history {kept:,} tok for the next turn "
              f"(compacted from {raw:,}; {len(after)}/{len(before)} messages)")
    except Exception:  # noqa: BLE001 — a telemetry line must never break a turn
        pass


def reset_session(applicant_id: str, stage: str) -> dict:
    """Drop this lane's accumulated history, returning what was discarded.

    The escape hatch for a conversation that has grown past the model's context
    window. `SESSIONS[key]["messages"]` only ever grows (see the two
    `sess["messages"] + all_new_messages` accumulations below), so once a full
    assessment turn has run, EVERY later message replays that whole transcript
    and a small-context endpoint rejects the request before the RM can say
    anything. Without this the only remedy is restarting the process.

    Deliberately scoped to one (applicant, stage) lane, not a global wipe: the
    RM's other cases are unaffected. The graph's own checkpointer is untouched —
    a new thread_id is minted per turn anyway (see _config), and clearing
    `thread_id`/`pending` here is what abandons any half-finished HITL gate so
    the next turn starts clean rather than trying to resume a dead interrupt.

    Case data is NOT touched: KPI progress, overrides and rendered letters live
    in their own stores and survive. This clears the conversation only.
    """
    key = _skey(applicant_id, stage)
    sess = SESSIONS.get(key) or {}
    dropped = len(sess.get("messages") or [])
    was_pending = bool(sess.get("pending"))
    SESSIONS.pop(key, None)
    return {"cleared": True, "messages_dropped": dropped,
            "had_pending_gate": was_pending}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False, default=str)}\n\n"


def _summary(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= _THINK_MAX_CHARS else text[:_THINK_MAX_CHARS].rstrip() + " …"


# Pull the leading clause number ("2.1", "1.3.1", "(a)") off a chunk title so the
# Sources strip can show "Clause 2.1" instead of the whole first line.
_CLAUSE_NUM_RE = re.compile(r"^\s*((?:\d+\.)+\d*|\d+\.?|\([a-z]+\))")


def _collect_policy_sources(result_str: str, acc: dict) -> None:
    """Parse a search_policy tool result and add each hit's {source, clause} to
    ``acc`` (keyed by (source, clause) so duplicates across queries collapse).
    Best-effort: a non-list / unparseable result is ignored."""
    try:
        hits = json.loads(result_str)
    except (json.JSONDecodeError, TypeError):
        return
    if not isinstance(hits, list):
        return
    for h in hits:
        if not isinstance(h, dict):
            continue
        source = h.get("source", "")
        title = h.get("title", "") or ""
        m = _CLAUSE_NUM_RE.match(title)
        clause = m.group(1).rstrip(".") if m else ""
        key = (source, clause)
        if source and key not in acc:
            acc[key] = {"source": source, "clause": clause}


def _config():
    return {
        "configurable": {"thread_id": uuid.uuid4().hex},
        "recursion_limit": MAX_TOOL_ROUNDS * len(AGENTS) + 8,
    }


def _sync_calc_rate_override(node_output: dict, sess: dict, ui_stage: str) -> None:
    """Mirror the interest rate a producing agent just computed with into the
    override store, so the read-only Loan Scenario card stays in sync when the RM
    changes the rate via the ORDINARY chat composer (e.g. types "change the rate to
    1.5%") instead of the HITL "Revise" button.

    The Revise button routes through replan, which emits a structured `overrides`
    dict the caller already captures. But a composer message re-runs the whole flow
    WITHOUT replan, so no override is emitted and the left-panel KPIs would go stale
    even though the agent (and the redrafted letter) used the new rate. Here we read
    the rate actually passed to this turn's `calculate_loan` calls and, when it
    differs from the applicant's CSV rate, record it as an override — exactly what a
    replan recompute would have stored. Rate only; price / n_props changes still go
    through the structured replan path. Best-effort; never raises."""
    try:
        aid = sess.get("applicant_id", "")
        if not aid:
            return
        # Find the rate this turn's calculate_loan actually used (last one wins).
        rate = None
        for entry in node_output.get("audit", []) or []:
            if entry.get("kind") != "tool_call":
                continue
            payload = entry.get("payload", {}) or {}
            if payload.get("name") != "calculate_loan":
                continue
            args = payload.get("args", {}) or {}
            if args.get("interest_rate_pct") is not None:
                rate = float(args["interest_rate_pct"])
        if rate is None:
            return
        # Only record it when it actually differs from the seed CSV rate (else a
        # normal flow would spuriously flag the card as "adjusted").
        loan_app = store.get_loan_application(aid)
        csv_rate = None
        if isinstance(loan_app, dict) and "error" not in loan_app:
            csv_rate = loan_app.get("interest_rate_pct")
        if csv_rate is not None and abs(float(csv_rate) - rate) < 1e-9:
            return
        case_overrides.apply_overrides(
            aid, sess.get("stage", ui_stage), {"interest_rate_pct": rate})
    except Exception:  # noqa: BLE001 — a KPI-sync convenience must never break the stream
        pass


def _render_letter(applicant_id: str, stage: str, body_text: str,
                   *, draft: bool) -> tuple[str, str]:
    """Render the letter PDF from the drafted BODY TEXT plus the figures/recipient
    the draft_letter tool registered, stash it, and return its (download URL,
    filename). ("", "") if the agent never registered the letter (didn't call
    draft_letter) or there is no body text yet.

    The body comes from the agent's final answer — NOT from a tool-call argument —
    so a long markdown body can never corrupt the tool-call JSON. A letter is a
    nice-to-have overlay: any failure returns ("", "") and never breaks the stream.

    Cache-busted with a short token so an approved (final) letter isn't served from
    a stale draft cache under the same lane URL."""
    from utils import letter_pdf

    stored = letter_store.get_body(applicant_id, stage)
    if not stored:
        return "", ""          # agent didn't call draft_letter this turn
    _, recipient, facts = stored
    body = (body_text or "").strip()
    if not body:
        return "", ""
    try:
        pdf, filename = letter_pdf.build_letter_pdf(
            applicant_id, stage, body, draft=draft,
            recipient=recipient, facts=facts)
    except Exception:
        return "", ""
    letter_store.put(applicant_id, stage, draft, pdf, filename)
    # Remember the body so an Approve can re-render the identical letter as final.
    letter_store.put_body(applicant_id, stage, body, recipient, facts)
    v = uuid.uuid4().hex[:8]
    url = (f"/api/letter/{applicant_id.upper()}/{stage.upper()}"
           f"?draft={1 if draft else 0}&v={v}")
    return url, filename


def _release_final_letter(applicant_id: str, stage: str) -> tuple[str, str]:
    """On Approve, re-render the identical drafted letter as the final (non-DRAFT)
    PDF from the body/facts stored at the draft gate. ("", "") if no draft body was
    stored (agent skipped draft_letter)."""
    stored = letter_store.get_body(applicant_id, stage)
    if not stored:
        return "", ""
    body, _, _ = stored
    return _render_letter(applicant_id, stage, body, draft=False)


def _explain_error(exc: Exception) -> str:
    """Operator-facing text for a graph-killing exception.

    The raw repr of an SDK error is close to useless on a locked-down host: a bare
    "InternalServerError: Internal Server Error" names neither the endpoint that
    failed nor anything you can act on, and on an intranet deployment it sits on
    screen next to an unrelated (and harmless) OpenRouter pricing warning, which
    sends people debugging the wrong thing entirely. So we name the configured
    endpoint and say what class of failure it is. metrics.error keeps the raw
    string — this only changes what the human reads.
    """
    raw = f"{type(exc).__name__}: {exc}"

    # Tool-round exhaustion is not an endpoint failure and must not read like one.
    # It means the agent kept calling tools without converging (observed 2026-08-04:
    # eleven identical interest_savings_tool calls). The RM can neither diagnose nor
    # act on "GraphRecursionError", so say what happened in their terms.
    if type(exc).__name__ == "GraphRecursionError" or "recursion limit" in str(exc).lower():
        return ("The assistant kept calling tools without settling on an answer, so "
                "the turn was stopped. Nothing was saved to the case. Try rephrasing "
                "the question, or narrow it to one thing at a time.")

    try:
        from utils import config
        endpoint = config.BASE_URL
        retries = config.MAX_RETRIES
    except Exception:  # noqa: BLE001 — never fail while reporting a failure
        return raw

    status = None
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            status = val
            break

    if status in (500, 502, 503, 504):
        return (f"{raw} — the LLM endpoint ({endpoint}) returned HTTP {status} and "
                f"was still failing after {retries} retries. This is the gateway "
                f"or the model behind it, not the case data; retry the turn, and if "
                f"it persists check the gateway logs.")
    if status in (401, 403):
        return (f"{raw} — the LLM endpoint ({endpoint}) rejected the credential. "
                f"Check the API key for this route in mas/.env.")
    if isinstance(exc, (ConnectionError, OSError)) or "name or service not known" in str(exc).lower():
        return (f"{raw} — could not reach the LLM endpoint ({endpoint}). On an "
                f"intranet host, confirm INTERNAL_LLM_BASE_URL points at a "
                f"reachable gateway.")
    # A local programming error never touched the network, so do NOT blame the
    # endpoint. Observed 2026-08-04: a KeyError raised inside the calculator was
    # reported as "KeyError: 'nationality' — while calling the LLM endpoint
    # (https://openrouter.ai/api/v1)", which sent the diagnosis straight at
    # OpenRouter when the real cause was an incomplete tool-call argument. The
    # catch-all below is for provider-shaped failures; these are ours.
    if isinstance(exc, (KeyError, AttributeError, IndexError, TypeError, ValueError)):
        return (f"{raw} — this is an internal error in the assistant, not the LLM "
                f"endpoint. The turn was stopped and nothing was saved to the case.")
    return f"{raw} — while calling the LLM endpoint ({endpoint})."


def _drive(stream_iter, sess, ui_stage, think_agent="", is_resume=False, finish=None):
    """Core event loop shared by a fresh chat turn and a HITL resume.

    Yields SSE strings. Tracks the currently-open thinking block so it can be
    sealed (thinking_close) when the next agent takes over / the run ends.
    The actual loop is in _drive_events; this wrapper turns any graph exception
    (e.g. an LLM auth/gateway error) into a clean SSE `error` event so the
    browser never just sees a dropped connection.

    Also records one operational-metrics row (eval/runs.csv) per stream: the
    token counter is reset here so it sums only this stream's LLM calls, and
    node/tool/route/interrupted/error are tallied on the RunMetrics `metrics`
    passed down into _drive_events. The row is written on every exit path.

    `finish(metrics)` — optional — takes over writing that row, for a caller with
    post-stream work the row must reflect (the Approve path releases the final PDF
    after the stream ends; the letter-delivery and time-to-letter fields are only
    true once it has). The caller MUST write the row.
    """
    metrics = RunMetrics(
        source="server",
        applicant_id=sess.get("applicant_id", ""),
        stage=(sess.get("stage") or ui_stage or ""),
        turn=sess.get("turn", 0),
        is_resume=is_resume,
    )
    # A resume stream never re-runs the orchestrator, so seed the route from the
    # case's last known route so resume rows still carry it.
    if is_resume:
        metrics.set_route(sess.get("last_route", ""))

    # North Star (time-to-letter): the span the RM experiences runs from the message
    # that started this letter to the approved PDF existing — across BOTH the drafting
    # request and the later Approve request. `case_t0` is stamped on the session (which
    # outlives a single HTTP request) by the drafting turn; the resume turn reads it
    # back and hands RunMetrics everything that elapsed before its own leg. A resume
    # with no stamp (server restarted mid-case) contributes 0, so time_to_letter_s
    # degrades to this stream's latency rather than to a wrong total.
    if not is_resume:
        sess["case_t0"] = time.time()
    else:
        t0 = sess.get("case_t0")
        metrics.case_elapsed_s = max(0.0, time.time() - t0) if t0 else 0.0
    token_counter.reset()   # count only this stream's LLM calls
    rewrite_probe.reset()   # count only this stream's policy-rewrite activity
    think_t0 = time.time() if think_agent else None
    # Shared, mutable view of the currently-open thinking block. _drive_events
    # advances this as agents take over; on a graph exception the except clause
    # below reads it to seal the block that was ACTUALLY open — not the stale
    # initial `think_agent` (which would mislabel the failing agent's block, e.g.
    # closing a borrower block with the orchestrator's "Request understood").
    cur = {"agent": think_agent, "t0": think_t0}

    def close_block(agent, t0):
        if not agent:
            return None
        ev = {"event": "thinking_close", "agent": agent,
              "title": _THINK_COPY.get(agent, ("", agent))[1]}
        if t0 is not None:
            ev["duration"] = round(time.time() - t0, 1)
        return ev

    try:
        yield from _drive_events(
            stream_iter, sess, ui_stage, think_agent, think_t0, close_block, cur, metrics)
    except Exception as e:  # surface a graceful error instead of dropping the SSE
        ev = close_block(cur["agent"], cur["t0"])
        if ev:
            yield _sse(ev)
        sess["pending"] = False
        sess["thread_id"] = None
        metrics.error = f"{type(e).__name__}: {e}"
        yield _sse({"event": "error", "text": _explain_error(e)})
        yield _sse({"event": "done"})
    finally:
        # Did this stream end with a letter the RM can actually download? Only the
        # approved (non-DRAFT) PDF counts, so the drafting turn that merely reaches
        # the gate is not credited with delivering one — its cost is instead folded
        # into the resume row's time_to_letter_s via case_t0.
        #
        # `finish` lets the CALLER release the final PDF before the row is written:
        # stream_resume renders it only after _drive returns, so checking the letter
        # store here would always miss it and pin time_to_letter_s blank on exactly
        # the turn that delivers the letter.
        if finish is None:
            metrics.check_letter(final_only=True)
            metrics.write()   # one row per stream, on every exit path
        else:
            finish(metrics)


def _drive_events(stream_iter, sess, ui_stage, think_agent, think_t0, close_block, cur, metrics):
    """Inner stream loop (separated so _drive can wrap it in one try/except).

    `cur` is the shared {agent, t0} view of the open thinking block; it is kept in
    sync with the local think_agent/think_t0 so _drive can seal the right block if
    the graph raises mid-run. `metrics` is the RunMetrics row being tallied.
    """
    final_answer = ""
    all_new_messages = []
    interrupted = False
    draft = ""
    replan_msg = ""
    misroute_emitted = False
    agent_summary = ""   # rolling "conclusion" for the current agent's thinking block
    # Policy clauses retrieved this turn (search_policy hits), for the answer's
    # "Sources" strip. Keyed by (source, title) so the same clause retrieved by
    # two queries is listed once; insertion order is preserved for display.
    policy_sources: dict[tuple, dict] = {}

    def close_with_summary(agent, t0, summ):
        ev = close_block(agent, t0)
        if ev and summ:
            ev["summary"] = _summary(summ)
        return ev

    for event in stream_iter:
        # LangGraph surfaces an interrupt under the special '__interrupt__' key.
        if "__interrupt__" in event:
            interrupted = True
            metrics.interrupted = True
            intr = event["__interrupt__"]
            payload = getattr(intr[0], "value", {}) if intr else {}
            if isinstance(payload, dict):
                draft = payload.get("draft", "")
                replan_msg = payload.get("replan_msg", "")
            break

        for node_name, node_output in event.items():
            if not isinstance(node_output, dict):
                continue

            # Capture an RM recompute override the moment replan emits it, so the
            # read-only Loan Scenario card can re-price at the adjusted value (e.g.
            # the RM repriced the letter to 1.2%). The seed CSV is never mutated.
            # This fires on the HITL "Revise" (reject→replan) path.
            if node_output.get("overrides"):
                case_overrides.apply_overrides(
                    sess.get("applicant_id", ""), sess.get("stage", ui_stage),
                    node_output["overrides"])

            # Also keep the card in sync when the RM changes the rate via the plain
            # chat composer (no replan, so no `overrides` above). Mirrors the rate
            # this turn's calculate_loan actually used into the override store.
            _sync_calc_rate_override(node_output, sess, ui_stage)

            base = node_name[:-4] if node_name.endswith("_llm") else node_name

            # Metrics: an *_llm node output is one agent-node visit. Re-entering
            # the same llm node (after a tool loop) is what loop_count counts.
            if node_name.endswith("_llm"):
                metrics.visit_node(base)

            # Audit completeness: every node that runs is bound by CLAUDE.md's audit
            # convention, so tally the node here and how many entries it logged. A
            # node that appears in nodes_ran but never in nodes_logged is a gap in
            # the trail — that is the KPI, not the raw entry count.
            metrics.saw_node(node_name)
            metrics.add_audit_entries(node_name, len(node_output.get("audit", []) or []))

            # Open a new thinking block when a new agent's node appears.
            if base in _THINK_COPY and base != think_agent and base != "hitl_review":
                ev = close_with_summary(think_agent, think_t0, agent_summary)
                if ev:
                    yield _sse(ev)
                agent_summary = ""
                think_agent, think_t0 = base, time.time()
                cur["agent"], cur["t0"] = think_agent, think_t0
                yield _sse({"event": "thinking_open", "agent": base,
                            "title": _THINK_COPY[base][0]})

            for entry in node_output.get("audit", []):
                kind = entry.get("kind", "")
                payload = entry.get("payload", {})
                if kind == "tool_call":
                    # Pass name+args so metrics can record which tools ran and check
                    # each call's applicant_id against the session's (leakage).
                    metrics.add_tool_call(payload.get("name", ""), payload.get("args", {}))
                    yield _sse({"event": "tool_call", "name": payload.get("name", ""),
                                "args": payload.get("args", {}),
                                "call_id": payload.get("call_id", "")})
                elif kind == "a2a":
                    role = payload.get("role", "")
                    if role == "routed":
                        # Routing-chain hop for the activity log (agent badge).
                        ag = AGENTS.get(payload.get("agent", ""))
                        yield _sse({"event": "routed",
                                    "agent": payload.get("agent", ""),
                                    "label": ag.label if ag else payload.get("to", ""),
                                    "color": ag.color if ag else "#6b7280"})
                    else:
                        # system / user / assistant message card.
                        yield _sse({"event": "a2a",
                                    "role": role,
                                    "from": payload.get("from", ""),
                                    "to": payload.get("to", ""),
                                    "content": payload.get("content", ""),
                                    "is_final": bool(payload.get("is_final"))})
                    if payload.get("is_final") and payload.get("content"):
                        final_answer = payload["content"]
                        agent_summary = payload["content"]
                elif kind == "answer":
                    final_answer = payload.get("text", "")
                    agent_summary = payload.get("text", "")

            # Orchestrator / replan are one-shot LLM nodes with no is_final a2a of
            # their own — build their thinking-block summary from the state update.
            if base == "orchestrator":
                route = node_output.get("route") or ""
                metrics.set_route(route)
                # Remember the route token so the END handler can tell an
                # assessment flow (full_*_assess) from a drafting flow.
                sess["last_route"] = route
                if route and route != "none":
                    stg = node_output.get("stage", "none")
                    app = node_output.get("applicant_id", "")
                    agent_summary = (f"Route: {route} · Stage: {stg}"
                                     + (f" · Applicant: {app}" if app else ""))
                chosen = (node_output.get("stage") or "none").upper()
                # If the orchestrator chose another lane, offer a "switch stage?" prompt.
                if (not misroute_emitted and chosen in _UI_STAGES
                        and ui_stage in _UI_STAGES and chosen != ui_stage):
                    misroute_emitted = True
                    yield _sse({"event": "misroute", "suggested_stage": chosen})
            elif base == "replan":
                kind_ = node_output.get("replan_kind", "")
                desc = {"recompute": "Recompute the case with the adjusted input(s).",
                        "redraft": "Rewrite the letter; the numbers stand.",
                        "reject_unchangeable": "The requested change is fixed by regulation or derived.",
                        }.get(kind_, kind_)
                msg_ = node_output.get("replan_msg", "")
                agent_summary = (desc + (f"\n{msg_}" if msg_ else "")).strip()

            for m in node_output.get("messages", []):
                all_new_messages.append(m)
                if isinstance(m, ToolMessage):
                    result_str = (m.content if isinstance(m.content, str)
                                  else json.dumps(m.content, ensure_ascii=False, default=str))
                    # Collect search_policy hits for the answer's Sources strip.
                    if getattr(m, "name", "") == "search_policy":
                        _collect_policy_sources(result_str, policy_sources)
                    yield _sse({"event": "tool_result", "name": getattr(m, "name", ""),
                                "result": result_str,
                                "call_id": getattr(m, "tool_call_id", "")})
                elif isinstance(m, AIMessage) and m.content and not getattr(m, "tool_calls", None):
                    # Control nodes (orchestrator/replan) speak through `route` /
                    # `replan_kind`, handled above — their raw AIMessage content is
                    # the routing JSON, NOT a user-facing reply. Never harvest it as
                    # the answer, or that JSON leaks into the RM chat bubble.
                    if base not in ("orchestrator", "replan"):
                        final_answer = m.content

    if interrupted:
        ev = close_with_summary(think_agent, think_t0, draft or agent_summary)
        if ev:
            yield _sse(ev)
        sess["pending"] = True
        # Keep the messages produced so far so the resume continues the history.
        # NOT compacted: this turn is paused mid-flight at the HITL gate, and the
        # resume re-enters drafting expecting its own working context intact. The
        # compaction happens when the turn actually finishes (both END paths below).
        sess["messages"] = sess["messages"] + all_new_messages
        # Render the DRAFT PDF from the drafted body text (this `draft`) plus the
        # figures the draft_letter tool registered. The body comes from the agent's
        # answer, NOT a tool-call arg, so a long letter can't corrupt the tool JSON.
        # No link if the agent didn't call draft_letter (nothing registered).
        pdf_url, pdf_name = _render_letter(
            sess.get("applicant_id", ""), sess.get("stage") or ui_stage,
            draft, draft=True)
        yield _sse({"event": "draft", "draft": draft, "replan_msg": replan_msg,
                    "assessment": _computes_scenario(sess.get("last_route")),
                    "pdf_url": pdf_url, "pdf_name": pdf_name})
        return

    # Ran to END.
    ev = close_with_summary(think_agent, think_t0, agent_summary or final_answer)
    if ev:
        yield _sse(ev)
    sess["pending"] = False
    sess["thread_id"] = None
    # The turn is over, so its scaffolding (agent system prompts, tool-call plumbing,
    # older tool payloads) has done its job. Compact before storing, or this lane's
    # history — replayed into every later turn — grows past the context window.
    _raw_history = sess["messages"] + all_new_messages
    sess["messages"] = _compact_history(_raw_history)
    _log_context(_raw_history, sess["messages"])
    # Tag the answer with whether this turn actually computed the scenario. The Loan
    # Scenario card fills only then, and its "Assistant's reasoning" footer must quote
    # THAT answer — not whatever the RM asked last.
    yield _sse({"event": "answer", "text": final_answer or "(no response)",
                "assessment": _computes_scenario(sess.get("last_route")),
                "sources": list(policy_sources.values())})
    yield _sse({"event": "done"})


def stream_chat(applicant_id: str, stage: str, message: str, role: str = "rm"):
    """SSE generator for a fresh RM message in a given (applicant, stage) lane.

    role="customer" drives the toC self-service lane: the session is keyed by the
    caller's pseudo-id (e.g. "GUEST"), but the graph state carries NO applicant_id
    — a customer session is never bound to a case, and the orchestrator hard-routes
    it to customer_assistant without an LLM decision."""
    if not message.strip():
        yield _sse({"event": "done"})
        return

    sess = _session(applicant_id, stage)
    config = _config()
    sess["thread_id"] = config["configurable"]["thread_id"]

    # Seed state with this lane's running history + the new message + applicant.
    state = new_state("" if role == "customer" else applicant_id, message, role=role)
    state["messages"] = list(sess["messages"]) + state["messages"]

    sess["turn"] = sess.get("turn", 0) + 1
    yield _sse({"event": "user", "text": message})
    # Open a new activity-log turn group; the orchestrator badge starts the chain.
    yield _sse({"event": "turn", "num": sess["turn"], "user_msg": message})
    # Pre-open the orchestrator block (events only fire on node completion).
    yield _sse({"event": "thinking_open", "agent": "orchestrator",
                "title": _THINK_COPY["orchestrator"][0]})
    yield _sse({"event": "routed", "agent": "orchestrator",
                "label": "🧭 Orchestrator", "color": "#f59e0b"})

    stream = graph.stream(state, config=config)
    yield from _drive(stream, sess, stage.upper(), think_agent="orchestrator", is_resume=False)

    # Milestone: an assessment flow (full_*_assess) that ran to END — not a draft
    # flow (it would pause at the HITL gate, leaving sess["pending"] True) — means
    # the RM got an eligibility/package verdict. Record it so the next-best-action
    # chips advance from "assess" to "draft the letter".
    if not sess.get("pending") and _is_assess_route(sess.get("last_route")):
        case_progress.mark_stage_assessed(applicant_id, stage)


def stream_resume(applicant_id: str, stage: str, approved: bool, feedback: str = ""):
    """SSE generator for an Approve / Reject decision on a paused HITL gate."""
    sess = _session(applicant_id, stage)
    if not sess.get("thread_id") or not sess.get("pending"):
        # No pending interrupt — nothing to resume.
        yield _sse({"event": "done"})
        return

    config = {"configurable": {"thread_id": sess["thread_id"]},
              "recursion_limit": MAX_TOOL_ROUNDS * len(AGENTS) + 8}
    yield _sse({"event": "thinking_open", "agent": "hitl_review",
                "title": _THINK_COPY["hitl_review"][0]})
    stream = graph.stream(
        Command(resume={"approved": approved, "feedback": feedback}), config=config)
    # Defer the metrics row until after the final PDF is released below, so the
    # North Star fields can see the letter this turn actually produced.
    #
    # The `finally` spans the drive AND the release, for two failure modes: a crash
    # while rendering the PDF, and the browser disconnecting mid-stream (which throws
    # GeneratorExit into this generator and would otherwise skip the write entirely,
    # silently losing the turn). Monitoring must not be the thing that loses data when
    # the feature under it breaks.
    held: list = []
    try:
        yield from _drive(stream, sess, stage.upper(), think_agent="hitl_review",
                          is_resume=True, finish=held.append)

        # Milestone: an approved draft that ran to END (not a reject that bounced back
        # to the gate via replan, which leaves sess["pending"] True) means the RM
        # signed off on this stage's letter — advance the deal-progress dots.
        if approved and not sess.get("pending"):
            case_progress.mark_stage_completed(applicant_id, stage)
            # Release the letter: re-render the approved draft as the final (non-DRAFT)
            # PDF and hand the browser the clean download link.
            final_url, final_name = _release_final_letter(applicant_id, stage)
            if final_url:
                yield _sse({"event": "letter_ready", "pdf_url": final_url,
                            "pdf_name": final_name})
    finally:
        for m in held:
            m.check_letter(final_only=True)   # now the final PDF exists, if it ever will
            m.write()
