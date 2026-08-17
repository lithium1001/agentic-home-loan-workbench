"""RunMetrics — collect operational KPIs for one graph.stream() call and
append them as a single row to artifacts/runs.csv.

Design (agreed 2026-07-06):
  - Granularity: one graph.stream() == one CSV row (run_id). A full case that
    pauses at the HITL gate spans several streams (chat + each resume); each is
    its own row, tagged with applicant/stage/turn/is_resume so rows can be
    aggregated back into a case downstream.
  - Single append-only file: artifacts/runs.csv, fixed schema, Excel-openable.
  - Two hook points feed one row:
      * token usage  — collected in utils.llm._generate (the ONLY place every
        LLM call passes through; the stream events do NOT carry token usage).
      * latency / loop count / tool calls / path / interrupted / error —
        collected in server.stream._drive from the events already flowing.

The token side uses a thread-local accumulator (`token_counter`) so a stream
handler can .reset() before driving the graph and .read() after, summing tokens
across every LLM call the graph made in between — without threading a metrics
object down into the LLM client.
"""

from __future__ import annotations

import csv
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from utils.config import BASE_URL, MODEL
from utils.telemetry import pricing

# runs.csv is a run artifact, not source: <repo>/artifacts/runs.csv.
# This file is mas/utils/telemetry/metrics.py, so the repo root is 4 levels up.
RUNS_CSV = Path(__file__).resolve().parents[3] / "artifacts" / "runs.csv"

# Fixed column order. run_evals.py MUST write the same header so live-monitoring
# rows and offline-eval rows land in one table. Golden-set columns (expected_*,
# success, routing_correct, tools_correct) are left blank by live monitoring and
# filled by run_evals.py — keeping them here means the schema never diverges.
FIELDNAMES = [
    "run_id",
    "ts",              # ISO-8601 UTC start time
    "source",          # "server" (live) | "eval" (run_evals.py)
    "applicant_id",
    "stage",           # IPA | LO | REPRICE | none
    "turn",            # RM message number within the (applicant, stage) lane
    "is_resume",       # 1 if this stream is a HITL Approve/Reject resume, else 0
    "route",           # orchestrator's chosen agent token (e.g. full_ipa, reprice_retention)
    # ── operational (no labels needed) ──
    "model",           # LLM model id used this stream (config.MODEL)
    "endpoint",        # base_url that served it (config.BASE_URL). Recorded because
                       # `model` alone cannot identify the backend: a single-model
                       # gateway reports its model as "default", and two runs against
                       # different backends would otherwise be indistinguishable in
                       # the CSV — which breaks comparability of any eval batch.
    "latency_s",       # wall-clock of the stream
    "tokens_prompt",
    "tokens_completion",
    "tokens_total",
    "cost_est_usd",    # estimated $ from published per-token prices (blank if unknown; 0 for :free).
                       # Blank is normal and not an error: a self-hosted gateway has no
                       # per-token price, and a host without public internet cannot look
                       # one up. Neither affects the run.
                       # A LOWER BOUND on the real bill, not the bill. Tokens are only
                       # counted when a call RETURNS usage, so two classes of spend are
                       # invisible here: (a) gateway retries — utils/llm.py re-sends the
                       # whole prompt on a 504/429 and the provider charges for the failed
                       # attempt, which reports no usage; (b) requests rejected on context
                       # length — the prompt was transmitted and billed, but the 400 carries
                       # no usage. Measured on the 07-30 full run: est. $0.4806 vs ~$0.80
                       # actually billed (1.66x), the gap traced to 5 context-limit 400s
                       # (~$0.12) plus retries (~$0.20). Report the real invoice figure and
                       # cite this column as the successful-call floor.
    "llm_calls",       # number of LLM invocations this stream
    "loop_count",      # agent-node revisits (tool-loop iterations); 0 = no re-entry
    "tool_calls",      # number of tool_call events
    "node_visits",     # total agent-llm node visits this stream
    "path",            # ">"-joined agent path, e.g. "orchestrator>borrower_profile>property_analysis"
    "audit_entries",   # total LogEntry rows appended to state["audit"] this stream
    "nodes_audited",   # how many DISTINCT nodes emitted >=1 audit entry
    "nodes_total",     # how many DISTINCT nodes ran (audited or not)
    "audit_complete",  # 1 if every node that ran also logged; else 0 (the KPI)
    "interrupted",     # 1 if the stream paused at the HITL gate, else 0
    "letter_produced", # 1 if draft_letter actually registered a letter this run, else 0.
                       # The gate for the North Star: a case that never produced a letter
                       # has no time-to-letter, and averaging it in as if it did would
                       # flatter the metric. Read from utils.letter_store, not from the
                       # tool name in tools_called — the tool can be CALLED and still
                       # fail to render, and the store is the same source the download
                       # endpoint serves, so "produced" means the RM could really get it.
    "time_to_letter_s",# wall-clock from the RM's first message of the case to the
                       # approved letter existing — SUMMED ACROSS TURNS (draft turn +
                       # HITL resume), which is why it is not just latency_s. Blank when
                       # no letter was produced. This is the North Star metric.
    "error",           # error text if the stream raised, else ""
    # ── golden-set columns (blank for live rows; filled by run_evals.py) ──
    "query_id",        # GQ001..GQ050 — which golden case this row is. REQUIRED for
                       # pass^k: k trials of one query share a query_id and differ by trial.
    "trial",           # 1..k — which repeat this is. 1 for deterministic single runs.
    "expected_route",
    "expected_stage",
    "success",         # 1 | 0 | "" — end-to-end task success (eval only)
    "routing_correct", # 1 | 0 | "" — route == expected_route (eval only)
    "tools_correct",   # 1 | 0 | "" — expected_tools ⊆ tools actually called (eval only)
    "tools_called",    # "|"-joined tool names actually invoked — the evidence behind
                       # tools_correct, and what the leakage check reads. Filled on
                       # LIVE rows too (it needs no golden label).
    "leakage",         # 1 if any tool_call carried an applicant_id != the session's.
                       # Tool-layer authorisation check, deterministic, no LLM judge —
                       # see the leakage-is-api-not-prompt decision. Live rows too.
    # ── agentic-RAG observability (filled on EVERY row; no golden label needed) ──
    "policy_searches",  # how many search_policy calls this run made (0 = no RAG used)
    "rewrite_fired",    # of those, how many produced a usable (non-noop) rewrite
    "rewrite_fused",    # of those, how many actually entered RRF fusion (§6.4 lever)
    "rewrite_noop",     # of those, how many degraded to the single-query baseline path
]


# ══════════════════════════════════════════════════════════════════════════
# Token accumulator — written by utils.llm._generate, read by the stream layer.
# ══════════════════════════════════════════════════════════════════════════
class _TokenCounter:
    """Process-wide, thread-local token accumulator.

    utils.llm._generate calls .add(usage) after every LLM response; the stream
    handler calls .reset() before graph.stream and .read() after to get the sum
    for that one stream. Thread-local so concurrent SSE requests don't mix.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _bucket(self) -> dict:
        b = getattr(self._local, "bucket", None)
        if b is None:
            b = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
            self._local.bucket = b
        return b

    def reset(self) -> None:
        self._local.bucket = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}

    def add(self, prompt: int = 0, completion: int = 0, total: int = 0) -> None:
        b = self._bucket()
        b["prompt"] += prompt or 0
        b["completion"] += completion or 0
        # Fall back to prompt+completion if the provider omitted a total.
        b["total"] += total or ((prompt or 0) + (completion or 0))
        b["calls"] += 1

    def read(self) -> dict:
        return dict(self._bucket())


# The single shared instance. Import as `from eval.metrics import token_counter`.
token_counter = _TokenCounter()


# ══════════════════════════════════════════════════════════════════════════
# Rewrite probe — written by utils.tools._agentic_search_policy, read by the
# stream/eval layer. Same thread-local pattern as the token counter: the search
# tool has no handle on the RunMetrics object (it is a bare @tool the graph
# invokes), so it reports what the agentic-RAG layer did via a process-wide,
# thread-local sink the stream driver drains once per run.
#
# WHY this exists: §6.4 of the LR reports the agentic query-rewrite lifting
# colloquial Hit@5 from 45.8% to 73.6% — but that is a single OFFLINE number.
# In production we currently cannot see whether the rewrite even fires, how often
# it no-ops, or whether the clause that answered the query came from the original
# or the rewritten query. This probe makes that observable per run WITHOUT an LLM
# judge and WITHOUT changing retrieval behaviour: it only records what happened.
# ══════════════════════════════════════════════════════════════════════════
class _RewriteProbe:
    """Per-run, thread-local accumulator of query-rewrite activity.

    _agentic_search_policy calls .record(...) once per search_policy invocation;
    the stream/eval driver calls .reset() before the run and .read() after to get
    a compact summary for that run's CSV row. Thread-local so concurrent requests
    don't mix. Multiple searches in one run are summed (calls) and OR-ed
    (fired/fused) — a run "used the rewrite" if ANY search in it did.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def _bucket(self) -> dict:
        b = getattr(self._local, "bucket", None)
        if b is None:
            b = {"searches": 0, "fired": 0, "noop": 0, "fused": 0, "set_iter": 0}
            self._local.bucket = b
        return b

    def reset(self) -> None:
        self._local.bucket = {"searches": 0, "fired": 0, "noop": 0, "fused": 0, "set_iter": 0}

    def record(self, fired: bool, fused: bool, set_iter: bool = False) -> None:
        """One search_policy call. `fired` = a non-empty rewrite differing from the
        original was produced; `fused` = that rewrite's ranking actually entered RRF
        fusion (vs no-op'ing back to the single-query path); `set_iter` = the
        set-question third ranking was added."""
        b = self._bucket()
        b["searches"] += 1
        if fired:
            b["fired"] += 1
        else:
            b["noop"] += 1
        if fused:
            b["fused"] += 1
        if set_iter:
            b["set_iter"] += 1

    def read(self) -> dict:
        return dict(self._bucket())


# The single shared instance. Import as `from utils.telemetry.metrics import rewrite_probe`.
rewrite_probe = _RewriteProbe()


def _migrate_header(path: Path, columns: list | None = None) -> None:
    """Widen an existing CSV written before a column was added.

    DictWriter only emits the header for a NEW file, so appending a wider row to
    an older CSV silently misaligns it: the surplus value lands under DictReader's
    None key and every consumer reads shifted data. Rewrite the file once with the
    current columns, leaving new ones blank on historical rows.

    Blank is the honest value — it means "this run predates the column", which is
    not the same as any particular endpoint, so it is never guessed. No-op when the
    header already matches, so the cost is one header read per write.

    `columns` defaults to runs.csv's FIELDNAMES; eval/run_evals.py passes its own
    EVAL_LOG_COLUMNS, because that file has exactly the same widening problem and
    two copies of this logic would be two places to fix it.
    """
    cols = list(columns) if columns else FIELDNAMES
    try:
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            header = list(rows[0].keys()) if rows else None
        if header is None or header == cols:
            return
        if not set(header) - set(cols):             # only ever widen, never drop
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    r.pop(None, None)              # stray surplus from a prior append
                    w.writerow({k: r.get(k, "") for k in cols})
            tmp.replace(path)
            print(f"  [metrics] widened {path.name} to {len(cols)} columns")
    except Exception as e:  # noqa: BLE001 — monitoring must not break the turn
        print(f"  [metrics] header migration skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════
# RunMetrics — accumulates one row, then .write() appends it.
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class RunMetrics:
    source: str = "server"
    applicant_id: str = ""
    stage: str = ""
    turn: int = 0
    is_resume: bool = False
    model: str = MODEL   # LLM model id in effect for this stream
    endpoint: str = BASE_URL   # which backend served it — see FIELDNAMES note

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    route: str = ""
    tool_calls: int = 0
    interrupted: bool = False
    error: str = ""

    # ── North Star: time-to-letter ─────────────────────────────────────────
    # A letter is not produced in one stream. `full_ipa` runs to the HITL gate and
    # STOPS; the letter only becomes real when the RM approves and the resume
    # stream re-renders it as the final PDF. So the metric a bank cares about —
    # "how long from asking to holding the letter" — spans several streams, while
    # latency_s deliberately measures only one. `case_elapsed_s` lets the caller
    # carry the earlier turns' wall-clock in, so time_to_letter_s can be the true
    # end-to-end span rather than just the last leg.
    letter_produced: bool = False
    case_elapsed_s: float = 0.0

    # node_visits[agent] = how many times that agent's *_llm node ran this stream.
    # loop_count = sum(visits-1) over agents seen >1 time (a tool loop re-enters
    # the same llm node); path is the ordered list of agents as they first appear.
    _node_visits: dict = field(default_factory=dict)
    _path: list = field(default_factory=list)

    # Audit-log completeness. CLAUDE.md's audit convention says EVERY node must
    # append a LogEntry to state["audit"]. A node that runs silently produces a
    # gap in the trail a human auditor would follow — so the KPI is not "how many
    # entries" but "did every node that ran actually log". We track the set of
    # nodes seen and the subset that emitted at least one entry.
    audit_entries: int = 0
    _nodes_ran: set = field(default_factory=set)
    _nodes_logged: set = field(default_factory=set)

    # Which tools were actually invoked, in first-call order. Feeds tools_correct
    # (expected ⊆ called) and is the evidence trail when a case fails.
    _tools_called: list = field(default_factory=list)

    # Tool-layer authorisation. The front-end lets an RM switch borrower, so
    # applicant_id is client-controlled (an IDOR surface). The honest check is not
    # "did a prompt trick the model into naming another customer" but "did any
    # tool_call actually carry an applicant_id other than this session's" — a
    # deterministic comparison of tool args against the session principal, needing
    # no LLM judge. Rows record the count and which IDs leaked.
    _leaked_ids: set = field(default_factory=set)

    # Golden-set labels (eval rows only; run_evals.py sets these).
    query_id: str = ""
    trial: int = 1
    expected_route: str = ""
    expected_stage: str = ""
    success: object = ""          # 1 | 0 | "" — "" means "not an eval row"
    routing_correct: object = ""
    tools_correct: object = ""

    _t0: float = field(default_factory=time.perf_counter)

    # ── event hooks (called by the stream driver) ──────────────────────────
    def visit_node(self, agent: str) -> None:
        """Record that agent's llm node ran (called once per node output)."""
        if not agent:
            return
        if agent not in self._node_visits:
            self._path.append(agent)
        self._node_visits[agent] = self._node_visits.get(agent, 0) + 1

    # Nodes NOT bound by the audit convention, so excluded from the completeness
    # denominator. LangGraph's built-in ToolNode ("tools") returns only messages
    # and cannot append to state["audit"] — but the tool call it ran is already
    # audited by the adjacent *_llm node (the AIMessage's tool_calls in and the
    # ToolMessage result out both land in that node's audit). Counting `tools` as
    # an unaudited node would pin audit_complete below 100% for every turn that
    # uses a tool, measuring a structural fact about a framework component rather
    # than a gap in OUR trail — which is the KPI.
    _AUDIT_EXEMPT_NODES = frozenset({"tools"})

    def saw_node(self, node: str) -> None:
        """Record that a graph node bound by the audit convention produced output.

        Every node WE author — orchestrator, each agent, the *_llm nodes,
        hitl_review, replan — must log. The built-in ToolNode is exempt (see
        _AUDIT_EXEMPT_NODES); its work is audited by the adjacent *_llm node.
        """
        if node and node not in self._AUDIT_EXEMPT_NODES:
            self._nodes_ran.add(node)

    def add_audit_entries(self, node: str, n: int) -> None:
        """Record that `node` emitted n audit entries (n may be 0)."""
        if not node:
            return
        self.audit_entries += n
        if n > 0:
            self._nodes_logged.add(node)

    def add_tool_call(self, name: str = "", args: dict | None = None, n: int = 1) -> None:
        """Record one tool invocation, its name, and check it for cross-customer access.

        The leakage check compares the applicant_id in the tool's arguments against
        the session's applicant_id. Any tool reaching for a DIFFERENT customer's row
        is a tool-layer authorisation failure — regardless of what the prompt said or
        why the model did it. Deterministic; no LLM judge.
        """
        self.tool_calls += n
        if name:
            self._tools_called.append(name)

        if not isinstance(args, dict) or not self.applicant_id:
            return
        for key in ("applicant_id", "aid"):
            other = args.get(key)
            if other and str(other).strip() and str(other).strip() != self.applicant_id:
                self._leaked_ids.add(str(other).strip())

    def set_route(self, route: str) -> None:
        if route:
            self.route = route

    def check_letter(self, *, final_only: bool = False) -> bool:
        """Ask utils.letter_store whether this (applicant, stage) produced a letter.

        Kept here, not in the callers, so the eval and the server can never disagree
        about what "produced a letter" means. Imported lazily: telemetry sits below
        utils.letter_store in the dependency order and must not import it at module
        load, or `utils.tools` -> telemetry -> letter_store cycles.

        Two evidence levels, because the PDF is rendered OUTSIDE the graph:
          * `letter_store.get(...)` — an actual rendered PDF. Only ever true on the
            server path; `stream.py` renders it, `draft_letter` does not.
          * `letter_store.get_body(...)` — the drafting agent called `draft_letter`,
            registering the recipient and the cleared figures. This is the strongest
            signal available to the offline eval, which drives `graph.stream()`
            directly and so never reaches the renderer at all.

        `final_only` asks for the approved (non-DRAFT) PDF and is meaningful only on
        the server; it is ignored when no PDF was rendered but a letter was
        registered, otherwise the eval would score every case as letter-less.
        """
        try:
            from utils import letter_store
        except Exception:  # noqa: BLE001 — monitoring must not break the turn
            return self.letter_produced

        stage = (self.stage or "").upper()
        if not self.applicant_id or not stage:
            return self.letter_produced

        drafts = (False,) if final_only else (False, True)
        rendered = any(letter_store.get(self.applicant_id, stage, d) for d in drafts)
        registered = letter_store.get_body(self.applicant_id, stage) is not None
        if rendered or registered:
            self.letter_produced = True
        return self.letter_produced

    # ── derived fields ─────────────────────────────────────────────────────
    @property
    def latency_s(self) -> float:
        return round(time.perf_counter() - self._t0, 3)

    @property
    def time_to_letter_s(self) -> float | str:
        """Wall-clock from the case's first RM message to the letter existing.

        Blank (not 0) when no letter was produced: a case that never reached a
        letter has no time-to-letter, and writing 0 would drag the median down with
        a value that means the opposite of fast.
        """
        if not self.letter_produced:
            return ""
        return round(self.case_elapsed_s + (time.perf_counter() - self._t0), 3)

    @property
    def node_visits(self) -> int:
        return sum(self._node_visits.values())

    @property
    def loop_count(self) -> int:
        return sum(v - 1 for v in self._node_visits.values() if v > 1)

    @property
    def path(self) -> str:
        return ">".join(self._path)

    @property
    def nodes_total(self) -> int:
        return len(self._nodes_ran)

    @property
    def nodes_audited(self) -> int:
        return len(self._nodes_logged & self._nodes_ran)

    @property
    def audit_complete(self) -> bool:
        """Every node that ran also logged at least one audit entry.

        Vacuously true when no node ran (an errored stream), which is why the
        raw nodes_audited/nodes_total are written alongside — the report should
        aggregate the ratio over rows where nodes_total > 0.
        """
        return not (self._nodes_ran - self._nodes_logged)

    @property
    def unaudited_nodes(self) -> list:
        """Nodes that ran without logging — the actual defect, for debugging."""
        return sorted(self._nodes_ran - self._nodes_logged)

    @property
    def tools_called(self) -> list:
        """Distinct tools invoked, in first-call order.

        De-duplicated because tools_correct asks "was the needed tool used", not
        "how many times" — a tool loop calling get_profile twice is not two tools.
        """
        seen, out = set(), []
        for t in self._tools_called:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @property
    def leaked_ids(self) -> list:
        """Other customers' applicant_ids that appeared in tool arguments."""
        return sorted(self._leaked_ids)

    @property
    def leakage(self) -> bool:
        return bool(self._leaked_ids)

    def score_against_golden(self, expected_tools: list) -> None:
        """Set tools_correct: every expected tool was actually called.

        Subset, not equality — a model that also reads an extra harmless table has
        not made an error. The metric asks "did it fetch what the answer required".
        """
        self.tools_correct = int(set(expected_tools) <= set(self.tools_called))

    # ── persistence ────────────────────────────────────────────────────────
    def to_row(self) -> dict:
        tok = token_counter.read()
        cost = pricing.estimate(self.model, tok["prompt"], tok["completion"])
        row = {
            "run_id": self.run_id,
            "ts": self.ts,
            "source": self.source,
            "applicant_id": self.applicant_id,
            "stage": self.stage,
            "turn": self.turn,
            "is_resume": int(self.is_resume),
            "route": self.route,
            "model": self.model,
            "endpoint": self.endpoint,
            "latency_s": self.latency_s,
            "tokens_prompt": tok["prompt"],
            "tokens_completion": tok["completion"],
            "tokens_total": tok["total"],
            "cost_est_usd": "" if cost is None else cost,
            "llm_calls": tok["calls"],
            "loop_count": self.loop_count,
            "tool_calls": self.tool_calls,
            "node_visits": self.node_visits,
            "path": self.path,
            "audit_entries": self.audit_entries,
            "nodes_audited": self.nodes_audited,
            "nodes_total": self.nodes_total,
            "audit_complete": int(self.audit_complete),
            "interrupted": int(self.interrupted),
            "letter_produced": int(self.letter_produced),
            "time_to_letter_s": self.time_to_letter_s,
            "error": self.error,
            # Golden-set columns. Blank on live rows (no labels to score against);
            # run_evals.py populates them. `tools_called` and `leakage` are the
            # exception — they need no golden label, so they are filled on EVERY row
            # and the leakage rate can be read off live traffic too.
            "query_id": self.query_id,
            "trial": self.trial,
            "expected_route": self.expected_route,
            "expected_stage": self.expected_stage,
            "success": self.success,
            "routing_correct": self.routing_correct,
            "tools_correct": self.tools_correct,
            "tools_called": "|".join(self.tools_called),
            "leakage": int(self.leakage),
            # Agentic-RAG observability. Drained from the thread-local rewrite_probe
            # the search tool wrote to during this run (see _RewriteProbe). Filled on
            # every row; a run that never touched policy RAG reports all-zero.
            **self._rewrite_row(),
        }
        return row

    def _rewrite_row(self) -> dict:
        """Read the rewrite probe for this run into its four CSV columns."""
        rw = rewrite_probe.read()
        return {
            "policy_searches": rw["searches"],
            "rewrite_fired": rw["fired"],
            "rewrite_fused": rw["fused"],
            "rewrite_noop": rw["noop"],
        }

    def write(self, path: Path | str = RUNS_CSV) -> None:
        """Append this run as one row, writing the header if the file is new.

        Best-effort: a metrics write must never break a live RM turn, so any
        I/O error is swallowed (printed, not raised).
        """
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not path.exists() or path.stat().st_size == 0
            if not new_file:
                _migrate_header(path)
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                if new_file:
                    w.writeheader()
                w.writerow(self.to_row())
        except Exception as e:  # noqa: BLE001 — monitoring must not break the turn
            print(f"  [metrics] failed to write run row: {e}")
