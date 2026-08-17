"""LLM client — ThrottledChatOpenAI over any OpenAI-compatible endpoint.

The endpoint is chosen by ``utils.config`` (OpenRouter by default; an internal
gateway when ``INTERNAL_LLM_BASE_URL`` is set) — nothing here is
provider-specific, since both speak the same ``/chat/completions`` contract.

Wraps ``langchain_openai.ChatOpenAI`` with two cross-cutting concerns required
by free-tier OpenRouter models:
  - a request-frequency throttle (see config.MIN_INTERVAL), and
  - ``_strip_think()`` which removes ``<think>…</think>`` reasoning spans from
    responses (DeepSeek / reasoning models emit them).

``make_llm()`` is the factory the graph layer uses; ``_strip_think`` is exposed
because it is also handy in tests and ad-hoc post-processing.
"""

import json
import re
import time
import uuid

import httpx
from langchain_openai import ChatOpenAI as _BaseChatOpenAI

from utils import config
from utils.config import (
    BASE_URL,
    LLM_API_KEY,
    MODEL,
    SSL_CA_BUNDLE,
    VERIFY_SSL,
    _endpoint_error,
)

# MIN_INTERVAL / MAX_RETRIES are deliberately NOT from-imported: they differ per
# endpoint (see config._resolve_throttle) and are rebound by config._refresh(), so
# a from-import would freeze one route's pacing onto the other.

_LAST_CALL_TIME = 0.0
_ENDPOINT_LOGGED = False

# Transient upstream/gateway failures worth retrying. OpenRouter often wraps an
# upstream timeout in an HTTP-200 body like {"error": {"code": 504, "message":
# "The operation was aborted"}}, which the openai SDK surfaces as a ValueError
# (NOT an APIStatusError), so the SDK's own ``max_retries`` never catches it.
# We retry on these gateway codes and on aborted/timeout phrasing regardless of
# the exception type the SDK chose to raise.
_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504, 524}
_RETRYABLE_PHRASES = ("aborted", "timeout", "timed out", "gateway", "temporarily unavailable")

# ── Learning the real context window from the provider's own 400 ─────────
# config.CONTEXT_LIMIT carries a per-endpoint DEFAULT (32768 for the internal
# gateway), but a single-model gateway reports its model as "default", so no
# heuristic can confirm that number is right for the box actually being served.
# The provider does tell us, though — in the 400 it raises:
#
#   "This model's maximum context length is 32768 tokens. However, you requested
#    8192 output tokens and your prompt contains at least 24577 input tokens"
#
# Parsing it turns a wrong default into a self-correcting one: the FIRST overflow
# on an unexpected gateway teaches the process the true window, and callers that
# budget against config.CONTEXT_LIMIT tighten from then on. Deliberately narrow —
# it only ever LOWERS the limit, never raises it, so a misparse cannot talk the
# app into sending more than it does today.
_CTX_LIMIT_RE = re.compile(r"maximum context length is\s+(\d+)\s+tokens", re.I)


def _learn_context_limit(exc: Exception) -> bool:
    """If `exc` is a context-overflow 400, record the window it names. True if learnt.

    Never raises: this runs on an error path that is already failing, and a bug
    here must not mask the original exception the caller needs to see.
    """
    try:
        m = _CTX_LIMIT_RE.search(str(exc))
        if not m:
            return False
        found = int(m.group(1))
        current = getattr(config, "CONTEXT_LIMIT", 0) or 0
        # Only tighten (or set a previously-uncapped route), never loosen.
        if found > 0 and (current == 0 or found < current):
            config.CONTEXT_LIMIT = found
            # Logging is best-effort and must not affect the outcome: this repo runs
            # on a GBK console (see the `py` launcher note), where printing a non-CP936
            # glyph raises UnicodeEncodeError — which the outer `except` would swallow
            # AFTER the limit was already recorded, making a successful learn report
            # failure. Keep the message ASCII and guard it anyway.
            try:
                print(f"  [warn] endpoint reports a {found}-token context window; "
                      f"recorded (was {current or 'uncapped'})")
            except Exception:  # noqa: BLE001 — console encoding must not gate the result
                pass
            return True
    except Exception:  # noqa: BLE001 — never mask the real error
        pass
    return False


def _throttle() -> None:
    global _LAST_CALL_TIME
    wait = config.MIN_INTERVAL - (time.time() - _LAST_CALL_TIME)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_TIME = time.time()


def _is_retryable(exc: Exception) -> bool:
    """True if the error looks like a transient upstream/gateway hiccup.

    Inspects both any numeric code carried on the exception (or in a wrapped
    dict, OpenRouter's habit) and the message text, so it catches the 504
    ValueError case the SDK retry misses.
    """
    # A numeric status/code attribute, if present.
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and val in _RETRYABLE_CODES:
            return True
    text = str(getattr(exc, "args", "") or exc).lower()
    # OpenRouter's {"error": {"code": 504, ...}} renders the code into the text.
    if any(str(c) in text for c in _RETRYABLE_CODES):
        return True
    return any(p in text for p in _RETRYABLE_PHRASES)


def _log(message: str) -> None:
    """Print a diagnostic without letting the console decide whether the run survives.

    This repo runs on a GBK console (see the `py` launcher note), where printing a
    glyph outside CP936 raises UnicodeEncodeError. That matters here because these
    messages are emitted from inside broad `except` blocks and mid-mutation recovery
    code: a raising print does not just lose the line, it aborts the very repair it
    was reporting. Keep messages ASCII, and swallow whatever the console still
    objects to.
    """
    try:
        print(message)
    except Exception:  # noqa: BLE001 — logging must never change behaviour
        pass


def _strip_think(text: str) -> str:
    if not text:
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# Chat-template scaffolding that leaks into the CONTENT when a Llama-family model
# keeps generating past its turn. Observed 2026-08-04 in the RM's chat bubble:
#   "[loan_calc_result = calculate_loan_tool()]assistant\n\ncalculate_loan_tool()"
# — a bare role token where a new turn should have started. These markers are the
# serving layer's business, never part of an answer, so they are removed wherever
# they appear. The bare-word forms are anchored to a line edge so the ordinary
# English words ("the assistant will call you") are left alone.
_ROLE_TOKENS = re.compile(
    r"<\|(?:eot_id|start_header_id|end_header_id|begin_of_text|im_start|im_end)\|>"
    r"|^\s*(?:assistant|user|system)\s*$"
    r"|(?<=[\]\)\}])(?:assistant|user|system)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _strip_role_tokens(text: str) -> str:
    """Remove chat-template role markers that leaked into the answer text."""
    if not text or not isinstance(text, str):
        return text
    return _ROLE_TOKENS.sub("", text).strip()


# Control characters that must never appear raw in a JSON string sent to the API.
# A free-tier model occasionally emits one inside its answer (an ESC/BEL, a stray
# backslash sequence), and that text is then replayed as an input message on the
# next turn — where the provider rejects the request body with HTTP 400
# "invalid escaped character in string" / "Extra data", aborting the whole graph
# run AFTER several agents have already worked. Stripping them on the way IN to
# every LLM call neutralises the poison regardless of which agent produced it.
_CTRL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# A backslash not starting a valid JSON escape (\" \\ \/ \b \f \n \r \t \uXXXX).
# Such a lone backslash is what triggers "invalid escaped character in string".
_BAD_ESCAPE = re.compile(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})')


def _sanitize_text(text: str) -> str:
    """Strip raw control chars and neutralise lone backslashes so the text is safe
    to serialise into an API request body. Idempotent; leaves normal text intact."""
    if not text or not isinstance(text, str):
        return text
    text = _CTRL_CHARS.sub("", text)
    text = _BAD_ESCAPE.sub(r"\\\\", text)   # lone "\" -> escaped "\\"
    return text


def _sanitize_obj(obj):
    """Recursively sanitise every string inside a str / list / dict, returning a
    cleaned copy (dicts/lists rebuilt, scalars returned as-is)."""
    if isinstance(obj, str):
        return _sanitize_text(obj)
    if isinstance(obj, list):
        return [_sanitize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize_obj(v) for k, v in obj.items()}
    return obj


def _repair_json_args(raw: str) -> str:
    """Normalise a tool-call `arguments` STRING into clean, re-parseable JSON.

    Free-tier models frequently emit a broken `tool_calls.function.arguments`
    value — trailing junk after the object ("Extra data"), an invalid \\escape,
    control chars, or two concatenated objects. The provider then 400s the NEXT
    request that replays it. Every observed 400 in eval/runs.csv traces to this.

    Strategy, in order:
      1. If it already parses, re-dump it (canonical, no stray whitespace).
      2. Character-clean it (control chars + lone backslashes) and re-try.
      3. Decode just the FIRST JSON value with raw_decode and drop any trailing
         "Extra data", then re-dump that.
    Returns the original string only if nothing parses (better to send it as-is
    than to fabricate)."""
    if not isinstance(raw, str) or not raw.strip():
        return raw
    for candidate in (raw, _sanitize_text(raw)):
        try:
            return json.dumps(json.loads(candidate), ensure_ascii=False)
        except Exception:
            pass
        # Take just the first complete JSON value, discarding trailing Extra data.
        try:
            obj, _end = json.JSONDecoder().raw_decode(candidate.lstrip())
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            continue
    return raw


# ── Text-emitted tool calls (Llama-family recovery) ──────────────────────
# Llama-4 models are markedly weaker at NATIVE function calling than gemma was:
# instead of populating the structured `tool_calls` field, they often write the
# call into the message TEXT. That is not a cosmetic problem. The graph decides a
# turn is finished with `is_final = not resp.tool_calls` (graph.py), so a
# text-emitted call means: the tool never runs, the turn is treated as a final
# answer, and stream.py renders the raw JSON straight into the RM's chat bubble.
# Same class of defect as the 2026-07-22 orchestrator routing-JSON leak, from a
# different source.
#
# We recover the call rather than merely suppressing the text, because the intent
# IS present — it is just in the wrong field. Four emission syntaxes are handled;
# all four have been observed from Llama-family checkpoints, and which one a
# given deployment produces depends on the chat template the provider applies:
#
#   1. <|python_tag|>{"name": "calculate_loan", "parameters": {...}}
#   2. a bare or ```json-fenced  {"name": ..., "arguments": {...}}
#   3. <function=calculate_loan{"loan_amount": 500000}></function>
#   4. Python-call syntax, usually a bracketed BATCH and not JSON at all:
#        [get_loan_application(applicant_id='APP0006'),
#         get_profile(applicant_id='APP0006')]
#      Observed 2026-08-03 on the Reprice agent (screenshot evidence): the audit
#      log showed "FINAL ANSWER → Tool", i.e. the whole plan leaked into the RM's
#      bubble and not one of the three tools ran.
#
# NOTE: this only fixes "the model meant to call a tool but formatted it wrong".
# It deliberately does NOT invent a call when the model emitted none — inferring
# which tool was intended would be fabrication, precisely the failure mode the
# InjectedState fix (see graph.py's draft_letter note) exists to prevent.
_PYTHON_TAG = re.compile(r"<\|python_tag\|>", re.I)
# <function=NAME{json}>  — the JSON body is grabbed by brace matching, not regex,
# so nested objects survive.
_FUNCTION_TAG = re.compile(r"<function\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*", re.I)
_JSON_FENCE = re.compile(r"```(?:json|tool_call|python)?\s*", re.I)
# Syntax 4: NAME( … ) — the argument list is taken by paren matching, not regex, so
# nested parens/brackets in a value survive. Anchored on a bound tool name by the
# caller, which is what keeps ordinary prose like "call the bank (today)" out.
_PY_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _balanced_json(text: str, start: int) -> tuple[dict | None, int]:
    """Decode one JSON object beginning at/after `start`; return (obj, end_index).

    Uses raw_decode from the first '{' so trailing prose after the object is
    ignored, which is exactly how these models emit it (JSON then commentary).
    Returns (None, start) when nothing parses.
    """
    brace = text.find("{", start)
    if brace == -1:
        return None, start
    try:
        obj, end = json.JSONDecoder().raw_decode(_sanitize_text(text[brace:]))
    except Exception:
        return None, start
    if not isinstance(obj, dict):
        return None, start
    return obj, brace + end


def _match_paren(text: str, open_idx: int) -> int:
    """Index just past the ')' matching the '(' at `open_idx`, or -1 if unbalanced.

    Quote-aware so a paren inside a string value (e.g. name='Lim (Alice)') does not
    close the call early.
    """
    depth = 0
    quote = ""
    i = open_idx
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _py_arg_value(node) -> tuple[bool, object]:
    """Read one keyword-argument value node. Returns (ok, value).

    Two layers, strictest first:

    1. `ast.literal_eval` — the correct reading for everything well-formed:
       quoted strings, numbers, True/False/None, nested lists/dicts. Nothing is
       executed.
    2. Bare identifiers / dotted names (`property_type=Condo`,
       `applicant_id=APP0001`) — observed live 2026-08-04 on the Reprice track.
       These are unquoted string values; Python reads them as variable names, so
       layer 1 raises and, under the all-or-nothing rule below, ONE such value
       used to void an otherwise perfect call and leak the whole batch. In a tool
       argument position a bare name cannot mean anything else, so it is taken as
       its source text. True/False/None are ast.Constant, not ast.Name, so they
       never reach this layer and stay real literals.
    3. Arithmetic over CONSTANTS ONLY (`cash_cpf_available=511000 + 86780.42`,
       observed 2026-08-04 on the compliance track). Every operand is stated in the
       text, so folding it is reading what the model wrote, not inferring: the same
       kind of normalisation as reading `1_200_000` as 1200000. The whole call used
       to be refused over this, meaning the tool never ran at all.

    An expression referencing ANY name (`income*0.55`, `salary + bonus`) is
    refused: the operand is not in the text, so a value would have to be invented —
    exactly the fabrication this layer exists to prevent. Ditto calls and
    f-strings. Refusing returns ok=False, and the caller then rejects the whole
    call rather than hand a tool a value the model did not actually state.
    """
    import ast

    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    if isinstance(node, ast.Name):                       # Condo -> "Condo"
        return True, node.id
    if isinstance(node, ast.Attribute):                  # PropertyType.CONDO
        try:
            return True, ast.unparse(node)
        except Exception:
            return False, None
    if isinstance(node, (ast.BinOp, ast.UnaryOp)):
        # This text comes from the model, so the expression is untrusted input and
        # the walk below is a whitelist, not a sanity check.
        for n in ast.walk(node):
            if isinstance(n, ast.BinOp):
                # `**` is barred outright: 10**10**10 hangs the process for minutes
                # and exhausts memory before any check on the RESULT could run, and
                # no real tool argument is written as an exponent.
                if not isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    return False, None
            elif isinstance(n, ast.UnaryOp):
                if not isinstance(n.op, (ast.UAdd, ast.USub)):
                    return False, None
            elif isinstance(n, (ast.operator, ast.unaryop)):
                continue                # the op nodes themselves, vetted above
            elif not isinstance(n, (ast.Constant, ast.Load)):
                # Anything else — a Name, call, attribute, f-string, comprehension —
                # references something not stated in the text. Inventing that
                # operand is the fabrication this layer exists to prevent.
                return False, None
            if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
                return False, None      # numeric operands only; blocks 'a'*100000
        try:
            value = eval(compile(ast.Expression(node), "<arg>", "eval"),  # noqa: S307
                         {"__builtins__": {}}, {})
        except Exception:
            return False, None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False, None
        return True, value
    return False, None


def _parse_py_call_args(arglist: str) -> dict | None:
    """Parse `applicant_id='APP0006', months=12` into a dict, or None.

    Values go through `_py_arg_value`, so nothing is ever executed. Positional args
    are rejected: without the tool schema we cannot know which parameter they bind
    to, and guessing an order is how you silently compute the right number for the
    wrong field.

    Recovery stays all-or-nothing per call: if any value cannot be read, the whole
    call is rejected. Dropping just the unreadable argument would hand the tool a
    call missing a required field, turning a visible leak into a silently wrong
    number, which is the worse failure.
    """
    import ast

    try:
        tree = ast.parse(f"_f({arglist})", mode="eval")
    except SyntaxError:
        return None
    call = tree.body
    if not isinstance(call, ast.Call) or call.args:      # positional -> reject
        return None
    args: dict = {}
    for kw in call.keywords:
        if kw.arg is None:                               # **kwargs splat -> reject
            return None
        ok, value = _py_arg_value(kw.value)
        if not ok:
            return None
        args[kw.arg] = value
    return args


def _py_call_reject_reason(text: str, allowed: set[str]) -> str | None:
    """Why a call-shaped span was refused, for the log. None when nothing was refused.

    Diagnostic only — never affects what is recovered. It re-walks the same spans so
    the operator can tell "the model wrote junk" apart from "our parser is too
    strict", which is the distinction that decides whether a leak needs a code fix.
    """
    import ast

    for m in _PY_CALL.finditer(text):
        if m.group(1) not in allowed:
            continue
        end = _match_paren(text, m.end() - 1)
        if end == -1:
            return "unbalanced parens"
        arglist = text[m.end():end - 1]
        if not arglist.strip():
            continue                                     # no-arg: treated as prose
        try:
            call = ast.parse(f"_f({arglist})", mode="eval").body
        except SyntaxError:
            return "arglist is not valid Python syntax"
        if getattr(call, "args", None):
            return "positional args (cannot bind without the tool schema)"
        for kw in call.keywords:
            if kw.arg is None:
                return "**kwargs splat"
            if not _py_arg_value(kw.value)[0]:
                kind = type(kw.value).__name__
                return f"unreadable value for '{kw.arg}' ({kind}) — not a stated literal"
    return None


def _is_standalone_noarg_call(text: str, start: int, end: int,
                              name: str, noarg: set[str]) -> bool:
    """True if `name()` at [start:end) reads as a CALL rather than a mention.

    Two conditions, both required. The tool must genuinely need no arguments (else
    empty parens are a fragment, not a call), and the call must stand on its own —
    nothing but whitespace or assignment/bracket punctuation around it.

    The second condition is what keeps prose intact. "per calculate_loan_tool(), the
    instalment is 4,490" names the tool mid-sentence: recovering that would fire a
    spurious calculation AND delete words from a real answer, which is the failure
    the blanket no-arg rejection used to prevent. The observed real emissions —
    "calculate_loan_tool()" alone on a line, and "[x = calculate_loan_tool()]" — both
    satisfy it.
    """
    if name not in noarg:
        return False
    # Look at the raw neighbours, NOT stripped ones: a newline before the call is
    # what marks it as its own statement, and rstrip() would erase that evidence.
    before_raw = text[:start]
    after_raw = text[end:]
    # Starts the text, follows a line break, or follows assignment/bracket/list
    # punctuation ("[x = tool()]"). A space alone is not enough — that is how a
    # mid-sentence mention looks.
    head_ok = (not before_raw.strip()) or before_raw.rstrip(" \t")[-1:] in ("[", "(", "{", "=", ",", ":", ";", "\n")
    # Ends the text, or is followed by a line break / closing bracket / full stop.
    # A comma or a word means the sentence continues around it, i.e. prose.
    tail_ok = (not after_raw.strip()) or after_raw.lstrip(" \t")[:1] in ("]", ")", "}", "\n", ".")
    return head_ok and tail_ok


def _extract_py_calls(text: str, allowed: set[str],
                      noarg: set[str] | None = None) -> tuple[list[dict], list[tuple[int, int]]]:
    """Find Python-call-syntax tool calls (syntax 4). Returns (calls, spans).

    Only names in `allowed` are even considered, so this cannot mistake prose for a
    call. A surrounding bracketed batch — the observed "[a(...), b(...)]" shape — has
    its brackets swallowed into the removed span so no "[ , ]" debris survives.

    `noarg` names the bound tools whose schema requires nothing, so `tool()` is a
    complete call rather than prose. Empty by default: without the schema we cannot
    tell `calculate_loan_tool()` (a real call) from "per calculate_loan_tool(), the
    instalment is …" (a reference), and the safe reading is prose.
    """
    calls: list[dict] = []
    spans: list[tuple[int, int]] = []
    unparsed = False              # a call-shaped span we could not read
    for m in _PY_CALL.finditer(text):
        name = m.group(1)
        if name not in allowed:
            continue
        end = _match_paren(text, m.end() - 1)
        if end == -1:
            continue
        args = _parse_py_call_args(text[m.end():end - 1])
        if args is None:
            # A bound tool name with a real argument list is unmistakably a call,
            # even though its values did not parse. Record nothing, but remember
            # it: promoting the siblings alone would let the span widening below
            # delete THIS call's text without ever running it.
            unparsed = True
            continue
        # An empty arg list is ambiguous: it is either a real call to a tool that
        # needs nothing, or prose naming a tool ("per calculate_loan_tool(), the
        # instalment is …"). The tool's own schema settles it — recover only when
        # the tool genuinely requires no arguments.
        #
        # Observed 2026-08-04: llama-4 emitted "[loan_calc_result =
        # calculate_loan_tool()]assistant\ncalculate_loan_tool()" and the blanket
        # rejection here dropped it, so the calculator never ran and that raw text
        # was rendered into the RM's chat. The old comment's premise ("every tool
        # here takes arguments") was simply untrue for calculate_loan_tool, whose
        # inputs all come from graph state via InjectedState.
        if not args and not _is_standalone_noarg_call(text, m.start(), end, name,
                                                      noarg or set()):
            continue
        tc = _as_tool_call({"name": name, "arguments": args}, allowed)
        if tc:
            calls.append(tc)
            spans.append((m.start(), end))
        else:
            unparsed = True       # e.g. args that survived parsing but not coercion
    if not calls:
        return [], []
    # Partial batch: at least one sibling call could not be read. Recovering only the
    # readable ones is the dangerous outcome — the agent would proceed believing it
    # has data it never fetched. Bail out entirely so the turn still looks final and
    # the text is left intact for the caller's marker-based stripping to handle.
    if unparsed:
        return [], []
    # Widen the removed region over a wrapping [...] batch and its separators, so a
    # multi-call batch leaves no bracket/comma residue in the bubble.
    lo = min(s for s, _ in spans)
    hi = max(e for _, e in spans)
    pre, post = text[:lo], text[hi:]
    open_br = pre.rstrip()
    if open_br.endswith("["):
        lo = len(open_br) - 1
        close = re.compile(r"\s*,?\s*\]").match(post)
        if close:
            hi += close.end()
    return calls, [(lo, hi)]


def _as_tool_call(obj: dict, allowed: set[str] | None) -> dict | None:
    """Coerce a decoded {"name":..., "arguments"/"parameters":...} dict into a
    LangChain tool_call, or None if it is not a tool call we may execute.

    A name outside `allowed` is rejected: the bound tool list is the authority on
    what this agent can call, and honouring a hallucinated name would either blow
    up in ToolNode or, worse, reach a tool the agent was deliberately denied (the
    withheld-compare_packages case in graph.py, and the tool-whitelist boundary
    that the toC role relies on as a hard security edge).
    """
    # "action"/"action_input" is the ReAct shape. Observed 2026-08-04 on the
    # COMPARE track: the model emitted {"action": "interest_savings_tool",
    # "action_input": "{...}"} as plain text, and because neither key was
    # recognised the blob was rendered verbatim into the RM's chat bubble.
    name = (obj.get("name") or obj.get("function") or obj.get("tool")
            or obj.get("action"))
    if not isinstance(name, str) or not name.strip():
        return None
    name = name.strip()
    if allowed is not None and name not in allowed:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if args is None:
        args = obj.get("action_input")
    if args is None:
        args = obj.get("args", {})
    if isinstance(args, str):                     # args as a JSON string
        try:
            args = json.loads(_repair_json_args(args))
        except Exception:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": name, "args": _sanitize_obj(args),
            "id": f"call_recovered_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def _extract_text_tool_calls(text: str, allowed: set[str] | None,
                             noarg: set[str] | None = None) -> tuple[list[dict], str]:
    """Pull text-emitted tool calls out of `text`.

    Returns (tool_calls, remaining_text). `remaining_text` has each recovered span
    removed, so the caller can blank the content and let the graph see a proper
    tool-calling turn. Recovery is all-or-nothing per span: a span that cannot be
    turned into a valid call is still STRIPPED when it carried an unmistakable
    marker (python_tag / function tag), because leaking that marker into the RM's
    chat is the very bug being fixed — but a plain unparseable JSON blob is left
    alone, since it may be legitimate prose containing braces.
    """
    if not text or not isinstance(text, str):
        return [], text
    calls: list[dict] = []
    spans: list[tuple[int, int]] = []             # (start, end) to remove

    # 1 & 2: python_tag / json-fence markers, then any bare leading JSON object.
    for marker in (_PYTHON_TAG, _JSON_FENCE):
        for m in list(marker.finditer(text)):
            obj, end = _balanced_json(text, m.end())
            if obj is None:
                continue
            tc = _as_tool_call(obj, allowed)
            if tc:
                calls.append(tc)
                spans.append((m.start(), end))
            elif marker is _PYTHON_TAG:
                spans.append((m.start(), end))     # strip the marker regardless

    # 3: <function=NAME{...}>
    for m in list(_FUNCTION_TAG.finditer(text)):
        obj, end = _balanced_json(text, m.end())
        args = obj if isinstance(obj, dict) else {}
        tc = _as_tool_call({"name": m.group(1), "arguments": args}, allowed)
        # Consume the optional closing "></function>" so no tag debris remains.
        tail = re.compile(r"\s*>\s*(?:</function>)?").match(text, end if obj else m.end())
        stop = tail.end() if tail else (end if obj else m.end())
        if tc:
            calls.append(tc)
        spans.append((m.start(), stop))

    # 4: Python-call syntax, e.g. [get_profile(applicant_id='APP0006'), ...].
    # Gated on `allowed` being known, since the bound tool name is the only thing
    # separating a real call from prose that happens to look like one.
    if not calls and allowed:
        py_calls, py_spans = _extract_py_calls(text, allowed, noarg)
        if py_calls:
            calls.extend(py_calls)
            spans.extend(py_spans)

    # A bare tool-call object as the WHOLE message (no marker at all) — the most
    # common Llama-4 shape. Only accepted when the stripped text starts with '{'
    # and the object names a bound tool, so ordinary prose is never touched.
    if not calls:
        stripped = text.strip()
        if stripped.startswith("{"):
            obj, end = _balanced_json(text, 0)
            if obj is not None:
                tc = _as_tool_call(obj, allowed)
                if tc:
                    calls.append(tc)
                    spans.append((text.find("{"), end))

    if not spans:
        return calls, text
    out = text
    for start, end in sorted(spans, reverse=True):        # right-to-left: keep indices valid
        out = out[:start] + out[end:]
    # Tidy leftover fence debris / whitespace from the removed spans.
    out = re.sub(r"```", "", out)
    return calls, out.strip()


def _emission_syntax(text: str) -> str:
    """Name the syntax the model used, for the log. Cheap markers, checked in the
    same order the extractor tries them.

    Worth distinguishing: each syntax is a different model behaviour, and knowing
    which one is trending tells us whether a checkpoint changed its emission style
    (fix here) or a prompt regressed (fix in skill.md).
    """
    if _PYTHON_TAG.search(text):
        return "python_tag"
    if _FUNCTION_TAG.search(text):
        return "function-tag"
    if _JSON_FENCE.search(text):
        return "json-fence"
    stripped = text.strip()
    if stripped.startswith("{"):
        return "ReAct-action" if '"action"' in stripped[:200] else "bare-JSON"
    return "python-call"


def _recover_tool_calls(result, allowed: set[str] | None,
                        noarg: set[str] | None = None) -> None:
    """In-place: promote text-emitted tool calls on each generation to real ones.

    Only touches a message that has NO structured tool_calls — if the model got it
    right, nothing here applies. Never raises: a recovery bug must not turn a
    working call into a failed run.

    `allowed` empty/None means this call bound no tools, so there is nothing to
    recover INTO and any JSON in the content is data, not a call — bail out. (The
    filtering helpers treat None as "do not filter", which is the right default
    for them but the wrong one here; the distinction is load-bearing.)
    """
    if not allowed:
        return
    try:
        for gen in getattr(result, "generations", []) or []:
            msg = getattr(gen, "message", None)
            if msg is None or getattr(msg, "tool_calls", None):
                continue
            if not isinstance(getattr(msg, "content", None), str):
                continue
            original = msg.content
            calls, remaining = _extract_text_tool_calls(original, allowed, noarg)
            if not calls and remaining == original:
                # Nothing recovered and nothing stripped. Usually a plain answer, but
                # it is also how a leak reaches the RM's bubble, so say why when the
                # text was call-SHAPED. Silence here is what made the 2026-08-04
                # unquoted-value leak look like the patch had simply stopped working.
                reason = _py_call_reject_reason(original, allowed)
                if reason:
                    _log(f"  [warn] text tool call NOT recovered "
                         f"[{_emission_syntax(original)}] — {reason}; "
                         f"text left in the answer")
                continue
            if calls:
                msg.tool_calls = calls
                # Mirror into additional_kwargs so the message replays to the
                # provider as a normal assistant tool-call turn on the next hop.
                ak = dict(getattr(msg, "additional_kwargs", None) or {})
                ak["tool_calls"] = [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"],
                                  "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                    for c in calls
                ]
                msg.additional_kwargs = ak
                _log(f"  [warn] recovered {len(calls)} text-emitted tool call(s) "
                     f"[{_emission_syntax(original)}] "
                     f"({', '.join(c['name'] for c in calls)}) — model wrote them as text")
            else:
                # Spans were stripped but no call came out: an unmistakable marker
                # whose payload was unusable (e.g. an unbound tool name). The bubble
                # is safe, but the tool did NOT run, so this turn is silently thinner
                # than it looks — the one case worth noticing in a clean-looking run.
                _log(f"  [warn] stripped a text tool call [{_emission_syntax(original)}] "
                     f"without recovering it — no tool ran this turn")
            # Keep the surrounding narration (minus the call syntax itself) rather
            # than blanking it. It cannot leak into the RM's chat bubble: BOTH answer
            # paths are gated on there being no tool_calls — stream.py's AIMessage
            # harvest checks `not m.tool_calls`, and the a2a path only harvests when
            # graph.py's `is_final = not resp.tool_calls` is True. Recovering the
            # calls closes both gates, which is the exact mirror of the bug (empty
            # tool_calls left both open). Meanwhile this text IS the agent's plan, so
            # it stays useful as the thinking-block summary and in the audit trail.
            msg.content = remaining
    except Exception:  # noqa: BLE001 — recovery must never break a working call
        pass


def _sanitize_messages(messages):
    """Sanitise EVERY string carried by each outgoing message — not just `content`,
    but also the tool-call arguments and `additional_kwargs` (where OpenAI-format
    tool_calls live as a JSON `arguments` string). A lone backslash / control char
    anywhere in there is what makes the provider reject the request body with a 400,
    so all string-bearing fields are cleaned. Never raises."""
    try:
        for m in messages or []:
            content = getattr(m, "content", None)
            if isinstance(content, (str, list)):
                cleaned = _sanitize_obj(content)
                if cleaned != content:
                    m.content = cleaned
            # LangChain tool_calls: [{'name','args': {...}, 'id', 'type'}]
            tcs = getattr(m, "tool_calls", None)
            if isinstance(tcs, list) and tcs:
                for tc in tcs:
                    if isinstance(tc, dict) and isinstance(tc.get("args"), (dict, list, str)):
                        tc["args"] = _sanitize_obj(tc["args"])
            # OpenAI-format tool_calls carried in additional_kwargs, whose
            # `function.arguments` is a JSON STRING the provider re-parses — the
            # single most common source of the 400s (Extra data / invalid \escape /
            # "expected JSON"). Repair each arguments string into clean JSON, then
            # character-sanitise the rest of the kwargs.
            ak = getattr(m, "additional_kwargs", None)
            if isinstance(ak, dict) and ak:
                ak = _sanitize_obj(ak)          # char-clean first
                for tc in ak.get("tool_calls") or []:  # then normalise each args JSON
                    if isinstance(tc, dict):
                        fn = tc.get("function")
                        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                            fn["arguments"] = _repair_json_args(fn["arguments"])
                m.additional_kwargs = ak
    except Exception:  # noqa: BLE001 — sanitisation must never break the call
        pass
    return messages


def _record_token_usage(result) -> None:
    """Feed one LLM call's token usage to the telemetry token_counter (best-effort).

    Reads OpenRouter/OpenAI-format usage from result.llm_output["token_usage"],
    falling back to the AIMessage's usage_metadata if the top-level block is
    absent. A model that reports no usage at all leaves that stream's token
    columns at 0. Never raises: metrics must not break an LLM call, and the
    telemetry package may not be importable in every entry point (e.g. a bare
    notebook), so the import is guarded.
    """
    try:
        from utils.telemetry import token_counter
    except Exception:
        return
    try:
        usage = (getattr(result, "llm_output", None) or {}).get("token_usage") or {}
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        # Fallback: langchain also attaches usage_metadata on the message.
        if prompt is None and completion is None:
            for gen in getattr(result, "generations", []) or []:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None) or {}
                if meta:
                    prompt = meta.get("input_tokens")
                    completion = meta.get("output_tokens")
                    total = meta.get("total_tokens")
                    break
        token_counter.add(prompt=prompt or 0, completion=completion or 0, total=total or 0)
    except Exception:  # noqa: BLE001 — never break an LLM call over metrics
        pass


def _bound_tool_names(kwargs) -> set[str] | None:
    """Names of the tools bound for THIS call, or None when the call binds none.

    bind_tools() passes the OpenAI-format schema list through as kwargs["tools"],
    so this reflects the per-call tool list — including graph.py's
    bound_llm_excluding(), where a tool is deliberately withheld. Returning the
    real set is what lets _as_tool_call reject a name the agent may not call.
    None means "no tools bound", and recovery is skipped entirely.
    """
    tools = kwargs.get("tools")
    if not tools:
        return None
    names = set()
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        name = (fn or {}).get("name") if isinstance(fn, dict) else t.get("name")
        if isinstance(name, str) and name:
            names.add(name)
    return names or None


def _noarg_tool_names(kwargs) -> set[str]:
    """Bound tools that are legitimately callable with NO arguments.

    Read from the same bind_tools() schema list as _bound_tool_names, so it stays
    true as tools change instead of hardcoding a list that silently rots. A tool
    qualifies when its schema declares no required properties — `calculate_loan_tool`
    is the important case: every real parameter is optional (the borrower and the
    case's own figures are read from graph state via InjectedState, which is hidden
    from the schema the model sees), so `calculate_loan_tool()` is a COMPLETE call,
    not a fragment. `get_sora_rate` and `list_loan_packages` are the same shape.
    """
    tools = kwargs.get("tools")
    if not tools:
        return set()
    out = set()
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        params = fn.get("parameters")
        if not isinstance(name, str) or not name:
            continue
        required = (params or {}).get("required") or []
        if not required:
            out.add(name)
    return out


class ThrottledChatOpenAI(_BaseChatOpenAI):
    """ChatOpenAI + rate-limit guard + <think> stripping + gateway-error retry
    + recovery of tool calls a Llama-family model emitted as plain text."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Neutralise raw control chars / lone backslashes in the outgoing messages
        # so a free-tier model's earlier junk output can't make the provider reject
        # THIS request body with a 400 mid-run. This is the single choke point every
        # LLM call passes through, so one pass here covers every agent.
        messages = _sanitize_messages(messages)
        # Retry transient gateway failures (e.g. OpenRouter 504 "operation was
        # aborted") with exponential backoff. attempt 0 is the initial try; we
        # make up to MAX_RETRIES additional attempts before re-raising.
        last_exc = None
        max_retries = config.MAX_RETRIES        # read once so the loop is consistent
        for attempt in range(max_retries + 1):
            _throttle()
            try:
                result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                break
            except Exception as exc:                       # noqa: BLE001 — re-raised below if not retryable
                # A context overflow is NOT retryable — resending the identical
                # oversized prompt fails identically — but the error names the real
                # window, so record it on the way past. The next call budgets against
                # the true number instead of our per-endpoint guess.
                _learn_context_limit(exc)
                if attempt >= max_retries or not _is_retryable(exc):
                    raise
                last_exc = exc
                backoff = 2.0 * (2 ** attempt)             # 2s, 4s, 8s, …
                _log(f"  [warn] transient LLM error ({exc}); retry "
                     f"{attempt + 1}/{max_retries} in {backoff:.0f}s")
                time.sleep(backoff)
        else:                                              # pragma: no cover — loop always breaks or raises
            raise last_exc

        # Record token usage for the eval/metrics layer. This is the ONLY place
        # every LLM call passes through, and result.llm_output carries the
        # OpenAI-format token_usage; the graph's stream events do NOT. Best-effort
        # and import-guarded so the LLM client stays usable without the eval pkg.
        _record_token_usage(result)

        for gen in result.generations:
            if hasattr(gen, "message") and isinstance(gen.message.content, str):
                gen.message.content = _strip_think(gen.message.content)
        # AFTER _strip_think, so a call written inside a reasoning span is gone
        # before we look, and we never promote a tool call the model was only
        # thinking about. Restricted to the tools bound for this very call.
        _recover_tool_calls(result, _bound_tool_names(kwargs), _noarg_tool_names(kwargs))
        # LAST: recovery works on character spans, so editing the text before it ran
        # would shift every index and delete the wrong words. Role tokens are the
        # serving layer's scaffolding ("…()]assistant"), never part of an answer, and
        # by here whatever remains is genuinely what the RM would read.
        for gen in result.generations:
            msg = getattr(gen, "message", None)
            if msg is not None and isinstance(getattr(msg, "content", None), str):
                msg.content = _strip_role_tokens(msg.content)
        return result


def _ssl_clients() -> dict:
    """httpx clients carrying a custom TLS setting, or {} to use httpx defaults.

    Only needed for an internal gateway serving a private-CA or self-signed
    certificate: the default chain check fails there and the request never
    leaves the machine. Two clients are built because both transports are live —
    the graph's own calls are sync, but the FastAPI SSE endpoint drives the model
    over the async path, and configuring only one leaves the web UI failing.
    """
    if SSL_CA_BUNDLE:
        verify = SSL_CA_BUNDLE       # trust this CA, keep verification ON
    elif not VERIFY_SSL:
        verify = False               # accept ANY cert — trusted networks only
    else:
        return {}                    # public CA chain: httpx defaults are right
    return {
        "http_client":       httpx.Client(verify=verify),
        "http_async_client": httpx.AsyncClient(verify=verify),
    }


def _log_endpoint() -> None:
    """Print which endpoint/model was selected, once per process.

    Worth the line: the two backends are chosen silently by one env var, and
    without this a wrong-backend misconfiguration is indistinguishable from a
    bad model — both just look like poor answers.
    """
    global _ENDPOINT_LOGGED
    if _ENDPOINT_LOGGED:
        return
    _ENDPOINT_LOGGED = True
    tls = ""
    if SSL_CA_BUNDLE:
        tls = f"  [TLS: custom CA {SSL_CA_BUNDLE}]"
    elif not VERIFY_SSL:
        tls = "  [TLS verify OFF]"
    print(f"LLM: {MODEL} @ {BASE_URL}{tls}")


def make_llm(temperature: float = 0.2) -> ThrottledChatOpenAI:
    """Build the shared chat client for the configured OpenAI-compatible endpoint.

    Which endpoint that is depends on ``INTERNAL_LLM_BASE_URL`` (see config).
    Raises ValueError if that endpoint's own API key is not configured — failing
    here names the missing variable, whereas letting the call proceed surfaces it
    as an opaque 401 several agents into a graph run.
    """
    error = _endpoint_error()
    if error:
        raise ValueError(error)
    _log_endpoint()
    # Cap the output reservation on endpoints that need it (internal gateway).
    # Left out entirely when 0 so the commercial route keeps the provider default —
    # see config._resolve_limits() for why the output cap is part of the CONTEXT
    # budget, not separate from it.
    extra = {}
    if config.MAX_OUTPUT_TOKENS:
        extra["max_tokens"] = config.MAX_OUTPUT_TOKENS
    return ThrottledChatOpenAI(
        model       = MODEL,
        api_key     = LLM_API_KEY,
        base_url    = BASE_URL,
        temperature = temperature,
        **extra,
        # Disable the SDK's own retry: our _generate handles retries (incl. the
        # 504 ValueError the SDK misses), so we keep a single, throttle-aware
        # backoff loop instead of nesting two retry mechanisms.
        max_retries = 0,
        **_ssl_clients(),
    )
