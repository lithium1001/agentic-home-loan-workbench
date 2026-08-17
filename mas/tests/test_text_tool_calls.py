"""Guards recovery of tool calls that a Llama-family model emitted as plain TEXT.

Regression: after switching the endpoint to Llama-4, tool calls started arriving in
the message CONTENT instead of the structured `tool_calls` field. The graph decides
a turn is done with `is_final = not resp.tool_calls`, so such a turn meant: the tool
never ran, the turn was treated as a final answer, and stream.py rendered the raw
JSON into the RM's chat bubble ("tool call 发到前台"). Same class as the 2026-07-22
orchestrator routing-JSON leak, different source.

`_recover_tool_calls` promotes those text-emitted calls to real ones and blanks the
text they came from. It must NOT invent a call when the model emitted none, and must
NOT honour a tool name outside the bound set. No LLM / network.
"""

import json

from utils.llm import (
    _as_tool_call,
    _bound_tool_names,
    _extract_text_tool_calls,
    _recover_tool_calls,
)

ALLOWED = {"calculate_loan", "get_profile", "search_policy"}


class _Msg:
    """Minimal AIMessage stand-in: the recovery only touches these attributes."""

    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.additional_kwargs = {}


class _Gen:
    def __init__(self, message):
        self.message = message


class _Result:
    def __init__(self, *messages):
        self.generations = [_Gen(m) for m in messages]


# ── The three emission syntaxes ───────────────────────────────────────────
def test_python_tag_is_recovered():
    text = '<|python_tag|>{"name": "calculate_loan", "parameters": {"loan_amount": 500000}}'
    calls, remaining = _extract_text_tool_calls(text, ALLOWED)
    assert len(calls) == 1
    assert calls[0]["name"] == "calculate_loan"
    assert calls[0]["args"] == {"loan_amount": 500000}
    assert remaining == ""


def test_bare_json_object_is_recovered():
    text = '{"name": "get_profile", "arguments": {"applicant_id": "APP0001"}}'
    calls, _ = _extract_text_tool_calls(text, ALLOWED)
    assert len(calls) == 1
    assert calls[0]["args"] == {"applicant_id": "APP0001"}


def test_json_fenced_call_is_recovered():
    text = '```json\n{"name": "search_policy", "arguments": {"query": "MSR cap"}}\n```'
    calls, remaining = _extract_text_tool_calls(text, ALLOWED)
    assert len(calls) == 1
    assert calls[0]["name"] == "search_policy"
    assert "```" not in remaining


def test_function_tag_syntax_is_recovered():
    text = '<function=calculate_loan{"loan_amount": 750000}></function>'
    calls, remaining = _extract_text_tool_calls(text, ALLOWED)
    assert len(calls) == 1
    assert calls[0]["args"] == {"loan_amount": 750000}
    assert remaining == ""          # no tag debris left to leak into the bubble


# ── Syntax 4: Python-call batch (the 2026-08-03 Reprice screenshot) ──────
# Observed live on the Reprice & Retention agent: the audit log read
# "FINAL ANSWER → Tool", so none of the three tools ran and the whole plan text
# landed in the RM's chat bubble.
_LEAKED = """To provide a retention script, we need to follow the steps outlined:

1. Identify the applicant and pull the case.
2. Determine the two packages to compare.

Let's start by pulling the necessary details.

First, we need to get the loan application details for APP0006.

[get_loan_application(applicant_id='APP0006'),
get_profile(applicant_id='APP0006'),
get_bank_credit(applicant_id='APP0006')]"""

_REPRICE_TOOLS = {"get_loan_application", "get_profile", "get_bank_credit",
                  "compare_packages", "calculate_loan"}


def test_python_call_batch_from_real_leak():
    calls, remaining = _extract_text_tool_calls(_LEAKED, _REPRICE_TOOLS)
    assert [c["name"] for c in calls] == [
        "get_loan_application", "get_profile", "get_bank_credit"]
    assert all(c["args"] == {"applicant_id": "APP0006"} for c in calls)
    # No call syntax and no bracket debris may survive into the bubble.
    for debris in ("get_loan_application", "[", "]", "applicant_id="):
        assert debris not in remaining


def test_single_python_call_recovered():
    calls, _ = _extract_text_tool_calls(
        "get_profile(applicant_id='APP0001')", _REPRICE_TOOLS)
    assert calls[0]["args"] == {"applicant_id": "APP0001"}


def test_python_call_literals_and_nesting():
    text = ("compare_packages(package_ids=['P1','P2'], competitor_rate=1.55, "
            "include_fees=True, notes=None)")
    calls, _ = _extract_text_tool_calls(text, _REPRICE_TOOLS)
    assert calls[0]["args"] == {"package_ids": ["P1", "P2"], "competitor_rate": 1.55,
                                "include_fees": True, "notes": None}


def test_python_call_with_paren_inside_string():
    """A ')' inside a quoted value must not close the call early."""
    calls, _ = _extract_text_tool_calls(
        "get_profile(applicant_id='APP0001', name='Lim (Alice)')", _REPRICE_TOOLS)
    assert calls[0]["args"]["name"] == "Lim (Alice)"


def test_python_call_positional_args_rejected():
    """Without the schema we cannot know which parameter a positional binds to;
    guessing an order is how the right number lands in the wrong field."""
    calls, remaining = _extract_text_tool_calls(
        "We ran calculate_loan(500000) already.", _REPRICE_TOOLS)
    assert calls == []
    assert remaining == "We ran calculate_loan(500000) already."


def test_python_call_no_args_is_prose_not_a_call():
    """Every tool here takes arguments, so empty parens are narration."""
    text = "Per calculate_loan(), the instalment is S$4,490."
    calls, remaining = _extract_text_tool_calls(text, _REPRICE_TOOLS)
    assert calls == []
    assert remaining == text            # words must not be deleted from an answer


def test_tool_name_merely_mentioned_is_not_a_call():
    for text in ("I will now use calculate_loan to price this.",
                 "Call compare_packages with our package as the baseline.",
                 "The TDSR is 42% (under the 55% cap), so we proceed."):
        calls, remaining = _extract_text_tool_calls(text, _REPRICE_TOOLS)
        assert calls == [], text
        assert remaining == text, text


def test_python_call_unbound_name_ignored():
    """An unbound name is not a call — and since there is no unambiguous marker,
    the prose is left intact rather than deleted."""
    text = "draft_letter(applicant_id='APP0006')"
    calls, remaining = _extract_text_tool_calls(text, {"get_profile"})
    assert calls == []
    assert remaining == text


def test_args_as_json_string_is_parsed():
    text = '<|python_tag|>{"name": "get_profile", "arguments": "{\\"applicant_id\\": \\"APP0002\\"}"}'
    calls, _ = _extract_text_tool_calls(text, ALLOWED)
    assert calls[0]["args"] == {"applicant_id": "APP0002"}


def test_trailing_prose_after_json_is_dropped():
    text = ('<|python_tag|>{"name": "calculate_loan", "parameters": {"loan_amount": 100}}'
            "\nNow I will compute the TDSR.")
    calls, remaining = _extract_text_tool_calls(text, ALLOWED)
    assert len(calls) == 1
    # The narration is not an answer; the caller blanks it once a call exists.
    assert "TDSR" in remaining


def test_nested_args_survive():
    text = ('{"name": "calculate_loan", "arguments": {"applicant": {"id": "APP0001", '
            '"income": {"base": 8000}}, "loan_amount": 500000}}')
    calls, _ = _extract_text_tool_calls(text, ALLOWED)
    assert calls[0]["args"]["applicant"]["income"]["base"] == 8000


# ── Boundaries: what must NOT be recovered ───────────────────────────────
def test_unbound_tool_name_is_rejected():
    """A name outside the bound set must never execute — bound_llm_excluding()
    withholds compare_packages on purpose, and the toC whitelist is a hard edge."""
    text = '<|python_tag|>{"name": "compare_packages", "parameters": {}}'
    calls, remaining = _extract_text_tool_calls(text, ALLOWED)
    assert calls == []
    # …but the marker is still stripped, so it cannot leak into the chat bubble.
    assert "python_tag" not in remaining
    assert "compare_packages" not in remaining


def test_ordinary_prose_is_untouched():
    text = "The TDSR is 42% which exceeds the 55% cap. I recommend a longer tenor."
    calls, remaining = _extract_text_tool_calls(text, ALLOWED)
    assert calls == []
    assert remaining == text


def test_prose_containing_braces_is_untouched():
    text = 'Use the {rate} placeholder in the letter template.'
    calls, remaining = _extract_text_tool_calls(text, ALLOWED)
    assert calls == []
    assert remaining == text


def test_json_that_is_not_a_tool_call_is_untouched():
    text = '{"tdsr": 0.42, "msr": 0.28}'
    calls, remaining = _extract_text_tool_calls(text, ALLOWED)
    assert calls == []
    assert remaining == text


def test_no_call_is_never_invented():
    """The 'silent no-op' failure mode is explicitly NOT this layer's job:
    guessing which tool was meant would be fabrication."""
    calls, _ = _extract_text_tool_calls("I have already computed the instalment.", ALLOWED)
    assert calls == []


def test_empty_and_non_string_content():
    assert _extract_text_tool_calls("", ALLOWED) == ([], "")
    assert _extract_text_tool_calls(None, ALLOWED) == ([], None)


# ── The in-place promotion on a result ───────────────────────────────────
def test_recover_promotes_the_call():
    msg = _Msg('<|python_tag|>{"name": "calculate_loan", "parameters": {"loan_amount": 500000}}')
    _recover_tool_calls(_Result(msg), ALLOWED)
    assert len(msg.tool_calls) == 1
    assert msg.tool_calls[0]["type"] == "tool_call"
    assert msg.tool_calls[0]["id"]                      # ToolNode needs an id
    # is_final in graph.py reads `not resp.tool_calls` — must now be False.
    assert bool(msg.tool_calls) is True
    # The call syntax itself is gone from the text.
    assert "python_tag" not in msg.content
    assert "calculate_loan" not in msg.content


def test_narration_is_kept_for_the_thinking_block():
    """The plan text is useful (thinking-block summary + audit trail) and cannot
    reach the chat bubble, so recovery keeps it instead of blanking it."""
    msg = _Msg("First, we need the loan application details for APP0006.\n\n"
               "[get_profile(applicant_id='APP0006')]")
    _recover_tool_calls(_Result(msg), ALLOWED | {"get_profile"})
    assert msg.tool_calls
    assert "First, we need the loan application details" in msg.content
    assert "get_profile(" not in msg.content            # call syntax still stripped


def test_kept_narration_cannot_become_the_answer():
    """Both answer paths are gated on there being no tool_calls. This is the exact
    mirror of the original bug, where empty tool_calls left both gates open."""
    msg = _Msg("Let's start by pulling the case.\n\n"
               "[get_profile(applicant_id='APP0006')]")
    _recover_tool_calls(_Result(msg), ALLOWED | {"get_profile"})

    # graph.py:568 — drives the a2a payload's is_final, which stream.py:440 requires.
    is_final = not bool(msg.tool_calls)
    assert is_final is False

    # stream.py:486 — the AIMessage harvest guard.
    harvested = bool(msg.content and not msg.tool_calls)
    assert harvested is False


def test_recover_mirrors_into_additional_kwargs():
    """The turn is replayed to the provider on the next hop, so it must look like a
    normal assistant tool-call turn in OpenAI format too."""
    msg = _Msg('{"name": "get_profile", "arguments": {"applicant_id": "APP0001"}}')
    _recover_tool_calls(_Result(msg), ALLOWED)
    tc = msg.additional_kwargs["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "get_profile"
    assert json.loads(tc["function"]["arguments"]) == {"applicant_id": "APP0001"}
    assert tc["id"] == msg.tool_calls[0]["id"]         # ids must match


def test_recover_leaves_correct_structured_calls_alone():
    """When the model gets it right, this layer is a no-op."""
    good = [{"name": "calculate_loan", "args": {"loan_amount": 1},
             "id": "call_abc", "type": "tool_call"}]
    msg = _Msg("Calling the calculator.", tool_calls=good)
    _recover_tool_calls(_Result(msg), ALLOWED)
    assert msg.tool_calls == good
    assert msg.content == "Calling the calculator."     # a real answer is preserved


def test_recover_skipped_when_no_tools_bound():
    """With no tools bound, JSON in the content is data, not a call."""
    msg = _Msg('{"name": "calculate_loan", "arguments": {}}')
    _recover_tool_calls(_Result(msg), None)
    assert msg.tool_calls == []
    assert msg.content == '{"name": "calculate_loan", "arguments": {}}'


def test_recover_never_raises_on_junk():
    class _Broken:
        generations = "not a list of gens"
    _recover_tool_calls(_Broken(), ALLOWED)             # must not raise


# ── The bound-tool-name plumbing ─────────────────────────────────────────
def test_bound_tool_names_reads_openai_schema():
    kwargs = {"tools": [
        {"type": "function", "function": {"name": "calculate_loan", "parameters": {}}},
        {"type": "function", "function": {"name": "get_profile", "parameters": {}}},
    ]}
    assert _bound_tool_names(kwargs) == {"calculate_loan", "get_profile"}


def test_bound_tool_names_none_when_unbound():
    assert _bound_tool_names({}) is None
    assert _bound_tool_names({"tools": []}) is None


def test_as_tool_call_accepts_any_name_when_allowed_is_none():
    """allowed=None means 'do not filter' — used only where the caller has already
    decided recovery applies; the _generate path never passes None with tools bound."""
    tc = _as_tool_call({"name": "anything", "arguments": {}}, None)
    assert tc["name"] == "anything"


# ── ReAct "action"/"action_input" shape ────────────────────────────────────
# Observed live 2026-08-04 on the COMPARE track: the model emitted
#   {"action": "interest_savings_tool", "action_input": "{ \"outstanding_loan\": ... }"}
# as plain text. Neither key was recognised, so no call was recovered and the blob
# was rendered verbatim into the RM's chat bubble (screenshot evidence) while the
# tool never ran.

_REACT_LEAK = json.dumps({
    "action": "calculate_loan",
    "action_input": '{ "loan_amount": 1200000, "interest_rate_pct": 2.0 }',
})


def test_react_action_shape_is_recovered():
    calls, remaining = _extract_text_tool_calls(_REACT_LEAK, ALLOWED)
    assert len(calls) == 1
    assert calls[0]["name"] == "calculate_loan"
    assert calls[0]["args"]["loan_amount"] == 1200000
    assert remaining.strip() == "", "the JSON must not survive into the chat bubble"


def test_react_action_input_may_be_an_object():
    """Some checkpoints emit action_input as a nested object rather than a string."""
    raw = json.dumps({"action": "get_profile",
                      "action_input": {"applicant_id": "APP0001"}})
    calls, _ = _extract_text_tool_calls(raw, ALLOWED)
    assert len(calls) == 1
    assert calls[0]["args"] == {"applicant_id": "APP0001"}


def test_react_action_still_honours_the_allowlist():
    """The whitelist is a security boundary (toC role), not a formatting detail —
    a new syntax must not become a way around it."""
    raw = json.dumps({"action": "draft_letter", "action_input": "{}"})
    calls, _ = _extract_text_tool_calls(raw, ALLOWED)
    assert calls == []


def test_react_recovery_promotes_to_a_real_tool_call():
    """End to end through _recover_tool_calls: the turn must stop looking final."""
    msg = _Msg(_REACT_LEAK)
    _recover_tool_calls(_Result(msg), ALLOWED)
    assert msg.tool_calls, "graph would treat this as a final answer and leak it"
    assert msg.tool_calls[0]["name"] == "calculate_loan"
    assert not (msg.content or "").strip()


# ── Unquoted string values (the 2026-08-04 Reprice screenshot, take 2) ─────
# The Python-call recovery above already worked, but only for QUOTED values. Live
# on the Reprice track the model emitted `applicant_id=APP0001` and
# `property_type=Condo` with no quotes. ast.literal_eval reads a bare word as a
# variable NAME, not a literal, so it raised; since one unreadable value voids the
# whole call, both calls were skipped and the entire batch was rendered into the
# RM's chat bubble verbatim. Values are now read in layers: literal first, then
# bare identifier as its own string.

_BARE_TOOLS = {"get_loan_application", "compare_packages_tool", "get_profile",
               "calculate_loan"}

_BARE_LEAK = (
    "[get_loan_application(applicant_id=APP0001), "
    "compare_packages_tool(packages=[{'label': 'Current Loan', 'interest_rate_pct': 2}, "
    "{'label': '1.55% Now', 'interest_rate_pct': 1.55}], "
    "borrowers=[{'age': 35, 'monthly_income': 10000, 'nationality': 'Singaporean'}], "
    "property_type=Condo, n_outstanding_loans=1, n_props_owned=1, "
    "monthly_car_loan=0, monthly_other=0, target_property_price=1200000)]"
)


def test_unquoted_values_from_real_leak():
    """The verbatim screenshot text: both calls recovered, nothing left to leak."""
    calls, remaining = _extract_text_tool_calls(_BARE_LEAK, _BARE_TOOLS)
    assert [c["name"] for c in calls] == [
        "get_loan_application", "compare_packages_tool"]
    assert calls[0]["args"] == {"applicant_id": "APP0001"}
    assert calls[1]["args"]["property_type"] == "Condo"      # bare -> string
    # The quoted/nested values in the same call must be unharmed.
    assert calls[1]["args"]["packages"][1]["interest_rate_pct"] == 1.55
    assert calls[1]["args"]["borrowers"][0]["nationality"] == "Singaporean"
    assert calls[1]["args"]["target_property_price"] == 1200000
    assert remaining.strip() == ""


def test_bare_word_does_not_swallow_true_false_none():
    """True/False/None are ast.Constant, not ast.Name, so the fallback never sees
    them. If it did, a boolean flag would silently become the string 'True'."""
    calls, _ = _extract_text_tool_calls(
        "get_profile(applicant_id=APP0001, include_fees=True, waived=False, notes=None)",
        _BARE_TOOLS)
    args = calls[0]["args"]
    assert args["include_fees"] is True
    assert args["waived"] is False
    assert args["notes"] is None
    assert args["applicant_id"] == "APP0001"


def test_dotted_name_value_is_read_as_text():
    calls, _ = _extract_text_tool_calls(
        "get_profile(property_type=PropertyType.CONDO)", _BARE_TOOLS)
    assert calls[0]["args"]["property_type"] == "PropertyType.CONDO"


def test_partial_batch_is_all_or_nothing():
    """One sibling unreadable -> recover NOTHING. Promoting only the readable call
    would let the batch's span widening delete the other call's text without ever
    running it, and the agent would proceed believing it had data it never fetched.
    """
    text = ("[get_profile(applicant_id=APP0001), "
            "calculate_loan(loan_amount=income*0.8)]")
    calls, remaining = _extract_text_tool_calls(text, _BARE_TOOLS)
    assert calls == []
    assert "calculate_loan" in remaining, "the unread call must not vanish silently"


# ── Constant arithmetic (the 2026-08-04 compliance screenshot) ─────────────
# Live on the compliance track the model wrote
#   calculate_loan_tool(..., cash_cpf_available=511000 + 86780.42, ...)
# Both operands are stated in the text, but the value was an ast.BinOp rather than a
# literal, so the whole call was refused and the tool never ran. Folding constants is
# reading what the model wrote (same kind of normalisation as 1_200_000 -> 1200000),
# NOT inferring — the line is drawn at whether an operand appears in the text.

_COMPLIANCE_LEAK = (
    '[calculate_loan_tool(property_type="Private Residential", n_outstanding_loans=0, '
    "n_props_owned=1, interest_rate_pct=1.31, monthly_car_loan=0, monthly_other=0, "
    "cash_cpf_available=511000 + 86780.42, target_property_price=1792000)]"
)


def test_constant_arithmetic_from_real_leak():
    calls, remaining = _extract_text_tool_calls(
        _COMPLIANCE_LEAK, {"calculate_loan_tool"})
    assert len(calls) == 1
    args = calls[0]["args"]
    assert args["cash_cpf_available"] == 597780.42      # 511000 + 86780.42, folded
    assert args["property_type"] == "Private Residential"
    assert args["target_property_price"] == 1792000
    assert remaining.strip() == ""


def test_constant_arithmetic_shapes():
    for expr, want in (("8000*12", 96000), ("(100+50)/2", 75.0),
                       ("-500", -500), ("1200000*0.75", 900000.0),
                       ("1_200_000", 1200000)):
        calls, _ = _extract_text_tool_calls(
            f"get_profile(v={expr})", _BARE_TOOLS)
        assert calls and calls[0]["args"]["v"] == want, expr


def test_expression_referencing_a_name_is_refused():
    """`income` is not in the text. Folding it would require inventing the operand —
    the fabrication this layer exists to prevent."""
    for expr in ("income*0.55", "salary + bonus", "511000 + unknown"):
        text = f"calculate_loan(loan_amount={expr})"
        calls, remaining = _extract_text_tool_calls(text, _BARE_TOOLS)
        assert calls == [], expr
        assert remaining == text, expr


def test_arithmetic_cannot_become_code_execution():
    """The expression text is model-controlled, so the operand walk is a whitelist.
    Calls, attributes and f-strings must never be evaluated."""
    for expr in ('len("ab")', '__import__("os").getcwd()', 'f"{x}"'):
        calls, _ = _extract_text_tool_calls(
            f"get_profile(v={expr})", _BARE_TOOLS)
        assert calls == [], expr


def test_arithmetic_cannot_exhaust_the_process():
    """`10**10**10` hangs for minutes and exhausts memory DURING eval, before any
    check on the result could run, so `**` and shifts are barred at the operator.
    A model that emits one must not be able to take the worker down."""
    import time

    for expr in ("10**10**10", "1<<999999999", '"a"*100000'):
        start = time.monotonic()
        calls, _ = _extract_text_tool_calls(f"get_profile(v={expr})", _BARE_TOOLS)
        assert calls == [], expr
        assert time.monotonic() - start < 1.0, f"{expr} was evaluated"


def test_arithmetic_error_is_refused_not_raised():
    calls, _ = _extract_text_tool_calls("get_profile(v=1/0)", _BARE_TOOLS)
    assert calls == []


def test_bare_value_call_promotes_end_to_end():
    msg = _Msg(_BARE_LEAK)
    _recover_tool_calls(_Result(msg), _BARE_TOOLS)
    assert len(msg.tool_calls) == 2
    assert not (msg.content or "").strip()
    assert bool(msg.tool_calls) is True          # is_final -> False, no leak


# ── Recovery logging ──────────────────────────────────────────────────────
# One flat line for every outcome is what made the unquoted-value leak look like
# the patch had stopped working: a refusal logged nothing at all, so the only
# symptom was text in the RM's bubble. The three outcomes must now read apart, and
# ordinary prose must stay silent or the warning means nothing.

def _log_of(text, allowed=None, capsys=None):
    msg = _Msg(text)
    _recover_tool_calls(_Result(msg), allowed or _BARE_TOOLS)
    return capsys.readouterr().out


def test_log_names_the_syntax_that_was_recovered(capsys):
    out = _log_of(_BARE_LEAK, capsys=capsys)
    assert "recovered 2" in out
    assert "[python-call]" in out
    assert "get_loan_application" in out         # which tools, for the audit trail


def test_log_distinguishes_each_syntax(capsys):
    cases = [
        ('<|python_tag|>{"name": "get_profile", "parameters": {"applicant_id": "A1"}}',
         "[python_tag]"),
        (json.dumps({"action": "get_profile", "action_input": {"applicant_id": "A1"}}),
         "[ReAct-action]"),
        ('{"name": "get_profile", "arguments": {"applicant_id": "A1"}}',
         "[bare-JSON]"),
        ("get_profile(applicant_id=APP0001)", "[python-call]"),
    ]
    for text, tag in cases:
        assert tag in _log_of(text, capsys=capsys), text


def test_refusal_is_logged_with_the_reason(capsys):
    """The 2026-08-04 blind spot: refusing to recover used to be silent, so a leak
    was indistinguishable from the patch not running at all."""
    out = _log_of("calculate_loan(loan_amount=income*0.75)", capsys=capsys)
    assert "NOT recovered" in out
    assert "loan_amount" in out                  # which argument defeated it
    assert "BinOp" in out                        # and what shape it was


def test_positional_refusal_names_its_own_reason(capsys):
    out = _log_of("calculate_loan(500000)", capsys=capsys)
    assert "positional" in out


def test_stripped_without_recovery_is_logged(capsys):
    """Bubble is clean but no tool ran — invisible without its own line."""
    out = _log_of('<|python_tag|>{"name": "draft_letter", "parameters": {}}',
                  capsys=capsys)
    assert "without recovering" in out
    assert "no tool ran" in out


def test_ordinary_answer_logs_nothing(capsys):
    """A warning on every normal turn would train the reader to ignore it."""
    assert _log_of("The TDSR is 42%, under the 55% cap.", capsys=capsys) == ""
    assert _log_of("I will use calculate_loan to price this.", capsys=capsys) == ""


# ── No-arg calls (2026-08-04) ────────────────────────────────────────────
# Observed in the RM's chat bubble on the Compliance agent:
#   "[loan_calc_result = calculate_loan_tool()]assistant\n\ncalculate_loan_tool()"
# The calculator never ran and that raw text was rendered as the answer. Two causes:
# the recovery rejected EVERY no-arg call ("every tool here takes arguments" — untrue
# for calculate_loan_tool, whose inputs come from graph state via InjectedState), and
# the trailing chat-template role token was never stripped.
_ALLOWED = {"calculate_loan_tool", "get_loan_application"}
_NOARG = {"calculate_loan_tool"}


def test_recovers_the_observed_noarg_emission():
    calls, remaining = _extract_text_tool_calls(
        "[loan_calc_result = calculate_loan_tool()]assistant\n\ncalculate_loan_tool()",
        _ALLOWED, _NOARG)
    assert [c["name"] for c in calls] == ["calculate_loan_tool"] * len(calls)
    assert calls, "the call must be recovered, not left as chat text"
    assert "calculate_loan_tool()" not in remaining


def test_recovers_a_bare_noarg_call():
    calls, remaining = _extract_text_tool_calls("calculate_loan_tool()", _ALLOWED, _NOARG)
    assert len(calls) == 1
    assert remaining == ""


def test_recovers_a_noarg_call_after_a_sentence():
    calls, _ = _extract_text_tool_calls(
        "Now I will compute.\ncalculate_loan_tool()", _ALLOWED, _NOARG)
    assert len(calls) == 1


def test_a_tool_that_requires_args_is_still_rejected_when_empty():
    """Empty parens on get_loan_application is a fragment, not a call — recovering it
    would fire a lookup with no applicant."""
    calls, remaining = _extract_text_tool_calls("get_loan_application()", _ALLOWED, _NOARG)
    assert calls == []
    assert remaining == "get_loan_application()"


def test_prose_mentioning_a_noarg_tool_is_left_alone():
    """The reason the blanket rejection existed. Recovering these would fire a
    spurious calculation AND delete words from a real answer."""
    for prose in (
        "per calculate_loan_tool(), the instalment is 4490.",
        "The calculate_loan_tool() result shows 4490 per month.",
        "I used calculate_loan_tool() earlier for this.",
        "Lets start by calling the calculate_loan_tool to get details.",
    ):
        calls, remaining = _extract_text_tool_calls(prose, _ALLOWED, _NOARG)
        assert calls == [], f"prose recovered as a call: {prose}"
        assert remaining == prose


def test_noarg_recovery_is_off_without_the_schema():
    """Default (no `noarg` set) keeps the old conservative behaviour: without the
    schema we cannot tell a call from a mention."""
    calls, _ = _extract_text_tool_calls("calculate_loan_tool()", _ALLOWED)
    assert calls == []


def test_noarg_names_read_from_the_bound_schema():
    """Derived from bind_tools()' own schema so it cannot rot as tools change."""
    from utils.llm import _noarg_tool_names

    kwargs = {"tools": [
        {"function": {"name": "calculate_loan_tool", "parameters": {"required": []}}},
        {"function": {"name": "get_profile", "parameters": {"required": ["applicant_id"]}}},
    ]}
    assert _noarg_tool_names(kwargs) == {"calculate_loan_tool"}


# ── Chat-template role tokens ────────────────────────────────────────────
def test_role_tokens_are_stripped_from_the_answer():
    from utils.llm import _strip_role_tokens

    assert _strip_role_tokens("Here is the answer.<|eot_id|>") == "Here is the answer."
    assert _strip_role_tokens("assistant\n\nThe TDSR is 42%.") == "The TDSR is 42%."
    assert _strip_role_tokens("[x = ]assistant\n\nmore").startswith("[x = ]")


def test_the_english_word_assistant_survives():
    """A bank letter says "your relationship assistant" — stripping that would edit
    the RM's actual answer."""
    from utils.llm import _strip_role_tokens

    for keep in ("The assistant will contact you shortly.",
                 "Please ask your relationship assistant about this."):
        assert _strip_role_tokens(keep) == keep
