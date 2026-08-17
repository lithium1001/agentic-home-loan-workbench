"""Context stays bounded — across turns (compaction) and within one (segmentation).

Both halves exist because `messages` only ever APPENDS (the `add_messages` reducer
in graph, and `sess["messages"] + all_new_messages` in stream). Measured on the real
skill.md prompts and real CSV tool payloads, one full-IPA turn reaches ~14.5k input
tokens and leaves the same behind for the next turn to replay — so the SECOND such
turn overflows a 32768-token endpoint and the RM is locked out of the chat.

The invariant these tests defend is not "smaller" but BOUNDED: cost must not scale
with how many agents ran or how many turns happened. The correctness constraint that
shapes both is tool-call pairing — a ToolMessage whose originating tool_call was
dropped is an orphan, and an OpenAI-compatible endpoint 400s the entire request,
which is a worse failure than the overflow being fixed.
"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from graph import _agent_segment
from server.stream import _compact_history


def _call(name, cid):
    return {"name": name, "args": {}, "id": cid, "type": "tool_call"}


def _agent_turn(sys_text, question, tool_name, cid, answer):
    """One agent's full working set, as make_agent_node + the tool loop produce it."""
    return [
        SystemMessage(content=sys_text),
        HumanMessage(content=question),
        AIMessage(content="", tool_calls=[_call(tool_name, cid)]),
        ToolMessage(content='{"ok": true}', name=tool_name, tool_call_id=cid),
        AIMessage(content=answer),
    ]


# ── B: per-agent segmentation, within one turn ───────────────────────────
def test_segment_starts_at_the_running_agents_system_prompt():
    """Each agent re-seeds its own SystemMessage, so the LAST one is the boundary."""
    msgs = (_agent_turn("BORROWER PROMPT", "q1", "get_profile", "c1", "persona done")
            + _agent_turn("PROPERTY PROMPT", "q2", "get_property_docs", "c2", "ltv done"))

    seg = _agent_segment(msgs)

    assert seg[0].content == "PROPERTY PROMPT"
    assert len(seg) == 5
    assert not any("BORROWER PROMPT" in str(getattr(m, "content", "")) for m in seg)


def test_segment_cost_does_not_grow_with_chain_length():
    """The whole point: a 5-agent chain must not cost more than a 1-agent one."""
    one = _agent_turn("SYS", "q", "get_profile", "c0", "done")
    five = []
    for i in range(5):
        five += _agent_turn(f"SYS{i}", f"q{i}", "get_profile", f"c{i}", f"done{i}")

    assert len(_agent_segment(five)) == len(_agent_segment(one)) == 5


def test_segment_keeps_tool_calls_paired():
    """A semantic cut, not a positional one — no ToolMessage may be orphaned."""
    msgs = (_agent_turn("A", "q1", "get_profile", "c1", "a1")
            + _agent_turn("B", "q2", "get_bank_credit", "c2", "a2"))

    seg = _agent_segment(msgs)

    live = {tc["id"] for m in seg if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])}
    for m in seg:
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in live


def test_segment_passes_through_when_no_system_message():
    """The customer lane can reach the LLM before any agent seeded a prompt."""
    msgs = [HumanMessage(content="hi")]
    assert _agent_segment(msgs) == msgs


# ── A: cross-turn compaction ─────────────────────────────────────────────
def test_compaction_keeps_conversation_drops_scaffolding():
    msgs = _agent_turn("HUGE SKILL.MD PROMPT", "assess APP0001", "get_profile", "c1",
                       "Borrower earns S$8,000/month.")

    kept = _compact_history(msgs, keep_tool_turns=0)

    assert [type(m) for m in kept] == [HumanMessage, AIMessage]
    assert kept[1].content == "Borrower earns S$8,000/month."
    assert not any(isinstance(m, SystemMessage) for m in kept)


def test_compaction_keeps_every_human_message():
    """The RM's own words are load-bearing: the reprice flow reads the competitor
    rate ("DBS offered 1.55%") out of what the RM said in an EARLIER turn."""
    msgs = ([HumanMessage(content="DBS offered 1.55%")]
            + _agent_turn("SYS", "q", "get_profile", "c1", "noted")
            + [HumanMessage(content="so should we match?")])

    kept = _compact_history(msgs, keep_tool_turns=0)

    texts = [m.content for m in kept if isinstance(m, HumanMessage)]
    assert "DBS offered 1.55%" in texts
    assert "so should we match?" in texts


def test_compaction_keeps_the_most_recent_turns_tool_results():
    """Conservative A: an immediate follow-up ("explain that TDSR") still has the
    raw figures rather than forcing a re-fetch."""
    old = _agent_turn("SYS1", "turn one", "get_profile", "c1", "first answer")
    new = _agent_turn("SYS2", "turn two", "calculate_loan", "c2", "second answer")

    kept = _compact_history(old + new, keep_tool_turns=1)

    tool_ids = [m.tool_call_id for m in kept if isinstance(m, ToolMessage)]
    assert tool_ids == ["c2"]          # newest kept, older dropped


def test_compaction_never_orphans_a_tool_message():
    """The 400-causing shape: a retained ToolMessage whose tool_call was dropped."""
    msgs = []
    for i in range(4):
        msgs += _agent_turn(f"SYS{i}", f"q{i}", "get_profile", f"c{i}", f"a{i}")

    for keep in (0, 1, 2, 3):
        kept = _compact_history(msgs, keep_tool_turns=keep)
        live = {tc["id"] for m in kept if isinstance(m, AIMessage)
                for tc in (m.tool_calls or [])}
        orphans = [m.tool_call_id for m in kept
                   if isinstance(m, ToolMessage) and m.tool_call_id not in live]
        assert orphans == [], f"orphaned ToolMessage at keep_tool_turns={keep}"


def test_compaction_is_bounded_across_many_turns():
    """Ten turns must not cost ten times one turn."""
    hist = []
    sizes = []
    for i in range(10):
        hist = _compact_history(
            hist + _agent_turn(f"SYS{i}", f"q{i}", "get_profile", f"c{i}", f"a{i}"))
        sizes.append(sum(len(str(getattr(m, "content", ""))) for m in hist))

    # Growth is linear in KEPT conversation only (2 short messages/turn), not in the
    # scaffolding, so ten turns stay far under ten times the raw turn size.
    raw_one_turn = sum(len(str(getattr(m, "content", "")))
                       for m in _agent_turn("SYS", "q", "get_profile", "c", "a"))
    assert sizes[-1] < raw_one_turn * 10


def test_compaction_is_idempotent():
    """It runs on every finished turn, so re-compacting must be a no-op."""
    msgs = (_agent_turn("SYS1", "q1", "get_profile", "c1", "a1")
            + _agent_turn("SYS2", "q2", "calculate_loan", "c2", "a2"))

    once = _compact_history(msgs)
    twice = _compact_history(once)

    assert [(type(m), getattr(m, "content", "")) for m in once] == \
           [(type(m), getattr(m, "content", "")) for m in twice]


def test_compaction_of_empty_history():
    assert _compact_history([]) == []


# ── The off switch ───────────────────────────────────────────────────────
# Both trims are on by default. The switch exists so "did trimming make the model
# dumber?" can be answered by running the same queries both ways — without it the
# only way back is editing code, which is not a control arm.
def test_switch_off_restores_send_everything(monkeypatch):
    from utils import config

    msgs = (_agent_turn("SYS1", "q1", "get_profile", "c1", "a1")
            + _agent_turn("SYS2", "q2", "calculate_loan", "c2", "a2"))

    monkeypatch.setattr(config, "TRIM_CONTEXT", False)
    assert _agent_segment(msgs) == msgs           # within-turn trim disabled
    assert _compact_history(msgs) == msgs         # cross-turn trim disabled

    monkeypatch.setattr(config, "TRIM_CONTEXT", True)
    assert len(_agent_segment(msgs)) < len(msgs)  # and back on again
    assert len(_compact_history(msgs)) < len(msgs)


def test_switch_is_read_live_not_frozen(monkeypatch):
    """Both modules must read config.TRIM_CONTEXT through the module, never via a
    from-import: an A/B run flips it mid-process and a frozen copy would silently
    keep the first arm's behaviour for the whole run."""
    import graph as graph_mod
    from server import stream as stream_mod

    assert "TRIM_CONTEXT" not in vars(graph_mod)
    assert "TRIM_CONTEXT" not in vars(stream_mod)
