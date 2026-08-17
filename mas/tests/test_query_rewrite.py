"""Tests for the agentic-RAG layer: LLM query rewriting (utils/query_rewrite.py)
and the rank-fusion retrieval it feeds (utils/tools._agentic_search_policy).

No network, no API key, no cost: every test injects a **stub LLM**. That is the
point of ``rewrite_query(query, llm=...)`` taking an injectable client — the LLM
is the only impure part, so stubbing it makes the whole agentic path unit-testable
offline, and lets us assert on the failure modes (model returns prose / junk /
raises) that we can't reliably provoke against a real endpoint.

Run from mas/:
    py -m pytest -q tests/test_query_rewrite.py
"""
from pathlib import Path

import pytest

from utils.query_rewrite import (
    _clean,
    _looks_usable,
    is_set_question,
    rerank_order,
    rewrite_query,
)
from utils.tools import _FUSION_POOL, _agentic_search_policy, _rrf_fuse

_HAS_PDFS = bool(list((Path(__file__).resolve().parents[1] / "t&c").glob("*.pdf"))) \
    if (Path(__file__).resolve().parents[1] / "t&c").is_dir() else False
needs_pdfs = pytest.mark.skipif(not _HAS_PDFS, reason="no policy PDFs under t&c/")


class _StubLLM:
    """Minimal stand-in for the chat client: returns a canned content string, or
    raises, so the degrade paths are testable."""

    def __init__(self, content: str = "", raises: Exception | None = None):
        self._content = content
        self._raises = raises
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self._raises:
            raise self._raises
        return type("_Msg", (), {"content": self._content})()


# ── rewrite_query: the happy path ──────────────────────────────────────────
def test_rewrite_returns_the_models_query():
    llm = _StubLLM("Building Under Construction BUC property eligibility")
    out = rewrite_query("my place isn't built yet, does it still count", llm=llm)
    assert out == "Building Under Construction BUC property eligibility"


def test_rewrite_sends_the_original_question_to_the_llm():
    llm = _StubLLM("lock-in period")
    rewrite_query("how long am I stuck", llm=llm)
    # System prompt + user message; the question must reach the model.
    sent = " ".join(m.content for m in llm.calls[0])
    assert "how long am I stuck" in sent


# ── rewrite_query: every failure degrades to the ORIGINAL query ─────────────
# This is the contract the whole design rests on — a broken rewrite must leave
# retrieval exactly as good as it is without one, never worse, never raising.
@pytest.mark.parametrize(
    "stub, why",
    [
        (_StubLLM(raises=RuntimeError("no api key")), "llm raises"),
        (_StubLLM(""), "empty response"),
        (_StubLLM("   "), "whitespace-only response"),
        (_StubLLM("x" * 400), "over-long response (model wrote an essay)"),
        (_StubLLM("Sure! The answer is yes. You qualify. Here is why. It says so."), "model answered instead of rewriting"),
    ],
)
def test_rewrite_degrades_to_original_query(stub, why):
    original = "can I apply under my company name"
    assert rewrite_query(original, llm=stub) == original, why


def test_rewrite_passes_through_empty_query():
    assert rewrite_query("", llm=_StubLLM("anything")) == ""


# ── _clean: strip the wrappers a chat model adds around a one-liner ─────────
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Search query: lock-in period", "lock-in period"),
        ('"prepayment penalty"', "prepayment penalty"),
        ("- guarantor joint borrower", "guarantor joint borrower"),
        ("fire insurance\n\n(This finds the clause about...)", "fire insurance"),
    ],
)
def test_clean_strips_model_wrappers(raw, expected):
    assert _clean(raw) == expected


def test_looks_usable_rejects_prose_and_junk():
    assert _looks_usable("body corporate company applicant")
    assert not _looks_usable("")
    assert not _looks_usable("x" * 500)
    # An answer, not a query: multiple sentences.
    assert not _looks_usable("Yes, you qualify. The clause says so. See 1.2.")


# ── rerank_order: listwise rerank, fail-open ────────────────────────────────
_DOCS = ["clause about gifts", "clause about eligibility", "clause about law"]


def test_rerank_parses_a_ranking():
    assert rerank_order("who can apply", _DOCS, llm=_StubLLM("2, 1, 3")) == [1, 0, 2]


def test_rerank_appends_omitted_indices_in_original_order():
    """The model may promote, but a clause can never vanish because it forgot one."""
    assert rerank_order("who can apply", _DOCS, llm=_StubLLM("3")) == [2, 0, 1]


def test_rerank_ignores_out_of_range_and_duplicate_numbers():
    assert rerank_order("q", _DOCS, llm=_StubLLM("9, 2, 2, 0, 1")) == [1, 0, 2]


@pytest.mark.parametrize(
    "stub, why",
    [
        (_StubLLM(raises=RuntimeError("boom")), "llm raises"),
        (_StubLLM(""), "empty reply"),
        (_StubLLM("The most relevant clause is clearly the one about eligibility, because " * 5), "essay reply"),
        (_StubLLM("no numbers here"), "no parseable numbers"),
    ],
)
def test_rerank_fails_open_to_none(stub, why):
    assert rerank_order("who can apply", _DOCS, llm=stub) is None, why


def test_rerank_declines_degenerate_inputs():
    assert rerank_order("", _DOCS, llm=_StubLLM("1")) is None
    assert rerank_order("q", ["single doc"], llm=_StubLLM("1")) is None


def test_rerank_sends_question_and_snippets():
    llm = _StubLLM("1, 2, 3")
    rerank_order("who can apply", _DOCS, llm=llm)
    sent = " ".join(m.content for m in llm.calls[0])
    assert "who can apply" in sent and "clause about gifts" in sent


# ── _rrf_fuse: rank-based fusion of two rankings ───────────────────────────
def _hits(*texts):
    return [{"text": t, "title": t, "source": "s.pdf", "score": 1.0} for t in texts]


def test_rrf_promotes_a_chunk_found_by_both_queries():
    """A clause both rankings surface accumulates two reciprocal-rank contributions,
    so it outranks a clause that only one ranking found higher."""
    a = _hits("only_a", "both")     # 'both' at rank 2
    b = _hits("only_b", "both")     # 'both' at rank 2
    fused = [h["text"] for h in _rrf_fuse([a, b], k=3)]
    assert fused[0] == "both"


def test_rrf_keeps_hits_that_only_one_query_found():
    a = _hits("a1", "a2")
    b = _hits("b1", "b2")
    fused = [h["text"] for h in _rrf_fuse([a, b], k=4)]
    assert set(fused) == {"a1", "a2", "b1", "b2"}


def test_rrf_truncates_to_k():
    a = _hits("a1", "a2", "a3")
    b = _hits("b1", "b2", "b3")
    assert len(_rrf_fuse([a, b], k=2)) == 2


def test_rrf_deduplicates_by_text():
    a = _hits("same", "a2")
    b = _hits("same", "b2")
    fused = [h["text"] for h in _rrf_fuse([a, b], k=5)]
    assert fused.count("same") == 1


# ── _agentic_search_policy: rewrite + dual retrieval + fusion, end to end ───
@needs_pdfs
def test_agentic_search_falls_back_to_plain_bm25_when_rewrite_fails():
    """Rewrite dead → behave exactly like today's single-query BM25."""
    from utils.policy_rag import search_policy as plain

    q = "is refinancing an existing loan eligible"
    broken = _StubLLM(raises=RuntimeError("502"))
    agentic = _agentic_search_policy(q, k=3, llm=broken)
    assert [h["text"] for h in agentic] == [h["text"] for h in plain(q, 3)]


@needs_pdfs
def test_agentic_search_rescues_a_colloquial_query_the_raw_wording_misses():
    """The whole point, as one test: 'my place isn't built yet' shares no tokens
    with the 'Building Under Construction' clause, so plain BM25 cannot find it —
    the rewrite can. (The rewrite is stubbed with what the real model produces, so
    this asserts the RETRIEVAL wiring, not the model's phrasing.)"""
    from utils.policy_rag import search_policy as plain

    q = "my place isn't built yet, does it still count"
    llm = _StubLLM("Building Under Construction BUC property eligibility")

    def _mentions_buc(hits):
        return any("under construction" in h["text"].lower() for h in hits)

    assert not _mentions_buc(plain(q, 3))                       # baseline misses it
    assert _mentions_buc(_agentic_search_policy(q, k=3, llm=llm))  # agentic finds it


@needs_pdfs
def test_agentic_search_respects_k():
    llm = _StubLLM("promotion eligibility criteria")
    assert len(_agentic_search_policy("who qualifies", k=2, llm=llm)) <= 2


def test_fusion_pool_is_deeper_than_a_typical_k():
    """Regression guard for a real bug: fusing two k-length lists into k slots means
    each rewrite hit EVICTS one of the original query's, which knocked the original's
    rank-4/5 answers off the end (measured: verbatim Hit@5 -1.8%). The pool must be
    deeper than the k callers ask for, so fusion — not truncation — picks the winners."""
    assert _FUSION_POOL > 5


# ── is_set_question: the gate on the OPTIONAL iterate path ─────────────────
@pytest.mark.parametrize(
    "verdict, expected",
    [("SET", True), ("ONE", False), ("set", True), ("SET — it asks for the list", True)],
)
def test_is_set_question_parses_the_verdict(verdict, expected):
    assert is_set_question("which applications are excluded", llm=_StubLLM(verdict)) is expected


def test_is_set_question_fails_closed():
    """An unavailable judge must mean 'retrieve normally', not 'loop' or 'raise' —
    the same degrade contract rewrite_query has."""
    assert is_set_question("anything", llm=_StubLLM(raises=RuntimeError("502"))) is False
    assert is_set_question("", llm=_StubLLM("SET")) is False


# ── iterate=True: an opt-in switch that must not change the default path ───
@needs_pdfs
def test_iterate_defaults_off_and_costs_nothing_extra():
    """The set-question judge must NOT be consulted on the default path. It is an extra
    LLM call per search, and ~98% of queries are answered by a single clause — paying
    for it on every retrieval is exactly the cost the reference walkthrough incurs and
    we chose not to.

    The default path budget is exactly TWO calls — the rewrite and the listwise
    rerank (default-on since the measured 07-17 win) — and neither of them may be
    the set judge. The rerank prompt is identifiable by its 'Ranking:' cue."""
    llm = _StubLLM("promotion eligibility")
    _agentic_search_policy("who qualifies", k=3, llm=llm)
    assert len(llm.calls) == 2
    sent = " ".join(m.content for call in llm.calls for m in call)
    assert "SET" not in sent                      # no set-question judgement
    assert any("Ranking:" in m.content for m in llm.calls[1])   # 2nd call is the rerank


@needs_pdfs
def test_rerank_false_makes_no_rerank_call():
    """`rerank=False` must fully disable the reranker — this is what keeps the
    eval's rewrite-only arm honest (and unpaid) now that the production default
    is True. Regression: the arm silently reranked with live calls once."""
    llm = _StubLLM("promotion eligibility")
    _agentic_search_policy("who qualifies", k=3, llm=llm, rerank=False)
    assert len(llm.calls) == 1                    # the rewrite only


@needs_pdfs
def test_injected_empty_order_keeps_fused_order_without_llm_call():
    """`_rerank_order=[]` is the eval's sentinel for a cached failed-open rerank:
    keep the fused order, and do NOT retry the judgement live."""
    llm = _StubLLM("promotion eligibility")
    baseline = _agentic_search_policy("who qualifies", k=3, llm=llm, rerank=False)
    llm2 = _StubLLM("promotion eligibility")
    injected = _agentic_search_policy("who qualifies", k=3, llm=llm2, _rerank_order=[])
    assert [h["text"] for h in injected] == [h["text"] for h in baseline]
    assert len(llm2.calls) == 1                   # rewrite yes, rerank no


@needs_pdfs
def test_iterate_returns_exactly_k_like_every_other_mode():
    """The iterate arm must not widen the result window. Returning 5 chunks for a k=3
    query would let it score a rigged Hit@3 against a top-3 baseline — the comparison
    has to use one yardstick."""
    llm = _StubLLM("exclusions not eligible")
    hits = _agentic_search_policy("what deals don't work with this promo",
                                  k=3, llm=llm, iterate=True, _is_set=True)
    assert len(hits) <= 3


@needs_pdfs
def test_iterate_with_a_single_answer_question_matches_the_default_path():
    """iterate=True on a NON-set question must retrieve exactly as the default does:
    the enumeration pass only fires when the question actually asks for a list."""
    q = "is refinancing an existing loan eligible"
    rw = "refinancing existing loan eligibility"
    plain = _agentic_search_policy(q, k=3, llm=_StubLLM(rw))
    with_flag = _agentic_search_policy(q, k=3, llm=_StubLLM(rw), iterate=True, _is_set=False)
    assert [h["text"] for h in plain] == [h["text"] for h in with_flag]
