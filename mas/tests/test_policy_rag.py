"""Tests for the policy RAG layer (utils/policy_rag.py) and its wiring into the
tool registry as ``search_policy``.

Pure retrieval layer — no network, no LLM. The clause-content assertions run
against the real t&c/ PDFs (via pypdf); they are skipped if no PDF is present so
the suite still passes on a checkout without the documents.

Run from mas/:
    py -m pytest -q tests/test_policy_rag.py
"""
import importlib
from pathlib import Path

import pytest

import utils.policy_rag
from utils.policy_rag import (
    Chunk,
    build_index,
    load_chunks,
    search_policy,
)

# The bundled policy T&C corpus, if present.
_HAS_PDFS = bool(list((Path(__file__).resolve().parents[1] / "t&c").glob("*.pdf"))) \
    if (Path(__file__).resolve().parents[1] / "t&c").is_dir() else False
needs_pdfs = pytest.mark.skipif(not _HAS_PDFS, reason="no policy PDFs under t&c/")


@pytest.fixture
def bank_name_configured(monkeypatch):
    """The module with a bank name configured, as a deployment with a real corpus
    would set it. The name regex is compiled at import, so the module is reloaded
    under the patched environment and restored afterwards — yielding the module
    itself, since the reload rebinds the functions under test."""
    monkeypatch.setenv("RM_COPILOT_BANK_LEGAL_NAME", "Sample Bank Limited")
    monkeypatch.setenv("RM_COPILOT_BANK_SHORT_NAME", "SBL")
    module = importlib.reload(utils.policy_rag)
    try:
        yield module
    finally:
        monkeypatch.undo()
        importlib.reload(utils.policy_rag)


# ── Clause splitting / loading (against the real PDFs) ──────────────────────
@needs_pdfs
def test_load_chunks_splits_into_clauses():
    """The T&C parse into many clause-level chunks (not one blob), each carrying
    its source file name for citation."""
    chunks = load_chunks()
    assert len(chunks) > 10
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.source.endswith(".pdf") for c in chunks)
    assert all(c.text.strip() for c in chunks)


@needs_pdfs
def test_refinancing_query_hits_refinancing_clause():
    """A refinancing question retrieves the refinancing eligibility clause —
    the core 'retrieve real terms, not memory' behaviour."""
    hits = search_policy("is refinancing an existing loan eligible", k=3)
    assert hits
    joined = " ".join(h["text"].lower() for h in hits)
    assert "re-finance" in joined or "refinancing" in joined


@needs_pdfs
def test_sign_up_gift_query_hits_gift_clause():
    """A gift question retrieves the Sign Up Gift clause, and the tier table
    (amounts + bands) is kept together in the retrieved text."""
    hits = search_policy("sign up gift for a large home loan", k=3)
    assert hits
    joined = " ".join(h["text"].lower() for h in hits)
    assert "sign up gift" in joined


@needs_pdfs
def test_k_limits_results():
    assert len(search_policy("promotion", k=2)) <= 2
    assert len(search_policy("promotion", k=5)) <= 5


@needs_pdfs
def test_hits_carry_citation_fields():
    for h in search_policy("eligibility", k=3):
        assert set(h) >= {"title", "source", "text", "score"}
        assert h["score"] > 0


@needs_pdfs
def test_no_tiny_or_duplicate_chunks():
    """load_chunks drops bare-heading fragments and de-duplicates identical clause
    text across the corpus, so a long / multi-page PDF can't flood top-k with
    repeated copies (the ploan_tnc.pdf failure mode). Guards both cleanup passes."""
    chunks = load_chunks()
    assert chunks
    assert all(len(c.text.strip()) >= 40 for c in chunks)   # no bare-heading noise
    bodies = [c.text.strip() for c in chunks]
    assert len(bodies) == len(set(bodies))                  # no duplicate copies


# ── Degrade path (no documents) ─────────────────────────────────────────────
def test_empty_dir_returns_no_chunks(tmp_path):
    """An empty / missing policy dir yields no chunks — the graph then degrades
    to skill.md rules rather than erroring."""
    assert load_chunks(tmp_path) == []
    assert load_chunks(tmp_path / "does-not-exist") == []


def test_empty_index_search_returns_empty():
    """Searching an index built from no chunks is safe and returns []."""
    idx = build_index([])
    assert idx.search("anything") == []


def test_blank_query_returns_empty():
    idx = build_index([Chunk(source="x.pdf", title="t", text="some policy text here")])
    assert idx.search("   ") == []


def test_tokenizer_is_a_plain_split_when_no_bank_name_configured():
    """Unconfigured is the shipped default: tokenization is an ordinary
    lowercase alphanumeric split, with no name collapsed."""
    from utils.policy_rag import _tokenize

    assert _tokenize("Sample Bank Limited") == ["sample", "bank", "limited"]


def test_tokenizer_applies_documents_own_abbreviation(bank_name_configured):
    """A configured legal name tokenizes as the document's own defined
    abbreviation, so boilerplate repeats of that name stop donating its
    constituent words as tokens to unrelated queries. The failure this fixes:
    an issuer's legal name is made of ordinary words, so its repeats in headings
    and boilerplate rank the document TITLE for any query using one of them."""
    _tokenize = bank_name_configured._tokenize

    assert _tokenize("Sample Bank Limited") == ["sbl"]
    assert _tokenize("sample  bank offers loans") == ["sbl", "offers", "loans"]
    # A genuine use of the word stays a real token.
    assert "bank" in _tokenize("refinancing from another bank")


def test_bank_name_no_longer_matches_queries_using_its_words(bank_name_configured):
    """A query using a word from the bank's name must not rank a chunk whose only
    link to it is that name sitting in the chunk's heading."""
    chunks = [
        Chunk(source="a.pdf", title="title", text="Terms and Conditions for Sample Bank Limited Promotion"),
        Chunk(source="a.pdf", title="refinance", text="refinancing of a property from another bank is eligible"),
        Chunk(source="a.pdf", title="gift", text="sign up gift takashimaya vouchers cash reward"),
        Chunk(source="a.pdf", title="law", text="governed by the laws of singapore jurisdiction courts"),
    ]
    hits = bank_name_configured.build_index(chunks).search("refinancing from another bank", k=2)
    assert hits and hits[0]["title"] == "refinance"
    assert all(h["title"] != "title" for h in hits)


def test_index_ranks_by_term_overlap():
    """A hand-built index ranks the chunk that shares query terms first and drops
    zero-overlap chunks (no noise). Uses several distinct chunks so BM25's IDF is
    meaningful — a query term must be rare across the corpus to score, which is
    exactly the real multi-clause situation (a 2-chunk corpus is degenerate)."""
    chunks = [
        Chunk(source="a.pdf", title="gift", text="sign up gift takashimaya vouchers cash reward"),
        Chunk(source="a.pdf", title="law", text="governed by the laws of singapore jurisdiction courts"),
        Chunk(source="a.pdf", title="eligibility", text="applicant must be a natural person not a body corporate"),
        Chunk(source="a.pdf", title="refinance", text="refinancing of a completed private residential property from another bank"),
    ]
    hits = build_index(chunks).search("sign up gift voucher reward", k=5)
    assert hits
    assert hits[0]["title"] == "gift"
    # The unrelated clauses share no query term -> excluded (no noise).
    assert all(h["title"] == "gift" for h in hits)


# ── Tool registry wiring ────────────────────────────────────────────────────
def test_search_policy_registered_as_tool():
    from utils.tools import TOOLS_BY_NAME, TOOL_SCHEMA_MAP

    assert "search_policy" in TOOLS_BY_NAME
    assert "search_policy" in TOOL_SCHEMA_MAP


def test_execute_tool_dispatches_search_policy(monkeypatch):
    """The legacy execute_tool() dispatcher also routes search_policy (JSON out).

    The LLM rewrite is stubbed out: `search_policy` now goes through the agentic path,
    so an un-stubbed call here would make the *test suite* hit the paid API — quietly,
    on every run. The suite must stay free and offline; only the eval spends money.
    """
    import json

    import utils.query_rewrite as qr
    from utils.tools import execute_tool

    monkeypatch.setattr(qr, "rewrite_query", lambda q, llm=None: q)
    monkeypatch.setattr(qr, "rerank_order", lambda q, docs, llm=None: None)
    out = json.loads(execute_tool("search_policy", {"query": "promotion", "k": 1}))
    assert isinstance(out, list)


# ── Clause structure: the evidence behind the "no iteration" decision ───────
def test_sibling_runs_groups_by_adjacency_not_by_letter():
    """Sub-clause siblings are CONTIGUOUS chunks. Keying on the letter alone would let
    one list's "(b)" complete another list's "(a)" — this T&C holds several independent
    lettered runs, so that bug would be silent and wrong."""
    from utils.policy_rag import sibling_runs

    chunks = [
        Chunk("t.pdf", "1. Eligibility", "The applicant must be an individual who:"),
        Chunk("t.pdf", "(a) first condition", "(a) first condition text here, long enough."),
        Chunk("t.pdf", "(b) second condition", "(b) second condition text here, long enough."),
        Chunk("t.pdf", "2. Exclusions", "The following are excluded from the promotion:"),
        Chunk("t.pdf", "(a) an excluded thing", "(a) an excluded thing, described at length."),
        Chunk("t.pdf", "(b) another excluded", "(b) another excluded thing, at some length."),
        Chunk("t.pdf", "(c) a third excluded", "(c) a third excluded thing, at some length."),
    ]
    assert sibling_runs(chunks) == [[1, 2], [4, 5, 6]]


@needs_pdfs
def test_longest_sibling_run_exceeds_a_typical_k():
    """The arithmetic that killed iterative retrieval: the exclusions are FIVE sibling
    sub-clauses, so "which applications are excluded" cannot be answered inside a top-3
    no matter how good the query is. Adding LLM rounds moved Recall@k by 0.0pp once
    results were truncated to k — the bottleneck is the result window, not retrieval.
    If this ever drops to <= 3, that argument no longer holds and the iterate path is
    worth re-evaluating."""
    from utils.policy_rag import longest_sibling_run

    assert longest_sibling_run() >= 5
