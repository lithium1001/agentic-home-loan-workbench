"""Query rewriting — the agentic step in front of the BM25 policy retriever.

Why this exists (measured, not assumed): ``eval/retrieval.py`` breaks Hit@k down
by query style, and the split is stark — verbatim questions that reuse the T&C's
own wording retrieve at ~95%+, while colloquial ones retrieve at ~33%. Every miss
is the same failure: **the index is fine, the query is wrong**. BM25 scores on
shared tokens, and an RM who types

    "my place isn't built yet, does it still count"

shares *zero* tokens with the clause that answers it, which says "Building Under
Construction (BUC)". No amount of re-chunking fixes that; the query has to meet
the document's vocabulary.

So this module does the one thing the reference agentic-RAG walkthrough does
before retrieving: an LLM rewrites the user's question into the document's
register (contract nouns, no chit-chat, no first person) and hands *that* to BM25.

Design constraints this file is written to respect:
  - ``utils/policy_rag.py`` stays **pure** (zero LLM, zero network) — it is the
    index, and its unit tests must keep running offline. The LLM lives here, one
    layer up, and the two are composed by the ``search_policy`` @tool wrapper.
  - **Never break the caller.** A missing API key, a 500, a model that answers
    with an essay — all degrade to returning the original query, so retrieval is
    exactly as good as it is today and the graph never dies on a rewrite.
  - The rewrite is *additive*: callers OR the two queries' results together
    (see ``utils.tools.search_policy``), so a rewrite that drifts off-topic can
    only add noise below the original query's hits — it cannot delete them. That
    is what protects the 95% verbatim baseline from regressing.
"""

from __future__ import annotations

import re

# Rewriting is a retrieval-time transformation, not a conversation: the model
# must emit ONE line of search terms and nothing else. Everything that could
# tempt it into prose (greetings, "Sure, here's...", explanations) is closed off
# explicitly, because any extra word becomes a BM25 token that dilutes the score.
#
# The few-shot examples below are deliberately drawn from home-loan contract
# vocabulary that appears NOWHERE in the retrieval golden set (lock-in period,
# prepayment penalty, guarantor, fire insurance). Seeding them with the actual
# terms the golden set's misses turn on (BUC, body corporate, repricing, …) would
# be test-set leakage: the eval would then be measuring how well we hand-fed the
# answers, not whether the model can generalise the colloquial→contract mapping.
_SYSTEM = """You rewrite a user's question into a search query for a keyword \
(BM25) search over a bank's home-loan Terms & Conditions document.

BM25 matches on shared words, so the rewritten query must use the words the \
CONTRACT would use, not the words a customer would use.

Rules:
- Replace colloquial phrasing with the formal/legal term a T&C document uses.
  "if I pay it off early do I get fined" -> "prepayment penalty early redemption fee"
  "how long am I stuck with this"        -> "lock-in period commitment term"
  "my dad will co-sign for me"           -> "guarantor joint borrower undertaking"
  "do I have to insure the flat"         -> "fire insurance mortgage requirement"
- Keep every distinctive noun already present (loan amount, product name, "gift").
- Expand the key concept with 1-3 likely synonyms from contract vocabulary.
- Drop first person, politeness, and questions words ("can I", "do I", "what if").
- Output ONE line of search terms. No punctuation-heavy prose, no explanation,
  no quotes, no preamble."""

_USER = """Question: {q}

Search query:"""

# A rewrite is only usable if it still looks like a query. These guard against the
# two ways a small model fails here: it answers the question instead of rewriting
# it (long prose), or it emits a refusal / empty string.
_MAX_REWRITE_CHARS = 300
_MIN_REWRITE_CHARS = 3


def _clean(text: str) -> str:
    """Strip the wrappers a chat model adds around a one-line answer: a leading
    'Search query:' echo, surrounding quotes, markdown bullets, trailing periods."""
    if not text:
        return ""
    line = text.strip()
    # Models often echo the label, or answer on the line after it.
    line = re.sub(r"(?i)^\s*(search\s+query|rewritten\s+query|query)\s*:\s*", "", line)
    # Take the first non-empty line — a stray explanation usually follows on line 2.
    for candidate in line.splitlines():
        candidate = candidate.strip()
        if candidate:
            line = candidate
            break
    line = line.strip().strip('"').strip("'").lstrip("-*• ").strip()
    return line


def _looks_usable(rewritten: str) -> bool:
    """True if the rewrite is a plausible search query rather than prose or junk."""
    if not (_MIN_REWRITE_CHARS <= len(rewritten) <= _MAX_REWRITE_CHARS):
        return False
    # An answer (rather than a query) reads like a sentence: many words + a verb-y
    # ending. Cheap proxy: a rewrite should not be a paragraph.
    return rewritten.count(".") <= 2


# Decoding is pinned as hard as a hosted endpoint allows, because the retrieval
# metric is only reproducible if the rewrite is.
#
# ``temperature=0`` alone is NOT enough, and we measured that: two runs of the same
# 99 golden queries with identical code and prompt scored Hit@3 = 83.8% and 78.8%.
# On a hosted, load-balanced endpoint (OpenRouter fans out across providers and GPU
# kernels) floating-point reduction order differs run to run, so near-ties in the
# logits break differently — greedy decoding is deterministic in theory, not in
# deployment. ``seed`` pins the sampler on providers that honour it, and top_p=1
# avoids nucleus truncation interacting with those ties.
#
# It is still best-effort: no hosted endpoint guarantees bit-identical decoding. That
# is precisely why the eval caches rewrites to disk (eval/rewrite_cache.py) — the
# cache, not the endpoint, is what makes a reported number re-derivable.
_SEED = 20260714


def _rewrite_llm():
    """The rewrite's own client: pinned decoding, built lazily so importing this
    module never requires an API key."""
    from utils.llm import make_llm

    llm = make_llm(temperature=0)
    return llm.bind(seed=_SEED, top_p=1)


def _ask(llm, system: str, user: str) -> str:
    """One pinned-decoding LLM turn returning stripped text. Shared by the three
    agentic steps so they cannot drift apart in how they call the model."""
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = (llm or _rewrite_llm()).invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    return (getattr(resp, "content", "") or "").strip()


def rewrite_query(query: str, llm=None) -> str:
    """Rewrite ``query`` into T&C vocabulary for BM25. Returns the ORIGINAL query
    unchanged on any failure (no key, network error, unusable output) — the caller
    then retrieves exactly as it does today.

    ``llm`` is injectable so tests and the eval harness can pass a stub / a
    shared client instead of building a new one per call.
    """
    if not query or not query.strip():
        return query
    try:
        if llm is None:
            llm = _rewrite_llm()
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = llm.invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=_USER.format(q=query.strip())),
        ])
        rewritten = _clean(getattr(resp, "content", "") or "")
    except Exception:  # noqa: BLE001 — a rewrite must never break retrieval
        return query
    return rewritten if _looks_usable(rewritten) else query


# ══════════════════════════════════════════════════════════════════════════
# Set-seeking detection — our replacement for the reference's _needs_more_context.
#
# The failure class: a question whose answer spans several clauses at once. "what
# deals don't work with this promo" is answered by 1.3.1(a) THROUGH (e) — five
# sub-clauses. A top-3 physically cannot hold five answers, so no rewrite fixes it;
# the retriever must fetch DEEPER and return the whole run.
#
# Why we do not copy the reference here. Its `_needs_more_context` shows the LLM the
# retrieved text after every round and asks "is this sufficient?", so a 3-round loop
# costs ~4 LLM calls per search — paid on all 99 queries, including the ~98% whose
# answer is a single clause and who never needed it. We instead classify the QUESTION,
# once, before retrieving:
#
#   "which applications are excluded"  -> wants the whole list   (set-seeking)
#   "is a bridging loan excluded"      -> wants one member of it (not set-seeking)
#
# Both retrieve the same chunks, so no signal computed from the RESULTS can separate
# them — we measured that: a structural "did we get all the siblings" test fires on
# 70/99 golden queries, because retrieving one sibling is the *correct* answer to the
# second question. The distinction lives in the question and nowhere else.
#
# One call, cacheable, and it fires only where it can help.
# ══════════════════════════════════════════════════════════════════════════

_SET_QUESTION_SYSTEM = """You classify a question about a bank's home-loan Terms & \
Conditions.

Answer SET if the question asks for ALL the members of a list — every exclusion, all \
the eligibility conditions, each tier, the full set of situations.
Answer ONE if the question asks whether ONE specific thing is covered, or asks for a \
single fact, even if that thing happens to belong to a list.

  "which applications are excluded from this promotion" -> SET
  "what deals don't work with this promo"               -> SET
  "is a bridging loan excluded"                         -> ONE
  "can I restructure an existing loan under this"       -> ONE
  "how much is the gift for a 1.2m loan"                -> ONE

Reply with exactly one word: SET or ONE."""


def is_set_question(question: str, llm=None) -> bool:
    """True if the question asks for a whole enumerated list rather than one fact.

    Fails CLOSED (False) on any error: an unavailable judge means we retrieve normally
    rather than loop, so the retriever degrades to the non-iterative path instead of
    breaking. Same principle as rewrite_query.
    """
    if not question or not question.strip():
        return False
    try:
        verdict = _ask(llm, _SET_QUESTION_SYSTEM, f"Question: {question}\n\nSET or ONE:")
    except Exception:  # noqa: BLE001 — the judge must never break retrieval
        return False
    return verdict.strip().upper().startswith("SET")


# ══════════════════════════════════════════════════════════════════════════
# Listwise rerank — the third agentic step, added after the 07-17 miss analysis.
#
# The per-style error analysis split the remaining misses into two classes:
#   - answer NOT in the fused candidate pool at all  → a vocabulary wall; only a
#     dense channel or new vocabulary can help. Out of scope here.
#   - answer IN the pool but ranked 6-19             → an ORDERING failure. BM25
#     ranks by token overlap, and a clause that shares two generic tokens can
#     outrank the right clause that shares one rare one. Paraphrase queries had
#     Hit@5 53% but Hit@10 73% — a 20-point gap that is pure ordering.
#
# The fix for the second class is a reranker: hand the top-N fused candidates to
# the LLM and ask it to order them by how well they ANSWER the question (semantic
# judgement, which BM25 cannot make). Standard IR practice — cross-encoder
# reranking in BEIR; LLM listwise reranking (RankGPT et al.) is its LLM form.
#
# Same contract as the other two steps: pinned decoding, and fail-OPEN — any
# error returns None and the caller keeps the fused BM25 order, so the reranker
# can only ever be additive on top of a working pipeline.
# ══════════════════════════════════════════════════════════════════════════

_RERANK_SYSTEM = """You rank clauses from a bank's home-loan Terms & Conditions \
by how well they ANSWER a user's question.

The best clause is the one whose content answers the question — not the one that \
shares the most words with it.

Reply with the clause numbers only, best first, comma-separated. No explanation, \
no other text.

Example reply: 3, 1, 4, 2, 5"""

# Snippet length per candidate. Long enough to show what a clause regulates,
# short enough that a 20-candidate prompt stays ~1.5k tokens.
_SNIPPET_CHARS = 280
# A sane reply is a short list of numbers; an essay means the model answered the
# question instead of ranking. Cap chosen well above "20 numbers + separators".
_MAX_RERANK_REPLY = 200


def rerank_order(query: str, docs: list[str], llm=None) -> list[int] | None:
    """Order for ``docs`` (0-based indices, best first) by how well each answers
    ``query`` — or None on ANY failure, in which case the caller keeps its
    existing (fused BM25) order. Fail-open, like rewrite_query.

    Indices the model omits are appended in their original order: the reranker
    may promote, but a clause can never vanish because the model forgot it.
    """
    if not query or not query.strip() or len(docs) < 2:
        return None
    lines = [f"{i + 1}. {d[:_SNIPPET_CHARS]}".replace("\n", " ") for i, d in enumerate(docs)]
    user = f"Question: {query.strip()}\n\nClauses:\n" + "\n".join(lines) + "\n\nRanking:"
    try:
        raw = _ask(llm, _RERANK_SYSTEM, user)
    except Exception:  # noqa: BLE001 — the reranker must never break retrieval
        return None
    # First non-empty line only — a stray explanation usually follows on line 2.
    line = next((ln.strip() for ln in (raw or "").splitlines() if ln.strip()), "")
    if not line or len(line) > _MAX_RERANK_REPLY:
        return None
    order: list[int] = []
    seen: set[int] = set()
    for tok in re.findall(r"\d+", line):
        i = int(tok) - 1
        if 0 <= i < len(docs) and i not in seen:
            seen.add(i)
            order.append(i)
    if not order:
        return None
    order.extend(i for i in range(len(docs)) if i not in seen)
    return order
