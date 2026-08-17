"""Policy retrieval layer — BM25 over the bank's PDF policy / T&C documents.

This is the knowledge source behind the ``search_policy`` tool (utils/tools.py):
an agent (currently Policy & Product) can retrieve the exact promotion / product
clauses instead of relying on the model's memory, so its answers cite real,
auditable terms.

Design (mirrors utils/calculator.py: pure, testable, zero LLM/network):
  - load_chunks(dir)  — read every ``*.pdf`` under ``dir`` (pypdf), split each
    into clause-numbered chunks (``1.1`` / ``2.3.1`` / ``(a)`` …). Table clauses
    (e.g. the Sign Up Gift tiers) are kept whole so an amount and its band stay
    in the same chunk even when the PDF flattens the table to a run of lines.
  - build_index(chunks) — a rank_bm25 ``BM25Okapi`` over a simple tokenizer.
  - search_policy(query, k) — top-k chunks as
    ``[{"title", "source", "text", "score"}]``.

The index is built lazily and cached at module level (like ``store`` in
utils.tools). It is a **best-effort** layer: if pypdf is missing, the directory
is empty, or nothing extracts, search_policy returns ``[]`` — the caller then
falls back to its skill.md rules and the graph never breaks.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

# Where the policy PDFs live. Relative to mas/ (the package root), matching the
# DATA_DIR convention in config.py. Override with RM_COPILOT_POLICY_DIR.
POLICY_DIR = os.getenv("RM_COPILOT_POLICY_DIR", "t&c")


@dataclass
class Chunk:
    source: str   # file name, e.g. "online-exclusive-tncs.pdf"
    title: str    # clause number + first line, e.g. "2.1 Under this Promotion, …"
    text: str     # the full clause text (the retrievable unit)


# ── PDF text extraction ────────────────────────────────────────────────────
def _extract_pdf_text(path: Path) -> str:
    """Best-effort full-text extraction of a PDF. Returns '' on any failure
    (missing pypdf, encrypted/scanned PDF, parse error) so the caller degrades
    to an empty index rather than raising."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


# A clause heading is a number like ``1.``, ``1.1``, ``2.3.1`` or a lettered
# sub-point ``(a)`` at the start of a line. We split on these so each numbered
# clause becomes one retrievable chunk; the heading itself is kept as the
# chunk's citation title.
_CLAUSE_RE = re.compile(r"(?m)^\s*((?:\d+\.)+\d*|\([a-z]+\)|\d+\.)\s+")


def _split_clauses(text: str, source: str) -> list[Chunk]:
    """Split one document's text into clause-numbered chunks.

    Splitting on clause headings keeps a table clause (e.g. the Sign Up Gift
    tiers under 2.1) whole — its amounts and bands stay in the same chunk even
    though the PDF flattens the table into a run of lines. Text before the first
    numbered clause (title / preamble) becomes a leading chunk so nothing is lost.
    """
    # Normalise whitespace-only lines; collapse runs of blank lines.
    text = re.sub(r"[ \t]+\n", "\n", text)
    matches = list(_CLAUSE_RE.finditer(text))
    if not matches:
        body = text.strip()
        return [Chunk(source=source, title=f"{source} (full text)", text=body)] if body else []

    chunks: list[Chunk] = []
    # Preamble before the first clause (document title, defined terms header).
    preamble = text[: matches[0].start()].strip()
    if preamble:
        first_line = preamble.splitlines()[0].strip()
        chunks.append(Chunk(source=source, title=first_line[:80], text=preamble))

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause = text[start:end].strip()
        if not clause:
            continue
        number = m.group(1).strip()
        # Title = clause number + first meaningful line, for a readable citation.
        first_line = clause.splitlines()[0].strip()
        title = first_line if first_line else number
        chunks.append(Chunk(source=source, title=title[:100], text=clause))
    return chunks


# Clause chunks shorter than this are almost always a bare heading the PDF split
# onto its own line ("2. Conditions Precedent") — too little text for BM25 to
# match on, so they are dropped as noise.
_MIN_CHUNK_CHARS = 40


def load_chunks(policy_dir: str | Path | None = None) -> list[Chunk]:
    """Load and clause-split every PDF under ``policy_dir`` (default POLICY_DIR).

    Two cleanup passes make multi-file / long-document corpora healthy:
      - drop bare-heading fragments (< _MIN_CHUNK_CHARS), which add BM25 noise;
      - de-duplicate identical clause text across the whole corpus. Some PDFs
        re-flow a clause onto several pages, so pypdf's per-page extraction emits
        the same text more than once; without this, a handful of clauses would
        fill every top-k result with their own copies.

    Returns [] if the directory is absent or holds no readable PDF text — the
    empty-index / degrade path."""
    base = Path(policy_dir if policy_dir is not None else POLICY_DIR)
    if not base.is_dir():
        return []
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for pdf in sorted(base.glob("*.pdf")):
        raw = _extract_pdf_text(pdf)
        if not raw.strip():
            continue
        for c in _split_clauses(raw, pdf.name):
            body = c.text.strip()
            if len(body) < _MIN_CHUNK_CHARS:
                continue                 # bare heading / stray fragment
            if body in seen:
                continue                 # identical clause already indexed
            seen.add(body)
            chunks.append(c)
    return chunks


# ── BM25 index ──────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A bank's T&C defines its own abbreviation up front — "<Full Legal Name>
# ('ABC')" — and then repeats the full name in headings and boilerplate. Left
# as-is, those repeats donate the name's constituent words to clauses that have
# nothing to do with a query using them: a legal name is built from ordinary
# words, so the measured failure was a query using one of them retrieving the
# document TITLE. Applying the document's own abbreviation at TOKENIZATION time
# (both sides — chunks and queries go through this same function) removes that
# false lexical channel. Chunk .text is untouched: citations must show the
# document's real wording.
#
# Which name to collapse depends entirely on the corpus loaded under POLICY_DIR,
# so it is configuration, not code: set both vars to the issuing institution's
# legal name and its defined short form. Left unset, tokenization is a plain
# lowercase split — correct for any corpus, just without this one refinement.
BANK_LEGAL_NAME = os.getenv("RM_COPILOT_BANK_LEGAL_NAME", "")
BANK_SHORT_NAME = os.getenv("RM_COPILOT_BANK_SHORT_NAME", "")

def _bank_name_pattern(legal_name: str) -> re.Pattern[str] | None:
    """A regex matching the configured legal name as a document writes it, or None
    if unconfigured. Whitespace matches any run of it, so a name broken across
    lines by PDF extraction still collapses, and a trailing incorporation suffix
    is optional because boilerplate drops it ('… Bank Limited' vs '… Bank')."""
    words = legal_name.split()
    if not words:
        return None
    if len(words) > 1 and words[-1].lower().rstrip(".") in {"limited", "ltd", "berhad"}:
        head, suffix = words[:-1], words[-1]
        body = r"\s+".join(re.escape(w) for w in head)
        return re.compile(rf"{body}(\s+{re.escape(suffix)})?", re.I)
    return re.compile(r"\s+".join(re.escape(w) for w in words), re.I)


_BANK_NAME_RE = _bank_name_pattern(BANK_LEGAL_NAME)


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer — sufficient for BM25 over clause text
    (no stemming/stopwords; clause search terms like 'refinancing', 'sign up
    gift', 'body corporate' are already discriminative). The one normalisation
    is the document's own defined abbreviation (see BANK_LEGAL_NAME)."""
    lowered = text.lower()
    if _BANK_NAME_RE is not None:
        lowered = _BANK_NAME_RE.sub(BANK_SHORT_NAME.lower(), lowered)
    return _TOKEN_RE.findall(lowered)


class PolicyIndex:
    """A BM25 index over policy chunks. Empty-safe: an index with no chunks
    returns no results rather than raising."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._bm25 = None
        if chunks:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def search(self, query: str, k: int = 4) -> list[dict]:
        if not self._bm25 or not query.strip():
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[dict] = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                break   # no term overlap beyond here — don't return noise
            c = self.chunks[i]
            out.append({
                "title": c.title,
                "source": c.source,
                "text": c.text,
                "score": round(float(scores[i]), 3),
            })
        return out


def build_index(chunks: list[Chunk]) -> PolicyIndex:
    return PolicyIndex(chunks)


# ── Clause structure ────────────────────────────────────────────────────────
# A lettered sub-clause heading: "(a) Private Home Loan involving …". The T&C
# enumerates its sets — the exclusions, the ineligible persons — as runs of these.
_SUBCLAUSE_RE = re.compile(r"^\(([a-z])\)\s")


def subclause_letter(chunk_title: str) -> str | None:
    """The letter of a lettered sub-clause heading ("(a) …"), else None."""
    m = _SUBCLAUSE_RE.match((chunk_title or "").strip())
    return m.group(1) if m else None


def sibling_runs(chunks: list[Chunk] | None = None) -> list[list[int]]:
    """Group the corpus's lettered sub-clauses into the enumerated LISTS they form.

    Siblings are *contiguous* chunks, not same-letter chunks: this T&C holds four
    independent lettered runs (eligibility steps, exclusions (a)-(e), liability
    carve-outs, ineligible persons), so keying on the letter alone would treat one
    list's "(b)" as completing another's.

    Used by the iterative retriever to know how many clauses a set-answer spans, and
    therefore how deep to retrieve — not to *decide* whether the question wants a set
    (that is a property of the question; see query_rewrite.is_set_question).
    """
    corpus = chunks if chunks is not None else _get_index().chunks
    runs: list[list[int]] = []
    current: list[int] = []
    for i, c in enumerate(corpus):
        if subclause_letter(c.title):
            if current and i == current[-1] + 1:
                current.append(i)
                continue
            if len(current) > 1:
                runs.append(current)
            current = [i]
        else:
            if len(current) > 1:
                runs.append(current)
            current = []
    if len(current) > 1:
        runs.append(current)
    return runs


def longest_sibling_run(chunks: list[Chunk] | None = None) -> int:
    """Size of the largest enumerated sub-clause list in the corpus.

    This number is why iterative retrieval cannot rescue a set-seeking question at a
    fixed k, and the retrieval eval bears it out: the exclusions run to FIVE sub-clauses
    (1.3.1(a)-(e)), so "which applications are excluded" needs five slots. At k=3 the
    answer does not fit — that is arithmetic, not retrieval quality, and no number of
    LLM rounds changes it. Measured: adding the iteration pass moved Recall@k by 0.0pp
    once results were truncated to k (see the A/B in this module's __main__).

    The real fixes it points at are a wider / adaptive top-k, or merging an enumerated
    run into a single chunk at split time — both future work.
    """
    runs = sibling_runs(chunks)
    return max((len(r) for r in runs), default=0)


# ── Lazy module-level singleton (like utils.tools.store) ────────────────────
_INDEX: PolicyIndex | None = None


def _get_index() -> PolicyIndex:
    global _INDEX
    if _INDEX is None:
        _INDEX = build_index(load_chunks())
    return _INDEX


def search_policy(query: str, k: int = 4) -> list[dict]:
    """Retrieve the top-k most relevant policy clauses for ``query``.

    Returns a list of ``{"title", "source", "text", "score"}`` dicts, or ``[]``
    if no document is indexed / nothing matches (the degrade path)."""
    return _get_index().search(query, k)
