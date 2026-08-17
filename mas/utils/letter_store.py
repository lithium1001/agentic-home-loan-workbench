"""In-memory store for the most recent letter PDF per (applicant, stage).

The `draft_letter` tool renders the draft PDF and stashes the bytes + the inputs
it used here. `stream.py` reads the draft PDF's link at the HITL gate and, on an
Approve, re-renders the identical letter as the final (non-DRAFT) PDF from the
stored body/facts. `app.py`'s `/api/letter` endpoint streams the file on download.

Session-scoped in memory only (lost on restart) — the same no-DB trade-off as
`case_overrides` / `case_progress` / `stream.SESSIONS`. Kept in its own module so
the tool (writer), stream (writer) and app (reader) share it without a cycle."""

from __future__ import annotations

# {(APPID, STAGE, is_draft): (pdf_bytes, filename)}
_LETTERS: dict[tuple[str, str, bool], tuple[bytes, str]] = {}
# {(APPID, STAGE): (body_markdown, recipient, facts)} — the inputs the draft_letter
# tool used, so an Approve can re-render the identical letter as the final (non-
# DRAFT) PDF without re-running the agent.
_BODIES: dict[tuple[str, str], tuple[str, dict, dict]] = {}


def _key(applicant_id: str, stage: str, draft: bool) -> tuple[str, str, bool]:
    return (applicant_id.upper(), stage.upper(), bool(draft))


def _bkey(applicant_id: str, stage: str) -> tuple[str, str]:
    return (applicant_id.upper(), stage.upper())


def put(applicant_id: str, stage: str, draft: bool, pdf: bytes, filename: str) -> None:
    _LETTERS[_key(applicant_id, stage, draft)] = (pdf, filename)


def get(applicant_id: str, stage: str, draft: bool) -> tuple[bytes, str] | None:
    return _LETTERS.get(_key(applicant_id, stage, draft))


def put_body(applicant_id: str, stage: str, body: str, recipient: dict, facts: dict) -> None:
    _BODIES[_bkey(applicant_id, stage)] = (body, recipient, facts)


def get_body(applicant_id: str, stage: str) -> tuple[str, dict, dict] | None:
    return _BODIES.get(_bkey(applicant_id, stage))


def clear(applicant_id: str, stage: str) -> None:
    """Forget any letter held for this (applicant, stage).

    Needed by the eval harness, which runs many independent cases in ONE process:
    the store is keyed by (applicant, stage) and several golden queries share a
    borrower and stage, so a letter left by an earlier case would be read as proof
    that a later one produced a letter — inflating letter delivery and inventing a
    time-to-letter for a case that never drafted anything. Called before each case
    so "a letter exists" always means "THIS run made it".
    """
    for draft in (True, False):
        _LETTERS.pop(_key(applicant_id, stage, draft), None)
    _BODIES.pop(_bkey(applicant_id, stage), None)
