"""Runtime case-progress store — what the RM has actually COMPLETED this session.

The CSV `application_status` is seed data: it never changes as the RM drives the
workflow, so the deal-progress dots could only reflect the canned starting state.
This module records the milestones that genuinely happen at runtime — an IPA letter
or a Letter of Offer being **approved by the RM at the HITL gate** — so the left-rail
progress (and the next-best-action chips that read it) advance as the RM works.

Milestone granularity = **HITL approval**, not "the flow ran". A draft is only a
milestone once the RM has signed off on it at the review gate; running the flow but
rejecting the draft does NOT advance the stage.

Scope: in-memory, keyed by applicant id, peer to stream.SESSIONS. Lost on process
restart (same trade-off as SESSIONS) — fine for the demo; swap for SQLite if cross-
restart persistence is ever needed. Kept in its own module so both case_service
(reader) and stream (writer) import it without a circular dependency.
"""

from __future__ import annotations

# Two milestone maps, both keyed by applicant id:
#   _ASSESSED  — the RM ran the eligibility/package ASSESSMENT flow to a verdict
#                (full_*_assess ran to END). This is the lighter, earlier step.
#   _COMPLETED — the RM APPROVED this stage's drafted letter at the HITL gate.
# Both feed the next-best-action chips; only _COMPLETED advances the deal dots.
_ASSESSED: dict[str, dict[str, bool]] = {}
_COMPLETED: dict[str, dict[str, bool]] = {}


def mark_stage_assessed(applicant_id: str, stage: str) -> None:
    """Record that the RM ran this stage's assessment flow to an eligibility/package
    verdict (no letter yet). Called when a full_*_assess flow runs to END."""
    stage = stage.upper()
    if stage not in ("IPA", "LO"):
        return
    _ASSESSED.setdefault(applicant_id.upper(), {})[stage] = True


def mark_stage_completed(applicant_id: str, stage: str) -> None:
    """Record that the RM approved this stage's draft at the HITL gate.

    Called from the stream layer's Approve path. REPRICE is a standalone advisory
    lane with no build/draft chain, so it never produces a HITL approval and is
    simply never marked here."""
    stage = stage.upper()
    if stage not in ("IPA", "LO"):
        return
    _COMPLETED.setdefault(applicant_id.upper(), {})[stage] = True


def assessed_stages(applicant_id: str) -> dict[str, bool]:
    """Stages the RM has assessed this session (empty dict if none yet)."""
    return dict(_ASSESSED.get(applicant_id.upper(), {}))


def completed_stages(applicant_id: str) -> dict[str, bool]:
    """Stages the RM has completed (draft approved) this session."""
    return dict(_COMPLETED.get(applicant_id.upper(), {}))
