"""Runtime calculation-override store — the recompute inputs the RM changed this
session via the reject→replan loop (e.g. "reprice the letter to 1.2%").

The reject→recompute path in the graph writes an `overrides` dict
(`interest_rate_pct` / `target_property_price` / `n_props_owned`) into the graph
state so the agent recomputes the draft's numbers. But that override never
touched the CSV loan-application row, so `case_service._loan_summary` kept pricing
the KPI card at the ORIGINAL rate — the RM would change the rate in chat and the
Loan Scenario card would not move.

This module remembers, per (applicant, stage), the last override the RM applied,
so `_loan_summary` can re-price the KPI card at the adjusted value. It is a
read-model overlay only: the seed CSV is never mutated (guide, don't mutate), and
the override is session-scoped in memory — lost on restart, same trade-off as
`case_progress` / `stream.SESSIONS`. Kept in its own module so both `stream`
(writer) and `case_service` (reader) import it without a circular dependency.
"""

from __future__ import annotations

# The only keys `calculate_loan` accepts as overrides (mirror of graph._ALLOWED).
_ALLOWED = {"interest_rate_pct", "target_property_price", "n_props_owned"}

# {(APPID, STAGE): {"interest_rate_pct": 1.2, ...}} — last override the RM applied.
_OVERRIDES: dict[tuple[str, str], dict] = {}


def _key(applicant_id: str, stage: str) -> tuple[str, str]:
    return (applicant_id.upper(), stage.upper())


def apply_overrides(applicant_id: str, stage: str, overrides: dict) -> None:
    """Merge an RM recompute override into this (applicant, stage)'s stored set.
    Silently drops any key that is not a real `calculate_loan` param."""
    if not overrides:
        return
    clean = {k: v for k, v in overrides.items() if k in _ALLOWED and v is not None}
    if not clean:
        return
    _OVERRIDES.setdefault(_key(applicant_id, stage), {}).update(clean)


def get_overrides(applicant_id: str, stage: str) -> dict:
    """The overrides the RM has applied to this (applicant, stage), or {}.

    IPA and LO share the same underlying loan scenario, so an override applied in
    one stage should also price the other's card — merge both onto the requested
    stage (the exact stage wins on a key clash)."""
    other = "LO" if stage.upper() == "IPA" else ("IPA" if stage.upper() == "LO" else None)
    merged: dict = {}
    if other:
        merged.update(_OVERRIDES.get(_key(applicant_id, other), {}))
    merged.update(_OVERRIDES.get(_key(applicant_id, stage), {}))
    return merged
