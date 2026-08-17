"""Guards the letter body against the agent's own workflow reaching letterhead.

Observed 2026-08-04 in a rendered IPA PDF: the body opened with the heading
"Step 2 — Write the IPA Letter Body" and closed with "The IPA letter body is ready
for rendering into the official bank PDF." Neither is letter prose — the first is a
heading copied out of skills/document_drafting/skill.md, which organises its
instructions as "**Step 1 — get the figures.** / **Step 2 — write the letter
body**"; the second is narration addressed to the system.

Same class as the header-block leak the renderer already strips, and the same
lesson recorded there: a prompt is not a constraint. skill.md now forbids both, but
the renderer is the only layer that can guarantee it, so the guard is tested here.
No LLM / network — `_draft_body_flowables` is a pure text scanner.
"""

import re

from utils.letter_pdf import _draft_body_flowables


def _text(body: str) -> str:
    """Flatten rendered flowables back to plain text for assertions."""
    out = []
    for f in _draft_body_flowables(body):
        raw = getattr(f, "text", "") or ""
        out.append(re.sub(r"<[^>]+>", "", raw))     # drop the inline markup tags
    return "\n".join(out)


# The verbatim shape from the rendered PDF.
_LEAKED_BODY = """Step 2 — Write the IPA Letter Body

We are pleased to inform you that the Bank is willing to grant you an In-Principle
Approval for a home loan of $1,224,743 for the purchase of the private residential
property at Canninghill Piers, valued at $1,792,000.

This In-Principle Approval is valid for 30 days from the date of this letter.

The IPA letter body is ready for rendering into the official bank PDF."""


def test_workflow_step_heading_is_not_printed():
    out = _text(_LEAKED_BODY)
    assert "Step 2" not in out
    assert "Write the IPA Letter Body" not in out


def test_progress_narration_is_not_printed():
    out = _text(_LEAKED_BODY)
    assert "ready for rendering" not in out
    assert "official bank PDF" not in out


def test_the_actual_letter_survives_intact():
    """The guard must remove the debris and nothing else — the offer, the figures
    and the validity clause are the letter."""
    out = _text(_LEAKED_BODY)
    assert "We are pleased to inform you" in out
    assert "$1,224,743" in out
    assert "$1,792,000" in out
    assert "Canninghill Piers" in out
    assert "valid for 30 days" in out


def test_step_heading_variants_are_all_dropped():
    for line in ("Step 1 — get the figures",
                 "**Step 2 — Write the IPA Letter Body**",
                 "## Step 3: sign off",
                 "step 2 - write the letter body"):
        assert "Step" not in _text(line + "\n\nDear Customer,"), line


def test_narration_variants_are_all_dropped():
    for line in ("The IPA letter body is ready for rendering into the official bank PDF.",
                 "The letter body is now complete.",
                 "**The draft is ready.**",
                 "The LO letter body is prepared for the PDF renderer."):
        assert "ready" not in _text(line).lower(), line
        assert "complete" not in _text(line).lower(), line


def test_ordinary_letter_prose_is_untouched():
    """Words like 'ready' and 'step' appear in legitimate letter prose. The guards
    are anchored to line starts and specific shapes so they cannot eat sentences."""
    body = ("We are pleased to inform you that your application is approved.\n\n"
            "Once you are ready to proceed, contact your Relationship Manager.\n\n"
            "The next step is to convert this approval to a formal application.\n\n"
            "Your funds will be ready for disbursement upon completion.")
    out = _text(body)
    for sentence in ("Once you are ready to proceed",
                     "The next step is to convert",
                     "ready for disbursement"):
        assert sentence in out, sentence


def test_next_steps_heading_still_allowed():
    """skill.md explicitly permits section headings inside the body (e.g. 'Next
    Steps'); only the instruction-brief's own step headings are forbidden."""
    out = _text("## Next Steps\n\nPlease contact your Relationship Manager.")
    assert "Next Steps" in out
