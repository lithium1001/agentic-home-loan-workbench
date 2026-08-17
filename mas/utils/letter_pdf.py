"""Formal bank-letter PDF renderer for the HITL draft gate.

The Document Drafting Agent produces a *markdown* draft letter (see
`skills/document_drafting/skill.md`). This module turns that text into a
professional, print-ready bank letter PDF: a letterhead, a reference line and
date, the applicant address block, a subject line, the drafted body, a signature
block, and a footer disclaimer.

Two renders of the same case:
  - draft=True   → a light "DRAFT" watermark + "DRAFT — pending RM review" footer.
  - draft=False  → the issued version: no watermark, an "Issued …· Ref …" footer.

Design decisions:
  - Pure renderer. The compliance-cleared figures and the recipient details are
    PASSED IN by the caller (the `draft_letter` tool) via `facts` / `recipient`;
    this module NEVER re-derives a loan number, so the Indicative Terms table can
    never disagree with the letter body (same rule the drafting agent follows).
  - No disk persistence — the bytes are returned to the caller, which holds them in
    `letter_store` and streams them on demand ("all in-memory, no DB" deploy).
  - `reportlab` is already a project dependency (see `documents/make_eval_pdf.py`).
  - Lives in utils/ (not server/) so `utils.tools.draft_letter` can import it
    without inverting the one-way utils → server dependency.

Public API:
    build_letter_pdf(applicant_id, stage, body, *, draft=True,
                     recipient=None, facts=None) -> (bytes, filename)
"""

from __future__ import annotations

import io
import os
import re
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── Bank identity. Deliberately a neutral placeholder: the issuing institution is
# anonymised, and every case in this system is synthetic data. BANK_REF is the
# prefix on reference numbers and letter filenames — keep it short and A-Z only,
# it lands in both a "Ref:" line and a filename. ────────────────────────────────
BANK_NAME = "The Bank"
BANK_ADDR = "Singapore"
BANK_META = "Synthetic data · demonstration only"
BANK_REF = "BNK"

# Names of the issuing institution that the model may reproduce from memory as a
# letterhead, comma-separated (e.g. "Some Bank Limited,SBL"). Deployment-specific
# and empty by default; see the letterhead alternative in _body_flowables().
BANK_ALIASES = [
    a.strip() for a in os.getenv("RM_COPILOT_BANK_ALIASES", "").split(",") if a.strip()
]
_ALIAS_ALT = "|".join(re.escape(a) for a in BANK_ALIASES)
NAVY = colors.HexColor("#003a70")   # deep navy letterhead accent
INK = colors.HexColor("#1a1a1a")
MUTE = colors.HexColor("#666666")

_STAGE_TITLE = {
    "IPA": "In-Principle Approval Letter",
    "LO": "Letter of Offer",
    "REPRICE": "Repricing Offer Letter",
}
_STAGE_REFTAG = {"IPA": "IPA", "LO": "LO", "REPRICE": "RP"}


# ── styles ───────────────────────────────────────────────────────────────────
_ss = getSampleStyleSheet()
S_BANK = ParagraphStyle("bank", parent=_ss["Normal"], fontName="Helvetica-Bold",
                        fontSize=13.5, textColor=NAVY, spaceAfter=1, leading=15)
S_BANKADDR = ParagraphStyle("bankaddr", parent=_ss["Normal"], fontSize=7.5,
                            textColor=MUTE, leading=9.5)
S_REF = ParagraphStyle("ref", parent=_ss["Normal"], fontSize=8.5, textColor=MUTE,
                       alignment=TA_RIGHT, leading=12)
S_ADDR = ParagraphStyle("addr", parent=_ss["Normal"], fontSize=9.5, leading=13,
                        textColor=INK)
S_SUBJECT = ParagraphStyle("subject", parent=_ss["Normal"], fontName="Helvetica-Bold",
                           fontSize=10.5, textColor=NAVY, spaceBefore=6, spaceAfter=6,
                           leading=14)
S_BODY = ParagraphStyle("body", parent=_ss["Normal"], fontSize=9.5, leading=14,
                        textColor=INK, spaceAfter=5)
S_H = ParagraphStyle("h", parent=_ss["Normal"], fontName="Helvetica-Bold",
                     fontSize=10, textColor=NAVY, spaceBefore=8, spaceAfter=3, leading=13)
S_LI = ParagraphStyle("li", parent=S_BODY, leftIndent=10, bulletIndent=0, spaceAfter=2)
S_TERM_K = ParagraphStyle("tk", parent=_ss["Normal"], fontSize=9.5, leading=12,
                          textColor=INK)
S_TERM_V = ParagraphStyle("tv", parent=_ss["Normal"], fontName="Helvetica-Bold",
                          fontSize=9.5, leading=12, textColor=INK, alignment=TA_RIGHT)
S_SIGN = ParagraphStyle("sign", parent=_ss["Normal"], fontSize=9.5, leading=14,
                        textColor=INK)
S_FOOT = ParagraphStyle("foot", parent=_ss["Normal"], fontSize=7.5, leading=10,
                        textColor=MUTE, alignment=TA_CENTER)
S_WM = ParagraphStyle("wm", parent=_ss["Normal"], fontSize=90,
                      textColor=colors.Color(0.85, 0.87, 0.92, alpha=0.35),
                      alignment=TA_CENTER)


# ── markdown → reportlab flowables ───────────────────────────────────────────
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITAL = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")


def _inline(text: str) -> str:
    """Minimal inline markdown → reportlab mini-HTML (bold/italic), escaped."""
    text = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITAL.sub(r"<i>\1</i>", text)
    return text


def _draft_body_flowables(draft_text: str) -> list:
    """Parse the drafting agent's markdown letter into body flowables.

    We drop the agent's own letterhead/date/footer lines (we render a proper
    letterhead + footer ourselves) and keep the substantive body: headings,
    bullet/numbered lists, and paragraphs. A pragmatic line-scanner — the draft is
    short and predictable, so this is enough without a full markdown engine."""
    out: list = []
    # Strip the agent's DRAFT footer marker and any leading/trailing whitespace.
    lines = draft_text.replace("\r\n", "\n").split("\n")
    para: list[str] = []

    def flush():
        if para:
            out.append(Paragraph(_inline(" ".join(para).strip()), S_BODY))
            para.clear()

    # Lines we skip because the PDF renders its own version of them. The prompt tells
    # the agent not to write a letterhead, date or subject line, but a prompt is not a
    # constraint — it has written e.g. "Date: 20 May 2026 Applicant: … Applicant ID: …"
    # right under the rendered RE: heading, restating (and contradicting) the header
    # block. Anything the header already carries is dropped here.
    skip_re = re.compile(
        r"^\s*(draft\s*[—-].*pending|—+\s*$|\*+\s*draft.*\*+|"
        # A letterhead the model wrote into the body. A specific institution's name
        # can match greedily (it only ever appears AS letterhead), but the neutral
        # BANK_NAME is ordinary English — "The Bank will assess your application…" is
        # real letter prose — so that form is stripped only when the line is the name
        # essentially alone. A deployment whose model has memorised its institution's
        # name from training data lists those forms in BANK_ALIASES to get the greedy
        # treatment; unset (the default here), only the neutral form is stripped.
        + (rf"({_ALIAS_ALT})\b.*$|" if _ALIAS_ALT else "") +
        r"the bank(\s+limited)?\s*[,.]?\s*(singapore.*)?$|"
        r"\**\s*(date|ref|reference|applicant(\s+id)?|nric(/fin)?|to|attn)\s*\**\s*:.*$|"
        r"\**\s*re\s*\**\s*:.*$)", re.I)

    # The agent's own WORKFLOW leaking into the letter. skill.md organises its
    # instructions as "**Step 1 — get the figures.** / **Step 2 — write the letter
    # body**", and the model has copied that heading straight onto bank letterhead
    # ("Step 2 — Write the IPA Letter Body", observed 2026-08-04). Same class as the
    # header-block leak above: the prompt's own structure becoming letter prose.
    step_re = re.compile(r"^\s*#*\s*\**\s*step\s+\d+\b.*$", re.I)
    # Narration ABOUT the letter, addressed to the system rather than the customer
    # ("The IPA letter body is ready for rendering into the official bank PDF.").
    meta_re = re.compile(
        r"^\s*\**\s*(the\s+)?(ipa|lo|letter|letter\s+body|body|draft)\b[^.]*\b"
        r"(is\s+)?(now\s+)?(ready|complete[d]?|prepared)\b.*$", re.I)

    for raw in lines:
        line = raw.rstrip()
        if skip_re.match(line) or step_re.match(line) or meta_re.match(line):
            continue
        h = re.match(r"^(#{1,4})\s+(.*)$", line)
        if h:
            flush()
            out.append(Paragraph(_inline(h.group(2)), S_H))
            continue
        li = re.match(r"^\s*(?:[-*]|\d+[.)])\s+(.*)$", line)
        if li:
            flush()
            out.append(Paragraph("• " + _inline(li.group(1)), S_LI))
            continue
        if line.strip() == "":
            flush()
            continue
        para.append(line)
    flush()
    return out


# ── key-terms table (echoes the case's cleared figures) ──────────────────────
def _fmt_money(n) -> str:
    try:
        return "S$" + format(round(float(n)), ",")
    except (TypeError, ValueError):
        return "—"


def _key_terms_table(facts: dict) -> Table | None:
    """The letter's key-terms block, rendered like a REAL bank letter: an inline
    definition list — label left, value right, a thin rule under each row, no
    shading and no boxed grid.

    Echoes ONLY the figures the drafting agent passed in (never re-derived), and
    ONLY the customer-facing ones: LTV is an internal risk metric and is omitted.
    None if no priced figures were supplied."""
    facts = facts or {}
    rate = facts.get("interest_rate_pct")
    rows_spec = [
        ("Property Value", _fmt_money(facts.get("property_price")) if facts.get("property_price") else None),
        ("Loan Amount", _fmt_money(facts.get("loan_amount")) if facts.get("loan_amount") else None),
        ("Loan Tenure", f"{facts.get('tenure_years')} years" if facts.get("tenure_years") else None),
        ("Interest Rate", f"{rate}% p.a." if rate else None),
        ("Monthly Instalment", _fmt_money(facts.get("monthly_repayment")) if facts.get("monthly_repayment") else None),
    ]
    rows = [(k, v) for k, v in rows_spec if v is not None]
    if not rows:
        return None
    data = [[Paragraph(k, S_TERM_K), Paragraph(v, S_TERM_V)] for k, v in rows]
    # Full text-width, right-aligned values; a hairline rule under every row.
    t = Table(data, colWidths=[70 * mm, 100 * mm])
    style = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d7dee8")),
    ]
    # A slightly stronger rule above the first row to open the block.
    style.append(("LINEABOVE", (0, 0), (-1, 0), 0.7, NAVY))
    t.setStyle(TableStyle(style))
    return t


# ── page furniture (letterhead band + footer + watermark) ────────────────────
def _make_page_decorator(draft: bool, ref: str):
    def decorate(canvas, doc):
        canvas.saveState()
        w, h = A4
        # Top navy rule under the compact letterhead band.
        canvas.setStrokeColor(NAVY)
        canvas.setLineWidth(1.2)
        canvas.line(20 * mm, h - 22 * mm, w - 20 * mm, h - 22 * mm)
        # DRAFT watermark, diagonal, low-opacity.
        if draft:
            canvas.saveState()
            canvas.translate(w / 2, h / 2)
            canvas.rotate(38)
            canvas.setFont("Helvetica-Bold", 96)
            canvas.setFillColor(colors.Color(0.82, 0.85, 0.90, alpha=0.30))
            canvas.drawCentredString(0, 0, "DRAFT")
            canvas.restoreState()
        # Footer: thin rule + disclaimer + status/ref line.
        canvas.setStrokeColor(colors.HexColor("#c8d4e4"))
        canvas.setLineWidth(0.5)
        canvas.line(20 * mm, 18 * mm, w - 20 * mm, 18 * mm)
        canvas.setFont("Helvetica", 6.8)
        canvas.setFillColor(MUTE)
        # Two short centred lines so the disclaimer never runs off the page edge.
        canvas.drawCentredString(w / 2, 15.6 * mm, f"{BANK_NAME} · {BANK_META}")
        canvas.drawCentredString(
            w / 2, 13.2 * mm,
            "Computer-generated. All figures are indicative and subject to the "
            "Bank's credit approval and prevailing MAS regulations.")
        status = ("DRAFT — pending RM review · not a binding offer"
                  if draft else f"Issued {date.today():%d %b %Y} · Ref {ref}")
        canvas.setFont("Helvetica-Bold", 7)
        canvas.setFillColor(NAVY if not draft else colors.HexColor("#b45309"))
        canvas.drawCentredString(w / 2, 11 * mm, status)
        # Page number, bottom-right.
        canvas.setFont("Helvetica", 6.8)
        canvas.setFillColor(MUTE)
        canvas.drawRightString(w - 20 * mm, 11 * mm, f"Page {doc.page}")
        canvas.restoreState()
    return decorate


def _ref_number(applicant_id: str, stage: str) -> str:
    tag = _STAGE_REFTAG.get(stage.upper(), stage.upper())
    return f"{BANK_REF}/{tag}/{applicant_id.upper()}/{date.today():%Y%m%d}"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", (name or "").strip()).strip("-") or "Applicant"


def build_letter_pdf(applicant_id: str, stage: str, draft_text: str,
                     *, draft: bool = True,
                     recipient: dict | None = None,
                     facts: dict | None = None) -> tuple[bytes, str]:
    """Render the drafted letter to a formal bank PDF.

    `recipient` = {name, nric, property_detail} for the address block.
    `facts` = the compliance-cleared figures the drafting agent passed
    (loan_amount / ltv_pct / tenure_years / interest_rate_pct / monthly_repayment /
    property_price). Both are supplied by the caller (the draft_letter tool) — this
    renderer NEVER re-derives numbers, so the Indicative Terms table always matches
    the letter body.

    Returns (pdf_bytes, filename). The filename carries '-DRAFT' before approval
    and drops it once released. No disk write — the bytes are returned to the
    caller to hold in memory and stream on request."""
    stage = stage.upper()
    recipient = recipient or {}
    facts = facts or {}
    applicant_name = recipient.get("name") or applicant_id.upper()
    borrower = {"nric": recipient.get("nric", "")}
    prop = {"detail": recipient.get("property_detail", "")}
    ref = _ref_number(applicant_id, stage)
    title = _STAGE_TITLE.get(stage, "Loan Letter")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=26 * mm, bottomMargin=22 * mm,
        title=f"{BANK_NAME} {title} — {applicant_name}",
        author=BANK_NAME,
    )

    story: list = []
    # Compact left-aligned letterhead (name + address), like a real bank letter.
    story.append(Paragraph(BANK_NAME, S_BANK))
    story.append(Paragraph(BANK_ADDR, S_BANKADDR))
    story.append(Spacer(1, 9))

    # Reference + date, right-aligned.
    story.append(Paragraph(f"Ref: {ref}<br/>Date: {date.today():%d %B %Y}", S_REF))
    story.append(Spacer(1, 6))

    # Applicant address block. This is the recipient's own details, so it carries no
    # internal case number — the applicant id already appears in the Ref: line above.
    addr_lines = [f"<b>{applicant_name}</b>"]
    if borrower.get("nric"):
        addr_lines.append(f"NRIC/FIN: {borrower['nric']}")
    if prop.get("detail"):
        addr_lines.append(prop["detail"])
    story.append(Paragraph("<br/>".join(addr_lines), S_ADDR))
    story.append(Spacer(1, 10))

    # Salutation + subject.
    story.append(Paragraph(f"Dear {applicant_name},", S_BODY))
    story.append(Paragraph(f"RE: {title.upper()}", S_SUBJECT))
    story.append(HRFlowable(width="100%", thickness=0.6, color=NAVY,
                            spaceBefore=2, spaceAfter=8))

    # Drafted body.
    story.extend(_draft_body_flowables(draft_text))

    # Key-terms block (echoes ONLY the figures the agent passed in). An IPA is
    # indicative; an LO states the binding facility details.
    terms = _key_terms_table(facts)
    if terms is not None:
        terms_heading = "Facility Details" if stage == "LO" else "Indicative Terms"
        story.append(Spacer(1, 8))
        story.append(Paragraph(terms_heading, S_H))
        story.append(Spacer(1, 2))
        story.append(terms)

    # Signature block.
    story.append(Spacer(1, 18))
    story.append(Paragraph("Yours sincerely,", S_SIGN))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<b>Home Loans, Personal Financial Services</b><br/>"
        f"{BANK_NAME}", S_SIGN))

    doc.build(story, onFirstPage=_make_page_decorator(draft, ref),
              onLaterPages=_make_page_decorator(draft, ref))

    pdf = buf.getvalue()
    buf.close()

    tag = _STAGE_REFTAG.get(stage, stage)
    base = f"{BANK_REF}-{tag}-Letter_{applicant_id.upper()}_{_safe_name(applicant_name)}"
    filename = base + ("-DRAFT.pdf" if draft else ".pdf")
    return pdf, filename