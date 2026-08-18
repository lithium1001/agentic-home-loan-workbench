"""Usage tracking — who reached the tool, and what they asked it.

Two append-only CSVs under ``usage_data/`` at the repo root:

  * ``visits.csv``          — one row per page view, keyed by a cookie ``cid``.
  * ``customer_inputs.csv`` — one row per "Show my results" click on /customer,
                              carrying the figures the customer typed in.

Both rows carry the SAME ``cid``, which is the whole point of the design: joining
the two files answers "of the people who opened the tool, how many actually ran a
calculation" — a conversion rate, not just two disconnected counts.

Separate from ``artifacts/runs.csv`` on purpose. That file is operational KPIs of
the agent graph (latency, tokens, routing). This is adoption data about *people*,
so it lives in its own folder — and it is gitignored too: ``visits.csv`` records
client IPs, which must never reach a shared repo. Collect it, analyse it locally,
and export only aggregates.

WHAT ``cid`` ACTUALLY MEASURES
------------------------------
A browser profile, not a person. The same human on a laptop and a phone counts
twice; clearing cookies or using a private window mints a new id. That is a
property of browsers, not a defect here, and the honest phrasing downstream is
"distinct browser sessions" — never "unique users". ``ua_ip_hash`` is recorded
alongside as a second, differently-biased estimate: it survives a cookie wipe but
merges distinct people behind one NAT. Cookie count is the lower bound, hash count
the upper; reporting both is more defensible than picking one.

FAILURE POLICY
--------------
Instrumentation must never break the page. Every write is wrapped, and a field
that could not be read is written as "" — never guessed, never defaulted. An empty
string means "not captured on this request", which is not the same as any real
value and must not be silently turned into one.
"""

from __future__ import annotations

import csv
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from utils.telemetry.metrics import _migrate_header

# <repo>/usage_data/. This file is mas/utils/telemetry/usage.py, so the repo root
# is 4 levels up — same arithmetic as metrics.RUNS_CSV, deliberately kept parallel.
USAGE_DIR = Path(__file__).resolve().parents[3] / "usage_data"
VISITS_CSV = USAGE_DIR / "visits.csv"
INPUTS_CSV = USAGE_DIR / "customer_inputs.csv"

COOKIE_NAME = "cid"
# Two years. The id is an opaque random token with no meaning outside these files;
# a long life is what makes "returning visitor" measurable at all.
COOKIE_MAX_AGE = 63_072_000

VISIT_FIELDS = [
    "ts",           # ISO-8601 UTC
    "cid",          # cookie id — the visitor key both files join on
    "ua_ip_hash",   # sha1(ip + user_agent)[:12] — cookie-independent cross-check
    "role",         # landing | rm | customer — which audience's page was opened
    "path",
    "is_new",       # 1 if this request minted the cookie (first-ever page view)
    "ip",
    "user_agent",
]

INPUT_FIELDS = [
    "ts",
    "cid",
    "mode",             # price | downpayment | instalment | explore
    "ok",               # 1 if the calculator returned figures, 0 if it returned
                        # an error. Failed attempts are still usage: dropping them
                        # would undercount demand and hide which inputs break.
    "error",            # the error text when ok=0, else ""
    "age",
    "monthly_fixed_income",
    "monthly_variable_income",
    "nationality",
    "property_type",
    "properties_owned_now",
    "outstanding_home_loans",
    "monthly_car_loan",
    "monthly_other",
    "interest_rate_pct",
    "target_property_price",
    "cash_cpf_available",
    "monthly_budget",
    "financial_assets",  # JSON blob — a nested {class: {amount, pledged}} dict has
                         # no fixed column shape, and one text cell keeps the file
                         # openable in Excel.
    # Headline outputs, so the file answers "what did they get told" without
    # re-running the calculator against a possibly-changed code version.
    "eligible_loan",
    "monthly_repayment",
]

# Only real page views. Without this filter every visit also logs the .js, the
# .css and each /api/* poll, and the file stops being a count of anything.
_TRACKED_PATHS = {
    "/":              "landing",
    "/rm":            "rm",
    "/index.html":    "rm",
    "/customer":      "customer",
    "/customer.html": "customer",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_cid() -> str:
    return uuid.uuid4().hex[:16]


def role_for(path: str) -> str:
    """Which audience a path belongs to, or "" if it is not a tracked page view."""
    return _TRACKED_PATHS.get(path, "")


def _record_ip(request) -> str:
    """Read the client IP, honouring a reverse proxy.

    Behind a proxy ``request.client.host`` is the PROXY's address, identical for
    every visitor, which would silently collapse all traffic onto one apparent
    machine. X-Forwarded-For's first entry is the original client.

    Isolated in its own function so that if the IP ever has to be hashed or
    dropped for privacy, exactly one line changes rather than the middleware.
    """
    try:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
        return getattr(request.client, "host", "") or ""
    except Exception:  # noqa: BLE001 — never let telemetry break the request
        return ""


def _ua_ip_hash(ip: str, ua: str) -> str:
    """Cookie-independent visitor estimate.

    Salted with nothing on purpose: this is a local, committed, private-repo file
    and the raw ``ip`` sits in the next column anyway, so hashing here buys
    correlation convenience, not secrecy. Do not present it as anonymisation.
    """
    if not ip and not ua:
        return ""
    return hashlib.sha1(f"{ip}|{ua}".encode("utf-8", "replace")).hexdigest()[:12]


def _append(path: Path, fieldnames: list, row: dict) -> None:
    """Append one row, writing the header if the file is new.

    Best-effort by contract: a usage write must never break a page load, so any
    I/O error is printed, not raised.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists() or path.stat().st_size == 0
        if not new_file:
            _migrate_header(path, fieldnames)
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if new_file:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception as e:  # noqa: BLE001 — monitoring must not break the page
        print(f"  [usage] failed to write {path.name}: {e}")


def record_visit(request, cid: str, *, is_new: bool, path: Path | str | None = None) -> None:
    """Log one page view. Called by the middleware; never raises.

    `path` resolves at CALL time, not as a default argument: a default would bind
    the module constant at import and make the destination impossible to redirect
    afterwards (which is also what lets the tests write to tmp).
    """
    try:
        path = path or VISITS_CSV
        url_path = request.url.path
        ip = _record_ip(request)
        ua = request.headers.get("user-agent", "") or ""
        _append(Path(path), VISIT_FIELDS, {
            "ts": _now(),
            "cid": cid,
            "ua_ip_hash": _ua_ip_hash(ip, ua),
            "role": role_for(url_path),
            "path": url_path,
            "is_new": int(is_new),
            "ip": ip,
            "user_agent": ua,
        })
    except Exception as e:  # noqa: BLE001
        print(f"  [usage] visit not recorded: {e}")


def record_customer_input(
    req,
    result: dict | None,
    cid: str,
    *,
    path: Path | str | None = None,
) -> None:
    """Log one "Show my results" click with the figures the customer entered.

    ``result`` is the calculator's return value; a dict carrying "error" means the
    attempt failed, which is recorded as ok=0 rather than dropped. ``path``
    resolves at call time — see record_visit.
    """
    try:
        path = path or INPUTS_CSV
        data = req.model_dump() if hasattr(req, "model_dump") else dict(req)
        result = result or {}
        err = result.get("error") or ""

        row = {
            "ts": _now(),
            "cid": cid or "",
            "ok": int(not err),
            "error": err,
            # Absent optional fields stay "" rather than becoming 0 — the customer
            # leaving a box empty is not the same as typing zero into it.
            **{k: ("" if data.get(k) is None else data.get(k))
               for k in INPUT_FIELDS if k in data},
            "eligible_loan": result.get("eligible_loan", ""),
            "monthly_repayment": result.get("monthly_repayment", ""),
        }
        assets = data.get("financial_assets")
        row["financial_assets"] = json.dumps(assets, ensure_ascii=False) if assets else ""
        _append(Path(path), INPUT_FIELDS, row)
    except Exception as e:  # noqa: BLE001
        print(f"  [usage] customer input not recorded: {e}")
