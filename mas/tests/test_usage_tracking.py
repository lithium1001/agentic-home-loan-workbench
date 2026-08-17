"""Usage tracking — visitor cookie + customer-input logging.

Guards three properties that are easy to break silently:

  1. The `cid` cookie is minted once and then STAYS. A regression that re-mints it
     per request would inflate the visitor count without any visible symptom.
  2. Only real page views are logged. Without the path filter, every visit also
     logs assets and API polls, and the row count stops meaning anything.
  3. Logging never breaks the page. Instrumentation that can 500 a request is
     worse than no instrumentation.
"""

from __future__ import annotations

import csv
import pathlib

import pytest
from fastapi.testclient import TestClient

from server.app import app
from utils.telemetry import usage


@pytest.fixture
def logs(tmp_path, monkeypatch):
    """Redirect both CSVs into tmp so tests never touch the committed usage_data/."""
    visits = tmp_path / "visits.csv"
    inputs = tmp_path / "customer_inputs.csv"
    monkeypatch.setattr(usage, "VISITS_CSV", visits)
    monkeypatch.setattr(usage, "INPUTS_CSV", inputs)
    return visits, inputs


def _rows(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


CALC = {
    "mode": "price",
    "age": 35,
    "monthly_fixed_income": 9000,
    "monthly_variable_income": 2000,
    "property_type": "Private",
    "properties_owned_now": 0,
    "interest_rate_pct": 3.5,
    "target_property_price": 1_500_000,
}


# ── 1. visitor identity ────────────────────────────────────────────────────
def test_cookie_is_minted_once_and_reused(logs):
    visits, _ = logs
    c = TestClient(app)

    c.get("/")
    cid = c.cookies.get("cid")
    assert cid, "first page view must set a cid cookie"

    c.get("/customer")
    c.get("/rm")
    assert c.cookies.get("cid") == cid, "cid must survive later requests"

    rows = _rows(visits)
    assert [r["cid"] for r in rows] == [cid] * 3
    # Only the first request minted the cookie — this is what distinguishes a
    # new visitor from a returning one in the data.
    assert [r["is_new"] for r in rows] == ["1", "0", "0"]
    assert [r["role"] for r in rows] == ["landing", "customer", "rm"]


def test_separate_browsers_are_separate_visitors(logs):
    visits, _ = logs
    TestClient(app).get("/customer")
    TestClient(app).get("/customer")
    assert len({r["cid"] for r in _rows(visits)}) == 2


def test_assets_and_api_calls_are_not_logged_as_visits(logs):
    visits, _ = logs
    c = TestClient(app)
    c.get("/")
    c.get("/styles.css")
    c.get("/app.js")
    c.get("/api/cases")
    assert len(_rows(visits)) == 1, "only the page view counts"


# ── 2. customer inputs ─────────────────────────────────────────────────────
def test_calculation_logs_inputs_and_headline_outputs(logs):
    _, inputs = logs
    c = TestClient(app)
    c.get("/customer")
    body = c.post("/api/customer/calc", json=CALC).json()

    row = _rows(inputs)[0]
    assert row["cid"] == c.cookies.get("cid"), "must join to the visit row"
    assert row["ok"] == "1"
    assert row["mode"] == "price"
    assert float(row["monthly_fixed_income"]) == 9000
    assert float(row["eligible_loan"]) == body["eligible_loan"]


def test_failed_calculation_is_still_recorded(logs):
    """A validation error is real usage: the person tried. Dropping these would
    undercount demand and hide which inputs people get stuck on."""
    _, inputs = logs
    c = TestClient(app)
    c.post("/api/customer/calc", json={**CALC, "target_property_price": None})

    row = _rows(inputs)[0]
    assert row["ok"] == "0"
    assert row["error"]
    assert row["eligible_loan"] == ""


def test_absent_optional_field_stays_blank_not_zero(logs):
    """Leaving a box empty is not the same as typing 0 into it, and writing 0
    would silently invent data the customer never entered."""
    _, inputs = logs
    TestClient(app).post("/api/customer/calc", json=CALC)
    assert _rows(inputs)[0]["monthly_budget"] == ""


# ── 3. failure policy ──────────────────────────────────────────────────────
def test_unwritable_log_does_not_break_the_page(monkeypatch):
    monkeypatch.setattr(usage, "VISITS_CSV", pathlib.Path("Z:/nope/visits.csv"))
    monkeypatch.setattr(usage, "INPUTS_CSV", pathlib.Path("Z:/nope/inputs.csv"))
    c = TestClient(app)
    assert c.get("/customer").status_code == 200
    r = c.post("/api/customer/calc", json=CALC)
    assert r.status_code == 200 and "eligible_loan" in r.json()


def test_ip_prefers_forwarded_header():
    """Behind a proxy request.client.host is the PROXY, identical for everyone —
    which would collapse all traffic onto one apparent machine."""
    class Req:
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}
        class client:
            host = "10.0.0.1"

    assert usage._record_ip(Req()) == "203.0.113.7"


def test_unreadable_request_degrades_to_blank():
    assert usage._record_ip(object()) == ""
    assert usage._ua_ip_hash("", "") == ""
