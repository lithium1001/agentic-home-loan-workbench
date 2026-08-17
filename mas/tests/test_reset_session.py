"""Clearing a lane's conversation — the escape hatch from an overgrown transcript.

`stream.SESSIONS[key]["messages"]` only ever GROWS: every turn appends its new
messages, and `stream_chat` replays the whole accumulation into the next turn's
state. On a small-context endpoint (the internal gateway's 32768-token window)
one full assessment turn is enough to fill it, after which every later message
is rejected before the RM can say anything — with no way back except restarting
the process. These tests pin the way back.

The scope is deliberate: one (applicant, stage) lane, conversation only. A wipe
that also dropped case progress / overrides / rendered letters would turn a
recovery action into data loss, and an RM reaching for "Clear" because the chat
is stuck has no reason to expect their case to be reset with it.
"""
import pytest

from server import stream


@pytest.fixture(autouse=True)
def clean_sessions():
    """SESSIONS is module-global and process-lived; isolate each test from it."""
    saved = dict(stream.SESSIONS)
    stream.SESSIONS.clear()
    yield
    stream.SESSIONS.clear()
    stream.SESSIONS.update(saved)


def test_reset_drops_the_accumulated_history():
    sess = stream._session("APP0007", "LO")
    sess["messages"] = ["m1", "m2", "m3"]

    out = stream.reset_session("APP0007", "LO")

    assert out["cleared"] is True
    assert out["messages_dropped"] == 3
    # The next turn must start from nothing, not from the old transcript.
    assert stream._session("APP0007", "LO")["messages"] == []


def test_reset_is_scoped_to_one_lane():
    """The RM's other cases (and the same case's other stage) are untouched —
    this is a recovery action, not a global wipe."""
    stream._session("APP0007", "LO")["messages"] = ["lo-1", "lo-2"]
    stream._session("APP0007", "IPA")["messages"] = ["ipa-1"]
    stream._session("APP0009", "LO")["messages"] = ["other-1"]

    stream.reset_session("APP0007", "LO")

    assert stream._session("APP0007", "IPA")["messages"] == ["ipa-1"]
    assert stream._session("APP0009", "LO")["messages"] == ["other-1"]


def test_reset_abandons_a_pending_hitl_gate():
    """A half-finished gate belongs to the discarded turn. Leaving thread_id set
    would let a later Approve try to resume an interrupt whose conversation is
    gone; the report lets the UI know a gate was dropped."""
    sess = stream._session("APP0007", "LO")
    sess["messages"] = ["m1"]
    sess["pending"] = True
    sess["thread_id"] = "deadbeef"

    out = stream.reset_session("APP0007", "LO")

    assert out["had_pending_gate"] is True
    fresh = stream._session("APP0007", "LO")
    assert fresh["pending"] is False
    assert fresh["thread_id"] is None


def test_reset_of_an_unknown_lane_is_harmless():
    """The UI may fire this on a lane that never ran a turn; it must not raise."""
    out = stream.reset_session("APP9999", "IPA")
    assert out == {"cleared": True, "messages_dropped": 0, "had_pending_gate": False}


def test_reset_endpoint_returns_the_report():
    """Wired as plain JSON, not SSE — nothing about clearing a list streams."""
    from fastapi.testclient import TestClient

    from server.app import app

    stream._session("APP0007", "LO")["messages"] = ["m1", "m2"]
    resp = TestClient(app).post(
        "/api/reset", json={"applicant_id": "APP0007", "stage": "LO"})

    assert resp.status_code == 200
    assert resp.json()["messages_dropped"] == 2
    assert stream._session("APP0007", "LO")["messages"] == []
