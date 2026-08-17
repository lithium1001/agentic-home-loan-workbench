"""An unreachable network must degrade silently, not stall and not stop the run.

Three features reach the public internet — the OpenRouter price lookup, the MAS
SORA API, and the competitor rate scrape — and on the intranet deployment none
of them can succeed. The requirement is that the user never has to know: no flag
to set, no stall, and above all no abort of the case workflow, which needs none
of the three.

The design is a per-process circuit breaker (utils.netguard) rather than a config
flag, so these tests pin down the two halves of that bargain:

* it OPENS on evidence the network has no path (DNS failure, refused, no route),
  and every later optional call then short-circuits instantly, and
* it does NOT open on a timeout or an HTTP error, which say nothing about
  reachability — otherwise one slow response on a healthy host would
  permanently disable live rates.

Network is asserted-against by monkeypatching the transport to explode: if a
guard regresses, the test fails loudly instead of quietly reaching the internet
on whichever machine happens to run CI.
"""
import socket

import pytest

from utils import netguard, rates, rates_scraper
from utils.telemetry import pricing


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Breaker and price cache are process-global; each test needs a clean slate."""
    netguard.reset()
    monkeypatch.setattr(pricing, "_PRICE_TABLE", None)
    monkeypatch.setattr(rates, "_cache", None)
    yield
    netguard.reset()


@pytest.fixture
def no_network(monkeypatch):
    """Make any real outbound call an immediate, obvious failure."""
    def boom(*a, **k):                      # noqa: ANN002, ANN003
        raise AssertionError("a network call escaped the offline guard")

    monkeypatch.setattr(pricing.urllib.request, "urlopen", boom)
    monkeypatch.setattr(rates.urllib3, "PoolManager", boom)
    monkeypatch.setattr(rates_scraper.requests, "get", boom)


def _dns_error() -> Exception:
    """The exact shape seen on the intranet host: urllib wrapping a gaierror."""
    inner = socket.gaierror(-2, "Name or service not known")
    outer = OSError("<urlopen error [Errno -2] Name or service not known>")
    outer.__cause__ = inner
    return outer


# ── what opens the breaker, and what must not ────────────────────────────
def test_dns_failure_opens_the_breaker():
    assert netguard.note_failure(_dns_error()) is True
    assert netguard.is_offline() is True


def test_connection_refused_opens_the_breaker():
    assert netguard.note_failure(ConnectionRefusedError("refused")) is True


def test_detects_the_error_through_a_wrapper_chain():
    """requests nests the real cause several levels deep."""
    outer = RuntimeError("HTTPSConnectionPool: Max retries exceeded")
    outer.__cause__ = socket.gaierror(-2, "Name or service not known")
    assert netguard.looks_unreachable(outer) is True


def test_timeout_does_not_open_the_breaker():
    """A slow endpoint is not an unreachable network — live rates must survive it."""
    assert netguard.note_failure(TimeoutError("timed out")) is False
    assert netguard.is_offline() is False


def test_http_error_does_not_open_the_breaker():
    """The host answered, just badly. Says nothing about reachability."""
    assert netguard.note_failure(RuntimeError("MAS API returned HTTP 503")) is False
    assert netguard.is_offline() is False


# ── once open, nothing else even tries ───────────────────────────────────
def test_all_three_features_skip_instantly_once_offline(no_network):
    netguard.note_failure(_dns_error())

    assert pricing._pricing_enabled() is False
    assert pricing.estimate("meta-llama/llama-4-maverick", 100, 100) is None

    with pytest.raises(rates.SoraUnavailable, match="no public internet"):
        rates.get_sora_3m(refresh=True)

    with pytest.raises(RuntimeError, match="no public internet"):
        rates_scraper.fetch(rates_scraper.URL)


def test_first_real_failure_trips_the_breaker_for_the_others(monkeypatch):
    """The point of sharing one breaker: only ONE feature pays the timeout."""
    monkeypatch.setattr(pricing.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(_dns_error()))
    from utils import config
    monkeypatch.setattr(config, "USE_INTERNAL_LLM", False)

    assert pricing.estimate("meta-llama/llama-4-maverick", 100, 100) is None
    assert netguard.is_offline() is True      # SORA/scrape now skip without trying


# ── the internal-gateway route never asks OpenRouter for a price ─────────
def test_pricing_skipped_on_internal_gateway(monkeypatch, no_network):
    from utils import config
    monkeypatch.setattr(config, "USE_INTERNAL_LLM", True)
    assert pricing.estimate("default", 100, 100) is None


def test_pricing_reads_config_live_not_frozen(monkeypatch, no_network):
    """Guards the from-import bug: config.USE_INTERNAL_LLM is rebound by
    _refresh(), so pricing must read it through the module every time."""
    from utils import config
    monkeypatch.setattr(config, "USE_INTERNAL_LLM", True)
    assert pricing._pricing_enabled() is False
    monkeypatch.setattr(config, "USE_INTERNAL_LLM", False)
    assert pricing._pricing_enabled() is True


# ── graceful degradation (the part that must never regress) ──────────────
def test_collect_reports_both_failures_without_raising(no_network, tmp_path):
    """One outage must not discard the other source, and neither may raise."""
    netguard.note_failure(_dns_error())
    summary = rates_scraper.collect(out=tmp_path / "market_rates.csv")
    assert summary["bank_rows"] == 0
    assert summary["sora_rows"] == 0
    assert len(summary["errors"]) == 2          # scrape + SORA, reported separately


def test_sora_tool_returns_payload_not_exception(no_network):
    """The agent-facing tool degrades to an explicit 'unavailable' payload, so a
    dead rate feed cannot abort a case run."""
    from utils.tools import _sora_rate_payload

    netguard.note_failure(_dns_error())
    payload = _sora_rate_payload()
    assert payload["sora_3m_pct"] is None
    assert payload["source"] == "unavailable"
    assert "do not quote" in payload["note"].lower()


def test_pricing_survives_a_network_error(monkeypatch):
    """Cost goes blank on failure — estimate() must never raise."""
    from utils import config
    monkeypatch.setattr(config, "USE_INTERNAL_LLM", False)
    monkeypatch.setattr(pricing.urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(_dns_error()))
    assert pricing.estimate("meta-llama/llama-4-maverick", 100, 100) is None


# ── operator-facing error text ───────────────────────────────────────────
# The screenshot that prompted this work showed a bare "InternalServerError:
# Internal Server Error" sitting directly above a harmless OpenRouter pricing
# warning. Nothing on screen named the endpoint that actually failed, so the
# pricing line looked like the cause. These assert the message carries enough
# to tell the two apart.
def test_error_text_names_the_endpoint_and_blames_the_gateway(monkeypatch):
    import httpx
    import openai

    from server.stream import _explain_error
    from utils import config

    monkeypatch.setattr(config, "BASE_URL", "https://gw.internal/v1")
    req = httpx.Request("POST", "https://gw.internal/v1/chat/completions")
    exc = openai.InternalServerError(
        "Internal Server Error", response=httpx.Response(500, request=req), body=None)

    text = _explain_error(exc)
    assert "https://gw.internal/v1" in text          # which endpoint failed
    assert "500" in text
    assert "not the case data" in text               # don't debug the wrong thing


def test_error_text_distinguishes_a_bad_credential(monkeypatch):
    import httpx
    import openai

    from server.stream import _explain_error
    from utils import config

    monkeypatch.setattr(config, "BASE_URL", "https://gw.internal/v1")
    req = httpx.Request("POST", "https://gw.internal/v1/chat/completions")
    exc = openai.AuthenticationError(
        "Unauthorized", response=httpx.Response(401, request=req), body=None)

    assert "rejected the credential" in _explain_error(exc)


def test_error_text_never_raises_on_a_plain_exception():
    """Reporting a failure must not become a second failure."""
    from server.stream import _explain_error

    assert "boom" in _explain_error(ValueError("boom"))


def test_tool_round_exhaustion_reads_as_a_stopped_turn(monkeypatch):
    """Tool-round exhaustion is not an endpoint failure and must not read like one.

    Observed 2026-08-04: the Reprice agent made eleven identical
    interest_savings_tool calls without converging. LangGraph then raises
    GraphRecursionError, which fell through to the generic branch and put a raw
    exception name in front of the RM — who can neither diagnose nor act on it.
    """
    from server.stream import _explain_error
    from utils import config

    monkeypatch.setattr(config, "BASE_URL", "https://gw.internal/v1")

    class GraphRecursionError(Exception):
        pass

    text = _explain_error(GraphRecursionError(
        "Recursion limit of 64 reached without hitting a stop condition."))

    assert "kept calling tools" in text
    assert "Nothing was saved to the case" in text
    # The RM must not be sent debugging the gateway for an agent-loop problem.
    assert "gw.internal" not in text
    assert "GraphRecursionError" not in text


# ── Error attribution (2026-08-04) ───────────────────────────────────────
# A KeyError raised inside the calculator was reported to the RM as
# "KeyError: 'nationality' — while calling the LLM endpoint (https://openrouter.ai/
# api/v1)". Nothing about it touched the network, and the label sent the diagnosis
# at the provider instead of at the incomplete tool-call argument that caused it.
def test_local_errors_are_not_blamed_on_the_llm_endpoint():
    from server.stream import _explain_error

    for exc in (KeyError("nationality"), AttributeError("x"), IndexError("y"),
                TypeError("z"), ValueError("w")):
        msg = _explain_error(exc)
        assert "internal error in the assistant" in msg
        assert "while calling the LLM endpoint" not in msg


def test_provider_shaped_failures_still_name_the_endpoint():
    """The catch-all must keep working for anything that really is the provider."""
    from server.stream import _explain_error

    assert "LLM endpoint" in _explain_error(Exception("upstream returned nonsense"))
