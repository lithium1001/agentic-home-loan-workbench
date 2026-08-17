"""Endpoint selection: OpenRouter by default, internal gateway when configured.

Guards the one-switch contract in config._resolve_endpoint(): setting
INTERNAL_LLM_BASE_URL must flip the model / key / TLS defaults with it, and
leaving it unset must reproduce the original OpenRouter behaviour exactly. Both
matter — a silent regression here points the whole app at the wrong backend,
which looks like a bad model rather than a misconfiguration.

_resolve_endpoint() is called directly with a patched _secret so the developer's
own .env (which pins RM_COPILOT_MODEL) cannot mask a broken default.
"""
import pytest

from utils import config


@pytest.fixture
def env(monkeypatch):
    """A _secret() that reads only from the dict this returns, then the default."""
    values: dict[str, str] = {}
    monkeypatch.setattr(config, "_secret",
                        lambda name, default="": values.get(name) or default)
    return values


def test_defaults_to_openrouter(env):
    """No INTERNAL_LLM_BASE_URL: unchanged from before the internal-gateway support."""
    env["OPENROUTER_API_KEY"] = "sk-or-test"
    base_url, model, api_key, verify = config._resolve_endpoint()
    assert base_url == "https://openrouter.ai/api/v1"
    assert model == "google/gemma-4-31b-it"
    assert api_key == "sk-or-test"
    assert verify is True            # public CA — verification stays ON


def test_internal_gateway_flips_dependent_defaults(env):
    """One env var switches endpoint, model naming and TLS together."""
    env["INTERNAL_LLM_BASE_URL"] = "https://gw.internal/v1"
    env["INTERNAL_LLM_API_KEY"] = "internal-key"
    base_url, model, api_key, verify = config._resolve_endpoint()
    assert base_url == "https://gw.internal/v1"
    assert model == "default"        # single-model gateway, not a route key
    assert api_key == "internal-key"
    assert verify is False           # self-signed / private CA expected


def test_openrouter_key_never_sent_to_internal_gateway(env):
    """The keys must not substitute for each other, in either direction.

    A fallback would ship the OpenRouter credential to whatever host
    INTERNAL_LLM_BASE_URL names — a third-party credential leak caused by a typo.
    """
    env["INTERNAL_LLM_BASE_URL"] = "https://gw.internal/v1"
    env["OPENROUTER_API_KEY"] = "sk-or-secret"
    assert config._resolve_endpoint()[2] == ""      # empty, NOT the OpenRouter key

    # ...and the internal key is never sent to OpenRouter either.
    env.pop("INTERNAL_LLM_BASE_URL")
    env["INTERNAL_LLM_API_KEY"] = "internal-key"
    assert config._resolve_endpoint()[2] == "sk-or-secret"


def test_both_routes_configured_at_once(env):
    """The whole point of per-route variables: fill in both, switch with the URL.

    Nothing is shared, so a value left set for the inactive route must never leak
    into the active one — otherwise switching would mean editing four lines
    instead of one, which is what this design exists to avoid.
    """
    env.update({
        "OPENROUTER_API_KEY": "sk-or-secret",
        "RM_COPILOT_MODEL": "google/gemma-4-31b-it",
        "INTERNAL_LLM_API_KEY": "internal-key",
        "INTERNAL_LLM_MODEL": "llama-4-70b",
    })

    # URL unset -> Route A, untouched by the Route B values sitting right there.
    base_url, model, api_key, _ = config._resolve_endpoint()
    assert (base_url, model, api_key) == ("https://openrouter.ai/api/v1",
                                          "google/gemma-4-31b-it", "sk-or-secret")

    # Setting the one switch flips model AND key together — no other edits.
    env["INTERNAL_LLM_BASE_URL"] = "https://gw.internal/v1"
    base_url, model, api_key, _ = config._resolve_endpoint()
    assert (base_url, model, api_key) == ("https://gw.internal/v1",
                                          "llama-4-70b", "internal-key")


def test_model_vars_do_not_cross_routes(env):
    """RM_COPILOT_MODEL is Route A's; INTERNAL_LLM_MODEL is Route B's.

    Sharing one variable meant an OpenRouter "vendor/model" id left in .env was
    silently sent to a gateway that names its model differently.
    """
    env["INTERNAL_LLM_BASE_URL"] = "https://gw.internal/v1"
    env["INTERNAL_LLM_API_KEY"] = "internal-key"
    env["RM_COPILOT_MODEL"] = "google/gemma-4-31b-it"   # Route A's, must be ignored
    assert config._resolve_endpoint()[1] == "default"

    env.pop("INTERNAL_LLM_BASE_URL")
    env["INTERNAL_LLM_MODEL"] = "llama-4-70b"           # Route B's, must be ignored
    assert config._resolve_endpoint()[1] == "google/gemma-4-31b-it"


def test_trailing_slash_stripped(env):
    """Avoids a '//chat/completions' path that some gateways 404 on."""
    env["INTERNAL_LLM_BASE_URL"] = "https://gw.internal/v1/"
    assert config._resolve_endpoint()[0] == "https://gw.internal/v1"


def test_explicit_vars_override_derived_defaults(env):
    """The derived defaults are conveniences, never a ceiling."""
    env.update({
        "INTERNAL_LLM_BASE_URL": "https://gw.internal/v1",
        "INTERNAL_LLM_MODEL": "llama-4-70b",
        "INTERNAL_LLM_API_KEY": "internal-key",
        "RM_COPILOT_VERIFY_SSL": "1",
    })
    base_url, model, api_key, verify = config._resolve_endpoint()
    assert (model, api_key, verify) == ("llama-4-70b", "internal-key", True)


@pytest.mark.parametrize("configured, missing_var", [
    ({"INTERNAL_LLM_BASE_URL": "https://gw.internal/v1"}, "INTERNAL_LLM_API_KEY"),
    ({}, "OPENROUTER_API_KEY"),
])
def test_missing_key_names_the_variable_to_set(monkeypatch, env, configured,
                                               missing_var):
    """A forgotten key must fail with the variable's name, not an opaque 401.

    make_llm() raises before any request, so the failure arrives at startup
    rather than several agents into a graph run.
    """
    from utils import llm

    env.update(configured)
    base_url, _model, api_key, _verify = config._resolve_endpoint()
    monkeypatch.setattr(config, "LLM_API_KEY", api_key)
    monkeypatch.setattr(config, "BASE_URL", base_url)
    monkeypatch.setattr(config, "USE_INTERNAL_LLM",
                        not base_url.startswith("https://openrouter.ai"))
    monkeypatch.setattr(llm, "LLM_API_KEY", api_key)

    with pytest.raises(ValueError, match=missing_var):
        llm.make_llm()


@pytest.mark.parametrize("ca_bundle, verify_ssl, expects_client", [
    ("", True, False),      # public CA: httpx defaults are correct, pass nothing
    ("", False, True),      # verify off: needs an explicit unverified client
])
def test_ssl_clients_only_built_when_needed(monkeypatch, ca_bundle, verify_ssl,
                                            expects_client):
    """Both transports get configured together, or neither does.

    The sync client alone is not enough: the FastAPI SSE endpoint drives the model
    over the async path, so configuring one and not the other leaves the web UI
    failing TLS while the tests pass.
    """
    from utils import llm

    monkeypatch.setattr(llm, "SSL_CA_BUNDLE", ca_bundle)
    monkeypatch.setattr(llm, "VERIFY_SSL", verify_ssl)
    clients = llm._ssl_clients()
    try:
        if expects_client:
            assert set(clients) == {"http_client", "http_async_client"}
        else:
            assert clients == {}
    finally:
        for c in clients.values():
            try:
                c.close()
            except Exception:  # noqa: BLE001 — async client needs aclose()
                pass


# ── Throttle tier ────────────────────────────────────────────────────────
# A self-hosted gateway is the one deployment that LOOKS unthrottled but is
# least able to absorb a burst (single GPU, no cross-user batching, queued).
# It used to inherit the *paid commercial* setting — no throttle at all — purely
# because its model is not named "*:free", and an overloaded gateway answers
# HTTP 500, which from the client is indistinguishable from a broken model.
@pytest.mark.parametrize("configured, expect_interval, expect_retries", [
    ({"RM_COPILOT_MODEL": "google/gemma-3-27b-it:free"}, 3.5, 3),  # free tier: 1 req/3s
    ({"RM_COPILOT_MODEL": "meta-llama/llama-4-maverick"}, 0.0, 3),  # paid: full speed
    ({"INTERNAL_LLM_BASE_URL": "https://gw.internal/v1"}, 1.0, 1),  # self-hosted: pace it
])
def test_throttle_tier_follows_the_endpoint(env, monkeypatch, configured,
                                            expect_interval, expect_retries):
    """Three tiers, resolved with no configuration by the operator."""
    monkeypatch.delenv("RM_COPILOT_MIN_INTERVAL", raising=False)
    monkeypatch.delenv("RM_COPILOT_MAX_RETRIES", raising=False)
    env.update(configured)
    base_url, model, _key, _verify = config._resolve_endpoint()
    monkeypatch.setattr(config, "MODEL", model)
    monkeypatch.setattr(config, "USE_INTERNAL_LLM",
                        not base_url.startswith("https://openrouter.ai"))

    _is_free, interval, retries = config._resolve_throttle()
    assert interval == expect_interval
    assert retries == expect_retries


def test_throttle_tier_reads_endpoint_not_model_name(env, monkeypatch):
    """A gateway's model is usually just called "default", so no name heuristic
    can identify it — the tier must come from the endpoint."""
    monkeypatch.delenv("RM_COPILOT_MIN_INTERVAL", raising=False)
    monkeypatch.setattr(config, "MODEL", "default")
    monkeypatch.setattr(config, "USE_INTERNAL_LLM", True)
    assert config._resolve_throttle()[1] == 1.0


def test_explicit_override_still_wins(env, monkeypatch):
    """The defaults are a floor, not a policy — an operator who knows their
    hardware can still say so."""
    monkeypatch.setenv("RM_COPILOT_MIN_INTERVAL", "2.5")
    monkeypatch.setenv("RM_COPILOT_MAX_RETRIES", "0")
    monkeypatch.setattr(config, "USE_INTERNAL_LLM", True)
    _is_free, interval, retries = config._resolve_throttle()
    assert (interval, retries) == (2.5, 0)


def test_llm_reads_throttle_live_not_frozen(monkeypatch):
    """utils.llm must NOT from-import these: they differ per route and are
    rebound by _refresh(), so a frozen copy would apply one route's pacing to
    the other."""
    from utils import llm

    monkeypatch.setattr(config, "MIN_INTERVAL", 0.0)
    monkeypatch.setattr(llm, "_LAST_CALL_TIME", 0.0)
    llm._throttle()                      # must not sleep, and must not NameError

    monkeypatch.setattr(config, "MAX_RETRIES", 7)
    assert config.MAX_RETRIES == 7       # read through the module, not a copy


# ── Context / output budget ──────────────────────────────────────────────
# A self-hosted gateway serves a much smaller window than a commercial API. The
# observed failure was HTTP 400 "maximum context length is 32768 tokens ... you
# requested 8192 output tokens and your prompt contains at least 24577 input
# tokens": the gateway's DEFAULT output reservation ate a quarter of the window
# before anything was sent. These pin the two halves of the fix — a per-endpoint
# budget, and never applying that budget to the commercial route.
@pytest.mark.parametrize("configured, expect_ctx, expect_out", [
    ({"OPENROUTER_API_KEY": "sk-or-test"}, 0, 0),                   # uncapped
    ({"INTERNAL_LLM_BASE_URL": "https://gw.internal/v1"}, 32768, 2048),
])
def test_budget_tier_follows_the_endpoint(env, monkeypatch, configured,
                                          expect_ctx, expect_out):
    """0 on the commercial route means "no client-side cap" — the pre-existing
    behaviour. Capping it there would throw away context the model really has."""
    monkeypatch.delenv("RM_COPILOT_CONTEXT_LIMIT", raising=False)
    monkeypatch.delenv("RM_COPILOT_MAX_TOKENS", raising=False)
    env.update(configured)
    base_url, _model, _key, _verify = config._resolve_endpoint()
    monkeypatch.setattr(config, "USE_INTERNAL_LLM",
                        not base_url.startswith("https://openrouter.ai"))

    ctx, out = config._resolve_limits()
    assert (ctx, out) == (expect_ctx, expect_out)


def test_budget_env_override_wins(env, monkeypatch):
    """Pointing at a different gateway (say a 64k one) must need no code change."""
    monkeypatch.setattr(config, "USE_INTERNAL_LLM", True)
    monkeypatch.setenv("RM_COPILOT_CONTEXT_LIMIT", "65536")
    monkeypatch.setenv("RM_COPILOT_MAX_TOKENS", "4096")
    assert config._resolve_limits() == (65536, 4096)


def test_max_tokens_omitted_when_uncapped(monkeypatch):
    """make_llm must not pass max_tokens=0 on the commercial route — that would
    be a real (and absurd) cap, not "use the provider default"."""
    from utils import llm

    captured = {}

    class _Fake:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(llm, "ThrottledChatOpenAI", _Fake)
    monkeypatch.setattr(llm, "_endpoint_error", lambda: "")
    monkeypatch.setattr(config, "MAX_OUTPUT_TOKENS", 0)
    llm.make_llm()
    assert "max_tokens" not in captured

    captured.clear()
    monkeypatch.setattr(config, "MAX_OUTPUT_TOKENS", 2048)
    llm.make_llm()
    assert captured["max_tokens"] == 2048


# ── Learning the true window from the provider's own 400 ─────────────────
_OVERFLOW_400 = (
    "BadRequestError: Error code: 400 - {'message': \"This model's maximum "
    "context length is 32768 tokens. However, you requested 8192 output tokens "
    "and your prompt contains at least 24577 input tokens, for a total of at "
    "least 32769 tokens.\"}"
)


def test_learns_context_limit_from_overflow_error(monkeypatch):
    """The per-endpoint default is a guess (a gateway's model is just "default"),
    so the first real overflow must correct it."""
    from utils import llm

    monkeypatch.setattr(config, "CONTEXT_LIMIT", 0)
    assert llm._learn_context_limit(Exception(_OVERFLOW_400)) is True
    assert config.CONTEXT_LIMIT == 32768


def test_learning_only_ever_tightens(monkeypatch):
    """A misparse must not be able to talk the app into sending MORE than it does
    today, so a larger reported window is ignored."""
    from utils import llm

    monkeypatch.setattr(config, "CONTEXT_LIMIT", 8192)
    assert llm._learn_context_limit(Exception(_OVERFLOW_400)) is False
    assert config.CONTEXT_LIMIT == 8192

    monkeypatch.setattr(config, "CONTEXT_LIMIT", 32768)
    assert llm._learn_context_limit(
        Exception(_OVERFLOW_400.replace("32768", "16384"))) is True
    assert config.CONTEXT_LIMIT == 16384


def test_unrelated_errors_do_not_touch_the_limit(monkeypatch):
    """The 504 path and the limit path share an except clause; only one may act."""
    from utils import llm

    monkeypatch.setattr(config, "CONTEXT_LIMIT", 32768)
    assert llm._learn_context_limit(Exception("operation was aborted")) is False
    assert config.CONTEXT_LIMIT == 32768


def test_learning_survives_a_gbk_console(monkeypatch, capsys):
    """Regression: the log line used an emoji, and on this repo's GBK console that
    raised UnicodeEncodeError INSIDE the guard — the limit was recorded but the
    function reported failure. Logging must never gate the result."""
    from utils import llm

    monkeypatch.setattr(config, "CONTEXT_LIMIT", 0)

    def _explode(*_a, **_k):
        raise UnicodeEncodeError("gbk", "x", 0, 1, "illegal multibyte sequence")

    monkeypatch.setattr("builtins.print", _explode)
    assert llm._learn_context_limit(Exception(_OVERFLOW_400)) is True
    assert config.CONTEXT_LIMIT == 32768


def test_llm_logging_survives_a_gbk_console(monkeypatch):
    """_log() exists because these diagnostics are emitted from inside broad except
    blocks and mid-mutation recovery code: on this repo's GBK console a non-CP936
    glyph raises UnicodeEncodeError, which would abort the repair being reported
    rather than merely losing the line."""
    from utils import llm

    def _explode(*_a, **_k):
        raise UnicodeEncodeError("gbk", "x", 0, 1, "illegal multibyte sequence")

    monkeypatch.setattr("builtins.print", _explode)
    llm._log("anything at all")          # must not raise
