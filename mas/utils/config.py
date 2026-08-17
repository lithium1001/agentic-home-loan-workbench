"""Centralised configuration — all secret / environment reads live here.

Importing this module triggers a single ``load_dotenv()`` so every other
module (llm, data, tools) sees the same values without re-reading ``.env``.
To change a default, set the corresponding env var in ``mas/.env`` (or the
shell) — do not hardcode elsewhere.

Secrets resolve through ``_secret()``, which works unchanged on a laptop
(``.env`` / shell) and inside Google Colab (the Secrets panel, key icon in
the left sidebar). Colab is tried first so a notebook demo needs no ``.env``
file uploaded to the VM; everything else is unaffected.

When neither source is available — most notably a Colab runtime driven from
VS Code, where ``google.colab.userdata`` is a *frontend* feature and so is
unreachable over the plain Jupyter protocol — call ``prompt_secrets()`` from
a notebook cell to type the key in by hand.
"""

import os

from dotenv import load_dotenv

load_dotenv()


# ── Secret resolution ────────────────────────────────────────────────────
def _colab_secret(name: str) -> str | None:
    """Return a Colab Secrets value, or None when not on Colab / not set.

    Every failure mode is non-fatal and looks the same to the caller: the
    import fails off-Colab, and userdata raises when the secret is missing
    (SecretNotFoundError) or when the notebook has not been granted access
    to it (NotebookAccessError). Any of those simply means "fall back to the
    environment", so they are caught broadly rather than by class — the
    exception types live in a module that does not exist off-Colab.
    """
    try:
        from google.colab import userdata  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return userdata.get(name) or None
    except Exception:
        return None


def _secret(name: str, default: str = "") -> str:
    """Read a secret: Colab Secrets first, then env var / .env, then default."""
    return _colab_secret(name) or os.getenv(name) or default


def prompt_secrets(*names: str, force: bool = False) -> None:
    """Type missing secrets in by hand, then refresh this module's constants.

    The escape hatch for environments where neither ``.env`` nor Colab Secrets
    is reachable (a Colab runtime attached from VS Code being the usual case).
    Call it from a notebook cell *before* importing the app:

        from utils import config; config.prompt_secrets()

    Input goes through ``getpass`` so the key is never echoed into the notebook
    and never persisted into the ``.ipynb`` — retyping it each session is the
    deliberate cost of not writing a credential to disk.

    Values are pushed into ``os.environ`` so that subprocesses (``uvicorn``)
    and any later re-import see them too, and the module-level constants are
    rebound because they were frozen at import time. Already-set secrets are
    skipped unless ``force``; blank input leaves a secret untouched.
    """
    from getpass import getpass

    # Default to whichever key the configured endpoint actually uses, so a Colab
    # demo against the internal gateway does not prompt for the OpenRouter key
    # (which that endpoint would reject) and vice versa.
    default_name = "INTERNAL_LLM_API_KEY" if USE_INTERNAL_LLM else "OPENROUTER_API_KEY"
    for name in (names or (default_name,)):
        if not force and _secret(name):
            print(f"{name}: already set — skipping (force=True to overwrite)")
            continue
        value = getpass(f"{name}: ").strip()
        if value:
            os.environ[name] = value
        else:
            print(f"{name}: left unset")

    _refresh()


def _resolve_endpoint() -> tuple[str, str, str, bool]:
    """Resolve which LLM endpoint to use: internal gateway, else OpenRouter.

    ``INTERNAL_LLM_BASE_URL`` is the switch, and each route then reads its OWN
    variables — nothing is shared, so both can stay filled in at once and
    switching endpoints means setting (or clearing) that one URL:

                    internal gateway         OpenRouter
        key         INTERNAL_LLM_API_KEY     OPENROUTER_API_KEY
        model       INTERNAL_LLM_MODEL       RM_COPILOT_MODEL
                      (default: "default")     (default: gemma-4-31b-it)

    The model is per-route because the two namespaces are incompatible: an
    OpenRouter id is a "vendor/model" route key, while a single-model gateway
    ignores the field entirely. One shared variable would mean a value left set
    for one route silently travels to the other.

    The credential is per-route for a stronger reason: a fallback would send the
    OpenRouter key to whatever host INTERNAL_LLM_BASE_URL names — a credential
    leak to a third party on nothing more than a typo — and would equally turn a
    forgotten INTERNAL_LLM_API_KEY into a confusing 401 instead of a clear "not
    configured". A missing key is reported by _endpoint_error() at startup.

    TLS verification also flips with the switch (off for the internal route,
    which typically serves a self-signed or private-CA certificate) but is shared
    rather than per-route, since it is a property of the connection, not of
    either provider.

    Leaving INTERNAL_LLM_BASE_URL unset reproduces the previous OpenRouter
    behaviour exactly. Every default is still overridable by its own env var.
    """
    internal = _secret("INTERNAL_LLM_BASE_URL", "").strip().rstrip("/")
    base_url = internal or "https://openrouter.ai/api/v1"
    if internal:
        model   = _secret("INTERNAL_LLM_MODEL", "default")
        api_key = _secret("INTERNAL_LLM_API_KEY")
    else:
        model   = _secret("RM_COPILOT_MODEL", "google/gemma-4-31b-it")
        api_key = _secret("OPENROUTER_API_KEY", "<YOUR_KEY_HERE>")
    # verify=False accepts ANY certificate — it disables the check that the peer
    # is who it claims to be. Acceptable on a trusted internal network; never on
    # a public endpoint. Prefer pointing RM_COPILOT_SSL_CA_BUNDLE at the private
    # CA file, which keeps verification on.
    verify   = _secret("RM_COPILOT_VERIFY_SSL", "0" if internal else "1") != "0"
    return base_url, model, api_key, verify


def _endpoint_error() -> str:
    """Describe a missing credential for the selected endpoint, else "".

    Returned rather than raised: importing config must stay side-effect-free for
    the test suite and for tooling that reads DATA_DIR without touching the LLM.
    make_llm() surfaces this at the point the credential is actually needed.
    """
    if not LLM_API_KEY or LLM_API_KEY == "<YOUR_KEY_HERE>":
        var = "INTERNAL_LLM_API_KEY" if USE_INTERNAL_LLM else "OPENROUTER_API_KEY"
        return (f"{var} is not set, but the configured LLM endpoint is {BASE_URL}. "
                f"Set {var} in mas/.env (each endpoint uses only its own key — "
                f"they are never substituted for each other).")
    return ""


def _refresh() -> None:
    """Re-resolve every constant from the current environment."""
    global OPENROUTER_API_KEY, MODEL, MAS_API_KEY
    global BASE_URL, LLM_API_KEY, VERIFY_SSL, SSL_CA_BUNDLE, USE_INTERNAL_LLM
    global IS_FREE_MODEL, MIN_INTERVAL, MAX_RETRIES
    global CONTEXT_LIMIT, MAX_OUTPUT_TOKENS
    OPENROUTER_API_KEY = _secret("OPENROUTER_API_KEY", "<YOUR_KEY_HERE>")
    MAS_API_KEY        = _secret("MAS_API_KEY", "")
    BASE_URL, MODEL, LLM_API_KEY, VERIFY_SSL = _resolve_endpoint()
    SSL_CA_BUNDLE      = _secret("RM_COPILOT_SSL_CA_BUNDLE", "")
    USE_INTERNAL_LLM   = not BASE_URL.startswith("https://openrouter.ai")
    # Must follow USE_INTERNAL_LLM / MODEL above — the throttle tier reads both.
    IS_FREE_MODEL, MIN_INTERVAL, MAX_RETRIES = _resolve_throttle()
    # Must follow the endpoint resolution above — the budget tier reads it too.
    CONTEXT_LIMIT, MAX_OUTPUT_TOKENS = _resolve_limits()


# ── Credentials / model / endpoint ───────────────────────────────────────
# Kept in sync with _refresh() above — both must set the same names, because
# these are frozen at import time and prompt_secrets() rebinds them later.
OPENROUTER_API_KEY = _secret("OPENROUTER_API_KEY", "<YOUR_KEY_HERE>")

# Set INTERNAL_LLM_BASE_URL to point the whole app at an internal
# OpenAI-compatible gateway; unset, everything resolves to OpenRouter as before.
# See _resolve_endpoint() for what each variable defaults to under either branch.
BASE_URL, MODEL, LLM_API_KEY, VERIFY_SSL = _resolve_endpoint()

# Optional private-CA bundle. Preferred over VERIFY_SSL=0: verification stays on,
# it just trusts this CA too. Takes precedence over VERIFY_SSL when both are set.
SSL_CA_BUNDLE      = _secret("RM_COPILOT_SSL_CA_BUNDLE", "")

USE_INTERNAL_LLM   = not BASE_URL.startswith("https://openrouter.ai")

# MAS APIMG portal key (Domestic Interest Rates / SORA — catalog 10484).
# Used by utils.rates.get_sora_3m(); falls back to a constant if absent.
MAS_API_KEY        = _secret("MAS_API_KEY", "")

# ── Data location ────────────────────────────────────────────────────────
# Relative to mas/ (the notebook's working directory and the package root).
DATA_DIR = os.getenv("RM_COPILOT_DATA_DIR", os.path.join("..", "csv_tables"))

# ── Rate-limit guard ─────────────────────────────────────────────────────
# Minimum seconds between LLM requests. Three tiers, all resolved automatically —
# nothing has to be configured for the common cases:
#   - free OpenRouter model  -> 3.5s  (free tier enforces ~1 req/3s; avoids 429s)
#   - internal gateway       -> 1.0s  (see below)
#   - paid commercial API    -> 0     (built to absorb bursts; run at full speed)
#
# The internal tier exists because a self-hosted gateway is the ONE case that
# looks unthrottled but is the least able to absorb a burst: typically a single
# GPU with no cross-user batching, fronted by a queue. This app fires several
# calls per agent turn back to back, and an overloaded gateway answers HTTP 500 —
# which is indistinguishable, from the client, from the model being broken. The
# earlier default sent that deployment the *paid commercial* setting (no throttle
# at all) purely because its model is not named "*:free"; 1.0s is the conservative
# default for hardware whose capacity we cannot know.
#
# The tier is read from the ENDPOINT, not from the model name: a gateway's model
# is often just called "default", so no name heuristic can identify it.
#
# NOTE: MIN_INTERVAL caps request *frequency*, not token spend — lowering it does
# not save tokens. To control cost, send fewer / smaller requests and set a
# credit cap with the provider.
def _resolve_throttle() -> tuple[bool, float, int]:
    """(is_free_model, min_interval_s, max_retries) for the current endpoint.

    A function, not inline constants, because _refresh() must be able to
    re-resolve these too: switching endpoints has to move the throttle with it,
    or a run would keep the previous route's pacing.
    """
    is_free = MODEL.strip().endswith(":free")
    if is_free:
        default_interval = "3.5"
    elif USE_INTERNAL_LLM:
        default_interval = "1.0"
    else:
        default_interval = "0"
    interval = float(os.getenv("RM_COPILOT_MIN_INTERVAL", default_interval))
    # Retries on a transient gateway error (429/5xx), with 2s/4s/8s backoff in
    # utils/llm.py. Fewer on the internal route: when a self-hosted gateway 500s it
    # is usually because its queue is already saturated, so re-sending the same
    # prompt three more times adds load to the exact thing that is failing. One
    # retry still absorbs a genuine one-off blip without piling on.
    retries = int(os.getenv("RM_COPILOT_MAX_RETRIES", "1" if USE_INTERNAL_LLM else "3"))
    return is_free, interval, retries


IS_FREE_MODEL, MIN_INTERVAL, MAX_RETRIES = _resolve_throttle()


# ── Context / output budget ──────────────────────────────────────────────
# A self-hosted gateway typically serves a far smaller context window than a
# commercial API. Observed on the internal route: HTTP 400 "This model's maximum
# context length is 32768 tokens. However, you requested 8192 output tokens and
# your prompt contains at least 24577 input tokens".
#
# Note what that message says: the OUTPUT reservation counts against the SAME
# budget as the input. Nothing in this app set max_tokens, so the gateway applied
# its own 8192 default and silently reserved a quarter of the window before a
# single message was sent. Setting it explicitly reclaims that space at no cost —
# no agent answer here comes close to 2048 tokens (the letter body is the longest
# and still lands well under it).
#
# Resolved per-ENDPOINT for the same reason as the throttle above: a gateway's
# model is often just called "default", so no name heuristic can identify it, and
# a hardcoded cap would wrongly constrain the OpenRouter route (whose models carry
# much larger windows). 0 means "impose no client-side cap" — the provider default
# applies, which is exactly the pre-existing behaviour on the commercial route.
#
# CONTEXT_LIMIT is the *total* window, and is what a trimming layer must budget
# against, since input and output share it. Both are env-overridable so pointing
# at a different gateway (say a 64k one) needs no code change.
def _resolve_limits() -> tuple[int, int]:
    """(context_limit, max_output_tokens) for the current endpoint; 0 = uncapped."""
    if USE_INTERNAL_LLM:
        default_ctx, default_out = "32768", "2048"
    else:
        default_ctx, default_out = "0", "0"
    ctx = int(os.getenv("RM_COPILOT_CONTEXT_LIMIT", default_ctx))
    out = int(os.getenv("RM_COPILOT_MAX_TOKENS", default_out))
    return ctx, out


CONTEXT_LIMIT, MAX_OUTPUT_TOKENS = _resolve_limits()

# ── Context trimming ─────────────────────────────────────────────────────
# Two independent trims keep the request bounded, both ON by default and both
# governed by this one switch (see server.stream._compact_history for the
# cross-turn half and graph._agent_segment for the within-turn half):
#
#   - within a turn: each agent's LLM call sees only its own segment, instead of
#     every earlier agent's system prompt and tool traffic;
#   - across turns: a finished turn's scaffolding is dropped from the replayed
#     history, keeping the RM's messages, each turn's answer, and the most recent
#     turn's tool results.
#
# Unlike CONTEXT_LIMIT above this is NOT endpoint-scoped: unbounded growth is wrong
# on every endpoint — a commercial model with a large window just takes longer to
# hit it, and pays for the waste in the meantime.
#
# The switch exists to make the change falsifiable. "Does trimming make the model
# dumber?" can only be answered by running the same queries both ways, and without
# a flag the only way back is editing code. Set RM_COPILOT_TRIM_CONTEXT=0 to restore
# the previous send-everything behaviour.
#
# Read live via `config.TRIM_CONTEXT` (never from-imported), so a test — or an eval
# comparing both arms in one process — can flip it with monkeypatch.
TRIM_CONTEXT = os.getenv("RM_COPILOT_TRIM_CONTEXT", "1") != "0"
