"""Optional per-token $ cost for the telemetry CSV's cost_est_usd column.

Entirely cosmetic: this fills in one reporting column and nothing in the app
reads it back. A blank cost column is a normal, healthy state — it is what you
get on a self-hosted gateway (no published per-token price exists) and on a host
without public internet (no catalogue to look one up in). Neither is an error and
neither affects a run.

When the app IS routed through a public model marketplace, that marketplace is
the authoritative and moving source of its own prices, so we do not hardcode a
rate table: on first use we fetch its model catalogue once and cache each model's
prompt/completion rate (USD per token) for the process lifetime. cost_est is then:

    prompt_tokens * prompt_price + completion_tokens * completion_price

`:free` models publish "0"/"0", so cost_est is genuinely $0 for those. Any
failure — unreachable network, parse error, unknown model — makes estimate()
return None and the caller leaves the column blank. Pricing must never break a
run or the metrics write.
"""

from __future__ import annotations

import json
import urllib.request

_MODELS_URL = "https://openrouter.ai/api/v1/models"

# model id -> (prompt_price_per_token, completion_price_per_token) as floats.
# None = not fetched yet; {} = fetched but empty (still cached, no re-fetch).
_PRICE_TABLE: dict[str, tuple[float, float]] | None = None


def _pricing_enabled() -> bool:
    """Whether we may call OpenRouter for prices at all.

    False on the internal-gateway route (a self-hosted model has no OpenRouter
    price to look up), and False once utils.netguard has established that this
    host cannot reach the public internet. Cost then stays blank, which is the
    honest answer rather than a guess.

    Any failure to read config is treated as "not enabled": on a locked-down host
    the safe default is to make no outbound request, not to attempt one.
    """
    try:
        from utils import config, netguard
        if netguard.is_offline():
            return False
        return not config.USE_INTERNAL_LLM
    except Exception:  # noqa: BLE001 — telemetry must never break on an import
        return False


def _load_table() -> dict[str, tuple[float, float]]:
    """Fetch and parse the model pricing table once; cache for the process.

    Best-effort: any network/parse failure caches an empty table so we don't
    hammer the endpoint on every run, and estimate() then returns None.
    """
    global _PRICE_TABLE
    if _PRICE_TABLE is not None:
        return _PRICE_TABLE
    table: dict[str, tuple[float, float]] = {}
    # A self-hosted / internal gateway bills nothing per token and its model id is
    # not in OpenRouter's catalog, so the fetch could only ever miss. Skip it: the
    # request costs 10s of startup latency (and may be firewalled) for no answer.
    #
    # Read config through the MODULE, never `from ... import USE_INTERNAL_LLM`:
    # that constant is rebound by config._refresh() (prompt_secrets), so a
    # from-import would freeze whatever value happened to be current when this
    # module was first imported and could send an OpenRouter request from a run
    # that is entirely on the internal gateway.
    if not _pricing_enabled():
        _PRICE_TABLE = table
        return table
    try:
        req = urllib.request.Request(_MODELS_URL, headers={"User-Agent": "rm-copilot-eval"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for m in data.get("data", []):
            mid = m.get("id")
            pricing = m.get("pricing") or {}
            if not mid:
                continue
            try:
                prompt = float(pricing.get("prompt", "0") or 0)
                completion = float(pricing.get("completion", "0") or 0)
            except (TypeError, ValueError):
                continue
            table[mid] = (prompt, completion)
    except Exception as e:  # noqa: BLE001 — pricing lookup must never raise
        # An unreachable network trips the shared breaker, so the OTHER optional
        # outbound features (SORA, rate scrape) skip their attempts too instead of
        # each paying its own timeout to learn the same thing.
        from utils import netguard
        if netguard.note_failure(e):
            # Said once per process, and only for a genuinely unreachable network.
            # Deliberately does NOT name the price source: this is a blank optional
            # column, and naming a provider here made it look like the system
            # depends on that provider, which it does not — the LLM endpoint is
            # configured separately and is unaffected by this.
            print("  [info] no public internet from this host — live rates and the "
                  "cost column are unavailable.")
    _PRICE_TABLE = table
    return table


def estimate(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimated USD cost for one run's token usage, or None if price unknown.

    None (not 0.0) signals "price unavailable" so the caller can leave the CSV
    column blank; a genuine free model returns 0.0 because its rates are "0".
    """
    if not model:
        return None
    rates = _load_table().get(model)
    if rates is None:
        return None
    prompt_price, completion_price = rates
    return round((prompt_tokens or 0) * prompt_price
                 + (completion_tokens or 0) * completion_price, 6)
