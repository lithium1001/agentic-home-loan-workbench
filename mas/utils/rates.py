"""Live market-rate access — currently the MAS 3-month compounded SORA.

Single source of truth for SORA so both the RM Copilot tools and the synthetic
data generator pull the same value. Successful fetches are cached process-wide
so a demo never hammers the API. There is deliberately NO fallback rate: if the
live value cannot be fetched this raises ``SoraUnavailable``, because a stale
hardcoded SORA silently corrupts every floating instalment derived from it.

Source: MAS APIMG portal, Domestic Interest Rates (Daily), field ``comp_sora_3m``
(catalog 10484). Auth via the ``keyid`` header (key in ``mas/.env`` as MAS_API_KEY).
"""

import json

import urllib3

from utils import netguard
from utils.config import MAS_API_KEY


class SoraUnavailable(RuntimeError):
    """Raised when the live MAS SORA cannot be fetched.

    Deliberately fatal: a stale hardcoded SORA silently produces wrong
    instalments downstream, which is worse than an explicit failure.
    """


_MAS_URL = (
    "https://eservices.mas.gov.sg/apimg-gw/server/"
    "monthly_statistical_bulletin_non610mssql/"
    "domestic_interest_rates_daily/views/domestic_interest_rates_daily"
)

# Process-level cache: {"rate": float, "as_of": "YYYY-MM-DD", "source": "live"}
# Only successful fetches land here — a failure raises, so a stale value can
# never be served from cache.
_cache: dict | None = None


def _fetch_records() -> list[dict]:
    """Raw MAS rows. Raises SoraUnavailable on any failure — never fabricates."""
    # This host has already proven it has no public internet — skip the attempt
    # rather than pay another 30s timeout to be told so again. The outcome is
    # identical (SoraUnavailable, no fabricated rate), just immediate.
    if netguard.is_offline():
        raise SoraUnavailable(
            "no public internet from this host — live SORA unavailable"
        )
    if not MAS_API_KEY:
        raise SoraUnavailable(
            "MAS_API_KEY is not set — cannot fetch live SORA (set it in mas/.env)"
        )
    try:
        http = urllib3.PoolManager()
        resp = http.request("GET", _MAS_URL, headers={"keyid": MAS_API_KEY}, timeout=30.0)
        if resp.status != 200:
            raise RuntimeError(f"MAS API returned HTTP {resp.status}")
        return json.loads(resp.data.decode("utf-8"))["elements"]
    except SoraUnavailable:
        raise
    except Exception as exc:
        netguard.note_failure(exc)
        raise SoraUnavailable(f"live SORA fetch failed: {exc}") from exc


def get_sora_history(days: int = 90, tenor: str = "comp_sora_3m") -> list[dict]:
    """Daily SORA history, newest last.

    One MAS call returns the full series (~6k rows back to 2005), so history is
    free once the endpoint is reachable — no need to accumulate it ourselves.
    ``days`` caps how many of the most recent observations are returned.

    Returns ``[{"as_of": "YYYY-MM-DD", "rate": <float %>}, ...]``.
    Raises ``SoraUnavailable`` if the fetch fails.
    """
    records = _fetch_records()
    valid = [r for r in records if r.get(tenor) is not None]
    if not valid:
        raise SoraUnavailable(f"MAS API returned no rows carrying {tenor}")
    valid.sort(key=lambda r: r["end_of_day"])
    return [
        {"as_of": r["end_of_day"], "rate": round(float(r[tenor]), 4)}
        for r in valid[-days:]
    ]


def get_sora_3m(refresh: bool = False) -> dict:
    """Return the latest 3-month compounded SORA.

    Returns ``{"rate": <float %>, "as_of": <date str>, "source": "live"}``.
    Only successful fetches are cached; pass ``refresh=True`` to force a re-fetch.

    Raises ``SoraUnavailable`` if the live value cannot be retrieved — there is
    no fallback number by design. Callers must handle the failure and say the
    rate is unavailable rather than showing a stale figure.
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache

    records = _fetch_records()
    # Rows are NOT sorted by date — pick the max end_of_day with a 3M value.
    valid = [r for r in records if r.get("comp_sora_3m") is not None]
    if not valid:
        raise SoraUnavailable("MAS API returned no rows carrying comp_sora_3m")
    latest = max(valid, key=lambda r: r["end_of_day"])

    _cache = {
        "rate": round(float(latest["comp_sora_3m"]), 4),
        "as_of": latest["end_of_day"],
        "source": "live",
    }
    return _cache
