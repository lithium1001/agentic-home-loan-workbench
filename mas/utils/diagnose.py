"""Print which LLM endpoint this host actually resolved, and why.

Run ON the deployment host, from mas/:

    py -m utils.diagnose

Answers the one question that decides everything else: is this process talking to
the internal gateway or to OpenRouter? A misconfiguration here does not announce
itself — the app simply runs against the wrong backend, and the symptom (an
opaque 500, a DNS error from an unrelated telemetry call) looks like a bug in the
application rather than in the environment.

Prints no secret VALUES, only whether each variable is set and how long it is, so
the output is safe to paste into a ticket or a chat.
"""
from __future__ import annotations

import os

from utils import config


def _shown(name: str) -> str:
    """'SET (n chars)' / '-- unset --', never the value itself."""
    raw = os.getenv(name)
    if raw is None:
        return "-- unset --"
    if not raw.strip():
        # The trap: a blank or whitespace-only value is stripped to "" and then
        # falls back to the OpenRouter default, so the operator believes the
        # internal route is configured while the app is on the public one.
        return f"!! SET BUT BLANK ({len(raw)} chars of whitespace) -> treated as UNSET"
    return f"SET ({len(raw.strip())} chars)"


def main() -> int:
    print("=" * 68)
    print("Environment variables (values never printed)")
    print("=" * 68)
    for name in ("INTERNAL_LLM_BASE_URL", "INTERNAL_LLM_API_KEY", "INTERNAL_LLM_MODEL",
                 "OPENROUTER_API_KEY", "RM_COPILOT_MODEL",
                 "RM_COPILOT_VERIFY_SSL", "RM_COPILOT_SSL_CA_BUNDLE", "MAS_API_KEY"):
        print(f"  {name:26} {_shown(name)}")

    internal = config.USE_INTERNAL_LLM
    print()
    print("=" * 68)
    print("Resolved configuration")
    print("=" * 68)
    print(f"  route            {'INTERNAL GATEWAY' if internal else 'OPENROUTER (public)'}")
    print(f"  BASE_URL         {config.BASE_URL}")
    print(f"  MODEL            {config.MODEL}")
    print(f"  key configured   {'yes' if config.LLM_API_KEY and config.LLM_API_KEY != '<YOUR_KEY_HERE>' else 'NO -- calls will 401'}")
    print(f"  TLS verify       {config.VERIFY_SSL}"
          f"{'  (custom CA: ' + config.SSL_CA_BUNDLE + ')' if config.SSL_CA_BUNDLE else ''}")

    print()
    print("=" * 68)
    print("Outbound calls this process may make")
    print("=" * 68)
    try:
        from utils.telemetry import pricing
        price_on = pricing._pricing_enabled()
    except Exception as exc:  # noqa: BLE001
        price_on = f"unknown ({exc})"
    auto = "attempted once, then auto-disabled if unreachable"
    print(f"  LLM (required)   {config.BASE_URL}")
    print()
    print("  Optional, cosmetic only — each self-disables on an unreachable network:")
    print(f"    token cost     {auto if price_on is True else 'no (skipped)'}")
    print(f"    live SORA      {auto}")
    print(f"    market rates   {auto}")
    print()
    print("  There is no flag to set for these. Only the LLM endpoint above has to")
    print("  be correct; the case workflow needs none of the optional calls.")

    print()
    if not internal:
        print("  >>> This host is on the PUBLIC OpenRouter route.")
        print("      If that is not intended, INTERNAL_LLM_BASE_URL is unset, blank,")
        print("      still commented out in mas/.env, or set in a different shell")
        print("      than the one running the server.")
    else:
        print("  >>> Internal gateway route. This is the intended intranet setup.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
