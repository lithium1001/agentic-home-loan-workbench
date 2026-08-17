"""Self-tuning guard for optional outbound calls.

Three features reach the public internet — the token-cost price lookup, the MAS
SORA API, and the competitor rate scrape. All three are *optional*: each already
degrades to "unavailable" rather than inventing a number, and the case workflow
needs none of them. The problem on an air-gapped host was never correctness, it
was that each one still paid a full DNS/connect timeout to rediscover that the
internet is unreachable, on every call, forever.

The obvious fix is a config flag, but a flag is the wrong shape: it has to be
known about, documented, and set correctly on every deployment, and if it is
missed the behaviour silently reverts to stalling. So this detects instead.

The rule is a per-host circuit breaker: the FIRST time a call fails with an error
that means "this network cannot reach that name" (DNS failure, no route,
connection refused), the breaker opens and every later call short-circuits
instantly for the rest of the process. An intranet host pays one timeout at
startup and is quiet forever after; an internet host with a momentarily flaky
endpoint is unaffected, because a timeout or an HTTP error does NOT open the
breaker — only errors that indicate the network itself has no path.

Deliberately per-process and not persisted: a laptop that moves between the
office VPN and home must re-evaluate on restart, and a wrong sticky "offline"
written to disk would be far more confusing than one slow first call.
"""
from __future__ import annotations

import socket
import threading

# Errors that mean "no path to that host", as opposed to "that host answered
# badly" or "that host was slow". Only the former is evidence about the NETWORK,
# which is what the breaker is allowed to generalise from; a 500 or a read
# timeout says nothing about whether the next call can connect.
_UNREACHABLE_TYPES = (socket.gaierror, ConnectionRefusedError, ConnectionAbortedError)
_UNREACHABLE_PHRASES = (
    "name or service not known",     # Linux DNS failure (Errno -2)
    "nodename nor servname",         # macOS DNS failure
    "getaddrinfo failed",            # Windows DNS failure
    "temporary failure in name resolution",
    "no route to host",
    "network is unreachable",
    "connection refused",
    "proxy",                         # blocked by a corporate proxy
)

_lock = threading.Lock()
_offline = False        # breaker state: True = skip optional outbound calls


def looks_unreachable(exc: BaseException) -> bool:
    """True if `exc` says the network has no path to the host.

    Walks the __cause__/__context__ chain because the interesting error is
    usually wrapped: urllib raises URLError(gaierror(...)), requests raises
    ConnectionError(MaxRetryError(NewConnectionError(gaierror(...)))).
    """
    seen = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, _UNREACHABLE_TYPES):
            return True
        text = str(cur).lower()
        if any(p in text for p in _UNREACHABLE_PHRASES):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def is_offline() -> bool:
    """True once this process has proven it cannot reach the public internet."""
    return _offline


def note_failure(exc: BaseException) -> bool:
    """Record an outbound failure. Returns True if it opened the breaker.

    Call this from the `except` block of any optional outbound request.
    """
    global _offline
    if _offline or not looks_unreachable(exc):
        return False
    with _lock:
        if _offline:
            return False
        _offline = True
    return True


def reset() -> None:
    """Close the breaker again (tests, and a manual refresh the user asked for)."""
    global _offline
    with _lock:
        _offline = False
