"""Pytest setup for the whole suite.

Adds the package root (``mas/``) to sys.path so any test can ``import utils.*``
regardless of the directory pytest is launched from. It sits at ``mas/`` rather
than inside ``tests/`` for that reason.

Run from ``mas/`` with:

    py -m pytest                          # everything
    py -m pytest -q tests/test_policy_rag.py
"""
import os
import sys

import pytest

_MAS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _MAS_ROOT not in sys.path:
    sys.path.insert(0, _MAS_ROOT)

# Several test modules import the agent graph at module scope, and config.py
# refuses to import without credentials. The suite never reaches the network
# (see _no_live_llm below), so a placeholder is enough and lets a fresh clone
# run the tests before any key is configured. A real key in the environment or
# in mas/.env is left untouched.
if not os.getenv("OPENROUTER_API_KEY") and not os.getenv("INTERNAL_LLM_BASE_URL"):
    os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-testing-placeholder-not-a-real-key"


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch, request):
    """Fail any test that sends a request to the paid LLM endpoint.

    The test suite must be free and offline. This is not
    hypothetical — when ``search_policy`` grew an LLM query-rewrite step, the existing
    ``test_execute_tool_dispatches_search_policy`` silently started billing on every
    run, because the tool it exercises now calls a model. A test that needs the LLM
    must inject a stub (see tests/test_query_rewrite.py::_StubLLM).

    Patches the request path, not client construction: some modules build a client at
    import time without ever calling it, and failing those would be a false alarm.
    """
    if request.node.get_closest_marker("live_llm"):
        return                                     # opt out explicitly, if ever needed

    import utils.llm as _llm

    def _boom(self, messages, stop=None, run_manager=None, **kw):
        raise AssertionError(
            "This test sent a request to the LLM. Tests must be free and offline — "
            "inject a stub client (llm=...) instead."
        )

    monkeypatch.setattr(_llm.ThrottledChatOpenAI, "_generate", _boom, raising=False)
