"""Runs the browser-side chip tests (tests/js/) inside the normal pytest suite.

The Package Comparison chips are interpolated in app.js from the panel's live
inputs, so the logic worth guarding lives in JavaScript. Rather than add a second
`npm test` entry point that nobody remembers to run, the Node checks are driven
from here — `py -m pytest` stays the one command for the whole suite.

Skipped (not failed) when Node is unavailable, so the Python suite still runs on a
machine without it. No LLM / network.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

_JS_DIR = Path(__file__).resolve().parent / "js"
_NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(_NODE is None, reason="node is not installed")


@pytest.mark.parametrize(
    "script", sorted(p.name for p in _JS_DIR.glob("*.test.js")) or ["<none found>"]
)
def test_js_suite(script):
    """Each tests/js/*.test.js is one pytest case; its output is shown on failure."""
    assert script != "<none found>", f"no *.test.js found in {_JS_DIR}"
    proc = subprocess.run(
        [_NODE, str(_JS_DIR / script)],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{script} failed (exit {proc.returncode})\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
