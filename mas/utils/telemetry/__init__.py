"""Runtime telemetry — operational KPIs recorded on every live graph.stream().

This is **production instrumentation, not an eval deliverable**: `utils/llm.py`
and `server/stream.py` import it, so it ships with the app. It records the KPIs
that need no golden labels — one CSV row per graph.stream() call:
  - latency (wall-clock of the stream)
  - token cost (prompt / completion / total, summed across every LLM call)
  - loop count (agent-node revisits = tool-loop iterations)
  - tool-call count, node path, interrupted?, error?

Rows land in ``artifacts/runs.csv`` (a run artifact, gitignored). The offline
evaluation harness in ``eval/`` writes the SAME schema with ``source="eval"``,
so live monitoring and offline eval share one table and never diverge.
"""

from utils.telemetry.metrics import RunMetrics, rewrite_probe, token_counter

__all__ = ["RunMetrics", "token_counter", "rewrite_probe"]
