# Agentic Home Loan Workbench — guided-workflow web UI

A FastAPI + static-HTML/JS replacement for the Gradio UI, built to look and feel
like a bank **information system** (SAP-/Salesforce-style) so a new RM can onboard
fast and drive the workflow by clicking buttons instead of writing prompts. The
visual language follows `reference_file/RM Copilot v2 - Standalone.html`.

## Run

```bash
cd mas
# .env must contain a valid OPENROUTER_API_KEY (and optional RM_COPILOT_MODEL)
py -m uvicorn server.app:app --reload
```

Then open <http://127.0.0.1:8000>. (On Windows set `PYTHONIOENCODING=utf-8` first so
emoji logging doesn't crash the console.)

## Layout (three panes)

- **Top bar** — brand + **Current Case** switcher (pick any of APP0001–APP0020).
- **Left rail** — **Deal Progress** (IPA → LO → REPRICE stage dots), **Loan Summary**
  card (priced live via `calculate_loan`), **Documents** checklist per stage.
- **Center** — chat scoped to the selected (customer, stage), with per-agent
  **thinking blocks**, **action chips**, a **misroute → "switch stage?"** prompt, and
  the **HITL Approve / Revise** draft gate.
- **Right** — slide-over **Activity log** (tool calls + results).

## How it is wired

```
browser (static/) ──fetch──▶ /api/cases, /api/case/{id}     ──▶ server/case_service.py ──▶ utils DataStore + calculate_loan
                  ──SSE────▶ /api/chat, /api/approve, /api/reject ──▶ server/stream.py ──▶ graph.py (graph.stream / Command resume)
```

- **`app.py`** — FastAPI app: read-only case endpoints + SSE chat/HITL endpoints +
  static mount.
- **`case_service.py`** — derives the left-rail panels from the **existing** DataStore
  (no new data, no DB).
- **`stream.py`** — drives `graph.stream(...)` and translates each LangGraph event into
  a small JSON SSE event (`thinking_open/close`, `tool_call/result`, `misroute`,
  `answer`, `draft`, `error`, `done`). Per-(applicant, stage) conversation state +
  the active HITL `thread_id` live server-side in `SESSIONS`.
- **`graph.py`** (in `mas/`) — the compiled agent graph, unchanged from the notebook
  (extracted from cell 13 so the server and the notebook share one source of truth).

The agent/tool/calculator logic is untouched: this is a UI shell swap only.
