# Agentic Home Loan Workbench

A multi-agent workbench for Singapore home loans. A LangGraph agent graph drives borrower, property, policy, compliance and drafting agents through an IPA → Letter-of-Offer → Reprice workflow, fronted by a FastAPI backend and a JavaScript web UI.

Eight tool-using agents plus an orchestrator and a replan sub-orchestrator share one compiled graph. Every loan figure comes from a deterministic MAS-compliant calculator (TDSR ≤ 55%, LTV tiers, 4% stress rate) rather than the LLM; policy answers are grounded in a product T&C corpus via BM25 retrieval; and any drafted letter pauses at a human-in-the-loop Approve / Revise gate before release.

The app serves **two audiences** from one graph: a customer-facing self-service explorer, and the RM workbench.

## Setup

Requires **Python 3.10 or newer** (on Windows invoke it as `py`, not `python.exe`) and access to an LLM — either an OpenRouter key or an internal OpenAI-compatible gateway.

```bash
py -m venv .venv
.venv\Scripts\activate                      # Windows
# source .venv/bin/activate                 # macOS / Linux

py -m pip install -r mas/requirements.txt
cp mas/.env.example mas/.env                # then fill in your key
```

`mas/.env` is the one place the app is configured. It holds secrets and is gitignored — never commit it. Nothing else is needed on a fresh clone: the data is synthetic CSVs and there is no database.

### Choosing an LLM endpoint

The agents run against any **OpenAI-compatible** endpoint, so there are two options:

| | Set | Model |
|---|---|---|
| **External** — OpenRouter | `OPENROUTER_API_KEY` | `RM_COPILOT_MODEL` |
| **Internal** — your own gateway | `INTERNAL_LLM_BASE_URL` + `INTERNAL_LLM_API_KEY` | `INTERNAL_LLM_MODEL` |

`INTERNAL_LLM_BASE_URL` is the whole switch: set it and the app uses the internal gateway, leave it unset and it uses OpenRouter. Each option reads only its own variables, so both can stay filled in permanently. An OpenRouter key comes from [openrouter.ai/keys](https://openrouter.ai/keys); an internal gateway must support **tool calling**, which the agent graph depends on and OpenAI-compatibility does not guarantee.

Everything else is optional, and `mas/.env.example` documents each variable inline — TLS, throttling, context limits, and `MAS_API_KEY` for live SORA in the Market Rates panel.

## Run the web app

```bash
cd mas
export PYTHONIOENCODING=utf-8       # so emoji logging doesn't crash the console
py -m uvicorn server.app:app --reload
```

On Windows `cmd`, use `set PYTHONIOENCODING=utf-8` instead of `export`.

Open <http://127.0.0.1:8000>. The landing page asks which workspace you want:

- **`/customer`** — a self-service explorer. Fill in the form, click *Show my results*, and the MAS calculator returns what the customer can borrow, the downpayment, and the monthly instalment. Hard-scoped to the customer assistant: it cannot reach case data.
- **`/rm`** — the RM workbench. Pick a case (`APP0001`–`APP0010`) from the top-bar switcher, choose a stage (IPA / LO / REPRICE), and chat. The left rail shows deal progress, a priced loan summary, and a per-stage document checklist; the right rail streams the assistant's reasoning and the audit log.

Ask for a full IPA and the graph runs borrower → property → document validation → compliance → drafting, then **pauses at the Approve / Revise gate** with a draft letter PDF you can download. Approving releases the final letter; rejecting sends your feedback to the replan sub-orchestrator, which either recomputes the numbers, redrafts the wording, or explains why the value you asked to change is fixed by regulation.

The chat needs a valid API key to reply. The read-only panels render without one.

## Policy corpus — where to put the T&C files

The `search_policy` tool answers product questions by retrieving clauses from PDFs under **`mas/t&c/`**. That directory does not exist in a fresh clone — **create it and drop your PDFs in**:

```bash
mkdir "mas/t&c"
# copy your product T&C PDFs into it, then restart the server
```

The path is resolved relative to `mas/`, so `mas/t&c/*.pdf` is what the tool reads. Point `RM_COPILOT_POLICY_DIR` elsewhere if you prefer another location.

**No corpus is bundled here**, because the documents used in the study are a bank's own published promotional terms and are not ours to redistribute. Everything else runs without it: the tool returns no results when the directory is absent or empty, the policy agent says so rather than inventing an answer, and the retrieval tests skip themselves (`no policy PDFs under t&c/`).

The chunker splits on numbered clauses such as `1.1`, `2.10`, `3.3(a)`, so any document organised that way indexes without changes. If your PDFs name a real issuing institution, set `RM_COPILOT_BANK_LEGAL_NAME` / `RM_COPILOT_BANK_SHORT_NAME` so boilerplate repeats of the legal name stop skewing retrieval, and `RM_COPILOT_BANK_ALIASES` to strip a letterhead the model writes from memory. All three are optional and documented in `mas/.env.example`; unset, the demo runs as a neutral "The Bank".

## Usage tracking

The app can track how it is used, so a demo or pilot produces data you can analyse afterwards. Two CSVs are written under `usage_data/` at runtime, created on first run:

- **`visits.csv`** — one row per page view of `/`, `/rm` or `/customer`.
- **`customer_inputs.csv`** — one row per "Show my results" click on `/customer`: which question was asked, the figures typed in, and whether the calculator succeeded.

Both files carry the same random `cid` cookie, so joining them answers the question worth asking — of the people who opened the tool, how many actually ran a calculation. Failed attempts are recorded too (`ok=0`), which is what tells you *which inputs break*. Note that `cid` identifies a browser profile rather than a person: the same person on a laptop and a phone counts as two.

`usage_data/` is gitignored and nothing collected is bundled here — `visits.csv` records client IPs, so keep the raw files local, analyse them there, and export only aggregates. Check your own jurisdiction's privacy rules before deploying this anywhere real.

## Layout

| Path | What |
|---|---|
| `mas/graph.py` | The compiled LangGraph agent graph — single source of truth |
| `mas/skills/<agent>/` | Each agent's system prompt (`skill.md`) — edit prompts here, never in `.py` |
| `mas/utils/` | Tools, MAS loan calculator, policy RAG, data store, LLM client, letter PDF |
| `mas/utils/config.py` | Every secret/env read, and the endpoint routing |
| `mas/.env.example` | Annotated template for `mas/.env` |
| `mas/server/` | FastAPI app + SSE stream, and the JS front-end in `static/` |
| `mas/tests/` | pytest regression suite (`cd mas && py -m pytest`) |
| `mas/t&c/` | Policy corpus for the `search_policy` tool — **you create this**, see above |
| `csv_tables/` | Synthetic star-schema data (10 borrowers, `APP0001`–`APP0010`) |
