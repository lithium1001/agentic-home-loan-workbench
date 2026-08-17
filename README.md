# Agentic Home Loan Workbench

A multi-agent workbench for Singapore bank Relationship Managers handling home loans. A LangGraph agent graph drives borrower, property, policy, compliance and drafting agents through an IPA → Letter-of-Offer → Reprice workflow, fronted by a FastAPI backend and a JavaScript web UI.

Seven tool-using agents plus an orchestrator and a replan sub-orchestrator share one compiled graph. Every loan figure comes from a deterministic MAS-compliant calculator (TDSR ≤ 55%, LTV tiers, 4% stress rate) rather than the LLM; policy answers are grounded in a product T&C corpus via BM25 retrieval; and any drafted letter pauses at a human-in-the-loop Approve / Revise gate before release.

## Setup

Requires **Python 3.11+** (on Windows invoke it as `py`, not `python.exe`) and access to an LLM — either an OpenRouter key or an internal OpenAI-compatible gateway.

```bash
py -m venv .venv
.venv\Scripts\activate                      # Windows
# source .venv/bin/activate                 # macOS / Linux

py -m pip install -r mas/requirements.txt
cp mas/.env.example mas/.env                # then fill in your key
```

`mas/.env` is the one place the app is configured. It holds secrets and is gitignored — never commit it. Nothing else is needed on a fresh clone: the data is synthetic CSVs and there is no database.

### Configuration

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

Open <http://127.0.0.1:8000>, pick a case (`APP0001`–`APP0010`) from the top-bar switcher, choose a stage (IPA / LO / REPRICE), and chat. The left rail shows deal progress, a priced loan summary, and a per-stage document checklist; the right rail streams the assistant's reasoning and the audit log.

Ask for a full IPA and the graph runs borrower → property → document validation → compliance → drafting, then **pauses at the Approve / Revise gate** with a draft letter PDF you can download. Approving releases the final letter; rejecting sends your feedback to the replan sub-orchestrator, which either recomputes the numbers, redrafts the wording, or explains why the value you asked to change is fixed by regulation.

The chat needs a valid API key to reply. The read-only panels render without one.

## Usage tracking

The app can record who opened it and what they asked, into two CSVs written under `usage_data/` at runtime. Each browser gets a random `cid` cookie, which appears in both files and is what links a visit to the calculations made during it. **No collected data is bundled in this repository** — the directory is created on first run, and the instrumentation was used to measure adoption during the study. Review it against your own jurisdiction's privacy rules before deploying: `visits.csv` records client IP and user-agent.

**`visits.csv`** — one row per page view of `/`, `/rm` or `/customer`:

| Column | Meaning |
|---|---|
| `ts` | When, in UTC |
| `cid` | The visitor's cookie id — the key both files join on |
| `ua_ip_hash` | Hash of IP + browser, as a cookie-independent cross-check |
| `role` | Which audience's page: `landing`, `rm` or `customer` |
| `path` | The URL opened |
| `is_new` | `1` if this request minted the cookie, i.e. a first-ever visit |
| `ip` | Client IP (the real one, read through any proxy) |
| `user_agent` | Browser identification string |

**`customer_inputs.csv`** — one row per "Show my results" click on `/customer`:

| Column | Meaning |
|---|---|
| `ts`, `cid` | When, and which visitor |
| `mode` | Which question was asked: `price`, `downpayment`, `instalment` or `explore` |
| `ok` | `1` if the calculator returned figures, `0` if it returned an error |
| `error` | The error text when `ok=0` |
| `age` … `monthly_budget` | Exactly what the customer typed into the form |
| `financial_assets` | Declared MAS 645 assets, as JSON (the shape varies, so one cell) |
| `eligible_loan`, `monthly_repayment` | The headline figures they were shown |

`cid` identifies a **browser profile, not a person**: the same person on a laptop and a phone appears as two, and clearing cookies mints a new id, so it counts distinct browser sessions rather than unique users. `ua_ip_hash` carries the opposite bias, surviving a cookie wipe but merging everyone behind one office network. A blank cell means the value could not be read, not zero.

## Policy corpus

The `search_policy` tool answers product questions by retrieving clauses from PDFs under `mas/t&c/`. **That corpus is not included in this repository**, because the documents evaluated in the study are a bank's own published promotional terms and are not ours to redistribute.

Everything else runs without it. The tool returns no results when the directory is absent or empty, the policy agent says so rather than inventing an answer, and the retrieval tests skip themselves (`no policy PDFs under t&c/`).

To supply your own corpus, drop one or more PDFs into `mas/t&c/` (or point `RM_COPILOT_POLICY_DIR` elsewhere). The chunker splits on numbered clauses such as `1.1`, `2.10`, `3.3(a)`, so any document organised that way indexes without changes. One thing is worth adapting to a new corpus, via environment variables rather than code (see `mas/.env.example`): `RM_COPILOT_BANK_LEGAL_NAME` / `RM_COPILOT_BANK_SHORT_NAME` normalise the issuing institution's legal name to its short form at tokenization time, so boilerplate repeats stop donating the name's constituent words to unrelated queries. Both are unset here, which makes tokenization a plain lowercase split. `RM_COPILOT_BANK_ALIASES` similarly strips a letterhead the model writes from memory instead of the neutral placeholder name.

## Layout

| Path | What |
|---|---|
| `mas/graph.py` | The compiled LangGraph agent graph — single source of truth |
| `mas/skills/<agent>/` | Each agent's system prompt (`skill.md`) — edit prompts here, never in `.py` |
| `mas/utils/` | Tools, MAS loan calculator, policy RAG, data store, LLM client, letter PDF |
| `mas/utils/config.py` | Every secret/env read, and the endpoint routing |
| `mas/.env.example` | Annotated template for `mas/.env` |
| `mas/utils/telemetry/` | Runtime instrumentation and usage tracking |
| `mas/server/` | FastAPI app + SSE stream, and the JS front-end in `static/` |
| `mas/tests/` | pytest regression suite |
| `mas/t&c/` | Policy corpus for the `search_policy` RAG tool — **not bundled**, see [Policy corpus](#policy-corpus) |
| `csv_tables/` | Synthetic star-schema data (10 borrowers, `APP0001`–`APP0010`) |
