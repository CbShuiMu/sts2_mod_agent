# STS2 Mod Agent

> A **Slay the Spire 2** modding agent: a RAG retrieval backend built on Flask + LangChain + Milvus, paired with a static web UI, that lets an LLM agent read and write C#/Godot source and localization files under `mods/` directly.

**Language**: [中文（默认）](./README.md) | **English** (current)

---

## Contents

- [Overview](#overview)
- [Repo layout](#repo-layout)
- [Installation](#installation)
- [Running](#running)
- [Usage](#usage)
- [Features](#features)

---

## Overview

STS2 Mod Agent chunks the game's source code and localization text (cards / relics / potions / orbs / enchantments / afflictions / rest-site UI / events) into semantic fragments and stores them in Milvus. The frontend sends a chat or query request; the backend retrieves context via RAG, then hands it to an LLM agent that calls MCP tools to generate mod code, write files, and verify assets.

| Layer | Stack |
| --- | --- |
| Backend | Python 3.14 · Flask 3 · LangChain 1.2 · `langchain-milvus` · `pymilvus` |
| Vector DB | Milvus 2.6 (defaults to Milvus Lite local `.db`; Docker Standalone also supported) |
| Embeddings | `codefuse-ai/F2LLM-v2-0.6B` (`transformers` + `torch`) |
| LLM | DeepSeek / any OpenAI-compatible endpoint (configured at `/api/config`) |
| Frontend | Plain HTML/CSS/JS (`/`, `/make`, `/query`, `/mod` pages) |
| MCP | `LocalFileMCP` for filesystem I/O + `rag_query` retrieval tool |

---

## Repo layout

```
sts2_mod_agent/
  backend/                   Flask API, RAG retrieval, agent main loop
    app.py                   Flask entry, /api/* routes, streaming chat
    agent.py                 Agent loop, tool dispatch, dedup / fallbacks
    sts2_core/               Retrieval core: embeddings, BM25 rerank, Milvus wrapper
    services/                LLM client, prompts, rag_query implementation
    mcp/                     LocalFileMCP filesystem sandbox
    scripts/                 Offline scripts (build.py etc. for vector DB)
    settings_store.py        /api/config persistence
    legacy/                  Older index code (kept for reference)

  frontend/                  Plain HTML/CSS/JS: /, /make, /query, /mod, /settings

  data/                      Local data (mostly .gitignored)
    localization/            Game-shipped localization JSON (zhs / eng × cards / powers …)
    Models/                  Decompiled game C# source (reference only)
    libs/                    Decompiled dependency DLLs
    settings/                rules.json and other runtime config
    milvus/                  docker-compose Milvus volumes (prebuilt vector DBs)
    logs/                    ai_chat_log and other per-request logs

  mods/                      Mod project root (agent's write sandbox)
    template/                Empty template — first turn copy_tree's from here
    <your_mod>/              Generated mod: src/Core/Models/*.cs, localization/, resources/

  tools/                     External tool scripts (mostly .gitignored — download separately)
    ExportMod.cmd            Bundle a mod for the game (dotnet build)
    lookup_symbol.py         Type.Member → decompiled source lookup

  docker-compose.milvus.yml  Standalone Milvus container (when not using Milvus Lite)
  .env.example               Env var template
```

---

## Installation

### Prerequisites

- **Python 3.14** (3.11 / 3.12 should also work, but `requirements.lock.txt` was not tested on lower versions)
- A package manager: **[uv](https://docs.astral.sh/uv/) (recommended)** or conda
- Optional: Docker (only needed when not using Milvus Lite)
- **External tools** (used by the export / decompile / build-mod workflow — download these, then put the absolute paths into the `.env` Tool paths section in Step 1):
  - **[Godot 4.5.1 Mono (win64)](https://godotengine.org/download/archive)** — opens / edits the mod's `.tres` resources and scenes. The **Mono** build is mandatory to run C# scripts.
  - **[GDRE Tools v2.5.0-beta.5 (windows)](https://github.com/GDRETools/gdsdecomp/releases)** — Godot asset decompiler, used to extract `.pck` contents from the shipped game for reference.
  - **[.NET 9 SDK](https://dotnet.microsoft.com/en-us/download/dotnet)** — `tools/ExportMod.cmd` shells out to `dotnet build` to compile the mod's C# project. Install the **SDK** (not just the Runtime).

### Step 1: clone and configure env vars

```bash
git clone <this-repo>
cd sts2_mod_agent
cp .env.example .env
```

Edit `.env`. The minimum to run is the LLM key; everything else is tunable:

```env
# Required (or fill via the /api/config UI)
deepseek_api_key=

# Embedding model & vector store
EMBEDDING_MODEL=codefuse-ai/F2LLM-v2-0.6B   # any HuggingFace feature-extraction model id
EMBEDDING_BATCH_SIZE=16
MILVUS_URI=http://127.0.0.1:19530           # leave blank for Milvus Lite
# MILVUS_TOKEN=
MILVUS_DB_NAME=                             # blank = default DB
DESC_COLLECTION_NAME=                       # blank = default collection name

# App server
APP_HOST=127.0.0.1
APP_PORT=7870

# RAG defaults (CLI flags still override these)
DESC_TOP_K=4
CONTEXT_N=3
CODE_CHARS=2200

# Optional: HF token / mirror
HF_TOKEN=
# HF_ENDPOINT=https://hf-mirror.com

# Tool paths (used by the export / Godot workflows; also editable from the UI under "Settings → Tool Paths")
GAME_ROOT=D:\SteamLibrary\steamapps\common\Slay the Spire 2
EXPORT_TOOL_PATH=E:\JellyProject\sts2_mod_agent\tools\ExportMod.cmd
GODOT_TOOL_PATH=E:\path\to\Godot_v4.5.1-stable_mono_win64.exe
```

> `GAME_ROOT` points at the STS2 install directory; `EXPORT_TOOL_PATH` is the bundled `tools/ExportMod.cmd` (packages a mod for the game to load); `GODOT_TOOL_PATH` is your local Godot Mono executable. All three are absolute paths — leave them blank and the matching actions are disabled in the UI.

Full list of keys in [.env.example](.env.example). CLI flags (e.g. `--port 8000`) always win over `.env`, which in turn wins over the hard-coded defaults in code.

### Step 2: install Python deps

Pick one (**uv is recommended** — single binary, zero config, much faster).

#### Option A · uv (recommended)

Install uv if you don't have it:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pipx / pip
pipx install uv      # or: pip install uv
```

Then from the repo root:

```bash
uv python install 3.14            # downloads Python 3.14 once
uv venv --python 3.14              # creates .venv/
# Activate:
#   Windows PowerShell: .venv\Scripts\Activate.ps1
#   Windows cmd:        .venv\Scripts\activate.bat
#   macOS / Linux:      source .venv/bin/activate

# Regular install
uv pip install -r backend/requirements.txt

# Or: fully reproduce the author's environment
uv pip install -r backend/requirements.lock.txt
```

#### Option B · conda + pip

```bash
conda create -n sts2_agent python=3.14 -y
conda activate sts2_agent

# Regular install
pip install -r backend/requirements.txt

# Or: fully reproduce the author's environment
pip install -r backend/requirements.lock.txt
```

> **CUDA torch (works for both options)**: install `torch==2.11.0` from the matching CUDA index per the [PyTorch site](https://pytorch.org/get-started/locally/) **first**, then run `pip install -r backend/requirements.lock.txt` (or `uv pip install ...`) — pip / uv will skip the already-satisfied torch.

### Step 3 (optional): standalone Milvus

If `milvus_lite` has no prebuilt wheel for your Python version:

```bash
docker compose -f docker-compose.milvus.yml up -d
```

Ports: `19530` (gRPC), `9091` (health/UI), `9001` (MinIO Console).

### Step 4: build the vector DBs

```bash
# uv users: activate .venv (see Step 2), or prefix with `uv run`
# conda users: conda activate sts2_agent
python backend/scripts/build.py --rebuild
```

This builds the main card/power description DB plus 7 localization shards (relics / potions / orbs / enchantments / afflictions / rest_site_ui / events) into `data/vector_db/`.

Rebuild a single shard:

```bash
python backend/scripts/build.py --target relics --rebuild --skip-preview
```

### Step 5: verify

```bash
python -c "import flask, langchain, langchain_openai, pymilvus, transformers, torch; print('OK')"
```

See [backend/INSTALL.md](./backend/INSTALL.md) and [backend/MILVUS.md](./backend/MILVUS.md) for more troubleshooting.

---

## Running

```bash
# uv users
.venv\Scripts\Activate.ps1            # Windows; macOS/Linux: source .venv/bin/activate
python backend/app.py --port 7870
# or without activating:
uv run python backend/app.py --port 7870

# conda users
conda activate sts2_agent
python backend/app.py --port 7870
```

Open <http://127.0.0.1:7870>.

| Route | Purpose |
| --- | --- |
| `/` | Landing page |
| `/make` | Conversational mod generation (Agent + RAG enabled, writes to `mods/`) |
| `/query` | Pure retrieval over the vector DBs |
| `/mod` | Browse existing mods: C# classes, asset mapping, missing resources |

Useful CLI flags:

```bash
python backend/app.py \
  --port 7870 \
  --desc-top-k 4 \
  --default-context-n 3 \
  --code-chars 2200
```

---

## Usage

1. **Configure the LLM provider**: in the settings panel (top-right of the home page), set `base_url`, `api_key`, and `model` for DeepSeek (or any OpenAI-compatible endpoint) and save as default.
2. **Generate a mod**: in `/make`, describe what you want (e.g. "make a card that draws 2 and heals 4"), enable **Agent + RAG**. The agent will:
   - call `rag_query` MCP to retrieve similar in-game cards and their localization;
   - call `local_file_read` / `local_file_write` / `local_file_replace` to materialize `.cs`, `.tres`, and `localization/*.csv` under `mods/<your_mod>/`;
   - stream `agent_trace` events so you can watch every tool call, its arguments, and its result.
3. **Browse retrieved content**: in `/query`, pick a domain (descriptions / relics / potions / ...), tune `desc_top_k` and `context_n`, and search.
4. **Inspect a mod**: `/mod` lists each mod's C# classes and their asset coverage based on `data/settings/rules.json`. Missing assets are highlighted in red.

### Recommended opening prompt

Use this verbatim:

```
Copy `template` under `mods/` and create a new project called <project_name>,
the mod contains <N> relics and <M> cards
```

### Provide fine-grained details with the visual editor (recommended)

1. Open **<https://sts2custom.shuimu.co.nz/>** and fill out each card / relic's stats, effects, and localization in a visual form;
2. Hit **Export JSON** to download the config file;
3. **Drag the JSON straight into the `/make` chat input** — a dialog pops up letting you assign a target domain (cards / relics / powers / …) to each description segment;
4. Send the opening prompt above first to scaffold the project, then drag the JSON to fill in the details.

---

## Features

### RAG retrieval

A hand-rolled hybrid pipeline ([retrieval.py](backend/sts2_core/retrieval.py)) built around the unusual "natural-language description ⇄ game source" mapping — does not reuse LangChain's stock retriever.

- **Description as index, code as answer** — indexed text is the title + description + smartDescription pulled from `data/localization/{eng,zhs}/{cards,powers}.json`; matched doc metadata resolves back to the corresponding `.cs` file.
- **Sharded vector store** — cards / powers in the main collection; relics / potions / orbs / enchantments / afflictions / rest_site_ui / events each get their own Milvus collection, filtered on demand.
- **Hand-rolled BM25 rerank** — vector search returns 1000 candidates; rerank ranks "title hits > description hits" in two tiers, with vector distance only as tie-breaker, so weak hits never outrank strong ones.
- **CN/EN-aware tokenizer** — English snake_case splitting plus Chinese character-level bigrams, with no segmenter dependency.
- **Paired-reference expansion** — a card hit also returns its paired Power + every referenced Power (and vice versa), giving the model a self-consistent code ring instead of an orphan file — the biggest fix against hallucinated APIs.
- **Engineered embedding text** — short descriptions are duplicated as `[raw, normalized, raw, normalized]` to compensate for sparsity in vector space.
- **Local inference** — `codefuse-ai/F2LLM-v2-0.6B` via transformers `feature-extraction`, with hand-written mean pooling and an RLock guarding concurrent requests.

### Description splitting + domain routing

Engineering measures iterated against "STS2 descriptions routinely span multiple domains" and "the model's recall collapses when it sends a full sentence to retrieval."

- **Upload JSON → split → assign domain** — `POST /api/descriptions/split` walks the uploaded JSON, splits every `*description*` field on CN/EN punctuation, and the UI pops a dialog where the user picks the target domain for each segment. The result is prefixed as `[Relics] On pickup` / `[Cards] upgrade all your Strikes` before reaching the agent — turning one fuzzy "search everywhere" call into N precise "single-segment × single-domain" queries.
- **Strict `rag_query` prompt** — the tool spec forbids `query` from containing `。，；、！？.,;!?\n\r`, mandates one call per segment and one domain per segment, and ships positive/negative examples.
- **Fan-out blocker** — if the model re-queries the same segment against a disjoint domain set, the call short-circuits with `skipped + reason`, forcing it to batch all needed domains into one call — collapses 3–4 wasted turns into one.
- **Domain aliases + keyword inference** — `card / cards / cardModel / CardModel` all map to `cards`; when the model omits `domains`, keyword inference picks the most likely one rather than falling back to a 9-domain full scan.
- **Top-3 auto-read after each retrieval** — `rag_query` reads the top 3 hit `.cs` files (truncated to 8 KB) into its result and tells the agent "these are already loaded — do NOT call `read_file` on them again," collapsing "retrieve → read_many" into a single turn.

### Agent main-loop stability

Generic robustness measures in [backend/app.py](backend/app.py).

- **Duplicate-call interception** — exact `(tool, args)` dedup; on a hit, the first result is fed back as `prior_result` so the agent doesn't burn a turn re-verifying.
- **`mods/`-scoped calls exempt from dedup** — generated mod files are live work-products the agent reads → writes → re-reads constantly; `mods/` calls skip dedup while the much larger `data/` reference tree stays strict.
- **Sliding-window trimming** — stale `ToolMessage`s are dropped at the top of every step (keep the most recent 8), so prompt size stays bounded.
- **Read-only streak guard** — after 3 consecutive read-only steps, a directive is injected forcing a write or a final answer — no infinite "investigation" loops.
- **Budget-exhaustion fallback** — if `stream_max_steps` is reached without a final answer, a tool-disabled wrap-up turn is appended so the user always gets a reply.
- **Completion checks** — verifies whether files were actually written / localization actually appended, and produces a targeted follow-up rather than a vague "continue".
- **Unparsed tool-call recovery** — synthesizes a `ToolMessage` with an error note for every unanswered `tool_call_id`, keeping the message sequence valid for OpenAI/DeepSeek APIs.

### Tooling & infrastructure

- **Agent + MCP toolchain** — the LLM directly calls `local_file_read / write / replace / search / list / read_many / copy_tree / create_dir / rag_query`; writes are sandboxed to `mods/`.
- **Streaming observability** — `/api/chat/stream` emits NDJSON events (`retrieval_start / retrieval_done / generation_start / token / reasoning_content / agent_trace / memory_updated / done`) for token-by-token rendering and a tool-trace panel.
- **Session memory** — each turn produces a `memory_summary` that's fed back next request — no manual context restating.
- **Rule-driven asset checks** — `data/settings/rules.json` declares the asset paths each entity type expects; `/mod` auto-resolves snake_case class names and renders missing files in red.
- **Per-request logging** — every conversation is appended to `ai_chat_log` with traces / reasoning / memory before-after / duration, making replay and tuning straightforward.
- **Local-first + unified hosting** — defaults to Milvus Lite + on-device embedding inference; Flask serves both `frontend/` and `/api/*`, no separate frontend process required.

---

## License

Bundled mod subprojects keep their own licenses; backend code is MIT.
