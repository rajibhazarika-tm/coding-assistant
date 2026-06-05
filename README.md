# 🤖 Coding Assistant

A local AI coding assistant powered by [Ollama](https://ollama.com) + `qwen2.5-coder:7b`.  
Runs entirely on your machine. No cloud. No API keys. No data leaves your machine.

---

## Features

| | |
|---|---|
| 💬 **Chat** | Multi-turn conversation grounded in your actual codebase |
| 🔍 **Ask** | Single questions with a 5-step Cursor-style retrieval pipeline |
| 📦 **Index** | Parallel embedding + large-batch ChromaDB writes (4–8× faster) |
| 🔬 **Review** | AI code review citing exact file paths and line numbers |
| 📖 **Explain** | Explain files, functions, flows, or error logs |
| ⚙️ **Generate** | Generate code that matches your codebase's style and conventions |
| 🖥️ **Desktop UI** | Native Windows app built with CustomTkinter — no browser needed |
| 🌐 **Web UI** | Optional browser interface via FastAPI + SSE streaming |
| ⚙️ **Configurable** | All settings editable from the UI or via environment variables |

---

## How It Works — 5-Step Retrieval Pipeline

Every query goes through a Cursor-style pipeline before reaching the model:

```
Your question
      │
      ▼
① Query Analysis   — LLM extracts exact grep terms, rewrites semantic
                      query, identifies likely file names, detects task type
      │
      ▼
② Grep / ripgrep   — exact literal search for function names, class names,
                      error strings, annotation names across the repo
      │
      ▼
③ Semantic Search  — vector + BM25 hybrid retrieval on the indexed codebase
                      using the rewritten query from step ①
      │
      ▼
④ Merge & Dedup    — combine grep + semantic results, collapse overlapping
                      spans, apply file-hint boost from step ①
      │
      ▼
⑤ Rerank           — LLM-free scoring: query overlap + diversity penalty +
                      test-file penalty + file-hint boost
      │
      ▼
Selected snippets → model → streamed answer
```

The pipeline trace is visible live in the Ask panel of both UIs.

---

## Hardware Requirements

Optimised for **4GB VRAM / 32GB RAM**.  
Benchmarked on a 72,024-file enterprise repo (192k chunks).

| Step | Time |
|---|---|
| Cold repo scan (72k files) | 10.5s |
| AST chunking (72k files) | ~14s |
| First-time embedding (CPU) | ~400 min — run once overnight |
| First-time embedding (GPU) | ~40 min |
| Incremental re-index (500 changed files) | ~15 min |
| Per-query pipeline overhead (warm) | <130ms |
| LLM inference per answer | 5–30s |

---

## Quick Start

### 1. Install Ollama and pull models

```bash
# Linux / macOS
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download
```

```bash
ollama pull qwen2.5-coder:7b   # ~4.1 GB — fits in 4GB VRAM
ollama pull nomic-embed-text    # ~270 MB — runs on CPU
```

### 2. Install the project

```bash
git clone https://github.com/rajibhazarika-tm/coding-assistant
cd coding-assistant
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Choose your interface

#### Desktop app (Windows — recommended)

```bash
python app.py
```

A native dark-themed window opens immediately. No browser required.

#### Web UI (any OS)

```bash
python api.py
# Open http://localhost:8000
```

#### CLI only

```bash
python -m cli.main index --path /path/to/repo --auto
python -m cli.main chat
```

---

## Build to a Standalone `.exe` (Windows)

```bat
pip install pyinstaller
pyinstaller app.spec
```

Output: `dist\CodingAssistant.exe` (~100 MB, single file).  
No Python installation required on the target machine.

---

## CLI Reference

| Command | Description |
|---|---|
| `index --path <dir> --auto` | Index a repo (auto-detects Spring / frontend / generic profile) |
| `index --path <dir> --force` | Force full re-index (ignores incremental cache) |
| `analyze --path <dir>` | Preview indexing strategy without indexing |
| `ask "question"` | Ask a single question with full pipeline |
| `chat` | Interactive multi-turn chat |
| `review --file <path>` | AI code review with line-level citations |
| `explain --file <path>` | Explain a file |
| `explain --file <path> --function <name>` | Explain a specific function |
| `generate "description"` | Generate code matching codebase style |
| `stats` | Show index statistics |

---

## Configuration

All settings are editable from the **Settings panel** in the UI, or via environment variables / `.env` file.

### Model & API

| Env var | Default | Description |
|---|---|---|
| `CODING_MODEL` | `qwen2.5-coder:7b` | LLM for answers |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### Context & Memory

| Env var | Default | Description |
|---|---|---|
| `MAX_CONTEXT_TOKENS` | `2048` | Token budget for context window |
| `TOP_K_CHUNKS` | `5` | Chunks retrieved per query |
| `CHUNK_MAX_LINES` | `50` | Max lines per AST chunk |
| `LLM_TEMPERATURE` | `0.1` | Lower = more deterministic code output |
| `LLM_MAX_TOKENS` | `1024` | Max tokens in model response |

### Indexing Performance

| Env var | Default | Description |
|---|---|---|
| `EMBED_WORKERS` | `2` | Parallel embedding threads (raise to 4–8 with more RAM) |
| `CHROMA_BATCH_SIZE` | `128` | ChromaDB upsert batch size |
| `EMBED_NUM_CTX` | `8192` | Ollama context window for embedding model |
| `EMBED_MAX_CHARS` | `2048` | Max chars per chunk sent to embedding model |
| `EMBED_QUERY_MAX_CHARS` | `1500` | Max chars for an embedded query |

> **Why `EMBED_MAX_CHARS=2048`?**  
> Dense Java/Kotlin code (annotations, generics, brackets) tokenises at ~2 chars/token.  
> 2048 chars ≈ 1024 tokens — safely within all Ollama builds.  
> Raise to `4096` if you have a newer Ollama version and want richer chunk context.

---

## Project Structure

```
coding-assistant/
├── app.py                  ← Desktop UI entry point (CustomTkinter)
├── api.py                  ← Web UI entry point (FastAPI)
├── app.spec                ← PyInstaller build spec for .exe
├── cli/
│   └── main.py             ← CLI entry point
├── config/
│   └── settings.py         ← All configuration + env var overrides
├── indexer/
│   ├── scanner.py          ← Repo walker, .gitignore-aware, incremental
│   ├── chunker.py          ← AST-based code chunker (tree-sitter)
│   ├── embedder.py         ← Parallel embedding + ChromaDB storage
│   └── strategy.py         ← Auto-detects Spring / frontend / generic profile
├── retriever/
│   ├── pipeline.py         ← 5-step Cursor-style retrieval pipeline
│   ├── hybrid_search.py    ← BM25 + vector hybrid search with corpus cache
│   └── context_builder.py  ← Token-budget-aware context assembly
├── assistant/
│   ├── llm.py              ← Ollama streaming wrapper + history trimming
│   └── prompts.py          ← Task-specific prompt templates
└── tests.py                ← 48 tests (37 core + 11 pipeline/app)
```

---

## Supported Languages

Python · Java · JavaScript · TypeScript · Go · Rust · C++ · C · C# · Ruby · PHP · Swift · Kotlin · Scala · Bash · SQL · YAML · JSON · TOML · XML · Properties · HTML · CSS · Markdown

---

## Known Fixes Applied

| Issue | Fix |
|---|---|
| Ollama 500 on 50-line Java chunks | `EMBED_NUM_CTX=8192` passed explicitly; `EMBED_MAX_CHARS` reduced to 2048 |
| Ollama 500 when pasting error logs | `_embed_query` now truncates to 1500 chars + retries on 500 |
| Embedding text truncation bug | Fixed `parts[:2]` slicing — metadata now built separately from content |
| BM25 rebuilt on every query | Corpus cached in-process, invalidated on index change (780× speedup) |
| Sequential embedding bottleneck | ThreadPoolExecutor with `EMBED_WORKERS=2` parallel requests |
| Duplicate grep chunk IDs | Deduplicate on `(file, window_start)` key, merge term names |
| `rank-bm25` unused dependency | Removed from requirements |
| `import requests` inside docstring | Moved to module level |
