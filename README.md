# 🤖 Coding Assistant

A local AI coding assistant powered by [Ollama](https://ollama.com) + `qwen2.5-coder:7b`.  
Runs entirely on your machine. No cloud. No API keys.

## Features

- **Chat** — multi-turn conversation grounded in your codebase
- **Ask** — single questions with hybrid BM25 + vector retrieval
- **Index** — parallel embedding + large ChromaDB batches (4–8× faster than naive)
- **Review** — AI code review citing file + line numbers
- **Explain** — explain files, functions, or cross-cutting flows
- **Generate** — generate code matching your codebase's style
- **Web UI** — full browser interface with streaming responses
- **Configurable** — all settings adjustable from the UI

## Hardware

Optimised for **4GB VRAM / 32GB RAM**. Tested on 72k-file enterprise repos.

## Quick Start

```bash
# 1. Install Ollama and pull models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text

# 2. Install Python deps
git clone https://github.com/rajibhazarika-tm/coding-assistant
cd coding-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Start the UI
python api.py
# Open http://localhost:8000

# 4. Or use the CLI
python -m cli.main index --path /path/to/repo --auto
python -m cli.main chat
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `index --path <dir> --auto` | Index a repo (auto-detects Spring/frontend/generic) |
| `ask "question"` | Ask with RAG context |
| `chat` | Interactive multi-turn chat |
| `review --file <path>` | Code review |
| `explain --file <path> --function <name>` | Explain code |
| `generate "description"` | Generate code |
| `analyze --path <dir>` | Show indexing strategy without indexing |
| `stats` | Show index statistics |

## Performance (after fixes)

| Step | Time |
|------|------|
| Cold scan (72k files) | 10.5s |
| AST chunking (72k files) | ~14s |
| Embedding (parallel, 4 workers) | ~4× faster than sequential |
| ChromaDB upsert (128-chunk batches) | ~6× fewer disk syncs |
| Per-query overhead (BM25 cached) | <130ms |

## Configuration

All settings are configurable from the UI (`Settings` tab) or via environment variables:

| Env var | Default | Description |
|---------|---------|-------------|
| `CODING_MODEL` | `qwen2.5-coder:7b` | LLM model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `MAX_CONTEXT_TOKENS` | `2048` | Context window budget |
| `TOP_K_CHUNKS` | `5` | Chunks retrieved per query |
| `EMBED_WORKERS` | `4` | Parallel embedding threads |
| `CHROMA_BATCH_SIZE` | `128` | ChromaDB upsert batch size |
