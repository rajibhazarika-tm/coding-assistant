# 🤖 Local Coding Assistant

A smart, memory-efficient coding assistant that runs entirely locally using Ollama.
Designed for **4GB VRAM / 32GB RAM** hardware constraints.

---

## 🧠 Model Recommendation

### Primary: `qwen2.5-coder:7b` (Q4_K_M quantization)

| Property | Value |
|---|---|
| Parameters | 7 Billion |
| VRAM Required | ~4.1 GB (Q4_K_M) |
| HumanEval Score | ~88% |
| Context Window | 32K tokens |
| License | Apache 2.0 (free, commercial OK) |
| Best For | Code generation, review, explanation |

```bash
ollama pull qwen2.5-coder:7b
```

**Why this model?**
- Fits entirely in your 4GB VRAM — no CPU offload, 5–10× faster inference
- Outperforms CodeLlama 13B despite being half the size
- Trained on 5.5 trillion tokens of code across 92+ languages
- Supports fill-in-the-middle (FIM) for code completion

### Fallback (even lighter): `qwen2.5-coder:3b`
If 7B is too slow, the 3B variant needs only ~2GB VRAM:
```bash
ollama pull qwen2.5-coder:3b
```

### Embedding model (required for RAG):
```bash
ollama pull nomic-embed-text
```

---

## 🏗️ Architecture

```
Your Repo
    │
    ▼
[1] SCANNER          — walks repo, respects .gitignore, detects languages
    │
    ▼
[2] AST CHUNKER      — tree-sitter parses each file into semantic units
    │  (functions, classes, methods — NOT arbitrary line splits)
    │
    ▼
[3] INDEXER          — embeds chunks via nomic-embed-text + stores in ChromaDB
    │  (incremental: only re-indexes changed files via file hashes)
    │
    ▼
[4] RETRIEVER        — hybrid BM25 + vector search, returns top-k chunks
    │  (re-ranks by recency + structural relevance)
    │
    ▼
[5] CONTEXT BUILDER  — smart token budget: fits context in ~2048 tokens
    │  (summary header + relevant chunks + file path metadata)
    │
    ▼
[6] OLLAMA LLM       — qwen2.5-coder:7b answers with minimal, precise context
```

---

## 📦 Installation

### 1. Install Ollama
```bash
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download
```

### 2. Pull models
```bash
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

### 3. Set up Python environment
```bash
cd coding-assistant
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Index your repo
```bash
python -m cli.main index --path /path/to/your/repo
```

### 5. Start chatting
```bash
python -m cli.main chat
```

---

## 🚀 Usage Examples

```bash
# Index a repository
python -m cli.main index --path ~/projects/myapp

# Ask a question
python -m cli.main ask "How does authentication work in this codebase?"

# Review a specific file
python -m cli.main review --file src/auth/login.py

# Generate code
python -m cli.main generate "Write a function to validate JWT tokens"

# Explain a function
python -m cli.main explain --file src/utils.py --function parse_config

# Interactive chat
python -m cli.main chat
```

---

## ⚙️ Configuration

Edit `config/settings.py` to tune for your hardware:

```python
MAX_CONTEXT_TOKENS = 2048    # Keep small for 4GB VRAM
TOP_K_CHUNKS = 5             # Chunks retrieved per query
CHUNK_MAX_LINES = 60         # Max lines per AST chunk
MODEL = "qwen2.5-coder:7b"
EMBED_MODEL = "nomic-embed-text"
```

---

## 🗂️ Project Structure

```
coding-assistant/
├── indexer/
│   ├── scanner.py       # Repo walker, .gitignore aware
│   ├── chunker.py       # AST-based code chunker (tree-sitter)
│   └── embedder.py      # Embedding + ChromaDB storage
├── retriever/
│   ├── hybrid_search.py # BM25 + vector hybrid retrieval
│   └── context_builder.py # Token-budget-aware context assembly
├── assistant/
│   ├── llm.py           # Ollama API wrapper
│   └── prompts.py       # Task-specific prompt templates
├── cli/
│   └── main.py          # CLI entry point
├── config/
│   └── settings.py      # All tuneable parameters
├── requirements.txt
└── README.md
```
