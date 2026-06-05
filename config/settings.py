# config/settings.py
"""
Central configuration for the coding assistant.
Tuned for 4GB VRAM / 32GB RAM hardware.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ─── Models ───────────────────────────────────────────────────────────────────
# Primary coding model — fits in 4GB VRAM at Q4_K_M quantization
# HumanEval: ~88%, supports 92+ languages, Apache 2.0 license
MODEL = os.getenv("CODING_MODEL", "qwen2.5-coder:7b")

# Lightweight embedding model — runs on CPU, ~270MB
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# Ollama API base
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Memory / Context Budget ──────────────────────────────────────────────────
# CRITICAL: Keep context small — 4GB VRAM means the model needs headroom
# qwen2.5-coder:7b supports 32K but with 4GB VRAM, stay under 4K for fast inference
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2048"))

# Number of chunks to retrieve per query
TOP_K_CHUNKS = int(os.getenv("TOP_K_CHUNKS", "5"))

# Approximate token budget per chunk (leaves room for system prompt + response)
TOKENS_PER_CHUNK = 300

# Reserve tokens for system prompt, question, and response
RESERVED_TOKENS = 512

# ─── Indexing ─────────────────────────────────────────────────────────────────
# Maximum lines per chunk (prevents oversized embeddings)
CHUNK_MAX_LINES = int(os.getenv("CHUNK_MAX_LINES", "60"))

# Minimum lines to bother chunking (skip tiny stubs)
CHUNK_MIN_LINES = 3

# ChromaDB collection name
CHROMA_COLLECTION = "code_index"

# Hash file to track changed files for incremental indexing
HASH_CACHE_FILE = INDEX_DIR / "file_hashes.json"

# ─── File Filtering ───────────────────────────────────────────────────────────
# Supported languages (tree-sitter parsers available for these)
SUPPORTED_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".xml": "xml",
    ".properties": "properties",
    ".gradle": "gradle",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
}

# Directories to always skip
SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".pytest_cache",
    "venv", ".venv", "env", ".env",
    "dist", "build", "target", "out",
    ".gradle", ".mvn", "logs", "log", "tmp", "temp",
    "generated", "generated-sources", "generated-test-sources",
    ".idea", ".vscode",
    "coverage", ".coverage",
    "migrations",  # usually auto-generated
}

# Files to skip
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "Pipfile.lock",
    "poetry.lock", "Cargo.lock", "go.sum",
}

# Max file size to index (skip giant generated files)
MAX_FILE_SIZE_KB = 500

# ─── Search / Retrieval ───────────────────────────────────────────────────────
# Weight for BM25 vs vector similarity in hybrid search (0.0 = pure vector, 1.0 = pure BM25)
BM25_WEIGHT = 0.3
VECTOR_WEIGHT = 0.7

# ─── Indexing performance ─────────────────────────────────────────────────────
# Parallel embedding workers — 4 is safe for most machines; raise to 8 if you
# have a fast CPU or a GPU-backed nomic-embed-text instance
EMBED_WORKERS = int(os.getenv("EMBED_WORKERS", "4"))

# ChromaDB upsert batch size — larger = fewer disk syncs, faster overall
# 128 is optimal for most SSDs; lower to 32 if you hit memory pressure
CHROMA_BATCH_SIZE = int(os.getenv("CHROMA_BATCH_SIZE", "128"))

# ─── LLM Generation ───────────────────────────────────────────────────────────
LLM_TEMPERATURE = 0.1       # Low temperature for deterministic code output
LLM_MAX_TOKENS = 1024       # Enough for a function or review; keep short for speed
LLM_TIMEOUT_SECONDS = 120   # Ollama can be slow on first load
