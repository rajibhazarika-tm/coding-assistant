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
MODEL      = os.getenv("CODING_MODEL", "qwen2.5-coder:7b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ─── Context / Memory ─────────────────────────────────────────────────────────
# Input token budget for the LLM.
# qwen2.5-coder:7b supports 32k but 4GB VRAM limits safe use to ~4096 input.
# KV cache at 4096 tokens ≈ 512MB — leaves ~3.5GB for model weights.
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "4096"))

# Number of chunks passed to the LLM context.
# Lower = more precise answers (only the most relevant code).
# Raise if answers seem incomplete (missing related functions).
# Default 3: for specific questions (getUserById) 1-2 chunks are ideal;
# for broader questions (auth flow) 3 gives enough coverage.
TOP_K_CHUNKS    = int(os.getenv("TOP_K_CHUNKS", "3"))

# Tokens reserved for: system prompt (~400) + question (~200) + sources header (~100)
# Raised from 512 to 700 to match actual system prompt sizes.
RESERVED_TOKENS = 700

# ─── Indexing ─────────────────────────────────────────────────────────────────
CHUNK_MAX_LINES  = int(os.getenv("CHUNK_MAX_LINES", "60"))
CHUNK_MIN_LINES  = 3
CHROMA_COLLECTION = "code_index"
HASH_CACHE_FILE  = INDEX_DIR / "file_hashes.json"

# ─── File Filtering ───────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".tsx": "tsx", ".jsx": "javascript", ".java": "java",
    ".go": "go", ".rs": "rust", ".cpp": "cpp", ".c": "c",
    ".cs": "c_sharp", ".rb": "ruby", ".php": "php",
    ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "bash", ".md": "markdown", ".yaml": "yaml", ".yml": "yaml",
    ".json": "json", ".toml": "toml", ".xml": "xml",
    ".properties": "properties", ".gradle": "gradle",
    ".sql": "sql", ".html": "html", ".css": "css", ".scss": "scss",
}

SKIP_DIRS = {
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".pytest_cache",
    "venv", ".venv", "env", ".env",
    "dist", "build", "target", "out",
    ".gradle", ".mvn", "logs", "log", "tmp", "temp",
    "generated", "generated-sources", "generated-test-sources",
    ".idea", ".vscode", "coverage", ".coverage", "migrations",
}

SKIP_FILES = {
    "package-lock.json", "yarn.lock", "Pipfile.lock",
    "poetry.lock", "Cargo.lock", "go.sum",
}

MAX_FILE_SIZE_KB = 500

# ─── Search / Retrieval ───────────────────────────────────────────────────────
BM25_WEIGHT   = 0.3
VECTOR_WEIGHT = 0.7

# ─── Indexing performance ─────────────────────────────────────────────────────
EMBED_WORKERS      = int(os.getenv("EMBED_WORKERS", "4"))
CHROMA_BATCH_SIZE  = int(os.getenv("CHROMA_BATCH_SIZE", "128"))
EMBED_NUM_CTX      = int(os.getenv("EMBED_NUM_CTX", "8192"))
EMBED_MAX_CHARS    = int(os.getenv("EMBED_MAX_CHARS", "2048"))
EMBED_QUERY_MAX_CHARS = int(os.getenv("EMBED_QUERY_MAX_CHARS", "1500"))

# ─── LLM Generation ───────────────────────────────────────────────────────────
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Task-adaptive max response tokens.
# Explanation and review need more room; chat stays concise.
LLM_MAX_TOKENS_BY_TASK = {
    "explain":  1500,
    "review":   1500,
    "generate": 2048,
    "debug":    1200,
    "general":  1024,
    "chat":      800,
}
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))  # fallback / settings UI

LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "180"))
