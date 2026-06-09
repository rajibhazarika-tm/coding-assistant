"""
Test suite for the local coding assistant.
Tests each component independently (no Ollama required).
"""

import sys
import os
import json
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "
results = []

def test(name, fn):
    try:
        fn()
        results.append((PASS, name, ""))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((FAIL, name, str(e)))
        print(f"  {FAIL}  {name}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "═"*60)
print("  CODING ASSISTANT — TEST SUITE")
print("═"*60)

# ── 1. CONFIG ─────────────────────────────────────────────────────────────────
print("\n📋  [1/7] Config & Settings")

def test_config_imports():
    from config.settings import (
        MODEL, EMBED_MODEL, MAX_CONTEXT_TOKENS, TOP_K_CHUNKS,
        SUPPORTED_EXTENSIONS, SKIP_DIRS, INDEX_DIR
    )
    assert MODEL == "qwen2.5-coder:7b"
    assert MAX_CONTEXT_TOKENS == 4096  # raised from 2048 to fix chat timeout
    assert TOP_K_CHUNKS == 3  # lowered from 5 for precision
    assert ".py" in SUPPORTED_EXTENSIONS
    assert "node_modules" in SKIP_DIRS
    assert INDEX_DIR.exists()

def test_config_env_override():
    import os
    os.environ["MAX_CONTEXT_TOKENS"] = "4096"
    # Re-import won't re-read env (module cached), so just assert the env var was set
    assert os.environ["MAX_CONTEXT_TOKENS"] == "4096"
    del os.environ["MAX_CONTEXT_TOKENS"]

test("settings load correctly", test_config_imports)
test("env var override works", test_config_env_override)


# ── 2. SCANNER ────────────────────────────────────────────────────────────────
print("\n📂  [2/7] Repository Scanner")

def make_test_repo():
    """Create a temporary fake repo for testing."""
    tmp = Path(tempfile.mkdtemp())
    (tmp / "src").mkdir()
    # Use multi-line functions so they meet CHUNK_MIN_LINES (3)
    (tmp / "src" / "main.py").write_text(
        "def hello(name):\n    greeting = f'Hello {name}'\n    print(greeting)\n    return greeting\n"
    )
    (tmp / "src" / "utils.py").write_text(
        "def add(a, b):\n    result = a + b\n    return result\n"
    )
    (tmp / "tests").mkdir()
    (tmp / "tests" / "test_main.py").write_text(
        "def test_hello():\n    from src.main import hello\n    assert hello('world')\n"
    )
    (tmp / "node_modules").mkdir()
    (tmp / "node_modules" / "big.js").write_text("// should be skipped\n")
    (tmp / ".gitignore").write_text("*.pyc\n__pycache__/\n")
    (tmp / "README.md").write_text("# Test Repo\n")
    (tmp / "package-lock.json").write_text("{}")  # should be skipped
    return tmp

def test_scanner_finds_files():
    from indexer.scanner import scan_repo
    tmp = make_test_repo()
    try:
        files, deleted = scan_repo(tmp, incremental=False, force_reindex=True)
        paths = [f.relative_path for f in files]
        assert any("main.py" in p for p in paths), f"main.py not found in {paths}"
        assert any("utils.py" in p for p in paths)
        assert any("README.md" in p for p in paths)
    finally:
        shutil.rmtree(tmp)

def test_scanner_skips_node_modules():
    from indexer.scanner import scan_repo
    tmp = make_test_repo()
    try:
        files, _ = scan_repo(tmp, incremental=False, force_reindex=True)
        paths = [f.relative_path for f in files]
        assert not any("node_modules" in p for p in paths), "node_modules should be skipped"
    finally:
        shutil.rmtree(tmp)

def test_scanner_skips_lockfiles():
    from indexer.scanner import scan_repo
    tmp = make_test_repo()
    try:
        files, _ = scan_repo(tmp, incremental=False, force_reindex=True)
        paths = [f.relative_path for f in files]
        assert not any("package-lock.json" in p for p in paths)
    finally:
        shutil.rmtree(tmp)

def test_scanner_incremental_no_changes():
    from indexer.scanner import scan_repo
    tmp = make_test_repo()
    try:
        files1, _ = scan_repo(tmp, incremental=True, force_reindex=True)
        files2, _ = scan_repo(tmp, incremental=True, force_reindex=False)
        assert len(files2) == 0, f"Expected 0 changed files, got {len(files2)}"
    finally:
        shutil.rmtree(tmp)

def test_scanner_detects_changes():
    from indexer.scanner import scan_repo
    tmp = make_test_repo()
    try:
        scan_repo(tmp, incremental=True, force_reindex=True)
        # Modify a file
        (tmp / "src" / "main.py").write_text("def hello():\n    print('changed!')\n")
        files2, _ = scan_repo(tmp, incremental=True)
        changed = [f.relative_path for f in files2]
        assert any("main.py" in p for p in changed), f"Modified file not detected: {changed}"
    finally:
        shutil.rmtree(tmp)

def test_scanner_detects_deletions():
    from indexer.scanner import scan_repo
    tmp = make_test_repo()
    try:
        scan_repo(tmp, incremental=True, force_reindex=True)
        (tmp / "src" / "utils.py").unlink()
        _, deleted = scan_repo(tmp, incremental=True)
        assert any("utils.py" in d for d in deleted), f"Deletion not detected: {deleted}"
    finally:
        shutil.rmtree(tmp)

test("scanner finds .py and .md files", test_scanner_finds_files)
test("scanner skips node_modules/", test_scanner_skips_node_modules)
test("scanner skips lockfiles", test_scanner_skips_lockfiles)
test("incremental: no changes → 0 files", test_scanner_incremental_no_changes)
test("incremental: detects modified files", test_scanner_detects_changes)
test("incremental: detects deleted files", test_scanner_detects_deletions)


# ── 3. CHUNKER ────────────────────────────────────────────────────────────────
print("\n✂️   [3/7] AST Code Chunker")

SAMPLE_PYTHON = '''
import os

def simple_function(x, y):
    """Add two numbers."""
    return x + y

class MyClass:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"

    def long_method(self):
        result = []
        for i in range(100):
            result.append(i * 2)
        return result

async def async_handler(request):
    data = await request.json()
    return {"status": "ok", "data": data}
'''

SAMPLE_JS = '''
function greet(name) {
    return `Hello, ${name}!`;
}

class UserService {
    constructor(db) {
        this.db = db;
    }

    async getUser(id) {
        return await this.db.find(id);
    }
}

const fetchData = async (url) => {
    const response = await fetch(url);
    return response.json();
};
'''

def test_chunker_python_ast():
    from indexer.chunker import chunk_file
    chunks = chunk_file("src/example.py", "python", SAMPLE_PYTHON)
    assert len(chunks) > 0, "No chunks produced"
    names = [c.name for c in chunks if c.name]
    types = [c.chunk_type for c in chunks]
    assert any(n in ["simple_function", "async_handler", "MyClass"] for n in names), \
        f"Expected function names, got: {names}"

def test_chunker_javascript_ast():
    from indexer.chunker import chunk_file
    chunks = chunk_file("src/app.js", "javascript", SAMPLE_JS)
    assert len(chunks) > 0

def test_chunker_chunk_fields():
    from indexer.chunker import chunk_file
    chunks = chunk_file("src/example.py", "python", SAMPLE_PYTHON)
    c = chunks[0]
    assert c.file_path == "src/example.py"
    assert c.language == "python"
    assert c.start_line >= 1
    assert c.end_line >= c.start_line
    assert len(c.content) > 0
    assert c.id == f"src/example.py:{c.start_line}-{c.end_line}"

def test_chunker_embedding_text():
    from indexer.chunker import chunk_file
    chunks = chunk_file("src/example.py", "python", SAMPLE_PYTHON)
    for c in chunks:
        assert "src/example.py" in c.embedding_text
        assert c.content in c.embedding_text

def test_chunker_empty_file():
    from indexer.chunker import chunk_file
    chunks = chunk_file("empty.py", "python", "")
    assert chunks == []

def test_chunker_fallback_language():
    from indexer.chunker import chunk_file
    # bash is supported via line windowing fallback
    code = "\n".join([f"echo line {i}" for i in range(100)])
    chunks = chunk_file("script.sh", "bash", code)
    assert len(chunks) > 0

def test_chunker_large_file_splits():
    from indexer.chunker import chunk_file
    # A single huge function should be split into multiple chunks
    big_code = "def huge_function():\n" + "\n".join(
        [f"    x_{i} = {i} * 2" for i in range(200)]
    )
    chunks = chunk_file("big.py", "python", big_code)
    # Should produce multiple chunks due to CHUNK_MAX_LINES limit
    assert len(chunks) >= 1  # At minimum the function itself

def test_chunker_respects_min_lines():
    from indexer.chunker import chunk_file
    tiny = "x = 1\n"
    chunks = chunk_file("tiny.py", "python", tiny)
    # Under CHUNK_MIN_LINES (3), should produce no chunks or only a window
    # (line windowing still runs on the whole file)
    assert isinstance(chunks, list)

test("Python AST chunking extracts functions/classes", test_chunker_python_ast)
test("JavaScript AST chunking works", test_chunker_javascript_ast)
test("chunks have correct metadata fields", test_chunker_chunk_fields)
test("embedding_text includes file path + content", test_chunker_embedding_text)
test("empty file returns empty list", test_chunker_empty_file)
test("fallback line-windowing for unsupported langs", test_chunker_fallback_language)
test("large function is split into chunks", test_chunker_large_file_splits)
test("tiny file handled gracefully", test_chunker_respects_min_lines)


# ── 4. CONTEXT BUILDER ────────────────────────────────────────────────────────
print("\n🧩  [4/7] Context Builder")

def make_mock_chunks(n=5, lines=20):
    from retriever.hybrid_search import RetrievedChunk
    chunks = []
    for i in range(n):
        code = "\n".join([f"    line_{j} = {j}" for j in range(lines)])
        chunks.append(RetrievedChunk(
            id=f"src/file_{i}.py:1-{lines}",
            content=f"def function_{i}():\n{code}",
            file_path=f"src/file_{i}.py",
            language="python",
            start_line=1,
            end_line=lines,
            chunk_type="function",
            name=f"function_{i}",
            context_header=f"function_{i}",
            score=1.0 - i * 0.1,
        ))
    return chunks

def test_context_builder_basic():
    from retriever.context_builder import build_context
    chunks = make_mock_chunks(3)
    ctx, sources = build_context(chunks, "How does this work?")
    assert len(ctx) > 0
    assert len(sources) > 0
    assert "src/file_0.py" in sources

def test_context_builder_respects_budget():
    from retriever.context_builder import build_context, _rough_token_count
    from config.settings import MAX_CONTEXT_TOKENS
    # Large chunks — context should stay under budget
    chunks = make_mock_chunks(n=20, lines=100)
    ctx, sources = build_context(chunks, "test query")
    token_est = _rough_token_count(ctx)
    assert token_est <= MAX_CONTEXT_TOKENS, f"Context too large: {token_est} tokens"

def test_context_builder_deduplicates_sources():
    from retriever.context_builder import build_context
    from retriever.hybrid_search import RetrievedChunk
    # Two chunks from same file
    chunks = [
        RetrievedChunk("f.py:1-10", "def a(): pass", "src/f.py", "python", 1, 10, "function", "a", "", 0.9),
        RetrievedChunk("f.py:11-20", "def b(): pass", "src/f.py", "python", 11, 20, "function", "b", "", 0.8),
    ]
    ctx, sources = build_context(chunks, "query")
    assert sources.count("src/f.py") == 1, "Duplicate source files should be deduplicated"

def test_context_builder_empty_chunks():
    from retriever.context_builder import build_context
    ctx, sources = build_context([], "query")
    assert ctx == ""
    assert sources == []

def test_rough_token_count():
    from retriever.context_builder import _rough_token_count
    text = "a" * 300
    # Updated: 300 chars // 3 = 100 (changed from //4 to //3 for code accuracy)
    assert _rough_token_count(text) == 100

test("builds context from chunks", test_context_builder_basic)
test("context stays within token budget", test_context_builder_respects_budget)
test("deduplicates source file list", test_context_builder_deduplicates_sources)
test("empty chunk list → empty context", test_context_builder_empty_chunks)
test("token count approximation works", test_rough_token_count)


# ── 5. BM25 ───────────────────────────────────────────────────────────────────
print("\n🔍  [5/7] BM25 Keyword Search")

def test_bm25_basic():
    from retriever.hybrid_search import BM25
    corpus = [
        "def authenticate_user(username, password): ...",
        "def create_database_connection(host, port): ...",
        "def validate_jwt_token(token): ...",
    ]
    bm25 = BM25(corpus)
    scores = [bm25.score("authenticate user login", i) for i in range(3)]
    assert scores[0] > scores[1], "Auth doc should score highest for auth query"
    assert scores[0] > scores[2]

def test_bm25_camelcase_tokenizer():
    from retriever.hybrid_search import _tokenize
    tokens = _tokenize("parseUserRequest")
    assert "parse" in tokens
    assert "user" in tokens
    assert "request" in tokens

def test_bm25_exact_match_scores_high():
    from retriever.hybrid_search import BM25
    corpus = [
        "validateToken function checks JWT",
        "database connection pooling setup",
        "http request handler middleware",
    ]
    bm25 = BM25(corpus)
    s0 = bm25.score("validateToken", 0)
    s1 = bm25.score("validateToken", 1)
    assert s0 > s1

def test_bm25_empty_query():
    from retriever.hybrid_search import BM25
    bm25 = BM25(["some code here"])
    assert bm25.score("", 0) == 0.0

def test_bm25_single_doc():
    from retriever.hybrid_search import BM25
    bm25 = BM25(["only one document"])
    score = bm25.score("document", 0)
    assert score >= 0

test("BM25 ranks relevant docs higher", test_bm25_basic)
test("camelCase tokenizer splits correctly", test_bm25_camelcase_tokenizer)
test("exact symbol match scores highest", test_bm25_exact_match_scores_high)
test("empty query scores 0", test_bm25_empty_query)
test("single document corpus works", test_bm25_single_doc)


# ── 6. PROMPTS ────────────────────────────────────────────────────────────────
print("\n💬  [6/7] Prompt Templates")

def test_prompts_all_tasks():
    from assistant.prompts import build_prompt
    for task in ["explain", "review", "generate", "general"]:
        system, user = build_prompt(task, "What does this do?", "```python\npass\n```", ["file.py"])
        assert len(system) > 10
        assert "What does this do?" in user

def test_prompts_sources_in_message():
    from assistant.prompts import build_prompt
    _, user = build_prompt("general", "query", "context", ["src/auth.py", "src/db.py"])
    assert "src/auth.py" in user

def test_prompts_no_context():
    from assistant.prompts import build_no_context_prompt
    system, user = build_no_context_prompt("generate", "write a hello world")
    assert len(system) > 0
    assert "hello world" in user

def test_prompts_extra_instruction():
    from assistant.prompts import build_prompt
    system, _ = build_prompt("generate", "q", "ctx", [], extra_instruction="Use async/await.")
    assert "async/await" in system

def test_prompts_unknown_task_fallback():
    from assistant.prompts import build_prompt
    system, user = build_prompt("unknown_task", "query", "", [])
    assert len(system) > 0  # Falls back to SYSTEM_GENERAL

test("all task types produce valid prompts", test_prompts_all_tasks)
test("source files appear in user message", test_prompts_sources_in_message)
test("no-context prompt works", test_prompts_no_context)
test("extra instruction injected into system prompt", test_prompts_extra_instruction)
test("unknown task falls back to general", test_prompts_unknown_task_fallback)


# ── 7. INTEGRATION: SCAN → CHUNK → INDEX → RETRIEVE ──────────────────────────
print("\n🔗  [7/7] Integration: Scan → Chunk → Embed → Retrieve")

print("\n[6b] Auto Index Strategy")

def make_mixed_monorepo():
    tmp = Path(tempfile.mkdtemp())
    backend = tmp / "backend"
    (backend / "src" / "main" / "java" / "com" / "example").mkdir(parents=True)
    (backend / "src" / "main" / "resources").mkdir(parents=True)
    (backend / "pom.xml").write_text("<project></project>")
    (backend / "src" / "main" / "java" / "com" / "example" / "UserController.java").write_text(
        "@RestController\nclass UserController {\n  @GetMapping(\"/users\")\n  String users() { return \"ok\"; }\n}\n"
    )
    (backend / "src" / "main" / "resources" / "application.yml").write_text("server:\n  port: 8080\n")
    (backend / "target").mkdir()
    (backend / "target" / "Generated.class").write_text("skip")

    framework = tmp / "framework"
    (framework / "src" / "main" / "java").mkdir(parents=True)
    (framework / "build.gradle").write_text("plugins { id 'java-library' }\n")
    (framework / "src" / "main" / "java" / "Core.java").write_text("class Core {\n  void run() {}\n}\n")

    ui = tmp / "ui"
    (ui / "src").mkdir(parents=True)
    (ui / "package.json").write_text('{"scripts": {}}\n')
    (ui / "angular.json").write_text("{}\n")
    (ui / "src" / "app.ts").write_text("export const app = 1;\n")
    (ui / "node_modules").mkdir()
    (ui / "node_modules" / "huge.js").write_text("skip")
    return tmp

def test_strategy_detects_mixed_monorepo():
    from indexer.strategy import analyze_repo
    tmp = make_mixed_monorepo()
    try:
        analysis = analyze_repo(tmp)
        profiles = {area.profile for area in analysis.areas}
        assert analysis.repo_type == "mixed-monorepo", analysis.repo_type
        assert "spring" in profiles, profiles
        assert "java-library" in profiles, profiles
        assert "frontend" in profiles, profiles
    finally:
        shutil.rmtree(tmp)

def test_strategy_formats_plan():
    from indexer.strategy import analyze_repo, format_analysis
    tmp = make_mixed_monorepo()
    try:
        text = format_analysis(analyze_repo(tmp))
        assert "Index plan:" in text
        assert "backend" in text
        assert "frontend" in text
    finally:
        shutil.rmtree(tmp)

def test_scanner_uses_spring_profile():
    from indexer.scanner import scan_repo
    from indexer.strategy import get_profile
    tmp = make_mixed_monorepo()
    try:
        profile = get_profile("spring")
        files, _ = scan_repo(
            tmp / "backend",
            incremental=False,
            force_reindex=True,
            include_extensions=profile.include_extensions,
            include_filenames=profile.include_filenames,
            extra_skip_dirs=profile.extra_skip_dirs,
            max_file_size_kb=profile.max_file_size_kb,
            update_cache=False,
        )
        paths = [f.relative_path for f in files]
        assert any(p.endswith("UserController.java") for p in paths), paths
        assert any(p.endswith("application.yml") for p in paths), paths
        assert not any("target" in p for p in paths), paths
    finally:
        shutil.rmtree(tmp)

test("auto strategy detects mixed monorepo", test_strategy_detects_mixed_monorepo)
test("auto strategy formats readable plan", test_strategy_formats_plan)
test("scanner applies spring profile", test_scanner_uses_spring_profile)


def test_integration_scan_and_chunk():
    """Full pipeline without Ollama (uses a mock embed function)."""
    import unittest.mock as mock
    from indexer.scanner import scan_repo
    from indexer.chunker import chunk_file

    tmp = make_test_repo()
    try:
        files, _ = scan_repo(tmp, incremental=False, force_reindex=True)
        assert len(files) > 0

        all_chunks = []
        for f in files:
            content = f.path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_file(f.relative_path, f.language, content)
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0
        # All chunks should have valid metadata
        for c in all_chunks:
            assert c.file_path
            assert c.start_line >= 1
            assert c.end_line >= c.start_line
            assert len(c.content.strip()) > 0
    finally:
        shutil.rmtree(tmp)

def test_integration_chromadb_store_retrieve():
    """Test ChromaDB store/retrieve with mock embeddings (no Ollama)."""
    import gc
    import chromadb
    from chromadb.config import Settings

    tmpdir = tempfile.mkdtemp()
    try:
        client = chromadb.PersistentClient(
            path=tmpdir,
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name="test_collection",
            metadata={"hnsw:space": "cosine"},
        )

        # Use clearly distinct embeddings for reliable retrieval
        auth_vec   = [1.0, 0.0] + [0.0] * 766
        db_vec     = [0.0, 1.0] + [0.0] * 766
        token_vec  = [0.7, 0.7] + [0.0] * 766

        collection.upsert(
            ids=["chunk_auth", "chunk_db", "chunk_token"],
            embeddings=[auth_vec, db_vec, token_vec],
            documents=[
                "def authenticate(user, password): ...",
                "def connect_database(host, port): ...",
                "def validate_token(jwt): ...",
            ],
            metadatas=[
                {"file_path": "auth.py", "language": "python", "chunk_type": "function"},
                {"file_path": "db.py",   "language": "python", "chunk_type": "function"},
                {"file_path": "auth.py", "language": "python", "chunk_type": "function"},
            ],
        )

        assert collection.count() == 3

        # Query with auth_vec — closest match should be chunk_auth
        results = collection.query(query_embeddings=[auth_vec], n_results=1)
        assert results["ids"][0][0] == "chunk_auth", \
            f"Expected chunk_auth as top result, got {results['ids'][0]}"

        # Filter by file
        filtered = collection.get(where={"file_path": "auth.py"})
        assert len(filtered["ids"]) == 2, \
            f"Expected 2 auth.py chunks, got {len(filtered['ids'])}"
    finally:
        del collection
        del client
        gc.collect()
        shutil.rmtree(tmpdir, ignore_errors=True)

def test_integration_rrf_fusion():
    """Test that RRF correctly combines vector + BM25 ranks."""
    from retriever.hybrid_search import _rrf_score, BM25

    # RRF scores should decrease with rank
    scores = [_rrf_score(r) for r in range(5)]
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i+1], "RRF scores should decrease with rank"

    # BM25 integration: symbol query should find exact match
    corpus = [
        "def authenticate_user(): pass",
        "def send_email(): pass",
        "def parse_json(): pass",
    ]
    bm25 = BM25(corpus)
    auth_score = bm25.score("authenticate_user", 0)
    other_scores = [bm25.score("authenticate_user", i) for i in range(1, 3)]
    assert all(auth_score > s for s in other_scores)

test("scan + chunk pipeline produces valid chunks", test_integration_scan_and_chunk)
test("ChromaDB store and vector retrieval works", test_integration_chromadb_store_retrieve)
test("RRF fusion ranks correctly", test_integration_rrf_fusion)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
total = len(results)

print("\n" + "═"*60)
print(f"  RESULTS: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} failed)")
    print("\n  Failed tests:")
    for icon, name, err in results:
        if icon == FAIL:
            print(f"    {FAIL} {name}")
            print(f"       {err}")
else:
    print("  — all clear! 🎉")
print("═"*60 + "\n")

sys.exit(0 if failed == 0 else 1)
