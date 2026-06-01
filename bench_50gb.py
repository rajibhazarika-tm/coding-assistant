"""
50GB-scale workspace performance benchmark.
Tests every pipeline layer against 72,024 files / 3.4M lines.
No Ollama required — embeddings are mocked.
"""
import sys, os, gc, time, tracemalloc, statistics, tempfile, shutil, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

REPO   = Path("/tmp/enterprise_repo")
TMPDIR = Path(tempfile.mkdtemp())
RESULTS = {}

# ── helpers ───────────────────────────────────────────────────────────────────
W = 68
def hr(title=""):
    print(f"\n{'─'*W}")
    if title: print(f"  {title}"); print(f"{'─'*W}")

def measure(label, fn, runs=1):
    times, peak_mb = [], 0
    result = None
    for _ in range(runs):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peak_mb = max(peak_mb, peak / 1_048_576)
    avg = statistics.mean(times)
    mn, mx = min(times), max(times)
    RESULTS[label] = dict(avg=round(avg,4), min=round(mn,4), max=round(mx,4), mem=round(peak_mb,2))
    icon = "🐢" if avg > 10 else ("⚠️ " if avg > 3 else "✅")
    if runs > 1:
        print(f"  {icon} {label:52s} avg={avg:.3f}s  mem={peak_mb:.1f}MB")
    else:
        print(f"  {icon} {label:52s} {avg:.3f}s  mem={peak_mb:.1f}MB")
    return result

# ── 1. SCAN ───────────────────────────────────────────────────────────────────
hr("1.  REPO SCAN  (72,024 files)")
from indexer.scanner import scan_repo, _hash_cache_file, repo_id_for_path

repo_id = repo_id_for_path(REPO)
hcf = _hash_cache_file(repo_id)
if hcf.exists(): hcf.unlink()

files, deleted = measure("Cold scan  (no cache)", lambda: scan_repo(REPO, force_reindex=True))
print(f"     → {len(files):,} files queued, {len(deleted)} deleted")

files2, _ = measure("Warm scan  (all cached, 0 changes)", lambda: scan_repo(REPO))
print(f"     → {len(files2):,} re-index (expected 0)")

# Touch 500 files
import random; random.seed(9)
py_files = list(REPO.rglob("*.py"))
touched = random.sample(py_files, min(500, len(py_files)))
for f in touched:
    f.write_text(f.read_text() + "\n# perf-touch\n")
files3, _ = measure("Incremental scan  (500 changed)", lambda: scan_repo(REPO))
print(f"     → {len(files3):,} changed files detected (expected ~500)")

total_items = len(list(REPO.rglob("*")))
scan_tput = total_items / RESULTS["Cold scan  (no cache)"]["avg"]
print(f"     → Throughput: {scan_tput:,.0f} fs-entries/sec")

# ── 2. STRATEGY ANALYSIS ─────────────────────────────────────────────────────
hr("2.  STRATEGY ANALYSIS  (auto profile detection)")
from indexer.strategy import analyze_repo, format_analysis

analysis = measure("analyze_repo  (single walk, no double-walk)", lambda: analyze_repo(REPO))
print(f"     → type={analysis.repo_type}, areas={len(analysis.areas)}, markers={analysis.markers[:5]}")
for a in analysis.areas[:3]:
    print(f"       {a.path.name}/ → {a.profile}  ~{a.estimated_files:,} files")

# ── 3. CHUNKING ───────────────────────────────────────────────────────────────
hr("3.  AST CHUNKING  (full 72k-file corpus)")
from indexer.chunker import chunk_file

# Refresh file list (use force to get all)
if hcf.exists(): hcf.unlink()
all_files, _ = scan_repo(REPO, force_reindex=True)

def chunk_all():
    chunks = []
    errors = 0
    for f in all_files:
        try:
            content = f.path.read_text(encoding="utf-8", errors="replace")
            chunks.extend(chunk_file(f.relative_path, f.language, content, repo_id=f.repo_id))
        except Exception:
            errors += 1
    return chunks, errors

all_chunks, chunk_errors = measure(f"Chunk all {len(all_files):,} files", chunk_all)
chunk_tput = len(all_chunks) / RESULTS[f"Chunk all {len(all_files):,} files"]["avg"]
print(f"     → {len(all_chunks):,} chunks | {chunk_errors} errors | {chunk_tput:,.0f} chunks/sec")

from collections import Counter
by_lang = Counter(c.language for c in all_chunks)
by_type = Counter(c.chunk_type for c in all_chunks)
sizes   = [c.end_line - c.start_line + 1 for c in all_chunks]
named   = sum(1 for c in all_chunks if c.name)
print(f"     → Languages: {dict(by_lang.most_common(5))}")
print(f"     → Types:     {dict(by_type.most_common(5))}")
print(f"     → Named:     {named:,}/{len(all_chunks):,} ({100*named//len(all_chunks)}%)")
p = sorted(sizes)
print(f"     → Lines/chunk: min={p[0]} p25={p[len(p)//4]} median={p[len(p)//2]} p95={p[int(len(p)*.95)]} max={p[-1]}")

# Chunking a single file (latency)
sample = next(f for f in all_files if f.language == "java")
content = sample.path.read_text(errors="replace")
measure(f"Chunk 1 Java file ({len(content)} bytes)", 
        lambda: chunk_file(sample.relative_path, "java", content), runs=20)

# ── 4. INDEX (ChromaDB + mock embeddings) ─────────────────────────────────────
hr("4.  CHROMADB INDEXING  (mock embeddings, full corpus)")

import chromadb, struct, hashlib, math as _math
from chromadb.config import Settings

client = chromadb.PersistentClient(path=str(TMPDIR), settings=Settings(anonymized_telemetry=False))
col = client.get_or_create_collection("bench50gb", metadata={"hnsw:space": "cosine"})

def fast_embed(text: str) -> list[float]:
    vec = [0.0] * 768
    words = text.lower().split()
    for i, w in enumerate(words[:150]):
        h = int(hashlib.md5(w.encode()).hexdigest(), 16)
        for d in range(4):
            vec[(h >> (d*8)) % 768] += 1.0 / (i + 1)
    norm = _math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x/norm for x in vec]

BATCH = 64
def index_all():
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i+BATCH]
        col.upsert(
            ids=[c.id for c in batch],
            embeddings=[fast_embed(c.embedding_text) for c in batch],
            documents=[c.content for c in batch],
            metadatas=[{"file_path":c.file_path,"language":c.language,
                        "start_line":c.start_line,"end_line":c.end_line,
                        "chunk_type":c.chunk_type,"name":c.name or "",
                        "repo_id":c.repo_id} for c in batch],
        )

index_all_result = measure(f"Index {len(all_chunks):,} chunks into ChromaDB (mock)", index_all)
idx_tput = len(all_chunks) / RESULTS[f"Index {len(all_chunks):,} chunks into ChromaDB (mock)"]["avg"]
print(f"     → {col.count():,} stored | mock throughput {idx_tput:,.0f} chunks/sec")
real_est = len(all_chunks) / 8   # ~8 chunks/sec on CPU nomic-embed-text
print(f"     → Real embed estimate (CPU nomic-embed-text ~8/s): {real_est/60:.0f} min")
print(f"     → With GPU (RTX 3060 ~80/s): {len(all_chunks)/80/60:.0f} min")

# ── 5. BM25 BUILD + QUERY SCALING ────────────────────────────────────────────
hr("5.  BM25 — build time & query latency at scale")
from retriever.hybrid_search import BM25

corpus_docs = [c.content for c in all_chunks]

# Build at different sizes to show scaling
for n in [1_000, 5_000, 10_000, 50_000, len(all_chunks)]:
    sample_docs = corpus_docs[:n]
    b = measure(f"  BM25 build  ({n:>7,} docs)", lambda d=sample_docs: BM25(d), runs=1)

# Query latency at full scale
bm25_full = BM25(corpus_docs)
QUERIES = [
    "authenticate user password jwt",
    "database connection pool retry",
    "cache invalidation eviction",
    "order payment billing process",
    "batch operations bulk insert",
    "health check status endpoint",
    "shutdown close cleanup graceful",
    "validate config host port timeout",
]
def bm25_all_queries():
    results = []
    for q in QUERIES:
        scores = [(i, bm25_full.score(q, i)) for i in range(len(corpus_docs))]
        top5 = sorted(scores, key=lambda x: -x[1])[:5]
        results.append(top5)
    return results

bm25_results = measure(
    f"  BM25 query  ({len(QUERIES)} queries × {len(corpus_docs):,} docs)",
    bm25_all_queries, runs=3)
qps = len(QUERIES) / RESULTS[f"  BM25 query  ({len(QUERIES)} queries × {len(corpus_docs):,} docs)"]["avg"]
print(f"     → {qps:.1f} queries/sec at {len(corpus_docs):,} docs")
print(f"     → Single query latency: {1000/qps/len(QUERIES):.0f}ms avg per query")

# ── 6. BM25 CACHE BENEFIT (Fix 4) ─────────────────────────────────────────────
hr("6.  BM25 CACHE BENEFIT  (Fix 4 — before vs after)")
from retriever.hybrid_search import _get_cached_corpus, _get_cached_bm25, _corpus_cache, _metadata_filter

# Simulate old behaviour: rebuild BM25 on every call
def old_retrieve_overhead():
    all_data = col.get(include=["documents","metadatas"])
    corpus = list(zip(all_data["ids"], all_data["documents"],
                      [m or {} for m in all_data["metadatas"]]))
    BM25([d for _,d,_ in corpus])  # rebuild every time

# New behaviour: cache hit after first build
_corpus_cache.clear()
def new_retrieve_overhead_cold():
    _get_cached_corpus(col, None, None)
    _get_cached_bm25(col, None, None)

def new_retrieve_overhead_warm():
    _get_cached_corpus(col, None, None)
    _get_cached_bm25(col, None, None)

measure("  OLD: ChromaDB load + BM25 rebuild per query", old_retrieve_overhead, runs=3)
_corpus_cache.clear()
measure("  NEW: cold (first call — builds cache)",       new_retrieve_overhead_cold, runs=1)
measure("  NEW: warm (subsequent calls — cache hit)",    new_retrieve_overhead_warm, runs=5)

old_t = RESULTS["  OLD: ChromaDB load + BM25 rebuild per query"]["avg"]
warm_t = RESULTS["  NEW: warm (subsequent calls — cache hit)"]["avg"]
print(f"     → Speedup on warm queries: {old_t/max(warm_t,0.001):.0f}× faster")

# ── 7. CONTEXT BUILDER ───────────────────────────────────────────────────────
hr("7.  CONTEXT BUILDER  (token budget)")
from retriever.hybrid_search import RetrievedChunk
from retriever.context_builder import build_context, _rough_token_count

def make_retrieved(n):
    return [RetrievedChunk(
        id=all_chunks[i].id, content=all_chunks[i].content,
        file_path=all_chunks[i].file_path, language=all_chunks[i].language,
        start_line=all_chunks[i].start_line, end_line=all_chunks[i].end_line,
        chunk_type=all_chunks[i].chunk_type, name=all_chunks[i].name or "",
        context_header=all_chunks[i].context_header, score=1-i*.05,
    ) for i in range(n)]

for n in [5, 10, 20]:
    r = make_retrieved(n)
    ctx, srcs = measure(f"  build_context  ({n} chunks)", lambda rr=r: build_context(rr, "query"), runs=100)
ctx_str, srcs = build_context(make_retrieved(5), "query")
print(f"     → ~{_rough_token_count(ctx_str)} tokens per context window")

# ── 8. END-TO-END QUERY PIPELINE ─────────────────────────────────────────────
hr("8.  END-TO-END QUERY  (excl. embed + LLM)")
from assistant.prompts import build_prompt

def e2e_query(q="How does the auth service handle retries?"):
    bm25_scores = [(i, bm25_full.score(q, i)) for i in range(len(corpus_docs))]
    top5_idx = sorted(bm25_scores, key=lambda x: -x[1])[:5]
    retrieved = [RetrievedChunk(
        id=all_chunks[i].id, content=all_chunks[i].content,
        file_path=all_chunks[i].file_path, language=all_chunks[i].language,
        start_line=all_chunks[i].start_line, end_line=all_chunks[i].end_line,
        chunk_type=all_chunks[i].chunk_type, name=all_chunks[i].name or "",
        context_header=all_chunks[i].context_header, score=sc,
    ) for i, sc in top5_idx]
    ctx, srcs = build_context(retrieved, q)
    system, user_msg = build_prompt("general", q, ctx, srcs)
    return ctx, srcs

ctx, srcs = measure("E2E: BM25 search + context + prompt build (warm)", e2e_query, runs=10)
print(f"     → {_rough_token_count(ctx)} tokens | {len(srcs)} source files")

# ── 9. MEMORY PROFILE ────────────────────────────────────────────────────────
hr("9.  MEMORY PROFILE")
import psutil, os as _os
proc = psutil.Process(_os.getpid())
rss = proc.memory_info().rss / 1_048_576

tracemalloc.start()
_ = list(all_chunks)  # ensure fully materialised
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"  Process RSS:            {rss:.0f} MB")
print(f"  {len(all_chunks):,} chunks in RAM:    {peak/1_048_576:.1f} MB alloc")
print(f"  Per-chunk overhead:     {peak/max(len(all_chunks),1):.0f} bytes")

tracemalloc.start()
_ = BM25(corpus_docs[:10_000])
_, bm25_peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"  BM25 (10k docs):        {bm25_peak/1_048_576:.1f} MB")
print(f"  BM25 proj ({len(all_chunks):,} docs):  ~{bm25_peak/1_048_576 * len(all_chunks)/10000:.0f} MB")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
shutil.rmtree(TMPDIR, ignore_errors=True)

print(f"\n{'═'*W}")
print("  PERFORMANCE SUMMARY  —  50GB-scale workspace")
print(f"  Repo: {len(all_files):,} files · {len(all_chunks):,} chunks · {sum(sizes):,} lines")
print(f"{'═'*W}\n")

rows = [
    ("Cold scan (72k files)",           "Cold scan  (no cache)",                         "s"),
    ("Warm scan (0 changes)",           "Warm scan  (all cached, 0 changes)",             "s"),
    ("Incremental (500 changes)",       "Incremental scan  (500 changed)",               "s"),
    ("Strategy analysis",               "analyze_repo  (single walk, no double-walk)",   "s"),
    ("Chunk all files",                 f"Chunk all {len(all_files):,} files",            "s"),
    ("BM25 build (full corpus)",        f"  BM25 build  ({len(all_chunks):>7,} docs)",   "s"),
    ("BM25 8 queries (full corpus)",    f"  BM25 query  ({len(QUERIES)} queries × {len(corpus_docs):,} docs)", "s"),
    ("Cache COLD (load+build)",         "  NEW: cold (first call — builds cache)",       "s"),
    ("Cache WARM (subsequent)",         "  NEW: warm (subsequent calls — cache hit)",    "s"),
    ("Context build (5 chunks)",        "  build_context  (5 chunks)",                   "s"),
    ("E2E query (warm, excl LLM)",      "E2E: BM25 search + context + prompt build (warm)", "s"),
]
print(f"  {'Step':<45} {'avg':>8}  {'mem':>7}")
print(f"  {'─'*45}  {'─'*7}  {'─'*7}")
for label, key, unit in rows:
    if key in RESULTS:
        r = RESULTS[key]
        flag = "  🐢" if r["avg"] > 10 else ("  ⚠️ " if r["avg"] > 3 else "")
        print(f"  {label:<45} {r['avg']:>6.3f}{unit}  {r['mem']:>6.1f}MB{flag}")

print(f"\n  Key estimates for real deployment:")
embed_min = len(all_chunks)/8/60
print(f"  • First-time embed (CPU nomic-embed-text):  ~{embed_min:.0f} min  ({len(all_chunks):,} chunks @ 8/s)")
print(f"  • First-time embed (GPU RTX 3060):          ~{len(all_chunks)/80/60:.0f} min")
print(f"  • Incremental re-embed (500 files changed): ~{len(files3)*6/8/60:.1f} min  (est {len(files3)*6} chunks)")
print(f"  • LLM inference per query:                  5-30s  (qwen2.5-coder:7b @ 4GB VRAM)")
print(f"  • Our overhead per query (warm):            <{RESULTS['E2E: BM25 search + context + prompt build (warm)']['avg']*1000:.0f}ms")
print(f"\n{'═'*W}\n")
