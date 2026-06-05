# indexer/embedder.py
"""
Embeds code chunks and stores them in ChromaDB.

Performance improvements:
- PARALLEL embedding via ThreadPoolExecutor (4–8 workers hit Ollama concurrently)
- LARGE ChromaDB upsert batches (128 chunks vs old 10) — far fewer disk syncs
- Progress callback for UI streaming
- Estimated time remaining shown during indexing
"""
from __future__ import annotations
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

from config.settings import (
    OLLAMA_BASE_URL, EMBED_MODEL, INDEX_DIR, CHROMA_COLLECTION,
    EMBED_WORKERS, CHROMA_BATCH_SIZE, EMBED_NUM_CTX,
)
from indexer.chunker import CodeChunk


def _get_chroma_client():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=str(INDEX_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _embed_one(text: str, retries: int = 3) -> list[float]:
    """Embed a single text with exponential-backoff retry.

    Three fixes for intermittent Ollama 500 errors with 50-line chunks:

    FIX A — pass num_ctx=8192 explicitly: Ollama's default num_ctx for
      nomic-embed-text is 2048 tokens. Long chunks with verbose Java/Kotlin
      identifiers can silently overflow that, causing Ollama to kill the runner
      and return HTTP 500. Setting num_ctx=8192 unlocks the model's full capacity.

    FIX B — retry on HTTP 500/503: these are transient Ollama overload errors
      (previously only network errors were retried, not HTTP error responses).

    FIX C — longer timeout for large chunks under memory pressure: 30s was too
      tight when 4 workers compete for the embedding model simultaneously.
    """
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={
                    "model": EMBED_MODEL,
                    "prompt": text,
                    "options": {
                        "num_ctx": EMBED_NUM_CTX,  # FIX A
                    },
                },
                timeout=60,  # FIX C: was 30s
            )
            # FIX B: retry on 500/503 (transient Ollama overload), not just network errors
            if r.status_code in (500, 503):
                raise requests.HTTPError(f"Ollama returned {r.status_code}: {r.text[:200]}", response=r)
            r.raise_for_status()
            return r.json()["embedding"]
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"   ↻ embed retry {attempt + 1}/{retries - 1} — {exc} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _embed_parallel(
    texts: list[str],
    workers: int = EMBED_WORKERS,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[list[float]]:
    """
    Embed texts in parallel using a thread pool.

    Ollama handles concurrent /api/embeddings requests fine — each runs
    in its own goroutine on the server side. With 4–8 workers we saturate
    the CPU embedding model without overloading it.

    Returns embeddings in the same order as input texts.
    """
    results: list[list[float]] = [[] for _ in texts]
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_embed_one, text): idx for idx, text in enumerate(texts)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            done += 1
            if on_progress:
                on_progress(done, len(texts))

    return results


def index_chunks(
    chunks: list[CodeChunk],
    show_progress: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> int:
    """
    Embed chunks in parallel and store in ChromaDB in large batches.

    on_progress(indexed, total, eta_str) — called after each ChromaDB upsert,
    useful for streaming progress to a UI.
    """
    if not chunks:
        return 0

    client = _get_chroma_client()
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(chunks)
    indexed = 0
    t_start = time.perf_counter()

    # ── Step 1: embed ALL chunks in parallel ──────────────────────────────────
    if show_progress:
        print(f"   🔀 Embedding {total} chunks with {EMBED_WORKERS} parallel workers...")

    embed_done = [0]
    def _embed_progress(done, n):
        embed_done[0] = done
        if show_progress and done % 50 == 0:
            elapsed = time.perf_counter() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            print(f"   ⚡ Embedding {done}/{n}  {rate:.0f}/s  ETA {eta:.0f}s", end="\r")

    all_texts = [c.embedding_text for c in chunks]
    try:
        all_embeddings = _embed_parallel(all_texts, on_progress=_embed_progress)
    except Exception as e:
        print(f"\n   ❌ Embedding failed: {e}")
        return 0

    if show_progress:
        elapsed = time.perf_counter() - t_start
        print(f"\n   ✅ Embedded {total} chunks in {elapsed:.1f}s  ({total/elapsed:.0f}/s)")

    # ── Step 2: upsert into ChromaDB in large batches ─────────────────────────
    t_db = time.perf_counter()
    for i in range(0, total, CHROMA_BATCH_SIZE):
        batch = chunks[i: i + CHROMA_BATCH_SIZE]
        embs  = all_embeddings[i: i + CHROMA_BATCH_SIZE]
        try:
            collection.upsert(
                ids=[c.id for c in batch],
                embeddings=embs,
                documents=[c.content for c in batch],
                metadatas=[{
                    "file_path":    c.file_path,
                    "language":     c.language,
                    "start_line":   c.start_line,
                    "end_line":     c.end_line,
                    "chunk_type":   c.chunk_type,
                    "name":         c.name or "",
                    "context_header": c.context_header,
                    "repo_id":      c.repo_id,
                } for c in batch],
            )
            indexed += len(batch)
        except Exception as e:
            print(f"   ⚠️  DB batch {i//CHROMA_BATCH_SIZE + 1} failed: {e}")

        if on_progress:
            elapsed = time.perf_counter() - t_start
            rate = indexed / elapsed if elapsed > 0 else 0
            eta  = (total - indexed) / rate if rate > 0 else 0
            on_progress(indexed, total, f"{eta:.0f}s")

        if show_progress and indexed % 500 == 0:
            db_elapsed = time.perf_counter() - t_db
            print(f"   💾 Stored {indexed}/{total} chunks  ({indexed/db_elapsed:.0f}/s DB write)", end="\r")

    if show_progress:
        total_elapsed = time.perf_counter() - t_start
        print(f"\n   ✅ Stored  {indexed} chunks  total time {total_elapsed:.1f}s")

    return indexed


def delete_chunks_for_files(file_paths: list[str], repo_id: str = "") -> int:
    if not file_paths:
        return 0
    client = _get_chroma_client()
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        return 0
    deleted = 0
    for fp in file_paths:
        try:
            where = {"$and": [{"file_path": fp}, {"repo_id": repo_id}]} if repo_id else {"file_path": fp}
            results = collection.get(where=where)
            if results["ids"]:
                collection.delete(ids=results["ids"])
                deleted += len(results["ids"])
        except Exception:
            pass
    return deleted


def delete_chunks_for_repo(repo_id: str) -> int:
    if not repo_id:
        return 0
    client = _get_chroma_client()
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
        results = collection.get(where={"repo_id": repo_id})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def get_collection_stats() -> dict:
    try:
        client = _get_chroma_client()
        collection = client.get_collection(CHROMA_COLLECTION)
        return {"total_chunks": collection.count(), "status": "ok"}
    except Exception as e:
        return {"total_chunks": 0, "status": f"error: {e}"}
