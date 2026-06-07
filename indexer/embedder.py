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
    from config.settings import INDEX_DIR as _idx_dir
    return chromadb.PersistentClient(
        path=str(_idx_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection():
    """Get or create the collection, reading CHROMA_COLLECTION at call time."""
    from config.settings import CHROMA_COLLECTION as _col_name
    client = _get_chroma_client()
    return client.get_or_create_collection(
        name=_col_name,
        metadata={"hnsw:space": "cosine"},
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


# Global cancel flag — set to True to stop embedding workers cleanly.
# Checked by _embed_one on each retry and by _embed_parallel's result loop.
_cancel_requested = False


def request_cancel() -> None:
    """Signal all embedding workers to stop after their current request."""
    global _cancel_requested
    _cancel_requested = True


def reset_cancel() -> None:
    """Clear the cancel flag before starting a new indexing run."""
    global _cancel_requested
    _cancel_requested = False


def _embed_parallel(
    texts: list[str],
    workers: int = EMBED_WORKERS,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list[list[float]]:
    """
    Embed texts in parallel using a thread pool.

    Ctrl+C / KeyboardInterrupt sets _cancel_requested=True, which causes:
    1. The result-collection loop to break immediately
    2. pool.shutdown(wait=False, cancel_futures=True) to drop queued work
    3. index_chunks to save what it has so far (resume picks up remainder)

    Returns however many embeddings completed before cancellation.
    Incomplete slots remain as empty lists — index_chunks filters them out.
    """
    global _cancel_requested
    results: list[list[float]] = [[] for _ in texts]
    done = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(_embed_one, text): idx for idx, text in enumerate(texts)}

    try:
        for future in as_completed(futures):
            if _cancel_requested:
                break
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception:
                results[idx] = []
            done += 1
            if on_progress:
                on_progress(done, len(texts))
    except KeyboardInterrupt:
        _cancel_requested = True
    finally:
        # cancel_futures=True drops anything not yet started (Python 3.9+)
        # wait=False means we don't block waiting for in-flight requests
        pool.shutdown(wait=False, cancel_futures=True)

    if _cancel_requested and done < len(texts):
        print(f"\n   ⚠️  Cancelled after {done}/{len(texts)} embeddings."
              f" Progress saved — restart to resume from here.")

    return results


def _get_already_indexed_ids(collection, chunk_ids: list[str]) -> set[str]:
    """
    Return the subset of chunk_ids that are already in ChromaDB.
    Checked in batches to avoid hitting ChromaDB's get() size limits.
    """
    existing: set[str] = set()
    batch_size = 5000  # ChromaDB handles up to ~5k IDs per get()
    for i in range(0, len(chunk_ids), batch_size):
        batch_ids = chunk_ids[i: i + batch_size]
        try:
            result = collection.get(ids=batch_ids, include=[])
            existing.update(result.get("ids", []))
        except Exception:
            pass
    return existing


def index_chunks(
    chunks: list[CodeChunk],
    show_progress: bool = True,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
    resume: bool = True,
) -> int:
    """
    Embed chunks in parallel and store in ChromaDB in large batches.

    resume=True (default): skips chunks whose IDs are already in ChromaDB,
    so interrupted indexing jobs can be continued without re-embedding
    everything from scratch.

    on_progress(indexed, total, eta_str) — called after each ChromaDB upsert.
    """
    if not chunks:
        return 0

    collection = _get_collection()

    # ── Resume: skip chunks already in the index ──────────────────────────────
    if resume:
        all_ids = [c.id for c in chunks]
        already_done = _get_already_indexed_ids(collection, all_ids)
        if already_done:
            chunks = [c for c in chunks if c.id not in already_done]
            if show_progress:
                print(f"   ⏭️  Skipping {len(already_done)} already-indexed chunks"
                      f" — {len(chunks)} remaining")
        if not chunks:
            if show_progress:
                print("   ✅ All chunks already indexed — nothing to do")
            return 0

    total = len(chunks)
    indexed = 0
    t_start = time.perf_counter()
    # Rolling rate tracker: use last 200 completions for accurate ETA
    # (avoids the ETA explosion at start when rate is measured over too few samples)
    _recent_times: list[float] = []

    # ── Step 1: embed ALL chunks in parallel ──────────────────────────────────
    if show_progress:
        print(f"   🔀 Embedding {total} chunks"
              f" with {EMBED_WORKERS} parallel workers...")

    embed_done = [0]
    def _embed_progress(done, n):
        embed_done[0] = done
        _recent_times.append(time.perf_counter())
        # Compute rate from last 50 completions (not from t=0)
        if len(_recent_times) > 200:
            _recent_times.pop(0)
        if len(_recent_times) >= 2 and show_progress and done % 20 == 0:
            window_s = _recent_times[-1] - _recent_times[0]
            window_n = len(_recent_times) - 1
            rate = window_n / window_s if window_s > 0 else 0
            eta_s = (n - done) / rate if rate > 0 else 0
            eta_str = (f"{eta_s/3600:.1f}h" if eta_s > 3600
                       else f"{eta_s/60:.0f}m" if eta_s > 60
                       else f"{eta_s:.0f}s")
            print(f"   ⚡ {done}/{n}  {rate:.1f}/s  ETA {eta_str}   ", end="\r")
            if on_progress:
                on_progress(done, n, eta_str)

    reset_cancel()  # clear any previous cancel before starting

    all_texts = [c.embedding_text for c in chunks]
    try:
        all_embeddings = _embed_parallel(all_texts, on_progress=_embed_progress)
    except Exception as e:
        print(f"\n   ❌ Embedding failed: {e}")
        return 0

    # Filter out slots that didn't complete (cancelled or errored)
    completed = [(c, emb) for c, emb in zip(chunks, all_embeddings) if emb]
    n_completed = len(completed)
    n_skipped   = total - n_completed

    if show_progress:
        elapsed = time.perf_counter() - t_start
        rate = n_completed / elapsed if elapsed > 0 else 0
        status = "Cancelled" if _cancel_requested else "Embedded"
        print(f"\n   ✅ {status} {n_completed}/{total} chunks in {elapsed:.1f}s  ({rate:.1f}/s)")
        if n_skipped:
            print(f"   ℹ️  {n_skipped} chunks skipped — restart to resume (they'll be detected automatically)")

    if not completed:
        return 0

    # ── Step 2: upsert into ChromaDB whatever completed ───────────────────────
    t_db = time.perf_counter()
    for i in range(0, n_completed, CHROMA_BATCH_SIZE):
        batch_pairs = completed[i: i + CHROMA_BATCH_SIZE]
        batch = [c for c, _ in batch_pairs]
        embs  = [e for _, e in batch_pairs]
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
