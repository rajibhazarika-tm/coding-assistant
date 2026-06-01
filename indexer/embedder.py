# indexer/embedder.py
"""
Embeds code chunks and stores them in ChromaDB (local vector store).

Uses nomic-embed-text via Ollama for embeddings:
- Runs on CPU — doesn't compete with the LLM for VRAM
- 768-dim embeddings, strong on code
- ~270MB model, fast enough for incremental indexing

ChromaDB is used as the vector store:
- Runs fully in-process, no server needed
- Persists to disk automatically
- Supports metadata filtering for targeted retrieval
"""

from __future__ import annotations
import time
import requests  # FIX 1: was accidentally placed inside the docstring of _embed_texts

from config.settings import (
    OLLAMA_BASE_URL, EMBED_MODEL, INDEX_DIR, CHROMA_COLLECTION
)
from indexer.chunker import CodeChunk


def _get_chroma_client():
    """Get or create a persistent ChromaDB client."""
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore

    return chromadb.PersistentClient(
        path=str(INDEX_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _embed_one(text: str, retries: int = 3) -> list[float]:
    """
    Embed a single text via Ollama. Retries up to `retries` times with
    exponential backoff so transient timeouts (common on first model load)
    don't silently drop chunks.
    FIX 2: replaces dead double-batching loop; FIX 3: adds per-text retry.
    """
    for attempt in range(retries):
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt  # 1s, 2s, 4s …
            print(f"   ↻ embed retry {attempt + 1}/{retries - 1} after error: {exc} (waiting {wait}s)")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # satisfy type checker


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, one request per text (Ollama is a single-text API)."""
    return [_embed_one(text) for text in texts]


def index_chunks(chunks: list[CodeChunk], show_progress: bool = True) -> int:
    """
    Embed and store chunks in ChromaDB.

    Returns number of chunks indexed.
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
    batch_size = 10

    for i in range(0, total, batch_size):
        batch = chunks[i: i + batch_size]

        ids = [c.id for c in batch]
        texts = [c.embedding_text for c in batch]
        metadatas = [
            {
                "file_path": c.file_path,
                "language": c.language,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "chunk_type": c.chunk_type,
                "name": c.name or "",
                "context_header": c.context_header,
                "repo_id": c.repo_id,
            }
            for c in batch
        ]
        documents = [c.content for c in batch]

        try:
            embeddings = _embed_texts(texts)
            # Upsert handles both new and updated chunks
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            indexed += len(batch)
        except Exception as e:
            print(f"   ⚠️  Batch {i//batch_size + 1} failed: {e}")

        if show_progress and (i + batch_size) % 50 == 0:
            print(f"   📊 {min(i + batch_size, total)}/{total} chunks indexed...")

    return indexed


def delete_chunks_for_files(file_paths: list[str], repo_id: str = "") -> int:
    """Remove all chunks belonging to deleted files."""
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
            where = {"file_path": fp}
            if repo_id:
                where = {"$and": [{"file_path": fp}, {"repo_id": repo_id}]}
            results = collection.get(where=where)
            if results["ids"]:
                collection.delete(ids=results["ids"])
                deleted += len(results["ids"])
        except Exception:
            pass

    return deleted


def delete_chunks_for_repo(repo_id: str) -> int:
    """Remove all chunks for one repository."""
    if not repo_id:
        return 0

    client = _get_chroma_client()
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception:
        return 0

    try:
        results = collection.get(where={"repo_id": repo_id})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
        return len(ids)
    except Exception:
        return 0


def get_collection_stats() -> dict:
    """Return basic stats about the index."""
    try:
        client = _get_chroma_client()
        collection = client.get_collection(CHROMA_COLLECTION)
        count = collection.count()
        return {"total_chunks": count, "status": "ok"}
    except Exception as e:
        return {"total_chunks": 0, "status": f"error: {e}"}
