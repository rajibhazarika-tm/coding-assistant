# retriever/hybrid_search.py
"""
Hybrid retrieval: vector search plus full-corpus BM25 reranking.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from config.settings import (
    BM25_WEIGHT,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    INDEX_DIR,
    OLLAMA_BASE_URL,
    TOP_K_CHUNKS,
    VECTOR_WEIGHT,
)


@dataclass
class RetrievedChunk:
    """A retrieved code chunk with its relevance score."""

    id: str
    content: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    chunk_type: str
    name: str
    context_header: str
    score: float

    @property
    def display_header(self) -> str:
        loc = f"{self.file_path}:{self.start_line}-{self.end_line}"
        if self.name:
            return f"{loc} [{self.chunk_type}: {self.name}]"
        return f"{loc} [{self.chunk_type}]"


def _embed_query(query: str) -> list[float]:
    """Embed the user query for vector search."""
    import requests

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": query},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def _tokenize(text: str) -> list[str]:
    """Split code-ish text into lowercase tokens, including camelCase/snake_case."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25:
    """In-memory BM25 scorer for a fixed corpus."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = [_tokenize(doc) for doc in corpus]
        self.N = len(corpus)
        self.avgdl = sum(len(t) for t in self.corpus_tokens) / max(self.N, 1)
        self.df: dict[str, int] = {}
        self.tf: list[dict[str, int]] = []

        for tokens in self.corpus_tokens:
            freq: dict[str, int] = {}
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
            self.tf.append(freq)
            for token in set(tokens):
                self.df[token] = self.df.get(token, 0) + 1

    def score(self, query: str, doc_id: int) -> float:
        query_tokens = _tokenize(query)
        doc_tf = self.tf[doc_id]
        dl = len(self.corpus_tokens[doc_id])
        score = 0.0

        for term in query_tokens:
            if term not in doc_tf:
                continue
            df = self.df.get(term, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf = doc_tf[term]
            norm_tf = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            )
            score += idf * norm_tf

        return score


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score."""
    return 1.0 / (k + rank)


def _collection():
    import chromadb  # type: ignore
    from chromadb.config import Settings  # type: ignore

    client = chromadb.PersistentClient(
        path=str(INDEX_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(CHROMA_COLLECTION)


# FIX 4: cache the BM25 corpus so it isn't rebuilt from scratch on every query.
# Key = (where_filter_repr, collection_count). Invalidated automatically when
# new chunks are indexed (count changes).
_corpus_cache: dict[str, tuple[int, list, BM25]] = {}


def _get_cached_corpus(
    collection,
    where_filter: Optional[dict],
    file_filter: Optional[str],
) -> list[tuple[str, str, dict]]:
    """
    Return (chunk_id, doc, meta) triples for the filtered corpus, rebuilding
    the BM25 index only when the collection size has changed.
    """
    count = collection.count()
    cache_key = f"{repr(where_filter)}|{file_filter or ''}"

    if cache_key in _corpus_cache:
        cached_count, cached_corpus, _ = _corpus_cache[cache_key]
        if cached_count == count:
            return cached_corpus  # cache hit

    all_data = collection.get(where=where_filter, include=["documents", "metadatas"])
    corpus = [
        (chunk_id, doc, meta or {})
        for chunk_id, doc, meta in zip(
            all_data.get("ids", []),
            all_data.get("documents", []),
            all_data.get("metadatas", []),
        )
        if _meta_matches(meta or {}, file_filter)
    ]
    bm25 = BM25([doc for _, doc, _ in corpus])
    _corpus_cache[cache_key] = (count, corpus, bm25)
    return corpus


def _get_cached_bm25(
    collection,
    where_filter: Optional[dict],
    file_filter: Optional[str],
) -> "BM25":
    """Return the cached BM25 index, building it if needed."""
    count = collection.count()
    cache_key = f"{repr(where_filter)}|{file_filter or ''}"
    if cache_key not in _corpus_cache or _corpus_cache[cache_key][0] != count:
        _get_cached_corpus(collection, where_filter, file_filter)
    return _corpus_cache[cache_key][2]


def _metadata_filter(language_filter: Optional[str], repo_filter: Optional[str]) -> Optional[dict]:
    filters = []
    if language_filter:
        filters.append({"language": language_filter})
    if repo_filter:
        filters.append({"repo_id": repo_filter})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def _meta_matches(meta: dict, file_filter: Optional[str]) -> bool:
    if not file_filter:
        return True
    wanted = file_filter.replace("\\", "/")
    actual = str(meta.get("file_path", "")).replace("\\", "/")
    return wanted in actual


def _build_result(chunk_id: str, doc: str, meta: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=chunk_id,
        content=doc,
        file_path=meta.get("file_path", ""),
        language=meta.get("language", ""),
        start_line=meta.get("start_line", 0),
        end_line=meta.get("end_line", 0),
        chunk_type=meta.get("chunk_type", "block"),
        name=meta.get("name", ""),
        context_header=meta.get("context_header", ""),
        score=score,
    )


def retrieve(
    query: str,
    top_k: int = TOP_K_CHUNKS,
    language_filter: Optional[str] = None,
    file_filter: Optional[str] = None,
    repo_filter: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant code chunks for a query.

    File-scoped retrieval uses BM25 only (exact symbol/path matching).
    General retrieval fuses vector + BM25 via RRF.
    BM25 corpus is cached in-process and only rebuilt when the index changes.
    """
    try:
        collection = _collection()
    except Exception:
        print("Index not found. Run `python -m cli.main index --path <repo>` first.")
        return []

    if collection.count() == 0:
        return []

    where_filter = _metadata_filter(language_filter, repo_filter)
    corpus = _get_cached_corpus(collection, where_filter, file_filter)
    if not corpus:
        return []

    bm25 = _get_cached_bm25(collection, where_filter, file_filter)
    bm25_scores = [bm25.score(query, i) for i in range(len(corpus))]
    fetch_k = min(max(top_k * 8, 20), len(corpus))
    bm25_ranks = sorted(range(len(corpus)), key=lambda i: -bm25_scores[i])[:fetch_k]

    if file_filter:
        return [
            _build_result(corpus[idx][0], corpus[idx][1], corpus[idx][2], _rrf_score(rank))
            for rank, idx in enumerate(bm25_ranks[:top_k])
        ]

    rrf_scores: dict[str, float] = {}

    try:
        kwargs: dict = {
            "query_embeddings": [_embed_query(query)],
            "n_results": fetch_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter
        vector_results = collection.query(**kwargs)
        vector_ids = vector_results.get("ids", [[]])[0]
        for rank, chunk_id in enumerate(vector_ids):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + VECTOR_WEIGHT * _rrf_score(rank)
    except Exception:
        pass  # Lexical-only mode when Ollama embeddings are unavailable

    for rank, idx in enumerate(bm25_ranks):
        chunk_id = corpus[idx][0]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + BM25_WEIGHT * _rrf_score(rank)

    corpus_by_id = {chunk_id: (doc, meta) for chunk_id, doc, meta in corpus}
    ranked_ids = [
        cid for cid in sorted(rrf_scores, key=lambda x: -rrf_scores[x])
        if cid in corpus_by_id
    ]

    return [
        _build_result(cid, corpus_by_id[cid][0], corpus_by_id[cid][1], rrf_scores[cid])
        for cid in ranked_ids[:top_k]
    ]
