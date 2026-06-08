# retriever/rag.py
"""
Full RAG pipeline for the coding assistant.

Implements the complete Retrieval-Augmented Generation loop:

  ┌─────────────────────────────────────────────────────────┐
  │  1. Query Understanding                                  │
  │     - Conversation-aware reformulation                   │
  │     - Multi-part question decomposition                  │
  │     - Hypothetical Document Embedding (HDE)              │
  ├─────────────────────────────────────────────────────────┤
  │  2. Multi-Strategy Retrieval                             │
  │     - HDE vector search (best for "explain X" queries)   │
  │     - Original query vector search                       │
  │     - BM25 keyword search (exact symbol names)           │
  │     - Grep/ripgrep (literal matches)                     │
  ├─────────────────────────────────────────────────────────┤
  │  3. Fusion & Reranking                                   │
  │     - Reciprocal Rank Fusion across all strategies       │
  │     - Diversity penalty (avoid same-file dominance)      │
  │     - Test-file penalty for non-debug tasks              │
  ├─────────────────────────────────────────────────────────┤
  │  4. Contextual Compression                               │
  │     - LLM extracts only relevant lines from each chunk   │
  │     - Cuts token usage by ~60% vs raw chunk              │
  │     - Falls back to full chunk if compression fails      │
  ├─────────────────────────────────────────────────────────┤
  │  5. Generation                                           │
  │     - Task-specific prompt (explain/review/generate/debug│
  │     - Streaming response                                 │
  ├─────────────────────────────────────────────────────────┤
  │  6. Faithfulness Check (optional)                        │
  │     - Verify answer uses retrieved context               │
  │     - Flag potential hallucinations                      │
  └─────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import requests

from config.settings import (
    OLLAMA_BASE_URL, MODEL, MAX_CONTEXT_TOKENS,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)
from retriever.hybrid_search import RetrievedChunk, retrieve as _hybrid_retrieve
from retriever.query_correction import correct_query, CorrectedQuery
from retriever.pipeline import (
    run_pipeline, grep_search, grep_matches_to_chunks,
    QueryPlan, analyse_query, merge_results, rerank,
)
from retriever.hybrid_search import _rrf_score
from retriever.context_builder import build_context, _rough_token_count
from assistant.prompts import build_prompt, build_no_context_prompt
from assistant.llm import stream_response, _trim_history_to_budget


# ── 1. Query Understanding ────────────────────────────────────────────────────

@dataclass
class RAGQuery:
    """Fully analysed query ready for multi-strategy retrieval."""
    original: str                    # what the user typed
    corrected: str                   # after spelling/abbreviation correction
    reformulated: str                # conversation-aware rewrite
    sub_queries: list[str]           # decomposed sub-questions
    hypothetical_answer: str         # HDE: fake answer for embedding
    plan: QueryPlan                  # grep terms, file hints, task type
    is_multi_hop: bool               # needs info from multiple files
    is_conversational: bool          # refers to previous turns
    corrections: list[str] = None    # list of changes made by corrector
    grep_terms: list[str] = None     # all grep search terms (LLM + corrector symbols)


_QUERY_UNDERSTANDING_SYSTEM = """You are a query analyser for a code search system. Given a user question, output a JSON object with these EXACT keys:

"search_terms": list of 1-6 EXACT code identifiers to grep for. Rules:
  - Use camelCase/PascalCase/snake_case as they appear verbatim in code
  - Include method names: "getUserById", "processPayment", "validateToken"
  - Include class names: "OrderService", "AuthController", "PaymentGateway"
  - Include annotation names: "@Transactional", "@Autowired", "@RestController"
  - Include exception/error names: "NullPointerException", "AuthenticationException"
  - Include config keys if relevant: "spring.datasource.url", "jwt.secret"
  - Do NOT use plain English like "user", "order", "get" — only identifiers that appear verbatim in source files
  - If unsure, prefer longer specific identifiers over short generic words

"reformulated": rewrite the question to be self-contained, clear, technical (resolve "it"/"this"/"that" from history)

"sub_queries": list of 1-3 focused sub-questions if complex, else ["<same as reformulated>"]

"hypothetical_answer": 3-5 line realistic code snippet that would answer the question — embedded to find similar real code

"file_hints": list of 0-3 partial filename patterns likely containing the answer (e.g. "AuthService", "OrderController")

"task": one of: explain, review, generate, debug, general

"is_multi_hop": true if the question requires info from multiple files/services

"reasoning": one sentence explaining your search strategy

Output ONLY valid JSON. No markdown fences. No text before or after the JSON."""


def understand_query(
    question: str,
    history: Optional[list[dict]] = None,
) -> RAGQuery:
    """
    Step 1: Deep query understanding using the LLM.

    First runs the fast (pure-Python) query correction pipeline:
      - spelling fixes ("authetication" → "authentication")
      - camelCase splitting ("getUserById" → "get user by id")
      - abbreviation expansion ("svc" → "service")
      - symbol/operator normalisation ("user.getName()" → "user getName")

    Then passes the corrected query to the LLM for deeper analysis
    (reformulation, HDE, multi-hop detection). This two-step approach
    means the LLM sees clean, unambiguous input.
    """
    # Run fast correction first (no LLM call, <1ms)
    correction = correct_query(question)
    working_question = correction.corrected  # feed corrected text to LLM

    history_text = ""
    if history:
        recent = history[-6:]  # last 3 turns
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:200]}"
            for m in recent
        )

    prompt = working_question
    if history_text:
        prompt = f"Conversation history:\n{history_text}\n\nNew question: {working_question}"

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": _QUERY_UNDERSTANDING_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 400},
            },
            timeout=20,
        )
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)

        plan = QueryPlan(
            search_terms=data.get("search_terms", [])[:5],
            semantic_query=data.get("reformulated", question),
            file_hints=data.get("file_hints", [])[:3],
            task=data.get("task", "general"),
            reasoning=data.get("reasoning", ""),
        )
        # Build combined grep terms: LLM terms + raw symbols from corrector
        # Raw symbols (camelCase, snake_case) are the most precise grep targets
        llm_terms = data.get("search_terms", [])[:5]
        raw_symbols = correction.symbols_found  # e.g. ["getUserById", "OrderService"]
        combined_terms = list(dict.fromkeys(llm_terms + raw_symbols))  # dedup, preserve order

        return RAGQuery(
            original=question,
            corrected=working_question,
            reformulated=data.get("reformulated", working_question),
            sub_queries=data.get("sub_queries", [working_question]),
            hypothetical_answer=data.get("hypothetical_answer", ""),
            plan=plan,
            is_multi_hop=data.get("is_multi_hop", False),
            is_conversational=bool(history),
            corrections=correction.corrections,
            grep_terms=combined_terms,
        )
    except Exception:
        # Graceful fallback — use corrector symbols as grep terms
        words = [w for w in working_question.split() if len(w) > 4]
        raw_symbols = correction.symbols_found
        combined_terms = list(dict.fromkeys(raw_symbols + words[:3]))
        plan = QueryPlan(
            search_terms=combined_terms[:5],
            semantic_query=working_question,
            file_hints=[],
            task="general",
            reasoning="(fallback — query analysis unavailable)",
        )
        return RAGQuery(
            original=question,
            corrected=working_question,
            reformulated=working_question,
            sub_queries=[working_question],
            hypothetical_answer="",
            plan=plan,
            is_multi_hop=False,
            is_conversational=bool(history),
            corrections=correction.corrections,
            grep_terms=combined_terms,
        )


# ── 2. Multi-Strategy Retrieval ───────────────────────────────────────────────

def _embed_text(text: str) -> list[float]:
    """Embed any text for HDE or query vector search."""
    from config.settings import EMBED_MODEL, EMBED_NUM_CTX, EMBED_QUERY_MAX_CHARS
    safe = text[:EMBED_QUERY_MAX_CHARS]
    for attempt in range(3):
        try:
            r = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": safe,
                      "options": {"num_ctx": EMBED_NUM_CTX}},
                timeout=60,
            )
            if r.status_code in (500, 503):
                raise requests.HTTPError(str(r.status_code))
            r.raise_for_status()
            return r.json()["embedding"]
        except (requests.RequestException, OSError):
            if attempt == 2:
                return []
            time.sleep(2 ** attempt)
    return []


def multi_strategy_retrieve(
    raq: RAGQuery,
    top_k: int = 5,
    repo_root: Optional[str] = None,
    language_filter: Optional[str] = None,
    repo_filter: Optional[str] = None,
) -> list[RetrievedChunk]:
    """
    Step 2: Run multiple retrieval strategies in parallel and fuse results.

    Strategies:
    A) HDE vector search  — embed the hypothetical answer, not the question
    B) Query vector search — embed the reformulated question
    C) BM25 keyword search — from the cached corpus
    D) Grep exact search   — for literal symbol names

    Returns merged, deduplicated, scored candidates for reranking.
    """
    from retriever.hybrid_search import (
        _collection, _get_cached_corpus, _get_cached_bm25,
        _metadata_filter, BM25, _rrf_score as rrf,
        _build_result, _meta_matches_filter,
    )

    all_results: dict[str, tuple[RetrievedChunk, float]] = {}  # id → (chunk, score)

    def _add(chunk: RetrievedChunk, weight: float, rank: int):
        score = weight * rrf(rank)
        if chunk.id in all_results:
            existing, existing_score = all_results[chunk.id]
            all_results[chunk.id] = (existing, existing_score + score)
        else:
            all_results[chunk.id] = (chunk, score)

    try:
        col = _collection()
    except Exception:
        return []

    where_filter = _metadata_filter(language_filter, repo_filter)
    corpus = _get_cached_corpus(col, where_filter, None)
    if not corpus:
        return []

    count = col.count()
    fetch_k = min(top_k * 6, count, 100)

    # Strategy A: HDE — embed the hypothetical answer
    if raq.hypothetical_answer:
        hde_vec = _embed_text(raq.hypothetical_answer)
        if hde_vec:
            try:
                kw: dict = {
                    "query_embeddings": [hde_vec],
                    "n_results": fetch_k,
                    "include": ["documents", "metadatas", "distances"],
                }
                if where_filter:
                    kw["where"] = where_filter
                hde_res = col.query(**kw)
                for rank, (cid, doc, meta, _dist) in enumerate(zip(
                    hde_res["ids"][0], hde_res["documents"][0],
                    hde_res["metadatas"][0], hde_res["distances"][0],
                )):
                    _add(_build_result(cid, doc, meta or {}, 0), weight=1.5, rank=rank)
            except Exception:
                pass

    # Strategy B: Reformulated query vector search
    q_vec = _embed_text(raq.reformulated)
    if q_vec:
        try:
            kw2: dict = {
                "query_embeddings": [q_vec],
                "n_results": fetch_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kw2["where"] = where_filter
            q_res = col.query(**kw2)
            for rank, (cid, doc, meta, _dist) in enumerate(zip(
                q_res["ids"][0], q_res["documents"][0],
                q_res["metadatas"][0], q_res["distances"][0],
            )):
                _add(_build_result(cid, doc, meta or {}, 0), weight=1.0, rank=rank)
        except Exception:
            pass

    # Strategy C: BM25 over full corpus
    bm25 = _get_cached_bm25(col, where_filter, None)
    if bm25 and corpus:
        docs = [doc for _, doc, _ in corpus]
        bm25_scores = [bm25.score(raq.reformulated, i) for i in range(len(docs))]
        bm25_ranked = sorted(range(len(docs)), key=lambda i: -bm25_scores[i])[:fetch_k]
        for rank, idx in enumerate(bm25_ranked):
            cid, doc, meta = corpus[idx]
            _add(_build_result(cid, doc, meta, 0), weight=0.8, rank=rank)

    # Strategy D: Grep using combined terms (LLM + raw corrector symbols)
    # raq.grep_terms is built in understand_query() by merging:
    #   - LLM-extracted search_terms (e.g. "authenticate", "validateToken")
    #   - Raw symbols from corrector (e.g. "getUserById", "OrderService")
    # Raw symbols are the best grep targets: exact camelCase that appears
    # verbatim in source code, before any splitting or normalisation.
    if repo_root:
        grep_terms = [t for t in (raq.grep_terms or raq.plan.search_terms or []) if t.strip()]
        if grep_terms:
            grep_raw = grep_search(grep_terms, repo_root=repo_root)
            grep_chunks = grep_matches_to_chunks(grep_raw, repo_root=repo_root)
            for rank, chunk in enumerate(grep_chunks):
                _add(chunk, weight=1.2, rank=rank)  # exact match bonus

    # Multi-hop: run sub-queries and merge their results
    if raq.is_multi_hop and len(raq.sub_queries) > 1:
        for sq in raq.sub_queries[1:]:
            sq_vec = _embed_text(sq)
            if sq_vec:
                try:
                    kw3: dict = {
                        "query_embeddings": [sq_vec],
                        "n_results": min(fetch_k // 2, count, 50),
                        "include": ["documents", "metadatas", "distances"],
                    }
                    if where_filter:
                        kw3["where"] = where_filter
                    sq_res = col.query(**kw3)
                    for rank, (cid, doc, meta, _) in enumerate(zip(
                        sq_res["ids"][0], sq_res["documents"][0],
                        sq_res["metadatas"][0], sq_res["distances"][0],
                    )):
                        _add(_build_result(cid, doc, meta or {}, 0), weight=0.7, rank=rank)
                except Exception:
                    pass

    # Collect and update scores
    candidates = []
    for chunk, score in all_results.values():
        chunk.score = round(score, 4)
        candidates.append(chunk)

    # Rerank with diversity + task awareness
    return rerank(candidates, raq.original, raq.plan, top_k=top_k * 2)


# ── 4. Contextual Compression ─────────────────────────────────────────────────

_COMPRESS_SYSTEM = """You are a code excerpt extractor. Given a code chunk and a question, 
extract ONLY the lines directly relevant to answering the question.
Return the extracted code as-is (no explanation, no markdown fences).
If the entire chunk is relevant, return it unchanged.
If nothing is relevant, return an empty string."""


def compress_chunk(chunk: RetrievedChunk, question: str) -> RetrievedChunk:
    """
    Step 4: Use LLM to extract only relevant lines from a chunk.
    Reduces token usage by ~60% for large chunks.
    Falls back to original chunk content on any error.
    """
    # Skip compression for small chunks — not worth an LLM call
    if len(chunk.content) < 400:
        return chunk

    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": _COMPRESS_SYSTEM},
                    {"role": "user", "content":
                        f"Question: {question}\n\nCode chunk from {chunk.file_path}:\n\n{chunk.content}"},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512},
            },
            timeout=15,
        )
        r.raise_for_status()
        compressed = r.json()["message"]["content"].strip()
        if compressed and len(compressed) > 20:
            from dataclasses import replace
            return RetrievedChunk(
                id=chunk.id,
                content=compressed,
                file_path=chunk.file_path,
                language=chunk.language,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                chunk_type=chunk.chunk_type,
                name=chunk.name,
                context_header=f"{chunk.context_header} [compressed]",
                score=chunk.score,
            )
    except Exception:
        pass
    return chunk  # fallback: original chunk unchanged


def compress_chunks(
    chunks: list[RetrievedChunk],
    question: str,
    compress: bool = True,
) -> list[RetrievedChunk]:
    """Compress top chunks to reduce context tokens. Runs sequentially."""
    if not compress:
        return chunks
    # Only compress the top 3 — the rest are already deprioritised
    compressed = [compress_chunk(c, question) for c in chunks[:3]]
    return compressed + chunks[3:]


# ── 6. Faithfulness Check ─────────────────────────────────────────────────────

_FAITHFULNESS_SYSTEM = """You are a faithfulness checker for a RAG system.
Given a question, retrieved context, and a generated answer, determine if the answer
is grounded in the context or contains hallucinated details.

Output a JSON object:
{
  "is_faithful": true/false,
  "confidence": 0.0-1.0,
  "unsupported_claims": ["list of specific claims not found in context"],
  "verdict": "one sentence summary"
}
Output ONLY valid JSON."""


def check_faithfulness(
    question: str,
    context: str,
    answer: str,
) -> dict:
    """
    Step 6: Verify the generated answer is grounded in the retrieved context.
    Returns a faithfulness report dict.
    """
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": _FAITHFULNESS_SYSTEM},
                    {"role": "user", "content":
                        f"Question: {question}\n\n"
                        f"Context:\n{context[:2000]}\n\n"
                        f"Answer:\n{answer[:1000]}"},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            },
            timeout=20,
        )
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "is_faithful": None,
            "confidence": None,
            "unsupported_claims": [],
            "verdict": f"(faithfulness check unavailable: {e})",
        }


# ── Full RAG Pipeline ─────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    """Complete result from the RAG pipeline."""
    query: RAGQuery
    chunks: list[RetrievedChunk]
    context: str
    sources: list[str]
    answer: str
    faithfulness: Optional[dict]
    timings: dict[str, float] = field(default_factory=dict)
    tokens_used: int = 0
    compressed: bool = False


def rag_stream(
    question: str,
    history: Optional[list[dict]] = None,
    top_k: int = 5,
    repo_root: Optional[str] = None,
    language_filter: Optional[str] = None,
    repo_filter: Optional[str] = None,
    use_compression: bool = False,   # adds 1 LLM call per chunk — slower but richer
    check_faithfulness_flag: bool = False,
    on_step: Optional[callable] = None,  # callback(step_name, data) for UI updates
) -> Iterator[str]:
    """
    Full RAG pipeline — yields tokens as they stream.

    on_step(name, data) is called at each pipeline step so the UI
    can show live progress without blocking the stream.

    Steps emitted:
      ('query_understood', RAGQuery)
      ('retrieved', list[RetrievedChunk])
      ('compressed', list[RetrievedChunk])
      ('context_built', (context, sources, token_count))
      ('generating', None)
      ('faithfulness', dict)
    """
    timings: dict[str, float] = {}
    t = time.perf_counter

    def _step(name: str, data):
        if on_step:
            on_step(name, data)

    # ── Step 1: Query understanding ───────────────────────────────────────────
    t0 = t()
    raq = understand_query(question, history)
    timings["1_understand"] = round(t() - t0, 2)
    _step("query_understood", raq)

    # ── Step 2: Multi-strategy retrieval ──────────────────────────────────────
    t0 = t()
    chunks = multi_strategy_retrieve(
        raq, top_k=top_k,
        repo_root=repo_root,
        language_filter=language_filter,
        repo_filter=repo_filter,
    )
    # Final rerank to top_k
    chunks = rerank(chunks, question, raq.plan, top_k=top_k)
    timings["2_retrieve"] = round(t() - t0, 2)
    _step("retrieved", chunks)

    # ── Step 3/4: Contextual compression (optional) ───────────────────────────
    if use_compression and chunks:
        t0 = t()
        chunks = compress_chunks(chunks, question, compress=True)
        timings["4_compress"] = round(t() - t0, 2)
        _step("compressed", chunks)

    # ── Step 5a: Build context ────────────────────────────────────────────────
    if chunks:
        context, sources = build_context(chunks, question, raq.plan.task)
        system, user_msg = build_prompt(raq.plan.task, question, context, sources)
    else:
        context, sources = "", []
        system, user_msg = build_no_context_prompt(raq.plan.task, question)

    token_count = _rough_token_count(context + user_msg)
    timings["5a_context"] = round(t() - t0, 2)
    _step("context_built", (context, sources, token_count))

    # History trimming
    trimmed_history = _trim_history_to_budget(
        system=system,
        history=history or [],
        user_message=user_msg,
        max_tokens=MAX_CONTEXT_TOKENS,
        reserve_for_response=LLM_MAX_TOKENS,
    )

    # ── Step 5b: Stream generation ────────────────────────────────────────────
    _step("generating", None)
    t0 = t()
    full_answer = ""
    for token in stream_response(system, user_msg, history=trimmed_history, task=raq.plan.task):
        full_answer += token
        yield token
    timings["5b_generate"] = round(t() - t0, 2)

    # ── Step 6: Faithfulness check (optional) ─────────────────────────────────
    if check_faithfulness_flag and full_answer and context:
        t0 = t()
        faith = check_faithfulness(question, context, full_answer)
        timings["6_faithfulness"] = round(t() - t0, 2)
        _step("faithfulness", faith)
