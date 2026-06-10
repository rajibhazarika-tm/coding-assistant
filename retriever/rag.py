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
    query_variants: list[str] = None # diverse phrasings for multi-angle retrieval


_QUERY_UNDERSTANDING_SYSTEM = """You are a query analyser for a code search system. Given a user question, output a JSON object with these EXACT keys:

"search_terms": list of 1-6 EXACT code identifiers to grep for. Rules:
  - Use camelCase/PascalCase/snake_case as they appear verbatim in code
  - Include method names: "getUserById", "processPayment", "validateToken"
  - Include class names: "OrderService", "AuthController", "PaymentGateway"
  - Include annotation names: "@Transactional", "@Autowired", "@RestController"
  - Include exception/error names: "NullPointerException", "AuthenticationException"
  - Only identifiers that appear verbatim in source files — no plain English words

"query_variants": list of 3-5 DIVERSE search queries that all express the same intent from different angles.
  These run as SEPARATE vector searches so they must use different vocabulary to find different relevant code.
  Include:
  - Conceptual angle: "how does X work" → "X implementation mechanism design pattern"
  - Implementation angle: specific method/class names, technical terms
  - Caller angle: "what calls X" / "where is X used" / "X invocation"
  - Problem angle: what problem does X solve, when would you need it
  - Synonym angle: alternative names/terms for the same concept
  Example for "how does authentication work":
    ["JWT token validation implementation",
     "authenticate user credentials verify password",
     "AuthService login method security filter chain",
     "session management token expiry refresh",
     "Spring Security WebSecurityConfigurerAdapter"]

"reformulated": single self-contained rewrite of the question (resolve "it"/"this"/"that" from history)

"hypothetical_answer": 3-5 line realistic code snippet that would answer the question — embedded for HDE search

"file_hints": list of 0-3 partial filename patterns likely containing the answer

"task": one of: explain, review, generate, debug, general

"is_multi_hop": true if the question requires info from multiple separate files/services

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
        llm_terms = data.get("search_terms", [])[:6]
        raw_symbols = correction.symbols_found
        combined_terms = list(dict.fromkeys(llm_terms + raw_symbols))

        # Build query variants — ensure reformulated is always first
        reformulated = data.get("reformulated", working_question)
        llm_variants = data.get("query_variants", [])[:4]  # cap at 4 so total stays ≤6
        # Combine: reformulated + LLM variants + working question (for coverage)
        # Deduplicate while preserving order
        all_variants = [reformulated] + llm_variants
        if working_question not in all_variants:
            all_variants.append(working_question)
        # Deduplicate
        seen_v: set = set()
        query_variants = []
        for v in all_variants:
            if v and v.strip() and v not in seen_v:
                seen_v.add(v); query_variants.append(v)

        return RAGQuery(
            original=question,
            corrected=working_question,
            reformulated=reformulated,
            sub_queries=data.get("sub_queries", [working_question]),
            hypothetical_answer=data.get("hypothetical_answer", ""),
            plan=plan,
            is_multi_hop=data.get("is_multi_hop", False),
            is_conversational=bool(history),
            corrections=correction.corrections,
            grep_terms=combined_terms,
            query_variants=query_variants,
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
            query_variants=[working_question],
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

    # Strategy B: Multi-angle vector search across ALL query variants
    #
    # Cursor-style improvement: instead of embedding the question once, we
    # embed 3-5 diverse phrasings and run a separate vector search for each.
    # Different phrasings activate different embedding dimensions, surfacing
    # code that a single query misses.
    #
    # Example — "how does authentication work?":
    #   Variant 1: "JWT token validation implementation"           → JwtFilter.java
    #   Variant 2: "authenticate user credentials verify password" → AuthService.java
    #   Variant 3: "AuthService login method security filter"      → SecurityConfig.java
    #   Variant 4: "session management token expiry refresh"       → TokenRepository.java
    #
    # RRF naturally boosts chunks appearing in multiple variant results.
    variants = raq.query_variants or [raq.reformulated]
    # Weights: reformulated=1.0, then decreasing for diversity variants
    variant_weights = [1.0, 0.85, 0.75, 0.65, 0.60]

    for v_idx, variant in enumerate(variants):
        v_weight = variant_weights[min(v_idx, len(variant_weights)-1)]
        v_vec = _embed_text(variant)
        if not v_vec:
            continue
        try:
            kw_v: dict = {
                "query_embeddings": [v_vec],
                "n_results": fetch_k,
                "include": ["documents", "metadatas", "distances"],
            }
            if where_filter:
                kw_v["where"] = where_filter
            v_res = col.query(**kw_v)
            for rank, (cid, doc, meta, _dist) in enumerate(zip(
                v_res["ids"][0], v_res["documents"][0],
                v_res["metadatas"][0], v_res["distances"][0],
            )):
                _add(_build_result(cid, doc, meta or {}, 0), weight=v_weight, rank=rank)
        except Exception:
            pass

    # Strategy C: BM25 across reformulated + first 2 variants
    bm25 = _get_cached_bm25(col, where_filter, None)
    if bm25 and corpus:
        docs = [doc for _, doc, _ in corpus]
        bm25_queries = list(dict.fromkeys([raq.reformulated] + variants[:2]))
        combined_bm25: dict[int, float] = {}
        for bq_idx, bq in enumerate(bm25_queries):
            bq_weight = 0.8 if bq_idx == 0 else 0.5
            for i, s in enumerate(bm25.score(bq, j) for j in range(len(docs))):
                combined_bm25[i] = combined_bm25.get(i, 0) + s * bq_weight
        bm25_ranked = sorted(combined_bm25, key=lambda i: -combined_bm25[i])[:fetch_k]
        for rank, idx in enumerate(bm25_ranked):
            cid, doc, meta = corpus[idx]
            _add(_build_result(cid, doc, meta, 0), weight=0.8, rank=rank)

    # Strategy D: Grep — exact camelCase/snake_case identifiers
    if repo_root:
        grep_terms = [t for t in (raq.grep_terms or raq.plan.search_terms or []) if t.strip()]
        if grep_terms:
            grep_raw = grep_search(grep_terms, repo_root=repo_root)
            grep_chunks = grep_matches_to_chunks(grep_raw, repo_root=repo_root)
            for rank, chunk in enumerate(grep_chunks):
                _add(chunk, weight=1.2, rank=rank)

    # Collect, filter weak candidates, and rerank
    candidates = []
    for chunk, score in all_results.values():
        chunk.score = round(score, 4)
        candidates.append(chunk)

    if not candidates:
        return []

    # Adaptive minimum score: only drop truly weak candidates.
    # Keep at least top_k*2 candidates so reranker has enough to work with.
    if len(candidates) > top_k * 2:
        scores = sorted(c.score for c in candidates)
        # Minimum = 20th percentile score — drops the bottom 20% of retrievals
        cutoff_idx = max(0, len(scores) - int(len(scores) * 0.80) - 1)
        min_score = scores[cutoff_idx] if cutoff_idx < len(scores) else 0
        filtered = [c for c in candidates if c.score >= min_score]
        candidates = filtered if len(filtered) >= top_k else candidates

    # Rerank with precision-focused scorer
    # Enrich plan.search_terms with exact grep_terms for the reranker's
    # exact-identifier bonus (chunk.name == search_term → +3.0 score)
    if raq.grep_terms:
        from retriever.pipeline import QueryPlan
        enriched_plan = QueryPlan(
            search_terms=list(dict.fromkeys((raq.plan.search_terms or []) + raq.grep_terms)),
            semantic_query=raq.plan.semantic_query,
            file_hints=raq.plan.file_hints,
            task=raq.plan.task,
            reasoning=raq.plan.reasoning,
        )
    else:
        enriched_plan = raq.plan

    return rerank(candidates, raq.original, enriched_plan, top_k=top_k * 2)


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
        # Intercept Ollama stats sentinel — emit via on_step, don't yield as text
        if isinstance(token, str) and token.startswith("__llm_stats__"):
            try:
                import json as _json
                stats = _json.loads(token[len("__llm_stats__"):-len("__end_stats__")])
                if on_step:
                    on_step("llm_stats", stats)
            except Exception:
                pass
            continue
        full_answer += token
        yield token
    timings["5b_generate"] = round(t() - t0, 2)

    # ── Step 6: Faithfulness check (optional) ─────────────────────────────────
    if check_faithfulness_flag and full_answer and context:
        t0 = t()
        faith = check_faithfulness(question, context, full_answer)
        timings["6_faithfulness"] = round(t() - t0, 2)
        _step("faithfulness", faith)
