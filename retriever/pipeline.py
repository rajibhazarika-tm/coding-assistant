# retriever/pipeline.py
"""
Cursor-style 5-step retrieval pipeline.

Flow:
  1. Query Analysis  — LLM extracts precise search terms, symbols, file hints
  2. Grep/Ripgrep    — exact literal matches (function names, class names, error strings)
  3. Semantic Search — vector + BM25 hybrid for conceptual matches
  4. Merge & Dedup   — combine results from both sources, remove overlaps
  5. Rerank          — LLM-free cross-encoder score: position + diversity + recency

Why this order?
- Grep catches exact symbols the user mentioned ("findById", "NullPointerException")
  which semantic search often misses because they're rare tokens
- Semantic catches intent ("how does auth work") which grep can't handle
- Combining both gives recall of grep + precision of semantic
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from config.settings import OLLAMA_BASE_URL, MODEL, MAX_CONTEXT_TOKENS
from retriever.hybrid_search import RetrievedChunk, retrieve as _semantic_retrieve
from retriever.context_builder import build_context


# ── Step 1: Query Analysis ────────────────────────────────────────────────────

@dataclass
class QueryPlan:
    """What the LLM decided to search for."""
    search_terms: list[str]        # exact strings for grep (symbols, error text)
    semantic_query: str            # rewritten query for vector search
    file_hints: list[str]          # file name patterns to prioritise (e.g. "Controller", "Service")
    task: str                      # explain | review | generate | general | debug
    reasoning: str                 # why these terms were chosen (for UI transparency)


_PLAN_SYSTEM = """You are a code search planner. Given a user question, output a JSON object with:
- "search_terms": list of 1-5 exact strings to grep for (function names, class names, error strings, annotation names). Be specific.
- "semantic_query": a rewritten, expanded version of the question optimised for semantic embedding search.
- "file_hints": list of 0-3 partial file name patterns likely to contain the answer (e.g. "Controller", "Service", "config").
- "task": one of: explain, review, generate, debug, general
- "reasoning": one sentence explaining your search strategy.

Output ONLY valid JSON. No markdown fences. No explanation outside the JSON."""


def analyse_query(question: str) -> QueryPlan:
    """Step 1: Ask the LLM to decompose the question into search directives."""
    try:
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": _PLAN_SYSTEM},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 256},
            },
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json()["message"]["content"].strip()
        # Strip markdown fences if model adds them anyway
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        return QueryPlan(
            search_terms=data.get("search_terms", [])[:5],
            semantic_query=data.get("semantic_query", question),
            file_hints=data.get("file_hints", [])[:3],
            task=data.get("task", "general"),
            reasoning=data.get("reasoning", ""),
        )
    except Exception:
        # Graceful fallback: treat question words as search terms
        words = [w for w in question.split() if len(w) > 4]
        return QueryPlan(
            search_terms=words[:3],
            semantic_query=question,
            file_hints=[],
            task="general",
            reasoning="(query analysis unavailable — using fallback)",
        )


# ── Step 2: Grep / Ripgrep ────────────────────────────────────────────────────

@dataclass
class GrepMatch:
    file_path: str
    line_number: int
    line_content: str
    term: str
    score: float = 1.0


def _find_repo_root() -> Optional[Path]:
    """Guess the indexed repo root from ChromaDB metadata."""
    try:
        from retriever.hybrid_search import _collection
        col = _collection()
        sample = col.get(limit=1, include=["metadatas"])
        if sample["metadatas"]:
            fp = sample["metadatas"][0].get("file_path", "")
            if fp:
                # file_path is relative; just return cwd as the root
                return Path.cwd()
    except Exception:
        pass
    return Path.cwd()


def grep_search(
    terms: list[str],
    repo_root: Optional[str | Path] = None,
    max_results_per_term: int = 15,
    context_lines: int = 3,
) -> list[GrepMatch]:
    """
    Step 2: Exact literal search using ripgrep (rg) or grep fallback.

    Returns up to max_results_per_term matches per term, with context_lines
    of surrounding code so we can build a meaningful chunk from grep hits.
    """
    if not terms:
        return []

    root = Path(repo_root) if repo_root else _find_repo_root() or Path.cwd()
    if not root.exists():
        return []

    # Prefer ripgrep (much faster on large repos), fall back to grep
    rg = shutil.which("rg") or shutil.which("ripgrep")
    use_rg = rg is not None

    matches: list[GrepMatch] = []
    seen: set[str] = set()

    for term in terms:
        if not term.strip():
            continue
        try:
            if use_rg:
                cmd = [
                    rg, "--json",
                    "--context", str(context_lines),
                    "--max-count", str(max_results_per_term),
                    "--type-add", "code:*.{py,java,js,ts,go,rs,cpp,c,cs,rb,kt,scala,swift}",
                    "--type", "code",
                    "--ignore-case",
                    "--", term, str(root),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                matches.extend(_parse_rg_json(result.stdout, term, seen))
            else:
                cmd = [
                    "grep", "-rn", "--include=*.py", "--include=*.java",
                    "--include=*.js", "--include=*.ts", "--include=*.go",
                    "-m", str(max_results_per_term),
                    "-i", term, str(root),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                matches.extend(_parse_grep_output(result.stdout, term, seen, root))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    return matches


def _parse_rg_json(output: str, term: str, seen: set) -> list[GrepMatch]:
    matches = []
    for line in output.splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") != "match":
                continue
            data = obj["data"]
            fp = data["path"]["text"]
            ln = data["line_number"]
            text = data["lines"]["text"].rstrip()
            key = f"{fp}:{ln}"
            if key in seen:
                continue
            seen.add(key)
            matches.append(GrepMatch(file_path=fp, line_number=ln, line_content=text, term=term))
        except Exception:
            continue
    return matches


def _parse_grep_output(output: str, term: str, seen: set, root: Path) -> list[GrepMatch]:
    matches = []
    for line in output.splitlines():
        try:
            fp, ln_str, content = line.split(":", 2)
            key = f"{fp}:{ln_str}"
            if key in seen:
                continue
            seen.add(key)
            # Make path relative to root
            try:
                rel = str(Path(fp).relative_to(root))
            except ValueError:
                rel = fp
            matches.append(GrepMatch(file_path=rel, line_number=int(ln_str),
                                     line_content=content.rstrip(), term=term))
        except Exception:
            continue
    return matches


def grep_matches_to_chunks(
    matches: list[GrepMatch],
    repo_root: Optional[str | Path] = None,
    window: int = 20,
) -> list[RetrievedChunk]:
    """
    Convert grep matches into RetrievedChunks by reading window lines
    around each match from disk.
    """
    root = Path(repo_root) if repo_root else _find_repo_root() or Path.cwd()
    chunks: list[RetrievedChunk] = []
    seen_files: dict[str, list[str]] = {}

    for m in matches:
        fp = m.file_path
        abs_path = root / fp if not Path(fp).is_absolute() else Path(fp)
        if not abs_path.exists():
            continue

        # Cache file lines
        if fp not in seen_files:
            try:
                seen_files[fp] = abs_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue

        file_lines = seen_files[fp]
        start = max(0, m.line_number - 1 - window // 2)
        end   = min(len(file_lines), m.line_number - 1 + window // 2)
        snippet = "\n".join(file_lines[start:end])

        ext = abs_path.suffix.lower()
        lang_map = {".py":"python",".java":"java",".js":"javascript",
                    ".ts":"typescript",".go":"go",".rs":"rust",".kt":"kotlin",
                    ".cs":"c_sharp",".rb":"ruby",".cpp":"cpp",".c":"c"}
        lang = lang_map.get(ext, "text")

        chunk_id = f"{fp}:{start+1}-{end}:grep"
        chunks.append(RetrievedChunk(
            id=chunk_id, content=snippet, file_path=fp, language=lang,
            start_line=start + 1, end_line=end,
            chunk_type="grep_match", name=m.term,
            context_header=f"grep:{m.term}", score=1.2,  # boosted — exact match
        ))

    return chunks


# ── Step 3: Semantic search (existing pipeline) ───────────────────────────────

def semantic_search(
    query: str,
    top_k: int = 8,
    language_filter: Optional[str] = None,
    file_filter: Optional[str] = None,
    repo_filter: Optional[str] = None,
) -> list[RetrievedChunk]:
    """Step 3: Vector + BM25 hybrid retrieval (existing pipeline)."""
    return _semantic_retrieve(
        query=query,
        top_k=top_k,
        language_filter=language_filter,
        file_filter=file_filter,
        repo_filter=repo_filter,
    )


# ── Step 4: Merge & Dedup ─────────────────────────────────────────────────────

def merge_results(
    grep_chunks: list[RetrievedChunk],
    semantic_chunks: list[RetrievedChunk],
    file_hints: list[str],
) -> list[RetrievedChunk]:
    """
    Step 4: Combine grep + semantic results, deduplicate overlapping spans,
    apply file-hint boost.
    """
    combined: dict[str, RetrievedChunk] = {}

    for chunk in grep_chunks:
        key = _overlap_key(chunk)
        if key not in combined:
            combined[key] = chunk
        else:
            # Keep higher score
            if chunk.score > combined[key].score:
                combined[key] = chunk

    for chunk in semantic_chunks:
        key = _overlap_key(chunk)
        if key not in combined:
            combined[key] = chunk
        else:
            # Merge score: if grep already found this file+area, boost it
            existing = combined[key]
            merged_score = existing.score + chunk.score * 0.5
            if chunk.chunk_type != "grep_match":
                # Replace with semantic chunk (better content boundary) but keep boosted score
                combined[key] = RetrievedChunk(
                    id=chunk.id, content=chunk.content, file_path=chunk.file_path,
                    language=chunk.language, start_line=chunk.start_line, end_line=chunk.end_line,
                    chunk_type=chunk.chunk_type, name=chunk.name,
                    context_header=chunk.context_header, score=merged_score,
                )

    # File-hint boost: chunks whose file path contains a hint get +0.3
    results = list(combined.values())
    for hint in file_hints:
        hint_lower = hint.lower()
        for chunk in results:
            if hint_lower in chunk.file_path.lower():
                chunk.score += 0.3

    return results


def _overlap_key(chunk: RetrievedChunk) -> str:
    """Key that groups chunks covering the same file region."""
    # Round line numbers to nearest 15 to collapse near-overlapping windows
    bucket = (chunk.start_line // 15) * 15
    return f"{chunk.file_path}:{bucket}"


# ── Step 5: Rerank ────────────────────────────────────────────────────────────

def rerank(
    chunks: list[RetrievedChunk],
    query: str,
    plan: QueryPlan,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Step 5: LLM-free cross-encoder reranking using:
    - Base retrieval score
    - Query term overlap (how many search terms appear in this chunk)
    - File hint match
    - Source diversity (avoid returning 5 chunks from the same file)
    - Penalise test files when task is not 'review' (tests rarely explain production logic)
    """
    if not chunks:
        return []

    query_tokens = set(_tokenize_simple(query))
    plan_tokens  = set(_tokenize_simple(" ".join(plan.search_terms)))

    scored: list[tuple[float, RetrievedChunk]] = []
    file_counts: dict[str, int] = {}

    for chunk in chunks:
        s = chunk.score

        # Term overlap boost
        chunk_tokens = set(_tokenize_simple(chunk.content))
        overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        plan_overlap = len(plan_tokens & chunk_tokens) / max(len(plan_tokens), 1)
        s += overlap * 0.4 + plan_overlap * 0.6

        # Penalise test files unless task is review/debug
        fp_lower = chunk.file_path.lower()
        is_test = any(t in fp_lower for t in ("test", "spec", "_test.", ".test."))
        if is_test and plan.task not in ("review", "debug"):
            s *= 0.6

        # Penalise if we already have chunks from this file (diversity)
        n_from_file = file_counts.get(chunk.file_path, 0)
        s *= (1.0 / (1 + n_from_file * 0.4))

        file_counts[chunk.file_path] = n_from_file + 1
        scored.append((s, chunk))

    scored.sort(key=lambda x: -x[0])
    result = [c for _, c in scored[:top_k]]

    # Update scores with final reranked values
    for i, (s, c) in enumerate(scored[:top_k]):
        result[i].score = round(s, 4)

    return result


def _tokenize_simple(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# ── Full pipeline ─────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    plan: QueryPlan
    grep_chunks: list[RetrievedChunk]
    semantic_chunks: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]
    context: str
    sources: list[str]
    # Timing for UI transparency
    timings: dict[str, float] = field(default_factory=dict)


def run_pipeline(
    question: str,
    repo_root: Optional[str | Path] = None,
    top_k: int = 5,
    language_filter: Optional[str] = None,
    repo_filter: Optional[str] = None,
    task: str = "general",
) -> PipelineResult:
    """
    Execute all 5 steps and return a PipelineResult.
    Safe to call even when Ollama is offline — degrades gracefully.
    """
    import time

    timings: dict[str, float] = {}

    # Step 1: Query analysis
    t0 = time.perf_counter()
    plan = analyse_query(question)
    if task != "general":
        plan.task = task  # CLI override
    timings["1_analyse"] = round(time.perf_counter() - t0, 3)

    # Step 2: Grep
    t0 = time.perf_counter()
    grep_raw = grep_search(plan.search_terms, repo_root=repo_root)
    grep_chunks = grep_matches_to_chunks(grep_raw, repo_root=repo_root)
    timings["2_grep"] = round(time.perf_counter() - t0, 3)

    # Step 3: Semantic
    t0 = time.perf_counter()
    sem_chunks = semantic_search(
        plan.semantic_query, top_k=top_k * 2,
        language_filter=language_filter, repo_filter=repo_filter,
    )
    timings["3_semantic"] = round(time.perf_counter() - t0, 3)

    # Step 4: Merge
    t0 = time.perf_counter()
    merged = merge_results(grep_chunks, sem_chunks, plan.file_hints)
    timings["4_merge"] = round(time.perf_counter() - t0, 3)

    # Step 5: Rerank
    t0 = time.perf_counter()
    final = rerank(merged, question, plan, top_k=top_k)
    timings["5_rerank"] = round(time.perf_counter() - t0, 3)

    # Build context
    context, sources = build_context(final, question, task)

    return PipelineResult(
        plan=plan,
        grep_chunks=grep_chunks,
        semantic_chunks=sem_chunks,
        final_chunks=final,
        context=context,
        sources=sources,
        timings=timings,
    )
