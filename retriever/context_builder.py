# retriever/context_builder.py
"""
Smart context assembly with strict token budgeting.

PROBLEM: qwen2.5-coder:7b on 4GB VRAM runs best with short contexts.
Too much context → slow inference, potential OOM, degraded output quality.

SOLUTION:
- Hard limit: MAX_CONTEXT_TOKENS total
- Allocate budget: system_prompt + chunks + question + response headroom
- Prioritize chunks by score, truncate at budget
- Add structural metadata so the model understands where code lives
"""

from __future__ import annotations
from retriever.hybrid_search import RetrievedChunk
from config.settings import MAX_CONTEXT_TOKENS, RESERVED_TOKENS, TOKENS_PER_CHUNK


def _rough_token_count(text: str) -> int:
    """
    Approximate token count (4 chars ≈ 1 token for code).
    Good enough for budget management without running a tokenizer.
    """
    return len(text) // 4


def build_context(
    chunks: list[RetrievedChunk],
    query: str,
    task_type: str = "general",
) -> tuple[str, list[str]]:
    """
    Assemble a context string from retrieved chunks, respecting token budget.

    Args:
        chunks: Ranked list of retrieved code chunks
        query: The user's question
        task_type: "explain", "review", "generate", or "general"

    Returns:
        (context_string, list_of_source_files_used)
    """
    budget = MAX_CONTEXT_TOKENS - RESERVED_TOKENS - _rough_token_count(query)
    budget = max(budget, 512)  # Always allow at least some context

    context_parts: list[str] = []
    sources_used: list[str] = []
    tokens_used = 0

    for chunk in chunks:
        chunk_text = _format_chunk(chunk)
        chunk_tokens = _rough_token_count(chunk_text)

        if tokens_used + chunk_tokens > budget:
            # Try a truncated version
            truncated = _truncate_chunk(chunk_text, budget - tokens_used)
            if _rough_token_count(truncated) > 50:  # Worth including
                context_parts.append(truncated + "\n# [... truncated for context budget]")
                tokens_used += _rough_token_count(truncated)
                if chunk.file_path not in sources_used:
                    sources_used.append(chunk.file_path)
            break

        context_parts.append(chunk_text)
        tokens_used += chunk_tokens
        if chunk.file_path not in sources_used:
            sources_used.append(chunk.file_path)

    if not context_parts:
        return "", []

    header = f"# Relevant code from {len(sources_used)} file(s) (~{tokens_used*4} chars)\n"
    return header + "\n\n".join(context_parts), sources_used


def _format_chunk(chunk: RetrievedChunk) -> str:
    """Format a chunk with location metadata as a code block."""
    lines = f"lines {chunk.start_line}-{chunk.end_line}"
    header = f"### {chunk.file_path} ({lines})"
    if chunk.name:
        header += f" — {chunk.chunk_type}: `{chunk.name}`"

    return f"{header}\n```{chunk.language}\n{chunk.content}\n```"


def _truncate_chunk(text: str, max_tokens: int) -> str:
    """Truncate a chunk to fit within token budget, keeping the start."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars]
