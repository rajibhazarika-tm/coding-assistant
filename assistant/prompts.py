# assistant/prompts.py
"""
Task-specific prompt templates optimized for qwen2.5-coder:7b.

KEY PRINCIPLES:
- Keep system prompts short (saves tokens, improves speed on small models)
- Be explicit about output format to reduce hallucination
- Reference file paths to ground the model in real code
- For code gen, ask for one function at a time
"""

from typing import Optional


SYSTEM_EXPLAIN = """You are a code explanation assistant. Given relevant code snippets from a repository, explain clearly and concisely what the code does. Reference specific file paths and function names. Be precise."""

SYSTEM_REVIEW = """You are a code reviewer. Analyze the provided code for bugs, security issues, performance problems, and style issues. Be specific: cite the file path and line number. Prioritize critical issues first."""

SYSTEM_GENERATE = """You are a code generation assistant. Generate clean, working code that matches the style of the existing codebase shown. Output ONLY the code, no explanations unless asked. Use the same language, patterns, and conventions visible in the context."""

SYSTEM_GENERAL = """You are a coding assistant with knowledge of the codebase shown. Answer questions accurately. Reference specific files and functions. If you don't know something, say so."""


def build_prompt(
    task: str,           # "explain" | "review" | "generate" | "general"
    query: str,
    context: str,
    sources: list[str],
    extra_instruction: Optional[str] = None,
) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) for the given task.

    Returns a tuple to pass to the Ollama chat API.
    """
    system_map = {
        "explain": SYSTEM_EXPLAIN,
        "review": SYSTEM_REVIEW,
        "generate": SYSTEM_GENERATE,
        "general": SYSTEM_GENERAL,
    }
    system = system_map.get(task, SYSTEM_GENERAL)

    if extra_instruction:
        system = system + "\n" + extra_instruction

    sources_note = ""
    if sources:
        files_list = "\n".join(f"  - {s}" for s in sources[:5])
        sources_note = f"\nRelevant files:\n{files_list}\n"

    if context:
        user_message = f"{sources_note}\n{context}\n\n---\n\n{query}"
    else:
        user_message = query

    return system, user_message


def build_no_context_prompt(task: str, query: str) -> tuple[str, str]:
    """Used when no relevant context was retrieved (model uses its own knowledge)."""
    system = SYSTEM_GENERATE if task == "generate" else SYSTEM_GENERAL
    return system, f"(No codebase context available for this query)\n\n{query}"
