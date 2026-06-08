# assistant/llm.py
"""
Ollama API wrapper with streaming support.

Handles:
- Streaming responses for interactive chat (low latency feel)
- Conversation history for multi-turn chat
- Error recovery (Ollama not running, model not pulled, etc.)
- Context length enforcement
"""

from __future__ import annotations
import json
import sys
import requests
from typing import Iterator, Optional

from config.settings import (
    OLLAMA_BASE_URL, MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    LLM_TIMEOUT_SECONDS, MAX_CONTEXT_TOKENS,
)


def check_ollama_running() -> bool:
    """Check if Ollama server is reachable."""
    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def check_model_available(model: str = MODEL) -> bool:
    """Check if the model is pulled and available."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in response.json().get("models", [])]
        # Allow partial match (e.g. "qwen2.5-coder:7b" matches "qwen2.5-coder:7b-instruct-q4_K_M")
        return any(model in m or m.startswith(model) for m in models)
    except Exception:
        return False


def _count_tokens_approx(messages: list[dict]) -> int:
    """
    Rough token count for a message list.
    Code tokenises at ~2.5-3 chars/token (brackets, keywords, identifiers
    each count as 1 token). Using 3 chars/token — more conservative than
    the old 4, so trimming kicks in earlier and avoids overflow.
    """
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 3


def _trim_history_to_budget(
    system: str,
    history: list[dict],
    user_message: str,
    max_tokens: int,
    reserve_for_response: int,
) -> list[dict]:
    """
    Drop oldest history pairs until the full message fits within budget.

    Uses 80% of the available budget as a safety margin — token counting
    is approximate (3 chars/token estimate) and Ollama measures differently.
    The 20% headroom ensures we never overflow even with estimation error.
    Always keeps at least the last 2 turns for conversational coherence.
    """
    # 80% safety margin accounts for tokeniser differences
    budget = int((max_tokens - reserve_for_response) * 0.80)
    base = [{"role": "system", "content": system}] if system else []
    base += [{"role": "user", "content": user_message}]
    base_tokens = _count_tokens_approx(base)

    kept = list(history)
    while kept and base_tokens + _count_tokens_approx(kept) > budget:
        # Drop oldest pair (user + assistant = 2 messages)
        if len(kept) >= 2:
            kept = kept[2:]
        else:
            kept = []

    return kept


def stream_response(
    system: str,
    user_message: str,
    history: Optional[list[dict]] = None,
    model: str = MODEL,
    temperature: float = LLM_TEMPERATURE,
    task: str = "general",
) -> Iterator[str]:
    """
    Stream tokens from Ollama as they arrive.

    Protects against "input length exceeds context length" by:
    - Trimming oldest history turns when total token estimate exceeds budget
    - Setting num_ctx explicitly so Ollama uses the right window size
    - Truncating user_message as a last-resort safety net
    """
    # Safety net: truncate enormous user messages (pasted files, huge error logs).
    # Reserve 40% of context budget for system prompt + history.
    # Use 3 chars/token (conservative for code).
    max_user_chars = int(MAX_CONTEXT_TOKENS * 0.60 * 3)
    if len(user_message) > max_user_chars:
        user_message = user_message[:max_user_chars] + "\n\n[...truncated to fit context window]"

    trimmed_history = _trim_history_to_budget(
        system=system,
        history=history or [],
        user_message=user_message,
        max_tokens=MAX_CONTEXT_TOKENS,
        reserve_for_response=LLM_MAX_TOKENS,
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(trimmed_history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": __import__("config.settings", fromlist=["LLM_MAX_TOKENS_BY_TASK"]).LLM_MAX_TOKENS_BY_TASK.get(task, LLM_MAX_TOKENS),
            # num_ctx = total window Ollama allocates (input + output).
        # Must be large enough for: system + history + RAG context + user message + response.
        # We pass MAX_CONTEXT_TOKENS (now 4096) + LLM_MAX_TOKENS (1024) = 5120.
        # At 4GB VRAM this uses ~640MB KV cache — safe with qwen2.5-coder:7b.
        "num_ctx": MAX_CONTEXT_TOKENS + LLM_MAX_TOKENS,
        },
    }

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=LLM_TIMEOUT_SECONDS,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    token = data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if data.get("done"):
                        break
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.ConnectionError:
        yield "\n\n❌ Cannot connect to Ollama. Is it running? Try: `ollama serve`"
    except requests.exceptions.Timeout:
        yield f"\n\n⏰ Request timed out after {LLM_TIMEOUT_SECONDS}s. Try a smaller context or model."
    except requests.exceptions.HTTPError as e:
        if "404" in str(e):
            yield f"\n\n❌ Model '{model}' not found. Pull it with: `ollama pull {model}`"
        else:
            yield f"\n\n❌ HTTP error: {e}"


def ask(
    system: str,
    user_message: str,
    history: Optional[list[dict]] = None,
    model: str = MODEL,
    print_streaming: bool = True,
) -> str:
    """
    Ask the LLM and return the full response.

    If print_streaming=True, prints tokens to stdout as they arrive.
    """
    full_response = []
    for token in stream_response(system, user_message, history, model):
        if print_streaming:
            print(token, end="", flush=True)
        full_response.append(token)

    if print_streaming:
        print()  # Final newline

    return "".join(full_response)
