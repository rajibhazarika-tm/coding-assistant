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


def stream_response(
    system: str,
    user_message: str,
    history: Optional[list[dict]] = None,
    model: str = MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> Iterator[str]:
    """
    Stream tokens from Ollama as they arrive.

    Yields string chunks suitable for printing to terminal.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_predict": LLM_MAX_TOKENS,
            "num_ctx": MAX_CONTEXT_TOKENS + LLM_MAX_TOKENS,  # Total context window
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
