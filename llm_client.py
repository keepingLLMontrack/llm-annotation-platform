"""
LLM client for OpenAI-compatible endpoints (LM Studio / Ollama / vLLM / hosted).

Key responsibilities for this project:
  - call a specific model (resolved via models.resolve_model)
  - pass the system prompt as a real system message (handled by the caller)
  - use fixed-ish decoding (temperature / top_p recorded per run)
  - separate any hidden reasoning / thinking trace from the final, user-facing
    answer, so annotators only ever label the final response.
"""

from __future__ import annotations

import re
from typing import Optional

from constants import DEFAULT_TEMPERATURE, DEFAULT_TOP_P

# Patterns some open-weight models (incl. gpt-oss / reasoning models) use to
# wrap their chain-of-thought inside the visible content.
_THINK_PATTERNS = [
    re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|>analysis(.*?)<\|channel\|>final", re.DOTALL | re.IGNORECASE),
]


def split_reasoning(text: str) -> tuple[str, str]:
    """
    Return (final_user_facing_text, reasoning_text).

    Strips known thinking-trace wrappers from the content. We never annotate
    the reasoning, but we keep it so it can be shown (greyed out) for context.
    """
    if not text:
        return "", ""
    reasoning_parts: list[str] = []
    cleaned = text
    for pat in _THINK_PATTERNS:
        for m in pat.findall(cleaned):
            reasoning_parts.append(m.strip())
        cleaned = pat.sub("", cleaned)
    return cleaned.strip(), "\n\n".join(p for p in reasoning_parts if p)


def _is_connection_error(error_text: str) -> bool:
    text = error_text.lower()
    markers = [
        "connection", "refused", "max retries exceeded",
        "failed to establish", "timeout", "api connection error", "localhost",
    ]
    return any(marker in text for marker in markers)


def _is_model_error(error_text: str) -> bool:
    text = error_text.lower()
    return "model" in text and any(
        s in text for s in (
            "not found", "not loaded", "does not exist",
            "unknown model", "invalid model",
        )
    )


def run_chat(
    messages: list[dict],
    model_cfg: dict,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = 4096,
) -> dict:
    """
    Call one model with a fully-assembled message list (system message already
    included by the caller).

    `model_cfg` is the dict returned by models.resolve_model().

    Returns a dict:
      {
        "ok": bool,
        "final": str,       # user-facing answer (reasoning stripped)
        "reasoning": str,   # hidden reasoning, if any (never annotated)
        "error": str,       # populated when ok is False
      }
    """
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError:
        return {
            "ok": False, "final": "", "reasoning": "",
            "error": "openai package not installed. Run: pip install openai",
        }

    try:
        client = OpenAI(
            base_url=model_cfg["base_url"],
            api_key=model_cfg["api_key"],
        )
        response = client.chat.completions.create(
            messages=messages,
            model=model_cfg["served_model_id"],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        msg = response.choices[0].message
        content = msg.content or ""

        # Some servers expose reasoning in a dedicated field rather than in
        # the content. Capture it but keep it out of the annotated answer.
        field_reasoning = ""
        for attr in ("reasoning", "reasoning_content"):
            val = getattr(msg, attr, None)
            if val:
                field_reasoning = str(val)
                break

        final, inline_reasoning = split_reasoning(content)
        reasoning = "\n\n".join(p for p in (field_reasoning, inline_reasoning) if p)
        return {"ok": True, "final": final, "reasoning": reasoning, "error": ""}

    except Exception as e:  # noqa: BLE001
        error_text = str(e)
        base_url = model_cfg.get("base_url", "?")
        if _is_connection_error(error_text):
            error_text = (
                f"Could not reach the model endpoint at '{base_url}'. "
                "Start your local server (e.g. LM Studio → Start Server) or "
                "check the base URL.\nOriginal error: " + error_text
            )
        elif _is_model_error(error_text):
            error_text = (
                f"Model '{model_cfg.get('served_model_id')}' not available at "
                f"'{base_url}'. Load it / fix the model id.\n"
                "Original error: " + error_text
            )
        return {"ok": False, "final": "", "reasoning": "", "error": error_text}


def build_messages(
    system_prompt: str,
    base_context: list[dict],
    distractor_response_pairs: list[tuple[str, str]],
    next_distractor: Optional[str] = None,
) -> list[dict]:
    """
    Assemble the message list for a distractor turn.

      [system]
      + base_context (real prior turns up to the injection point)
      + interleaved (distractor_k, model_response_k) for completed turns
      + [user: next_distractor]  (if provided)
    """
    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for turn in base_context:
        role = turn.get("role")
        if role in ("user", "assistant") and turn.get("content"):
            messages.append({"role": role, "content": turn["content"]})
    for distractor, response in distractor_response_pairs:
        messages.append({"role": "user", "content": distractor})
        if response:
            messages.append({"role": "assistant", "content": response})
    if next_distractor:
        messages.append({"role": "user", "content": next_distractor})
    return messages
