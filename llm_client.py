"""
LLM client using LM Studio local OpenAI-compatible API.
"""

from typing import Optional, Generator, Iterable
from constants import DEFAULT_CHAT_MODEL
from dataset_io import get_lmstudio_api_key, get_lmstudio_base_url


def _is_connection_error(error_text: str) -> bool:
    text = error_text.lower()
    markers = [
        "connection",
        "refused",
        "max retries exceeded",
        "failed to establish",
        "timeout",
        "api connection error",
        "localhost",
    ]
    return any(marker in text for marker in markers)


def _is_model_error(error_text: str) -> bool:
    text = error_text.lower()
    return "model" in text and (
        "not found" in text
        or "not loaded" in text
        or "does not exist" in text
        or "unknown model" in text
        or "invalid model" in text
    )

def chat_completion(
    messages: list[dict],
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_CHAT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = False,
) -> str | Iterable:
    """
    Send a chat completion request to LM Studio local API.

    Args:
        messages: list of {role, content} dicts (user/assistant turns)
        system_prompt: prepended as a system message if provided
        model: local model name loaded in LM Studio
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        stream: if True, returns a generator for streaming output

    Returns:
        Full response string (stream=False) or a generator (stream=True).
    """
    try:
        from openai import OpenAI  # noqa: PLC0415
    except ImportError:
        err = "ERROR: openai package not installed. Run: pip install openai"
        return (x for x in [err]) if stream else err

    base_url = get_lmstudio_base_url()
    api_key = get_lmstudio_api_key()

    # Build final message list
    all_messages: list[dict] = []
    if system_prompt:
        all_messages.append({"role": "system", "content": system_prompt})
    all_messages.extend(messages)

    try:
        client = OpenAI(base_url=base_url, api_key=api_key)

        if stream:
            return client.chat.completions.create(
                messages=all_messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )

        response = client.chat.completions.create(
            messages=all_messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        return response.choices[0].message.content or ""

    except Exception as e:
        error_text = str(e)

        if _is_connection_error(error_text):
            err = (
                "LM Studio connection error. Could not reach local server at "
                f"'{base_url}'.\n\n"
                "Start LM Studio, load your model, and click 'Start Server' "
                "(OpenAI-compatible endpoint, usually http://localhost:1234/v1).\n"
                f"Original error: {error_text}"
            )
            return (x for x in [err]) if stream else err

        if _is_model_error(error_text):
            err = (
                f"LM Studio model error for '{model}'.\n\n"
                "Make sure this exact model name is loaded in LM Studio. "
                "You can also set LLM_MODEL to the loaded model name.\n"
                f"Original error: {error_text}"
            )
            return (x for x in [err]) if stream else err

        err = f"LLM API error: {error_text}"
        return (x for x in [err]) if stream else err


def stream_to_string(stream_gen) -> Generator[str, None, None]:
    """
    Yield incremental text from a streaming chat_completion response.
    Each yielded value is the full accumulated string so far (for Streamlit's write).
    """
    full_text = ""
    try:
        for chunk in stream_gen:
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    full_text += delta.content
                    yield full_text
    except Exception as e:
        full_text += f"\n\n[Stream error: {e}]"
        yield full_text
