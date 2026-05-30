"""
LLM client using HuggingFace Inference API.
Uses InferenceClient which supports streaming chat completions.
"""

from typing import Optional, Generator
from utils.dataset_io import get_hf_token
from utils.constants import DEFAULT_CHAT_MODEL


def chat_completion(
    messages: list[dict],
    system_prompt: Optional[str] = None,
    model: str = DEFAULT_CHAT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    stream: bool = False,
) -> str | Generator:
    """
    Send a chat completion request to the HF Inference API.

    Args:
        messages: list of {role, content} dicts (user/assistant turns)
        system_prompt: prepended as a system message if provided
        model: HF model ID
        max_tokens: maximum tokens to generate
        temperature: sampling temperature
        stream: if True, returns a generator for streaming output

    Returns:
        Full response string (stream=False) or a generator (stream=True).
    """
    try:
        from huggingface_hub import InferenceClient  # noqa: PLC0415
    except ImportError:
        err = "ERROR: huggingface_hub not installed. Run: pip install huggingface_hub"
        return (x for x in [err]) if stream else err

    token = get_hf_token()

    # Build final message list
    all_messages: list[dict] = []
    if system_prompt:
        all_messages.append({"role": "system", "content": system_prompt})
    all_messages.extend(messages)

    try:
        client = InferenceClient(model=model, token=token)

        if stream:
            return client.chat_completion(
                messages=all_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
        else:
            response = client.chat_completion(
                messages=all_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            )
            return response.choices[0].message.content or ""

    except Exception as e:
        err = f"LLM API error: {e}"
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
