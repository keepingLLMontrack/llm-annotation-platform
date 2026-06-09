"""
Model registry for the two fixed target models.

Serving is done "as currently": an OpenAI-compatible endpoint (LM Studio /
Ollama / vLLM / any hosted gateway). Each model resolves its own base_url,
api_key and served model-id from environment / Streamlit secrets, falling back
to the shared LM Studio defaults used elsewhere in the app.

To serve the two models from different places (e.g. Llama on local LM Studio,
gpt-oss on a hosted endpoint), set the per-model *_BASE_URL / *_API_KEY /
*_MODEL variables. Otherwise both use the shared OPENAI_BASE_URL.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st

from constants import (
    CANONICAL_MODEL_NAME,
    MODEL_GPTOSS,
    MODEL_LLAMA,
    TARGET_MODELS,
)


def _get(name: str) -> Optional[str]:
    """Read a setting from env first, then Streamlit secrets."""
    val = os.environ.get(name)
    if val:
        return val
    try:
        return st.secrets.get(name)
    except Exception:
        return None


def _session_get(key: str, default=None):
    """Read from st.session_state, tolerating a non-running Streamlit context."""
    try:
        return st.session_state.get(key, default)
    except Exception:
        return default


def _shared_base_url() -> str:
    return (
        _get("OPENAI_BASE_URL")
        or _get("LMSTUDIO_BASE_URL")
        or "http://localhost:1234/v1"
    )


def _shared_api_key() -> str:
    return _get("OPENAI_API_KEY") or _get("LMSTUDIO_API_KEY") or "lm-studio"


# Per-model override variable names + the served-model-id default.
# The default served-model-id is the canonical HF id; override it (via the
# Model Setup page or the *_MODEL env var) if your server expects a different
# name (e.g. an LM Studio model label).
_REGISTRY = {
    MODEL_LLAMA: {
        "base_url_env": "LLAMA_BASE_URL",
        "api_key_env": "LLAMA_API_KEY",
        "model_id_env": "LLAMA_MODEL",
        "default_model_id": "meta-llama/Llama-3.1-8B-Instruct",
    },
    MODEL_GPTOSS: {
        "base_url_env": "GPTOSS_BASE_URL",
        "api_key_env": "GPTOSS_API_KEY",
        "model_id_env": "GPTOSS_MODEL",
        "default_model_id": "openai/gpt-oss-20b",
    },
}

# Session-state keys for per-user UI overrides / enable toggles.
OVERRIDES_KEY = "model_overrides"   # {display_name: {served_model_id, base_url, api_key}}
ENABLED_KEY = "model_enabled"       # {display_name: bool}


def list_models() -> list[str]:
    """Display names of the two fixed target models."""
    return list(TARGET_MODELS)


def is_model_enabled(display_name: str) -> bool:
    """
    Whether the user has chosen to load/run this model. Default: enabled.
    Toggled on the Model Setup page; lets a user run only one model if the
    other isn't loaded.
    """
    enabled = _session_get(ENABLED_KEY, None) or {}
    return bool(enabled.get(display_name, True))


def resolve_model(display_name: str) -> dict:
    """
    Return everything needed to call a model and to label the export row.

    Resolution order for each field: UI override (session) → env/secrets →
    code default.

    Keys:
      display_name    – friendly name shown in the UI
      canonical_name  – HF id written to the export's `model_name` column
      served_model_id – id passed to the OpenAI-compatible API
      base_url        – endpoint
      api_key         – api key
      enabled         – whether the user has chosen to load/run it
    """
    spec = _REGISTRY[display_name]
    ov = (_session_get(OVERRIDES_KEY, {}) or {}).get(display_name, {})
    return {
        "display_name": display_name,
        "canonical_name": CANONICAL_MODEL_NAME[display_name],
        "served_model_id": ov.get("served_model_id") or _get(spec["model_id_env"]) or spec["default_model_id"],
        "base_url": ov.get("base_url") or _get(spec["base_url_env"]) or _shared_base_url(),
        "api_key": ov.get("api_key") or _get(spec["api_key_env"]) or _shared_api_key(),
        "default_model_id": spec["default_model_id"],
        "enabled": is_model_enabled(display_name),
    }
