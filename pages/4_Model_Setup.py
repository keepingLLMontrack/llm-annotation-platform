"""
Page 4 — Model Setup.

The two target models ship with their canonical ids already in the code. Here
each user chooses whether to LOAD/run each model and can override its served
id / endpoint / key to match their own server (e.g. LM Studio). Choices are
per-session.

This does NOT remove a model from the export — both models always produce a
row. Disabling one just stops you from running it (useful when only one model
is loaded locally).
"""

from __future__ import annotations

import streamlit as st

from constants import TARGET_MODELS
from llm_client import run_chat
from models import ENABLED_KEY, OVERRIDES_KEY, resolve_model
from ui import render_sidebar

st.set_page_config(page_title="Model Setup", page_icon="⚙️", layout="wide")
render_sidebar("Model Setup")

st.title("⚙️ 4 · Model Setup")
st.caption(
    "Both models are pre-configured with their canonical ids. Choose which to "
    "load, and override the served id / endpoint if your server uses different "
    "values. Settings apply to your current session."
)

st.session_state.setdefault(OVERRIDES_KEY, {})
st.session_state.setdefault(ENABLED_KEY, {})

st.info(
    "Serving is via any OpenAI-compatible endpoint (LM Studio, vLLM, a hosted "
    "gateway, …). For LM Studio: load the model, **Start Server**, and set the "
    "served id below to the exact name LM Studio shows. `gpt-oss-20b` is large — "
    "if you can't run it locally, point it at a hosted endpoint or leave it "
    "disabled and document that.",
    icon="ℹ️",
)

for model_display in TARGET_MODELS:
    cfg = resolve_model(model_display)
    with st.container(border=True):
        st.subheader(model_display)
        st.caption(f"Export `model_name`: `{cfg['canonical_name']}`")

        enabled = st.checkbox(
            "Enabled — loaded & ready to run",
            value=cfg["enabled"],
            key=f"enabled_{model_display}",
            help="Uncheck if this model isn't loaded; you can still run the other one.",
        )
        st.session_state[ENABLED_KEY][model_display] = enabled

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            served = st.text_input(
                "Served model id (sent to the API)",
                value=cfg["served_model_id"],
                key=f"served_{model_display}",
                help=f"Code default: {cfg['default_model_id']}",
            )
        with c2:
            base_url = st.text_input(
                "Base URL", value=cfg["base_url"], key=f"baseurl_{model_display}"
            )
        with c3:
            api_key = st.text_input(
                "API key", value=cfg["api_key"], type="password",
                key=f"apikey_{model_display}",
            )

        st.session_state[OVERRIDES_KEY][model_display] = {
            "served_model_id": served.strip(),
            "base_url": base_url.strip(),
            "api_key": api_key.strip(),
        }

        if st.button(f"🔌 Test connection — {model_display}", key=f"test_{model_display}"):
            if not enabled:
                st.warning("Model is disabled. Enable it first.")
            else:
                with st.spinner("Pinging endpoint..."):
                    result = run_chat(
                        [{"role": "user", "content": "Reply with the single word: ok"}],
                        resolve_model(model_display),
                        max_tokens=8,
                    )
                if result["ok"]:
                    st.success(f"✅ Reachable. Reply: {result['final'][:120]}")
                else:
                    st.error(result["error"])

st.divider()
st.caption(
    "Durable/group-wide endpoints can also be set via env vars or "
    "`.streamlit/secrets.toml` (`OPENAI_BASE_URL`, `LLAMA_MODEL`, `GPTOSS_MODEL`, "
    "`GPTOSS_BASE_URL`, …). UI overrides here take precedence for your session."
)
