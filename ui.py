"""
Shared UI components for the Distractor Annotation Tool.
Call render_sidebar() at the top of every page.
"""

import os
import streamlit as st

from dataset_io import get_annotations_repo, get_hf_token


def render_sidebar(page_title: str = "") -> str:
    """
    Render the common sidebar. Returns the current annotator name.
    Uses a unique widget key per page to avoid Streamlit key collisions.
    """
    with st.sidebar:
        st.markdown("## 🎯 Annotation Tool")
        st.caption("NLP with DL — Research")
        st.divider()

        # ── Annotator identity ──────────────────────────────────────────────
        current_name = st.session_state.get("annotator_name", "")

        if current_name:
            st.success(f"👤 **{current_name}**")
            if st.button("✏️ Change name", use_container_width=True):
                st.session_state["annotator_name"] = ""
                st.rerun()
        else:
            name = st.text_input(
                "Your name",
                placeholder="e.g. Alice",
                # unique key per page avoids DuplicateWidgetID errors
                key=f"sidebar_name__{page_title.replace(' ', '_')}",
            )
            if name.strip():
                st.session_state["annotator_name"] = name.strip()
                st.rerun()
            else:
                st.warning("⚠️ Enter your name to annotate")

        st.divider()

        # ── Config status ───────────────────────────────────────────────────
        token_ok = bool(get_hf_token())
        repo_ok = bool(get_annotations_repo())

        if token_ok and repo_ok:
            st.caption("🔗 Connected to shared repo")
        else:
            if not token_ok:
                st.error("HF_TOKEN missing")
            if not repo_ok:
                st.error("ANNOTATIONS_REPO_ID missing")

        st.divider()
        st.caption("📌 Links")
        st.markdown("[Base Dataset](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset)")
        st.markdown("[EMNLP Paper](https://aclanthology.org/2024.findings-emnlp.713)")
        st.markdown("[arXiv:2511.05018](https://arxiv.org/abs/2511.05018)")
        st.markdown("[Annotation Dataset](https://huggingface.co/datasets/keepingLLMontrack/distractor-annotations)")
        
    return st.session_state.get("annotator_name", "")


def render_conversation(conversation: list[dict], highlight_turn: str = "") -> None:
    """Render a conversation as chat bubbles. Highlights matching bot_turn if given."""
    for turn in conversation:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        is_highlight = highlight_turn and role == "assistant" and content == highlight_turn
        with st.chat_message(role):
            if is_highlight:
                st.markdown(f"**⬇️ Injection point:**\n\n{content}")
            else:
                st.write(content)


def render_distractor_multiturn(distractor: dict, idx: int) -> None:
    """Render a single multi-turn distractor entry."""
    turns = distractor.get("turns", [])
    st.markdown(f"**Subject:** {distractor.get('off_topic_subject', '—')}")
    st.markdown(f"**Tactic:** `{distractor.get('tactic_used', '—')}`")
    st.markdown(f"**Injected after:** _{distractor.get('bot_turn', '—')[:80]}..._")
    st.markdown(f"**Turns:** {len(turns)}")
    with st.expander("Show turns"):
        for t in turns:
            with st.chat_message(t.get("role", "user")):
                st.write(t.get("content", ""))
