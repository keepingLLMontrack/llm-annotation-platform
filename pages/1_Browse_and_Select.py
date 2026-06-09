"""
Page 1 — Browse & Select.

Pick an existing conversation from the NVIDIA dataset, then pick the user turn
where a distractor will be injected. This seeds a new scenario draft that you
then fill in on the Annotate page.
"""

from __future__ import annotations

import streamlit as st

from constants import KNOWN_DOMAINS
from scenarios import load_source_conversations, new_scenario
from ui import render_sidebar

st.set_page_config(page_title="Browse & Select", page_icon="🔎", layout="wide")
annotator = render_sidebar("Browse & Select")

st.title("🔎 1 · Browse & Select a conversation")
st.caption(
    "Choose a real conversation, then choose the user turn to replace with your "
    "first distractor. The model only ever sees the turns **before** that point."
)

with st.spinner("Loading source dataset (first time can take a minute)..."):
    conversations = load_source_conversations()

if not conversations:
    st.error(
        "Could not load the source dataset. Make sure `datasets` is installed; "
        "for gated access set `HF_TOKEN`."
    )
    st.stop()

# -- Filter ------------------------------------------------------------------
col_a, col_b = st.columns([1, 2])
with col_a:
    domains_present = [d for d in KNOWN_DOMAINS if any(c["domain"] == d for c in conversations)]
    domain = st.selectbox("Domain", options=domains_present or KNOWN_DOMAINS)
with col_b:
    search = st.text_input("Filter scenarios (substring match)", "")

subset = [c for c in conversations if c["domain"] == domain]
if search.strip():
    s = search.lower()
    subset = [c for c in subset if s in c.get("scenario", "").lower()]

st.caption(f"{len(subset)} conversation(s) in **{domain}**.")
if not subset:
    st.stop()

# -- Pick a conversation -----------------------------------------------------
def _label(c: dict) -> str:
    return f"{c['conversation_id']} — {c.get('scenario', '')[:80]}"

choice = st.selectbox(
    "Conversation",
    options=subset,
    format_func=_label,
)

st.markdown(f"**Scenario:** {choice.get('scenario', '')}")
with st.expander("📜 System instruction (domain policy)", expanded=False):
    st.write(choice.get("system_instruction", ""))

# -- Show conversation with turn indices -------------------------------------
st.subheader("Conversation turns")
conversation = choice.get("conversation", [])
user_turn_indices = []
for idx, turn in enumerate(conversation):
    role = turn.get("role", "?")
    content = turn.get("content", "")
    if role == "user":
        user_turn_indices.append(idx)
    with st.chat_message(role if role in ("user", "assistant") else "user"):
        st.markdown(f"`turn {idx}` · **{role}**")
        st.write(content)

if not user_turn_indices:
    st.warning("This conversation has no user turn to replace.")
    st.stop()

st.divider()

# -- Pick injection point ----------------------------------------------------
st.subheader("Pick the injection point")
st.caption(
    "Select the **user** turn to replace with your distractor. A good point is "
    "one where a user could naturally try to derail the conversation."
)

def _turn_label(i: int) -> str:
    preview = conversation[i].get("content", "")[:90]
    return f"turn {i}: {preview}"

selected_turn_id = st.selectbox(
    "User turn to replace",
    options=user_turn_indices,
    format_func=_turn_label,
)

base_context = conversation[:selected_turn_id]
original_user_message = conversation[selected_turn_id].get("content", "")

st.markdown("**Context the model will receive (turns before the injection):**")
if base_context:
    for t in base_context:
        with st.chat_message(t.get("role", "user")):
            st.write(t.get("content", ""))
else:
    st.info("No prior turns — the distractor will be the very first user message.")

st.markdown(f"**Original user message at this turn (will be replaced):**\n\n> {original_user_message}")

st.divider()
if st.button("➡️ Start a scenario from this turn", type="primary", use_container_width=True):
    if not annotator:
        st.error("Enter your name in the sidebar first.")
    else:
        scenario = new_scenario(annotator)
        scenario["domain"] = choice["domain"]
        scenario["conversation_id"] = choice["conversation_id"]
        scenario["selected_turn_id"] = selected_turn_id
        scenario["system_prompt"] = choice.get("system_instruction", "")
        scenario["system_prompt_or_policy_summary"] = (
            choice.get("system_instruction", "")[:300]
        )
        scenario["base_context"] = base_context
        scenario["original_user_message"] = original_user_message
        st.session_state["active_scenario"] = scenario
        st.success(
            f"Scenario `{scenario['scenario_id']}` created. "
            "Open **2 · Annotate** in the sidebar to continue."
        )
        st.page_link("pages/2_Annotate.py", label="➡️ Go to Annotate", icon="✍️")
