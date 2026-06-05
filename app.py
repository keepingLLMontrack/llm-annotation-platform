"""
Dashboard - home page of the Distractor Annotation Tool.

Streamlit multipage: this file is the entry point.
All other pages live in pages/.
"""

import uuid
from collections import Counter
from datetime import datetime
import re

import pandas as pd
import streamlit as st

from constants import AVAILABLE_CHAT_MODELS, DEFAULT_CHAT_MODEL, KNOWN_DOMAINS
from dataset_io import (
    get_annotations_repo,
    get_hf_token,
    get_lmstudio_base_url,
    get_lmstudio_default_model,
    get_seed_data_path,
    load_annotations,
    load_seed_data,
    save_export_file,
    save_annotations,
)
from llm_client import chat_completion
from ui import render_sidebar

# -- Page config (must be first Streamlit call) -----------------------------
st.set_page_config(
    page_title="Distractor Annotation Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

annotator = render_sidebar("Dashboard")


def reset_new_entry_wizard() -> None:
    st.session_state["new_entry_step"] = 0
    st.session_state["new_entry_draft"] = {}
    st.session_state["new_entry_chat_messages"] = []


def _slugify_for_filename(text: str) -> str:
    value = (text or "").strip().lower().replace(" ", "_")
    value = re.sub(r"[^a-z0-9_-]", "", value)
    return value or "unknown"


st.session_state.setdefault("new_entry_step", 0)
st.session_state.setdefault("new_entry_draft", {})
st.session_state.setdefault("new_entry_chat_messages", [])
st.session_state.setdefault("finished_annotations", [])

# -- Header -----------------------------------------------------------------
st.title("🎯 Distractor Annotation Tool")
st.markdown(
    "**NLP - Keeping LLMs on Track in Task-Oriented Dialogue**  \n"
    "Collaborative annotation workspace for your research group."
)
st.divider()

# -- Load annotations --------------------------------------------------------
with st.spinner("Loading shared annotations..."):
    try:
        annotations = load_annotations()
    except Exception as exc:
        st.error(f"Error loading annotations: {exc}")
        annotations = []

# -- Summary metrics ---------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📝 Total Entries", len(annotations))
c2.metric("✅ Approved", sum(1 for a in annotations if a.get("_review_status") == "approved"))
c3.metric("📋 Draft", sum(1 for a in annotations if a.get("_review_status") == "draft"))
c4.metric(
    "🎭 Distractors",
    sum(len(a.get("distractors_multiturn", [])) for a in annotations),
)
c5.metric(
    "👥 Annotators",
    len({a.get("_annotator", "?") for a in annotations} - {"seed_data", "?"}),
)

st.divider()

# -- Configuration status ----------------------------------------------------
st.subheader("⚙️ Configuration")
hf_token = get_hf_token()
ann_repo = get_annotations_repo()

cfg1, cfg2 = st.columns(2)
with cfg1:
    if hf_token:
        st.success("✅ `HF_TOKEN` is set")
    else:
        st.error("❌ `HF_TOKEN` not set - add it to your Space secrets")
with cfg2:
    if ann_repo:
        st.success(f"✅ Annotations repo: `{ann_repo}`")
    else:
        st.error("❌ `ANNOTATIONS_REPO_ID` not set - add it to your Space secrets")

st.divider()

# -- New entry + LLM chat assistant -----------------------------------------
st.subheader("🧩 New Entry Assistant (LLM)")
st.caption("Create a new entry draft step-by-step, then open a model-backed chat window.")

overview_left, overview_right = st.columns([3, 1])
with overview_left:
    st.caption("Use the wizard to build entries, then export all finished annotations.")
with overview_right:
    st.metric("📦 Ready To Export", len(st.session_state.get("finished_annotations", [])))
    if st.button("📤 Export", use_container_width=True):
        finished = st.session_state.get("finished_annotations", [])
        if not finished:
            st.warning("No finished annotations to export yet.")
        elif not annotator:
            st.error("Enter your name in the sidebar first.")
        else:
            domains = {entry.get("domain", "") for entry in finished if entry.get("domain")}
            domain_name = next(iter(domains)) if len(domains) == 1 else "mixed"

            now = datetime.now()
            date_part = now.strftime("%d%m")
            time_part = now.strftime("%H%M")
            filename = (
                f"distractors_{_slugify_for_filename(annotator)}_"
                f"{_slugify_for_filename(domain_name)}_{date_part}_{time_part}.json"
            )

            if save_export_file(filename, finished):
                st.success(
                    f"✅ Exported {len(finished)} annotations to `{filename}` "
                    "in your configured HF dataset repo."
                )
                st.session_state["finished_annotations"] = []
                st.rerun()

act1, act2 = st.columns([2, 1])
with act1:
    if st.button("➕ Start New Entry", use_container_width=True, type="primary"):
        if not annotator:
            st.error("Enter your name in the sidebar first.")
        else:
            st.session_state["new_entry_draft"] = {
                "_annotator": annotator,
                "_created_at": datetime.now().isoformat(),
                "_updated_at": datetime.now().isoformat(),
                "_review_status": "draft",
                "_needs_human_review": True,
                "_llm_test_results": [],
            }
            st.session_state["new_entry_chat_messages"] = []
            st.session_state["new_entry_step"] = 1
            st.rerun()
with act2:
    if st.session_state.get("new_entry_step", 0) > 0:
        if st.button("↺ Reset Wizard", use_container_width=True):
            reset_new_entry_wizard()
            st.rerun()

step = st.session_state.get("new_entry_step", 0)
draft = st.session_state.get("new_entry_draft", {})

if step == 0:
    st.info("Click **Start New Entry** to begin.")

if step == 1:
    st.markdown("**Step 1/3 - Domain and Scenario**")
    with st.form("new_entry_domain_scenario_form"):
        chosen_domain = st.selectbox(
            "Select domain",
            options=KNOWN_DOMAINS,
            index=(
                KNOWN_DOMAINS.index(draft.get("domain"))
                if draft.get("domain") in KNOWN_DOMAINS
                else 0
            ),
            key="new_entry_domain",
        )
        scenario_text = st.text_area(
            "Scenario description",
            value=draft.get("scenario", ""),
            placeholder="Describe the user scenario you want to annotate...",
            height=140,
            key="new_entry_scenario",
        )
        confirm_step_1 = st.form_submit_button("Confirm Domain + Scenario")

    if confirm_step_1:
        scenario_text = (scenario_text or "").strip()
        if not scenario_text:
            st.error("Scenario description cannot be empty.")
        else:
            draft["domain"] = chosen_domain
            draft["scenario"] = scenario_text
            draft["_updated_at"] = datetime.now().isoformat()
            st.session_state["new_entry_draft"] = draft
            st.session_state["new_entry_step"] = 2
            st.rerun()

if step == 2:
    st.markdown("**Step 2/3 - Model + System Instruction**")
    with st.form("new_entry_model_instruction_form"):
        default_local_model = get_lmstudio_default_model() or DEFAULT_CHAT_MODEL
        candidate_models = AVAILABLE_CHAT_MODELS or [default_local_model]
        if default_local_model not in candidate_models:
            candidate_models = [default_local_model, *candidate_models]
        selected_model = st.selectbox(
            "Choose a chat model",
            options=candidate_models,
            index=(
                candidate_models.index(draft.get("model"))
                if draft.get("model") in candidate_models
                else (
                    candidate_models.index(default_local_model)
                    if default_local_model in candidate_models
                    else 0
                )
            ),
            key="new_entry_model_dropdown",
        )

        system_instruction = st.text_area(
            "System instruction",
            value=draft.get(
                "system_instruction",
                "",
            ),
            height=170,
            key="new_entry_system_instruction",
        )
        confirm_step_2 = st.form_submit_button("Open Chat Window")

    if confirm_step_2:
        system_instruction = (system_instruction or "").strip()
        if not system_instruction:
            st.error("System instruction cannot be empty.")
        else:
            draft["model"] = selected_model
            draft["system_instruction"] = system_instruction
            draft["_updated_at"] = datetime.now().isoformat()
            st.session_state["new_entry_draft"] = draft
            st.session_state["new_entry_step"] = 3
            st.rerun()

if step == 3:
    st.markdown("**Step 3/3 - Chat Window**")
    st.caption(
        f"Annotator: {draft.get('_annotator', '?')} | Domain: {draft.get('domain', '?')} | "
        f"Model: {draft.get('model', get_lmstudio_default_model() or DEFAULT_CHAT_MODEL)}"
    )
    st.caption(f"LM Studio endpoint: `{get_lmstudio_base_url()}`")
    st.caption(f"Finished annotations pending export: **{len(st.session_state.get('finished_annotations', []))}**")
    st.write(f"**Scenario:** {draft.get('scenario', '')}")
    st.write(f"**System instruction:** {draft.get('system_instruction', '')}")

    chat_messages = st.session_state.get("new_entry_chat_messages", [])
    for msg in chat_messages:
        with st.chat_message(msg.get("role", "assistant")):
            st.write(msg.get("content", ""))

    prompt = st.chat_input(
        "Send a message to the selected model",
        key="new_entry_chat_input",
    )
    if prompt:
        chat_messages.append({"role": "user", "content": prompt})
        response = chat_completion(
            messages=chat_messages,
            system_prompt=draft.get("system_instruction") or None,
            model=draft.get("model", get_lmstudio_default_model() or DEFAULT_CHAT_MODEL),
            stream=False,
        )
        chat_messages.append({"role": "assistant", "content": str(response)})
        st.session_state["new_entry_chat_messages"] = chat_messages
        st.rerun()

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        if st.button("✅ Finish", use_container_width=True):
            if not annotator:
                st.error("Enter your name in the sidebar first.")
            elif not chat_messages:
                st.error("Current annotation is empty. Send at least one chat message first.")
            else:
                conversation = [
                    {
                        "role": msg.get("role", ""),
                        "content": msg.get("content", ""),
                    }
                    for msg in chat_messages
                    if msg.get("role") in {"user", "assistant"} and msg.get("content")
                ]
                finished_entry = {
                    "domain": draft.get("domain", ""),
                    "scenario": draft.get("scenario", ""),
                    "system_instructions": draft.get("system_instruction", ""),
                    "conversation": conversation,
                }
                st.session_state["finished_annotations"] = [
                    *st.session_state.get("finished_annotations", []),
                    finished_entry,
                ]
                reset_new_entry_wizard()
                st.success("Annotation finished and queued for export.")
                st.rerun()

    with row1_col2:
        if st.button("✅ Finish + Export", use_container_width=True):
            if not annotator:
                st.error("Enter your name in the sidebar first.")
            elif not chat_messages:
                st.error("Current annotation is empty. Send at least one chat message first.")
            else:
                conversation = [
                    {
                        "role": msg.get("role", ""),
                        "content": msg.get("content", ""),
                    }
                    for msg in chat_messages
                    if msg.get("role") in {"user", "assistant"} and msg.get("content")
                ]
                finished_entry = {
                    "domain": draft.get("domain", ""),
                    "scenario": draft.get("scenario", ""),
                    "system_instructions": draft.get("system_instruction", ""),
                    "conversation": conversation,
                }
                finished = [
                    *st.session_state.get("finished_annotations", []),
                    finished_entry,
                ]

                domains = {entry.get("domain", "") for entry in finished if entry.get("domain")}
                domain_name = next(iter(domains)) if len(domains) == 1 else "mixed"

                now = datetime.now()
                date_part = now.strftime("%d%m")
                time_part = now.strftime("%H%M")
                filename = (
                    f"distractors_{_slugify_for_filename(annotator)}_"
                    f"{_slugify_for_filename(domain_name)}_{date_part}_{time_part}.json"
                )

                if save_export_file(filename, finished):
                    st.success(
                        f"✅ Exported {len(finished)} annotations to `{filename}` "
                        "in your configured HF dataset repo."
                    )
                    st.session_state["finished_annotations"] = []
                    reset_new_entry_wizard()
                    st.rerun()

    with row2_col1:
        if st.button("🚪 Exit", use_container_width=True):
            reset_new_entry_wizard()
            st.info("Current in-progress annotation was discarded.")
            st.rerun()

    with row2_col2:
        if st.button("🛑 Exit + Export", use_container_width=True):
            finished = st.session_state.get("finished_annotations", [])
            if not finished:
                st.warning("No finished annotations to export yet.")
            elif not annotator:
                st.error("Enter your name in the sidebar first.")
            else:
                domains = {entry.get("domain", "") for entry in finished if entry.get("domain")}
                domain_name = next(iter(domains)) if len(domains) == 1 else "mixed"

                now = datetime.now()
                date_part = now.strftime("%d%m")
                time_part = now.strftime("%H%M")
                filename = (
                    f"distractors_{_slugify_for_filename(annotator)}_"
                    f"{_slugify_for_filename(domain_name)}_{date_part}_{time_part}.json"
                )

                if save_export_file(filename, finished):
                    st.success(
                        f"✅ Exported {len(finished)} annotations to `{filename}` "
                        "in your configured HF dataset repo."
                    )
                    st.session_state["finished_annotations"] = []
                    reset_new_entry_wizard()
                    st.rerun()

st.divider()

# -- data import --------------------------------------------------------
st.subheader("📦 Import Annotation Data")
seed = load_seed_data()
if seed:
    st.write(
        f"**{len(seed)}** entries are bundled as seed data "
    )
    if len(annotations) == 0:
        st.info("The shared repo is empty. Import the seed data to get started.")

    col_imp, col_exp, col_info = st.columns([1, 1, 2])
    with col_imp:
        if st.button("🌱 Import Data", use_container_width=True, type="primary"):
            if not annotator:
                st.error("Enter your name in the sidebar first.")
            elif len(annotations) > 0:
                st.warning(
                    "Shared repo already has entries. "
                    "Clear all annotations on the **👥 Annotations** page first if you want to re-import."
                )
            else:
                tagged = []
                for entry in seed:
                    ec = dict(entry)
                    ec["_id"] = ec.get("_id") or str(uuid.uuid4())
                    ec.setdefault("_annotator", "seed_data")
                    ec.setdefault("_created_at", datetime.now().isoformat())
                    ec["_updated_at"] = datetime.now().isoformat()
                    ec.setdefault("_review_status", "draft")
                    ec.setdefault("_needs_human_review", True)
                    ec.setdefault("_llm_test_results", [])
                    tagged.append(ec)
                if save_annotations(tagged):
                    st.success(f"✅ Imported {len(tagged)} seed entries into the shared repo!")
                    st.rerun()
    with col_exp:
        if st.button("📤 Export Imported Data", use_container_width=True):
            if not annotator:
                st.error("Enter your name in the sidebar first.")
            elif not annotations:
                st.warning("No imported data found to export yet.")
            else:
                now = datetime.now()
                date_part = now.strftime("%d%m")
                time_part = now.strftime("%H%M")
                filename = (
                    f"import_export_{_slugify_for_filename(annotator)}_"
                    f"{date_part}_{time_part}.json"
                )

                if save_export_file(filename, annotations):
                    st.success(
                        f"✅ Exported {len(annotations)} imported entries to `{filename}` "
                        "in your configured HF dataset repo."
                    )
    with col_info:
        st.caption(
            "Seed data was generated with Gemma-4-E2B running locally via LM Studio, "
            "then reviewed and partially rewritten by your groupmate."
        )
else:
    expected_paths = [
        "seed_data/draft_distractors.json",
        "data/draft_distractors.json",
    ]
    st.warning(
        "Seed data file not found. Add `draft_distractors.json` to one of these locations: "
        + ", ".join(f"`{path}`" for path in expected_paths)
    )
    resolved_seed_path = get_seed_data_path()
    if resolved_seed_path is not None:
        st.caption(f"Detected seed data path: `{resolved_seed_path}`")

st.divider()

# -- Two-column section ------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Annotator Progress")
    if annotations:
        stats: dict[str, dict] = {}
        for a in annotations:
            ann = a.get("_annotator", "Unknown")
            stats.setdefault(ann, {"Entries": 0, "Distractors": 0, "Approved": 0})
            stats[ann]["Entries"] += 1
            stats[ann]["Distractors"] += len(a.get("distractors_multiturn", []))
            if a.get("_review_status") == "approved":
                stats[ann]["Approved"] += 1
        df = pd.DataFrame(
            [{"Annotator": k, **v} for k, v in stats.items()]
        ).sort_values("Entries", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No annotations yet - import seed data below or go to **Annotate**.")

with col_right:
    st.subheader("📋 Status & Domain Breakdown")
    if annotations:
        statuses = Counter(a.get("_review_status", "draft") for a in annotations)
        df_s = pd.DataFrame([{"Status": k, "Count": v} for k, v in statuses.most_common()])
        st.dataframe(df_s, use_container_width=True, hide_index=True)

        st.markdown("**By domain:**")
        domains = Counter(a.get("domain", "unknown") for a in annotations)
        for dom, cnt in domains.most_common():
            st.write(f"  - {dom}: **{cnt}**")
    else:
        st.info("No data yet.")

st.divider()

# -- Recent activity ---------------------------------------------------------
if annotations:
    st.subheader("🕒 Recent Activity")
    recent = sorted(
        annotations,
        key=lambda x: x.get("_updated_at", ""),
        reverse=True,
    )[:8]
    for entry in recent:
        scenario_preview = entry.get("scenario", "Unknown")[:70]
        status_emoji = {"approved": "✅", "draft": "📋", "failed": "❌"}.get(
            entry.get("_review_status", "draft"), "❓"
        )
        with st.expander(
            f"{status_emoji} [{entry.get('domain', '?')}] {scenario_preview}"
        ):
            rc1, rc2, rc3 = st.columns(3)
            rc1.write(f"**Annotator:** {entry.get('_annotator', '?')}")
            rc2.write(f"**Status:** {entry.get('_review_status', 'draft')}")
            rc3.write(f"**Distractors:** {len(entry.get('distractors_multiturn', []))}")
            updated = entry.get("_updated_at", "")[:16].replace("T", " ")
            st.caption(f"Last updated: {updated}")
