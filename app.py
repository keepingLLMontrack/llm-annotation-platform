"""
Dashboard — home page of the Distractor Annotation Tool.

Streamlit multipage: this file is the entry point.
All other pages live in pages/.
"""

import os
import uuid
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.dataset_io import load_annotations, load_seed_data, save_annotations
from utils.ui import render_sidebar

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="Distractor Annotation Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

annotator = render_sidebar("Dashboard")

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🎯 Distractor Annotation Tool")
st.markdown(
    "**MSc NLP — Keeping LLMs on Track in Task-Oriented Dialogue**  \n"
    "Collaborative annotation workspace for your research group."
)
st.divider()

# ── Load annotations ───────────────────────────────────────────────────────
with st.spinner("Loading shared annotations…"):
    try:
        annotations = load_annotations()
    except Exception as exc:
        st.error(f"Error loading annotations: {exc}")
        annotations = []

# ── Summary metrics ────────────────────────────────────────────────────────
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

# ── Two-column section ─────────────────────────────────────────────────────
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
        st.info("No annotations yet — import seed data below or go to **Annotate**.")

with col_right:
    st.subheader("📋 Status & Domain Breakdown")
    if annotations:
        statuses = Counter(a.get("_review_status", "draft") for a in annotations)
        df_s = pd.DataFrame([{"Status": k, "Count": v} for k, v in statuses.most_common()])
        st.dataframe(df_s, use_container_width=True, hide_index=True)

        st.markdown("**By domain:**")
        domains = Counter(a.get("domain", "unknown") for a in annotations)
        for dom, cnt in domains.most_common():
            st.write(f"  • {dom}: **{cnt}**")
    else:
        st.info("No data yet.")

st.divider()

# ── Recent activity ────────────────────────────────────────────────────────
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

    st.divider()

# ── Seed data import ───────────────────────────────────────────────────────
st.subheader("📦 Seed Data (Initial Group Work)")
seed = load_seed_data()
if seed:
    st.write(
        f"**{len(seed)}** entries are bundled as seed data "
        f"(20 legal-domain scenarios generated and reviewed by the group)."
    )
    if len(annotations) == 0:
        st.info("The shared repo is empty. Import the seed data to get started.")

    col_imp, col_info = st.columns([1, 3])
    with col_imp:
        if st.button("🌱 Import Seed Data", use_container_width=True, type="primary"):
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
    with col_info:
        st.caption(
            "Seed data was generated with Gemma-4-E2B running locally via LM Studio, "
            "then reviewed and partially rewritten by your groupmate."
        )
else:
    st.warning("seed_data/draft_distractors.json not found.")

st.divider()

# ── Configuration status ───────────────────────────────────────────────────
st.subheader("⚙️ Configuration")
hf_token = os.environ.get("HF_TOKEN") or (
    st.secrets.get("HF_TOKEN") if hasattr(st, "secrets") else None
)
ann_repo = os.environ.get("ANNOTATIONS_REPO_ID") or (
    st.secrets.get("ANNOTATIONS_REPO_ID") if hasattr(st, "secrets") else None
)

cfg1, cfg2 = st.columns(2)
with cfg1:
    if hf_token:
        st.success("✅ `HF_TOKEN` is set")
    else:
        st.error("❌ `HF_TOKEN` not set — add it to your Space secrets")
with cfg2:
    if ann_repo:
        st.success(f"✅ Annotations repo: `{ann_repo}`")
    else:
        st.error("❌ `ANNOTATIONS_REPO_ID` not set — add it to your Space secrets")
