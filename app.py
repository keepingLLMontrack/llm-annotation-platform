"""
Distractor Annotation Tool — Dashboard (entry point).

Workflow (see the pages in the sidebar):
  1. Browse & Select — pick a conversation from the dataset and an injection turn
  2. Annotate        — define the distractor, run both models, label responses
  3. Review & Export — review scenarios and export the required 2-rows-per-scenario table
"""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from constants import (
    KNOWN_DOMAINS,
    SCENARIOS_PER_DOMAIN_TARGET,
    TARGET_MODELS,
    TOTAL_SCENARIOS_TARGET,
)
from scenarios import load_scenarios
from ui import render_sidebar

st.set_page_config(
    page_title="Distractor Annotation Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

annotator = render_sidebar("Dashboard")

st.title("🎯 Distractor Annotation Tool")
st.markdown(
    "**NLP — Keeping LLMs on Track in Task-Oriented Dialogue**  \n"
    "Create multi-turn distractor scenarios, run them against "
    f"**{TARGET_MODELS[0]}** and **{TARGET_MODELS[1]}**, and annotate whether "
    "each model stays on track."
)
st.divider()

with st.spinner("Loading scenarios..."):
    scenarios = load_scenarios()


def _model_run_complete(run: dict) -> bool:
    turns = run.get("turns", [])
    return len(turns) >= 1 and bool(run.get("overall_outcome"))


def _scenario_complete(s: dict) -> bool:
    runs = s.get("runs", {})
    return all(_model_run_complete(runs.get(m, {})) for m in TARGET_MODELS)


total = len(scenarios)
complete = sum(1 for s in scenarios if _scenario_complete(s))

c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Scenarios", total)
c2.metric("✅ Fully annotated", complete)
c3.metric("🎯 Target", TOTAL_SCENARIOS_TARGET)
c4.metric(
    "👥 Annotators",
    len({s.get("annotator_name", "") for s in scenarios} - {""}),
)

st.progress(min(total / TOTAL_SCENARIOS_TARGET, 1.0) if TOTAL_SCENARIOS_TARGET else 0.0)
st.caption(
    f"{total} / {TOTAL_SCENARIOS_TARGET} scenarios created "
    f"(target ≈ {SCENARIOS_PER_DOMAIN_TARGET} per domain across {len(KNOWN_DOMAINS)} domains)."
)
st.divider()

# -- Per-domain quota --------------------------------------------------------
st.subheader("📊 Progress by domain")
counts = Counter(s.get("domain", "?") for s in scenarios)
complete_counts = Counter(s.get("domain", "?") for s in scenarios if _scenario_complete(s))
rows = []
for dom in KNOWN_DOMAINS:
    rows.append(
        {
            "Domain": dom,
            "Scenarios": counts.get(dom, 0),
            "Fully annotated": complete_counts.get(dom, 0),
            "Target": SCENARIOS_PER_DOMAIN_TARGET,
            "Status": "✅" if counts.get(dom, 0) >= SCENARIOS_PER_DOMAIN_TARGET else "⬜",
        }
    )
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.divider()

# -- Getting started ---------------------------------------------------------
if total == 0:
    st.info(
        "No scenarios yet. Open **1 · Browse & Select** in the sidebar to pick a "
        "conversation and start your first distractor scenario."
    )
else:
    st.subheader("🕒 Recent scenarios")
    recent = sorted(scenarios, key=lambda x: x.get("_updated_at", ""), reverse=True)[:8]
    for s in recent:
        done = "✅" if _scenario_complete(s) else "📝"
        with st.expander(
            f"{done} [{s.get('domain', '?')}] {s.get('scenario_id', '?')} — "
            f"{(s.get('distractor_goal') or 'no goal yet')[:70]}"
        ):
            st.write(f"**Annotator:** {s.get('annotator_name', '?')}")
            st.write(f"**Conversation:** `{s.get('conversation_id', '?')}` "
                     f"turn `{s.get('selected_turn_id', '?')}`")
            st.write(f"**Distractor goal:** {s.get('distractor_goal', '—')}")
            for m in TARGET_MODELS:
                run = s.get("runs", {}).get(m, {})
                st.write(
                    f"- **{m}** — {len(run.get('turns', []))} turn(s), "
                    f"outcome: _{run.get('overall_outcome') or 'not set'}_"
                )

st.divider()
st.caption(
    "Configure secrets (`HF_TOKEN`, `ANNOTATIONS_REPO_ID`) to share scenarios across the "
    "group. Set the model endpoints via `OPENAI_BASE_URL` (and optional per-model "
    "`LLAMA_*` / `GPTOSS_*` overrides)."
)
