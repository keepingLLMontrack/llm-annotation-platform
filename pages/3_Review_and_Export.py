"""
Page 3 — Review & Export.

Review all scenarios, inspect the flattened (one row per scenario+model) table,
and export it in the exact required column schema as CSV/JSON. Also offers a
quick view to compare scenarios that share annotators (inter-rater).
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from constants import EXPORT_COLUMNS, TARGET_MODELS
from scenarios import (
    delete_scenario,
    load_scenarios,
    scenario_to_rows,
    scenarios_to_csv,
    upload_export_file,
)
from ui import render_sidebar

st.set_page_config(page_title="Review & Export", page_icon="📤", layout="wide")
annotator = render_sidebar("Review & Export")

st.title("📤 3 · Review & Export")

scenarios = load_scenarios()
if not scenarios:
    st.info("No saved scenarios yet.")
    st.stop()

# -- Flattened preview -------------------------------------------------------
rows = []
for s in scenarios:
    rows.extend(scenario_to_rows(s))
df = pd.DataFrame(rows, columns=EXPORT_COLUMNS)

st.subheader(f"Export preview — {len(scenarios)} scenarios → {len(df)} rows")
st.caption("One row per (scenario, model); both rows share the same `scenario_id`.")
st.dataframe(df, use_container_width=True, hide_index=True)

# -- Completeness check ------------------------------------------------------
def _incomplete_reasons(s: dict) -> list[str]:
    issues = []
    if not s.get("distractor_goal"):
        issues.append("no distractor_goal")
    if not s.get("shared_distractor_turn_1"):
        issues.append("no first distractor turn")
    for m in TARGET_MODELS:
        run = s.get("runs", {}).get(m, {})
        if not run.get("turns"):
            issues.append(f"{m}: no runs")
        elif not all(t.get("label") for t in run["turns"]):
            issues.append(f"{m}: missing response label")
        if not run.get("overall_outcome"):
            issues.append(f"{m}: no overall outcome")
    return issues

incomplete = {s["scenario_id"]: _incomplete_reasons(s) for s in scenarios}
incomplete = {k: v for k, v in incomplete.items() if v}
if incomplete:
    with st.expander(f"⚠️ {len(incomplete)} scenario(s) incomplete", expanded=False):
        for sid, issues in incomplete.items():
            st.write(f"- `{sid}`: {', '.join(issues)}")

st.divider()

# -- Export ------------------------------------------------------------------
st.subheader("Download / upload export")
stamp = datetime.now().strftime("%Y%m%d_%H%M")
csv_text = scenarios_to_csv(scenarios)
json_text = json.dumps(scenarios, indent=2, ensure_ascii=False)

d1, d2, d3 = st.columns(3)
with d1:
    st.download_button(
        "⬇️ Download CSV (required format)",
        data=csv_text.encode("utf-8"),
        file_name=f"distractor_annotations_{stamp}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with d2:
    st.download_button(
        "⬇️ Download raw JSON (scenarios)",
        data=json_text.encode("utf-8"),
        file_name=f"scenarios_{stamp}.json",
        mime="application/json",
        use_container_width=True,
    )
with d3:
    if st.button("☁️ Upload CSV to HF repo", use_container_width=True):
        if upload_export_file(
            f"exports/distractor_annotations_{stamp}.csv",
            csv_text.encode("utf-8"),
        ):
            st.success("Uploaded CSV export to the dataset repo.")

st.divider()

# -- Inter-rater comparison --------------------------------------------------
st.subheader("👥 Inter-rater quick view")
st.caption("Scenarios that record a second annotator and/or disagreement notes.")
ir = [s for s in scenarios if s.get("second_annotator_name") or s.get("disagreement_notes")]
if ir:
    for s in ir:
        with st.expander(f"`{s['scenario_id']}` — {s.get('annotator_name')} / {s.get('second_annotator_name','?')}"):
            st.write(f"**Disagreement notes:** {s.get('disagreement_notes', '—')}")
            for m in TARGET_MODELS:
                run = s.get("runs", {}).get(m, {})
                labels = [t.get("label", "") for t in run.get("turns", [])]
                st.write(f"- **{m}** outcome _{run.get('overall_outcome','?')}_, labels: {labels}")
else:
    st.caption("None yet.")

st.divider()

# -- Manage scenarios --------------------------------------------------------
with st.expander("🗑️ Delete a scenario"):
    sid = st.selectbox(
        "Scenario to delete",
        options=[s["scenario_id"] for s in scenarios],
    )
    if st.button("Delete", type="secondary"):
        if delete_scenario(sid):
            st.success(f"Deleted `{sid}`.")
            st.rerun()
