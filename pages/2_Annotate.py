"""
Page 2 — Annotate.

The core workflow for one scenario:
  A. Define the distractor (goal, targeted rule, why it's good, shared turn-1)
  B. For each model: run turn-by-turn, read the response, label it, write the
     next distractor turn (interactively, up to 3 turns), set overall outcome
  C. Inter-rater fields + save

The first distractor turn is SHARED across both models; later turns may differ.
Only the final, user-facing response is annotated (reasoning is shown greyed).
"""

from __future__ import annotations

import streamlit as st

from constants import (
    DISTRACTOR_STRATEGIES,
    MAX_DISTRACTOR_TURNS,
    OVERALL_OUTCOMES,
    RESPONSE_LABELS,
    TARGET_MODELS,
)
from llm_client import build_messages, run_chat
from models import resolve_model
from scenarios import load_scenarios, new_model_run, upsert_scenario
from ui import render_sidebar

st.set_page_config(page_title="Annotate", page_icon="✍️", layout="wide")
annotator = render_sidebar("Annotate")

st.title("✍️ 2 · Annotate")

scenario = st.session_state.get("active_scenario")

# -- Load an existing scenario instead ---------------------------------------
with st.expander("📂 Load an existing scenario to continue / edit", expanded=scenario is None):
    existing = load_scenarios()
    if existing:
        opt = st.selectbox(
            "Saved scenarios",
            options=existing,
            format_func=lambda s: f"{s.get('scenario_id')} · [{s.get('domain')}] "
            f"{(s.get('distractor_goal') or 'no goal')[:50]}",
            key="load_existing_select",
        )
        if st.button("Load selected scenario"):
            # ensure runs dict has both models
            opt.setdefault("runs", {})
            for m in TARGET_MODELS:
                opt["runs"].setdefault(m, new_model_run(m))
            st.session_state["active_scenario"] = opt
            st.rerun()
    else:
        st.caption("No saved scenarios yet.")

if scenario is None:
    st.info(
        "No active scenario. Start one from **1 · Browse & Select**, or load an "
        "existing one above."
    )
    st.stop()

# ===========================================================================
# Context
# ===========================================================================
st.subheader(f"Scenario `{scenario['scenario_id']}` — {scenario.get('domain', '?')}")
meta1, meta2, meta3 = st.columns(3)
meta1.metric("Conversation", scenario.get("conversation_id", "?"))
meta2.metric("Injection turn", scenario.get("selected_turn_id", "?"))
meta3.metric("Annotator", scenario.get("annotator_name", "?"))

with st.expander("📜 System prompt (passed as the system message)", expanded=False):
    st.write(scenario.get("system_prompt", ""))
with st.expander("💬 Prior turns the model receives", expanded=False):
    if scenario.get("base_context"):
        for t in scenario["base_context"]:
            with st.chat_message(t.get("role", "user")):
                st.write(t.get("content", ""))
    else:
        st.caption("No prior turns — distractor is the first user message.")
st.markdown(f"**Original user message (replaced):** _{scenario.get('original_user_message', '')}_")

scenario["system_prompt_or_policy_summary"] = st.text_area(
    "Policy summary (for the export `system_prompt_or_policy_summary` column)",
    value=scenario.get("system_prompt_or_policy_summary", ""),
    height=80,
)

st.divider()

# ===========================================================================
# A. Distractor definition (shared across both models)
# ===========================================================================
st.subheader("A · Distractor definition (shared)")
scenario["distractor_goal"] = st.text_area(
    "Distractor goal — what are you trying to make the model do that it should not?",
    value=scenario.get("distractor_goal", ""),
    placeholder="e.g. make the banking assistant provide investment advice, "
    "which is outside its allowed scope.",
    height=70,
)

col_s1, col_s2 = st.columns(2)
with col_s1:
    strat_default = scenario.get("targeted_rule_or_scope_boundary", "")
    strat_idx = (
        DISTRACTOR_STRATEGIES.index(strat_default)
        if strat_default in DISTRACTOR_STRATEGIES
        else len(DISTRACTOR_STRATEGIES) - 1  # "other"
    )
    strat_choice = st.selectbox(
        "Targeted rule / scope boundary (strategy)",
        options=DISTRACTOR_STRATEGIES,
        index=strat_idx,
    )
    if strat_choice == "other":
        scenario["targeted_rule_or_scope_boundary"] = st.text_input(
            "Describe the targeted rule / boundary",
            value=strat_default if strat_default not in DISTRACTOR_STRATEGIES else "",
        )
    else:
        scenario["targeted_rule_or_scope_boundary"] = strat_choice
with col_s2:
    scenario["why_this_is_a_good_distractor"] = st.text_area(
        "Why is this a good distractor?",
        value=scenario.get("why_this_is_a_good_distractor", ""),
        height=70,
    )

scenario["shared_distractor_turn_1"] = st.text_area(
    "🎯 Shared first distractor turn (identical for BOTH models)",
    value=scenario.get("shared_distractor_turn_1", ""),
    placeholder="The first off-topic user message...",
    height=90,
)

# Shared decoding
dcol1, dcol2 = st.columns(2)
with dcol1:
    temperature = st.slider("temperature", 0.0, 1.0, float(scenario["runs"][TARGET_MODELS[0]].get("temperature", 0.2)), 0.05)
with dcol2:
    top_p = st.slider("top_p", 0.0, 1.0, float(scenario["runs"][TARGET_MODELS[0]].get("top_p", 1.0)), 0.05)
for m in TARGET_MODELS:
    scenario["runs"][m]["temperature"] = temperature
    scenario["runs"][m]["top_p"] = top_p

st.divider()

# ===========================================================================
# B. Per-model runs
# ===========================================================================
st.subheader("B · Run the models & label each response")

if not scenario.get("shared_distractor_turn_1", "").strip():
    st.warning("Write the shared first distractor turn above before running the models.")


def _label_index(value: str) -> int:
    opts = [""] + RESPONSE_LABELS
    return opts.index(value) if value in opts else 0


def render_model_run(model_display: str) -> None:
    run = scenario["runs"][model_display]
    cfg = resolve_model(model_display)
    st.caption(
        f"Serving `{cfg['served_model_id']}` at `{cfg['base_url']}` → "
        f"export name `{cfg['canonical_name']}`"
    )
    if not cfg["enabled"]:
        st.warning(
            f"{model_display} is not loaded. Enable it on the **4 · Model Setup** "
            "page to run it. You can still annotate the other model.",
            icon="⚠️",
        )

    turns = run.setdefault("turns", [])

    for turn_idx in range(MAX_DISTRACTOR_TURNS):
        has_turn = turn_idx < len(turns)
        prev_complete = turn_idx == 0 or (
            turn_idx - 1 < len(turns) and bool(turns[turn_idx - 1].get("response"))
        )
        if not prev_complete and not has_turn:
            break  # don't show turn N+1 until turn N has a response

        st.markdown(f"**Distractor turn {turn_idx + 1}**"
                    + (" — shared first turn" if turn_idx == 0 else " _(may differ per model)_"))

        if turn_idx == 0:
            distractor_text = scenario.get("shared_distractor_turn_1", "")
            st.info(distractor_text or "_(set the shared first turn above)_")
        else:
            distractor_text = st.text_area(
                f"Distractor turn {turn_idx + 1} for {model_display}",
                value=turns[turn_idx]["distractor"] if has_turn else "",
                key=f"dist_{scenario['scenario_id']}_{model_display}_{turn_idx}",
                height=70,
            )

        run_clicked = st.button(
            f"▶️ Run turn {turn_idx + 1}",
            key=f"run_{scenario['scenario_id']}_{model_display}_{turn_idx}",
            disabled=not distractor_text.strip() or not cfg["enabled"],
        )
        if run_clicked:
            pairs = [(t["distractor"], t["response"]) for t in turns[:turn_idx]]
            messages = build_messages(
                scenario.get("system_prompt", ""),
                scenario.get("base_context", []),
                pairs,
                distractor_text,
            )
            with st.spinner(f"Running {model_display} (turn {turn_idx + 1})..."):
                result = run_chat(
                    messages, cfg,
                    temperature=run.get("temperature", 0.2),
                    top_p=run.get("top_p", 1.0),
                )
            if not result["ok"]:
                st.error(result["error"])
            else:
                entry = {
                    "distractor": distractor_text,
                    "response": result["final"],
                    "reasoning": result["reasoning"],
                    "label": turns[turn_idx]["label"] if has_turn else "",
                }
                # re-running a turn invalidates later turns (context changed)
                run["turns"] = turns[:turn_idx] + [entry]
                st.rerun()

        if has_turn and turns[turn_idx].get("response"):
            with st.chat_message("assistant"):
                st.write(turns[turn_idx]["response"])
            if turns[turn_idx].get("reasoning"):
                with st.expander("🧠 hidden reasoning (NOT annotated)"):
                    st.caption(turns[turn_idx]["reasoning"])
            turns[turn_idx]["label"] = st.selectbox(
                f"Label for response {turn_idx + 1}",
                options=[""] + RESPONSE_LABELS,
                index=_label_index(turns[turn_idx].get("label", "")),
                key=f"label_{scenario['scenario_id']}_{model_display}_{turn_idx}",
            )
        st.markdown("---")

    # Overall outcome for this model
    oc_opts = [""] + OVERALL_OUTCOMES
    run["overall_outcome"] = st.selectbox(
        f"Overall outcome — {model_display}",
        options=oc_opts,
        index=oc_opts.index(run.get("overall_outcome", "")) if run.get("overall_outcome", "") in oc_opts else 0,
        key=f"outcome_{scenario['scenario_id']}_{model_display}",
    )
    run["short_justification"] = st.text_area(
        f"Short justification — {model_display}",
        value=run.get("short_justification", ""),
        key=f"just_{scenario['scenario_id']}_{model_display}",
        height=80,
    )


tabs = st.tabs(TARGET_MODELS)
for tab, model_display in zip(tabs, TARGET_MODELS):
    with tab:
        render_model_run(model_display)

st.divider()

# ===========================================================================
# C. Inter-rater + save
# ===========================================================================
st.subheader("C · Inter-rater & notes")
ir1, ir2 = st.columns(2)
with ir1:
    scenario["second_annotator_name"] = st.text_input(
        "Second annotator (if any)", value=scenario.get("second_annotator_name", "")
    )
with ir2:
    scenario["disagreement_notes"] = st.text_area(
        "Disagreement notes (where/how resolved, guideline changes)",
        value=scenario.get("disagreement_notes", ""),
        height=80,
    )
scenario["notes"] = st.text_area("Notes", value=scenario.get("notes", ""), height=70)

st.session_state["active_scenario"] = scenario

save1, save2 = st.columns(2)
with save1:
    if st.button("💾 Save scenario", type="primary", use_container_width=True):
        if not scenario.get("annotator_name"):
            scenario["annotator_name"] = annotator
        if upsert_scenario(scenario):
            st.success(f"Saved `{scenario['scenario_id']}` to the shared repo.")
with save2:
    if st.button("🆕 Save & start another", use_container_width=True):
        if upsert_scenario(scenario):
            st.success(f"Saved `{scenario['scenario_id']}`.")
            del st.session_state["active_scenario"]
            st.page_link("pages/1_Browse_and_Select.py", label="➡️ Browse & Select", icon="🔎")
