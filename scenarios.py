"""
Scenario storage, source-dataset loading, and export.

A *scenario* is one (conversation, injection-point, distractor-goal) item.
Each scenario is run against BOTH target models, so it flattens to TWO export
rows (one per model) sharing the same scenario_id.

Persistence: scenarios.json in the private HF dataset repo
(ANNOTATIONS_REPO_ID), same mechanism the app already uses.
"""

from __future__ import annotations

import io
import csv
import json
import uuid
from datetime import datetime
from typing import Optional

import streamlit as st

from constants import (
    BASE_DATASET_CONFIG,
    BASE_DATASET_ID,
    BASE_DATASET_SPLITS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    EXPORT_COLUMNS,
    TARGET_MODELS,
)
from dataset_io import get_annotations_repo, get_hf_token
from models import resolve_model

SCENARIOS_FILE = "scenarios.json"


# ===========================================================================
# Source dataset (nvidia/CantTalkAboutThis) – read-only, cached
# ===========================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_source_conversations() -> list[dict]:
    """
    Load conversations from the base dataset with stable synthetic ids.

    Returns a list of:
      {
        conversation_id, domain, scenario, system_instruction,
        conversation: [{role, content}, ...], split, row_index,
        distractors: [{bot turn, distractor}, ...]   # original synthetic ones
      }
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        st.error("`datasets` not installed. Run: pip install datasets")
        return []

    token = get_hf_token()
    out: list[dict] = []
    for split in BASE_DATASET_SPLITS:
        try:
            ds = load_dataset(
                BASE_DATASET_ID,
                BASE_DATASET_CONFIG,
                split=split,
                token=token,
            )
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not load split '{split}': {e}")
            continue
        for i, item in enumerate(ds):
            out.append(
                {
                    "conversation_id": f"{split}-{i}",
                    "domain": item.get("domain", ""),
                    "scenario": item.get("scenario", ""),
                    "system_instruction": item.get("system_instruction", ""),
                    "conversation": list(item.get("conversation", []) or []),
                    "distractors": list(item.get("distractors", []) or []),
                    "split": split,
                    "row_index": i,
                }
            )
    return out


# ===========================================================================
# Scenario schema
# ===========================================================================

def new_model_run(model_display_name: str) -> dict:
    """Empty per-model run block (up to 3 distractor turns)."""
    return {
        "model_display_name": model_display_name,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        # turns: list of {distractor, response, reasoning, label}
        "turns": [],
        "overall_outcome": "",
        "short_justification": "",
    }


def new_scenario(annotator: str = "") -> dict:
    """Blank scenario matching the export schema + everything needed to run."""
    now = datetime.now().isoformat()
    return {
        "scenario_id": f"sc-{uuid.uuid4().hex[:8]}",
        "annotator_name": annotator,
        "second_annotator_name": "",
        # --- source context (filled from Browse page) ---
        "domain": "",
        "conversation_id": "",
        "selected_turn_id": None,
        "system_prompt": "",                 # full policy – passed to the model
        "system_prompt_or_policy_summary": "",  # short summary for the export
        "base_context": [],                  # real turns before the injection
        "original_user_message": "",         # the user turn we replace
        # --- distractor definition (shared across both models) ---
        "distractor_goal": "",
        "targeted_rule_or_scope_boundary": "",
        "why_this_is_a_good_distractor": "",
        "shared_distractor_turn_1": "",      # SAME first turn for both models
        # --- per-model runs ---
        "runs": {m: new_model_run(m) for m in TARGET_MODELS},
        # --- inter-rater + misc ---
        "disagreement_notes": "",
        "notes": "",
        "_status": "draft",
        "_created_at": now,
        "_updated_at": now,
    }


# ===========================================================================
# Persistence (scenarios.json in the private HF dataset repo)
# ===========================================================================

def load_scenarios() -> list[dict]:
    repo_id = get_annotations_repo()
    token = get_hf_token()
    if not repo_id or not token:
        return []
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        path = hf_hub_download(
            repo_id=repo_id,
            filename=SCENARIOS_FILE,
            repo_type="dataset",
            token=token,
            force_download=True,
        )
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _ensure_repo(repo_id: str, token: str) -> None:
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
        HfApi().repo_info(repo_id=repo_id, repo_type="dataset", token=token)
    except Exception:
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415
            HfApi().create_repo(
                repo_id=repo_id, repo_type="dataset", private=True, token=token,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not create repo '{repo_id}': {e}")


def save_scenarios(scenarios: list[dict]) -> bool:
    repo_id = get_annotations_repo()
    token = get_hf_token()
    if not repo_id:
        st.error("ANNOTATIONS_REPO_ID is not set.")
        return False
    if not token:
        st.error("HF_TOKEN is not set.")
        return False
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
        _ensure_repo(repo_id, token)
        content = json.dumps(scenarios, indent=2, ensure_ascii=False)
        HfApi().upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=SCENARIOS_FILE,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Update scenarios [{datetime.now():%Y-%m-%d %H:%M}]",
        )
        return True
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to save scenarios: {e}")
        return False


def upsert_scenario(scenario: dict) -> bool:
    scenario["_updated_at"] = datetime.now().isoformat()
    current = load_scenarios()
    for i, s in enumerate(current):
        if s.get("scenario_id") == scenario.get("scenario_id"):
            current[i] = scenario
            return save_scenarios(current)
    current.append(scenario)
    return save_scenarios(current)


def delete_scenario(scenario_id: str) -> bool:
    current = load_scenarios()
    updated = [s for s in current if s.get("scenario_id") != scenario_id]
    if len(updated) == len(current):
        st.warning("Scenario not found.")
        return False
    return save_scenarios(updated)


# ===========================================================================
# Export (2 rows per scenario -> the 27 required columns)
# ===========================================================================

def _turn(run: dict, idx: int, key: str) -> str:
    turns = run.get("turns", [])
    if idx < len(turns):
        return turns[idx].get(key, "") or ""
    return ""


def scenario_to_rows(scenario: dict) -> list[dict]:
    """Flatten one scenario into one row per target model."""
    rows = []
    for model_display in TARGET_MODELS:
        run = scenario.get("runs", {}).get(model_display, {})
        rows.append(
            {
                "annotator_name": scenario.get("annotator_name", ""),
                "second_annotator_name_if_any": scenario.get("second_annotator_name", ""),
                "scenario_id": scenario.get("scenario_id", ""),
                "domain": scenario.get("domain", ""),
                "conversation_id": scenario.get("conversation_id", ""),
                "selected_turn_id": scenario.get("selected_turn_id", ""),
                "system_prompt_or_policy_summary": scenario.get(
                    "system_prompt_or_policy_summary", ""
                ),
                "original_user_message": scenario.get("original_user_message", ""),
                "distractor_goal": scenario.get("distractor_goal", ""),
                "targeted_rule_or_scope_boundary": scenario.get(
                    "targeted_rule_or_scope_boundary", ""
                ),
                "why_this_is_a_good_distractor": scenario.get(
                    "why_this_is_a_good_distractor", ""
                ),
                "model_name": resolve_model(model_display)["canonical_name"],
                "temperature": run.get("temperature", DEFAULT_TEMPERATURE),
                "top_p": run.get("top_p", DEFAULT_TOP_P),
                "distractor_turn_1": _turn(run, 0, "distractor"),
                "model_response_1": _turn(run, 0, "response"),
                "response_label_1": _turn(run, 0, "label"),
                "distractor_turn_2": _turn(run, 1, "distractor"),
                "model_response_2": _turn(run, 1, "response"),
                "response_label_2": _turn(run, 1, "label"),
                "distractor_turn_3_if_any": _turn(run, 2, "distractor"),
                "model_response_3_if_any": _turn(run, 2, "response"),
                "response_label_3_if_any": _turn(run, 2, "label"),
                "overall_outcome": run.get("overall_outcome", ""),
                "short_justification": run.get("short_justification", ""),
                "disagreement_notes_if_any": scenario.get("disagreement_notes", ""),
                "notes": scenario.get("notes", ""),
            }
        )
    return rows


def scenarios_to_csv(scenarios: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for scenario in scenarios:
        for row in scenario_to_rows(scenario):
            writer.writerow(row)
    return buf.getvalue()


def upload_export_file(filename: str, content: bytes, mime_note: str = "") -> bool:
    """Upload an arbitrary export artifact (csv/json) to the HF dataset repo."""
    repo_id = get_annotations_repo()
    token = get_hf_token()
    if not repo_id or not token:
        st.error("Set HF_TOKEN and ANNOTATIONS_REPO_ID to upload exports.")
        return False
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
        _ensure_repo(repo_id, token)
        HfApi().upload_file(
            path_or_fileobj=content,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Add export {filename} {mime_note}".strip(),
        )
        return True
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to upload '{filename}': {e}")
        return False
