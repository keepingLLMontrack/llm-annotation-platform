from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

try:
    from openai import OpenAI
except Exception:  # optional
    OpenAI = None

APP_TITLE = "🧭 LLM Annotation Platform"
DEFAULT_SOURCE_REPO = os.environ.get("SOURCE_DATASET_REPO", "nvidia/CantTalkAboutThis-Topic-Control-Dataset")
DEFAULT_SOURCE_SPLITS = os.environ.get("SOURCE_DATASET_SPLITS", "train,test")
DEFAULT_ANNOTATION_REPO = os.environ.get("ANNOTATION_REPO_ID", "YOUR_USERNAME/llm-distractor-annotations")
DEFAULT_HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_HUB_TOKEN", ""))

CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/data/cache"))
DRAFT_DIR = Path(os.environ.get("DRAFT_DIR", "/data/drafts"))
EXPORT_DIR = Path(os.environ.get("EXPORT_DIR", "/data/exports"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DRAFT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

TACTICS = [
    "rephrasing the request",
    "claiming urgency",
    "just a general explanation",
    "flattery + redirect",
    "hypothetical loophole",
    "persisting after refusal",
    "false framing",
    "other",
]

TURN_ROLES = ["user", "assistant", "system", "tool"]

DEFAULT_OUTPUT_TEMPLATE = {
    "domain": "",
    "scenario": "",
    "system_instruction": "",
    "conversation": [],
    "distractors": [],
    "distractors_multiturn": [],
    "conversation_with_distractors": [],
    "split": "train",
    "_review_status": "draft",
    "_needs_human_review": True,
}

# ---------------------------------------------------------
# Small utilities
# ---------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(text: str, default: str = "item") -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or default


def safe_json_loads(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return fallback


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def row_to_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, pd.Series):
        return row.to_dict()
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


def series_get(record: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def ensure_list_of_dicts(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        value = safe_json_loads(value, [])
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, dict):
            out.append(item)
        else:
            out.append({"value": str(item)})
    return out


def ensure_turns(value: Any) -> List[Dict[str, str]]:
    turns = ensure_list_of_dicts(value)
    out = []
    for t in turns:
        out.append({
            "role": str(t.get("role", "user")),
            "content": str(t.get("content", t.get("text", ""))),
        })
    return out


def normalize_conversation(raw: Any) -> List[Dict[str, str]]:
    return ensure_turns(raw)


def normalize_distractors(raw: Any) -> List[Dict[str, str]]:
    items = ensure_list_of_dicts(raw)
    out = []
    for d in items:
        out.append({
            "bot_turn": str(d.get("bot_turn", d.get("bot turn", ""))),
            "distractor": str(d.get("distractor", d.get("user_turn", d.get("content", "")))),
        })
    return out


def normalize_multiturn(raw: Any) -> List[Dict[str, Any]]:
    items = ensure_list_of_dicts(raw)
    out = []
    for d in items:
        turns = d.get("turns", [])
        if isinstance(turns, str):
            turns = safe_json_loads(turns, [])
        out.append({
            "off_topic_subject": str(d.get("off_topic_subject", "")),
            "tactic_used": str(d.get("tactic_used", "")),
            "bot_turn": str(d.get("bot_turn", d.get("bot turn", ""))),
            "turns_json": pretty_json(ensure_turns(turns)) if turns else "[]",
        })
    return out


def build_conversation_with_distractors(conversation: List[Dict[str, str]], multiturn: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple automatic build:
    - keep the base conversation as the first item
    - add a variant conversation for each multiturn distractor by appending the user turns
      after the matching bot_turn when possible.
    """
    if not conversation:
        return []

    variants = [{"variant": "base", "conversation": conversation}]
    for idx, d in enumerate(multiturn):
        turns = safe_json_loads(d.get("turns_json", "[]"), [])
        if not isinstance(turns, list):
            turns = []
        bot_turn = str(d.get("bot_turn", "")).strip()

        conv = []
        inserted = False
        for turn in conversation:
            conv.append(turn)
            if not inserted and bot_turn and turn.get("role", "").lower() == "assistant" and turn.get("content", "").strip() == bot_turn:
                conv.extend(ensure_turns(turns))
                inserted = True
        if not inserted:
            # Fallback: append to end
            conv.extend(ensure_turns(turns))
        variants.append({
            "variant": f"distractor_{idx+1}",
            "conversation": conv,
        })
    return variants


def record_from_inputs(
    domain: str,
    scenario: str,
    system_instruction: str,
    conversation: List[Dict[str, str]],
    distractors: List[Dict[str, str]],
    multiturn: List[Dict[str, Any]],
    conversation_with_distractors: Any,
    split: str,
    review_status: str,
    needs_review: bool,
    source_split: str = "",
    source_index: Optional[int] = None,
    source_repo: str = "",
    annotator: str = "",
) -> Dict[str, Any]:
    record = {
        "domain": domain.strip(),
        "scenario": scenario.strip(),
        "system_instruction": system_instruction.strip(),
        "conversation": conversation,
        "distractors": distractors,
        "distractors_multiturn": multiturn,
        "conversation_with_distractors": conversation_with_distractors,
        "split": split,
        "_review_status": review_status,
        "_needs_human_review": needs_review,
        "_annotator": annotator,
        "_source_repo": source_repo,
        "_source_split": source_split,
        "_source_index": source_index,
        "_created_at": now_iso(),
        "_updated_at": now_iso(),
    }
    return record


def record_to_exportable(record: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(record)
    # keep the same top-level structure as the source file, but preserve provenance
    return out


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_hf_split(repo_id: str, split: str) -> List[Dict[str, Any]]:
    ds = load_dataset(repo_id, split=split)
    return [dict(r) for r in ds]


@st.cache_data(show_spinner=False)
def load_hf_all_splits(repo_id: str, splits_csv: str) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for split in [s.strip() for s in splits_csv.split(",") if s.strip()]:
        try:
            rows = load_hf_split(repo_id, split)
            for i, row in enumerate(rows):
                row = dict(row)
                row.setdefault("split", split)
                row.setdefault("_source_split", split)
                row.setdefault("_source_index", i)
                row.setdefault("_source_repo", repo_id)
                all_rows.append(row)
        except Exception:
            continue
    return all_rows


def load_local_json(path: Path, split_default: str = "train") -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        for i, row in enumerate(rows):
            row.setdefault("split", split_default)
            row.setdefault("_source_split", split_default)
            row.setdefault("_source_index", i)
        return rows
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Local JSON must contain a list of records.")
    for i, row in enumerate(data):
        row.setdefault("split", split_default)
        row.setdefault("_source_split", split_default)
        row.setdefault("_source_index", i)
    return data


def coerce_source_records(raw_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, r in enumerate(raw_records):
        rec = dict(r)
        rec.setdefault("split", rec.get("_source_split", "train"))
        rec.setdefault("_source_index", i)
        out.append(rec)
    return out


# ---------------------------------------------------------
# HF persistence
# ---------------------------------------------------------

def hf_client() -> HfApi:
    return HfApi(token=DEFAULT_HF_TOKEN or None)


def ensure_annotation_repo(repo_id: str) -> None:
    if not repo_id or repo_id.startswith("YOUR_"):
        return
    hf_client().create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)


def upload_record_to_hf(repo_id: str, record: Dict[str, Any], annotator: str) -> str:
    ensure_annotation_repo(repo_id)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = slugify(f"{annotator}-{record.get('domain','')}-{record.get('scenario','')}", "entry")
    filename = f"entries/{slugify(annotator, 'annotator')}/{stamp}_{safe_name}_{uuid.uuid4().hex[:8]}.json"

    tmp_dir = DRAFT_DIR / "_tmp_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_file = tmp_dir / f"{uuid.uuid4().hex}.json"
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(record_to_exportable(record), f, ensure_ascii=False, indent=2)

    hf_client().upload_file(
        path_or_fileobj=str(tmp_file),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Add annotation entry by {annotator}",
    )
    return filename


def list_uploaded_files(repo_id: str) -> List[str]:
    if not repo_id or repo_id.startswith("YOUR_"):
        return []
    try:
        return hf_client().list_repo_files(repo_id, repo_type="dataset")
    except Exception:
        return []


# ---------------------------------------------------------
# Local drafts / state
# ---------------------------------------------------------

def annotator_draft_path(annotator: str) -> Path:
    safe = slugify(annotator, "annotator")
    return DRAFT_DIR / f"{safe}.json"


def save_draft_local(annotator: str, payload: Dict[str, Any]) -> Path:
    path = annotator_draft_path(annotator)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_draft_local(annotator: str) -> Dict[str, Any]:
    path = annotator_draft_path(annotator)
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def append_submission_index(entry: Dict[str, Any]) -> None:
    idx = DRAFT_DIR / "submissions_index.jsonl"
    with idx.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------
# Editing helpers
# ---------------------------------------------------------

def df_from_turns(turns: List[Dict[str, str]]) -> pd.DataFrame:
    if not turns:
        return pd.DataFrame([{"role": "user", "content": ""}])
    return pd.DataFrame(turns)


def turns_from_df(df: pd.DataFrame) -> List[Dict[str, str]]:
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        role = str(row.get("role", "")).strip()
        content = str(row.get("content", "")).strip()
        if role or content:
            out.append({"role": role or "user", "content": content})
    return out


def df_from_simple_distractors(items: List[Dict[str, str]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame([{"bot_turn": "", "distractor": ""}])
    return pd.DataFrame(items)


def simple_distractors_from_df(df: pd.DataFrame) -> List[Dict[str, str]]:
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        bot_turn = str(row.get("bot_turn", "")).strip()
        distractor = str(row.get("distractor", "")).strip()
        if bot_turn or distractor:
            out.append({"bot_turn": bot_turn, "distractor": distractor})
    return out


def df_from_multiturn(items: List[Dict[str, Any]]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame([{"off_topic_subject": "", "tactic_used": TACTICS[0], "bot_turn": "", "turns_json": "[]"}])
    return pd.DataFrame(items)


def multiturn_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        subject = str(row.get("off_topic_subject", "")).strip()
        tactic = str(row.get("tactic_used", "")).strip()
        bot_turn = str(row.get("bot_turn", "")).strip()
        turns_json = str(row.get("turns_json", "[]")).strip()
        turns = safe_json_loads(turns_json, [])
        if isinstance(turns, list):
            turns = ensure_turns(turns)
        else:
            turns = []
        if subject or bot_turn or turns:
            out.append({
                "off_topic_subject": subject,
                "tactic_used": tactic,
                "bot_turn": bot_turn,
                "turns_json": pretty_json(turns) if turns else "[]",
            })
    return out


def normalize_draft_from_record(record: Dict[str, Any], source_repo: str = "", source_split: str = "", source_index: Optional[int] = None) -> Dict[str, Any]:
    conversation = normalize_conversation(record.get("conversation"))
    distractors = normalize_distractors(record.get("distractors"))
    multiturn = normalize_multiturn(record.get("distractors_multiturn"))
    convwd = record.get("conversation_with_distractors", [])
    if not isinstance(convwd, list):
        convwd = []
    if not convwd and multiturn:
        convwd = build_conversation_with_distractors(conversation, multiturn)
    return {
        "domain": str(series_get(record, "domain", default="")),
        "scenario": str(series_get(record, "scenario", default="")),
        "system_instruction": str(series_get(record, "system_instruction", default="")),
        "conversation": conversation,
        "distractors": distractors,
        "distractors_multiturn": multiturn,
        "conversation_with_distractors": convwd,
        "split": str(series_get(record, "split", "_source_split", default="train")),
        "_review_status": str(series_get(record, "_review_status", default="draft")),
        "_needs_human_review": bool(record.get("_needs_human_review", True)),
        "_source_repo": source_repo or str(series_get(record, "_source_repo", default="")),
        "_source_split": source_split or str(series_get(record, "_source_split", default="")),
        "_source_index": source_index if source_index is not None else record.get("_source_index"),
        "_annotator": str(series_get(record, "_annotator", default="")),
    }


def make_blank_draft() -> Dict[str, Any]:
    return dict(DEFAULT_OUTPUT_TEMPLATE)


def generate_llm_distractor_draft(draft: Dict[str, Any], base_url: str, model: str, mode: str = "simple") -> Optional[Dict[str, Any]]:
    if OpenAI is None:
        st.error("The openai package is not installed.")
        return None

    client = OpenAI(base_url=base_url, api_key="lm-studio")
    convo = draft.get("conversation", [])
    sysinst = draft.get("system_instruction", "")
    domain = draft.get("domain", "")
    scenario = draft.get("scenario", "")

    if mode == "simple":
        prompt = f"""
You are helping create a human-made distractor dataset for a task-oriented assistant.

Domain: {domain}
Scenario: {scenario}

System instruction:
{sysinst}

Conversation:
{json.dumps(convo, ensure_ascii=False, indent=2)}

Write ONE realistic off-topic distractor pair:
- bot_turn: exact assistant turn from the conversation to anchor after
- distractor: the user's off-topic message

Return only valid JSON with keys bot_turn and distractor.
"""
    else:
        prompt = f"""
You are helping create a human-made multi-turn distractor dataset for a task-oriented assistant.

Domain: {domain}
Scenario: {scenario}

System instruction:
{sysinst}

Conversation:
{json.dumps(convo, ensure_ascii=False, indent=2)}

Write ONE multi-turn distractor item:
- off_topic_subject
- tactic_used
- bot_turn
- turns: a JSON list of 3-5 turns that starts with a user off-topic request and escalates politely after refusals.

Return only valid JSON with keys off_topic_subject, tactic_used, bot_turn, turns.
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=1500,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            raw = raw.replace("json\n", "", 1)
        return json.loads(raw)
    except Exception as e:
        st.error(f"Local LLM generation failed: {e}")
        return None


# ---------------------------------------------------------
# UI components
# ---------------------------------------------------------

def render_preview_df(records: List[Dict[str, Any]], split_filter: str, search_text: str = "") -> pd.DataFrame:
    rows = []
    search_text = search_text.lower().strip()
    for i, r in enumerate(records):
        if split_filter and split_filter != "All" and str(r.get("split", r.get("_source_split", ""))) != split_filter:
            continue
        domain = str(series_get(r, "domain", default=""))
        scenario = str(series_get(r, "scenario", default=""))
        if search_text:
            joined = " ".join([domain, scenario, str(series_get(r, "system_instruction", default=""))]).lower()
            if search_text not in joined:
                continue
        convo = normalize_conversation(r.get("conversation"))
        preview = ""
        if convo:
            for t in reversed(convo):
                if str(t.get("role", "")).lower() == "user":
                    preview = str(t.get("content", ""))
                    break
            if not preview:
                preview = str(convo[-1].get("content", ""))
        rows.append({
            "#": i,
            "split": str(r.get("split", r.get("_source_split", ""))),
            "domain": domain,
            "scenario": scenario,
            "conversation_preview": (preview[:120] + "…") if len(preview) > 120 else preview,
            "distractor_count": len(r.get("distractors", [])) if isinstance(r.get("distractors"), list) else 0,
            "multi_count": len(r.get("distractors_multiturn", [])) if isinstance(r.get("distractors_multiturn"), list) else 0,
        })
    return pd.DataFrame(rows)


def current_source_record(records: List[Dict[str, Any]], idx: int) -> Optional[Dict[str, Any]]:
    if idx < 0 or idx >= len(records):
        return None
    return records[idx]


def clean_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].fillna("")
    return df


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
    st.title(APP_TITLE)
    st.caption("Simple collaborative editor for human-made distractor datasets.")

    # Session defaults
    for key, default in [
        ("annotator", "annotator_1"),
        ("source_mode", "HF dataset"),
        ("source_repo", DEFAULT_SOURCE_REPO),
        ("source_splits", DEFAULT_SOURCE_SPLITS),
        ("annotation_repo", DEFAULT_ANNOTATION_REPO),
        ("source_file_name", ""),
        ("source_row_idx", 0),
        ("draft", make_blank_draft()),
        ("draft_source_idx", None),
        ("draft_source_split", "train"),
        ("draft_mode", "new"),
        ("last_saved_message", ""),
        ("llm_base_url", "http://localhost:1234/v1"),
        ("llm_model", "gemma-4-e2b-it"),
        ("llm_mode", "simple"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # Sidebar
    st.sidebar.header("Workspace")
    st.session_state["annotator"] = st.sidebar.text_input("Annotator name", value=st.session_state["annotator"])
    st.session_state["source_mode"] = st.sidebar.radio("Source mode", ["HF dataset", "Upload local JSON/JSONL"], index=0 if st.session_state["source_mode"] == "HF dataset" else 1)
    st.session_state["source_repo"] = st.sidebar.text_input("Source dataset repo", value=st.session_state["source_repo"])
    st.session_state["source_splits"] = st.sidebar.text_input("Source splits (comma-separated)", value=st.session_state["source_splits"])
    st.session_state["annotation_repo"] = st.sidebar.text_input("Annotation dataset repo", value=st.session_state["annotation_repo"])
    st.sidebar.divider()
    st.session_state["llm_base_url"] = st.sidebar.text_input("Local LLM base URL", value=st.session_state["llm_base_url"])
    st.session_state["llm_model"] = st.sidebar.text_input("Local LLM model", value=st.session_state["llm_model"])
    st.session_state["llm_mode"] = st.sidebar.selectbox("LLM generation mode", ["simple", "multiturn"], index=0 if st.session_state["llm_mode"] == "simple" else 1)
    st.sidebar.caption("For LM Studio / OpenAI-compatible local servers, keep the base URL like http://localhost:1234/v1.")
    st.sidebar.divider()

    uploaded_file = None
    if st.session_state["source_mode"] == "Upload local JSON/JSONL":
        uploaded_file = st.sidebar.file_uploader("Upload source file", type=["json", "jsonl"])
        if uploaded_file is not None:
            st.session_state["source_file_name"] = uploaded_file.name

    page = st.sidebar.radio("Page", ["Browse", "Edit / Create", "Drafts", "Export / Sync"], index=0)
    st.sidebar.caption(f"HF token present: {'yes' if DEFAULT_HF_TOKEN else 'no'}")
    st.sidebar.caption(f"Draft folder: {DRAFT_DIR}")

    # Load source records
    if "source_records" not in st.session_state:
        st.session_state["source_records"] = None

    if st.session_state["source_records"] is None:
        with st.spinner("Loading source data..."):
            try:
                if st.session_state["source_mode"] == "HF dataset":
                    records = load_hf_all_splits(st.session_state["source_repo"], st.session_state["source_splits"])
                else:
                    if uploaded_file is not None:
                        suffix = Path(uploaded_file.name).suffix.lower()
                        tmp_path = DRAFT_DIR / f"uploaded_source{suffix}"
                        tmp_path.write_bytes(uploaded_file.getbuffer())
                        records = load_local_json(tmp_path)
                    else:
                        records = []
                st.session_state["source_records"] = coerce_source_records(records)
            except Exception as e:
                st.session_state["source_records"] = []
                st.error(f"Could not load source data: {e}")

    records: List[Dict[str, Any]] = st.session_state["source_records"] or []

    if page == "Browse":
        st.subheader("Browse source dataset")
        split_choices = ["All"] + sorted({str(r.get("split", r.get("_source_split", ""))) for r in records if str(r.get("split", r.get("_source_split", "")))} )
        col1, col2 = st.columns([1, 1])
        with col1:
            split_filter = st.selectbox("Filter split", split_choices, index=0)
        with col2:
            search_text = st.text_input("Search text (domain / scenario / instruction)", value="")
        preview_df = render_preview_df(records, split_filter, search_text)
        st.write(f"Rows loaded: {len(preview_df)}")
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        if preview_df.empty:
            st.info("No rows match the current filter.")
        else:
            picked = st.number_input("Pick row number (#)", min_value=0, max_value=max(0, len(preview_df) - 1), value=0, step=1)
            if st.button("Load selected row into editor"):
                selected_global_idx = int(preview_df.iloc[int(picked)]["#"])
                st.session_state["draft_mode"] = "clone"
                st.session_state["draft_source_idx"] = selected_global_idx
                st.session_state["draft_source_split"] = str(records[selected_global_idx].get("split", records[selected_global_idx].get("_source_split", "train")))
                st.session_state["draft"] = normalize_draft_from_record(
                    records[selected_global_idx],
                    source_repo=st.session_state["source_repo"],
                    source_split=st.session_state["draft_source_split"],
                    source_index=selected_global_idx,
                )
                st.success(f"Loaded row {selected_global_idx} into the editor.")
                st.rerun()

        st.markdown("### Record inspector")
        if preview_df.empty:
            st.stop()
        idx = int(preview_df.iloc[int(picked)]["#"])
        rec = records[idx]
        st.json({
            "domain": rec.get("domain", ""),
            "scenario": rec.get("scenario", ""),
            "split": rec.get("split", rec.get("_source_split", "")),
            "keys": list(rec.keys()),
        })
        st.markdown("**Conversation preview**")
        st.code(pretty_json(rec.get("conversation", [])), language="json")
        st.markdown("**Distractors preview**")
        st.code(pretty_json(rec.get("distractors", [])), language="json")

    elif page == "Edit / Create":
        st.subheader("Create or edit an entry")
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1:
                if st.button("New blank entry"):
                    st.session_state["draft_mode"] = "new"
                    st.session_state["draft_source_idx"] = None
                    st.session_state["draft"] = make_blank_draft()
                    st.success("Blank entry created.")
                    st.rerun()
            with c2:
                if st.button("Reset draft from source row"):
                    idx = st.session_state.get("draft_source_idx")
                    if idx is not None and 0 <= idx < len(records):
                        st.session_state["draft"] = normalize_draft_from_record(
                            records[idx],
                            source_repo=st.session_state["source_repo"],
                            source_split=str(records[idx].get("split", records[idx].get("_source_split", "train"))),
                            source_index=idx,
                        )
                        st.success("Draft reset from source row.")
                    else:
                        st.warning("No source row selected.")
            with c3:
                if st.button("Auto-build conversation_with_distractors"):
                    d = st.session_state["draft"]
                    d["conversation_with_distractors"] = build_conversation_with_distractors(d.get("conversation", []), d.get("distractors_multiturn", []))
                    st.session_state["draft"] = d
                    st.success("Built conversation_with_distractors.")
                    st.rerun()

            st.markdown("### Source row")
            row_idx = st.number_input(
                "Source row index",
                min_value=0,
                max_value=max(0, len(records) - 1),
                value=int(st.session_state.get("draft_source_idx") or 0),
                step=1,
            )
            source_split_guess = ""
            if records:
                source_split_guess = str(records[int(row_idx)].get("split", records[int(row_idx)].get("_source_split", "train")))
            st.write("Detected source split:", source_split_guess or "n/a")
            if st.button("Load this row"):
                idx = int(row_idx)
                if 0 <= idx < len(records):
                    st.session_state["draft_mode"] = "clone"
                    st.session_state["draft_source_idx"] = idx
                    st.session_state["draft_source_split"] = str(records[idx].get("split", records[idx].get("_source_split", "train")))
                    st.session_state["draft"] = normalize_draft_from_record(
                        records[idx],
                        source_repo=st.session_state["source_repo"],
                        source_split=st.session_state["draft_source_split"],
                        source_index=idx,
                    )
                    st.success(f"Loaded source row {idx}.")
                    st.rerun()

            draft = st.session_state["draft"]

            top1, top2, top3 = st.columns(3)
            with top1:
                draft["split"] = st.selectbox("Entry split", ["train", "test"], index=0 if str(draft.get("split", "train")) == "train" else 1)
            with top2:
                draft["_review_status"] = st.selectbox("Review status", ["draft", "approved", "failed"], index=["draft", "approved", "failed"].index(str(draft.get("_review_status", "draft"))))
            with top3:
                draft["_needs_human_review"] = st.checkbox("Needs human review", value=bool(draft.get("_needs_human_review", True)))

            draft["domain"] = st.text_input("Domain", value=str(draft.get("domain", "")))
            draft["scenario"] = st.text_input("Scenario", value=str(draft.get("scenario", "")))
            draft["system_instruction"] = st.text_area("System instruction", value=str(draft.get("system_instruction", "")), height=180)

            st.markdown("#### Conversation")
            conv_df = clean_editor_df(pd.DataFrame(draft.get("conversation", [{"role": "user", "content": ""}])))
            conv_df = st.data_editor(
                conv_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "role": st.column_config.SelectboxColumn("role", options=TURN_ROLES, required=True),
                    "content": st.column_config.TextColumn("content", required=True),
                },
                hide_index=True,
                key="conversation_editor",
            )
            draft["conversation"] = turns_from_df(conv_df)
            if st.button("Clear conversation"):
                draft["conversation"] = []
                st.session_state["draft"] = draft
                st.rerun()

            st.markdown("#### Simple distractors")
            simple_df = clean_editor_df(pd.DataFrame(draft.get("distractors", [{"bot_turn": "", "distractor": ""}])))
            simple_df = st.data_editor(
                simple_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "bot_turn": st.column_config.TextColumn("bot_turn"),
                    "distractor": st.column_config.TextColumn("distractor"),
                },
                hide_index=True,
                key="simple_distractors_editor",
            )
            draft["distractors"] = simple_distractors_from_df(simple_df)
            if st.button("Clear simple distractors"):
                draft["distractors"] = []
                st.session_state["draft"] = draft
                st.rerun()

            st.markdown("#### Multi-turn distractors")
            multi_df = clean_editor_df(pd.DataFrame(draft.get("distractors_multiturn", [{"off_topic_subject": "", "tactic_used": TACTICS[0], "bot_turn": "", "turns_json": "[]"}])))
            multi_df = st.data_editor(
                multi_df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "off_topic_subject": st.column_config.TextColumn("off_topic_subject"),
                    "tactic_used": st.column_config.SelectboxColumn("tactic_used", options=TACTICS, required=False),
                    "bot_turn": st.column_config.TextColumn("bot_turn"),
                    "turns_json": st.column_config.TextColumn("turns_json", help="JSON list of turns, e.g. [{\"role\":\"user\",\"content\":\"...\"}]"),
                },
                hide_index=True,
                key="multi_distractors_editor",
            )
            draft["distractors_multiturn"] = multiturn_from_df(multi_df)
            if st.button("Clear multi-turn distractors"):
                draft["distractors_multiturn"] = []
                st.session_state["draft"] = draft
                st.rerun()

            st.markdown("#### Conversation with distractors")
            if st.button("Auto-generate conversation_with_distractors from current draft"):
                draft["conversation_with_distractors"] = build_conversation_with_distractors(draft.get("conversation", []), draft.get("distractors_multiturn", []))
                st.session_state["draft"] = draft
            convwd_text = st.text_area(
                "conversation_with_distractors (JSON)",
                value=pretty_json(draft.get("conversation_with_distractors", [])),
                height=200,
            )
            if st.button("Apply conversation_with_distractors JSON"):
                draft["conversation_with_distractors"] = safe_json_loads(convwd_text, [])
                st.session_state["draft"] = draft

            st.markdown("#### Quick LLM assist")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Generate draft with local LLM"):
                    out = generate_llm_distractor_draft(
                        draft,
                        base_url=st.session_state["llm_base_url"],
                        model=st.session_state["llm_model"],
                        mode=st.session_state["llm_mode"],
                    )
                    if out:
                        if st.session_state["llm_mode"] == "simple":
                            draft.setdefault("distractors", [])
                            draft["distractors"].append({
                                "bot_turn": out.get("bot_turn", ""),
                                "distractor": out.get("distractor", ""),
                            })
                        else:
                            draft.setdefault("distractors_multiturn", [])
                            draft["distractors_multiturn"].append({
                                "off_topic_subject": out.get("off_topic_subject", ""),
                                "tactic_used": out.get("tactic_used", ""),
                                "bot_turn": out.get("bot_turn", ""),
                                "turns_json": pretty_json(ensure_turns(out.get("turns", []))),
                            })
                        st.session_state["draft"] = draft
                        st.success("LLM draft inserted into the editor.")
                        st.rerun()
            with c2:
                st.caption("This calls a local OpenAI-compatible server such as LM Studio.")

            st.markdown("#### Save / submit")
            if st.button("Save draft locally"):
                draft["_annotator"] = st.session_state["annotator"]
                draft["_updated_at"] = now_iso()
                path = save_draft_local(st.session_state["annotator"], draft)
                st.success(f"Draft saved: {path}")
            if st.button("Submit current entry to HF repo"):
                final_record = record_from_inputs(
                    domain=draft.get("domain", ""),
                    scenario=draft.get("scenario", ""),
                    system_instruction=draft.get("system_instruction", ""),
                    conversation=draft.get("conversation", []),
                    distractors=draft.get("distractors", []),
                    multiturn=draft.get("distractors_multiturn", []),
                    conversation_with_distractors=draft.get("conversation_with_distractors", []),
                    split=str(draft.get("split", "train")),
                    review_status=str(draft.get("_review_status", "draft")),
                    needs_review=bool(draft.get("_needs_human_review", True)),
                    source_split=st.session_state.get("draft_source_split", ""),
                    source_index=st.session_state.get("draft_source_idx"),
                    source_repo=st.session_state["source_repo"],
                    annotator=st.session_state["annotator"],
                )
                try:
                    filename = upload_record_to_hf(st.session_state["annotation_repo"], final_record, st.session_state["annotator"])
                    append_submission_index({
                        "annotator": st.session_state["annotator"],
                        "uploaded_file": filename,
                        "split": final_record.get("split", ""),
                        "domain": final_record.get("domain", ""),
                        "scenario": final_record.get("scenario", ""),
                        "created_at": now_iso(),
                    })
                    save_draft_local(st.session_state["annotator"], draft)
                    st.success(f"Submitted to HF as {filename}")
                except Exception as e:
                    st.error(f"HF upload failed: {e}")
                    st.warning("The draft remains saved locally in the bucket.")

        with right:
            st.markdown("### Current draft preview")
            st.json(st.session_state["draft"])
            st.markdown("### Quick notes")
            st.write("The output keeps the same top-level structure as the source file and adds provenance fields such as split, annotator, and source index.")
            st.write("You can edit each cell directly in the tables, add rows dynamically, and clear whole sections with the buttons on the left.")

    elif page == "Drafts":
        st.subheader("Drafts and submissions")
        draft = load_draft_local(st.session_state["annotator"])
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Saved local draft")
            if draft:
                st.json(draft)
            else:
                st.info("No draft saved for this annotator yet.")
        with c2:
            st.markdown("### Submission index")
            idx_file = DRAFT_DIR / "submissions_index.jsonl"
            if idx_file.exists():
                lines = idx_file.read_text(encoding="utf-8").splitlines()
                rows = [json.loads(x) for x in lines if x.strip()]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No submissions recorded yet.")

    else:
        st.subheader("Export / Sync")
        st.write("Export current source + drafts as a merged JSONL or CSV, or inspect HF uploads.")
        current_draft = st.session_state.get("draft", make_blank_draft())

        # Build a merged dataset view from source records plus local draft if populated
        merged = [dict(r) for r in records]
        if current_draft and current_draft.get("domain") and current_draft.get("scenario"):
            merged.append(record_to_exportable(current_draft))

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Write merged JSONL export"):
                path = EXPORT_DIR / "merged_dataset.jsonl"
                with path.open("w", encoding="utf-8") as f:
                    for r in merged:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                st.success(f"Wrote {path}")
                st.download_button("Download merged JSONL", data=path.read_text(encoding="utf-8"), file_name=path.name, mime="application/json")
        with c2:
            if st.button("Write merged CSV export"):
                path = EXPORT_DIR / "merged_dataset.csv"
                pd.json_normalize(merged).to_csv(path, index=False)
                st.success(f"Wrote {path}")
                st.download_button("Download merged CSV", data=path.read_text(encoding="utf-8"), file_name=path.name, mime="text/csv")
        with c3:
            if st.button("Refresh HF file list"):
                st.rerun()

        st.markdown("### Uploaded files in annotation repo")
        files = list_uploaded_files(st.session_state["annotation_repo"])
        if files:
            st.dataframe(pd.DataFrame({"file": files}), use_container_width=True, hide_index=True)
        else:
            st.info("No repository files listed yet, or repo is not configured.")

        st.markdown("### Repository settings to remember")
        st.code(
            f"SOURCE_DATASET_REPO={st.session_state['source_repo']}\n"
            f"SOURCE_DATASET_SPLITS={st.session_state['source_splits']}\n"
            f"ANNOTATION_REPO_ID={st.session_state['annotation_repo']}\n"
            f"HF_TOKEN={'set' if DEFAULT_HF_TOKEN else 'missing'}",
            language="text",
        )


if __name__ == "__main__":
    main()
