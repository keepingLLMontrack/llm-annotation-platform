from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download

APP_TITLE = "🧭 LLM Annotation Platform"
DEFAULT_SOURCE_DATASET = os.environ.get(
    "SOURCE_DATASET_REPO",
    "nvidia/CantTalkAboutThis-Topic-Control-Dataset",
)
DEFAULT_SOURCE_SPLIT = os.environ.get("SOURCE_DATASET_SPLIT", "train")
DEFAULT_ANNOTATION_REPO = os.environ.get(
    "ANNOTATION_REPO_ID",
    "YOUR_ORG/llm-distractor-annotations",
)
DEFAULT_CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/data/hf_annotation_cache"))
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_DRAFT_DIR = Path(os.environ.get("DRAFT_DIR", "/data/hf_annotation_drafts"))
LOCAL_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_EXPORT_DIR = Path(os.environ.get("EXPORT_DIR", "/data/hf_annotation_exports"))
LOCAL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_OPTIONS = {
    "distractor_kind": [
        "benign off-topic",
        "smooth bridge",
        "policy-evasive",
        "roleplay / impersonation",
        "pressure / persistence",
        "urgency / time pressure",
        "loophole seeking",
        "multi-turn escalation",
        "other",
    ],
    "assistant_behavior": [
        "perfect refusal + redirect",
        "helpful redirection",
        "partial engagement",
        "full engagement / derailment",
        "over-refusal",
        "unclear",
    ],
    "transition_style": [
        "abrupt",
        "smooth bridge",
        "follow-up clarification",
        "rephrasing",
        "escalation",
        "roleplay",
        "ambiguity exploitation",
        "other",
    ],
    "policy_target": [
        "medical advice",
        "financial advice",
        "legal advice",
        "competitor discussion",
        "politics",
        "unsafe content",
        "personal data",
        "company-specific policy",
        "tone / style policy",
        "other",
    ],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def token() -> Optional[str]:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def api() -> HfApi:
    return HfApi(token=token())


def annotation_file_name(item_id: str, annotator: str) -> str:
    safe_annotator = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in annotator.strip().lower()) or "annotator"
    safe_item = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in item_id.strip()) or "item"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"annotations/{safe_annotator}/{stamp}_{safe_item}_{uuid.uuid4().hex[:8]}.json"


def draft_path(annotator: str) -> Path:
    safe_annotator = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in annotator.strip().lower()) or "annotator"
    return LOCAL_DRAFT_DIR / f"{safe_annotator}.json"


def cache_annotations_dir() -> Path:
    path = DEFAULT_CACHE_DIR / "annotations_snapshot"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_repo_exists(repo_id: str) -> None:
    if repo_id.startswith("YOUR_ORG/") or not repo_id.strip():
        return
    api().create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)


def load_source_dataset(repo_id: str, split: str) -> List[Dict[str, Any]]:
    ds = load_dataset(repo_id, split=split)
    return [dict(row) for row in ds]


def normalize_turns(turns: Any) -> List[Dict[str, Any]]:
    if turns is None:
        return []
    if isinstance(turns, str):
        try:
            turns = json.loads(turns)
        except Exception:
            return []
    if not isinstance(turns, list):
        return []
    out = []
    for turn in turns:
        if isinstance(turn, dict):
            role = turn.get("role") or turn.get("speaker") or turn.get("type") or "unknown"
            content = turn.get("content") or turn.get("text") or turn.get("utterance") or ""
            out.append({"role": str(role), "content": str(content)})
        else:
            out.append({"role": "unknown", "content": str(turn)})
    return out


def safe_sample_id(record: Dict[str, Any], fallback_index: int) -> str:
    for key in ("sample_id", "id", "_id", "row_id"):
        if record.get(key) not in (None, ""):
            return str(record[key])
    domain = str(record.get("domain", "sample")).replace(" ", "_")
    scenario = str(record.get("scenario", "")).replace(" ", "_")
    return f"{domain}-{scenario}-{fallback_index}"


def expand_record(record: Dict[str, Any], idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sample_id = safe_sample_id(record, idx)
    conversation = normalize_turns(record.get("conversation"))
    distractors = record.get("distractors") or []
    if isinstance(distractors, str):
        try:
            distractors = json.loads(distractors)
        except Exception:
            distractors = []
    if not isinstance(distractors, list):
        distractors = []

    sample = {
        "sample_id": sample_id,
        "domain": str(record.get("domain", "")),
        "scenario": str(record.get("scenario", "")),
        "system_instruction": str(record.get("system_instruction", "")),
        "conversation_json": json.dumps(conversation, ensure_ascii=False),
        "distractors_json": json.dumps(distractors, ensure_ascii=False),
        "conversation_with_distractors_json": json.dumps(record.get("conversation_with_distractors", []), ensure_ascii=False),
        "raw_json": json.dumps(record, ensure_ascii=False),
    }

    items = []
    for distractor_index, d in enumerate(distractors):
        bot_turn = ""
        distractor_text = ""
        if isinstance(d, dict):
            bot_turn = str(
                d.get("bot turn")
                or d.get("bot_turn")
                or d.get("assistant_turn")
                or d.get("assistant")
                or ""
            )
            distractor_text = str(
                d.get("distractor")
                or d.get("distractor user turn")
                or d.get("user_turn")
                or d.get("user")
                or d.get("text")
                or ""
            )
        else:
            distractor_text = str(d)

        items.append(
            {
                "item_id": f"{sample_id}::{distractor_index}",
                "sample_id": sample_id,
                "distractor_index": distractor_index,
                "bot_turn": bot_turn,
                "distractor_text": distractor_text,
            }
        )
    return sample, items


def seed_source_index(records: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    samples = []
    items = []
    for idx, record in enumerate(records):
        sample, record_items = expand_record(record, idx)
        samples.append(sample)
        items.extend(record_items)
    return pd.DataFrame(samples), pd.DataFrame(items)


def read_json_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_all_hub_annotations(annotation_repo_id: str) -> pd.DataFrame:
    """
    Each submission is stored as a separate JSON file, which avoids write conflicts.
    """
    if annotation_repo_id.startswith("YOUR_ORG/") or not annotation_repo_id.strip():
        return pd.DataFrame(columns=["item_id", "annotator", "labels", "notes", "status", "created_at", "file_path"])

    cache_dir = cache_annotations_dir()
    file_list = api().list_repo_files(annotation_repo_id, repo_type="dataset")
    ann_files = [f for f in file_list if f.startswith("annotations/") and f.endswith(".json")]

    rows = []
    for file_path in ann_files:
        try:
            local_path = hf_hub_download(
                repo_id=annotation_repo_id,
                repo_type="dataset",
                filename=file_path,
                token=token(),
                local_dir=str(cache_dir),
                local_dir_use_symlinks=False,
            )
            payload = read_json_file(Path(local_path))
            rows.append(
                {
                    "item_id": payload.get("item_id", ""),
                    "sample_id": payload.get("sample_id", ""),
                    "annotator": payload.get("annotator", ""),
                    "labels": payload.get("labels", {}),
                    "notes": payload.get("notes", ""),
                    "status": payload.get("status", "submitted"),
                    "created_at": payload.get("created_at", ""),
                    "file_path": file_path,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "item_id": "",
                    "sample_id": "",
                    "annotator": "",
                    "labels": {},
                    "notes": f"Failed to load {file_path}: {e}",
                    "status": "load_error",
                    "created_at": "",
                    "file_path": file_path,
                }
            )

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["item_id", "sample_id", "annotator", "labels", "notes", "status", "created_at", "file_path"])


def save_draft(annotator: str, payload: Dict[str, Any]) -> Path:
    path = draft_path(annotator)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_draft(annotator: str) -> Dict[str, Any]:
    path = draft_path(annotator)
    if not path.exists():
        return {}
    try:
        return read_json_file(path)
    except Exception:
        return {}


def build_labels_from_state(prefix: str = "") -> Dict[str, Any]:
    return {
        "distractor_kind": st.session_state.get(f"{prefix}distractor_kind", LABEL_OPTIONS["distractor_kind"][0]),
        "transition_style": st.session_state.get(f"{prefix}transition_style", LABEL_OPTIONS["transition_style"][0]),
        "policy_target": st.session_state.get(f"{prefix}policy_target", []),
        "difficulty": int(st.session_state.get(f"{prefix}difficulty", 3)),
        "realism": int(st.session_state.get(f"{prefix}realism", 3)),
        "assistant_behavior": st.session_state.get(f"{prefix}assistant_behavior", LABEL_OPTIONS["assistant_behavior"][0]),
        "multi_turn_escalation": bool(st.session_state.get(f"{prefix}multi_turn_escalation", False)),
        "rule_followed": bool(st.session_state.get(f"{prefix}rule_followed", True)),
        "needs_review": bool(st.session_state.get(f"{prefix}needs_review", False)),
        "confidence": int(st.session_state.get(f"{prefix}confidence", 3)),
    }


def preview_text(text: str, limit: int = 280) -> str:
    txt = (text or "").strip().replace("\n", " ")
    if len(txt) <= limit:
        return txt
    return txt[:limit - 1] + "…"


def render_turns(turns: List[Dict[str, Any]]) -> None:
    if not turns:
        st.info("No conversation turns found.")
        return
    for i, turn in enumerate(turns, 1):
        role = str(turn.get("role", "unknown")).lower()
        content = str(turn.get("content", "")).strip()
        css_cls = "user" if role == "user" else "assistant" if role in {"assistant", "bot"} else "system"
        st.markdown(
            f"""
            <div class="turn {css_cls}">
                <span class="badge">{role.upper()}</span>
                <span class="smallmono">Turn {i}</span>
                <div style="margin-top:0.35rem; white-space:pre-wrap;">{content.replace(chr(10), '<br>')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def annotation_exists_for_item(df_anns: pd.DataFrame, item_id: str, annotator: str) -> bool:
    if df_anns.empty:
        return False
    sub = df_anns[(df_anns["item_id"] == item_id) & (df_anns["annotator"] == annotator)]
    return not sub.empty


def compute_agreement(df_anns: pd.DataFrame, label_key: str = "assistant_behavior") -> Dict[str, Any]:
    if df_anns.empty:
        return {"paired_items": 0, "raw_agreement": None, "cohen_kappa": None}

    rows = []
    for _, r in df_anns.iterrows():
        labels = r.get("labels", {}) or {}
        rows.append({"item_id": r["item_id"], "annotator": r["annotator"], label_key: labels.get(label_key)})
    tmp = pd.DataFrame(rows)
    pivot = tmp.pivot_table(index="item_id", columns="annotator", values=label_key, aggfunc="first")
    pivot = pivot.dropna(axis=0, how="any")
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return {"paired_items": int(pivot.shape[0]), "raw_agreement": None, "cohen_kappa": None}

    from sklearn.metrics import cohen_kappa_score

    a = pivot.iloc[:, 0].astype(str)
    b = pivot.iloc[:, 1].astype(str)
    return {
        "paired_items": int(pivot.shape[0]),
        "raw_agreement": float((a == b).mean()),
        "cohen_kappa": float(cohen_kappa_score(a, b)),
    }


def push_annotation_to_hub(annotation_repo_id: str, payload: Dict[str, Any]) -> str:
    ensure_repo_exists(annotation_repo_id)
    file_rel_path = annotation_file_name(payload["item_id"], payload["annotator"])
    local_path = LOCAL_DRAFT_DIR / file_rel_path.replace("/", "__")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    api().upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=file_rel_path,
        repo_id=annotation_repo_id,
        repo_type="dataset",
        token=token(),
        commit_message=f"Add annotation for {payload['item_id']} by {payload['annotator']}",
    )
    return file_rel_path


def get_current_item_id() -> Optional[str]:
    return st.session_state.get("current_item_id")


def set_current_item_id(item_id: Optional[str]) -> None:
    st.session_state["current_item_id"] = item_id
    try:
        st.query_params["item_id"] = item_id or ""
    except Exception:
        pass


def main() -> None:
    st.set_page_config(page_title="LLM Annotation Platform", page_icon="🧭", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 2rem;}
        .smallmono {font-size: 0.84rem; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;}
        .cardbox {
            border: 1px solid rgba(120,120,120,0.22);
            border-radius: 18px;
            padding: 1rem 1rem 0.75rem 1rem;
            background: rgba(255,255,255,0.03);
        }
        .turn {
            border-left: 4px solid rgba(120,120,120,0.45);
            padding: 0.6rem 0.85rem;
            margin: 0.55rem 0;
            border-radius: 0.6rem;
            background: rgba(128,128,128,0.06);
        }
        .turn.user {border-left-color: #8b5cf6;}
        .turn.assistant, .turn.bot {border-left-color: #06b6d4;}
        .turn.system {border-left-color: #f59e0b;}
        .badge {
            display:inline-block; padding:0.18rem 0.5rem; border-radius: 999px;
            background: rgba(120,120,120,0.16); margin-right: 0.35rem; font-size: 0.78rem;
        }
        hr {margin: 0.7rem 0 0.9rem 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(APP_TITLE)
    st.caption("A Hugging Face–native annotation tool for multi-turn distractors, inter-rater review, and dataset versioning.")

    if "annotator" not in st.session_state:
        st.session_state["annotator"] = "annotator_1"
    if "current_item_id" not in st.session_state:
        st.session_state["current_item_id"] = None
    if "source_records" not in st.session_state:
        st.session_state["source_records"] = None
    if "source_index" not in st.session_state:
        st.session_state["source_index"] = None
    if "annotations_df" not in st.session_state:
        st.session_state["annotations_df"] = None
    if "draft_loaded" not in st.session_state:
        st.session_state["draft_loaded"] = False

    with st.sidebar:
        st.header("Workspace")
        annotator = st.text_input("Annotator name", value=st.session_state["annotator"])
        st.session_state["annotator"] = annotator.strip() or "annotator_1"

        source_repo = st.text_input("Source dataset repo", value=DEFAULT_SOURCE_DATASET)
        source_split = st.text_input("Source split", value=DEFAULT_SOURCE_SPLIT)
        annotation_repo = st.text_input("Annotation dataset repo", value=DEFAULT_ANNOTATION_REPO)

        st.divider()
        st.caption("HF token is needed only for upload / repo creation.")
        st.write("HF token present:", "yes" if token() else "no")
        st.write("Cache:", str(DEFAULT_CACHE_DIR))
        st.write("Drafts:", str(LOCAL_DRAFT_DIR))

        if st.button("Reload Hub data", use_container_width=True):
            st.session_state["source_records"] = None
            st.session_state["source_index"] = None
            st.session_state["annotations_df"] = None
            st.rerun()

        page = st.radio("Page", ["Annotate", "Review", "Dashboard", "Export"], index=0)

    if st.session_state["source_records"] is None:
        with st.spinner("Loading source dataset from the Hub..."):
            source_records = load_source_dataset(source_repo, source_split)
            samples_df, items_df = seed_source_index(source_records)
            st.session_state["source_records"] = source_records
            st.session_state["source_index"] = {"samples_df": samples_df, "items_df": items_df}

    if st.session_state["annotations_df"] is None:
        with st.spinner("Loading annotations from the annotation dataset repo..."):
            try:
                anns_df = load_all_hub_annotations(annotation_repo)
            except Exception as e:
                anns_df = pd.DataFrame(columns=["item_id", "sample_id", "annotator", "labels", "notes", "status", "created_at", "file_path"])
                st.warning(f"Could not load annotations from Hub yet: {e}")
            st.session_state["annotations_df"] = anns_df

    samples_df = st.session_state["source_index"]["samples_df"]
    items_df = st.session_state["source_index"]["items_df"]
    anns_df = st.session_state["annotations_df"]

    if not st.session_state["draft_loaded"]:
        try:
            q_item = st.query_params.get("item_id")
        except Exception:
            q_item = None
        if q_item:
            st.session_state["current_item_id"] = q_item
        draft = load_draft(st.session_state["annotator"])
        if draft.get("current_item_id") and not st.session_state["current_item_id"]:
            st.session_state["current_item_id"] = draft["current_item_id"]
        st.session_state["draft_loaded"] = True

    my_annotated_item_ids = set(
        anns_df.loc[anns_df["annotator"] == st.session_state["annotator"], "item_id"].dropna().astype(str).tolist()
    ) if not anns_df.empty else set()

    def current_item_row() -> Optional[Dict[str, Any]]:
        item_id = get_current_item_id()
        if not item_id:
            return None
        match = items_df[items_df["item_id"] == item_id]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        sample = samples_df[samples_df["sample_id"] == row["sample_id"]]
        if not sample.empty:
            row.update(sample.iloc[0].to_dict())
        return row

    def queue_df() -> pd.DataFrame:
        return items_df[~items_df["item_id"].astype(str).isin(my_annotated_item_ids)].copy()

    if page == "Annotate":
        st.subheader("Annotate a distractor item")
        left, right = st.columns([1.05, 0.95], gap="large")

        with left:
            top_a, top_b, top_c = st.columns([1, 1, 1])
            with top_a:
                if st.button("Claim next item", use_container_width=True):
                    q = queue_df()
                    if q.empty:
                        st.warning("No remaining items in your queue.")
                    else:
                        set_current_item_id(q.iloc[0]["item_id"])
                        st.rerun()
            with top_b:
                if st.button("Reload annotations from Hub", use_container_width=True):
                    st.session_state["annotations_df"] = load_all_hub_annotations(annotation_repo)
                    st.rerun()
            with top_c:
                if st.button("Clear current", use_container_width=True):
                    set_current_item_id(None)
                    st.rerun()

            item = current_item_row()
            if item is None:
                st.info("Claim an item to start. The app keeps a per-annotator queue so multiple people can work in parallel.")
            
                q = queue_df().head(10)
            
                # DEBUG: inspect actual dataset schema
                st.write("Dataset columns:", list(q.columns))
            
                if not q.empty:
            
                    # Only use columns that actually exist
                    available_cols = [
                        c for c in [
                            "item_id",
                            "sample_id",
                            "domain",
                            "scenario",
                            "distractor_index"
                        ]
                        if c in q.columns
                    ]
            
                    display = q[available_cols].copy()
            
                    if "distractor_text" in q.columns:
                        display["preview"] = q["distractor_text"].map(preview_text)
            
                    st.dataframe(display, use_container_width=True, hide_index=True)
            
                return

            st.markdown(
                f"""
                <div class="cardbox">
                    <div><span class="badge">Domain</span> {item.get("domain", "")}</div>
                    <div style="margin-top:0.35rem;"><span class="badge">Scenario</span> {item.get("scenario", "")}</div>
                    <div style="margin-top:0.35rem;"><span class="badge">Sample</span> <span class="smallmono">{item.get("sample_id", "")}</span></div>
                    <div style="margin-top:0.35rem;"><span class="badge">Item</span> <span class="smallmono">{item.get("item_id", "")}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.divider()

            tabs = st.tabs(["Context", "Distractor", "Existing annotations"])
            with tabs[0]:
                st.markdown("**System instruction**")
                st.code(item.get("system_instruction", ""), language="text")
                st.markdown("**Conversation**")
                render_turns(json.loads(item.get("conversation_json", "[]")))
            with tabs[1]:
                st.markdown("**Previous assistant turn**")
                st.code(item.get("bot_turn", "") or "(missing)", language="text")
                st.markdown("**Distractor user turn**")
                st.code(item.get("distractor_text", "") or "(missing)", language="text")
            with tabs[2]:
                existing = anns_df[anns_df["item_id"] == item["item_id"]].copy()
                if existing.empty:
                    st.caption("No annotations yet.")
                else:
                    for _, row in existing.iterrows():
                        st.write(f"**{row['annotator']}** · {row['status']} · {row['created_at']}")
                        st.json(row["labels"])
                        if row.get("notes"):
                            st.caption(row["notes"])
                        st.divider()

        with right:
            st.markdown("### Annotation form")
            current_draft = load_draft(st.session_state["annotator"])
            draft_labels = current_draft.get("labels", {}) if current_draft else {}

            with st.form("annotation_form", clear_on_submit=False):
                st.selectbox(
                    "Distractor kind",
                    LABEL_OPTIONS["distractor_kind"],
                    index=LABEL_OPTIONS["distractor_kind"].index(draft_labels.get("distractor_kind", LABEL_OPTIONS["distractor_kind"][0]))
                    if draft_labels.get("distractor_kind") in LABEL_OPTIONS["distractor_kind"]
                    else 0,
                    key="distractor_kind",
                )
                st.selectbox(
                    "Transition style",
                    LABEL_OPTIONS["transition_style"],
                    index=LABEL_OPTIONS["transition_style"].index(draft_labels.get("transition_style", LABEL_OPTIONS["transition_style"][0]))
                    if draft_labels.get("transition_style") in LABEL_OPTIONS["transition_style"]
                    else 0,
                    key="transition_style",
                )
                st.multiselect(
                    "Policy target(s)",
                    LABEL_OPTIONS["policy_target"],
                    default=draft_labels.get("policy_target", []),
                    key="policy_target",
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.slider("Difficulty", 1, 5, value=int(draft_labels.get("difficulty", 3)), key="difficulty")
                    st.slider("Realism", 1, 5, value=int(draft_labels.get("realism", 3)), key="realism")
                with c2:
                    st.selectbox(
                        "Assistant behavior",
                        LABEL_OPTIONS["assistant_behavior"],
                        index=LABEL_OPTIONS["assistant_behavior"].index(draft_labels.get("assistant_behavior", LABEL_OPTIONS["assistant_behavior"][0]))
                        if draft_labels.get("assistant_behavior") in LABEL_OPTIONS["assistant_behavior"]
                        else 0,
                        key="assistant_behavior",
                    )
                    st.slider("Confidence", 1, 5, value=int(draft_labels.get("confidence", 3)), key="confidence")

                st.checkbox(
                    "Multi-turn escalation / persistence",
                    value=bool(draft_labels.get("multi_turn_escalation", False)),
                    key="multi_turn_escalation",
                )
                st.checkbox(
                    "Assistant followed the rule",
                    value=bool(draft_labels.get("rule_followed", True)),
                    key="rule_followed",
                )
                st.checkbox(
                    "Borderline / needs review",
                    value=bool(draft_labels.get("needs_review", False)),
                    key="needs_review",
                )
                notes = st.text_area(
                    "Notes",
                    value=current_draft.get("notes", ""),
                    height=150,
                    placeholder="Explain ambiguity, likely disagreement, or policy edge cases.",
                )
                submitted = st.form_submit_button("Submit to Hugging Face", use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Save draft locally", use_container_width=True):
                    payload = {
                        "current_item_id": item["item_id"],
                        "labels": build_labels_from_state(),
                        "notes": notes,
                        "saved_at": now_iso(),
                    }
                    path = save_draft(st.session_state["annotator"], payload)
                    st.success(f"Draft saved to {path}")
            with c2:
                if st.button("Sync annotation cache", use_container_width=True):
                    st.session_state["annotations_df"] = load_all_hub_annotations(annotation_repo)
                    st.success("Reloaded annotation index from Hub.")

            if submitted:
                labels = build_labels_from_state()
                payload = {
                    "annotation_id": str(uuid.uuid4()),
                    "item_id": item["item_id"],
                    "sample_id": item["sample_id"],
                    "annotator": st.session_state["annotator"],
                    "created_at": now_iso(),
                    "status": "submitted",
                    "labels": labels,
                    "notes": notes,
                    "source": {
                        "source_dataset_repo": source_repo,
                        "source_dataset_split": source_split,
                        "domain": item.get("domain", ""),
                        "scenario": item.get("scenario", ""),
                        "distractor_index": int(item.get("distractor_index", 0)),
                    },
                }
                try:
                    path_in_repo = push_annotation_to_hub(annotation_repo, payload)
                    st.session_state["annotations_df"] = pd.concat(
                        [
                            anns_df,
                            pd.DataFrame(
                                [
                                    {
                                        "item_id": payload["item_id"],
                                        "sample_id": payload["sample_id"],
                                        "annotator": payload["annotator"],
                                        "labels": payload["labels"],
                                        "notes": payload["notes"],
                                        "status": payload["status"],
                                        "created_at": payload["created_at"],
                                        "file_path": path_in_repo,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                    save_draft(
                        st.session_state["annotator"],
                        {
                            "current_item_id": item["item_id"],
                            "labels": labels,
                            "notes": notes,
                            "saved_at": now_iso(),
                        },
                    )
                    st.success(f"Submitted to Hugging Face as {path_in_repo}")
                    q = queue_df()
                    if not q.empty:
                        set_current_item_id(q.iloc[0]["item_id"])
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed. Saved locally only. Error: {e}")
                    save_draft(
                        st.session_state["annotator"],
                        {
                            "current_item_id": item["item_id"],
                            "labels": labels,
                            "notes": notes,
                            "saved_at": now_iso(),
                        },
                    )

            st.caption("Each submission is a separate file in the annotation dataset repo, so multiple annotators can work in parallel without write conflicts.")

    elif page == "Review":
        st.subheader("Inter-rater review")
        multi = (
            anns_df.groupby("item_id")["annotator"].nunique().reset_index(name="n_annotators")
            if not anns_df.empty
            else pd.DataFrame(columns=["item_id", "n_annotators"])
        )
        multi = multi[multi["n_annotators"] >= 2] if not multi.empty else multi

        if multi.empty:
            st.info("No items with at least two annotations yet.")
        else:
            selected_item = st.selectbox("Item with multiple annotations", multi["item_id"].tolist())
            row = items_df[items_df["item_id"] == selected_item].iloc[0].to_dict()
            sample = samples_df[samples_df["sample_id"] == row["sample_id"]].iloc[0].to_dict()
            row.update(sample)

            st.markdown("### Context")
            st.code(row["system_instruction"], language="text")
            st.code(row["bot_turn"] or "", language="text")
            st.code(row["distractor_text"] or "", language="text")

            st.markdown("### Annotations")
            sub = anns_df[anns_df["item_id"] == selected_item].copy()
            cols = st.columns(min(len(sub), 3)) if len(sub) > 0 else st.columns(1)
            for idx, (_, ann) in enumerate(sub.iterrows()):
                with cols[idx % len(cols)]:
                    st.write(f"**{ann['annotator']}**")
                    st.caption(f"{ann['status']} · {ann['created_at']}")
                    st.json(ann["labels"])
                    if ann.get("notes"):
                        st.caption(ann["notes"])

            agreement = compute_agreement(sub, label_key="assistant_behavior")
            c1, c2, c3 = st.columns(3)
            c1.metric("Paired items", agreement["paired_items"])
            c2.metric("Raw agreement", f"{agreement['raw_agreement']:.2%}" if agreement["raw_agreement"] is not None else "n/a")
            c3.metric("Cohen's κ", f"{agreement['cohen_kappa']:.3f}" if agreement["cohen_kappa"] is not None else "n/a")

    elif page == "Dashboard":
        st.subheader("Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Source samples", len(samples_df))
        c2.metric("Source items", len(items_df))
        c3.metric("Annotation files", len(anns_df))
        c4.metric("My queue", len(queue_df()))

        st.markdown("### Progress by annotator")
        if anns_df.empty:
            st.info("No annotations yet.")
        else:
            by_ann = anns_df.groupby("annotator")["item_id"].nunique().reset_index(name="annotated_items").sort_values("annotated_items", ascending=False)
            st.dataframe(by_ann, use_container_width=True, hide_index=True)

            st.markdown("### Progress by domain")
            joined = anns_df.merge(items_df[["item_id", "domain"]], on="item_id", how="left")
            by_domain = joined.groupby("domain")["item_id"].nunique().reset_index(name="annotated_items").sort_values("annotated_items", ascending=False)
            st.dataframe(by_domain, use_container_width=True, hide_index=True)

            st.markdown("### Agreement snapshot")
            metric = compute_agreement(anns_df, label_key="assistant_behavior")
            st.write(metric)

            st.markdown("### Recent annotation previews")
            recent = anns_df.sort_values("created_at", ascending=False).head(20).copy()
            if "labels" in recent.columns:
                recent["assistant_behavior"] = recent["labels"].apply(lambda x: x.get("assistant_behavior") if isinstance(x, dict) else None)
                recent["distractor_kind"] = recent["labels"].apply(lambda x: x.get("distractor_kind") if isinstance(x, dict) else None)
            st.dataframe(
                recent[["annotator", "item_id", "status", "created_at", "assistant_behavior", "distractor_kind", "notes"]],
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.subheader("Export")
        st.write("Export the merged dataset for downstream analysis or model training.")

        merged = items_df.merge(samples_df, on="sample_id", how="left")
        if not anns_df.empty:
            export_df = merged.merge(anns_df[["item_id", "annotator", "labels", "notes", "status", "created_at"]], on="item_id", how="left")
        else:
            export_df = merged.copy()
            export_df["annotator"] = None
            export_df["labels"] = None
            export_df["notes"] = None
            export_df["status"] = None
            export_df["created_at"] = None

        c1, c2 = st.columns(2)
        with c1:
            jsonl = LOCAL_EXPORT_DIR / "annotations_export.jsonl"
            if st.button("Generate JSONL export", use_container_width=True):
                with jsonl.open("w", encoding="utf-8") as f:
                    for _, r in export_df.iterrows():
                        f.write(json.dumps(r.where(pd.notna(r), None).to_dict(), ensure_ascii=False) + "\n")
                st.success(f"Wrote {jsonl}")
                st.download_button("Download JSONL", jsonl.read_text(encoding="utf-8"), file_name=jsonl.name, mime="application/json")
        with c2:
            csv = LOCAL_EXPORT_DIR / "annotations_export.csv"
            if st.button("Generate CSV export", use_container_width=True):
                export_df.to_csv(csv, index=False)
                st.success(f"Wrote {csv}")
                st.download_button("Download CSV", csv.read_text(encoding="utf-8"), file_name=csv.name, mime="text/csv")

        st.markdown("### Repository handoff")
        st.code(
            f"Source repo: {source_repo}\nAnnotation repo: {annotation_repo}\nSplit: {source_split}\nAnnotator: {st.session_state['annotator']}",
            language="text",
        )


if __name__ == "__main__":
    main()
