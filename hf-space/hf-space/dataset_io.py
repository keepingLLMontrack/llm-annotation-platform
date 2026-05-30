"""
Dataset I/O utilities.

Reads:
  - nvidia/CantTalkAboutThis-Topic-Control-Dataset (public HF dataset)
  - seed_data/draft_distractors.json (bundled seed from initial group work)
  - {ANNOTATIONS_REPO_ID}/annotations.json (private HF dataset repo)

Writes:
  - {ANNOTATIONS_REPO_ID}/annotations.json via HF Hub API (requires HF_TOKEN)
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_hf_token() -> Optional[str]:
    """Return HF token from env or Streamlit secrets."""
    val = os.environ.get("HF_TOKEN")
    if val:
        return val
    try:
        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


def get_annotations_repo() -> Optional[str]:
    """Return the annotations dataset repo ID."""
    val = os.environ.get("ANNOTATIONS_REPO_ID")
    if val:
        return val
    try:
        return st.secrets.get("ANNOTATIONS_REPO_ID")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Base dataset (public, cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_base_dataset() -> list[dict]:
    """
    Load the nvidia/CantTalkAboutThis dataset from HF Hub.
    Returns a flat list of entries across all splits.
    Cached for 1 hour to avoid hammering the HF API.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError:
        return []

    try:
        token = get_hf_token()
        ds = load_dataset(
            "nvidia/CantTalkAboutThis-Topic-Control-Dataset",
            token=token,
        )
        entries = []
        for split_name, split in ds.items():
            for item in split:
                entry = dict(item)
                entry["_split"] = split_name
                entry["_source"] = "base_dataset"
                entries.append(entry)
        return entries
    except Exception as e:
        st.warning(f"Could not load base dataset from HF Hub: {e}")
        return []


# ---------------------------------------------------------------------------
# Seed data (bundled with the repo)
# ---------------------------------------------------------------------------

def load_seed_data() -> list[dict]:
    """Load the group's initial draft entries from seed_data/."""
    seed_path = Path(__file__).parent.parent / "seed_data" / "draft_distractors.json"
    if seed_path.exists():
        with open(seed_path, encoding="utf-8") as f:
            data = json.load(f)
        for entry in data:
            entry.setdefault("_source", "seed_data")
        return data
    return []


# ---------------------------------------------------------------------------
# Annotations (private HF dataset repo)
# ---------------------------------------------------------------------------

def _ensure_repo_exists(repo_id: str, token: str) -> None:
    """Create the annotations dataset repo if it doesn't exist yet."""
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
        api = HfApi()
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
    except Exception:
        # Repo doesn't exist — create it
        try:
            from huggingface_hub import HfApi  # noqa: PLC0415
            api = HfApi()
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=True,
                token=token,
            )
        except Exception as create_err:
            st.error(f"Could not create annotations repo '{repo_id}': {create_err}")


def load_annotations() -> list[dict]:
    """
    Load saved annotations from the private HF dataset repo.
    Falls back to seed data if the repo or file doesn't exist yet.
    """
    repo_id = get_annotations_repo()
    token = get_hf_token()

    if not repo_id or not token:
        return load_seed_data()

    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        # bust_cache=True ensures we always get fresh data
        filepath = hf_hub_download(
            repo_id=repo_id,
            filename="annotations.json",
            repo_type="dataset",
            token=token,
            force_download=True,
        )
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # File not in repo yet → seed data
        return load_seed_data()


def save_annotations(annotations: list[dict]) -> bool:
    """
    Overwrite annotations.json in the private HF dataset repo.
    Returns True on success, False on failure.
    """
    repo_id = get_annotations_repo()
    token = get_hf_token()

    if not repo_id:
        st.error("ANNOTATIONS_REPO_ID is not set. Set it in your Space secrets.")
        return False
    if not token:
        st.error("HF_TOKEN is not set. Set it in your Space secrets.")
        return False

    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
        _ensure_repo_exists(repo_id, token)
        api = HfApi()
        content = json.dumps(annotations, indent=2, ensure_ascii=False)
        api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo="annotations.json",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            commit_message=f"Update annotations [{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}]",
        )
        return True
    except Exception as e:
        st.error(f"Failed to save annotations: {e}")
        return False


def upsert_annotation(annotation: dict) -> bool:
    """
    Add (if new) or update (if existing) a single annotation by _id.
    Performs a load-modify-save cycle to avoid overwriting concurrent changes.
    """
    current = load_annotations()

    ann_id = annotation.get("_id")
    annotation["_updated_at"] = datetime.now().isoformat()

    if ann_id:
        for i, a in enumerate(current):
            if a.get("_id") == ann_id:
                current[i] = annotation
                return save_annotations(current)

    # New annotation
    annotation["_id"] = str(uuid.uuid4())
    annotation.setdefault("_created_at", datetime.now().isoformat())
    current.append(annotation)
    return save_annotations(current)


def delete_annotation(ann_id: str) -> bool:
    """Remove an annotation by _id."""
    current = load_annotations()
    updated = [a for a in current if a.get("_id") != ann_id]
    if len(updated) == len(current):
        st.warning("Annotation not found.")
        return False
    return save_annotations(updated)


def build_empty_annotation(annotator_name: str = "") -> dict:
    """Return a blank annotation dict matching the project schema."""
    return {
        "_id": str(uuid.uuid4()),
        "_annotator": annotator_name,
        "_review_status": "draft",
        "_needs_human_review": True,
        "_created_at": datetime.now().isoformat(),
        "_updated_at": datetime.now().isoformat(),
        "_llm_test_results": [],
        "domain": "",
        "scenario": "",
        "system_instruction": "",
        "conversation": [],
        "distractors": [],               # simple format, dataset-compatible
        "distractors_multiturn": [],     # rich multi-turn format
        "conversation_with_distractors": [],
    }
