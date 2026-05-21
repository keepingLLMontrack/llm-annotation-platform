# LLM Annotation Platform — Hugging Face native

This version removes the external database layer.

## What it uses

- **Hugging Face Space** for the Streamlit app
- **Hugging Face dataset repo** for the canonical annotation store
- **Hugging Face Storage Bucket** only for persistent local cache / drafts in the Space
- **No Supabase**
- **No separate backend platform**

Hugging Face Spaces provide ephemeral disk by default, and Hugging Face recommends attaching Storage Buckets to persist data across restarts. Buckets are mounted into the Space container as local volumes. citeturn322583view0

## Repository structure

```text
app.py
scripts/seed.py
requirements.txt
README.md
```

## Behavior

Each annotation is written as its own JSON file into the dataset repository:
```text
annotations/<annotator>/<timestamp>_<item_id>_<uuid>.json
```

That design avoids write conflicts between annotators because each submission is a new file, not an overwrite of a shared database row. Repository files on the Hub are versioned, and the Hub supports uploading files to dataset repositories. citeturn322583view1turn322583view4

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to set it up on Hugging Face

### 1. Create two dataset repositories

Create:
- one dataset repo for the **source / seed data**
- one dataset repo for the **annotations**

Hugging Face dataset repositories are created from the Hub UI, and dataset files plus revision history are stored in the repository. citeturn322583view1

### 2. Create a Space

Create a **Streamlit** Space and connect it to your GitHub repository. Spaces host apps directly on the Hub and support Streamlit as a built-in SDK. citeturn322583view2

### 3. Attach a Storage Bucket

Attach a Storage Bucket to the Space and mount it at `/data`.

This is the only stateful storage used by the app. It stores drafts and cache files and survives restarts. Hugging Face documents Storage Buckets as the recommended persistence mechanism for Spaces. citeturn322583view0

### 4. Add secrets

In the Space settings, add:
- `HF_TOKEN` — a Hugging Face token with **write** permission
- `SOURCE_DATASET_REPO`
- `SOURCE_DATASET_SPLIT`
- `ANNOTATION_REPO_ID`

Hugging Face recommends using Space secrets or environment variables instead of hard-coding sensitive values. A write token is required to create repositories or push content to the Hub. citeturn322583view2turn322583view4

### 5. Deploy

Commit the repo to GitHub. Once the Space is linked, it will build from the repository, and the app can upload annotation files to the dataset repo using the Hub API. Hugging Face’s Hub client supports `upload_file()` and `create_commit()` for repository writes. citeturn322583view3turn322583view4

## Suggested workflow for your group

- each person uses a stable annotator name
- each submission creates a new JSON file in the annotation repo
- the Review page shows items with 2+ annotations
- the Dashboard shows per-annotator and per-domain progress
- exports are generated from the merged source + annotation view

## Why this is a good fit

The original source dataset can still be loaded with `datasets.load_dataset(...)`, and the Hugging Face ecosystem is designed for pushing and versioning datasets directly on the Hub. The `datasets` library also provides a `push_to_hub()` path for dataset publishing, while `huggingface_hub` provides lower-level file upload methods when you want more control over file layout. citeturn674332search1turn674332search3turn322583view3
