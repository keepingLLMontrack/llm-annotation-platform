# LLM Annotation Platform

A simple Streamlit app for collaborative editing of a human-made distractor dataset.

## What it supports

- browse source data from a Hugging Face dataset repo or a local JSON/JSONL file
- load a row by index into an editor
- create a new blank entry
- edit:
  - `domain`
  - `scenario`
  - `system_instruction`
  - `conversation`
  - `distractors`
  - `distractors_multiturn`
  - `conversation_with_distractors`
- mark the entry with a `split` value (`train` / `test`)
- save drafts in the HF Space bucket path (`/data/drafts`)
- submit each finished entry as a separate JSON file to a Hugging Face dataset repo
- optionally ask a local OpenAI-compatible LLM server such as LM Studio to draft one distractor at a time

## Output shape

The app keeps the source structure and adds provenance fields:

- `split`
- `_review_status`
- `_needs_human_review`
- `_annotator`
- `_source_repo`
- `_source_split`
- `_source_index`
- `_created_at`
- `_updated_at`

That means the final file can still be merged into one dataset later.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Environment variables

Set these in your GitHub repo / HF Space:

- `SOURCE_DATASET_REPO`
- `SOURCE_DATASET_SPLITS`  
  Example: `train,test`
- `ANNOTATION_REPO_ID`
- `HF_TOKEN`

Optional local LLM settings:
- `LLM_BASE_URL` is entered in the sidebar inside the app
- `LLM_MODEL` is entered in the sidebar inside the app

## HF Space setup

Use a Docker Space, mount persistent storage at `/data`, and set the environment variables above. The app stores drafts and submission logs in the bucket path.

## GitHub structure

```text
app.py
requirements.txt
README.md
Dockerfile
.streamlit/config.toml
```
