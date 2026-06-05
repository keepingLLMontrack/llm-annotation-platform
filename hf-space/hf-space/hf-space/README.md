---
title: Distractor Annotation Tool
emoji: 🎯
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.36.0
app_file: app.py
pinned: false
---

# 🎯 Distractor Annotation Tool

Collaborative annotation GUI for the MSc NLP research project **"Keeping LLMs on Track in Task-Oriented Dialogue"**.

## One-time Setup

### 1. Create a private HF Dataset repo for shared annotations
Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset), make it **private**, and note the repo ID (e.g. `yourgroup/distractor-annotations`).

### 2. Set secrets in your HF Space
In your Space → Settings → Repository secrets, add:
| Secret | Value |
|---|---|
| `HF_TOKEN` | Your HF token with **write** access |
| `ANNOTATIONS_REPO_ID` | e.g. `yourgroup/distractor-annotations` |

### 3. Set secrets in your GitHub repo
In GitHub → Settings → Secrets and variables → Actions, add the same `HF_TOKEN`.

### 4. Update the sync workflow
In `.github/workflows/sync_to_hf.yml`, replace `YOUR_HF_USERNAME` and `YOUR_SPACE_NAME` with your actual values.

### 5. Import seed data
On first run, go to the **Dashboard** and click **Import Seed Data** to populate the shared repo with the group's initial entries.

## Workflow

| Page | Purpose |
|---|---|
| 🏠 Dashboard | Stats overview, seed import, config check |
| 📚 Browse | Explore the base nvidia dataset and seed entries |
| ✏️ Annotate | Create multi-turn distractor entries |
| 👥 Annotations | View, edit, review all group work |
| 💬 Test LLM | Send distractors to a live LLM, judge if it gets distracted |

## Annotation Schema

Each annotation follows the `nvidia/CantTalkAboutThis` schema, extended with:
- `distractors_multiturn`: rich multi-turn distractor sequences
- `_id`, `_annotator`, `_review_status`, `_created_at`, `_updated_at`
- `_llm_test_results`: logged results from the Test LLM page

## Base Dataset
[nvidia/CantTalkAboutThis-Topic-Control-Dataset](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset)

## Related Papers
- [2024.findings-emnlp.713](https://aclanthology.org/2024.findings-emnlp.713)
- [arXiv:2511.05018](https://arxiv.org/abs/2511.05018)
