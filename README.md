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

### 1. Create a private HF Dataset repo for shared annotations
Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset), make it **private**, and note the repo ID (e.g. `yourgroup/distractor-annotations`).

### 2. Set secrets in your HF Space
In your Space → Settings → Repository secrets, add:
| Secret | Value |
|---|---|
| `HF_TOKEN` | Your HF token with **write** access |
| `ANNOTATIONS_REPO_ID` | e.g. `yourgroup/distractor-annotations` |

### 3. Set up LM Studio (local LLM server)
1. Install LM Studio.
2. Download and load a chat model.
3. Start the local server in LM Studio (OpenAI-compatible API).
4. Confirm the server endpoint is available (default: `http://localhost:1234/v1`).
5. Note the loaded model name exactly as shown in LM Studio.

### 4. Create local Streamlit secrets (`.streamlit/secrets.toml`)
Create a file at `.streamlit/secrets.toml` with:

```toml
# Hugging Face dataset sync
HF_TOKEN = "hf_xxx"
ANNOTATIONS_REPO_ID = "yourgroup/distractor-annotations"

# LM Studio (local OpenAI-compatible server)
OPENAI_BASE_URL = "http://localhost:1234/v1"
OPENAI_API_KEY = "lm-studio"
LLM_MODEL = "google_gemma-4-e2b-it"
```

Notes:
- `OPENAI_API_KEY` can be any non-empty value for LM Studio local usage.
- `LLM_MODEL` must match the model id loaded in LM Studio.

### 5. Set secrets in your GitHub repo
In GitHub → Settings → Secrets and variables → Actions, add the same `HF_TOKEN`.

### 6. Update the sync workflow
In `.github/workflows/sync_to_hf.yml`, replace `YOUR_HF_USERNAME` and `YOUR_SPACE_NAME` with your actual values.

### 7. Run Locally

Run the following in a PowerShell terminal from the project root:

```powershell
$env:OPENAI_BASE_URL = "http://localhost:1234/v1"
$env:LLM_MODEL = "google_gemma-4-e2b-it"
$env:OPENAI_API_KEY = "lm-studio"
py -m streamlit run app.py
```

This starts the Streamlit app and opens it in your default browser.

### 8. Enter name into field on the left hand side

### 9. Check configuration settings

### 10. You're good to go


## Base Dataset
[nvidia/CantTalkAboutThis-Topic-Control-Dataset](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset)

## Related Papers
- [2024.findings-emnlp.713](https://aclanthology.org/2024.findings-emnlp.713)
- [arXiv:2511.05018](https://arxiv.org/abs/2511.05018)
