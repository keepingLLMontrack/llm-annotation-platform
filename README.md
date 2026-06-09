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

Annotation platform for the MSc NLP project **"Keeping LLMs on Track in
Task-Oriented Dialogue."** It implements the required workflow end-to-end:

1. **Browse & Select** — pick a real conversation from
   [`nvidia/CantTalkAboutThis-Topic-Control-Dataset`](https://huggingface.co/datasets/nvidia/CantTalkAboutThis-Topic-Control-Dataset)
   and choose the user turn where a distractor is injected.
2. **Annotate** — define the distractor goal, write a **shared first distractor
   turn**, then run it against **Llama 3.1 8B Instruct** and **gpt-oss-20b**
   interactively (run → read response → write the next turn, up to 3 turns).
   Label every response and assign an overall outcome per model.
3. **Review & Export** — export one row per (scenario, model) in the exact
   required column schema (CSV/JSON).

The system prompt (domain policy) is always passed as a real **system message**.
Only the final, user-facing response is annotated — hidden reasoning is shown
greyed-out and never labelled.

## Target: 18 scenarios (~2 per domain × 9 domains), optionally 27.

---

## Setup

### 1. Create a private HF dataset repo for shared annotations
Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset), make it
**private**, and note the repo ID (e.g. `yourgroup/distractor-annotations`).

### 2. Configure secrets
Locally, create `.streamlit/secrets.toml` (or use environment variables — see
`.env.example`):

```toml
HF_TOKEN = "hf_xxx"
ANNOTATIONS_REPO_ID = "yourgroup/distractor-annotations"

# Model serving (OpenAI-compatible endpoint, e.g. LM Studio)
OPENAI_BASE_URL = "http://localhost:1234/v1"
OPENAI_API_KEY  = "lm-studio"

# Optional per-model overrides (e.g. gpt-oss on a hosted endpoint)
# LLAMA_MODEL   = "meta-llama-3.1-8b-instruct"
# GPTOSS_BASE_URL = "https://your-endpoint/v1"
# GPTOSS_API_KEY  = "..."
# GPTOSS_MODEL    = "openai/gpt-oss-20b"
```

In an HF Space, add `HF_TOKEN` and `ANNOTATIONS_REPO_ID` under
**Settings → Repository secrets**.

### 3. Serve the two models
Both are called through one OpenAI-compatible API. With **LM Studio**: load the
model(s), Start Server, and set the served-model ids via `LLAMA_MODEL` /
`GPTOSS_MODEL` to match what LM Studio shows. `gpt-oss-20b` is large — if it is
infeasible locally, point `GPTOSS_BASE_URL` / `GPTOSS_API_KEY` at a hosted
endpoint, or document the limitation.

### 4. Decoding
Fixed defaults: `temperature = 0.2`, `top_p = 1.0` (adjustable per scenario).
Both values are recorded in every export row.

---

## Run locally (Windows PowerShell)

```powershell
cd C:\Users\elipe\Desktop\NLP\llm-annotation-platform
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Enter your name in the sidebar, then work through the pages **1 → 2 → 3**.

---

## Export schema (one row per scenario+model)

```
annotator_name, second_annotator_name_if_any, scenario_id, domain,
conversation_id, selected_turn_id, system_prompt_or_policy_summary,
original_user_message, distractor_goal, targeted_rule_or_scope_boundary,
why_this_is_a_good_distractor, model_name, temperature, top_p,
distractor_turn_1, model_response_1, response_label_1,
distractor_turn_2, model_response_2, response_label_2,
distractor_turn_3_if_any, model_response_3_if_any, response_label_3_if_any,
overall_outcome, short_justification, disagreement_notes_if_any, notes
```

The Llama row and the gpt-oss row for the same original scenario share the same
`scenario_id`.

## Related
- Paper: [2024.findings-emnlp.713](https://aclanthology.org/2024.findings-emnlp.713)
- [arXiv:2511.05018](https://arxiv.org/abs/2511.05018)
