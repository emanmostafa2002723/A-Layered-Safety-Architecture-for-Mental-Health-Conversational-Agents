# Layered Safety Guardrails for Mental Health Conversational Agents

**Queen's University — M.S. Graduation Project**
**Supervisor:** Prof. M. Anwar Hossain (ahossain@queensu.ca)
**Code deadline:** October 15, 2026

---

## Team

| Person | Role | Branch |
|--------|------|--------|
| Eman Elsayed (P1) | Evaluation + Data Pipeline | `p1-evaluation-memory` |
| Rania Atia (P2) | Layer 1 Input Guardrail | `p2-layer1-input` |
| Nourhan Salem (P3) | Layer 2 Output Guardrail | `p3-layer2-output` |
| Lamya Mamdouh Sayed (P4) | Layer 3 + LangGraph Pipeline | `p4-layer3-pipeline` |

---

## What We Are Building

A three-layer safety system wrapping around the WellBeing Agent (Shafi, 2025).

```
User Message
    ↓
[G1 — Input Guardrail]          ← Rania (P2)
    ↓
[WellBeing Agent — LLaMA 3.2 3B] ←→ [Tool Guardrail]   ← Lamya (P4)
    ↓
[G2 — Output Guardrail]         ← Nourhan (P3)
    ↓
[G3 — Escalation & Audit]       ← Lamya (P4)
    ↓
Safe Response to User
```

Crisis bypass: if G1 crisis confidence > 0.75, route directly to G3 (skip LLM).

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/emanmostafa2002723/wellbeing-guardrails.git
cd wellbeing-guardrails

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# 4. Copy and fill in your API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

---

## Folder Structure

```
wellbeing-guardrails/
├── data/
│   ├── raw/                 ← Downloaded datasets (not committed to git)
│   ├── processed/
│   │   ├── crisis/          ← Final train/val/test splits (Task 1.1)
│   │   ├── boundary_input/  ← Boundary classifier data (Task 1.2)
│   │   └── boundary_output/ ← Output boundary data (Task 1.3)
│   └── adversarial/         ← Red-team prompts (Task 1.4)
├── guardrails/
│   ├── layer1/              ← P2: crisis, boundary, PII, injection
│   ├── layer2/              ← P3: hallucination, boundary, empathy
│   ├── layer3/              ← P4: escalation, audit
│   └── pipeline/            ← P4: LangGraph state + graph
├── finetuning/              ← LLaMA fine-tuning scripts
├── evaluation/              ← P1: metrics harness
├── models/                  ← Saved checkpoints (not committed)
├── notebooks/               ← Exploratory notebooks
└── scripts/
    └── task1_1/             ← P1+P2: data pipeline scripts (run in order)
```

---

## Running Task 1.1 — Crisis Classifier Data Pipeline

Run scripts in this exact order:

```bash
cd scripts/task1_1

# Step 1 — Download Arnaiz-Rodriguez
bash step01_download_arnaiz.sh

# Step 2 — Inspect what you downloaded
python step02_inspect_arnaiz.py

# Step 3 — Download all 12 HuggingFace source datasets
python step03_download_hf_sources.py

# Step 4 — Load SHINES dataset
python step04_load_shines.py

# Step 5 — Define and test the judge prompt (read this file, edit if needed)
# No run needed — this file is imported by later steps

# Step 6 — Run pilot test on 100 Arnaiz examples (costs ~$0.05)
python step06_pilot_test.py

# Step 7 — Run full LLM labelling on 239K corpus (costs ~$10)
# Only run after Step 6 passes!
python step07_full_labeling.py

# Step 8 — Sample 10% of CRADLE Bench for domain adaptation
python step08_cradle_sample.py

# Step 9 — Merge all sources into one dataset
python step09_merge_all.py

# Step 10 — Check class distribution
python step10_check_distribution.py

# Step 11 — Balance / oversample underrepresented categories
python step11_balance.py

# Step 12 — Create stratified train/val/test splits
python step12_split.py

# Step 14 — Generate data report
python step14_data_report.py
```

---

## Git Workflow

```bash
# Always work on your own branch
git checkout p1-evaluation-memory

# Before starting each day
git pull origin main

# After finishing a script
git add scripts/task1_1/stepXX_...py
git commit -m "task1.1: add stepXX description"
git push origin p1-evaluation-memory

# When a full task is done, open a Pull Request to main
```
