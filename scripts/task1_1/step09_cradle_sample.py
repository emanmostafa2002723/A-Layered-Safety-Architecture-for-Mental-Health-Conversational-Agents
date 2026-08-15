"""
Task 1.1 — Step 09: Sample 10% of CRADLE Bench for domain adaptation
Owner: Eman (P1)
Update CRADLE_PATH to your actual file location.
Run from repo root: python scripts/task1_1/step09_cradle_sample.py
"""
import os, sys, glob, json
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from step06_judge_prompt import classify_with_judge, get_client

CRADLE_PATH    = "data/raw/cradle"
TRAIN_OUT      = "data/raw/cradle_train_10pct.jsonl"
TEST_OUT       = "data/raw/cradle_test_90pct.jsonl"
RANDOM_STATE   = 42

CRADLE_LABEL_MAP = {
    "suicidal":"suicidal_ideation","suicide":"suicidal_ideation",
    "self-harm":"self_harm","self_harm":"self_harm",
    "violence":"violence","domestic":"violence",
    "substance":"substance_abuse","anxiety":"anxiety_crisis",
    "risk":"risk_taking","no_crisis":"no_crisis","safe":"no_crisis",
}

def map_label(raw):
    if pd.isna(raw): return "UNMAPPED"
    clean = str(raw).strip().lower()
    for k,v in CRADLE_LABEL_MAP.items():
        if k in clean: return v
    return "UNMAPPED"

def main():
    if not os.path.isdir(CRADLE_PATH):
        print(f"[step09] CRADLE not found at '{CRADLE_PATH}' — skip for now.")
        print("         Continue with step10_merge_all.py")
        return
    files = sorted(glob.glob(os.path.join(CRADLE_PATH,"**","*.csv"),recursive=True) +
                   glob.glob(os.path.join(CRADLE_PATH,"**","*.jsonl"),recursive=True))
    dfs = [pd.read_csv(f) if f.endswith(".csv") else pd.read_json(f,lines=True) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    text_col  = next((c for c in ["text","content","utterance","sentence"] if c in combined.columns), None)
    label_col = next((c for c in ["label","category","crisis_type"] if c in combined.columns), None)
    if not text_col: print("No text column found."); return
    if label_col:
        combined["category"] = combined[label_col].apply(map_label)
    else:
        client = get_client()
        from tqdm import tqdm
        combined["category"] = [classify_with_judge(t, client)
                                 for t in tqdm(combined[text_col].fillna("").tolist())]
    clean = pd.DataFrame({"text":combined[text_col].fillna(""),"category":combined["category"],"source":"cradle"})
    clean = clean[clean["category"]!="UNMAPPED"].copy()
    try:
        train_df, test_df = train_test_split(clean, test_size=0.9, stratify=clean["category"], random_state=RANDOM_STATE)
    except:
        train_df, test_df = train_test_split(clean, test_size=0.9, random_state=RANDOM_STATE)
    train_df.to_json(TRAIN_OUT, orient="records", lines=True, force_ascii=False)
    test_df.to_json(TEST_OUT,  orient="records", lines=True, force_ascii=False)
    print(f"[step09] Train (10%): {len(train_df)} → {TRAIN_OUT}")
    print(f"[step09] Test  (90%): {len(test_df)} → {TEST_OUT}  ← DO NOT TOUCH until Phase 4")
    print("[step09] ✓ Next: python scripts/task1_1/step10_merge_all.py")

if __name__ == "__main__":
    main()
