"""
Task 1.1 — Step 12: Create stratified train/val/test splits (70/15/15)
Owner: Eman (P1)
Run from repo root: python scripts/task1_1/step12_split.py
"""
import os, json
import pandas as pd
from sklearn.model_selection import train_test_split

BALANCED_PATH = "data/raw/combined_balanced.jsonl"
OUT_DIR       = "data/processed/crisis"
RANDOM_STATE  = 42
OUR_CATEGORIES = ["suicidal_ideation","self_harm","violence",
                  "anxiety_crisis","substance_abuse","risk_taking","no_crisis"]

def main():
    if not os.path.exists(BALANCED_PATH):
        print("Run step11 first."); return
    df = pd.read_json(BALANCED_PATH, lines=True)
    print(f"[step12] Loaded {len(df)} rows")
    try:
        train_df, hold = train_test_split(df, test_size=0.30, stratify=df["category"], random_state=RANDOM_STATE)
        val_df, test_df = train_test_split(hold, test_size=0.50, stratify=hold["category"], random_state=RANDOM_STATE)
    except:
        train_df, hold = train_test_split(df, test_size=0.30, random_state=RANDOM_STATE)
        val_df, test_df = train_test_split(hold, test_size=0.50, random_state=RANDOM_STATE)
    print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    os.makedirs(OUT_DIR, exist_ok=True)
    train_df.to_json(f"{OUT_DIR}/train.jsonl", orient="records", lines=True, force_ascii=False)
    val_df.to_json(f"{OUT_DIR}/val.jsonl",   orient="records", lines=True, force_ascii=False)
    test_df.to_json(f"{OUT_DIR}/test.jsonl",  orient="records", lines=True, force_ascii=False)
    summary = {"total":len(df),"train":len(train_df),"val":len(val_df),"test":len(test_df),
               "split":"70/15/15","random_state":RANDOM_STATE,
               "train_dist":train_df["category"].value_counts().to_dict(),
               "val_dist":val_df["category"].value_counts().to_dict(),
               "test_dist":test_df["category"].value_counts().to_dict()}
    with open(f"{OUT_DIR}/split_summary.json","w") as f: json.dump(summary,f,indent=2)
    print(f"\n[step12] Saved to {OUT_DIR}/")
    print("         ⚠ Do NOT use test.jsonl until Phase 4!")
    print("[step12] ✓ Next: python scripts/task1_1/step14_data_report.py")

if __name__ == "__main__":
    main()
