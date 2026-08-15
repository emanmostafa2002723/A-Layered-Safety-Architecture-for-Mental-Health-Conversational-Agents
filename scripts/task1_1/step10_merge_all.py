"""
Task 1.1 — Step 10: Merge all sources into one dataset
Owner: Eman (P1)
Run from repo root: python scripts/task1_1/step10_merge_all.py
"""
import os, glob, hashlib
import pandas as pd

SOURCES = {"arnaiz":"data/raw/arnaiz_mapped.jsonl",
           "shines":"data/raw/shines_extracted.jsonl",
           "cradle":"data/raw/cradle_train_10pct.jsonl"}
HF_DIR   = "data/raw/hf_labelled"
OUT_PATH = "data/raw/combined_all.jsonl"
OUR_CATEGORIES = ["suicidal_ideation","self_harm","violence",
                  "anxiety_crisis","substance_abuse","risk_taking","no_crisis"]

def hash_text(t): return hashlib.md5(str(t).strip().lower().encode()).hexdigest()

def main():
    all_dfs = []
    for name, path in SOURCES.items():
        if os.path.exists(path):
            df = pd.read_json(path, lines=True)
            print(f"  {name}: {len(df)} rows")
            all_dfs.append(df[["text","category","source"]])
        else:
            print(f"  {name}: not found — skipping")
    hf_files = glob.glob(os.path.join(HF_DIR,"*.jsonl"))
    hf_total = 0
    for fp in sorted(hf_files):
        df = pd.read_json(fp, lines=True)
        if "source" not in df.columns:
            df["source"] = os.path.splitext(os.path.basename(fp))[0]
        all_dfs.append(df[["text","category","source"]])
        hf_total += len(df)
    if hf_total: print(f"  hf_labelled: {hf_total} rows from {len(hf_files)} files")
    combined = pd.concat(all_dfs, ignore_index=True)
    combined["text"] = combined["text"].fillna("").astype(str).str.strip()
    combined = combined[combined["text"].str.len() >= 10]
    combined = combined[combined["category"].isin(OUR_CATEGORIES)]
    combined["_h"] = combined["text"].apply(hash_text)
    before = len(combined)
    combined = combined.drop_duplicates(subset="_h").drop(columns="_h")
    print(f"\nDeduplicated: {before - len(combined)} removed")
    print(f"Final: {len(combined)} rows\n")
    print("Category distribution:")
    dist = combined["category"].value_counts()
    for cat in OUR_CATEGORIES:
        n = dist.get(cat,0)
        flag = " ⚠ LOW" if n < 100 else ""
        print(f"  {cat:<22} {n:>6}{flag}")
    combined.to_json(OUT_PATH, orient="records", lines=True, force_ascii=False)
    print(f"\n[step10] Saved {len(combined)} rows → {OUT_PATH}")
    print("[step10] ✓ Next: python scripts/task1_1/step11_check_and_balance.py")

if __name__ == "__main__":
    main()
