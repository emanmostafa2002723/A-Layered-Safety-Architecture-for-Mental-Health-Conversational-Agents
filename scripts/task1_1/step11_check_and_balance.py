"""
Task 1.1 — Step 11: Check distribution and balance
Owner: Eman (P1) checks; Rania (P2) applies balancing
Run from repo root: python scripts/task1_1/step11_check_and_balance.py
"""
import os, re, random
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split

COMBINED_PATH = "data/raw/combined_all.jsonl"
OUT_PATH      = "data/raw/combined_balanced.jsonl"
OUR_CATEGORIES = ["suicidal_ideation","self_harm","violence",
                  "anxiety_crisis","substance_abuse","risk_taking"]
RANDOM_STATE  = 42
MIN_PER_CAT   = 300
NO_CRISIS_MULT = 3
random.seed(RANDOM_STATE); np.random.seed(RANDOM_STATE)

def light_augment(text):
    ops = [lambda t:t, lambda t:t+" ", lambda t:re.sub(r"\s+"," ",t)]
    return random.choice(ops)(str(text).strip())

def oversample(df, cat, target):
    cat_df = df[df["category"]==cat].copy()
    if len(cat_df) >= target: return cat_df
    extra = cat_df.sample(target-len(cat_df), replace=True, random_state=RANDOM_STATE)
    extra["text"] = extra["text"].apply(light_augment)
    extra["source"] = extra["source"]+"_oversampled"
    return pd.concat([cat_df,extra], ignore_index=True)

def main():
    if not os.path.exists(COMBINED_PATH):
        print(f"Run step10 first."); return
    df = pd.read_json(COMBINED_PATH, lines=True)
    print(f"[step11] Loaded {len(df)} rows\nBEFORE:")
    dist = df["category"].value_counts()
    for cat in OUR_CATEGORIES+["no_crisis"]:
        print(f"  {cat:<22} {dist.get(cat,0):>6}{'  ⚠' if dist.get(cat,0)<MIN_PER_CAT and cat!='no_crisis' else ''}")
    parts = []
    counts = {}
    for cat in OUR_CATEGORIES:
        odf = oversample(df, cat, MIN_PER_CAT)
        parts.append(odf); counts[cat] = len(odf)
    cap = min(counts.values()) * NO_CRISIS_MULT
    nc = df[df["category"]=="no_crisis"].copy()
    if len(nc) > cap: nc = nc.sample(cap, random_state=RANDOM_STATE)
    parts.append(nc)
    balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print("\nAFTER:")
    dist2 = balanced["category"].value_counts()
    for cat in OUR_CATEGORIES+["no_crisis"]:
        print(f"  {cat:<22} {dist2.get(cat,0):>6}")
    balanced.to_json(OUT_PATH, orient="records", lines=True, force_ascii=False)
    print(f"\n[step11] Saved {len(balanced)} → {OUT_PATH}")
    print("[step11] ✓ Next: python scripts/task1_1/step12_split.py")

if __name__ == "__main__":
    main()
