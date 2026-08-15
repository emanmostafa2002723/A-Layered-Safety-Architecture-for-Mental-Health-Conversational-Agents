"""
Task 1.1 — Step 05: Load and inspect the SHINES dataset
Owner: Eman (P1)
Update SHINES_PATH below to wherever you have the SHINES files.
If not yet available, this script prints a skip message.

Run from repo root:
  python scripts/task1_1/step05_load_shines.py
"""
import os, glob
import pandas as pd

SHINES_PATH = "data/raw/shines"
OUT_PATH    = "data/raw/shines_extracted.jsonl"

IMPLICIT_KEYWORDS = [
    "unalive", "relief lines", "13'ing", "sleeping forever",
    "tired of fighting", "not wake up", "disappear",
    "don't want to be here", "making plans",
    "giving things away", "saying goodbye",
]

def find_files(root):
    return sorted(
        glob.glob(os.path.join(root,"**","*.csv"), recursive=True) +
        glob.glob(os.path.join(root,"**","*.jsonl"), recursive=True) +
        glob.glob(os.path.join(root,"**","*.json"), recursive=True)
    )

def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":   return pd.read_csv(path)
    if ext == ".json":  return pd.read_json(path)
    if ext == ".jsonl": return pd.read_json(path, lines=True)

def is_implicit(text):
    t = str(text).lower()
    return any(kw.lower() in t for kw in IMPLICIT_KEYWORDS)

def main():
    if not os.path.isdir(SHINES_PATH):
        print(f"[step05] SHINES not found at '{SHINES_PATH}' — skip for now.")
        print("         Continue with step06_judge_prompt.py")
        return
    files = find_files(SHINES_PATH)
    dfs = [load_file(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    text_col = next((c for c in ["text","content","post","message"] if c in combined.columns), None)
    if not text_col:
        print("Could not find text column — inspect manually."); return
    combined["is_implicit"] = combined[text_col].apply(is_implicit)
    implicit_df = combined[combined["is_implicit"]].copy()
    out = pd.DataFrame({"text": implicit_df[text_col].fillna(""),
                        "category": "self_harm", "source": "shines", "signal_type": "implicit"})
    out.to_json(OUT_PATH, orient="records", lines=True, force_ascii=False)
    print(f"[step05] Saved {len(out)} implicit examples → {OUT_PATH}")
    print("[step05] ✓ Next: python scripts/task1_1/step06_judge_prompt.py")

if __name__ == "__main__":
    main()
