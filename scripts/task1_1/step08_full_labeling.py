"""
Task 1.1 — Step 08: Full LLM-as-judge labelling on 239K HF corpus
Owner: Rania (P2) runs; Eman (P1) reviews output
Cost: ~$8-12 for full corpus

WARNING: Only run AFTER step07 pilot passes (accuracy >= 75%).

Run from repo root:
  python scripts/task1_1/step08_full_labeling.py
"""
import os, sys, json, time
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(__file__))
from step06_judge_prompt import classify_with_judge, get_client

INVENTORY = "data/raw/hf_sources_inventory.json"
OUT_DIR   = "data/raw/hf_labelled"

PRIORITY_KEYWORDS = {
    "violence":      ["want to hurt","kill them","harm someone","intrusive thought",
                      "violent thought","can't control my anger","rage"],
    "risk_taking":   ["don't care","stopped caring","reckless","drunk driving",
                      "driving drunk","don't care if","risky","death wish"],
    "anxiety_crisis":["panic attack","can't breathe","heart racing","dissociat",
                      "shaking uncontrollably","paralyzed"],
}

def has_priority_kw(text):
    t = str(text).lower()
    return any(kw in t for kws in PRIORITY_KEYWORDS.values() for kw in kws)

def get_text(row, col):
    return str(row.get(col,"") or "")

def label_dataset(name, info, client, out_dir):
    if info.get("status") != "OK":
        print(f"  Skipping {name} — download failed"); return
    out_path = os.path.join(out_dir, f"{name}.jsonl")
    text_col = info.get("text_col")
    if not text_col:
        print(f"  No text col for {name}"); return
    already = set()
    if os.path.exists(out_path):
        existing = pd.read_json(out_path, lines=True)
        already = set(range(len(existing)))
        print(f"  Resuming — {len(already)} rows done")
    ds  = load_dataset(info["hf_id"], cache_dir=f"data/raw/hf_sources/{name}")
    spl = list(ds.keys())[0]
    rows = ds[spl]
    total = len(rows)
    to_do = [i for i in range(total) if i not in already]
    if total >= 5000:
        to_do = [i for i in to_do if has_priority_kw(get_text(rows[i], text_col))]
        print(f"  Pre-filtered: {len(to_do)}/{total}")
    with open(out_path, "a", encoding="utf-8") as f:
        for idx in tqdm(to_do, desc=f"  {name}"):
            text  = get_text(rows[idx], text_col)
            if not text.strip(): continue
            label = classify_with_judge(text, client)
            if label == "no_crisis": continue
            f.write(json.dumps({"text":text[:512],"category":label,"source":name,"row_idx":idx}, ensure_ascii=False)+"\n")
            time.sleep(0.02)

def main():
    pilot = "data/raw/pilot_results.jsonl"
    if not os.path.exists(pilot):
        print("Run step07_pilot_test.py first."); return
    acc = pd.read_json(pilot, lines=True)["match"].mean()
    if acc < 0.75:
        print(f"Pilot accuracy {acc:.1%} < 75% — fix the prompt first."); return
    print(f"[step08] Pilot {acc:.1%} ✓ — starting full labelling (~$8-12)")
    os.makedirs(OUT_DIR, exist_ok=True)
    client = get_client()
    with open(INVENTORY) as f: inventory = json.load(f)
    for name, info in inventory.items():
        print(f"\n── {name} ──")
        try: label_dataset(name, info, client, OUT_DIR)
        except Exception as e: print(f"  ERROR: {e}")
    total = sum(len(pd.read_json(os.path.join(OUT_DIR,f), lines=True))
                for f in os.listdir(OUT_DIR) if f.endswith(".jsonl"))
    print(f"\n[step08] Total labelled: {total}")
    print("[step08] ✓ Next: python scripts/task1_1/step09_cradle_sample.py")

if __name__ == "__main__":
    main()
