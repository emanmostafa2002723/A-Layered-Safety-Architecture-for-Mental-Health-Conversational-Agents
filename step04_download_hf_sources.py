"""
Task 1.1 — Step 04: Download all 12 HuggingFace source datasets
Owner: Eman (P1)

What this does:
  - Downloads all 12 datasets listed in Arnaiz-Rodriguez Table 3
  - Saves each to data/raw/hf_sources/<name>/
  - Prints a summary table: name, size, columns, first row
  - Saves data/raw/hf_sources_inventory.json for later steps

Run from repo root:
  python scripts/task1_1/step04_download_hf_sources.py

Note: This can take 10-30 minutes depending on your connection.
      Let it run in the background.
"""

import os
import json
from datasets import load_dataset

RAW_DIR      = "data/raw/hf_sources"
INVENTORY    = "data/raw/hf_sources_inventory.json"

# ─────────────────────────────────────────────────────────────────────────────
# The 12 source datasets from Arnaiz-Rodriguez Table 3
# Also includes SWMH which is large and useful for underrepresented categories
# ─────────────────────────────────────────────────────────────────────────────
HF_SOURCES = {
    "suicidal_ideation":    "cypsiSAS/transformed_Suicidal_ideation",
    "mental_disorder":      "sajjadhadi/Mental-Disorder-Detection-Data",
    "mental_health":        "fadodr/mental_health_dataset",
    "mental_therapy":       "fadodr/mental_health_therapy",
    "mental_panda":         "marmikpandya/mental-health",
    "psyset":               "psycode1/psyset",
    "self_harm_eval":       "arianaazarbal/self-harm-synthetic-eval",
    "self_harm_levels":     "fanyin3639/test_test_self_harm_all_levels",
    "suicidal_finetune":    "richie-ghost/suicidal_finetune",
    "swmh":                 "AIMH/SWMH",
    # These two may require slightly more time to load:
    "nart_100k":            "jerryjalapeno/nart-100k-synthetic",
    "dreaddit":             "e-turcan/dreaddit",     # confirmed CC-BY-4.0
}

# Column that contains the raw text — we try these in order
TEXT_CANDIDATES = ["text", "content", "post", "message", "sentence",
                   "input", "body", "comment"]


def find_text_col(cols: list[str]) -> str | None:
    for c in TEXT_CANDIDATES:
        if c in cols:
            return c
    return None


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    inventory = {}

    print(f"[step04] Downloading {len(HF_SOURCES)} HuggingFace datasets")
    print(f"         Save location: {RAW_DIR}/\n")

    for name, hf_id in HF_SOURCES.items():
        print(f"  ── {name} ({hf_id}) ──────────────────────")
        save_dir = os.path.join(RAW_DIR, name)

        try:
            ds = load_dataset(hf_id, cache_dir=save_dir)
            split = list(ds.keys())[0]
            cols  = ds[split].column_names
            n     = len(ds[split])
            text_col = find_text_col(cols)

            print(f"     Rows:    {n:,}")
            print(f"     Columns: {cols}")
            print(f"     Text col (detected): {text_col}")
            print(f"     Example row: {dict(list(ds[split][0].items())[:3])}")

            inventory[name] = {
                "hf_id":     hf_id,
                "split":     split,
                "n_rows":    n,
                "columns":   cols,
                "text_col":  text_col,
                "save_dir":  save_dir,
                "status":    "OK",
            }

        except Exception as e:
            print(f"     ⚠  FAILED: {e}")
            inventory[name] = {
                "hf_id":  hf_id,
                "status": "FAILED",
                "error":  str(e),
            }

        print()

    # Save inventory
    with open(INVENTORY, "w") as f:
        json.dump(inventory, f, indent=2)

    # Print summary table
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_ok   = sum(1 for v in inventory.values() if v["status"] == "OK")
    total_rows = sum(v.get("n_rows", 0) for v in inventory.values())
    print(f"Downloaded: {total_ok}/{len(HF_SOURCES)} datasets")
    print(f"Total rows: {total_rows:,}")
    print()

    failed = [(k, v) for k, v in inventory.items() if v["status"] == "FAILED"]
    if failed:
        print("FAILED datasets — check internet connection or HF ID:")
        for k, v in failed:
            print(f"  {k}: {v.get('error', '?')}")

    print(f"\n[step04] Inventory saved → {INVENTORY}")
    print("[step04] ✓ Next step: python scripts/task1_1/step05_load_shines.py")


if __name__ == "__main__":
    main()
