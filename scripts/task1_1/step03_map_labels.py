"""
Task 1.1 — Step 03: Map Arnaiz-Rodriguez labels to our 6-category taxonomy
Owner: Eman (P1)

What this does:
  - Loads the Arnaiz dataset
  - Maps raw label strings → our 6 standardised categories
  - Flags any unmapped labels so you can fix the dict
  - Saves a clean DataFrame to data/raw/arnaiz_mapped.jsonl

Run from repo root:
  python scripts/task1_1/step03_map_labels.py

IMPORTANT: After running step02, look at the printed label names.
           Update LABEL_MAP below to match exactly what you saw.
           The map below covers the most likely variants — add any missing ones.
"""

import os
import json
import glob
import pandas as pd

RAW_DIR  = "data/raw/arnaiz-rodriguez"
OUT_PATH = "data/raw/arnaiz_mapped.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# UPDATE THIS after running step02 and seeing the real label names
# Keys = exactly what Arnaiz uses (lowercase stripped)
# Values = our 6 taxonomy categories or "no_crisis"
# ─────────────────────────────────────────────────────────────────────────────
LABEL_MAP = {
    # Suicidal ideation variants
    "suicidal ideation":        "suicidal_ideation",
    "suicidal_ideation":        "suicidal_ideation",
    "suicide":                  "suicidal_ideation",
    "suicidal":                 "suicidal_ideation",
    "si":                       "suicidal_ideation",

    # Self-harm variants
    "self-harm":                "self_harm",
    "self_harm":                "self_harm",
    "self harm":                "self_harm",
    "selfharm":                 "self_harm",
    "sh":                       "self_harm",

    # Violence variants
    "violence":                 "violence",
    "violent":                  "violence",
    "violent thoughts":         "violence",
    "aggression":               "violence",
    "harm to others":           "violence",

    # Anxiety crisis variants
    "anxiety crisis":           "anxiety_crisis",
    "anxiety_crisis":           "anxiety_crisis",
    "anxiety":                  "anxiety_crisis",
    "panic":                    "anxiety_crisis",
    "panic attack":             "anxiety_crisis",

    # Substance abuse variants
    "substance abuse":          "substance_abuse",
    "substance_abuse":          "substance_abuse",
    "substance":                "substance_abuse",
    "drug":                     "substance_abuse",
    "alcohol":                  "substance_abuse",

    # Risk-taking variants
    "risk-taking":              "risk_taking",
    "risk_taking":              "risk_taking",
    "risk taking":              "risk_taking",
    "reckless":                 "risk_taking",
    "risky behavior":           "risk_taking",

    # Non-crisis
    "no crisis":                "no_crisis",
    "no_crisis":                "no_crisis",
    "non-crisis":               "no_crisis",
    "non_crisis":               "no_crisis",
    "safe":                     "no_crisis",
    "normal":                   "no_crisis",
    "neutral":                  "no_crisis",
    "0":                        "no_crisis",
}

OUR_CATEGORIES = [
    "suicidal_ideation",
    "self_harm",
    "violence",
    "anxiety_crisis",
    "substance_abuse",
    "risk_taking",
    "no_crisis",
]


def find_best_file(root: str) -> str | None:
    """Return the most likely main data file."""
    for ext in ["*.csv", "*.jsonl", "*.json"]:
        matches = glob.glob(os.path.join(root, "**", ext), recursive=True)
        # Prefer files NOT in subdirs called 'test' only
        main_files = [m for m in matches if "test" not in m.lower() or
                      "train" in m.lower() or len(matches) == 1]
        if main_files:
            return main_files[0]
    return None


def load_df(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext == ".json":
        return pd.read_json(path)
    elif ext == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported extension: {ext}")


def find_text_col(df: pd.DataFrame) -> str:
    for candidate in ["text", "content", "message", "input", "post", "sentence"]:
        if candidate in df.columns:
            return candidate
    # Fall back to longest string column
    str_cols = df.select_dtypes(include="object").columns.tolist()
    if str_cols:
        return max(str_cols, key=lambda c: df[c].str.len().mean())
    raise ValueError("Could not find a text column. Inspect and set manually.")


def find_label_col(df: pd.DataFrame) -> str:
    for candidate in ["label", "category", "crisis_type", "class",
                      "crisis_category", "type"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Could not find a label column. Inspect and set manually.")


def map_label(raw_label) -> str:
    if pd.isna(raw_label):
        return "UNMAPPED"
    clean = str(raw_label).strip().lower()
    return LABEL_MAP.get(clean, "UNMAPPED")


def main():
    print("[step03] Mapping Arnaiz-Rodriguez labels to 6-category taxonomy")

    path = find_best_file(RAW_DIR)
    if path is None:
        print(f"ERROR: No data file found in {RAW_DIR}")
        print("Run step01 first.")
        return

    print(f"         Loading: {path}")
    df = load_df(path)
    print(f"         Loaded {len(df)} rows, {len(df.columns)} columns")

    text_col  = find_text_col(df)
    label_col = find_label_col(df)
    print(f"         Text column:  '{text_col}'")
    print(f"         Label column: '{label_col}'")

    # Apply mapping
    df["category"] = df[label_col].apply(map_label)

    # Report any unmapped labels
    unmapped = df[df["category"] == "UNMAPPED"]
    if len(unmapped) > 0:
        print(f"\n⚠  {len(unmapped)} rows have UNMAPPED labels!")
        print("   Add these to LABEL_MAP in this script and re-run:")
        for lbl in unmapped[label_col].unique():
            print(f'   "{str(lbl).lower()}": "???",')
    else:
        print("\n✓ All labels mapped successfully")

    # Summary
    print("\n[step03] Category distribution after mapping:")
    dist = df["category"].value_counts()
    for cat in OUR_CATEGORIES:
        count = dist.get(cat, 0)
        pct   = count / len(df) * 100
        flag  = " ⚠ LOW" if count < 50 else ""
        print(f"  {cat:<22} {count:>5}  ({pct:5.1f}%){flag}")
    if "UNMAPPED" in dist:
        print(f"  {'UNMAPPED':<22} {dist['UNMAPPED']:>5}  ← FIX THESE")

    # Save clean version (text + mapped category only)
    out_df = pd.DataFrame({
        "text":     df[text_col].fillna(""),
        "category": df["category"],
        "source":   "arnaiz_rodriguez",
    })

    # Drop unmapped rows before saving
    clean_df = out_df[out_df["category"] != "UNMAPPED"].copy()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    clean_df.to_json(OUT_PATH, orient="records", lines=True, force_ascii=False)

    print(f"\n[step03] Saved {len(clean_df)} clean rows → {OUT_PATH}")
    print("         (Dropped unmapped rows. Fix LABEL_MAP to recover them.)")
    print("\n[step03] ✓ Next step: python scripts/task1_1/step04_download_hf_sources.py")


if __name__ == "__main__":
    main()
