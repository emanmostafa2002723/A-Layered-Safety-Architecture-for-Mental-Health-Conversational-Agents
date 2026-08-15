"""
Task 1.1 — Step 02: Inspect the Arnaiz-Rodriguez dataset
Owner: Eman (P1)

What this does:
  - Finds the CSV/JSON files inside the cloned repo
  - Prints columns, shape, label distribution, and 3 example rows
  - Saves a summary to data/raw/arnaiz_inspection.txt

Run from repo root:
  python scripts/task1_1/step02_inspect_arnaiz.py
"""

import os
import glob
import json
import pandas as pd

RAW_DIR  = "data/raw/arnaiz-rodriguez"
OUT_FILE = "data/raw/arnaiz_inspection.txt"

lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))


def find_data_files(root: str) -> list[str]:
    """Recursively find all CSV and JSON files."""
    csvs  = glob.glob(os.path.join(root, "**", "*.csv"),  recursive=True)
    jsons = glob.glob(os.path.join(root, "**", "*.json"), recursive=True)
    jsonls = glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)
    return sorted(csvs + jsons + jsonls)


def inspect_file(path: str) -> pd.DataFrame | None:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            df = pd.read_json(path)
        elif ext == ".jsonl":
            df = pd.read_json(path, lines=True)
        else:
            return None
        return df
    except Exception as e:
        log(f"  Could not load {path}: {e}")
        return None


def main():
    if not os.path.isdir(RAW_DIR):
        print(f"ERROR: {RAW_DIR} does not exist.")
        print("Run step01 first: bash scripts/task1_1/step01_download_arnaiz.sh")
        return

    log("=" * 70)
    log("ARNAIZ-RODRIGUEZ DATASET INSPECTION")
    log("=" * 70)

    files = find_data_files(RAW_DIR)
    if not files:
        log(f"No CSV/JSON/JSONL files found under {RAW_DIR}")
        log("The repo may use a different format. Listing all files:")
        for root, dirs, fnames in os.walk(RAW_DIR):
            for fname in fnames:
                log(f"  {os.path.join(root, fname)}")
        return

    log(f"\nFound {len(files)} data file(s):\n")
    for f in files:
        log(f"  {f}")

    log("")
    for path in files:
        log("-" * 70)
        log(f"FILE: {path}")
        log("-" * 70)

        df = inspect_file(path)
        if df is None:
            continue

        log(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        log(f"Columns: {df.columns.tolist()}")
        log("")

        # Find the label / category column
        label_col = None
        for candidate in ["label", "category", "crisis_type", "class",
                          "crisis_category", "type", "split"]:
            if candidate in df.columns:
                label_col = candidate
                break

        if label_col:
            log(f"Label column: '{label_col}'")
            log("\nLabel distribution:")
            log(df[label_col].value_counts().to_string())
        else:
            log("⚠  Could not find a label column automatically.")
            log("   Inspect manually and update LABEL_MAP in step03_map_labels.py")

        log("\n3 Example rows:")
        log(df.head(3).to_string())
        log("")

        # Save all unique label values for the mapping step
        if label_col:
            unique_labels = df[label_col].dropna().unique().tolist()
            labels_path = "data/raw/arnaiz_unique_labels.json"
            with open(labels_path, "w") as f_out:
                json.dump({
                    "source_file": path,
                    "label_column": label_col,
                    "unique_labels": unique_labels
                }, f_out, indent=2)
            log(f"→ Unique labels saved to {labels_path}")
            log(f"  Labels: {unique_labels}")

    log("")
    log("=" * 70)
    log("NEXT STEP: python scripts/task1_1/step03_map_labels.py")
    log("           (after reviewing the label distribution above)")
    log("=" * 70)

    # Save the full report
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[step02] Full report saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
