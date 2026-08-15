"""
Task 1.1 — Step 07: Pilot test — run LLM judge on 100 Arnaiz examples
Owner: Both (Eman + Rania)

What this does:
  - Loads 100 random rows from arnaiz_mapped.jsonl (ground truth known)
  - Runs classify_with_judge() on each
  - Compares judge labels to ground truth
  - Prints accuracy, confusion, and any bad examples
  - Cost: ~$0.05

DECISION RULE:
  - Accuracy >= 75%  → prompt is good, proceed to step08 (full run)
  - Accuracy <  75%  → look at the disagreements, fix the prompt in
                       step06_judge_prompt.py, re-run this pilot

Run from repo root:
  python scripts/task1_1/step07_pilot_test.py
"""

import os
import sys
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# Add scripts dir to path so we can import step06
sys.path.insert(0, os.path.dirname(__file__))
from step06_judge_prompt import classify_with_judge, get_client, OUR_CATEGORIES

ARNAIZ_MAPPED = "data/raw/arnaiz_mapped.jsonl"
PILOT_OUT     = "data/raw/pilot_results.jsonl"
PILOT_N       = 100
RANDOM_STATE  = 42


def main():
    # ── Load Arnaiz ground truth ──────────────────────────────────────────────
    if not os.path.exists(ARNAIZ_MAPPED):
        print(f"ERROR: {ARNAIZ_MAPPED} not found.")
        print("Run step03_map_labels.py first.")
        return

    df = pd.read_json(ARNAIZ_MAPPED, lines=True)
    print(f"[step07] Loaded {len(df)} Arnaiz examples")

    # Stratified sample — get examples from multiple categories
    sample_parts = []
    per_cat = max(1, PILOT_N // len(OUR_CATEGORIES))
    for cat in OUR_CATEGORIES:
        cat_df = df[df["category"] == cat]
        if len(cat_df) > 0:
            n = min(per_cat, len(cat_df))
            sample_parts.append(cat_df.sample(n, random_state=RANDOM_STATE))
    pilot = pd.concat(sample_parts, ignore_index=True)
    if len(pilot) < PILOT_N:
        extra = df.sample(PILOT_N - len(pilot), random_state=RANDOM_STATE)
        pilot = pd.concat([pilot, extra], ignore_index=True)
    pilot = pilot.head(PILOT_N)

    print(f"[step07] Pilot sample: {len(pilot)} rows")
    print(f"         Ground truth distribution:")
    print(pilot["category"].value_counts().to_string())

    # ── Run judge ─────────────────────────────────────────────────────────────
    print(f"\n[step07] Running LLM judge on {len(pilot)} examples...")
    print("         Estimated cost: ~$0.05")
    client = get_client()

    results = []
    for _, row in tqdm(pilot.iterrows(), total=len(pilot)):
        judge_label = classify_with_judge(row["text"], client)
        results.append({
            "text":        row["text"][:200],
            "ground_truth": row["category"],
            "judge_label": judge_label,
            "match":       row["category"] == judge_label,
        })

    results_df = pd.DataFrame(results)
    results_df.to_json(PILOT_OUT, orient="records", lines=True, force_ascii=False)

    # ── Print report ──────────────────────────────────────────────────────────
    accuracy = results_df["match"].mean()
    invalids = (results_df["judge_label"] == "INVALID").sum()
    errors   = results_df["judge_label"].str.startswith("ERROR").sum()

    print(f"\n{'='*60}")
    print(f"PILOT RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:       {accuracy:.1%}  ({results_df['match'].sum()} / {len(results_df)} correct)")
    print(f"INVALID labels: {invalids}")
    print(f"API errors:     {errors}")

    print(f"\nClassification Report:")
    valid_results = results_df[~results_df["judge_label"].isin(["INVALID"]) &
                               ~results_df["judge_label"].str.startswith("ERROR")]
    if len(valid_results) > 0:
        print(classification_report(
            valid_results["ground_truth"],
            valid_results["judge_label"],
            labels=OUR_CATEGORIES,
            zero_division=0
        ))

    # Show disagreements so you can fix the prompt
    disagreements = results_df[~results_df["match"]].head(10)
    if len(disagreements) > 0:
        print(f"\nFirst {len(disagreements)} disagreements (to help you fix the prompt):")
        for _, row in disagreements.iterrows():
            print(f"\n  True:  {row['ground_truth']}")
            print(f"  Judge: {row['judge_label']}")
            print(f"  Text:  {row['text'][:150]!r}")

    print(f"\n{'='*60}")
    if accuracy >= 0.75:
        print(f"✓ ACCURACY {accuracy:.1%} >= 75% — prompt is good!")
        print("  Proceed to: python scripts/task1_1/step08_full_labeling.py")
    else:
        print(f"⚠  ACCURACY {accuracy:.1%} < 75% — prompt needs improvement.")
        print("   1. Study the disagreements above")
        print("   2. Edit JUDGE_SYSTEM_PROMPT in step06_judge_prompt.py")
        print("   3. Re-run this pilot")
    print(f"{'='*60}")

    print(f"\n[step07] Pilot results saved → {PILOT_OUT}")


if __name__ == "__main__":
    main()
