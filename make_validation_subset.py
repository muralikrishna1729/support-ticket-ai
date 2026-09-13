"""
Build a stratified hand-labeling subset from the test set.

Samples ~250 rows (proportional to category size, seed 42) from
artifacts/test.csv and writes artifacts/validation_subset.csv with empty
`true_category` / `true_issue_type` / `notes` columns for manual review.

Purpose: score the model on human-verified labels to separate
  - model error  (model wrong, label right)
  - label error  (label wrong / ambiguous, model reasonable)
from the noisy full-test-set number. Label ~250 rows (~30-45 min of work),
then evaluate with:
  python -m sklearn  ... or a small script computing
  f1_score(df.true_category, df.predicted_category, average="macro")

Usage: python make_validation_subset.py [--size 250]
"""

import argparse
import os

import pandas as pd

TEST_PATH = os.path.join("artifacts", "test.csv")
OUT_PATH = os.path.join("artifacts", "validation_subset.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=250)
    args = ap.parse_args()

    test = pd.read_csv(TEST_PATH)
    frac = min(1.0, args.size / len(test))
    sub = (test.groupby("category", group_keys=False)
               .sample(frac=frac, random_state=42)
               .sort_values("category")
               .reset_index(drop=True))

    sub["true_category"] = ""
    sub["true_issue_type"] = ""
    sub["notes"] = ""
    sub = sub[["category", "issue_type", "true_category",
               "true_issue_type", "notes", "text"]]
    sub.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(sub)} rows -> {OUT_PATH}")
    print("\nPer-category sample counts:")
    print(sub["category"].value_counts().to_string())
    print("""
How to use:
  1. Open the CSV and fill `true_category` (+ optionally `true_issue_type`)
     ONLY where you are confident; leave blank to mark 'unsure'.
  2. When done, compare against `category` (the dataset label) and the
     model prediction to split errors into model error vs label error.
  3. The share of rows where the DATASET label itself looks wrong is your
     label-noise ceiling estimate on held-out data.
""")


if __name__ == "__main__":
    main()
