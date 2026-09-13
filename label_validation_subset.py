"""
Quick CLI labeling tool for artifacts/validation_subset.csv

Usage:
    python label_validation_subset.py

- Shows one ticket at a time with its current (model-assigned) category.
- Press the number for the correct category, or Enter to accept the shown one.
- Progress is saved after every single row, so you can Ctrl+C and resume anytime.
- Produces a `true_category` column used later to split model-error vs label-error.
"""

import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 safety

INPUT_PATH = "artifacts/validation_subset.csv"

CATEGORIES = [
    "Billing and Payments",
    "Customer Service",
    "General Inquiry",
    "Human Resources",
    "IT Support",
    "Product Support",
    "Returns and Exchanges",
    "Sales and Pre-Sales",
    "Service Outages and Maintenance",
    "Technical Support",
]


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Can't find {INPUT_PATH} — run this from the project root.")
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)

    if "true_category" not in df.columns:
        df["true_category"] = ""

    # pandas 3.0: an all-blank column loads as float64 and rejects string
    # assignment (TypeError on first label). Coerce to object dtype first.
    df["true_category"] = df["true_category"].fillna("").astype("object")

    text_col = "text" if "text" in df.columns else df.columns[0]
    pred_col = "category" if "category" in df.columns else None

    remaining = df[df["true_category"].isna() | (df["true_category"] == "")]
    total = len(df)
    done_count = total - len(remaining)

    print(f"\n{done_count}/{total} already labeled. {len(remaining)} left.\n")
    print("Categories:")
    for i, cat in enumerate(CATEGORIES, 1):
        print(f"  {i}. {cat}")
    print("\nFor each ticket: type a number (1-10), or press Enter to ACCEPT the shown label, or 'q' to quit.\n")

    for idx in remaining.index:
        row = df.loc[idx]
        shown_label = row[pred_col] if pred_col else "(unknown)"

        print("-" * 70)
        print(f"[{done_count + 1}/{total}]")
        print(f"TEXT: {str(row[text_col])[:500]}")
        print(f"Model/current label: {shown_label}")
        ans = input("Your answer (number / Enter=accept / q=quit): ").strip()

        if ans.lower() == "q":
            print("\nSaving and exiting.")
            break

        if ans == "":
            df.at[idx, "true_category"] = shown_label
        elif ans.isdigit() and 1 <= int(ans) <= len(CATEGORIES):
            df.at[idx, "true_category"] = CATEGORIES[int(ans) - 1]
        else:
            print("Didn't understand that — skipping, you can redo it next run.")
            continue

        done_count += 1
        df.to_csv(INPUT_PATH, index=False)  # save after every row

    print(f"\nDone for now: {done_count}/{total} labeled. Progress saved to {INPUT_PATH}.")


if __name__ == "__main__":
    main()
