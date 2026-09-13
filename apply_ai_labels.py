"""Apply AI-assisted labels (artifacts/ai_labels.json) onto the validation
subset's `true_category` column, with strict validation before writing.

Provenance: these labels were produced by an AI assistant reading each
ticket blind (no dataset label / model prediction in view). They are NOT
human-verified — review artifacts/validation_disagreements_for_human_review.csv
to convert them into human-verified labels."""

import json
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SUBSET = os.path.join("artifacts", "validation_subset.csv")
LABELS = os.path.join("artifacts", "ai_labels.json")

VALID = {
    "Billing and Payments", "Customer Service", "General Inquiry",
    "Human Resources", "IT Support", "Product Support",
    "Returns and Exchanges", "Sales and Pre-Sales",
    "Service Outages and Maintenance", "Technical Support",
}


def main():
    with open(LABELS) as f:
        data = json.load(f)
    labels, notes = data["labels"], data.get("notes", {})

    df = pd.read_csv(SUBSET)
    assert set(labels) == {str(i) for i in range(len(df))}, \
        "label keys must cover every row exactly once"
    assert set(labels.values()) <= VALID, "unknown category in labels"

    df["true_category"] = df["true_category"].fillna("").astype("object")
    df["notes"] = df["notes"].fillna("").astype("object")  # pandas 3.0 float64 trap
    for i, lab in labels.items():
        df.at[int(i), "true_category"] = lab
    for i, note in notes.items():
        df.at[int(i), "notes"] = note
    df.to_csv(SUBSET, index=False)

    match = (df["true_category"] == df["category"]).mean()
    print(f"Applied {len(labels)} labels (+{len(notes)} notes) -> {SUBSET}")
    print(f"AI-label agreement with dataset labels: {match * 100:.1f}%")


if __name__ == "__main__":
    main()
