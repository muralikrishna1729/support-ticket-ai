"""
Split hand-labeled validation results into model-error vs label-error.

Reads artifacts/validation_subset.csv AFTER labeling (label_validation_subset.py),
predicts with the current production model, and buckets every labeled row:

  D = dataset label, M = model prediction, H = hand label
  all_agree                        D == M == H
  label_error_model_right          M == H != D    dataset label wrong, model right
  model_error                      D == H != M    model got it wrong
  label_error_model_learned_noise  D == M != H    dataset label wrong, model copied it
  all_disagree                     none of the above

Reports the model's "true ceiling" F1 against HAND labels, the dataset-vs-hand
agreement (label-noise estimate on held-out data), and where disagreements
concentrate by category.

Outputs: artifacts/validation_review.csv (per-row) + validation_review.json
Usage:   python validation_breakdown.py [--selftest]
         (--selftest simulates 30 hand labels in memory to smoke-test the
          pipeline; nothing is written and the CSV is never modified)
"""

import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.components.data_transformation import DataTransformation
from src.utils import load_object

SUBSET_PATH = os.path.join("artifacts", "validation_subset.csv")
REVIEW_CSV = os.path.join("artifacts", "validation_review.csv")
REVIEW_JSON = "validation_review.json"


def model_predictions(texts):
    """Predict categories exactly the way the production pipeline does."""
    dt = DataTransformation()
    tfidf = load_object("models/tfidf_vectorizer.pkl")
    clf = load_object("models/clf_category.pkl")
    le = load_object("models/le_category.pkl")
    cleaned = [dt.clean_text(t) for t in texts]
    X = tfidf.transform(cleaned)
    return list(le.inverse_transform(clf.predict(X)))


def bucket(d, m, h):
    if d == m == h:
        return "all_agree"
    if m == h:
        return "label_error_model_right"
    if d == h:
        return "model_error"
    if d == m:
        return "label_error_model_learned_noise"
    return "all_disagree"


def analyze(df):
    preds = model_predictions(df["text"].tolist())
    review = pd.DataFrame({
        "dataset_label":    df["category"].values,
        "model_prediction": preds,
        "hand_label":       df["true_category"].values,
        "text":             df["text"].values,
    })
    review["bucket"] = [
        bucket(d, m, h)
        for d, m, h in zip(review["dataset_label"],
                           review["model_prediction"],
                           review["hand_label"])
    ]

    n = len(review)
    counts = review["bucket"].value_counts().to_dict()
    macro = round(float(f1_score(review["hand_label"], review["model_prediction"],
                                 average="macro")), 4)
    weighted = round(float(f1_score(review["hand_label"], review["model_prediction"],
                                    average="weighted")), 4)
    disagreements = review[review["bucket"] != "all_agree"]
    summary = {
        "n_labeled": n,
        "model_vs_hand_macro_f1": macro,
        "model_vs_hand_weighted_f1": weighted,
        "dataset_vs_hand_agreement_pct":
            round(float((review["dataset_label"] == review["hand_label"]).mean() * 100), 1),
        "buckets": {k: {"count": int(v), "pct": round(100 * v / n, 1)}
                    for k, v in counts.items()},
        "disagreements_by_dataset_category":
            {k: int(v) for k, v in
             disagreements["dataset_label"].value_counts().items()},
    }
    return review, summary



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="simulate 30 hand labels (all = dataset label) to "
                         "smoke-test the pipeline; nothing is written")
    args = ap.parse_args()

    df = pd.read_csv(SUBSET_PATH)
    # pandas 3.0: all-blank column loads as float64; coerce for safe compares
    df["true_category"] = df["true_category"].fillna("").astype("object")
    labeled = df[df["true_category"].notna()
                 & (df["true_category"].astype(str).str.strip() != "")]

    if args.selftest:
        df = df.copy()
        df.loc[df.index[:30], "true_category"] = df["category"].head(30).values
        labeled = df[df["true_category"].notna()
                     & (df["true_category"].astype(str).str.strip() != "")]
        print("[selftest] simulating 30 hand labels (hand = dataset label)\n")

    print(f"Labeled rows: {len(labeled)} / {len(df)} "
          f"({len(df) - len(labeled)} still unlabeled)")
    if len(labeled) == 0:
        print("\nNo hand labels yet — run:  python label_validation_subset.py")
        return

    review, summary = analyze(labeled)

    print("\n=== VALIDATION BREAKDOWN (model vs hand labels) ===")
    print(f"model vs hand  : macro F1 {summary['model_vs_hand_macro_f1']} | "
          f"weighted F1 {summary['model_vs_hand_weighted_f1']}   <- 'true ceiling'")
    print(f"dataset vs hand agreement: {summary['dataset_vs_hand_agreement_pct']}%  "
          f"(label-noise estimate on held-out data)")
    print("buckets:")
    for name in ("all_agree", "model_error", "label_error_model_right",
                 "label_error_model_learned_noise", "all_disagree"):
        b = summary["buckets"].get(name, {"count": 0, "pct": 0.0})
        print(f"  {name:<32} {b['count']:>4}  ({b['pct']}%)")
    if summary["disagreements_by_dataset_category"]:
        print("disagreements by dataset category:")
        for cat, cnt in summary["disagreements_by_dataset_category"].items():
            print(f"  {cat:<35} {cnt}")

    if args.selftest:
        print("\n[selftest] OK — pipeline verified, nothing written.")
        return

    review.to_csv(REVIEW_CSV, index=False)
    with open(REVIEW_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {REVIEW_CSV} , {REVIEW_JSON}")


if __name__ == "__main__":
    main()

