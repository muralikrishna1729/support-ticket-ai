"""
Taxonomy diagnosis: label-noise quantification + confusion analysis + merge test.

A. Label consistency - cluster near-duplicate tickets (TF-IDF word 1-2 grams,
   cosine similarity >= threshold, connected components), then measure the
   majority-label purity of each cluster. Quantifies how often the SAME
   ticket content carries DIFFERENT labels in the source data.

B. Confusion analysis - row-normalized confusion matrix of the current
   negation-tagged TF-IDF model on the test set. Shows which category
   boundaries the data does not actually support.

C. Merge test - merge the confused cluster into a single category and
   compare macro/weighted F1 three ways on the SAME test rows:
     (1) the production 10-category model
     (2) the same model with true+predicted labels merged post-hoc
     (3) a model retrained on the merged 7-category taxonomy

Outputs: printed report + taxonomy_results.json
Usage:   python diagnose_taxonomy.py [--sim 0.7] [--skip-retrain]
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.components.data_transformation import DataTransformation
from src.utils import load_object

DATA_PATH = os.path.join("artifacts", "data.csv")
TRAIN_PATH = os.path.join("artifacts", "train.csv")
TEST_PATH = os.path.join("artifacts", "test.csv")
RESULTS_PATH = "taxonomy_results.json"

# The four-way confusion cluster (diagnosed in step B; defined here so the
# merge test uses exactly what the data pointed at).
MERGE_TARGETS = {"Technical Support", "Product Support",
                 "IT Support", "Customer Service"}
MERGED_NAME = "Technical & Customer Support"


def _light_clean(text: str) -> str:
    text = str(text).lower()
    return re.sub(r"\s+", " ", text).strip()


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[ri] = rj



# ---------------------------------------------------------------------------
# A. Label-consistency quantification
# ---------------------------------------------------------------------------
def label_consistency(df: pd.DataFrame, sim_threshold: float):
    texts = df["text"].map(_light_clean).tolist()
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True)
    X = vec.fit_transform(texts)

    nn = NearestNeighbors(n_neighbors=11, metric="cosine", algorithm="brute")
    nn.fit(X)
    dist, idx = nn.kneighbors(X)
    sim = 1.0 - dist

    uf = _UnionFind(len(texts))
    pair_count = 0
    for i in range(len(texts)):
        for pos in range(1, 11):          # skip self-match at pos 0
            j = idx[i, pos]
            if sim[i, pos] >= sim_threshold:
                uf.union(i, j)
                pair_count += 1

    comps = {}
    for i in range(len(texts)):
        comps.setdefault(uf.find(i), []).append(i)
    groups = [members for members in comps.values() if len(members) >= 2]

    grouped_tickets = 0
    impure_tickets = 0            # tickets whose group has >1 distinct label
    heavily_impure_tickets = 0    # tickets in groups with purity < 0.75
    purity_sum = 0.0
    worst = []
    for members in groups:
        labels = [df["category"].iloc[k] for k in members]
        counts = Counter(labels)
        purity = max(counts.values()) / len(labels)
        purity_sum += purity
        grouped_tickets += len(members)
        if len(counts) > 1:
            impure_tickets += len(members)
            worst.append({"size": len(members),
                          "purity": round(purity, 3),
                          "labels": dict(counts)})
        if purity < 0.75:
            heavily_impure_tickets += len(members)

    worst.sort(key=lambda g: g["purity"])   # most impure first
    result = {
        "sim_threshold": sim_threshold,
        "near_dup_pairs": pair_count,
        "n_groups": len(groups),
        "grouped_tickets": grouped_tickets,
        "pct_tickets_in_groups": round(100 * grouped_tickets / len(df), 2),
        "pct_grouped_tickets_label_inconsistent":
            round(100 * impure_tickets / grouped_tickets, 2) if grouped_tickets else 0.0,
        "pct_grouped_tickets_heavily_inconsistent":
            round(100 * heavily_impure_tickets / grouped_tickets, 2) if grouped_tickets else 0.0,
        "mean_group_purity": round(purity_sum / len(groups), 4) if groups else None,
        "worst_groups": worst[:8],
    }
    return result


# ---------------------------------------------------------------------------
# B. Confusion analysis of the current production model
# ---------------------------------------------------------------------------
def confusion_analysis():
    dt = DataTransformation()
    tfidf = load_object("models/tfidf_vectorizer.pkl")
    clf = load_object("models/clf_category.pkl")
    le = load_object("models/le_category.pkl")

    test = pd.read_csv(TEST_PATH)
    cleaned = [dt.clean_text(t) for t in test["text"]]
    keep = [i for i, c in enumerate(cleaned) if len(c.split()) > 2]
    y_true = le.transform(test["category"].iloc[keep])
    X = tfidf.transform([cleaned[i] for i in keep])
    y_pred = clf.predict(X)
    classes = list(le.classes_)

    macro = round(float(f1_score(y_true, y_pred, average="macro")), 4)
    weighted = round(float(f1_score(y_true, y_pred, average="weighted")), 4)
    cm = confusion_matrix(y_true, y_pred)
    rownorm = cm / cm.sum(axis=1, keepdims=True)

    bleed = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and rownorm[i, j] >= 0.05:
                bleed.append({
                    "from": classes[i], "to": classes[j],
                    "pct_of_true_class": round(100 * rownorm[i, j], 1),
                    "count": int(cm[i, j]),
                })
    bleed.sort(key=lambda b: b["pct_of_true_class"], reverse=True)
    return {"classes": classes, "macro_f1": macro, "weighted_f1": weighted,
            "n_test_rows": len(keep), "confusion_bleed": bleed,
            "y_true": y_true, "y_pred": y_pred}

# ---------------------------------------------------------------------------
# C. Merge test: 10-category vs merged taxonomy, three ways
# ---------------------------------------------------------------------------
def _merge_label(label):
    return MERGED_NAME if label in MERGE_TARGETS else label


def merge_experiment(conf: dict, skip_retrain: bool):
    classes = conf["classes"]
    mapping = {c: _merge_label(c) for c in classes}

    # (2) Post-hoc merge of the SAME 10-category model's predictions
    # (y_true/y_pred are label-encoded ints; decode to names first)
    y_true7 = [_merge_label(classes[i]) for i in conf["y_true"]]
    y_pred7 = [_merge_label(classes[i]) for i in conf["y_pred"]]
    macro_posthoc = round(float(f1_score(y_true7, y_pred7, average="macro")), 4)
    weighted_posthoc = round(float(f1_score(y_true7, y_pred7, average="weighted")), 4)

    # (3) Retrain on the merged taxonomy with the exact production recipe
    macro_retrained = weighted_retrained = None
    bucket_sizes = None
    best_C = None
    if not skip_retrain:
        dt = DataTransformation()
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
        train["cat_merged"] = train["category"].map(_merge_label)
        test["cat_merged"] = test["category"].map(_merge_label)
        bucket_sizes = {
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "merged_bucket_total": int((train["cat_merged"] == MERGED_NAME).sum()
                                       + (test["cat_merged"] == MERGED_NAME).sum()),
            "pct_merged_of_total": round(
                100 * ((train["cat_merged"] == MERGED_NAME).sum()
                       + (test["cat_merged"] == MERGED_NAME).sum())
                / (len(train) + len(test)), 1),
        }

        tr_clean = [dt.clean_text(t) for t in train["text"]]
        te_clean = [dt.clean_text(t) for t in test["text"]]
        tr_keep = [i for i, c in enumerate(tr_clean) if len(c.split()) > 2]
        te_keep = [i for i, c in enumerate(te_clean) if len(c.split()) > 2]

        le = LabelEncoder()
        y_tr = le.fit_transform(train["cat_merged"].iloc[tr_keep])
        y_te = le.transform(test["cat_merged"].iloc[te_keep])

        cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cwd = dict(zip(np.unique(y_tr), cw))

        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 3),
                              min_df=2, max_df=0.85, sublinear_tf=True)
        X_tr = vec.fit_transform([tr_clean[i] for i in tr_keep])
        X_te = vec.transform([te_clean[i] for i in te_keep])

        grid = GridSearchCV(
            LinearSVC(class_weight=cwd, max_iter=2000, random_state=42),
            {"C": [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]},
            cv=5, scoring="f1_weighted", n_jobs=-1, verbose=0)
        grid.fit(X_tr, y_tr)
        y_pred = grid.predict(X_te)
        macro_retrained = round(float(f1_score(y_te, y_pred, average="macro")), 4)
        weighted_retrained = round(float(f1_score(y_te, y_pred, average="weighted")), 4)
        best_C = grid.best_params_["C"]

    return {
        "n_categories_after_merge": 7,
        "merged_category": MERGED_NAME,
        "merged_from": sorted(MERGE_TARGETS),
        "ten_cat_macro_f1": conf["macro_f1"],
        "ten_cat_weighted_f1": conf["weighted_f1"],
        "posthoc_merged_macro_f1": macro_posthoc,
        "posthoc_merged_weighted_f1": weighted_posthoc,
        "retrained_macro_f1": macro_retrained,
        "retrained_weighted_f1": weighted_retrained,
        "retrained_best_C": best_C,
        "bucket_sizes": bucket_sizes,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=float, default=0.7,
                    help="cosine similarity threshold for near-duplicates")
    ap.add_argument("--skip-retrain", action="store_true")
    args = ap.parse_args()

    # Accumulate results across runs instead of overwriting: label
    # consistency is keyed by threshold, and a --skip-retrain run keeps
    # previously trained merge-test numbers.
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            results = json.load(f)
    else:
        results = {}
    results.pop("sim_threshold", None)        # ambiguous after multiple runs
    results.pop("label_consistency", None)    # legacy pre-threshold key
    results.setdefault("label_consistency_by_threshold", {})

    print("=== A. LABEL CONSISTENCY (near-duplicate clustering) ===")
    df = pd.read_csv(DATA_PATH)
    print(f"Corpus: {len(df):,} labeled rows from {DATA_PATH}")
    t0 = time.time()
    a = label_consistency(df, args.sim)
    results["label_consistency_by_threshold"][f"sim_{args.sim}"] = a
    print(f"  near-dup pairs (sim>={args.sim}) : {a['near_dup_pairs']:,}")
    print(f"  multi-ticket groups              : {a['n_groups']:,} "
          f"covering {a['grouped_tickets']:,} tickets "
          f"({a['pct_tickets_in_groups']}% of corpus)")
    print(f"  mean group purity                : {a['mean_group_purity']}")
    print(f"  >> grouped tickets with >1 label : {a['pct_grouped_tickets_label_inconsistent']}%")
    print(f"  >> tickets in groups purity<0.75 : {a['pct_grouped_tickets_heavily_inconsistent']}%")
    print("  worst groups (same content, mixed labels):")
    for g in a["worst_groups"]:
        print(f"    size={g['size']:>3} purity={g['purity']:<6} {g['labels']}")
    print(f"  ({time.time() - t0:.0f}s)")

    print("\n=== B. CONFUSION ANALYSIS (current tfidf_neg model) ===")
    t0 = time.time()
    conf = confusion_analysis()
    results["confusion_analysis"] = {
        k: v for k, v in conf.items() if k not in ("y_true", "y_pred")}
    print(f"  test rows: {conf['n_test_rows']}")
    print(f"  10-category macro F1    : {conf['macro_f1']}")
    print(f"  10-category weighted F1 : {conf['weighted_f1']}")
    print("  category bleed (>=5% of a true class misrouted):")
    for b in conf["confusion_bleed"]:
        print(f"    {b['from']:<33} -> {b['to']:<33} {b['pct_of_true_class']:>5}%  ({b['count']} tickets)")
    print(f"  ({time.time() - t0:.0f}s)")

    print("\n=== C. MERGE TEST (merge the confused cluster -> 1 category) ===")
    t0 = time.time()
    c = merge_experiment(conf, args.skip_retrain)
    prev = results.get("merge_test") or {}
    if args.skip_retrain and prev.get("retrained_macro_f1") is not None:
        # preserve the earlier full run's trained numbers
        c.update({k: prev[k] for k in ("retrained_macro_f1",
                                       "retrained_weighted_f1",
                                       "retrained_best_C",
                                       "bucket_sizes")})
    results["merge_test"] = c
    print(f"  merge: {sorted(MERGE_TARGETS)} -> '{MERGED_NAME}'")
    if c["bucket_sizes"]:
        bs = c["bucket_sizes"]
        print(f"  merged bucket: {bs['merged_bucket_total']:,} rows "
              f"({bs['pct_merged_of_total']}% of corpus)")
    print(f"  (1) 10-category model            : macro {c['ten_cat_macro_f1']} | weighted {c['ten_cat_weighted_f1']}")
    print(f"  (2) same model, merged post-hoc  : macro {c['posthoc_merged_macro_f1']} | weighted {c['posthoc_merged_weighted_f1']}")
    if c["retrained_macro_f1"] is not None:
        print(f"  (3) retrained on 7 categories    : macro {c['retrained_macro_f1']} | "
              f"weighted {c['retrained_weighted_f1']} | best C {c['retrained_best_C']}")
    print(f"  ({time.time() - t0:.0f}s)")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=list)
    print(f"\nResults saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()

