"""
Benchmark feature-extraction techniques for ticket classification.

All techniques train on the SAME train/test split (artifacts/train.csv,
artifacts/test.csv - produced by src/components/data_ingestion.py with
random_state=42, stratified by category) and use the SAME classifiers as
src/components/model_trainer.py:
  - category   : LinearSVC + GridSearchCV(C=[0.05..5.0], cv=5, f1_weighted)
  - issue_type : LinearSVC(C=1.0, class_weight='balanced')

Techniques:
  tfidf        TF-IDF(1-3, 20k, sublinear) on aggressively cleaned text     (v1 recipe)
  embed        all-MiniLM-L6-v2 sentence embeddings on lightly cleaned text (v2 recipe)
  concat       SVD-150-compressed TF-IDF (+) MiniLM embeddings              (v3 recipe)
  tfidf_neg    TF-IDF with negation tagging ("cannot access" -> "not NEG_access")
  concat_neg   negation-tagged TF-IDF (+) MiniLM embeddings

Usage:
  python benchmark_techniques.py --technique tfidf
  python benchmark_techniques.py --technique all

Results accumulate in experiment_results.json; a comparison table is printed
after every run. MiniLM embeddings are cached in artifacts/cache/ (content
hashed) so they are only computed once across runs.
"""

import argparse
import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

TRAIN_PATH = os.path.join("artifacts", "train.csv")
TEST_PATH = os.path.join("artifacts", "test.csv")
CACHE_DIR = os.path.join("artifacts", "cache")
RESULTS_PATH = "experiment_results.json"
EMBED_MODEL = "all-MiniLM-L6-v2"

TECHNIQUES = ["tfidf", "embed", "concat", "tfidf_neg", "concat_neg"]

# ---------------------------------------------------------------------------
# Cleaning - replicates src/components/data_transformation.py
# ---------------------------------------------------------------------------
CUSTOM_SW = {
    'data', 'support', 'issue', 'issues', 'information',
    'provide', 'request', 'assistance', 'customer',
    'appreciate', 'regards', 'thanks', 'thank', 'dear',
    'team', 'help', 'looking', 'forward', 'greatly',
    'soon', 'problem', 'great', 'problems'
}
STOP_WORDS = ENGLISH_STOP_WORDS.union(CUSTOM_SW)

NEG_CUES = {
    "not", "no", "nor", "never", "neither", "nothing", "nobody", "none",
    "cannot", "unable", "without", "lack", "lacks", "lacking", "missing",
    "fail", "fails", "failed", "failing", "deny", "denied",
    "refuse", "refused", "broken", "outage",
}
NEG_WINDOW = 3


def light_clean(text: str) -> str:
    """Minimal cleaning - same as DataTransformation.light_clean (embed branch).
    Preserves negation words on purpose."""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    return text


def _expand_contractions(text: str) -> str:
    """Turn n't contractions into standalone 'not' BEFORE punctuation
    stripping, so the negation cue survives the aggressive cleaner."""
    text = re.sub(r"n't\b", " not", text)          # don't / doesn't / isn't ...
    text = re.sub(r"\bcannot\b", "can not", text)  # cannot -> can not
    return text


def clean_tfidf(text: str, negation_aware: bool = False) -> str:
    """Aggressive cleaning - same as DataTransformation.clean_text.
    With negation_aware=True, cue words are kept and the next NEG_WINDOW
    content words get a NEG_ prefix so TF-IDF can separate polarity
    ("can access" vs "not NEG_access")."""
    text = str(text).lower()
    if negation_aware:
        text = _expand_contractions(text)
    text = re.sub(r"[^a-z]", " ", text)
    out, neg_left = [], 0
    for w in text.split():
        if negation_aware and w in NEG_CUES:
            out.append(w)
            neg_left = NEG_WINDOW
            continue
        if w in STOP_WORDS or len(w) <= 2:
            continue
        if negation_aware and neg_left > 0:
            out.append("NEG_" + w)
            neg_left -= 1
        else:
            out.append(w)
    return " ".join(out)


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def text_digest(texts) -> str:
    return hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()


def get_embeddings(texts, split: str) -> np.ndarray:
    os.makedirs(CACHE_DIR, exist_ok=True)
    vec_path = os.path.join(CACHE_DIR, f"embed_{split}.npy")
    key_path = os.path.join(CACHE_DIR, f"embed_{split}.sha256")
    digest = text_digest(texts)
    if os.path.exists(vec_path) and os.path.exists(key_path):
        with open(key_path) as f:
            if f.read() == digest:
                print(f"  [cache] {split} embeddings loaded from {vec_path}")
                return np.load(vec_path)
    from sentence_transformers import SentenceTransformer

    print(f"  [encode] encoding {len(texts):,} {split} texts with {EMBED_MODEL} "
          f"(cached after this run)...")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, batch_size=64, show_progress_bar=False,
                       convert_to_numpy=True)
    np.save(vec_path, emb)
    with open(key_path, "w") as f:
        f.write(digest)
    print(f"  [encode] done in {time.time() - t0:.0f}s -> shape {emb.shape}")
    return emb


def build_tfidf(train_texts, test_texts):
    """Same vectorizer params as the v1 pipeline."""
    tfidf = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 3),
        min_df=2, max_df=0.85, sublinear_tf=True,
    )
    Xtr = tfidf.fit_transform(train_texts)
    Xte = tfidf.transform(test_texts)
    return tfidf, Xtr, Xte


def build_svd(Xtr_sp, Xte_sp, n_components=150):
    """Compress sparse TF-IDF to a dense n_components representation
    for concatenation with embeddings."""
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    Xtr = svd.fit_transform(Xtr_sp)
    Xte = svd.transform(Xte_sp)
    return svd, Xtr, Xte


# ---------------------------------------------------------------------------
# Shared training / evaluation (mirrors model_trainer.py)
# ---------------------------------------------------------------------------
def train_and_eval(tag, X_train, X_test, y_train_cat, y_test_cat,
                   y_train_type, y_test_type, class_weight_dict):
    t0 = time.time()

    cat_pipe = Pipeline([
        ("clf", LinearSVC(class_weight=class_weight_dict,
                          max_iter=2000, random_state=42)),
    ])
    grid = GridSearchCV(
        cat_pipe,
        {"clf__C": [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]},
        cv=5, scoring="f1_weighted", n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train_cat)
    best_cat = grid.best_estimator_

    issue_clf = LinearSVC(class_weight="balanced", C=1.0,
                          max_iter=2000, random_state=42)
    issue_clf.fit(X_train, y_train_type)

    cat_pred = best_cat.predict(X_test)
    type_pred = issue_clf.predict(X_test)
    f1_cat = round(float(f1_score(y_test_cat, cat_pred, average="weighted")), 4)
    f1_type = round(float(f1_score(y_test_type, type_pred, average="weighted")), 4)

    rep = classification_report(y_test_cat, cat_pred, output_dict=True,
                                zero_division=0)
    per_class = {
        k: round(v["f1-score"], 3)
        for k, v in rep.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }

    result = {
        "technique": tag,
        "f1_category": f1_cat,
        "f1_issue_type": f1_type,
        "best_C": grid.best_params_["clf__C"],
        "train_seconds": round(time.time() - t0, 1),
        "per_class_category_f1": per_class,
    }
    print(f"  => F1 category: {f1_cat} | issue_type: {f1_type} "
          f"| best C: {result['best_C']} | {result['train_seconds']}s")
    return result


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------
def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def reference_rows():
    """Previously recorded runs, shown for context in the table."""
    refs = []
    p = os.path.join("models_v1_tfidf", "model_scores_v1.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        refs.append({"technique": "(recorded) tfidf v1 - older 16k dataset",
                     "f1_category": d["f1_category"],
                     "f1_issue_type": d["f1_issue_type"],
                     "best_C": d.get("best_C"), "train_seconds": None})
    p = os.path.join("models", "model_scores.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        refs.append({"technique": "(recorded) embed v2 run",
                     "f1_category": d["f1_category"],
                     "f1_issue_type": d["f1_issue_type"],
                     "best_C": d.get("best_C"), "train_seconds": None})
    return refs


def print_table(all_results):
    rows = [r for r in all_results if r.get("f1_category") is not None]
    rows.sort(key=lambda r: r["f1_category"], reverse=True)
    print("\n=== COMPARISON (sorted by category F1) ===")
    print(f"{'technique':<42} {'f1_cat':>8} {'f1_type':>8} {'best_C':>8} {'time':>7}")
    print("-" * 78)
    for r in rows:
        secs = f"{r['train_seconds']:.0f}s" if r.get("train_seconds") else "-"
        print(f"{r['technique']:<42} {r['f1_category']:>8} "
              f"{r['f1_issue_type']:>8} {str(r.get('best_C', '-')):>8} {secs:>7}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--technique", default="all",
                    choices=TECHNIQUES + ["all"])
    args = ap.parse_args()
    todo = TECHNIQUES if args.technique == "all" else [args.technique]

    print("=== Loading data ===")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # ONE shared row filter so every technique sees identical rows/labels.
    train_df["clean_light"] = train_df["text"].apply(light_clean)
    test_df["clean_light"] = test_df["text"].apply(light_clean)
    train_df = train_df[train_df["clean_light"].str.split().str.len() > 2].reset_index(drop=True)
    test_df = test_df[test_df["clean_light"].str.split().str.len() > 2].reset_index(drop=True)
    print(f"Rows: train={len(train_df):,} test={len(test_df):,}")

    le_cat, le_type = LabelEncoder(), LabelEncoder()
    y_train_cat = le_cat.fit_transform(train_df["category"])
    y_test_cat = le_cat.transform(test_df["category"])
    y_train_type = le_type.fit_transform(train_df["issue_type"])
    y_test_type = le_type.transform(test_df["issue_type"])

    classes = np.unique(y_train_cat)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_cat)
    cwd = dict(zip(classes, weights))

    tfidf_cache = {}   # key: 'plain' | 'neg' -> (vectorizer, Xtr_sp, Xte_sp)
    emb_cache = {}     # key: split -> ndarray

    def get_tfidf_feats(negation: bool):
        key = "neg" if negation else "plain"
        if key not in tfidf_cache:
            label = "negation-tagged" if negation else "plain"
            print(f"  [features] building TF-IDF ({label})...")
            tr_texts = train_df["text"].apply(
                lambda t: clean_tfidf(t, negation)).tolist()
            te_texts = test_df["text"].apply(
                lambda t: clean_tfidf(t, negation)).tolist()
            tfidf_cache[key] = build_tfidf(tr_texts, te_texts)
            print(f"  [features] TF-IDF shape: {tfidf_cache[key][1].shape}")
        return tfidf_cache[key]

    def get_embed_feats():
        if "train" not in emb_cache:
            emb_cache["train"] = get_embeddings(
                train_df["clean_light"].tolist(), "train")
            emb_cache["test"] = get_embeddings(
                test_df["clean_light"].tolist(), "test")
        return emb_cache["train"], emb_cache["test"]

    results = load_results()
    for tech in todo:
        print(f"\n=== TECHNIQUE: {tech} ===")
        if tech == "tfidf":
            _, Xtr, Xte = get_tfidf_feats(False)
            res = train_and_eval(tech, Xtr, Xte, y_train_cat, y_test_cat,
                                 y_train_type, y_test_type, cwd)
        elif tech == "embed":
            Xtr, Xte = get_embed_feats()
            res = train_and_eval(tech, Xtr, Xte, y_train_cat, y_test_cat,
                                 y_train_type, y_test_type, cwd)
        elif tech == "concat":
            _, Xtr_sp, Xte_sp = get_tfidf_feats(False)
            _, Xtr_svd, Xte_svd = build_svd(Xtr_sp, Xte_sp)
            Xtr_e, Xte_e = get_embed_feats()
            Xtr, Xte = np.hstack([Xtr_svd, Xtr_e]), np.hstack([Xte_svd, Xte_e])
            print(f"  [features] concat shape: {Xtr.shape}")
            res = train_and_eval(tech, Xtr, Xte, y_train_cat, y_test_cat,
                                 y_train_type, y_test_type, cwd)
        elif tech == "tfidf_neg":
            _, Xtr, Xte = get_tfidf_feats(True)
            res = train_and_eval(tech, Xtr, Xte, y_train_cat, y_test_cat,
                                 y_train_type, y_test_type, cwd)
        elif tech == "concat_neg":
            _, Xtr_sp, Xte_sp = get_tfidf_feats(True)
            _, Xtr_svd, Xte_svd = build_svd(Xtr_sp, Xte_sp)
            Xtr_e, Xte_e = get_embed_feats()
            Xtr, Xte = np.hstack([Xtr_svd, Xtr_e]), np.hstack([Xte_svd, Xte_e])
            print(f"  [features] concat_neg shape: {Xtr.shape}")
            res = train_and_eval(tech, Xtr, Xte, y_train_cat, y_test_cat,
                                 y_train_type, y_test_type, cwd)

        results = [r for r in results if r["technique"] != tech]
        results.append(res)
        save_results(results)

    print_table(reference_rows() + results)


if __name__ == "__main__":
    main()

