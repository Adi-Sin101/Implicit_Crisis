"""
TF-IDF + Logistic Regression baseline for 3-class crisis severity classification.

This baseline is built to be directly comparable with the existing BERT
experiment in `ipnyb/bert_crisis_classifier (1).ipynb`. It therefore reuses,
without deviation:

    dataset : data/final_datasets/merged_severity_dataset_not_keyword_keyword.csv
    input   : the `content` column only
    target  : `severity`, collapsed 0 -> 0, 1 -> 1, 2..6 -> 2
    split   : stratified 70 / 15 / 15, random_state = 42
    metric  : macro F1 for model selection

Leakage control: the TF-IDF vectorizer is fitted on the TRAINING texts only.
Validation and test texts are only ever transformed with that fitted
vectorizer. Model selection uses validation macro F1; the test set is touched
exactly once, after the configuration is frozen.

Run:
    python experiments/tfidf_baseline/run_tfidf_baseline.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Fixed experimental constants -- these mirror the BERT notebook exactly
# and must not be changed, or the two experiments stop being comparable.
# ----------------------------------------------------------------------
SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
MODEL_DIR = REPO_ROOT / "models" / "tfidf_logistic_regression"

DATA_PATH = (
    REPO_ROOT
    / "data"
    / "final_datasets"
    / "merged_severity_dataset_not_keyword_keyword.csv"
)
TEXT_COL = "content"
SEVERITY_COL = "severity"

LABEL_NAMES = ["Non-Crisis", "Implicit Crisis", "Explicit Crisis"]
IMPLICIT_IDX = 1

# Columns that must never reach the model. They are labels, annotations or
# metadata; using any of them as a feature would leak the target.
FORBIDDEN_FEATURE_COLS = [
    "severity",
    "gpt_label",
    "claude_label",
    "gemini_label",
    "llama_label",
    "mistral_label",
    "url",
    "author",
    "created",
]

# A light, brand-neutral palette so every figure reads as one system.
C_CLASS = ["#4C6EF5", "#F59F00", "#E03131"]
C_ACCENT = "#4C6EF5"
C_GRID = "#DEE2E6"
C_TEXT = "#212529"


def map_severity_to_3class(sev):
    """severity 0 -> 0, severity 1 -> 1, severity 2-6 -> 2 (same as BERT notebook)."""
    sev = int(sev)
    if sev == 0:
        return 0
    elif sev == 1:
        return 1
    else:
        return 2


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=9)
    ax.yaxis.label.set_color(C_TEXT)
    ax.xaxis.label.set_color(C_TEXT)
    ax.title.set_color(C_TEXT)


# ======================================================================
# 1. LOAD + LABEL MAPPING
# ======================================================================
def load_data():
    print("=" * 72)
    print("1. LOAD DATA")
    print("=" * 72)
    print("Dataset: " + DATA_PATH.relative_to(REPO_ROOT).as_posix())

    raw = pd.read_csv(DATA_PATH)
    print("Rows loaded : {}".format(len(raw)))
    print("Columns     : {}".format(list(raw.columns)))

    # Same cleaning step as the BERT notebook -- nothing more.
    df = raw.dropna(subset=[TEXT_COL, SEVERITY_COL]).reset_index(drop=True)
    print(
        "Rows after dropna(content, severity): {} (removed {})".format(
            len(df), len(raw) - len(df)
        )
    )

    df["label"] = df[SEVERITY_COL].apply(map_severity_to_3class)

    assert set(df["label"].unique()).issubset({0, 1, 2}), "Labels must be 0/1/2 only"

    print("\n3-class distribution:")
    counts = df["label"].value_counts().sort_index()
    for lab in range(3):
        n = int(counts.get(lab, 0))
        print(
            "  {} = {:<16} {:>5}  ({:5.2f}%)".format(
                lab, LABEL_NAMES[lab], n, 100.0 * n / len(df)
            )
        )

    class_dist = pd.DataFrame(
        {
            "label": list(range(3)),
            "class_name": LABEL_NAMES,
            "count": [int(counts.get(i, 0)) for i in range(3)],
            "proportion": [
                round(float(counts.get(i, 0)) / len(df), 4) for i in range(3)
            ],
        }
    )
    class_dist.to_csv(RESULTS_DIR / "class_distribution.csv", index=False)

    return df


# ======================================================================
# 2. DUPLICATE HANDLING
# ======================================================================
def handle_duplicates(df):
    """Report duplicates and, if any exist, drop duplicate texts before the split.

    Identical texts landing in different splits would inflate scores, so
    duplicates are removed *before* splitting rather than after. Every removal
    is counted and reported; nothing is dropped silently.
    """
    print()
    print("=" * 72)
    print("2. DUPLICATE CHECK (before splitting)")
    print("=" * 72)

    n_total = len(df)
    n_dup_rows = int(df.duplicated().sum())
    dup_content_mask = df[TEXT_COL].astype(str).duplicated(keep="first")
    n_dup_content = int(dup_content_mask.sum())

    print("Total rows           : {}".format(n_total))
    print("Duplicate rows       : {}".format(n_dup_rows))
    print("Duplicate content    : {}".format(n_dup_content))

    if n_dup_content > 0:
        conflicting = int(
            (df.groupby(df[TEXT_COL].astype(str))["label"].nunique() > 1).sum()
        )
        print("Duplicate texts with conflicting labels: {}".format(conflicting))
        df = df[~dup_content_mask].reset_index(drop=True)
        print(
            "Removed {} duplicate-content rows (kept first occurrence).".format(
                n_dup_content
            )
        )
    else:
        print("No duplicate content found -- no rows removed.")

    print("Rows remaining       : {}".format(len(df)))

    dup_report = pd.DataFrame(
        [
            ("total_rows", n_total),
            ("duplicate_rows", n_dup_rows),
            ("duplicate_content", n_dup_content),
            ("rows_removed", n_total - len(df)),
            ("rows_remaining", len(df)),
        ],
        columns=["metric", "value"],
    )
    dup_report.to_csv(RESULTS_DIR / "duplicate_report.csv", index=False)

    return df, {
        "total_rows": n_total,
        "duplicate_rows": n_dup_rows,
        "duplicate_content": n_dup_content,
        "rows_removed": n_total - len(df),
        "rows_remaining": len(df),
    }


# ======================================================================
# 3. SPLIT
# ======================================================================
def make_split(df):
    """Stratified 70 / 15 / 15, seed 42 -- identical protocol to the BERT run."""
    print()
    print("=" * 72)
    print("3. TRAIN / VALIDATION / TEST SPLIT (stratified 70/15/15, seed 42)")
    print("=" * 72)

    texts = df[TEXT_COL].astype(str).tolist()
    labels = df["label"].tolist()
    idx = list(range(len(df)))

    train_texts, temp_texts, train_labels, temp_labels, train_idx, temp_idx = (
        train_test_split(
            texts,
            labels,
            idx,
            test_size=0.30,
            random_state=SEED,
            stratify=labels,
        )
    )
    val_texts, test_texts, val_labels, test_labels, val_idx, test_idx = (
        train_test_split(
            temp_texts,
            temp_labels,
            temp_idx,
            test_size=0.50,
            random_state=SEED,
            stratify=temp_labels,
        )
    )

    n = len(df)
    print(
        "Train: {:>5}  ({:5.2f}%)".format(len(train_texts), 100.0 * len(train_texts) / n)
    )
    print("Val  : {:>5}  ({:5.2f}%)".format(len(val_texts), 100.0 * len(val_texts) / n))
    print(
        "Test : {:>5}  ({:5.2f}%)".format(len(test_texts), 100.0 * len(test_texts) / n)
    )

    # --- integrity checks ---
    assert len(train_idx) + len(val_idx) + len(test_idx) == n, "split sizes must sum"
    s_tr, s_va, s_te = set(train_idx), set(val_idx), set(test_idx)
    assert not (s_tr & s_va), "train/val row overlap"
    assert not (s_tr & s_te), "train/test row overlap"
    assert not (s_va & s_te), "val/test row overlap"
    assert not (set(train_texts) & set(test_texts)), "train/test TEXT overlap"
    assert not (set(train_texts) & set(val_texts)), "train/val TEXT overlap"
    assert not (set(val_texts) & set(test_texts)), "val/test TEXT overlap"
    print("Index overlap checks: PASS (no row and no text overlap between splits)")

    print("\nPer-split class counts:")
    header = "  {:<6} {:>12} {:>16} {:>17}".format(
        "split", LABEL_NAMES[0], LABEL_NAMES[1], LABEL_NAMES[2]
    )
    print(header)
    split_rows = []
    for name, y in [
        ("train", train_labels),
        ("val", val_labels),
        ("test", test_labels),
    ]:
        c = pd.Series(y).value_counts().sort_index()
        row = [int(c.get(i, 0)) for i in range(3)]
        print("  {:<6} {:>12} {:>16} {:>17}".format(name, row[0], row[1], row[2]))
        split_rows.append(
            {
                "split": name,
                "n": len(y),
                "Non-Crisis": row[0],
                "Implicit Crisis": row[1],
                "Explicit Crisis": row[2],
            }
        )
    pd.DataFrame(split_rows).to_csv(RESULTS_DIR / "split_distribution.csv", index=False)

    # --- persist the split assignment for the future BERT comparison ---
    split_col = np.empty(n, dtype=object)
    split_col[train_idx] = "train"
    split_col[val_idx] = "val"
    split_col[test_idx] = "test"
    split_df = pd.DataFrame(
        {
            "row_index": range(n),
            "url": df["url"] if "url" in df.columns else "",
            "content": df[TEXT_COL].astype(str),
            "severity": df[SEVERITY_COL].astype(int),
            "label": df["label"].astype(int),
            "class_name": [LABEL_NAMES[i] for i in df["label"]],
            "split": split_col,
        }
    )
    split_df.to_csv(RESULTS_DIR / "data_split.csv", index=False)
    print("\nSaved split assignment: experiments/tfidf_baseline/results/data_split.csv")

    return {
        "train": (train_texts, train_labels, train_idx),
        "val": (val_texts, val_labels, val_idx),
        "test": (test_texts, test_labels, test_idx),
    }


# ======================================================================
# 4. HYPERPARAMETER SEARCH (validation only)
# ======================================================================
# Stage A: vectorizer geometry, with the classifier held fixed.
VECTORIZER_CONFIGS = [
    {"name": "word_1_1_mindf1", "analyzer": "word", "ngram_range": (1, 1),
     "min_df": 1, "max_df": 1.0, "sublinear_tf": True, "max_features": None},
    {"name": "word_1_1_mindf2", "analyzer": "word", "ngram_range": (1, 1),
     "min_df": 2, "max_df": 0.9, "sublinear_tf": True, "max_features": None},
    {"name": "word_1_1_mindf2_nosub", "analyzer": "word", "ngram_range": (1, 1),
     "min_df": 2, "max_df": 0.9, "sublinear_tf": False, "max_features": None},
    {"name": "word_1_2_mindf1", "analyzer": "word", "ngram_range": (1, 2),
     "min_df": 1, "max_df": 1.0, "sublinear_tf": True, "max_features": None},
    {"name": "word_1_2_mindf2", "analyzer": "word", "ngram_range": (1, 2),
     "min_df": 2, "max_df": 0.9, "sublinear_tf": True, "max_features": None},
    {"name": "word_1_2_mindf3", "analyzer": "word", "ngram_range": (1, 2),
     "min_df": 3, "max_df": 0.9, "sublinear_tf": True, "max_features": None},
    {"name": "word_1_2_max20k", "analyzer": "word", "ngram_range": (1, 2),
     "min_df": 2, "max_df": 0.9, "sublinear_tf": True, "max_features": 20000},
    {"name": "char_wb_3_5", "analyzer": "char_wb", "ngram_range": (3, 5),
     "min_df": 2, "max_df": 0.9, "sublinear_tf": True, "max_features": 50000},
    {"name": "char_wb_2_5", "analyzer": "char_wb", "ngram_range": (2, 5),
     "min_df": 3, "max_df": 0.9, "sublinear_tf": True, "max_features": 50000},
]

# Stage B: classifier regularisation and class weighting, on the best vectorizer.
LR_C_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]
LR_CLASS_WEIGHTS = [None, "balanced"]

STAGE_A_C = 1.0
STAGE_A_CLASS_WEIGHT = "balanced"
LR_SOLVER = "lbfgs"
LR_MAX_ITER = 5000


def build_vectorizer(cfg):
    return TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=cfg["ngram_range"],
        min_df=cfg["min_df"],
        max_df=cfg["max_df"],
        sublinear_tf=cfg["sublinear_tf"],
        max_features=cfg["max_features"],
        lowercase=True,
        strip_accents="unicode",
    )


def fit_and_score(vec_cfg, C, class_weight, splits, verbose=True):
    """Fit vectorizer on TRAIN ONLY, transform val, return validation metrics."""
    train_texts, train_labels, _ = splits["train"]
    val_texts, val_labels, _ = splits["val"]

    vec = build_vectorizer(vec_cfg)
    X_train = vec.fit_transform(train_texts)  # FIT: training text only
    X_val = vec.transform(val_texts)  # TRANSFORM only

    clf = LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        random_state=SEED,
    )
    clf.fit(X_train, train_labels)

    val_pred = clf.predict(X_val)
    acc = accuracy_score(val_labels, val_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        val_labels, val_pred, average="macro", zero_division=0
    )
    p_cls, r_cls, f_cls, _ = precision_recall_fscore_support(
        val_labels, val_pred, labels=[0, 1, 2], zero_division=0
    )

    n_iter = int(np.max(clf.n_iter_))
    converged = n_iter < LR_MAX_ITER

    row = {
        "vectorizer_name": vec_cfg["name"],
        "analyzer": vec_cfg["analyzer"],
        "ngram_range": str(vec_cfg["ngram_range"]),
        "min_df": vec_cfg["min_df"],
        "max_df": vec_cfg["max_df"],
        "sublinear_tf": vec_cfg["sublinear_tf"],
        "max_features": (
            "None" if vec_cfg["max_features"] is None else vec_cfg["max_features"]
        ),
        "n_features_fitted": X_train.shape[1],
        "C": C,
        "class_weight": "None" if class_weight is None else class_weight,
        "solver": LR_SOLVER,
        "max_iter": LR_MAX_ITER,
        "n_iter_used": n_iter,
        "converged": converged,
        "val_accuracy": round(float(acc), 4),
        "val_macro_precision": round(float(mp), 4),
        "val_macro_recall": round(float(mr), 4),
        "val_macro_f1": round(float(mf1), 4),
        "val_implicit_precision": round(float(p_cls[IMPLICIT_IDX]), 4),
        "val_implicit_recall": round(float(r_cls[IMPLICIT_IDX]), 4),
        "val_implicit_f1": round(float(f_cls[IMPLICIT_IDX]), 4),
    }

    if verbose:
        print(
            "  {:<22} C={:<5} cw={:<9} feats={:<7} val_macroF1={:.4f}  "
            "val_acc={:.4f}  implicitF1={:.4f}{}".format(
                vec_cfg["name"],
                C,
                str(class_weight),
                X_train.shape[1],
                mf1,
                acc,
                f_cls[IMPLICIT_IDX],
                "" if converged else "  [NOT CONVERGED]",
            )
        )

    return row


def hyperparameter_search(splits):
    print()
    print("=" * 72)
    print("4. HYPERPARAMETER SEARCH (validation set only)")
    print("=" * 72)

    rows = []

    print(
        "\nStage A -- TF-IDF configurations "
        "(LogisticRegression held fixed at C={}, class_weight={}):".format(
            STAGE_A_C, STAGE_A_CLASS_WEIGHT
        )
    )
    for cfg in VECTORIZER_CONFIGS:
        row = fit_and_score(cfg, STAGE_A_C, STAGE_A_CLASS_WEIGHT, splits)
        row["stage"] = "A_vectorizer"
        rows.append(row)

    stage_a = pd.DataFrame(rows)
    best_vec_name = stage_a.loc[stage_a["val_macro_f1"].idxmax(), "vectorizer_name"]
    best_vec_cfg = next(c for c in VECTORIZER_CONFIGS if c["name"] == best_vec_name)
    print(
        "\nBest TF-IDF configuration by validation macro F1: {} ({:.4f})".format(
            best_vec_name, stage_a["val_macro_f1"].max()
        )
    )

    print("\nStage B -- LogisticRegression C and class_weight on '{}':".format(
        best_vec_name
    ))
    for C in LR_C_VALUES:
        for cw in LR_CLASS_WEIGHTS:
            row = fit_and_score(best_vec_cfg, C, cw, splits)
            row["stage"] = "B_classifier"
            rows.append(row)

    all_results = pd.DataFrame(rows)
    all_results.insert(0, "config_id", range(1, len(all_results) + 1))
    all_results = all_results.sort_values(
        "val_macro_f1", ascending=False
    ).reset_index(drop=True)
    all_results.to_csv(RESULTS_DIR / "validation_results.csv", index=False)

    best = all_results.iloc[0]
    print()
    print("-" * 72)
    print("SELECTED CONFIGURATION (highest validation macro F1)")
    print("-" * 72)
    print("  TF-IDF        : {}".format(best["vectorizer_name"]))
    print("    analyzer    : {}".format(best["analyzer"]))
    print("    ngram_range : {}".format(best["ngram_range"]))
    print("    min_df      : {}".format(best["min_df"]))
    print("    max_df      : {}".format(best["max_df"]))
    print("    sublinear_tf: {}".format(best["sublinear_tf"]))
    print("    max_features: {}".format(best["max_features"]))
    print("    n_features  : {}".format(best["n_features_fitted"]))
    print("  LogisticRegression")
    print("    C           : {}".format(best["C"]))
    print("    class_weight: {}".format(best["class_weight"]))
    print("    solver      : {}".format(best["solver"]))
    print("    max_iter    : {}".format(best["max_iter"]))
    print("  Validation macro F1 : {:.4f}".format(best["val_macro_f1"]))
    print("  Validation accuracy : {:.4f}".format(best["val_accuracy"]))
    print("  Validation implicit F1: {:.4f}".format(best["val_implicit_f1"]))
    print("\nConfiguration is now FROZEN. Test set evaluated once, below.")

    best_cfg = next(
        c for c in VECTORIZER_CONFIGS if c["name"] == best["vectorizer_name"]
    )
    best_C = float(best["C"])
    best_cw = None if best["class_weight"] == "None" else best["class_weight"]

    return all_results, best_cfg, best_C, best_cw, best


# ======================================================================
# 5. FINAL TRAINING + TEST EVALUATION
# ======================================================================
def final_evaluation(splits, best_cfg, best_C, best_cw):
    print()
    print("=" * 72)
    print("5. FINAL MODEL -- TRAIN ON TRAIN SPLIT, EVALUATE ONCE ON TEST")
    print("=" * 72)

    train_texts, train_labels, _ = splits["train"]
    val_texts, val_labels, _ = splits["val"]
    test_texts, test_labels, _ = splits["test"]

    vec = build_vectorizer(best_cfg)
    X_train = vec.fit_transform(train_texts)  # FIT on training text only
    X_val = vec.transform(val_texts)
    X_test = vec.transform(test_texts)
    print(
        "Vectorizer fitted on {} training documents -> {} features".format(
            X_train.shape[0], X_train.shape[1]
        )
    )
    print(
        "Validation matrix {}, test matrix {} (transform only)".format(
            X_val.shape, X_test.shape
        )
    )

    clf = LogisticRegression(
        C=best_C,
        class_weight=best_cw,
        solver=LR_SOLVER,
        max_iter=LR_MAX_ITER,
        random_state=SEED,
    )
    clf.fit(X_train, train_labels)
    print(
        "LogisticRegression converged in {} iterations (max_iter={})".format(
            int(np.max(clf.n_iter_)), LR_MAX_ITER
        )
    )

    y_true = np.array(test_labels)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    p_cls, r_cls, f_cls, s_cls = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )

    print()
    print("-" * 72)
    print("FINAL TEST RESULTS (n = {})".format(len(y_true)))
    print("-" * 72)
    print("  Accuracy           : {:.4f}".format(acc))
    print("  Macro Precision    : {:.4f}".format(mp))
    print("  Macro Recall       : {:.4f}".format(mr))
    print("  Macro F1           : {:.4f}".format(mf1))
    print("  Weighted Precision : {:.4f}".format(wp))
    print("  Weighted Recall    : {:.4f}".format(wr))
    print("  Weighted F1        : {:.4f}".format(wf1))

    print()
    print(
        classification_report(
            y_true, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES,
            digits=4, zero_division=0,
        )
    )

    # --- overall metrics file ---
    pd.DataFrame(
        [
            ("accuracy", round(float(acc), 4)),
            ("macro_precision", round(float(mp), 4)),
            ("macro_recall", round(float(mr), 4)),
            ("macro_f1", round(float(mf1), 4)),
            ("weighted_precision", round(float(wp), 4)),
            ("weighted_recall", round(float(wr), 4)),
            ("weighted_f1", round(float(wf1), 4)),
            ("n_test", len(y_true)),
        ],
        columns=["metric", "value"],
    ).to_csv(RESULTS_DIR / "test_results.csv", index=False)

    # --- per-class classification report ---
    report_df = pd.DataFrame(
        {
            "class_name": LABEL_NAMES,
            "label": [0, 1, 2],
            "precision": [round(float(x), 4) for x in p_cls],
            "recall": [round(float(x), 4) for x in r_cls],
            "f1": [round(float(x), 4) for x in f_cls],
            "support": [int(x) for x in s_cls],
        }
    )
    report_df.to_csv(RESULTS_DIR / "classification_report.csv", index=False)

    # ------------------------------------------------------------------
    # Implicit Crisis, called out explicitly -- the project's focus class
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    implicit_correct = int(cm[IMPLICIT_IDX, IMPLICIT_IDX])
    implicit_missed = int(cm[IMPLICIT_IDX, :].sum() - implicit_correct)
    false_implicit = int(cm[:, IMPLICIT_IDX].sum() - implicit_correct)

    print("-" * 72)
    print("IMPLICIT CRISIS -- EXPLICIT BREAKDOWN")
    print("-" * 72)
    print("  Precision : {:.4f}".format(p_cls[IMPLICIT_IDX]))
    print("  Recall    : {:.4f}".format(r_cls[IMPLICIT_IDX]))
    print("  F1        : {:.4f}".format(f_cls[IMPLICIT_IDX]))
    print("  Support   : {}".format(int(s_cls[IMPLICIT_IDX])))
    print("  Correctly detected Implicit Crisis : {}".format(implicit_correct))
    print("  Missed Implicit Crisis             : {}".format(implicit_missed))
    print(
        "    -> predicted Non-Crisis      : {}".format(int(cm[IMPLICIT_IDX, 0]))
    )
    print(
        "    -> predicted Explicit Crisis : {}".format(int(cm[IMPLICIT_IDX, 2]))
    )
    print("  Other classes wrongly called Implicit Crisis: {}".format(false_implicit))
    print("    <- true Non-Crisis      : {}".format(int(cm[0, IMPLICIT_IDX])))
    print("    <- true Explicit Crisis : {}".format(int(cm[2, IMPLICIT_IDX])))

    pd.DataFrame(
        [
            ("implicit_precision", round(float(p_cls[IMPLICIT_IDX]), 4)),
            ("implicit_recall", round(float(r_cls[IMPLICIT_IDX]), 4)),
            ("implicit_f1", round(float(f_cls[IMPLICIT_IDX]), 4)),
            ("implicit_support", int(s_cls[IMPLICIT_IDX])),
            ("implicit_correctly_detected", implicit_correct),
            ("implicit_missed_total", implicit_missed),
            ("implicit_missed_as_non_crisis", int(cm[IMPLICIT_IDX, 0])),
            ("implicit_missed_as_explicit_crisis", int(cm[IMPLICIT_IDX, 2])),
            ("false_implicit_total", false_implicit),
            ("false_implicit_from_non_crisis", int(cm[0, IMPLICIT_IDX])),
            ("false_implicit_from_explicit_crisis", int(cm[2, IMPLICIT_IDX])),
        ],
        columns=["metric", "value"],
    ).to_csv(RESULTS_DIR / "implicit_crisis_metrics.csv", index=False)

    # --- confusion matrices ---
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    idx = ["true_" + n for n in LABEL_NAMES]
    cols = ["pred_" + n for n in LABEL_NAMES]
    pd.DataFrame(cm, index=idx, columns=cols).to_csv(
        RESULTS_DIR / "confusion_matrix.csv"
    )
    pd.DataFrame(np.round(cm_norm, 4), index=idx, columns=cols).to_csv(
        RESULTS_DIR / "normalized_confusion_matrix.csv"
    )

    print()
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=idx, columns=cols))

    # --- model comparison file (TF-IDF row only) ---
    pd.DataFrame(
        [
            {
                "model": "TF-IDF + Logistic Regression",
                "accuracy": round(float(acc), 4),
                "macro_precision": round(float(mp), 4),
                "macro_recall": round(float(mr), 4),
                "macro_f1": round(float(mf1), 4),
                "weighted_precision": round(float(wp), 4),
                "weighted_recall": round(float(wr), 4),
                "weighted_f1": round(float(wf1), 4),
                "implicit_precision": round(float(p_cls[IMPLICIT_IDX]), 4),
                "implicit_recall": round(float(r_cls[IMPLICIT_IDX]), 4),
                "implicit_f1": round(float(f_cls[IMPLICIT_IDX]), 4),
            }
        ]
    ).to_csv(RESULTS_DIR / "model_comparison_ready.csv", index=False)

    metrics = {
        "accuracy": float(acc),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf1),
        "weighted_precision": float(wp),
        "weighted_recall": float(wr),
        "weighted_f1": float(wf1),
        "per_class_precision": [float(x) for x in p_cls],
        "per_class_recall": [float(x) for x in r_cls],
        "per_class_f1": [float(x) for x in f_cls],
        "per_class_support": [int(x) for x in s_cls],
        "implicit_correct": implicit_correct,
        "implicit_missed": implicit_missed,
        "false_implicit": false_implicit,
    }

    return vec, clf, cm, cm_norm, report_df, metrics


# ======================================================================
# 6. FEATURE ANALYSIS
# ======================================================================
def feature_analysis(vec, clf, top_n=20):
    print()
    print("=" * 72)
    print("6. TOP TF-IDF FEATURES PER CLASS (LogisticRegression coefficients)")
    print("=" * 72)
    print(
        "These are features the linear model weights most heavily for each class.\n"
        "They describe the learned decision boundary -- not causal indicators."
    )

    feature_names = np.array(vec.get_feature_names_out())
    coefs = clf.coef_  # shape (3, n_features) for multinomial

    rows = []
    for lab in range(coefs.shape[0]):
        order = np.argsort(coefs[lab])[::-1][:top_n]
        print("\n{} (top {}):".format(LABEL_NAMES[lab], top_n))
        terms = []
        for rank, j in enumerate(order, start=1):
            terms.append("{} ({:.3f})".format(feature_names[j], coefs[lab, j]))
            rows.append(
                {
                    "class_label": lab,
                    "class_name": LABEL_NAMES[lab],
                    "rank": rank,
                    "feature": feature_names[j],
                    "coefficient": round(float(coefs[lab, j]), 5),
                }
            )
        print("  " + ", ".join(terms[:10]))
        print("  " + ", ".join(terms[10:]))

    top_df = pd.DataFrame(rows)
    top_df.to_csv(RESULTS_DIR / "top_features.csv", index=False)
    print("\nSaved: experiments/tfidf_baseline/results/top_features.csv")
    return top_df


# ======================================================================
# 7. FIGURES
# ======================================================================
def make_figures(df, splits, val_results, cm, cm_norm, report_df, top_df):
    print()
    print("=" * 72)
    print("7. FIGURES")
    print("=" * 72)

    # --- 1. class distribution ---
    counts = [int((df["label"] == i).sum()) for i in range(3)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(LABEL_NAMES, counts, color=C_CLASS, width=0.6)
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(counts) * 0.015,
            "{}\n({:.1f}%)".format(c, 100.0 * c / sum(counts)),
            ha="center", va="bottom", fontsize=9, color=C_TEXT,
        )
    ax.set_ylabel("Number of posts")
    ax.set_title("Class distribution (n = {})".format(sum(counts)), fontsize=12, pad=12)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", color=C_GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close(fig)

    # --- 2. text length distribution ---
    word_len = df["content"].astype(str).str.split().str.len()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(word_len, bins=40, color=C_ACCENT, alpha=0.85, edgecolor="white")
    axes[0].set_xlabel("Words per post")
    axes[0].set_ylabel("Number of posts")
    axes[0].set_title("Text length, all posts", fontsize=11, pad=10)
    axes[0].grid(axis="y", color=C_GRID, linewidth=0.7, alpha=0.7)
    axes[0].set_axisbelow(True)
    style_axes(axes[0])

    data = [word_len[df["label"] == i].values for i in range(3)]
    bp = axes[1].boxplot(
        data, tick_labels=LABEL_NAMES, patch_artist=True, widths=0.5,
        medianprops={"color": C_TEXT, "linewidth": 1.4},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4,
                    "markerfacecolor": "#868E96", "markeredgecolor": "none"},
    )
    for patch, color in zip(bp["boxes"], C_CLASS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor(color)
    axes[1].set_ylabel("Words per post")
    axes[1].set_title("Text length by class", fontsize=11, pad=10)
    axes[1].grid(axis="y", color=C_GRID, linewidth=0.7, alpha=0.7)
    axes[1].set_axisbelow(True)
    style_axes(axes[1])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "text_length_distribution.png", dpi=160)
    plt.close(fig)

    # --- 3. validation hyperparameter performance ---
    vr = val_results.sort_values("val_macro_f1", ascending=True).reset_index(drop=True)
    labels = [
        "{} | C={} | cw={}".format(r["vectorizer_name"], r["C"], r["class_weight"])
        for _, r in vr.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(11, max(6, 0.32 * len(vr))))
    colors = [C_ACCENT] * len(vr)
    colors[-1] = "#2B8A3E"  # best configuration
    ax.barh(range(len(vr)), vr["val_macro_f1"], color=colors, height=0.68)
    ax.set_yticks(range(len(vr)))
    ax.set_yticklabels(labels, fontsize=8)
    for i, v in enumerate(vr["val_macro_f1"]):
        ax.text(v + 0.004, i, "{:.4f}".format(v), va="center", fontsize=8, color=C_TEXT)
    ax.set_xlabel("Validation macro F1")
    ax.set_xlim(0, max(vr["val_macro_f1"]) * 1.15)
    ax.set_title(
        "Validation macro F1 by configuration (best highlighted)", fontsize=12, pad=12
    )
    ax.grid(axis="x", color=C_GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_hyperparameters.png", dpi=160)
    plt.close(fig)

    # --- 4. per-class metrics ---
    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k, (metric, color) in enumerate(
        [("precision", "#4C6EF5"), ("recall", "#F59F00"), ("f1", "#2B8A3E")]
    ):
        vals = report_df[metric].values
        bars = ax.bar(x + (k - 1) * w, vals, width=w, label=metric.capitalize(),
                      color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    "{:.3f}".format(v), ha="center", va="bottom", fontsize=8,
                    color=C_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["{}\n(n={})".format(n, s)
         for n, s in zip(LABEL_NAMES, report_df["support"])]
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title("Test-set performance per class", fontsize=12, pad=12)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    ax.grid(axis="y", color=C_GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_metrics.png", dpi=160)
    plt.close(fig)

    # --- 5 & 6. confusion matrices ---
    def plot_cm(matrix, fname, title, fmt, cmap):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        im = ax.imshow(matrix, cmap=cmap, vmin=0,
                       vmax=matrix.max() if matrix.max() > 0 else 1)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LABEL_NAMES, fontsize=9)
        ax.set_yticklabels(LABEL_NAMES, fontsize=9, rotation=90, va="center")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(title, fontsize=12, pad=14)
        thresh = matrix.max() * 0.55
        for i in range(3):
            for j in range(3):
                ax.text(
                    j, i, fmt.format(matrix[i, j]), ha="center", va="center",
                    fontsize=12,
                    color="white" if matrix[i, j] > thresh else C_TEXT,
                )
        fig.colorbar(im, ax=ax, shrink=0.78)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / fname, dpi=160)
        plt.close(fig)

    plot_cm(cm, "confusion_matrix.png",
            "Confusion matrix (counts)", "{:d}", "Blues")
    plot_cm(cm_norm, "normalized_confusion_matrix.png",
            "Confusion matrix (row-normalised)", "{:.2f}", "Blues")

    # --- 7. top features ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    for lab in range(3):
        sub = top_df[top_df["class_label"] == lab].head(15).iloc[::-1]
        axes[lab].barh(range(len(sub)), sub["coefficient"], color=C_CLASS[lab],
                       height=0.7)
        axes[lab].set_yticks(range(len(sub)))
        axes[lab].set_yticklabels(sub["feature"], fontsize=8)
        axes[lab].set_xlabel("Coefficient")
        axes[lab].set_title(LABEL_NAMES[lab], fontsize=11, pad=10)
        axes[lab].grid(axis="x", color=C_GRID, linewidth=0.7, alpha=0.7)
        axes[lab].set_axisbelow(True)
        style_axes(axes[lab])
    fig.suptitle(
        "Top 15 TF-IDF features by Logistic Regression coefficient "
        "(learned weights, not causal indicators)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES_DIR / "top_features.png", dpi=160)
    plt.close(fig)

    for f in sorted(FIGURES_DIR.glob("*.png")):
        print("  wrote figures/{}".format(f.name))


# ======================================================================
# 8. SAVE + RELOAD TEST
# ======================================================================
def save_and_verify(vec, clf, best_cfg, best_C, best_cw, splits, metrics, dup_info):
    print()
    print("=" * 72)
    print("8. SAVE ARTIFACTS AND VERIFY RELOAD")
    print("=" * 72)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    vec_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    clf_path = MODEL_DIR / "logistic_regression.joblib"
    joblib.dump(vec, vec_path)
    joblib.dump(clf, clf_path)
    print("  saved models/tfidf_logistic_regression/tfidf_vectorizer.joblib")
    print("  saved models/tfidf_logistic_regression/logistic_regression.joblib")

    config = {
        "experiment": "TF-IDF + Logistic Regression baseline",
        "dataset": DATA_PATH.relative_to(REPO_ROOT).as_posix(),
        "text_column": TEXT_COL,
        "target_column": SEVERITY_COL,
        "label_mapping": {
            "severity_0": "0 = Non-Crisis",
            "severity_1": "1 = Implicit Crisis",
            "severity_2_to_6": "2 = Explicit Crisis",
        },
        "label_names": LABEL_NAMES,
        "excluded_from_features": FORBIDDEN_FEATURE_COLS,
        "seed": SEED,
        "split": {
            "train": 0.70, "validation": 0.15, "test": 0.15,
            "stratified": True, "random_state": SEED,
            "n_train": len(splits["train"][0]),
            "n_val": len(splits["val"][0]),
            "n_test": len(splits["test"][0]),
        },
        "duplicates": dup_info,
        "tfidf": {
            "analyzer": best_cfg["analyzer"],
            "ngram_range": list(best_cfg["ngram_range"]),
            "min_df": best_cfg["min_df"],
            "max_df": best_cfg["max_df"],
            "sublinear_tf": best_cfg["sublinear_tf"],
            "max_features": best_cfg["max_features"],
            "lowercase": True,
            "strip_accents": "unicode",
            "n_features_fitted": int(len(vec.get_feature_names_out())),
            "fitted_on": "training split only",
        },
        "logistic_regression": {
            "C": best_C,
            "class_weight": best_cw,
            "solver": LR_SOLVER,
            "max_iter": LR_MAX_ITER,
            "random_state": SEED,
            "n_iter_used": int(np.max(clf.n_iter_)),
        },
        "selection_metric": "validation macro F1",
        "test_metrics": {
            k: round(v, 4) for k, v in metrics.items()
            if isinstance(v, float)
        },
    }
    for p in [EXP_DIR / "config.json", MODEL_DIR / "config.json"]:
        p.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("  saved experiments/tfidf_baseline/config.json")
    print("  saved models/tfidf_logistic_regression/config.json")

    # --- reload verification ---
    print("\nReload verification:")
    vec2 = joblib.load(vec_path)
    clf2 = joblib.load(clf_path)
    test_texts, test_labels, _ = splits["test"]

    X_a = vec.transform(test_texts)
    X_b = vec2.transform(test_texts)
    same_matrix = (X_a - X_b).nnz == 0
    print("  reloaded vectorizer produces identical matrix : {}".format(same_matrix))

    pred_a = clf.predict(X_a)
    pred_b = clf2.predict(X_b)
    same_preds = bool(np.array_equal(pred_a, pred_b))
    print("  reloaded model produces identical predictions : {}".format(same_preds))

    acc_reload = accuracy_score(test_labels, pred_b)
    print("  reloaded model test accuracy                  : {:.4f}".format(acc_reload))

    probs = clf2.predict_proba(X_b)
    probs_ok = bool(np.allclose(probs.sum(axis=1), 1.0)) and probs.shape[1] == 3
    print("  reloaded model predict_proba valid            : {}".format(probs_ok))

    assert same_matrix, "Reloaded vectorizer did not reproduce the feature matrix"
    assert same_preds, "Reloaded model did not reproduce predictions"
    assert probs_ok, "Reloaded model produced invalid probabilities"
    print("  RELOAD TEST: PASS")


# ======================================================================
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)

    df = load_data()
    df, dup_info = handle_duplicates(df)
    splits = make_split(df)
    val_results, best_cfg, best_C, best_cw, best_row = hyperparameter_search(splits)
    vec, clf, cm, cm_norm, report_df, metrics = final_evaluation(
        splits, best_cfg, best_C, best_cw
    )
    top_df = feature_analysis(vec, clf)
    make_figures(df, splits, val_results, cm, cm_norm, report_df, top_df)
    save_and_verify(vec, clf, best_cfg, best_C, best_cw, splits, metrics, dup_info)

    print()
    print("=" * 72)
    print("DONE -- TF-IDF baseline complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
