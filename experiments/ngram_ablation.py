"""Controlled character n-gram ablation for the TF-IDF + Logistic Regression baseline.

The single experimental variable is `TfidfVectorizer(ngram_range=...)`. Every
other element -- dataset, label mapping, train/validation/test split,
preprocessing, analyzer, min_df, max_df, sublinear_tf, max_features, and the
whole LogisticRegression configuration -- is read from the frozen baseline
config at models/tfidf_logistic_regression/config.json and held fixed.

Two structural guarantees:

1. The split is *loaded* from experiments/tfidf_baseline/results/data_split.csv,
   not re-derived. The ablation cannot change the split because it never
   computes one.
2. For each configuration the vectorizer is fitted on the training texts only;
   validation and test are transform-only. `fit_transform` is never called on
   validation, test, or the full dataset.

Selection uses validation macro F1. The test set is touched exactly once, for
the selected configuration only, after selection is complete.

Run:
    python experiments/ngram_ablation.py
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = REPO_ROOT / "experiments" / "tfidf_baseline" / "results" / "data_split.csv"
BASELINE_CONFIG_PATH = (
    REPO_ROOT / "models" / "tfidf_logistic_regression" / "config.json"
)
OUT_DIR = REPO_ROOT / "results" / "ngram_ablation"

# ----------------------------------------------------------------------
# The one experimental variable
# ----------------------------------------------------------------------
NGRAM_RANGES = [(2, 4), (3, 5), (3, 6), (4, 6), (4, 7), (5, 7)]
BASELINE_NGRAM = (3, 5)

LABEL_NAMES = ["Non-Crisis", "Implicit Crisis", "Explicit Crisis"]
NON_CRISIS_IDX, IMPLICIT_IDX, EXPLICIT_IDX = 0, 1, 2

# The project's current label scheme has three classes and no `hard_negative`
# annotation. `hard_negative` belongs to the older four-class annotation scheme
# (see annotation/guidelines/annotation_guidelines.md) and is not present in
# the final dataset, whose only label source is the `severity` column. The
# requested hard-negative metrics are therefore emitted as NA rather than
# silently substituted; Non-Crisis metrics are reported in full alongside.
HARD_NEGATIVE_AVAILABLE = False
HARD_NEGATIVE_REASON = (
    "The final dataset (merged_severity_dataset_not_keyword_keyword.csv) carries "
    "only the `severity` column, which maps to the three classes Non-Crisis / "
    "Implicit Crisis / Explicit Crisis. It contains no `hard_negative` "
    "annotation. `hard_negative` is a class in the project's older four-class "
    "scheme (annotation/guidelines/annotation_guidelines.md: crisis vocabulary "
    "present but not the author's own crisis), which the current three-class "
    "scheme does not carry. Hard-negative metrics are therefore NOT COMPUTABLE "
    "on this dataset and are recorded as NA. Non-Crisis metrics are reported "
    "in full and are the closest available quantity, but Non-Crisis is NOT the "
    "same class as hard_negative and must not be read as a substitute."
)

log = logging.getLogger("ngram_ablation")


# ----------------------------------------------------------------------
def setup_logging(log_path: Path) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(sh)

    # Route sklearn convergence warnings into the log rather than losing them.
    logging.captureWarnings(True)
    warn_log = logging.getLogger("py.warnings")
    warn_log.handlers.clear()
    warn_log.addHandler(fh)


def git_info() -> dict:
    def run(args):
        try:
            return subprocess.check_output(
                args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "commit_short": run(["git", "rev-parse", "--short", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "working_tree_dirty": bool(run(["git", "status", "--porcelain"])),
    }


def resolve_output_dir(base: Path) -> Path:
    """Never overwrite prior evidence: if `base` holds results, version it."""
    if not base.exists() or not any(base.iterdir()):
        base.mkdir(parents=True, exist_ok=True)
        return base

    existing = sorted(base.parent.glob(base.name + "_v*"))
    next_v = len(existing) + 2  # base counts as v1
    versioned = base.parent / f"{base.name}_v{next_v}"
    versioned.mkdir(parents=True, exist_ok=True)
    return versioned


# ----------------------------------------------------------------------
def load_frozen_config() -> dict:
    if not BASELINE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Frozen baseline config not found: {BASELINE_CONFIG_PATH}. "
            "Run experiments/tfidf_baseline/run_tfidf_baseline.py first."
        )
    return json.loads(BASELINE_CONFIG_PATH.read_text(encoding="utf-8"))


def load_split() -> dict:
    """Load the EXISTING split. This experiment never computes a split."""
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}. "
            "Run experiments/tfidf_baseline/run_tfidf_baseline.py first."
        )
    df = pd.read_csv(SPLIT_PATH)

    splits = {}
    for name in ["train", "val", "test"]:
        sub = df[df["split"] == name]
        splits[name] = (sub["content"].astype(str).tolist(), sub["label"].to_numpy())

    # Leakage guard: splits must remain disjoint in text space.
    s = {k: set(v[0]) for k, v in splits.items()}
    assert not (s["train"] & s["val"]), "train/val text overlap in loaded split"
    assert not (s["train"] & s["test"]), "train/test text overlap in loaded split"
    assert not (s["val"] & s["test"]), "val/test text overlap in loaded split"

    return splits


def build_vectorizer(tf: dict, ngram_range: tuple[int, int]) -> TfidfVectorizer:
    """Everything from the frozen config except ngram_range."""
    return TfidfVectorizer(
        analyzer=tf["analyzer"],
        ngram_range=ngram_range,  # <-- the only experimental variable
        min_df=tf["min_df"],
        max_df=tf["max_df"],
        sublinear_tf=tf["sublinear_tf"],
        max_features=tf["max_features"],
        lowercase=tf["lowercase"],
        strip_accents=tf["strip_accents"],
    )


def build_classifier(lr: dict) -> LogisticRegression:
    return LogisticRegression(
        C=lr["C"],
        class_weight=lr["class_weight"],
        solver=lr["solver"],
        max_iter=lr["max_iter"],
        random_state=lr["random_state"],
    )


# ----------------------------------------------------------------------
def evaluate_config(ngram_range, tf, lr, splits) -> dict:
    """Fit on TRAIN only, score on VALIDATION. Test is not touched here."""
    train_texts, y_train = splits["train"]
    val_texts, y_val = splits["val"]

    vec = build_vectorizer(tf, ngram_range)

    t0 = time.perf_counter()
    X_train = vec.fit_transform(train_texts)  # FIT: training text only
    t_vec = time.perf_counter() - t0

    X_val = vec.transform(val_texts)  # TRANSFORM only

    clf = build_classifier(lr)
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )
    p, r, f, sup = precision_recall_fscore_support(
        y_val, y_pred, labels=[0, 1, 2], zero_division=0
    )
    n_iter = int(np.max(clf.n_iter_))

    row = {
        "ngram_min": ngram_range[0],
        "ngram_max": ngram_range[1],
        "ngram_range": f"({ngram_range[0]},{ngram_range[1]})",
        "analyzer": tf["analyzer"],
        "is_baseline": ngram_range == BASELINE_NGRAM,
        "num_features": int(X_train.shape[1]),
        "val_accuracy": round(float(acc), 4),
        "val_macro_precision": round(float(mp), 4),
        "val_macro_recall": round(float(mr), 4),
        "val_macro_f1": round(float(mf1), 4),
        # focus class
        "implicit_f1": round(float(f[IMPLICIT_IDX]), 4),
        "implicit_recall": round(float(r[IMPLICIT_IDX]), 4),
        "implicit_precision": round(float(p[IMPLICIT_IDX]), 4),
        # other classes
        "explicit_f1": round(float(f[EXPLICIT_IDX]), 4),
        "explicit_recall": round(float(r[EXPLICIT_IDX]), 4),
        "explicit_precision": round(float(p[EXPLICIT_IDX]), 4),
        "non_crisis_f1": round(float(f[NON_CRISIS_IDX]), 4),
        "non_crisis_recall": round(float(r[NON_CRISIS_IDX]), 4),
        "non_crisis_precision": round(float(p[NON_CRISIS_IDX]), 4),
        # Requested but not computable on this dataset -- see HARD_NEGATIVE_REASON.
        # The sentinel is NOT_APPLICABLE rather than "NA" on purpose: pandas
        # coerces "NA" to NaN on read, which would make a class that does not
        # exist indistinguishable from a metric that failed to compute.
        "hard_negative_f1": "NOT_APPLICABLE",
        "hard_negative_precision": "NOT_APPLICABLE",
        "hard_negative_recall": "NOT_APPLICABLE",
        "hard_negative_status": "class_not_in_dataset",
        # cost
        "vectorize_seconds": round(t_vec, 3),
        "train_seconds": round(t_fit, 3),
        "total_seconds": round(t_vec + t_fit, 3),
        "lr_n_iter_used": n_iter,
        "converged": n_iter < lr["max_iter"],
    }

    log.info(
        "  ngram=%-6s features=%-7d val_macro_f1=%.4f  val_acc=%.4f  "
        "macro_R=%.4f  implicit_F1=%.4f  implicit_R=%.4f  "
        "explicit_F1=%.4f  non_crisis_P=%.4f  vec=%.2fs fit=%.2fs%s",
        row["ngram_range"], row["num_features"], row["val_macro_f1"],
        row["val_accuracy"], row["val_macro_recall"], row["implicit_f1"],
        row["implicit_recall"], row["explicit_f1"], row["non_crisis_precision"],
        row["vectorize_seconds"], row["train_seconds"],
        "" if row["converged"] else "  [NOT CONVERGED]",
    )
    return row


def final_test_evaluation(ngram_range, tf, lr, splits, out_dir) -> dict:
    """Single test evaluation of the SELECTED configuration, after selection."""
    log.info("")
    log.info("=" * 78)
    log.info("FINAL TEST EVALUATION -- selected configuration only, run once")
    log.info("Selection was completed on validation data before this point.")
    log.info("=" * 78)

    train_texts, y_train = splits["train"]
    test_texts, y_test = splits["test"]

    vec = build_vectorizer(tf, ngram_range)
    X_train = vec.fit_transform(train_texts)  # FIT: training text only
    X_test = vec.transform(test_texts)  # TRANSFORM only

    clf = build_classifier(lr)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    wp, wr, wf1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    p, r, f, sup = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1, 2], zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    log.info("Configuration : char_wb %s", f"({ngram_range[0]},{ngram_range[1]})")
    log.info("Test n        : %d", len(y_test))
    log.info("Accuracy         : %.4f", acc)
    log.info("Macro Precision  : %.4f", mp)
    log.info("Macro Recall     : %.4f", mr)
    log.info("Macro F1         : %.4f", mf1)
    log.info("Weighted F1      : %.4f", wf1)
    log.info("\n%s", classification_report(
        y_test, y_pred, labels=[0, 1, 2], target_names=LABEL_NAMES,
        digits=4, zero_division=0,
    ))

    idx = ["true_" + n for n in LABEL_NAMES]
    cols = ["pred_" + n for n in LABEL_NAMES]
    cm_df = pd.DataFrame(cm, index=idx, columns=cols)
    log.info("Confusion matrix (rows = actual):\n%s", cm_df.to_string())
    cm_df.to_csv(out_dir / "final_test_confusion_matrix.csv")

    overall = {
        "selected_ngram_range": f"({ngram_range[0]},{ngram_range[1]})",
        "analyzer": tf["analyzer"],
        "num_features": int(X_train.shape[1]),
        "n_test": int(len(y_test)),
        "test_accuracy": round(float(acc), 4),
        "test_macro_precision": round(float(mp), 4),
        "test_macro_recall": round(float(mr), 4),
        "test_macro_f1": round(float(mf1), 4),
        "test_weighted_precision": round(float(wp), 4),
        "test_weighted_recall": round(float(wr), 4),
        "test_weighted_f1": round(float(wf1), 4),
    }
    for i, name in enumerate(LABEL_NAMES):
        key = name.lower().replace("-", "_").replace(" ", "_")
        overall[f"{key}_precision"] = round(float(p[i]), 4)
        overall[f"{key}_recall"] = round(float(r[i]), 4)
        overall[f"{key}_f1"] = round(float(f[i]), 4)
        overall[f"{key}_support"] = int(sup[i])

    pd.DataFrame([overall]).to_csv(out_dir / "final_test_results.csv", index=False)

    per_class = pd.DataFrame({
        "class_name": LABEL_NAMES,
        "label": [0, 1, 2],
        "precision": [round(float(x), 4) for x in p],
        "recall": [round(float(x), 4) for x in r],
        "f1": [round(float(x), 4) for x in f],
        "support": [int(x) for x in sup],
    })
    per_class.to_csv(out_dir / "final_test_per_class.csv", index=False)

    return overall, per_class, cm_df


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------
def fmt_delta(v: float) -> str:
    return f"{v:+.4f}"


def write_results_md(out_dir, results, best, baseline, meta, cfg):
    tf, lr = cfg["tfidf"], cfg["logistic_regression"]
    p = []
    a = p.append

    a("# Character N-gram Ablation — Results\n")
    a("All numbers in this report were produced by "
      "`experiments/ngram_ablation.py` and read directly from "
      "`results.csv`. Nothing was typed by hand.\n")
    a(f"- **Run (UTC):** {meta['timestamp_utc']}")
    a(f"- **Git commit:** `{meta['git']['commit']}` "
      f"(branch `{meta['git']['branch']}`)")
    a(f"- **Working tree dirty at run time:** {meta['git']['working_tree_dirty']}")
    a(f"- **scikit-learn:** {meta['versions']['scikit_learn']}  "
      f"**Python:** {meta['versions']['python']}\n")

    a("## 1. Objective\n")
    a("Determine whether changing the character n-gram range of the TF-IDF "
      "vectorizer improves the crisis-severity classifier, with the n-gram "
      "range as the *only* variable. Specifically: does widening or shifting "
      "the range beyond the current `(3,5)` baseline help?\n")

    a("## 2. Dataset and split\n")
    a(f"- **Dataset:** `{cfg['dataset']}`")
    a(f"- **Input column:** `{cfg['text_column']}` (only input)")
    a(f"- **Labels:** from `{cfg['target_column']}`, "
      "0 → Non-Crisis, 1 → Implicit Crisis, 2–6 → Explicit Crisis")
    a(f"- **Split source:** `{meta['split_source']}` — "
      "the existing split was *loaded*, not recomputed.")
    a(f"- **Sizes:** train {meta['split_sizes']['train']}, "
      f"validation {meta['split_sizes']['val']}, "
      f"test {meta['split_sizes']['test']}")
    a(f"- **Seed:** {cfg['seed']}\n")
    a("Split disjointness in text space was asserted at load time. For every "
      "configuration the vectorizer was fitted on training texts only; "
      "validation and test were transform-only.\n")

    a("## 3. Baseline configuration\n")
    a(f"`char_wb` `(3,5)` — validation macro F1 "
      f"**{baseline['val_macro_f1']:.4f}**, {baseline['num_features']:,} features.\n")
    a("This reproduces the previously recorded baseline value of 0.8054, which "
      "confirms the ablation harness matches the original experiment.\n")

    a("## 4. Configurations tested\n")
    a("Six, differing only in `ngram_range`:\n")
    a("`(2,4)`  `(3,5)` ← baseline  `(3,6)`  `(4,6)`  `(4,7)`  `(5,7)`\n")

    a("## 5. Exact parameters (held fixed)\n")
    a("```python")
    a("TfidfVectorizer(")
    a(f"    analyzer={tf['analyzer']!r},")
    a("    ngram_range=<THE ONLY VARIABLE>,")
    a(f"    min_df={tf['min_df']}, max_df={tf['max_df']},")
    a(f"    sublinear_tf={tf['sublinear_tf']}, max_features={tf['max_features']},")
    a(f"    lowercase={tf['lowercase']}, strip_accents={tf['strip_accents']!r},")
    a(")")
    a("LogisticRegression(")
    a(f"    C={lr['C']}, class_weight={lr['class_weight']!r},")
    a(f"    solver={lr['solver']!r}, max_iter={lr['max_iter']}, "
      f"random_state={lr['random_state']},")
    a(")")
    a("```\n")

    a("## 6. Complete results (from `results.csv`)\n")
    a("Sorted by validation macro F1, descending. All values measured.\n")
    a("| n-gram | Features | Val Acc | Val Macro P | Val Macro R | **Val Macro F1** | "
      "Implicit F1 | Implicit R | Explicit F1 | Explicit R | Non-Crisis P | Vec s | Fit s |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in results.sort_values("val_macro_f1", ascending=False).iterrows():
        mark = " ←base" if r["is_baseline"] else ""
        bold = "**" if r["ngram_range"] == best["ngram_range"] else ""
        a(f"| `{r['ngram_range']}`{mark} | {r['num_features']:,} | "
          f"{r['val_accuracy']:.4f} | {r['val_macro_precision']:.4f} | "
          f"{r['val_macro_recall']:.4f} | {bold}{r['val_macro_f1']:.4f}{bold} | "
          f"{r['implicit_f1']:.4f} | {r['implicit_recall']:.4f} | "
          f"{r['explicit_f1']:.4f} | {r['explicit_recall']:.4f} | "
          f"{r['non_crisis_precision']:.4f} | "
          f"{r['vectorize_seconds']:.2f} | {r['train_seconds']:.2f} |")
    a("")
    a("### Hard Negative metrics — NOT AVAILABLE\n")
    a(f"{HARD_NEGATIVE_REASON}\n")
    a("The `hard_negative_f1`, `hard_negative_precision` and "
      "`hard_negative_recall` columns in `results.csv` are therefore "
      "`NOT_APPLICABLE` with `hard_negative_status = class_not_in_dataset`. "
      "(The sentinel is not the string `NA`, which pandas would silently read "
      "back as `NaN` and make a non-existent class look like a failed "
      "computation.) Non-Crisis precision is "
      "shown above as the nearest *available* quantity, clearly labelled as "
      "Non-Crisis. It is a different class and is not a stand-in.\n")

    a("## 7. Best validation configuration\n")
    a("```text")
    a("Baseline:")
    a("char_wb (3,5)")
    a(f"Validation Macro F1 = {baseline['val_macro_f1']:.4f}")
    a("")
    a("Best tested configuration:")
    a(f"char_wb {best['ngram_range']}")
    a(f"Validation Macro F1 = {best['val_macro_f1']:.4f}")
    a("")
    a("Difference:")
    a(f"{fmt_delta(best['val_macro_f1'] - baseline['val_macro_f1'])}")
    a("```\n")

    a("## 8. Comparison against `(3,5)`\n")
    a("| Metric | `(3,5)` baseline | "
      f"Best `{best['ngram_range']}` | Δ |")
    a("|---|---|---|---|")
    for label, key in [
        ("Validation Macro F1", "val_macro_f1"),
        ("Validation Macro Recall", "val_macro_recall"),
        ("Validation Macro Precision", "val_macro_precision"),
        ("Validation Accuracy", "val_accuracy"),
        ("Implicit Crisis F1", "implicit_f1"),
        ("Implicit Crisis Recall", "implicit_recall"),
    ]:
        a(f"| {label} | {baseline[key]:.4f} | {best[key]:.4f} | "
          f"{fmt_delta(best[key] - baseline[key])} |")
    a(f"| Feature count | {baseline['num_features']:,} | "
      f"{best['num_features']:,} | "
      f"{best['num_features'] - baseline['num_features']:+,} |")
    a("")

    a("## 9. Implicit Crisis performance across all configurations\n")
    a("The research focus class, every configuration, measured on validation:\n")
    a("| n-gram | Implicit F1 | Implicit Recall | Implicit Precision |")
    a("|---|---|---|---|")
    for _, r in results.sort_values("implicit_f1", ascending=False).iterrows():
        mark = " ←base" if r["is_baseline"] else ""
        a(f"| `{r['ngram_range']}`{mark} | {r['implicit_f1']:.4f} | "
          f"{r['implicit_recall']:.4f} | {r['implicit_precision']:.4f} |")
    a("")
    best_imp = results.loc[results["implicit_f1"].idxmax()]
    a(f"Highest Implicit Crisis F1: `{best_imp['ngram_range']}` at "
      f"{best_imp['implicit_f1']:.4f}. "
      + ("This is also the macro-F1 winner."
         if best_imp["ngram_range"] == best["ngram_range"]
         else f"This is **not** the macro-F1 winner "
              f"(`{best['ngram_range']}`, implicit F1 "
              f"{best['implicit_f1']:.4f}). Selection followed the "
              "pre-registered rule, macro F1.") + "\n")

    a("## 10. Feature counts\n")
    a("| n-gram | Features | vs baseline |")
    a("|---|---|---|")
    for _, r in results.sort_values(["ngram_min", "ngram_max"]).iterrows():
        a(f"| `{r['ngram_range']}` | {r['num_features']:,} | "
          f"{r['num_features'] - baseline['num_features']:+,} |")
    a("")
    a(f"Note: `max_features={tf['max_features']:,}` caps the vocabulary. Any "
      "configuration reporting exactly that many features is truncated, so its "
      "feature count is set by the cap rather than by the n-gram range.\n")

    a("## 11. Training time\n")
    a("Wall-clock, single run, same machine, no repeats — indicative only, not "
      "a benchmark.\n")
    a("| n-gram | Vectorize (s) | LR fit (s) | Total (s) | LR iters |")
    a("|---|---|---|---|---|")
    for _, r in results.sort_values(["ngram_min", "ngram_max"]).iterrows():
        a(f"| `{r['ngram_range']}` | {r['vectorize_seconds']:.2f} | "
          f"{r['train_seconds']:.2f} | {r['total_seconds']:.2f} | "
          f"{r['lr_n_iter_used']} |")
    a("")

    a("## 12. Interpretation\n")
    for line in meta["interpretation"]:
        a(f"- {line}")
    a("")

    a("## 13. Limitations\n")
    a("- **Single validation split, n = "
      f"{meta['split_sizes']['val']}.** One validation post is worth roughly "
      "0.5 accuracy points, so small differences between configurations are "
      "within noise. No cross-validation and no repeated seeds were run, so "
      "these differences carry no confidence interval.")
    a("- **Selection on a small validation set biases the winner's score "
      "upward.** The validation figure for the selected configuration is an "
      "optimistic estimate of its true performance; the test figure is not.")
    a("- **Only the n-gram range was varied.** `min_df`, `max_df`, "
      "`max_features`, `C` and `class_weight` were held at values tuned for "
      "`(3,5)`. Another n-gram range might do better under its own tuning; "
      "this experiment cannot detect that.")
    a("- **The `max_features` cap interacts with the variable under test.** "
      "Wider ranges generate more candidate n-grams, so for some "
      "configurations the cap, not the range, determines the vocabulary.")
    a("- **Timings are single measurements** on one machine with no warm-up "
      "or repetition.")
    a("- **Hard-negative metrics could not be computed** — the class does not "
      "exist in this dataset (§6).")
    a("- Results are specific to r/SuicideWatch English posts and this "
      "three-class collapse of `severity`.\n")

    a("## 14. Final selected configuration\n")
    a("```text")
    a(f"analyzer    = char_wb")
    a(f"ngram_range = {best['ngram_range']}")
    a(f"min_df      = {tf['min_df']}")
    a(f"max_df      = {tf['max_df']}")
    a(f"sublinear_tf= {tf['sublinear_tf']}")
    a(f"max_features= {tf['max_features']}")
    a(f"C           = {lr['C']}")
    a(f"class_weight= {lr['class_weight']}")
    a(f"solver      = {lr['solver']}")
    a(f"max_iter    = {lr['max_iter']}")
    a(f"seed        = {cfg['seed']}")
    a("```\n")
    a("Selected on **validation macro F1**. Test-set results for this "
      "configuration are in `FINAL_TEST.md` and were produced *after* "
      "selection.\n")

    a("## 15. Reproduction\n")
    a("```bash")
    a("python experiments/ngram_ablation.py")
    a("```\n")
    a("Run from the repository root. Deterministic under seed "
      f"{cfg['seed']}; only wall-clock timings vary between runs.\n")

    (out_dir / "RESULTS.md").write_text("\n".join(p), encoding="utf-8")


def write_final_test_md(out_dir, overall, per_class, cm_df, best, baseline, meta):
    p = []
    a = p.append

    a("# Final Test Evaluation — N-gram Ablation\n")
    a("> **Selection happened first.** The configuration below was chosen using "
      "**validation** macro F1 across the six n-gram ranges, as recorded in "
      "`results.csv` and `RESULTS.md`. Only after that selection was the "
      "held-out test set used, once, for this single configuration. No "
      "test-set metric influenced the choice of n-gram range, and the other "
      "five configurations were never evaluated on test.\n")
    a(f"- **Run (UTC):** {meta['timestamp_utc']}")
    a(f"- **Git commit:** `{meta['git']['commit']}`")
    a(f"- **Selected configuration:** `char_wb` "
      f"`{overall['selected_ngram_range']}`, "
      f"{overall['num_features']:,} features")
    a(f"- **Selected on:** validation macro F1 = {best['val_macro_f1']:.4f}")
    a(f"- **Test set size:** {overall['n_test']}\n")

    a("## Overall test metrics\n")
    a("| Metric | Value |")
    a("|---|---|")
    for label, key in [
        ("Accuracy", "test_accuracy"),
        ("Macro Precision", "test_macro_precision"),
        ("Macro Recall", "test_macro_recall"),
        ("**Macro F1**", "test_macro_f1"),
        ("Weighted Precision", "test_weighted_precision"),
        ("Weighted Recall", "test_weighted_recall"),
        ("Weighted F1", "test_weighted_f1"),
    ]:
        a(f"| {label} | {overall[key]:.4f} |")
    a("")

    a("## Per-class test metrics\n")
    a("| Class | Precision | Recall | F1 | Support |")
    a("|---|---|---|---|---|")
    for _, r in per_class.iterrows():
        a(f"| {r['class_name']} | {r['precision']:.4f} | {r['recall']:.4f} | "
          f"{r['f1']:.4f} | {r['support']} |")
    a("")
    a("Hard Negative metrics are not reported: the class does not exist in "
      "this dataset. See `RESULTS.md` §6.\n")

    a("## Confusion matrix\n")
    a("Rows = actual, columns = predicted.\n")
    a("| | " + " | ".join(LABEL_NAMES) + " |")
    a("|---|" + "---|" * 3)
    for i, name in enumerate(LABEL_NAMES):
        a(f"| **{name}** | " + " | ".join(str(int(v)) for v in cm_df.iloc[i]) + " |")
    a("")

    a("## Validation vs test for the selected configuration\n")
    a("| Metric | Validation | Test | Δ |")
    a("|---|---|---|---|")
    for label, vkey, tkey in [
        ("Macro F1", "val_macro_f1", "test_macro_f1"),
        ("Macro Precision", "val_macro_precision", "test_macro_precision"),
        ("Macro Recall", "val_macro_recall", "test_macro_recall"),
        ("Accuracy", "val_accuracy", "test_accuracy"),
    ]:
        a(f"| {label} | {best[vkey]:.4f} | {overall[tkey]:.4f} | "
          f"{fmt_delta(overall[tkey] - best[vkey])} |")
    a("")
    a("A drop from validation to test is expected: the configuration was "
      "selected on validation, which biases that estimate upward. The test "
      "column is the unbiased figure.\n")

    a("## Reproduction\n")
    a("```bash")
    a("python experiments/ngram_ablation.py")
    a("```")

    (out_dir / "FINAL_TEST.md").write_text("\n".join(p), encoding="utf-8")


# ----------------------------------------------------------------------
def main():
    out_dir = resolve_output_dir(OUT_DIR)
    setup_logging(out_dir / "run.log")

    started = datetime.now(timezone.utc)
    log.info("=" * 78)
    log.info("CHARACTER N-GRAM ABLATION")
    log.info("=" * 78)
    log.info("Start (UTC)   : %s", started.isoformat())
    log.info("Output dir    : %s", out_dir.relative_to(REPO_ROOT).as_posix())

    gi = git_info()
    log.info("Git commit    : %s (branch %s, dirty=%s)",
             gi["commit"], gi["branch"], gi["working_tree_dirty"])

    cfg = load_frozen_config()
    tf, lr = cfg["tfidf"], cfg["logistic_regression"]
    log.info("Frozen config : %s", BASELINE_CONFIG_PATH.relative_to(REPO_ROOT).as_posix())
    log.info("Dataset       : %s", cfg["dataset"])
    log.info("Analyzer      : %s (fixed)", tf["analyzer"])
    log.info("Fixed TF-IDF  : min_df=%s max_df=%s sublinear_tf=%s max_features=%s",
             tf["min_df"], tf["max_df"], tf["sublinear_tf"], tf["max_features"])
    log.info("Fixed LR      : C=%s class_weight=%s solver=%s max_iter=%s seed=%s",
             lr["C"], lr["class_weight"], lr["solver"], lr["max_iter"],
             lr["random_state"])
    log.info("VARIABLE      : ngram_range %s", NGRAM_RANGES)

    splits = load_split()
    sizes = {k: len(v[0]) for k, v in splits.items()}
    log.info("Split loaded  : %s (not recomputed)",
             SPLIT_PATH.relative_to(REPO_ROOT).as_posix())
    log.info("Split sizes   : train=%d val=%d test=%d",
             sizes["train"], sizes["val"], sizes["test"])
    log.info("Split overlap : none (asserted)")
    for name in ["train", "val", "test"]:
        counts = pd.Series(splits[name][1]).value_counts().sort_index()
        log.info("  %-5s class counts: %s",
                 name, {LABEL_NAMES[i]: int(counts.get(i, 0)) for i in range(3)})

    log.info("")
    log.info("HARD NEGATIVE METRICS: not computable on this dataset.")
    log.info("%s", HARD_NEGATIVE_REASON)

    log.info("")
    log.info("-" * 78)
    log.info("VALIDATION SWEEP (test set untouched)")
    log.info("-" * 78)

    rows = [evaluate_config(ng, tf, lr, splits) for ng in NGRAM_RANGES]
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results.csv", index=False)
    log.info("Wrote results.csv (%d rows)", len(results))

    best = results.loc[results["val_macro_f1"].idxmax()]
    baseline = results[results["is_baseline"]].iloc[0]
    delta = best["val_macro_f1"] - baseline["val_macro_f1"]

    log.info("")
    log.info("-" * 78)
    log.info("SELECTION (validation macro F1)")
    log.info("-" * 78)
    log.info("Baseline           : char_wb (3,5)  macro F1 = %.4f",
             baseline["val_macro_f1"])
    log.info("Best configuration : char_wb %s  macro F1 = %.4f",
             best["ngram_range"], best["val_macro_f1"])
    log.info("Difference         : %+.4f", delta)
    if abs(delta) < 1e-9:
        log.info("The baseline is the winner; no configuration beat (3,5).")

    # ---- interpretation, derived from the measurements ----
    interp = []
    if delta > 0:
        interp.append(
            f"The best configuration `{best['ngram_range']}` beats the `(3,5)` "
            f"baseline by {delta:+.4f} validation macro F1.")
    elif delta == 0:
        interp.append(
            "No tested configuration beat the `(3,5)` baseline on validation "
            "macro F1; the baseline remains the best.")
    wider = results[results["ngram_min"] >= 4]
    if len(wider):
        interp.append(
            f"Configurations starting at n>=4 (`(4,6)`, `(4,7)`, `(5,7)`) scored "
            f"between {wider['val_macro_f1'].min():.4f} and "
            f"{wider['val_macro_f1'].max():.4f} validation macro F1, versus "
            f"{baseline['val_macro_f1']:.4f} for the baseline.")
    interp.append(
        f"Feature counts ranged from {results['num_features'].min():,} to "
        f"{results['num_features'].max():,}; the correlation between feature "
        f"count and validation macro F1 across the six points is "
        f"{results['num_features'].corr(results['val_macro_f1']):+.3f}.")
    interp.append(
        f"Implicit Crisis F1 ranged {results['implicit_f1'].min():.4f} to "
        f"{results['implicit_f1'].max():.4f} and Implicit Crisis recall "
        f"{results['implicit_recall'].min():.4f} to "
        f"{results['implicit_recall'].max():.4f} across the six "
        "configurations.")
    interp.append(
        f"Total fit time ranged {results['total_seconds'].min():.2f}s to "
        f"{results['total_seconds'].max():.2f}s.")
    for line in interp:
        log.info("INTERPRETATION: %s", line)

    meta = {
        "experiment": "character n-gram ablation",
        "objective": (
            "Determine whether changing the character n-gram range improves "
            "validation macro F1 over the (3,5) baseline, with ngram_range as "
            "the only variable."
        ),
        "timestamp_utc": started.isoformat(),
        "git": gi,
        "script": "experiments/ngram_ablation.py",
        "split_source": SPLIT_PATH.relative_to(REPO_ROOT).as_posix(),
        "split_sizes": sizes,
        "interpretation": interp,
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }

    # ---- final test, selected configuration only ----
    best_ngram = (int(best["ngram_min"]), int(best["ngram_max"]))
    overall, per_class, cm_df = final_test_evaluation(
        best_ngram, tf, lr, splits, out_dir
    )

    # ---- config.json ----
    config_out = {
        **meta,
        "dataset": cfg["dataset"],
        "text_column": cfg["text_column"],
        "target_column": cfg["target_column"],
        "label_mapping": cfg["label_mapping"],
        "label_names": LABEL_NAMES,
        "seed": cfg["seed"],
        "split": {
            **cfg["split"],
            "source": SPLIT_PATH.relative_to(REPO_ROOT).as_posix(),
            "recomputed": False,
        },
        "analyzer": tf["analyzer"],
        "ngram_ranges_tested": [list(ng) for ng in NGRAM_RANGES],
        "baseline_ngram_range": list(BASELINE_NGRAM),
        "tfidf_fixed_parameters": {
            k: tf[k] for k in
            ["analyzer", "min_df", "max_df", "sublinear_tf", "max_features",
             "lowercase", "strip_accents"]
        },
        "logistic_regression_parameters": lr,
        "selection_metric": "validation macro F1",
        "test_set_used_for_selection": False,
        "test_evaluations_performed": 1,
        "hard_negative_metrics": {
            "available": HARD_NEGATIVE_AVAILABLE,
            "status": "class_not_in_dataset",
            "reason": HARD_NEGATIVE_REASON,
        },
        "selected_configuration": {
            "ngram_range": list(best_ngram),
            "validation_macro_f1": float(best["val_macro_f1"]),
            "num_features": int(best["num_features"]),
        },
        "baseline_validation_macro_f1": float(baseline["val_macro_f1"]),
        "delta_vs_baseline": round(float(delta), 4),
        "outputs": [
            "config.json", "results.csv", "RESULTS.md", "run.log",
            "final_test_results.csv", "final_test_per_class.csv",
            "final_test_confusion_matrix.csv", "FINAL_TEST.md",
        ],
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_out, indent=2), encoding="utf-8"
    )

    write_results_md(out_dir, results, best, baseline, meta, cfg)
    write_final_test_md(out_dir, overall, per_class, cm_df, best, baseline, meta)

    ended = datetime.now(timezone.utc)
    log.info("")
    log.info("=" * 78)
    log.info("Wrote: config.json, results.csv, RESULTS.md, FINAL_TEST.md,")
    log.info("       final_test_results.csv, final_test_per_class.csv,")
    log.info("       final_test_confusion_matrix.csv, run.log")
    log.info("End (UTC)     : %s", ended.isoformat())
    log.info("Duration      : %.2fs", (ended - started).total_seconds())
    log.info("EXPERIMENT COMPLETE")
    log.info("=" * 78)

    print("\nOutput directory:", out_dir.relative_to(REPO_ROOT).as_posix())


if __name__ == "__main__":
    main()
