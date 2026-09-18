"""
Dataset audit for the TF-IDF + Logistic Regression baseline.

Audits ONLY the project's final dataset:
    data/final_datasets/merged_severity_dataset_not_keyword_keyword.csv

The audit does not select, compare, or substitute a dataset. The dataset is
fixed by the project; this script only validates and describes it so the
baseline experiment rests on known ground.

Run:
    python experiments/tfidf_baseline/audit_dataset.py
"""

from pathlib import Path

import pandas as pd

# --- Fixed experimental constants (must match the BERT reference notebook) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = (
    REPO_ROOT
    / "data"
    / "final_datasets"
    / "merged_severity_dataset_not_keyword_keyword.csv"
)
TEXT_COL = "content"
SEVERITY_COL = "severity"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"

LABEL_NAMES = ["Non-Crisis", "Implicit Crisis", "Explicit Crisis"]


def map_severity_to_3class(sev):
    """severity 0 -> 0, severity 1 -> 1, severity 2-6 -> 2 (same as BERT notebook)."""
    sev = int(sev)
    if sev == 0:
        return 0
    elif sev == 1:
        return 1
    else:
        return 2


def describe(series, name):
    print("\n" + name + ":")
    print("  count  {}".format(int(series.count())))
    print("  mean   {:.2f}".format(series.mean()))
    print("  std    {:.2f}".format(series.std()))
    print("  min    {}".format(int(series.min())))
    print("  25%    {:.1f}".format(series.quantile(0.25)))
    print("  median {:.1f}".format(series.median()))
    print("  75%    {:.1f}".format(series.quantile(0.75)))
    print("  95%    {:.1f}".format(series.quantile(0.95)))
    print("  max    {}".format(int(series.max())))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DATASET AUDIT")
    print("=" * 70)
    print("Dataset: " + DATA_PATH.relative_to(REPO_ROOT).as_posix())
    if not DATA_PATH.exists():
        raise FileNotFoundError("Dataset not found: {}".format(DATA_PATH))

    raw = pd.read_csv(DATA_PATH)

    print("\nRaw rows   : {}".format(len(raw)))
    print("Raw columns: {}".format(raw.shape[1]))
    print("Column names: {}".format(list(raw.columns)))

    if TEXT_COL not in raw.columns:
        raise KeyError("Expected text column '{}' not found.".format(TEXT_COL))
    if SEVERITY_COL not in raw.columns:
        raise KeyError("Expected label column '{}' not found.".format(SEVERITY_COL))

    # --- Missing values, per column ---
    print("\nMissing values per column:")
    missing = raw.isna().sum()
    for col, n in missing.items():
        print("  {:<16} {}".format(col, n))

    # --- Empty / whitespace-only content (before dropna) ---
    content_str = raw[TEXT_COL].astype("string")
    n_content_na = int(raw[TEXT_COL].isna().sum())
    n_content_empty = int((content_str.fillna("").str.strip() == "").sum())
    print("\nContent NaN             : {}".format(n_content_na))
    print("Content empty/whitespace: {}".format(n_content_empty))

    # --- Severity distribution (raw values, before collapsing) ---
    print("\nSeverity distribution (raw):")
    sev_counts = raw[SEVERITY_COL].value_counts(dropna=False).sort_index()
    for sev, n in sev_counts.items():
        print("  severity={:<8} {}".format(str(sev), n))

    # --- Same cleaning step as the BERT notebook ---
    df = raw.dropna(subset=[TEXT_COL, SEVERITY_COL]).reset_index(drop=True)
    n_after_dropna = len(df)
    print(
        "\nRows after dropna(content, severity): {} (removed {})".format(
            n_after_dropna, len(raw) - n_after_dropna
        )
    )

    df["label"] = df[SEVERITY_COL].apply(map_severity_to_3class)

    print("\nFinal 3-class distribution:")
    label_counts = df["label"].value_counts().sort_index()
    for lab in range(3):
        n = int(label_counts.get(lab, 0))
        print(
            "  {} = {:<16} {:>6}  ({:5.2f}%)".format(
                lab, LABEL_NAMES[lab], n, 100.0 * n / len(df)
            )
        )

    # --- Duplicates ---
    n_dup_full_rows = int(df.duplicated().sum())
    dup_content_mask = df[TEXT_COL].astype(str).duplicated(keep="first")
    n_dup_content = int(dup_content_mask.sum())
    n_unique_content = int(df[TEXT_COL].astype(str).nunique())

    print("\nDuplicate full rows           : {}".format(n_dup_full_rows))
    print("Duplicate content (keep=first): {}".format(n_dup_content))
    print("Unique content values         : {}".format(n_unique_content))

    # Do duplicated texts ever carry conflicting labels?
    conflicting = 0
    if n_dup_content:
        grp = df.groupby(df[TEXT_COL].astype(str))["label"].nunique()
        conflicting = int((grp > 1).sum())
        print("Duplicate texts with conflicting labels: {}".format(conflicting))

    # --- Text length statistics ---
    char_len = df[TEXT_COL].astype(str).str.len()
    word_len = df[TEXT_COL].astype(str).str.split().str.len()

    describe(char_len, "Text length (characters)")
    describe(word_len, "Text length (words)")

    print("\nMean word length per class:")
    for lab in range(3):
        sub = word_len[df["label"] == lab]
        if len(sub):
            print(
                "  {:<16} mean={:7.2f}  median={:7.1f}  n={}".format(
                    LABEL_NAMES[lab], sub.mean(), sub.median(), len(sub)
                )
            )

    # ------------------------------------------------------------------
    # Persist audit artifacts
    # ------------------------------------------------------------------
    audit_rows = [
        ("dataset_path", DATA_PATH.relative_to(REPO_ROOT).as_posix()),
        ("text_column", TEXT_COL),
        ("severity_column", SEVERITY_COL),
        ("n_rows_raw", len(raw)),
        ("n_columns_raw", raw.shape[1]),
        ("column_names", "|".join(raw.columns)),
        ("n_content_missing", n_content_na),
        ("n_content_empty_or_whitespace", n_content_empty),
        ("n_severity_missing", int(raw[SEVERITY_COL].isna().sum())),
        ("n_rows_after_dropna", n_after_dropna),
        ("n_rows_removed_by_dropna", len(raw) - n_after_dropna),
        ("n_duplicate_full_rows", n_dup_full_rows),
        ("n_duplicate_content", n_dup_content),
        ("n_unique_content", n_unique_content),
        ("n_duplicate_texts_with_conflicting_labels", conflicting),
        ("char_len_mean", round(float(char_len.mean()), 2)),
        ("char_len_median", float(char_len.median())),
        ("char_len_min", int(char_len.min())),
        ("char_len_max", int(char_len.max())),
        ("word_len_mean", round(float(word_len.mean()), 2)),
        ("word_len_median", float(word_len.median())),
        ("word_len_min", int(word_len.min())),
        ("word_len_max", int(word_len.max())),
    ]
    for lab in range(3):
        audit_rows.append(
            (
                "n_class_{}_{}".format(lab, LABEL_NAMES[lab].replace(" ", "_")),
                int(label_counts.get(lab, 0)),
            )
        )
    for sev, n in sev_counts.items():
        audit_rows.append(("n_severity_{}".format(sev), int(n)))

    audit_df = pd.DataFrame(audit_rows, columns=["metric", "value"])
    audit_df.to_csv(RESULTS_DIR / "dataset_audit.csv", index=False)

    missing_df = pd.DataFrame({"column": missing.index, "n_missing": missing.values})
    missing_df.to_csv(RESULTS_DIR / "missing_values_per_column.csv", index=False)

    stats_df = pd.DataFrame(
        {
            "statistic": [
                "count", "mean", "std", "min", "p25", "median", "p75", "p95", "max",
            ],
            "characters": [
                int(char_len.count()), char_len.mean(), char_len.std(),
                char_len.min(), char_len.quantile(0.25), char_len.median(),
                char_len.quantile(0.75), char_len.quantile(0.95), char_len.max(),
            ],
            "words": [
                int(word_len.count()), word_len.mean(), word_len.std(),
                word_len.min(), word_len.quantile(0.25), word_len.median(),
                word_len.quantile(0.75), word_len.quantile(0.95), word_len.max(),
            ],
        }
    )
    stats_df.to_csv(RESULTS_DIR / "text_statistics.csv", index=False)

    print("\nWrote: experiments/tfidf_baseline/results/dataset_audit.csv")
    print("Wrote: experiments/tfidf_baseline/results/missing_values_per_column.csv")
    print("Wrote: experiments/tfidf_baseline/results/text_statistics.csv")
    print("\nAudit complete.")


if __name__ == "__main__":
    main()
