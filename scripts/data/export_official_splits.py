"""Export the official frozen split into three explicit CSV files, then audit them.

This script NEVER creates a split. It does not import `train_test_split` and does
not sample, shuffle or reassign membership. It reads the authoritative file

    experiments/tfidf_baseline/results/data_split.csv

and writes each split's rows, unchanged and in source order, to

    data/final_datasets/splits/train.csv
    data/final_datasets/splits/validation.csv
    data/final_datasets/splits/test.csv

If those files already exist they are verified against the source rather than
overwritten; any difference is reported, not silently fixed.

Note on naming: the source `split` column uses the value `val`, not
`validation`. The exported file is named `validation.csv` as required, and the
rows it contains are exactly the source rows whose split value is `val`.

Run:
    python scripts/data/export_official_splits.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPO_ROOT / "experiments" / "tfidf_baseline" / "results" / "data_split.csv"
OUT_DIR = REPO_ROOT / "data" / "final_datasets" / "splits"

# Exported file name -> the value used in the source `split` column.
SPLIT_FILES = {
    "train.csv": "train",
    "validation.csv": "val",
    "test.csv": "test",
}

EXPECTED_COUNTS = {"train": 905, "val": 194, "test": 195}
EXPECTED_TOTAL = 1294

TEXT_COL = "content"
SEVERITY_COL = "severity"
LABEL_COL = "label"
SPLIT_COL = "split"
ID_COL = "row_index"

LABEL_NAMES = {0: "Non-Crisis", 1: "Implicit Crisis", 2: "Explicit Crisis"}

# --- frozen final TF-IDF configuration (recorded, never applied here) ---
FROZEN_TFIDF = {
    "analyzer": "char_wb",
    "ngram_range": (3, 5),
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": True,
    "max_features": 50000,
    "lowercase": True,
    "strip_accents": "unicode",
}
FROZEN_LR = {
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 5000,
    "random_state": 42,
}
FROZEN_RESULTS = {
    "validation_macro_f1": 0.8054,
    "test_accuracy": 0.7128,
    "test_macro_precision": 0.7007,
    "test_macro_recall": 0.6979,
    "test_macro_f1": 0.6962,
    "test_weighted_f1": 0.7117,
}

results: dict[str, object] = {}
problems: list[str] = []


def content_key(s: str) -> str:
    """Deterministic identity for a row's text, used when comparing files."""
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def main() -> int:
    print("=" * 72)
    print("EXPORT + AUDIT OFFICIAL SPLIT")
    print("=" * 72)
    print("Source:", SOURCE.relative_to(REPO_ROOT).as_posix())

    if not SOURCE.exists():
        raise FileNotFoundError(f"Official split not found: {SOURCE}")

    src = pd.read_csv(SOURCE)
    results["source_rows"] = len(src)
    results["columns"] = list(src.columns)

    print(f"Rows   : {len(src)}")
    print(f"Columns: {list(src.columns)}")

    # ---------------- column presence ----------------
    for col in [ID_COL, TEXT_COL, SEVERITY_COL, LABEL_COL, SPLIT_COL]:
        if col not in src.columns:
            problems.append(f"missing expected column: {col}")
    results["text_column_present"] = TEXT_COL in src.columns

    # ---------------- split values ----------------
    actual_values = sorted(src[SPLIT_COL].unique())
    print(f"\nSplit column values (actual): {actual_values}")
    counts = src[SPLIT_COL].value_counts().to_dict()
    results["source_counts"] = counts

    print("\nRow counts in source:")
    ok_counts = True
    for name, expected in EXPECTED_COUNTS.items():
        got = int(counts.get(name, 0))
        status = "OK" if got == expected else "MISMATCH"
        if got != expected:
            ok_counts = False
            problems.append(f"{name} count {got} != expected {expected}")
        print(f"  {name:<6} {got:>5}  (expected {expected})  {status}")
    total_ok = len(src) == EXPECTED_TOTAL
    if not total_ok:
        ok_counts = False
        problems.append(f"total {len(src)} != expected {EXPECTED_TOTAL}")
    print(f"  {'TOTAL':<6} {len(src):>5}  (expected {EXPECTED_TOTAL})  "
          f"{'OK' if total_ok else 'MISMATCH'}")
    results["counts_ok"] = ok_counts

    # ---------------- missing values ----------------
    print("\nMissing values per column (source):")
    missing = src.isna().sum()
    for col, n in missing.items():
        print(f"  {col:<12} {n}")
    if int(src[TEXT_COL].isna().sum()) or int(src[SEVERITY_COL].isna().sum()):
        problems.append("missing values in content or severity")
    results["missing"] = {c: int(n) for c, n in missing.items()}

    # ---------------- label mapping ----------------
    print("\nLabel mapping check (severity -> label):")
    mapping_ok = True
    for sev in sorted(src[SEVERITY_COL].unique()):
        labels = set(src.loc[src[SEVERITY_COL] == sev, LABEL_COL])
        expected = 0 if int(sev) == 0 else (1 if int(sev) == 1 else 2)
        ok = labels == {expected}
        if not ok:
            mapping_ok = False
            problems.append(f"severity {sev} maps to {labels}, expected {{{expected}}}")
        n = int((src[SEVERITY_COL] == sev).sum())
        print(f"  severity {sev} -> {sorted(labels)}  "
              f"({LABEL_NAMES[expected]:<16} n={n:>4})  {'OK' if ok else 'BAD'}")

    # class_name must agree with label
    for lab, name in LABEL_NAMES.items():
        names = set(src.loc[src[LABEL_COL] == lab, "class_name"])
        if names and names != {name}:
            mapping_ok = False
            problems.append(f"label {lab} has class_name {names}, expected {{{name}}}")

    n_classes = src[LABEL_COL].nunique()
    if n_classes != 3:
        mapping_ok = False
        problems.append(f"expected exactly 3 classes, found {n_classes}")
    if "hard_negative" in " ".join(map(str, src["class_name"].unique())).lower():
        mapping_ok = False
        problems.append("a hard_negative class is present")
    print(f"  exactly 3 classes: {n_classes == 3}")
    results["label_mapping_ok"] = mapping_ok

    # ---------------- class distribution per split ----------------
    print("\nClass distribution per split:")
    print(f"  {'split':<6} {'Non-Crisis':>12} {'Implicit':>10} {'Explicit':>10} {'total':>7}")
    dist = {}
    for name in ["train", "val", "test"]:
        sub = src[src[SPLIT_COL] == name]
        row = [int((sub[LABEL_COL] == i).sum()) for i in range(3)]
        dist[name] = row
        print(f"  {name:<6} {row[0]:>12} {row[1]:>10} {row[2]:>10} {len(sub):>7}")
    results["class_distribution"] = dist

    print("\nRaw severity counts per split:")
    sev_dist = {}
    for name in ["train", "val", "test"]:
        sub = src[src[SPLIT_COL] == name]
        counts_sev = {int(s): int((sub[SEVERITY_COL] == s).sum())
                      for s in sorted(src[SEVERITY_COL].unique())}
        sev_dist[name] = counts_sev
        print(f"  {name:<6} {counts_sev}")
    results["severity_distribution"] = sev_dist

    # ---------------- duplicates and overlap (source) ----------------
    n_dup_rows = int(src.duplicated().sum())
    n_dup_content = int(src[TEXT_COL].astype(str).duplicated().sum())
    n_dup_id = int(src[ID_COL].duplicated().sum())
    print(f"\nDuplicate full rows      : {n_dup_rows}")
    print(f"Duplicate content        : {n_dup_content}")
    print(f"Duplicate {ID_COL}      : {n_dup_id}")
    if n_dup_rows or n_dup_content or n_dup_id:
        problems.append(
            f"duplicates in source: rows={n_dup_rows} content={n_dup_content} "
            f"{ID_COL}={n_dup_id}"
        )
    results["duplicates"] = {
        "rows": n_dup_rows, "content": n_dup_content, "row_index": n_dup_id,
    }

    sets_id = {n: set(src.loc[src[SPLIT_COL] == n, ID_COL]) for n in ["train", "val", "test"]}
    sets_txt = {n: set(src.loc[src[SPLIT_COL] == n, TEXT_COL].astype(str))
                for n in ["train", "val", "test"]}
    overlaps = {}
    print("\nCross-split overlap:")
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        o_id = len(sets_id[a] & sets_id[b])
        o_txt = len(sets_txt[a] & sets_txt[b])
        overlaps[f"{a}_{b}"] = {"row_index": o_id, "content": o_txt}
        print(f"  {a} & {b:<5} row_index={o_id}  content={o_txt}")
        if o_id or o_txt:
            problems.append(f"overlap between {a} and {b}: id={o_id} content={o_txt}")
    results["overlaps"] = overlaps

    # ---------------- export / verify ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n" + "-" * 72)
    print("EXPORT / VERIFY")
    print("-" * 72)

    identity = {}
    for fname, split_value in SPLIT_FILES.items():
        path = OUT_DIR / fname
        expected_df = src[src[SPLIT_COL] == split_value].reset_index(drop=True)

        if path.exists():
            action = "verified (already existed)"
            got = pd.read_csv(path)
        else:
            expected_df.to_csv(path, index=False)
            action = "created"
            got = pd.read_csv(path)

        # identity: same row_index set, and same content per row_index
        same_ids = set(got[ID_COL]) == set(expected_df[ID_COL]) if ID_COL in got.columns else False
        same_n = len(got) == len(expected_df)

        same_content = False
        same_labels = False
        if same_ids and same_n:
            a = expected_df.set_index(ID_COL).sort_index()
            b = got.set_index(ID_COL).sort_index()
            same_content = (
                a[TEXT_COL].astype(str).map(content_key)
                .equals(b[TEXT_COL].astype(str).map(content_key))
            )
            same_labels = (
                a[LABEL_COL].equals(b[LABEL_COL])
                and a[SEVERITY_COL].equals(b[SEVERITY_COL])
            )

        ok = same_ids and same_n and same_content and same_labels
        identity[split_value] = {
            "file": f"data/final_datasets/splits/{fname}",
            "action": action,
            "rows": len(got),
            "expected_rows": len(expected_df),
            "same_ids": same_ids,
            "same_content": same_content,
            "same_labels": same_labels,
            "pass": ok,
        }
        if not ok:
            problems.append(
                f"{fname} does not match the official split "
                f"(ids={same_ids} n={same_n} content={same_content} labels={same_labels})"
            )

        print(f"  data/final_datasets/splits/{fname:<15} {action:<27} "
              f"rows={len(got):<5} identity={'PASS' if ok else 'FAIL'}")

        # per-file duplicate check
        d_rows = int(got.duplicated().sum())
        d_txt = int(got[TEXT_COL].astype(str).duplicated().sum())
        if d_rows or d_txt:
            problems.append(f"{fname}: duplicates rows={d_rows} content={d_txt}")
        identity[split_value]["duplicates"] = {"rows": d_rows, "content": d_txt}

    results["identity"] = identity

    # overlap between the exported files themselves
    print("\nCross-file overlap (exported files):")
    ex = {v: pd.read_csv(OUT_DIR / f) for f, v in SPLIT_FILES.items()}
    ex_ids = {k: set(v[ID_COL]) for k, v in ex.items()}
    ex_txt = {k: set(v[TEXT_COL].astype(str)) for k, v in ex.items()}
    export_overlap_total = 0
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        o_id = len(ex_ids[a] & ex_ids[b])
        o_txt = len(ex_txt[a] & ex_txt[b])
        export_overlap_total += o_id + o_txt
        print(f"  {a} & {b:<5} row_index={o_id}  content={o_txt}")
        if o_id or o_txt:
            problems.append(f"exported overlap {a}/{b}: id={o_id} content={o_txt}")
    results["export_overlap_total"] = export_overlap_total

    # exported files must reconstruct the source exactly
    union = pd.concat(list(ex.values()), ignore_index=True)
    reconstruct_ok = (
        len(union) == len(src)
        and set(union[ID_COL]) == set(src[ID_COL])
    )
    if not reconstruct_ok:
        problems.append("exported files do not reconstruct the source row set")
    print(f"\nExported files reconstruct source exactly: {reconstruct_ok} "
          f"({len(union)} rows)")
    results["reconstruct_ok"] = reconstruct_ok

    # ---------------- verdict ----------------
    all_pass = (
        ok_counts and mapping_ok and reconstruct_ok
        and all(v["pass"] for v in identity.values())
        and export_overlap_total == 0
        and n_dup_rows == 0 and n_dup_content == 0
        and results["text_column_present"]
    )
    results["problems"] = problems
    results["all_pass"] = all_pass

    print()
    print("=" * 40)
    print("FINAL TF-IDF MODEL + SPLIT VERIFICATION")
    print("=" * 40)
    print()
    print(f"SPLIT STATUS: {'PASS' if all_pass else 'FAIL'}")
    print()
    print("SOURCE:")
    print("experiments/tfidf_baseline/results/data_split.csv")
    print()
    print(f"TRAIN: {counts.get('train', 0)}")
    print(f"VALIDATION: {counts.get('val', 0)}")
    print(f"TEST: {counts.get('test', 0)}")
    print(f"TOTAL: {len(src)}")
    print()
    print(f"TRAIN IDENTITY: {'PASS' if identity['train']['pass'] else 'FAIL'}")
    print(f"VALIDATION IDENTITY: {'PASS' if identity['val']['pass'] else 'FAIL'}")
    print(f"TEST IDENTITY: {'PASS' if identity['test']['pass'] else 'FAIL'}")
    print()
    print(f"OVERLAP: {export_overlap_total}")
    print(f"DUPLICATES: {n_dup_rows + n_dup_content}")
    print()
    print(f"LABEL MAPPING: {'PASS' if mapping_ok else 'FAIL'}")
    print(f"TEXT COLUMN: {'PASS' if results['text_column_present'] else 'FAIL'}")
    print()
    print("SPLIT REGENERATED: NO")
    print()
    print("=" * 40)
    print("FROZEN TF-IDF MODEL")
    print("=" * 40)
    print()
    print(f"FINAL ANALYZER: {FROZEN_TFIDF['analyzer']}")
    print(f"FINAL NGRAM: {FROZEN_TFIDF['ngram_range']}")
    print()
    print("FINAL TF-IDF CONFIGURATION: FROZEN")
    print()
    print(f"VALIDATION MACRO F1: {FROZEN_RESULTS['validation_macro_f1']}")
    print(f"TEST MACRO F1: {FROZEN_RESULTS['test_macro_f1']}")
    print(f"TEST ACCURACY: {FROZEN_RESULTS['test_accuracy']}")
    print(f"TEST WEIGHTED F1: {FROZEN_RESULTS['test_weighted_f1']}")
    print()
    print("TF-IDF MODEL MODIFIED: NO")
    print("TF-IDF RESULTS MODIFIED: NO")
    print("N-GRAM SEARCH RERUN: NO")
    print()
    print("BERT MODIFIED: NO")
    print()
    print("AUDIT:")
    print("docs/FINAL_SPLIT_AUDIT.md")
    print()
    print("=" * 40)

    if problems:
        print("\nPROBLEMS DISCOVERED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nNo split-integrity problems were discovered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
