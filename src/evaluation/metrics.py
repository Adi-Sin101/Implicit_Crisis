"""Evaluation, with the disaggregated reporting the research question needs.

Pooled accuracy is not the result of this project. The result is what happens
on the implicit slice specifically, and how each model's precision holds up
against hard negatives. ``slice_report`` is therefore the primary output and
``overall_report`` the secondary one.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.utils.labels import EVALUATION_SLICES, LABELS


def overall_report(y_true, y_pred, labels=LABELS) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "n": len(y_true),
    }


def per_class_report(y_true, y_pred, labels=LABELS) -> pd.DataFrame:
    rep = classification_report(
        y_true, y_pred, labels=list(labels), output_dict=True, zero_division=0
    )
    rows = [
        {"label": label, **{k: float(v) for k, v in rep[label].items()}}
        for label in labels
        if label in rep
    ]
    return pd.DataFrame(rows)


def slice_report(y_true, y_pred, slices=EVALUATION_SLICES) -> pd.DataFrame:
    """One row per evaluation slice, restricted to items whose gold label is that slice.

    Recall is the meaningful figure for the crisis slices (what fraction of
    genuine cases were caught); precision is the meaningful figure for hard
    negatives (how often crisis vocabulary alone triggered a false positive),
    so both are reported for every slice and read according to the slice.
    """
    y_true = pd.Series(list(y_true))
    y_pred = pd.Series(list(y_pred))
    rows = []
    for slice_label in slices:
        mask = y_true == slice_label
        n = int(mask.sum())
        if n == 0:
            rows.append({"slice": slice_label, "n": 0, "recall": None,
                         "precision": None, "f1": None})
            continue
        true_binary = mask.map({True: slice_label, False: "other"})
        pred_binary = (y_pred == slice_label).map({True: slice_label, False: "other"})
        p, r, f1, _ = precision_recall_fscore_support(
            true_binary,
            pred_binary,
            labels=[slice_label],
            average="binary",
            pos_label=slice_label,
            zero_division=0,
        )
        rows.append(
            {
                "slice": slice_label,
                "n": n,
                "recall": float(r),
                "precision": float(p),
                "f1": float(f1),
                "slice_accuracy": float((y_pred[mask] == slice_label).mean()),
            }
        )
    return pd.DataFrame(rows)


def confusion(y_true, y_pred, labels=LABELS) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    return pd.DataFrame(matrix, index=list(labels), columns=list(labels))


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """Assemble the Section 6 reporting template from measured results only."""
    rows = []
    for model_name, res in results.items():
        row = {"model": model_name, "overall_macro_f1": res["overall"]["macro_f1"],
               "accuracy": res["overall"]["accuracy"]}
        for _, slice_row in res["slices"].iterrows():
            row[f"{slice_row['slice']}_f1"] = slice_row["f1"]
        rows.append(row)
    return pd.DataFrame(rows)
