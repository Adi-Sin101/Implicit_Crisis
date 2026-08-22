"""Inter-annotator agreement.

Implements Cohen's kappa (two annotators), Fleiss' kappa (three or more) and a
per-class breakdown, plus the confusion matrix used to decide which parts of
the guideline need revising.

No agreement figure is ever hard-coded or assumed here; everything is computed
from the annotation files actually present on disk.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from src.utils.labels import LABELS


def align(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Join per-annotator files on ``id``, keeping only items everyone labelled."""
    merged = None
    for name, df in frames.items():
        sub = df[["id", "label"]].rename(columns={"label": name}).dropna()
        merged = sub if merged is None else merged.merge(sub, on="id", how="inner")
    if merged is None or merged.empty:
        raise ValueError("No overlapping annotated items across annotators")
    return merged


def cohens_kappa(a: pd.Series, b: pd.Series, labels=LABELS) -> float:
    labels = list(labels)
    n = len(a)
    observed = float((a.values == b.values).mean())
    pa = a.value_counts(normalize=True)
    pb = b.value_counts(normalize=True)
    expected = float(sum(pa.get(k, 0.0) * pb.get(k, 0.0) for k in labels))
    if np.isclose(expected, 1.0):
        return 1.0
    return (observed - expected) / (1.0 - expected)


def fleiss_kappa(matrix: pd.DataFrame) -> float:
    """``matrix``: one row per item, one column per label, counts of raters."""
    counts = matrix.to_numpy(dtype=float)
    n_items, _ = counts.shape
    n_raters = counts.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Fleiss' kappa requires the same number of raters per item")
    n = n_raters[0]
    p_i = ((counts**2).sum(axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n_items * n)
    p_e = float((p_j**2).sum())
    if np.isclose(p_e, 1.0):
        return 1.0
    return float((p_bar - p_e) / (1.0 - p_e))


def rating_matrix(aligned: pd.DataFrame, annotators, labels=LABELS) -> pd.DataFrame:
    rows = []
    for _, row in aligned.iterrows():
        rows.append([sum(row[a] == label for a in annotators) for label in labels])
    return pd.DataFrame(rows, columns=list(labels), index=aligned["id"])


def pairwise_cohen(aligned: pd.DataFrame, annotators) -> pd.DataFrame:
    out = []
    for a, b in combinations(annotators, 2):
        out.append({"annotator_a": a, "annotator_b": b,
                    "cohens_kappa": cohens_kappa(aligned[a], aligned[b]),
                    "raw_agreement": float((aligned[a] == aligned[b]).mean()),
                    "n_items": len(aligned)})
    return pd.DataFrame(out)


def per_class_agreement(aligned: pd.DataFrame, annotators, labels=LABELS) -> pd.DataFrame:
    """One-vs-rest kappa per class - shows which category the guideline fails on."""
    rows = []
    for label in labels:
        binarised = {a: (aligned[a] == label).map({True: label, False: "other"}) for a in annotators}
        kappas = [
            cohens_kappa(binarised[a], binarised[b], labels=(label, "other"))
            for a, b in combinations(annotators, 2)
        ]
        rows.append({"label": label, "mean_pairwise_kappa": float(np.mean(kappas)) if kappas else np.nan})
    return pd.DataFrame(rows)


def disagreement_matrix(aligned: pd.DataFrame, a: str, b: str, labels=LABELS) -> pd.DataFrame:
    return pd.crosstab(
        aligned[a].astype(pd.CategoricalDtype(list(labels))),
        aligned[b].astype(pd.CategoricalDtype(list(labels))),
        dropna=False,
    )


def interpret(kappa: float) -> str:
    """Landis & Koch (1977) descriptive bands. Descriptive only, not a threshold."""
    if kappa < 0.0:
        return "poor"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"
