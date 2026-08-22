"""Validation of a completed annotation file / gold dataset."""
from __future__ import annotations

import pandas as pd

from src.utils.labels import LABELS

REQUIRED = ("id", "text", "label")


def validate(df: pd.DataFrame, strict: bool = True) -> list[str]:
    """Return a list of problems. Empty list means the file is well-formed."""
    problems: list[str] = []

    for col in REQUIRED:
        if col not in df.columns:
            problems.append(f"missing required column: {col}")
    if problems:
        return problems

    blank = df["label"].isna() | (df["label"].astype(str).str.strip() == "")
    if blank.any():
        problems.append(f"{int(blank.sum())} rows have no label")

    bad = sorted(set(df.loc[~blank, "label"].astype(str).str.strip()) - set(LABELS))
    if bad:
        problems.append(f"unrecognised labels: {bad} (allowed: {list(LABELS)})")

    dupes = df["id"].duplicated().sum()
    if dupes:
        problems.append(f"{int(dupes)} duplicate ids")

    empty_text = df["text"].astype(str).str.strip().eq("").sum()
    if empty_text:
        problems.append(f"{int(empty_text)} rows have empty text")

    if strict:
        counts = df.loc[~blank, "label"].value_counts()
        for label in LABELS:
            if counts.get(label, 0) == 0:
                problems.append(f"class '{label}' has no examples")

    return problems


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df["label"].value_counts().rename("n").rename_axis("label").reset_index()
    )
