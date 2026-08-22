"""Annotator-facing sheets and the key that maps them back to the pool.

The sheet an annotator opens carries only ``id``, ``text``, and the empty
columns they fill in. Stratum, source label and lexicon hit stay behind in the
key file: showing an annotator that a post was routed as "likely implicit", or
that it tripped the crisis lexicon, would anchor the judgement the annotation
is supposed to produce independently.
"""
from __future__ import annotations

import pandas as pd

from src.utils.labels import ANNOTATOR_COLUMNS, LABELS

BLIND_COLUMNS = ("stratum", "src_risk", "explicit_lex", "explicit_terms", "src_meta", "source")


def make_sheet(candidates: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Blind, shuffled sheet with empty ``label`` / ``confidence`` / ``notes``."""
    missing = [c for c in ("id", "text") if c not in candidates.columns]
    if missing:
        raise ValueError(f"Candidates are missing columns: {missing}")
    sheet = candidates[["id", "text"]].sample(frac=1.0, random_state=seed).copy()
    sheet["label"] = ""
    sheet["confidence"] = ""
    sheet["notes"] = ""
    return sheet[list(ANNOTATOR_COLUMNS)].reset_index(drop=True)


def make_key(candidates: pd.DataFrame) -> pd.DataFrame:
    """The withheld metadata, kept separately for post-hoc analysis only."""
    cols = ["id"] + [c for c in BLIND_COLUMNS if c in candidates.columns]
    return candidates[cols].reset_index(drop=True)


def assert_blind(sheet: pd.DataFrame) -> None:
    leaked = [c for c in BLIND_COLUMNS if c in sheet.columns]
    if leaked:
        raise AssertionError(f"Annotation sheet leaks biasing columns: {leaked}")


def label_options() -> str:
    return " | ".join(LABELS)
