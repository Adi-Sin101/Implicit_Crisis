"""Loader for CAMS - Causal Analysis of Mental health issues (Garg et al., LREC 2022).

CAMS annotates the *cause* of a mental-health issue (six categories plus an
explanation span), not a risk level. Its files use inconsistent column names
across releases, which is handled below.

The numeric category codes are carried through verbatim in ``src_meta``; their
meaning must be verified against the paper before being reported in writing
(see DATASETS.md).
"""
from __future__ import annotations

import pandas as pd

from src.data.loaders.base import normalise
from src.utils.config import dataset_path
from src.utils.io import read_table

SOURCE = "cams"
FILES = ("cams_main", "cams_added", "cams_intent_train", "cams_intent_test")

_TEXT_COLS = ("text", "selftext")
_LABEL_COLS = ("category", "cause", "ANNOTATIONS")


def _pick(df: pd.DataFrame, candidates) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"None of {candidates} present in {list(df.columns)}")


def load(keys=FILES) -> pd.DataFrame:
    frames = []
    for key in keys:
        df = read_table(dataset_path(key))
        text_col = _pick(df, _TEXT_COLS)
        label_col = _pick(df, _LABEL_COLS)
        frames.append(
            pd.DataFrame(
                {
                    "source": SOURCE,
                    "src_id": key + "-" + df.index.astype(str),
                    "text": df[text_col],
                    "src_risk": "cause",
                    "src_meta": f"file={key};cause=" + df[label_col].astype(str),
                }
            )
        )
    return normalise(pd.concat(frames, ignore_index=True), SOURCE)
