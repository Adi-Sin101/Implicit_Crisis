"""Loader for SDCNL (Haque et al., ICANN 2021).

The combined set holds 1,895 posts with ``title``, ``selftext`` and a binary
``is_suicide`` flag. SDCNL draws a suicide-vs-depression distinction, so its
negatives are topically close to the positives - a useful source of
hard-negative candidates.
"""
from __future__ import annotations

import pandas as pd

from src.data.loaders.base import normalise
from src.utils.config import dataset_path
from src.utils.io import read_table

SOURCE = "sdcnl"


def load(path=None) -> pd.DataFrame:
    path = path or dataset_path("sdcnl_combined")
    df = read_table(path)
    title = df["title"].fillna("").astype(str)
    body = df["selftext"].fillna("").astype(str)
    out = pd.DataFrame(
        {
            "source": SOURCE,
            "src_id": df.index.astype(str),
            "text": (title + "\n\n" + body).str.strip(),
            "src_risk": df["is_suicide"].map({1: "positive", 0: "negative"}).fillna("none"),
            "src_meta": "is_suicide=" + df["is_suicide"].astype(str),
        }
    )
    return normalise(out, SOURCE)
