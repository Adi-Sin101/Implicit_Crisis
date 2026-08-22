"""Loader for the Komati 'Suicide and Depression Detection' Reddit corpus.

Columns in the released CSV: ``Unnamed: 0``, ``text``, ``class``
(``suicide`` / ``non-suicide``), ~232k rows, labels derived from the
subreddit of origin (distant supervision).
"""
from __future__ import annotations

import pandas as pd

from src.data.loaders.base import normalise
from src.utils.config import dataset_path
from src.utils.io import read_table

SOURCE = "komati"


def load(path=None, nrows: int | None = None) -> pd.DataFrame:
    path = path or dataset_path("komati")
    df = read_table(path, nrows=nrows)
    out = pd.DataFrame(
        {
            "source": SOURCE,
            "src_id": df.iloc[:, 0].astype(str),
            "text": df["text"],
            "src_risk": df["class"].map({"suicide": "positive", "non-suicide": "negative"}),
            "src_meta": "class=" + df["class"].astype(str),
        }
    )
    return normalise(out, SOURCE)
