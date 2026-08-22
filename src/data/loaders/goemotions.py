"""Loader for GoEmotions (Demszky et al., ACL 2020), Apache-2.0.

The curated TSV splits are tab-separated: text, comma-separated label ids,
comment id - over 27 emotions plus neutral. Used here as a bulk source of
ordinary, non-mental-health text for the non-crisis stratum.
"""
from __future__ import annotations

import pandas as pd

from src.data.loaders.base import normalise
from src.utils.config import dataset_path
from src.utils.io import read_table

SOURCE = "goemotions"
SPLITS = ("goemotions_train", "goemotions_dev", "goemotions_test")


def load_label_names(path=None) -> list[str]:
    path = path or dataset_path("goemotions_labels")
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _decode(ids: str, names: list[str]) -> str:
    return ",".join(names[int(i)] for i in str(ids).split(",") if i.strip().isdigit())


def load(keys=SPLITS) -> pd.DataFrame:
    names = load_label_names()
    frames = []
    for key in keys:
        df = read_table(
            dataset_path(key), header=None, names=["text", "labels", "comment_id"]
        )
        emotions = df["labels"].apply(lambda ids: _decode(ids, names))
        frames.append(
            pd.DataFrame(
                {
                    "source": SOURCE,
                    "src_id": df["comment_id"].astype(str),
                    "text": df["text"],
                    "src_risk": "none",
                    "src_meta": "emotions=" + emotions,
                }
            )
        )
    return normalise(pd.concat(frames, ignore_index=True), SOURCE)
