"""Exact and near-duplicate removal.

The source corpora overlap: SDCNL was scraped from the same subreddits as the
Komati corpus, and CAMS re-uses SDCNL posts (its ``IntentSDCNL_*`` files).
Without deduplication the same post could be annotated twice, or leak across
the train/test split.
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

_NORM_RE = re.compile(r"[^a-z0-9 ]+")


def normalise_for_hash(text: str) -> str:
    return _NORM_RE.sub("", str(text).lower()).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalise_for_hash(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    tokens = normalise_for_hash(text).split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def drop_exact_duplicates(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    out = df.copy()
    out["_hash"] = out[text_col].map(text_hash)
    out = out.drop_duplicates(subset="_hash", keep="first").drop(columns="_hash")
    return out.reset_index(drop=True)


def drop_near_duplicates(
    df: pd.DataFrame, text_col: str = "text", threshold: float = 0.9, k: int = 5
) -> pd.DataFrame:
    """Bucket by first shingle, then compare Jaccard within buckets.

    Exact bucketing keeps this tractable on a pool of this size; it will miss
    near-duplicates whose openings differ, which is acceptable because exact
    deduplication has already run.
    """
    kept_rows: list[int] = []
    buckets: dict[str, list[set[str]]] = {}
    for idx, text in df[text_col].items():
        sh = shingles(text, k=k)
        key = next(iter(sorted(sh)), "")
        bucket = buckets.setdefault(key, [])
        if any(jaccard(sh, other) >= threshold for other in bucket):
            continue
        bucket.append(sh)
        kept_rows.append(idx)
    return df.loc[kept_rows].reset_index(drop=True)
