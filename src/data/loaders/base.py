"""Common contract for dataset-specific loaders.

Every loader returns a DataFrame with the same normalised columns so that the
corpora can be concatenated into a single candidate pool:

    source    str   dataset identifier ('komati', 'irf', 'sdcnl', 'cams', 'goemotions')
    src_id    str   identifier within the source dataset
    text      str   the post text, unmodified apart from field concatenation
    src_risk  str   the source dataset's own risk signal, verbatim and untranslated:
                    'positive' | 'negative' | 'risk_factor' | 'cause' | 'none'
    src_meta  str   any extra source annotation, kept for provenance only

``src_risk`` is a candidate signal, never a gold label.
"""
from __future__ import annotations

import pandas as pd

NORMALISED_COLUMNS = ["source", "src_id", "text", "src_risk", "src_meta"]


def normalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate and order the normalised frame produced by a loader."""
    missing = [c for c in NORMALISED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{source} loader is missing columns: {missing}")
    out = df[NORMALISED_COLUMNS].copy()
    out["source"] = source
    out["src_id"] = out["src_id"].astype(str)
    out["text"] = out["text"].astype(str)
    return out.reset_index(drop=True)
