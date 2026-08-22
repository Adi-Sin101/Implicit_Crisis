"""Loader for IRF - Interpersonal Risk Factors (Garg et al., Findings of ACL 2023).

Columns: ``text``, ``belong`` (thwarted belongingness, 0/1), ``belong_exp``,
``burden`` (perceived burdensomeness, 0/1), ``burden_exp``.

A risk-factor flag is NOT a crisis label. It is carried through as
``src_risk='risk_factor'`` purely to route posts into the implicit stratum.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.base import normalise
from src.utils.config import dataset_path
from src.utils.io import read_table

SOURCE = "irf"
SPLITS = ("irf_train", "irf_val", "irf_test")


def load(paths=None) -> pd.DataFrame:
    paths = paths or [dataset_path(k) for k in SPLITS]
    frames = []
    for split_path in paths:
        df = read_table(split_path)
        split = Path(split_path).stem.split("_")[0]
        any_flag = (df["belong"].fillna(0) + df["burden"].fillna(0)) > 0
        frames.append(
            pd.DataFrame(
                {
                    "source": SOURCE,
                    "src_id": split + "-" + df.iloc[:, 0].astype(str),
                    "text": df["text"],
                    "src_risk": any_flag.map({True: "risk_factor", False: "none"}),
                    "src_meta": (
                        "belong=" + df["belong"].astype(str)
                        + ";burden=" + df["burden"].astype(str)
                        + ";split=" + split
                    ),
                }
            )
        )
    return normalise(pd.concat(frames, ignore_index=True), SOURCE)
