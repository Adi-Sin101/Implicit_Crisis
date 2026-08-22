"""Small IO helpers shared by the pipeline scripts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: Path | str, **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV/Parquet file, tolerating the mixed encodings in the raw corpora."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix in {".tsv", ".tab"}:
        kwargs.setdefault("sep", "\t")
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Could not decode {path}")


def write_table(df: pd.DataFrame, path: Path | str, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False, **kwargs)
    else:
        df.to_csv(path, index=False, encoding="utf-8", **kwargs)
    return path
