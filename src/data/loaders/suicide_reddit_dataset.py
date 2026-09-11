"""Loader for the archived ``Suicide Reddit Dataset`` source.

The supplied raw source is a RAR archive containing six CSV files, each with
``title`` and ``usertext`` columns.  The file stem is retained as
``source_label``: it identifies the source collection, not a project label and
not a judgement of the author's risk.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cleaning import dedup
from src.data.loaders.base import normalise
from src.utils.config import dataset_path
from src.utils.io import read_table

SOURCE = "suicide_reddit_dataset"


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract a RAR archive with the system ``tar`` command.

    Windows' bundled bsdtar can read the supplied RAR archive.  Keeping this
    operation in a temporary directory ensures the protected raw archive is
    never unpacked into, or modified within, ``data/raw``.
    """
    tar = shutil.which("tar")
    if tar is None:
        raise RuntimeError("Cannot read Suicide Reddit Dataset: system 'tar' was not found.")
    result = subprocess.run(
        [tar, "-xf", str(archive), "-C", str(destination)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Could not extract {archive.name}: {detail}")


def read_extracted_directory(directory: Path) -> pd.DataFrame:
    """Read extracted CSVs and preserve the archive member name as provenance."""
    frames: list[pd.DataFrame] = []
    for path in sorted(directory.rglob("*.csv")):
        raw = read_table(path)
        required = {"title", "usertext"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
        # Titles are always present in the supplied data.  A blank body is
        # valid, so concatenate rather than discard it.
        text = raw["title"].fillna("").astype(str).str.strip()
        body = raw["usertext"].fillna("").astype(str).str.strip()
        text = (text + "\n\n" + body).str.strip()
        source_label = path.stem
        frames.append(
            pd.DataFrame(
                {
                    "source": SOURCE,
                    "src_id": [f"{source_label}:{i}" for i in raw.index],
                    "text": text,
                    "src_risk": "none",
                    "src_meta": f"source_label={source_label}",
                    "source_label": source_label,
                }
            )
        )
    if not frames:
        raise ValueError(f"No CSV files found in extracted archive directory: {directory}")
    return pd.concat(frames, ignore_index=True)


def load_records(path: Path | str | None = None) -> pd.DataFrame:
    """Load the archive into normalised records plus ``source_label``.

    ``src_risk='none'`` is deliberately neutral: archive member names are
    provenance only and must not create project gold labels or strata claims.
    """
    archive = Path(path or dataset_path("suicide_reddit_dataset"))
    if not archive.exists():
        raise FileNotFoundError(f"Suicide Reddit Dataset archive not found: {archive}")
    with tempfile.TemporaryDirectory(prefix="suicide-reddit-") as temp:
        directory = Path(temp)
        extract_archive(archive, directory)
        return read_extracted_directory(directory)


def inspect_archive(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return a read-only, per-member audit of the supplied archive."""
    archive = Path(path or dataset_path("suicide_reddit_dataset"))
    if not archive.exists():
        raise FileNotFoundError(f"Suicide Reddit Dataset archive not found: {archive}")
    audit: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="suicide-reddit-") as temp:
        directory = Path(temp)
        extract_archive(archive, directory)
        for member in sorted(directory.rglob("*.csv")):
            raw = read_table(member)
            required = {"title", "usertext"}
            missing = required - set(raw.columns)
            if missing:
                raise ValueError(f"{member.name} is missing required columns: {sorted(missing)}")
            joined = (
                raw["title"].fillna("").astype(str).str.strip()
                + "\n\n"
                + raw["usertext"].fillna("").astype(str).str.strip()
            ).str.strip()
            hashes = joined.map(dedup.text_hash)
            audit.append(
                {
                    "member": str(member.relative_to(directory)),
                    "format": "CSV",
                    "rows": len(raw),
                    "columns": list(raw.columns),
                    "missing_values": {column: int(count) for column, count in raw.isna().sum().items()},
                    "empty_title": int(raw["title"].fillna("").astype(str).str.strip().eq("").sum()),
                    "empty_usertext": int(raw["usertext"].fillna("").astype(str).str.strip().eq("").sum()),
                    "exact_duplicate_records_within_member": int(hashes.duplicated().sum()),
                }
            )
    return audit


def load(path: Path | str | None = None) -> pd.DataFrame:
    """Return the standard project loader contract."""
    return normalise(load_records(path), SOURCE)
