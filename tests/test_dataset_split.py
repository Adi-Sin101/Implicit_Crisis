from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data.create_dataset_split import COLUMNS, create_artifacts, read_dataset  # noqa: E402


def make_dataset() -> pd.DataFrame:
    rows = []
    for label in ("explicit_crisis", "implicit_crisis", "hard_negative", "non_crisis"):
        for number in range(10):
            rows.append({
                "id": f"{label}-{number}",
                "text": f"Exact text {label} {number}.",
                "label": label,
                "confidence": "high",
                "notes": "preserved note",
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def test_canonical_split_is_deterministic_and_preserves_records():
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary_dir:
        tmp_path = Path(temporary_dir)
        source = make_dataset()
        input_path = tmp_path / "gold.csv"
        source.to_csv(input_path, index=False, encoding="utf-8")
        first = create_artifacts(input_path, tmp_path / "splits_a", tmp_path / "profile_a.json")
        second = create_artifacts(input_path, tmp_path / "splits_b", tmp_path / "profile_b.json")

        assert first["actual_split_counts"] == {"train": 28, "validation": 6, "test": 6}
        assert first["split_assignment_sha256"] == second["split_assignment_sha256"]
        assert first["validation"]["passed"]

        combined = pd.concat([
            read_dataset(tmp_path / "splits_a" / f"{name}.csv")
            for name in ("train", "validation", "test")
        ], ignore_index=True)
        assert list(combined.columns) == COLUMNS
        assert len(combined) == len(source)
        assert combined["id"].nunique() == len(source)
        assert set(combined["id"]) == set(source["id"])
        assert dict(zip(combined["id"], combined["text"])) == dict(zip(source["id"], source["text"]))
        for name in ("train", "validation", "test"):
            assert set(read_dataset(tmp_path / "splits_a" / f"{name}.csv")["label"]) == set(source["label"])

        profile = json.loads((tmp_path / "profile_a.json").read_text(encoding="utf-8"))
        assert profile["total_records"] == 40
        assert profile["duplicate_id_count"] == 0
