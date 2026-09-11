from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.extraction.suicide_reddit_hard_negatives import (  # noqa: E402
    candidate_reason,
    extract_candidates,
    is_obvious_self_risk,
)
from src.data.loaders.suicide_reddit_dataset import read_extracted_directory  # noqa: E402


def test_obvious_self_risk_is_excluded_but_prevention_is_not():
    assert is_obvious_self_risk("I am planning to kill myself tonight")
    assert not is_obvious_self_risk("This article explains suicide prevention resources")


def test_candidate_reason_is_routing_metadata_not_a_label():
    assert candidate_reason("New suicide prevention hotline campaign") == "prevention"
    assert candidate_reason("A movie quote mentions suicide") == "quote_or_fiction"


def test_extractor_deduplicates_against_existing_data_and_omits_label():
    records = pd.DataFrame(
        {
            "text": [
                "An article about suicide prevention resources for communities.",
                "An article about suicide prevention resources for communities!",
                "I am planning to kill myself tonight.",
            ],
            "source": ["suicide_reddit_dataset"] * 3,
            "src_id": ["x:0", "x:1", "x:2"],
            "source_label": ["AskReddit"] * 3,
            "src_risk": ["none"] * 3,
            "src_meta": ["source_label=AskReddit"] * 3,
        }
    )
    candidates, stats = extract_candidates(records, limit=10)
    assert len(candidates) == 1
    assert candidates.iloc[0]["candidate_reason"] == "prevention"
    assert candidates.iloc[0]["source_id"] == "x:0"
    assert "label" not in candidates.columns
    assert stats["within_source_duplicates_removed"] == 1
    assert stats["obvious_self_risk_excluded"] == 1


def test_archive_member_loader_preserves_member_name():
    # Avoid pytest's shared temporary base: it may be unavailable on locked
    # Windows profiles even though Python's per-test temporary directory works.
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        extracted = root / "data"
        extracted.mkdir()
        pd.DataFrame({"title": ["Title"], "usertext": ["Body text"]}).to_csv(
            extracted / "Jokes.csv", index=False
        )
        records = read_extracted_directory(root)
    assert records.iloc[0]["source_label"] == "Jokes"
    assert records.iloc[0]["src_id"] == "Jokes:0"
