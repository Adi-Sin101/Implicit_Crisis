from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.integration.suicide_reddit_qc import append_additions, duplicate_report, prepare_retained  # noqa: E402


def test_prepare_and_append_retained_qc_candidates_without_labels():
    qc = pd.DataFrame({
        "text": ["A report discussed suicide prevention statistics 0."],
        "source": ["suicide_reddit_dataset"], "source_id": ["AskReddit:1"],
        "source_label": ["AskReddit"], "matched_terms": ["suicid"],
        "candidate_reason": ["research"], "qc_status": ["strong_hard_negative_candidate"],
    })
    # The production guard requires the expected 105 retained records.
    qc = pd.concat([qc] * 105, ignore_index=True)
    qc["source_id"] = [f"AskReddit:{i}" for i in range(105)]
    qc["text"] = [f"A report discussed suicide prevention statistics {i}." for i in range(105)]
    additions = prepare_retained(qc)
    existing = pd.DataFrame(columns=additions.columns)
    duplicates, counts = duplicate_report(existing, additions)
    # Identical text is correctly surfaced rather than silently appended.
    assert len(duplicates) == 0
    assert counts["genuinely_new"] == 105
    combined = append_additions(existing, additions)
    assert len(combined) == 105
    assert "label" not in combined.columns
    assert set(combined["stratum"]) == {"C_hard_negative"}
