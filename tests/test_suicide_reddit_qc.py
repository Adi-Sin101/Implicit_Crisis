from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.qc.suicide_reddit_hard_negative_qc import review  # noqa: E402


def test_qc_statuses_are_not_gold_labels_and_flags_direct_self_risk():
    frame = pd.DataFrame(
        {
            "text": [
                "I am planning to kill myself tonight.",
                "This article reports suicide prevention statistics.",
                "I called a suicide hotline after feeling low.",
                "Suicide is mentioned here without enough context.",
            ],
            "source_id": ["a", "b", "c", "d"],
            "source": ["suicide_reddit_dataset"] * 4,
            "source_label": ["AskReddit"] * 4,
            "matched_terms": ["kill myself", "suicid", "suicid", "suicid"],
            "candidate_reason": ["prevention", "news", "prevention", "crisis_term_context"],
        }
    )
    reviewed, counts = review(frame)
    assert reviewed.loc[0, "qc_status"] == "obvious_current_self_crisis"
    assert reviewed.loc[1, "qc_status"] == "strong_hard_negative_candidate"
    assert reviewed.loc[2, "qc_status"] == "borderline_needs_annotator_judgment"
    assert reviewed.loc[3, "qc_status"] == "borderline_needs_annotator_judgment"
    assert "label" not in reviewed.columns
    assert counts["obvious_current_self_crisis"] == 1


def test_qc_marks_near_duplicates_separately():
    frame = pd.DataFrame(
        {
            "text": [
                "A news article discussed suicide prevention statistics for schools today.",
                "A news article discussed suicide prevention statistics for schools today!",
            ],
            "source_id": ["a", "b"], "source": ["x", "x"],
            "source_label": ["x", "x"], "matched_terms": ["suicid", "suicid"],
            "candidate_reason": ["news", "news"],
        }
    )
    reviewed, counts = review(frame)
    assert reviewed.loc[1, "qc_status"] == "duplicate_or_near_duplicate"
    assert reviewed.loc[1, "qc_duplicate_of_source_id"] == "a"
    assert counts["duplicate_or_near_duplicate"] == 1
