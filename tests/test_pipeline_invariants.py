"""Tests for the invariants that keep the gold labels honest.

These are not model tests. They guard the two rules the whole study depends on:
a source label never becomes a gold label, and the annotator never sees the
signals that would anchor their judgement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation import sheets, validate  # noqa: E402
from src.data.cleaning import dedup, text_cleaning  # noqa: E402
from src.data.pooling import lexicon, strata  # noqa: E402
from src.utils.labels import ANNOTATOR_COLUMNS, LABELS, STRATA  # noqa: E402


def test_strata_and_labels_are_disjoint_vocabularies():
    """A stratum name must never be usable as a label, or vice versa."""
    assert set(STRATA).isdisjoint(set(LABELS))


def test_stratum_assignment():
    assert strata.assign_stratum("positive", True) == "A_likely_explicit"
    assert strata.assign_stratum("positive", False) == "B_likely_implicit"
    assert strata.assign_stratum("risk_factor", False) == "B_likely_implicit"
    assert strata.assign_stratum("negative", True) == "C_hard_negative"
    assert strata.assign_stratum("negative", False) == "D_likely_noncrisis"
    assert strata.assign_stratum("none", False) == "D_likely_noncrisis"


def test_annotation_sheet_hides_biasing_columns():
    candidates = pd.DataFrame(
        {
            "id": ["a", "b"],
            "text": ["some post text", "another post"],
            "source": ["sdcnl", "irf"],
            "src_risk": ["positive", "risk_factor"],
            "stratum": ["A_likely_explicit", "B_likely_implicit"],
            "explicit_lex": [True, False],
        }
    )
    sheet = sheets.make_sheet(candidates)
    assert list(sheet.columns) == list(ANNOTATOR_COLUMNS)
    for column in ("stratum", "src_risk", "explicit_lex", "source"):
        assert column not in sheet.columns
    sheets.assert_blind(sheet)


def test_key_retains_metadata_for_later_analysis():
    candidates = pd.DataFrame(
        {"id": ["a"], "text": ["t"], "stratum": ["C_hard_negative"],
         "src_risk": ["negative"], "explicit_lex": [True]}
    )
    key = sheets.make_key(candidates)
    assert set(key.columns) >= {"id", "stratum", "src_risk", "explicit_lex"}


def test_lexicon_flags_explicit_terms():
    assert lexicon.has_explicit_term("I want to kill myself tonight")
    assert lexicon.has_explicit_term("an article about suicide prevention")
    assert not lexicon.has_explicit_term("everyone would be better off without me")


def test_lexicon_cannot_separate_crisis_from_discussion():
    """Documents why the detector must never assign a label."""
    crisis = "I am going to kill myself"
    discussion = "the documentary discussed people who kill myself is a phrase from"
    assert lexicon.has_explicit_term(crisis)
    assert lexicon.has_explicit_term(discussion)


def test_cleaning_preserves_text_and_filters_length():
    df = pd.DataFrame({"text": ["one two three four five six", "short", "[removed]"]})
    out = text_cleaning.clean_frame(df, min_words=5, max_words=400)
    assert len(out) == 1
    assert out.iloc[0]["text"] == "one two three four five six"
    assert out.iloc[0]["n_words"] == 6


def test_url_masking_keeps_the_sentence():
    cleaned = text_cleaning.clean_text("see https://example.com/x for more")
    assert cleaned == "see <URL> for more"


def test_exact_dedup_ignores_case_and_punctuation():
    df = pd.DataFrame({"text": ["I feel awful.", "i feel awful", "something else"]})
    assert len(dedup.drop_exact_duplicates(df)) == 2


def test_validate_rejects_unknown_labels():
    df = pd.DataFrame({"id": ["a"], "text": ["t"], "label": ["B_likely_implicit"]})
    problems = validate.validate(df, strict=False)
    assert any("unrecognised labels" in p for p in problems)


def test_validate_rejects_blank_and_duplicate():
    df = pd.DataFrame(
        {"id": ["a", "a"], "text": ["t", "t"], "label": ["non_crisis", ""]}
    )
    problems = validate.validate(df, strict=False)
    assert any("no label" in p for p in problems)
    assert any("duplicate ids" in p for p in problems)


def test_validate_accepts_a_well_formed_gold_file():
    df = pd.DataFrame(
        {"id": list("abcd"), "text": ["w", "x", "y", "z"], "label": list(LABELS)}
    )
    assert validate.validate(df, strict=True) == []


def test_sampling_never_writes_a_label_column():
    pool = pd.DataFrame(
        {
            "id": [str(i) for i in range(20)],
            "text": [f"post {i}" for i in range(20)],
            "source": ["sdcnl"] * 20,
            "stratum": ["B_likely_implicit"] * 20,
        }
    )
    sample = strata.sample_strata(pool, {"B_likely_implicit": 5}, seed=1)
    assert len(sample) == 5
    assert "label" not in sample.columns


@pytest.mark.parametrize("label", LABELS)
def test_label_vocabulary_is_snake_case(label):
    assert label == label.lower()
    assert " " not in label
