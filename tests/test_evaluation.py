"""Tests for the disaggregated evaluation, which is the project's actual result."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotation import agreement  # noqa: E402
from src.evaluation import error_analysis, metrics  # noqa: E402


def test_slice_report_covers_all_three_slices():
    y_true = ["explicit_crisis", "implicit_crisis", "hard_negative", "non_crisis"]
    y_pred = ["explicit_crisis", "non_crisis", "implicit_crisis", "non_crisis"]
    report = metrics.slice_report(y_true, y_pred)
    assert set(report["slice"]) == {"explicit_crisis", "implicit_crisis", "hard_negative"}


def test_slice_recall_is_computed_over_gold_items_only():
    y_true = ["implicit_crisis"] * 4
    y_pred = ["implicit_crisis", "implicit_crisis", "non_crisis", "non_crisis"]
    row = metrics.slice_report(y_true, y_pred, slices=("implicit_crisis",)).iloc[0]
    assert row["n"] == 4
    assert row["recall"] == 0.5


def test_hard_negative_false_positives_are_isolated():
    frame = error_analysis.build_frame(
        ["a", "b", "c"],
        ["hard_negative", "hard_negative", "non_crisis"],
        {"tfidf": ["explicit_crisis", "hard_negative", "non_crisis"]},
    )
    fps = error_analysis.hard_negative_false_positives(frame, "tfidf")
    assert len(fps) == 1
    assert fps.iloc[0]["text"] == "a"


def test_missed_implicit_is_isolated():
    frame = error_analysis.build_frame(
        ["a", "b"],
        ["implicit_crisis", "implicit_crisis"],
        {"bert": ["implicit_crisis", "non_crisis"]},
    )
    assert len(error_analysis.missed_implicit(frame, "bert")) == 1


def test_disagreement_buckets_partition_the_test_set():
    frame = error_analysis.build_frame(
        list("abcd"),
        ["non_crisis"] * 4,
        {"tfidf": ["non_crisis", "non_crisis", "hard_negative", "hard_negative"],
         "bert": ["non_crisis", "hard_negative", "non_crisis", "hard_negative"]},
    )
    buckets = error_analysis.disagreements(frame, "tfidf", "bert")
    assert sum(len(b) for b in buckets.values()) == len(frame)


def test_cohens_kappa_is_one_on_perfect_agreement():
    a = pd.Series(["explicit_crisis", "non_crisis", "hard_negative"])
    assert agreement.cohens_kappa(a, a.copy()) == 1.0


def test_cohens_kappa_below_one_on_disagreement():
    a = pd.Series(["explicit_crisis", "non_crisis", "hard_negative", "implicit_crisis"])
    b = pd.Series(["explicit_crisis", "non_crisis", "implicit_crisis", "hard_negative"])
    assert agreement.cohens_kappa(a, b) < 1.0


def test_align_keeps_only_shared_items():
    frames = {
        "A": pd.DataFrame({"id": ["x", "y"], "label": ["non_crisis", "non_crisis"]}),
        "B": pd.DataFrame({"id": ["y", "z"], "label": ["non_crisis", "hard_negative"]}),
    }
    assert list(agreement.align(frames)["id"]) == ["y"]


def test_interpret_bands():
    assert agreement.interpret(0.85) == "almost perfect"
    assert agreement.interpret(0.5) == "moderate"
    assert agreement.interpret(-0.1) == "poor"
