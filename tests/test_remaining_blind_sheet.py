from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.annotation.create_remaining_blind_sheet import make_remaining_sheet  # noqa: E402


def test_remaining_sheet_is_blind_seeded_and_excludes_completed_ids(monkeypatch):
    import scripts.annotation.create_remaining_blind_sheet as module

    monkeypatch.setattr(module, "EXPECTED_COMPLETED", 2)
    monkeypatch.setattr(module, "EXPECTED_REMAINING", 3)
    candidates = pd.DataFrame({"id": list("abcde"), "text": [f"text {x}" for x in "abcde"], "source": ["x"] * 5})
    a = pd.DataFrame({"id": list("ab"), "label": ["non_crisis", "explicit_crisis"]})
    b = a.copy()
    blind, completed = make_remaining_sheet(candidates, a, b, seed=42)
    assert len(blind) == 3
    assert not set(blind["id"]) & completed
    assert list(blind.columns) == ["id", "text", "label", "confidence", "notes"]
    assert blind[["label", "confidence", "notes"]].fillna("").eq("").all().all()


def test_remaining_sheet_stops_when_completed_id_sets_differ(monkeypatch):
    import scripts.annotation.create_remaining_blind_sheet as module

    monkeypatch.setattr(module, "EXPECTED_COMPLETED", 1)
    candidates = pd.DataFrame({"id": ["a", "b"], "text": ["x", "y"]})
    a = pd.DataFrame({"id": ["a"], "label": ["non_crisis"]})
    b = pd.DataFrame({"id": ["b"], "label": ["non_crisis"]})
    with pytest.raises(ValueError, match="differ"):
        make_remaining_sheet(candidates, a, b, seed=42)
