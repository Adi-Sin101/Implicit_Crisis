"""Error analysis: where the two models disagree, and why (Step 8)."""
from __future__ import annotations

import pandas as pd


def build_frame(texts, y_true, preds: dict[str, list]) -> pd.DataFrame:
    frame = pd.DataFrame({"text": list(texts), "gold": list(y_true)})
    for model_name, y_pred in preds.items():
        frame[model_name] = list(y_pred)
        frame[f"{model_name}_correct"] = frame[model_name] == frame["gold"]
    return frame


def disagreements(frame: pd.DataFrame, model_a: str, model_b: str) -> dict[str, pd.DataFrame]:
    a_ok = frame[f"{model_a}_correct"]
    b_ok = frame[f"{model_b}_correct"]
    return {
        f"{model_a}_only": frame[a_ok & ~b_ok],
        f"{model_b}_only": frame[b_ok & ~a_ok],
        "both_wrong": frame[~a_ok & ~b_ok],
        "both_right": frame[a_ok & b_ok],
    }


def hard_negative_false_positives(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Hard negatives the model called a crisis - the lexical-shortcut failure mode."""
    is_hard_negative = frame["gold"] == "hard_negative"
    called_crisis = frame[model].isin(["explicit_crisis", "implicit_crisis"])
    return frame[is_hard_negative & called_crisis]


def missed_implicit(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    """Implicit-crisis items the model failed to recognise - the central failure mode."""
    return frame[(frame["gold"] == "implicit_crisis") & (frame[model] != "implicit_crisis")]


def summary(frame: pd.DataFrame, model_a: str, model_b: str) -> pd.DataFrame:
    blocks = disagreements(frame, model_a, model_b)
    return pd.DataFrame(
        [{"bucket": name, "n": len(block)} for name, block in blocks.items()]
    )
