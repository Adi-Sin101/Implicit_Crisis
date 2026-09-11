"""Write and validate AI-assisted pre-annotations for blind batch 009.

This is an annotation hand-off artefact, never a human-gold or final-dataset
builder.  It reads only the blind sheet and prior AI-assisted batch IDs.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BLIND = ROOT / "data/gold/annotation/remaining_1837_blind.csv"
OUTDIR = ROOT / "data/gold/annotation/ai_assisted"
OUT = OUTDIR / "batch_009_annotations.csv"
ALLOWED_LABELS = {"explicit_crisis", "implicit_crisis", "hard_negative", "non_crisis"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}

# Each decision was made by reading the blind text under the project guideline.
# These are AI-assisted pre-labels for later human review, not gold labels.
DECISIONS = {
    "sdcnl-ca1fc1e7b3e3": ("explicit_crisis", "high", "Author states a personal suicide plan and current suicidal concern."),
    "komati-0711815d8f9f": ("explicit_crisis", "medium", "Author personally asks others to persuade them to die."),
    "sdcnl-240017e02ad2": ("explicit_crisis", "high", "Author describes repeated thoughts of ending life and dying by their own hands."),
    "irf-fe026641b00a": ("non_crisis", "high", "Loneliness is described without crisis-level finality."),
    "komati-7212c9473a13": ("hard_negative", "medium", "Past suicidal thoughts are discussed while offering support and describing improvement."),
    "irf-357b033c8fae": ("non_crisis", "high", "Discusses depression and sexual drive without crisis meaning."),
    "irf-895f4ae957cf": ("non_crisis", "high", "Describes depression-related academic difficulty without crisis-level meaning."),
    "komati-e032b07dae40": ("implicit_crisis", "medium", "Burdensomeness and questions about being missed imply self-directed crisis without direct intent."),
    "sdcnl-463209a4af6f": ("explicit_crisis", "high", "Author directly wishes to die and describes personal death scenarios."),
    "komati-20f4d8097c15": ("non_crisis", "high", "Frustration with an online game has no crisis meaning."),
    "komati-d80bc86bacf5": ("explicit_crisis", "high", "Direct first-person statement of wanting to kill oneself."),
    "cams-39398711ba5c": ("explicit_crisis", "medium", "Author reports personal suicide thoughts emerging during distress."),
    "sdcnl-305815deedec": ("explicit_crisis", "high", "Seeks information about a suicide method for personal use."),
    "irf-9765028b3ba1": ("non_crisis", "high", "Loneliness and social worry are described without crisis-level meaning."),
    "komati-2fd82f629cd6": ("implicit_crisis", "high", "Peace with a final decision and farewell framing convey crisis without overt suicide terms."),
    "komati-7db2be97e679": ("explicit_crisis", "high", "Author explicitly says they might kill themselves over current distress."),
    "komati-6f09c3a9a2c4": ("hard_negative", "high", "Discusses suicide as a social consequence, not the author's own crisis."),
    "sdcnl-72d44841cc8a": ("explicit_crisis", "high", "Author repeatedly presents killing themselves as a personal option."),
    "sdcnl-c823d883aeb6": ("explicit_crisis", "high", "Author wants to die and describes looking for ways to kill themselves."),
    "komati-5a95af2f0124": ("implicit_crisis", "medium", "Persistent emptiness, hopelessness, and wishing to avoid waking suggest crisis-level escape framing."),
    "irf-757c75ead500": ("non_crisis", "high", "Work and relationship stress are described without crisis-level meaning."),
    "komati-2624ca3d9d7d": ("implicit_crisis", "medium", "Author frames death as escape from self and sees no tolerable future."),
    "sdcnl-e44dda896426": ("non_crisis", "low", "Severe breakup distress and desire to stop feeling are present, but finality is not clear."),
    "cams-8be6c8740ffe": ("implicit_crisis", "medium", "Author cannot imagine a happy future and rejects continuing through life."),
    "irf-33e79235e78a": ("explicit_crisis", "high", "Directly asks why they should not kill themselves right now."),
    "irf-b6f80d06b96b": ("non_crisis", "high", "Reflects on loneliness without crisis-level finality."),
    "irf-acf913d46351": ("implicit_crisis", "medium", "Describes emptiness, lack of meaning, and wanting to stop everything without explicit suicide terminology."),
    "komati-b73cac9ffd68": ("explicit_crisis", "medium", "Author explicitly wishes to die while describing personal distress."),
    "cams-700053c8b78c": ("implicit_crisis", "medium", "Although suicide is rejected, the author expresses a personal wish not to exist."),
    "cams-ab18ce006345": ("non_crisis", "high", "Self-critical voices and a request about self-worth lack crisis-level finality."),
    "komati-7e20d6b952d4": ("explicit_crisis", "high", "Author reports approaching a personal suicide attempt."),
    "komati-fbe6e8f2f6d0": ("explicit_crisis", "high", "Author states they want to die and will end it all."),
    "irf-4bf50ab5e119": ("explicit_crisis", "high", "Author directly says they want to end it all amid severe distress."),
    "irf-a795e62851d2": ("explicit_crisis", "high", "Author says they cannot continue living and discusses ending their life."),
    "sdcnl-d2128870e55a": ("explicit_crisis", "high", "Author gives a stated timeframe until killing themselves."),
    "irf-1910e29115a9": ("non_crisis", "high", "New-year loneliness is described without crisis-level meaning."),
    "cams-7233d062d57f": ("non_crisis", "low", "The referent of nighttime thoughts is unclear, so no crisis meaning can be established."),
    "sdcnl-b45e36389f89": ("hard_negative", "medium", "Discusses a past suicide attempt and current depression without current self-crisis language."),
    "irf-9977257db532": ("non_crisis", "high", "Asks about handling a difficult friendship without crisis content."),
    "sdcnl-436b6f2aeedd": ("explicit_crisis", "high", "Author explicitly describes personal suicide methods and intent to end life."),
    "cams-13b00c6f4e71": ("explicit_crisis", "medium", "Author states they might kill themselves in response to anticipated failure."),
    "komati-969f1df82ec5": ("non_crisis", "high", "A humorous water review contains no crisis meaning."),
    "cams-a2bf7ac21136": ("hard_negative", "high", "Violent game-like statement uses death casually rather than expressing personal crisis."),
    "irf-f1ec4c49d03a": ("explicit_crisis", "high", "Author describes self-harm and says suicide is their only option."),
    "komati-2b3350734019": ("implicit_crisis", "low", "Current abuse and inability to cope suggest crisis, but explicit suicide discussion is framed as contemplation rather than current intent."),
    "sdcnl-d01bdc3c9017": ("explicit_crisis", "high", "Past attempt is followed by daily current appraisal of suicide as a better option."),
    "komati-256339bfc339": ("explicit_crisis", "high", "Author asks for help while describing personal thoughts about how to end life."),
    "komati-2c3966a3b48d": ("implicit_crisis", "medium", "States being on the brink and losing the will to continue without overt suicide wording."),
    "komati-cf2478314bb9": ("explicit_crisis", "medium", "Reports a current self-inflicted wound; direct self-harm is expressed."),
    "irf-517daa8e447d": ("non_crisis", "high", "Seeks treatment information for depression and anxiety without crisis meaning."),
}


def main() -> int:
    prior_files = sorted(OUTDIR.glob("batch_*_annotations.csv"))
    annotated_ids: set[str] = set()
    for path in prior_files:
        if path == OUT:
            continue
        annotated_ids.update(pd.read_csv(path, usecols=["id"])["id"].astype(str))
    blind = pd.read_csv(BLIND)
    selected = blind[~blind["id"].astype(str).isin(annotated_ids)].head(50).copy()
    if len(selected) != 50:
        raise ValueError(f"Expected 50 available rows, found {len(selected)}")
    if set(selected["id"]) != set(DECISIONS):
        raise ValueError("Decision IDs do not exactly match the next 50 blind IDs.")
    selected[["label", "confidence", "notes"]] = [DECISIONS[item] for item in selected["id"]]
    output = selected[["id", "text", "label", "confidence", "notes"]].reset_index(drop=True)
    if output["id"].duplicated().any() or set(output["id"]) & annotated_ids:
        raise AssertionError("Duplicate or previously annotated ID in batch.")
    if not set(output["label"]).issubset(ALLOWED_LABELS) or not set(output["confidence"]).issubset(ALLOWED_CONFIDENCE):
        raise AssertionError("Invalid AI-assisted annotation values.")
    if list(output.columns) != ["id", "text", "label", "confidence", "notes"]:
        raise AssertionError("Unexpected output schema.")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False, encoding="utf-8")
    written = pd.read_csv(OUT)
    if not written.fillna("").astype(str).equals(output.fillna("").astype(str)):
        raise AssertionError("Saved batch does not match the validated output.")
    print(f"AI-assisted pre-label batch 009 written: {len(written)} rows")
    print(f"First ID: {written.iloc[0]['id']}\nLast ID: {written.iloc[-1]['id']}")
    print("Labels:\n" + written["label"].value_counts().to_string())
    print("Confidence:\n" + written["confidence"].value_counts().to_string())


if __name__ == "__main__":
    main()
