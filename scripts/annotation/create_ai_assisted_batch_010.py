"""Write and validate AI-assisted pre-annotations for blind batch 010.

This script reads only the blind sheet and AI-assisted batch IDs. Its labels
are AI-assisted pre-labels for later human review, never gold labels.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BLIND = ROOT / "data/gold/annotation/remaining_1837_blind.csv"
OUTDIR = ROOT / "data/gold/annotation/ai_assisted"
OUT = OUTDIR / "batch_010_annotations.csv"
LABELS = {"explicit_crisis", "implicit_crisis", "hard_negative", "non_crisis"}
CONFIDENCE = {"high", "medium", "low"}

DECISIONS = {
    "komati-e3c2fc1585dc": ("non_crisis", "high", "Asks about being ghosted by peers without crisis meaning."),
    "irf-6f56934c4d03": ("explicit_crisis", "high", "Author directly wants to die and wishes for an accidental death."),
    "sdcnl-b87ed7c889e4": ("hard_negative", "high", "Seeks help for a friend's grief and depression after another person's suicide."),
    "cams-4e5907718ce3": ("explicit_crisis", "high", "Author describes personal suicidal and self-harm behavior amid current abuse."),
    "komati-137996be06cd": ("non_crisis", "high", "Explicitly denies being suicidal while seeking anonymous conversation."),
    "irf-dcc69bb1ea6a": ("hard_negative", "medium", "Discusses past or noncurrent suicidal states while seeking depression treatment."),
    "komati-472a4a56d873": ("non_crisis", "high", "Celebrates passing a traffic test with no crisis content."),
    "sdcnl-d0cf10b8a8a6": ("non_crisis", "medium", "Expresses existential frustration without self-directed crisis or finality."),
    "suicide_reddit_dataset-49eeaa3fd189": ("explicit_crisis", "medium", "Personally asks whether to die by suicide amid severe current distress."),
    "irf-f1a0b93bc65d": ("non_crisis", "medium", "Describes a disturbing reading and life dissatisfaction without crisis-level self-directed meaning."),
    "irf-434704274756": ("non_crisis", "high", "Social anxiety about joining activities lacks crisis meaning."),
    "sdcnl-d4c33c92fdde": ("explicit_crisis", "high", "Author states a plan to be gone by tonight despite reluctance to die."),
    "komati-edc5675c7321": ("explicit_crisis", "high", "Author describes contemplating self-harm and a prior attempt to die."),
    "sdcnl-7f081226c26b": ("non_crisis", "low", "Brief, unclear statement does not establish crisis meaning."),
    "irf-af43d0dddb9f": ("non_crisis", "medium", "Depression and loneliness are described without self-directed finality."),
    "komati-db50ce4f91a3": ("explicit_crisis", "high", "Author says they want to die and reports a prior overdose attempt."),
    "sdcnl-4f5e0d186103": ("hard_negative", "high", "Reports concern that another person may attempt suicide."),
    "cams-5167a1d16f40": ("non_crisis", "high", "Fear and grief about future family deaths are not the author's crisis intent."),
    "komati-024556f85c6e": ("explicit_crisis", "high", "Author describes a personal suicide method, preparation, and prior failed attempt."),
    "sdcnl-19e17294acbc": ("explicit_crisis", "high", "Author declares an intended personal suicide deadline and desire for relief."),
    "irf-2890492a3dd9": ("non_crisis", "high", "Describes receiving support and feeling less alone without current crisis content."),
    "sdcnl-3ffd7bc600ed": ("explicit_crisis", "high", "Author is considering suicide methods and asks about electrocution."),
    "irf-a6e1a580e212": ("explicit_crisis", "high", "Author repeatedly wants to die and imagines jumping from a bridge."),
    "cams-1b735002ee04": ("explicit_crisis", "high", "Author directly asks to die and wants to kill themselves."),
    "sdcnl-f2e5a970b82e": ("explicit_crisis", "high", "Author implies a personal suicide act in response to family conflict."),
    "cams-3c3dc7d8585b": ("explicit_crisis", "medium", "After quoting criticism, author personally says they should kms."),
    "komati-d33b8c8be4db": ("hard_negative", "high", "Discusses a hypothetical zombie scenario and casual group talk about death."),
    "goemotions-d637a1e26bbc": ("non_crisis", "low", "Text is fragmentary and contains no interpretable crisis meaning."),
    "irf-f538fee6715b": ("explicit_crisis", "high", "Author asks when they will have courage to end their life."),
    "cams-d7cf1718156d": ("non_crisis", "low", "Brief reference to an unspecified conflict is too unclear to establish crisis meaning."),
    "suicide_reddit_dataset-9f09b5902e1b": ("hard_negative", "high", "A joke uses suicide-bomber language without the author's crisis."),
    "suicide_reddit_dataset-d0bc4da9b73b": ("explicit_crisis", "high", "Author describes longstanding attempts, current deadline, and intent not to fail again."),
    "komati-6a464650fd16": ("explicit_crisis", "high", "Farewell framing and imminent departure express the author's current crisis."),
    "suicide_reddit_dataset-887f61bb0240": ("hard_negative", "high", "Author seeks advice about a suicidal friend, not their own crisis."),
    "cams-04e429701030": ("hard_negative", "high", "Prevention-style encouragement mentions ending life without expressing the author's crisis."),
    "sdcnl-2c029bf433f6": ("explicit_crisis", "high", "Author frames personal suicide as an alternative amid severe current distress."),
    "sdcnl-4a6d43faa3f1": ("explicit_crisis", "high", "Author seriously seeks a fatal illness or sudden death as escape from life."),
    "sdcnl-83f5610ed763": ("explicit_crisis", "high", "Author says they want to die and cannot keep going."),
    "komati-e2ab7b8423bc": ("explicit_crisis", "high", "Author rejects being kept alive and directly discusses their own death."),
    "sdcnl-dc16b3ac27e3": ("explicit_crisis", "high", "Author says they are thinking about ending life and believes nothing will help."),
    "komati-3f5046dd4010": ("hard_negative", "high", "Mentions suicide and cutting while describing friends' harmful comments, not personal crisis."),
    "suicide_reddit_dataset-d4830cf52b02": ("explicit_crisis", "high", "Author reports sustained suicidality, a plan, research, and being close to acting."),
    "irf-e3b14159fe4b": ("implicit_crisis", "medium", "Author's belief their cat would be better off with someone else suggests burdensomeness."),
    "sdcnl-4f108ae430c8": ("explicit_crisis", "high", "Author wants to die and explicitly considers killing themselves over current hardship."),
    "sdcnl-cb28b11038a0": ("explicit_crisis", "high", "Author describes personally imagining serious suicide methods."),
    "komati-f258a6eaf0b1": ("implicit_crisis", "medium", "Total hopelessness and no imaginable improvement suggest crisis without overt suicide language."),
    "komati-0efd2db254d4": ("explicit_crisis", "high", "Author discusses permanent self-ending, methods, and suicide as their future."),
    "komati-bf490ee7a19e": ("non_crisis", "high", "Appearance insecurity and frustration lack crisis-level meaning."),
    "cams-9ab0dba811c3": ("non_crisis", "low", "Poetic distress is ambiguous but does not clearly communicate self-directed finality."),
    "cams-afba2e7ce03f": ("non_crisis", "medium", "Author explicitly denies being suicidal despite depression and lack of motivation."),
}


def main() -> int:
    seen: set[str] = set()
    for path in sorted(OUTDIR.glob("batch_*_annotations.csv")):
        if path != OUT:
            seen.update(pd.read_csv(path, usecols=["id"])["id"].astype(str))
    blind = pd.read_csv(BLIND)
    selected = blind[~blind["id"].astype(str).isin(seen)].head(50).copy()
    if len(selected) != 50 or set(selected["id"]) != set(DECISIONS):
        raise ValueError("Batch 010 decisions do not match the next 50 blind records.")
    selected[["label", "confidence", "notes"]] = [DECISIONS[item] for item in selected["id"]]
    output = selected[["id", "text", "label", "confidence", "notes"]].reset_index(drop=True)
    if output["id"].duplicated().any() or set(output["id"]) & seen:
        raise AssertionError("Duplicate or previously annotated IDs found.")
    if not set(output["label"]).issubset(LABELS) or not set(output["confidence"]).issubset(CONFIDENCE):
        raise AssertionError("Invalid annotation value.")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT, index=False, encoding="utf-8")
    written = pd.read_csv(OUT)
    if not written.fillna("").astype(str).equals(output.fillna("").astype(str)):
        raise AssertionError("Saved CSV differs from validated output.")
    print(f"AI-assisted pre-label batch 010 written: {len(written)} rows")
    print(f"First ID: {written.iloc[0]['id']}\nLast ID: {written.iloc[-1]['id']}")
    print("Labels:\n" + written["label"].value_counts().to_string())
    print("Confidence:\n" + written["confidence"].value_counts().to_string())


if __name__ == "__main__":
    main()
