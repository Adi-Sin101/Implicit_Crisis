"""Create and validate the adjudicated calibration disagreement artifact."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

LABELS = {"explicit_crisis", "implicit_crisis", "hard_negative", "non_crisis"}
SOURCE_COLUMNS = [
    "id", "text", "A label", "B label", "A confidence", "B confidence",
    "A notes", "B notes",
]
OUTPUT_COLUMNS = SOURCE_COLUMNS + [
    "final label", "adjudication confidence", "adjudication rationale",
]
DECISIONS = {
    "cams-24c0c25e6a33": (
        "explicit_crisis", "high",
        "The author explicitly includes themselves in the statement about thinking suicide, making this self-referential suicidal ideation despite the broader discussion of suicide forums.",
    ),
    "cams-687449dcc3c5": (
        "non_crisis", "high",
        "The author refers to having had \"darkest moments\" in the past but does not provide evidence that those moments involved suicidal or crisis-level distress. The current post focuses on seeking help and social connection.",
    ),
    "irf-5f27a24f37fa": (
        "explicit_crisis", "medium",
        "The passage contains \"might be better to just end it all\" together with detailed self-harm/cutting content. Although written largely in second person, the passage reads as a description of lived personal experience rather than a clearly external or educational discussion.",
    ),
    "irf-a11b2a6bb141": (
        "implicit_crisis", "medium",
        "The author describes persistent self-loathing and growing misery associated with isolation and loneliness. There is no explicit suicidal or self-harm statement, but the distress exceeds ordinary transient sadness.",
    ),
    "komati-c0f53895b28d": (
        "hard_negative", "high",
        "Suicide-related terminology appears only as part of a list of subreddit names and is not an expression of the author's own crisis or suicidal state.",
    ),
    "komati-e335a390c977": (
        "explicit_crisis", "medium",
        "\"make me want to off myself\" is a first-person suicidal euphemism. Although the trigger may suggest hyperbole, the context does not clearly establish that the statement is purely figurative or joking.",
    ),
    "sdcnl-9a7998c1beff": (
        "non_crisis", "high",
        "The author discusses depression and confusion but explicitly describes having avoided alcohol, drugs, self-harm, and suicide. There is insufficient evidence of current crisis-level distress.",
    ),
    "sdcnl-ca2a4047d113": (
        "non_crisis", "medium",
        "\"nothing is good anymore\" expresses negative mood or possible hopelessness, but the extremely short text provides insufficient evidence to establish severe crisis-level distress.",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    source_path = root / "data/gold/annotation/agreement/calibration_150_disagreements.csv"
    output_path = root / "data/gold/annotation/agreement/calibration_150_adjudicated.csv"
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    source = pd.read_csv(source_path, encoding="utf-8")
    if not set(SOURCE_COLUMNS).issubset(source.columns):
        raise ValueError(f"Missing required source columns: {list(source.columns)}")
    if len(source) != 8 or source["id"].duplicated().any():
        raise ValueError("Source disagreement file must contain eight unique rows")
    if set(source["id"]) != set(DECISIONS):
        raise ValueError("Source IDs do not match the eight adjudication decisions")

    result = source.copy()
    result[["final label", "adjudication confidence", "adjudication rationale"]] = result["id"].map(DECISIONS).apply(pd.Series)
    if set(result["final label"]) - LABELS or result["adjudication rationale"].eq("").any():
        raise ValueError("Invalid adjudication values")
    result = result[list(source.columns) + [
        "final label", "adjudication confidence", "adjudication rationale",
    ]]
    result.to_csv(output_path, index=False, encoding="utf-8")

    reloaded = pd.read_csv(output_path, encoding="utf-8")
    if len(reloaded) != 8 or set(reloaded["id"]) != set(DECISIONS) or reloaded["id"].duplicated().any():
        raise AssertionError("Adjudicated row accounting failed")
    for column in source.columns:
        if not reloaded[column].equals(source[column]):
            raise AssertionError(f"Source column changed: {column}")
    if not reloaded["adjudication rationale"].fillna("").str.strip().ne("").all():
        raise AssertionError("An adjudication rationale is empty")
    print(f"Created: {output_path}")
    print(f"Rows: {len(reloaded)}")
    print(f"Final distribution: {reloaded['final label'].value_counts().reindex(sorted(LABELS), fill_value=0).to_dict()}")
    print(f"Source SHA-256: {sha256(source_path)}")
    print(f"Output SHA-256: {sha256(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())