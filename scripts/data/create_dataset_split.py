"""Profile and create the canonical deterministic split of frozen gold data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["id", "text", "label", "confidence", "notes"]
LABELS = ["explicit_crisis", "implicit_crisis", "hard_negative", "non_crisis"]
CONFIDENCES = {"high", "medium", "low"}
SEED, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO = 42, 0.70, 0.15, 0.15
SCRIPT_VERSION = "1.0"


def read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


def distribution(frame: pd.DataFrame, column: str, values: list[str]) -> dict[str, dict[str, float | int]]:
    counts = frame[column].value_counts()
    return {
        value: {
            "count": int(counts.get(value, 0)),
            "percentage": round((counts.get(value, 0) / len(frame) * 100) if len(frame) else 0, 6),
        }
        for value in values
    }


def profile(frame: pd.DataFrame) -> dict[str, object]:
    chars, words = frame["text"].str.len(), frame["text"].str.split().str.len()
    stats = lambda series: {
        "min": int(series.min()), "max": int(series.max()),
        "mean": round(float(series.mean()), 6), "median": float(series.median()),
    }
    return {
        "total_records": len(frame),
        "unique_ids": int(frame["id"].nunique()),
        "label_distribution": distribution(frame, "label", LABELS),
        "confidence_distribution": distribution(frame, "confidence", ["high", "medium", "low"]),
        "text_length_statistics": {"characters": stats(chars), "whitespace_delimited_words": stats(words)},
        "duplicate_id_count": int(frame["id"].duplicated().sum()),
        "duplicate_text_count": int(frame["text"].duplicated().sum()),
        "missing_value_counts": {column: int(frame[column].eq("").sum()) for column in COLUMNS},
    }


def validate_input(frame: pd.DataFrame) -> None:
    problems = []
    if list(frame.columns) != COLUMNS:
        problems.append(f"columns must be exactly {COLUMNS}")
    if frame["id"].eq("").any() or frame["id"].duplicated().any():
        problems.append("blank or duplicate IDs")
    if frame["text"].eq("").any():
        problems.append("blank text")
    if frame["label"].eq("").any() or not set(frame["label"]).issubset(LABELS):
        problems.append("blank or invalid labels")
    if frame["confidence"].eq("").any() or not set(frame["confidence"]).issubset(CONFIDENCES):
        problems.append("blank or invalid confidence values")
    if frame["notes"].eq("").any():
        problems.append("blank notes")
    if problems:
        raise ValueError("; ".join(problems))


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Use two fixed-size, label-stratified held-out draws for exact near-ratios."""
    validation_count, test_count = round(len(frame) * VALIDATION_RATIO), round(len(frame) * TEST_RATIO)
    remaining, test = train_test_split(frame, test_size=test_count, random_state=SEED, stratify=frame["label"])
    train, validation = train_test_split(
        remaining, test_size=validation_count, random_state=SEED, stratify=remaining["label"]
    )
    return {name: split.reset_index(drop=True) for name, split in {
        "train": train, "validation": validation, "test": test
    }.items()}


def validate_splits(source: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> dict[str, object]:
    combined = pd.concat([splits[name] for name in ("train", "validation", "test")], ignore_index=True)
    source_ids, split_ids = set(source["id"]), set(combined["id"])
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    id_overlaps = {f"{left}_{right}": sorted(set(splits[left]["id"]) & set(splits[right]["id"])) for left, right in pairs}
    text_overlaps = {f"{left}_{right}": set(splits[left]["text"]) & set(splits[right]["text"]) for left, right in pairs}
    source_by_id = source.set_index("id")["text"]
    text_mismatches = [row.id for row in combined.itertuples(index=False) if row.text != source_by_id.at[row.id]]
    input_duplicate_texts = set(source.loc[source["text"].duplicated(keep=False), "text"])
    unexpected_text_overlaps = sorted(set().union(*text_overlaps.values()) - input_duplicate_texts)
    result = {
        "split_counts": {name: len(splits[name]) for name in ("train", "validation", "test")},
        "total_split_records": len(combined),
        "unique_ids_across_splits": int(combined["id"].nunique()),
        "duplicate_ids_across_splits": int(combined["id"].duplicated().sum()),
        "missing_ids": sorted(source_ids - split_ids),
        "unexpected_ids": sorted(split_ids - source_ids),
        "id_overlap": id_overlaps,
        "id_overlap_count": sum(map(len, id_overlaps.values())),
        "text_mismatch_ids": text_mismatches,
        "text_mismatch_count": len(text_mismatches),
        "source_duplicate_text_count": int(source["text"].duplicated().sum()),
        "exact_text_overlap_between_splits": {key: len(value) for key, value in text_overlaps.items()},
        "exact_text_overlap_count": len(set().union(*text_overlaps.values())),
        "unexpected_exact_text_overlap": unexpected_text_overlaps,
        "all_split_columns_exact": all(list(split.columns) == COLUMNS for split in splits.values()),
        "all_labels_valid": all(set(split["label"]).issubset(LABELS) for split in splits.values()),
        "all_confidences_valid": all(set(split["confidence"]).issubset(CONFIDENCES) for split in splits.values()),
        "all_labels_represented": {name: all(label in set(split["label"]) for label in LABELS) for name, split in splits.items()},
    }
    result["passed"] = (
        result["total_split_records"] == len(source)
        and result["unique_ids_across_splits"] == len(source)
        and result["duplicate_ids_across_splits"] == 0
        and not result["missing_ids"] and not result["unexpected_ids"]
        and result["id_overlap_count"] == 0 and result["text_mismatch_count"] == 0
        and not result["unexpected_exact_text_overlap"] and result["all_split_columns_exact"]
        and result["all_labels_valid"] and result["all_confidences_valid"]
        and all(result["all_labels_represented"].values())
    )
    return result


def assignment_hash(splits: dict[str, pd.DataFrame]) -> str:
    frame = pd.concat([split.assign(split=name)[["id", "split"]] for name, split in splits.items()])
    return hashlib.sha256(frame.sort_values("id").to_csv(index=False).encode("utf-8")).hexdigest()


def create_artifacts(input_path: Path, outdir: Path, profile_path: Path) -> dict[str, object]:
    source = read_dataset(input_path)
    validate_input(source)
    splits = split_frame(source)
    validation = validate_splits(source, splits)
    if not validation["passed"]:
        raise RuntimeError(f"Split validation failed: {json.dumps(validation)}")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        split.to_csv(outdir / f"{name}.csv", index=False, encoding="utf-8")
    metadata = {
        "input_dataset_path": input_path.as_posix(), "total_records": len(source),
        "random_seed": SEED, "train_ratio": TRAIN_RATIO, "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO, "stratification_column": "label",
        "script": "scripts/data/create_dataset_split.py", "script_version": SCRIPT_VERSION,
        "split_assignment_sha256": assignment_hash(splits), "actual_split_counts": validation["split_counts"],
        "label_distribution": {"original": distribution(source, "label", LABELS), **{
            name: distribution(split, "label", LABELS) for name, split in splits.items()
        }},
        "validation": validation,
    }
    with (outdir / "split_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with profile_path.open("w", encoding="utf-8") as handle:
        json.dump(profile(source), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/gold/gold.csv")
    parser.add_argument("--outdir", type=Path, default=ROOT / "data/gold/splits")
    parser.add_argument("--profile", type=Path, default=ROOT / "data/gold/gold_dataset_profile.json")
    args = parser.parse_args()
    metadata = create_artifacts(args.input, args.outdir, args.profile)
    print(f"Split validation PASSED: {metadata['actual_split_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
