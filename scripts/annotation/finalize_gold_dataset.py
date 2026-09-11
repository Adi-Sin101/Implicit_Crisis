"""Create and validate the operational annotation dataset while retaining its backup."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "data/gold/annotation/ai_assisted"
SOURCE_FINAL = SOURCE_DIR / "all_1837_ai_assisted_annotations.csv"
DEST_BATCH_DIR = ROOT / "data/gold/annotation_batches"
DEST_FINAL = ROOT / "data/gold/gold.csv"
REPORT = ROOT / "data/gold/final_gold_validation_report.json"
BLIND = ROOT / "data/gold/annotation/remaining_1837_blind.csv"
COLUMNS = ["id", "text", "label", "confidence", "notes"]
LABELS = {"explicit_crisis", "implicit_crisis", "hard_negative", "non_crisis"}
CONFIDENCES = {"high", "medium", "low"}
EXPECTED_BATCHES = [f"batch_{number:03d}_annotations.csv" for number in range(1, 38)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def validate(rows: list[dict[str, str]], headers: list[str], blind_rows: list[dict[str, str]]) -> dict[str, object]:
    ids = [row.get("id", "") for row in rows]
    labels = [row.get("label", "") for row in rows]
    confidences = [row.get("confidence", "") for row in rows]
    pairs = [(row.get("id", ""), row.get("text", "")) for row in rows]
    blind = {row.get("id", ""): row.get("text", "") for row in blind_rows}
    id_set, blind_id_set = set(ids), set(blind)
    text_mismatches = [
        row["id"] for row in rows
        if row.get("id", "") in blind and row.get("text", "") != blind[row["id"]]
    ]
    return {
        "expected_records": len(blind_rows),
        "annotated_records": len(rows),
        "unique_ids": len(id_set),
        "duplicate_ids": len(ids) - len(id_set),
        "missing_ids": sorted(blind_id_set - id_set),
        "unexpected_ids": sorted(id_set - blind_id_set),
        "text_mismatch_ids": text_mismatches,
        "text_mismatches": len(text_mismatches),
        "columns": headers,
        "columns_exactly_match_required": headers == COLUMNS,
        "blank_ids": sum(not value.strip() for value in ids),
        "blank_labels": sum(not value.strip() for value in labels),
        "invalid_labels": sorted(set(labels) - LABELS),
        "invalid_confidence_values": sorted(set(confidences) - CONFIDENCES),
        "duplicate_id_text_records": len(pairs) - len(set(pairs)),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "confidence_distribution": dict(sorted(Counter(confidences).items())),
    }


def passed(result: dict[str, object]) -> bool:
    return (
        result["expected_records"] == 1837
        and result["annotated_records"] == 1837
        and result["unique_ids"] == 1837
        and result["duplicate_ids"] == 0
        and not result["missing_ids"]
        and not result["unexpected_ids"]
        and result["text_mismatches"] == 0
        and result["columns_exactly_match_required"]
        and result["blank_ids"] == 0
        and result["blank_labels"] == 0
        and not result["invalid_labels"]
        and not result["invalid_confidence_values"]
        and result["duplicate_id_text_records"] == 0
    )


def main() -> int:
    source_headers, source_rows = read_rows(SOURCE_FINAL)
    _, blind_rows = read_rows(BLIND)
    source_validation = validate(source_rows, source_headers, blind_rows)
    if not passed(source_validation):
        raise RuntimeError("Source consolidated dataset failed validation; no files were copied.")

    DEST_BATCH_DIR.mkdir(parents=True, exist_ok=True)
    batch_copy_validation = {}
    for filename in EXPECTED_BATCHES:
        source, destination = SOURCE_DIR / filename, DEST_BATCH_DIR / filename
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)
        batch_copy_validation[filename] = {
            "source_sha256": sha256(source),
            "destination_sha256": sha256(destination),
        }
    batches_byte_for_byte_identical = all(
        item["source_sha256"] == item["destination_sha256"]
        for item in batch_copy_validation.values()
    )
    if not batches_byte_for_byte_identical:
        raise RuntimeError("One or more copied batch files differ from the source backup.")

    shutil.copy2(SOURCE_FINAL, DEST_FINAL)
    final_headers, final_rows = read_rows(DEST_FINAL)
    final_validation = validate(final_rows, final_headers, blind_rows)
    final_byte_for_byte_identical_to_source = sha256(SOURCE_FINAL) == sha256(DEST_FINAL)
    report = {
        "dataset_type": "AI-assisted pre-labels; not human-validated gold labels",
        "source_backup_directory": SOURCE_DIR.relative_to(ROOT).as_posix(),
        "source_consolidated_dataset": SOURCE_FINAL.relative_to(ROOT).as_posix(),
        "operational_dataset": DEST_FINAL.relative_to(ROOT).as_posix(),
        "source_backup_retained": SOURCE_DIR.is_dir(),
        "batch_destination": DEST_BATCH_DIR.relative_to(ROOT).as_posix(),
        "expected_batch_count": 37,
        "copied_batch_count": len(batch_copy_validation),
        "batches_byte_for_byte_identical_to_backup": batches_byte_for_byte_identical,
        "batch_copy_validation": batch_copy_validation,
        "gold_csv_byte_for_byte_identical_to_source_consolidated_file": final_byte_for_byte_identical_to_source,
        "global_validation": final_validation,
        "validation_passed": passed(final_validation)
        and batches_byte_for_byte_identical
        and final_byte_for_byte_identical_to_source,
    }
    with REPORT.open("w", encoding="utf-8", newline="") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Final gold validation {'PASSED' if report['validation_passed'] else 'FAILED'}")
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
