"""Validate and consolidate AI-assisted annotation batches without altering them."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BATCH_DIR = ROOT / "data/gold/annotation/ai_assisted"
BLIND_PATH = ROOT / "data/gold/annotation/remaining_1837_blind.csv"
OUT_CSV = BATCH_DIR / "all_1837_ai_assisted_annotations.csv"
OUT_REPORT = BATCH_DIR / "global_annotation_validation_report.json"
EXPECTED_BATCHES = [f"batch_{number:03d}_annotations.csv" for number in range(1, 38)]
COLUMNS = ["id", "text", "label", "confidence", "notes"]
VALID_LABELS = {"explicit_crisis", "implicit_crisis", "hard_negative", "non_crisis"}
VALID_CONFIDENCES = {"high", "medium", "low"}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def problems_for(rows: list[dict[str, str]], headers: list[str]) -> dict[str, object]:
    ids = [row.get("id", "") for row in rows]
    pairs = [(row.get("id", ""), row.get("text", "")) for row in rows]
    labels = [row.get("label", "") for row in rows]
    confidences = [row.get("confidence", "") for row in rows]
    invalid_labels = sorted(set(labels) - VALID_LABELS)
    invalid_confidences = sorted(set(confidences) - VALID_CONFIDENCES)
    return {
        "record_count": len(rows),
        "columns_exactly_match_required": headers == COLUMNS,
        "columns_found": headers,
        "blank_id_count": sum(not value.strip() for value in ids),
        "blank_label_count": sum(not value.strip() for value in labels),
        "invalid_labels": invalid_labels,
        "invalid_confidence_values": invalid_confidences,
        "duplicate_id_count": len(ids) - len(set(ids)),
        "duplicate_id_text_record_count": len(pairs) - len(set(pairs)),
        "label_distribution": dict(sorted(Counter(labels).items())),
        "confidence_distribution": dict(sorted(Counter(confidences).items())),
    }


def main() -> int:
    discovered = sorted(BATCH_DIR.glob("batch_*_annotations.csv"))
    discovered_names = [path.name for path in discovered]
    missing_batches = sorted(set(EXPECTED_BATCHES) - set(discovered_names))
    unexpected_batch_files = sorted(set(discovered_names) - set(EXPECTED_BATCHES))

    all_rows: list[dict[str, str]] = []
    batches: dict[str, dict[str, object]] = {}
    for filename in EXPECTED_BATCHES:
        path = BATCH_DIR / filename
        if not path.exists():
            continue
        headers, rows = read_csv(path)
        result = problems_for(rows, headers)
        result["passed"] = (
            result["columns_exactly_match_required"]
            and result["blank_id_count"] == 0
            and result["blank_label_count"] == 0
            and not result["invalid_labels"]
            and not result["invalid_confidence_values"]
            and result["duplicate_id_count"] == 0
            and result["duplicate_id_text_record_count"] == 0
        )
        batches[filename] = result
        all_rows.extend(rows)

    blind_headers, blind_rows = read_csv(BLIND_PATH)
    blind_by_id = {row.get("id", ""): row.get("text", "") for row in blind_rows}
    combined = problems_for(all_rows, COLUMNS)
    ids = [row.get("id", "") for row in all_rows]
    annotated_id_set = set(ids)
    blind_id_set = set(blind_by_id)
    text_mismatches = [
        row["id"] for row in all_rows
        if row.get("id", "") in blind_by_id and row.get("text", "") != blind_by_id[row["id"]]
    ]
    combined.update({
        "expected_records": len(blind_rows),
        "annotated_records": len(all_rows),
        "unique_ids": len(annotated_id_set),
        "missing_ids": sorted(blind_id_set - annotated_id_set),
        "unexpected_ids": sorted(annotated_id_set - blind_id_set),
        "text_mismatch_ids": text_mismatches,
        "every_text_exactly_matches_blind_sheet": not text_mismatches,
    })
    checks = {
        "expected_records_is_1837": len(blind_rows) == 1837,
        "annotated_records_is_1837": len(all_rows) == 1837,
        "unique_ids_is_1837": len(annotated_id_set) == 1837,
        "duplicate_ids_is_0": combined["duplicate_id_count"] == 0,
        "missing_ids_is_0": not combined["missing_ids"],
        "unexpected_ids_is_0": not combined["unexpected_ids"],
        "every_text_exactly_matches_blind_sheet": not text_mismatches,
        "text_mismatches_is_0": not text_mismatches,
        "exactly_five_annotation_columns": combined["columns_exactly_match_required"],
        "all_labels_valid": not combined["invalid_labels"],
        "all_confidence_values_valid": not combined["invalid_confidence_values"],
        "blank_labels_is_0": combined["blank_label_count"] == 0,
        "blank_ids_is_0": combined["blank_id_count"] == 0,
        "duplicate_id_text_records_is_0": combined["duplicate_id_text_record_count"] == 0,
    }
    report = {
        "dataset_type": "AI-assisted pre-labels; not human-validated gold labels",
        "source_blind_sheet": BLIND_PATH.relative_to(ROOT).as_posix(),
        "expected_batch_count": 37,
        "discovered_batch_count": len(discovered),
        "expected_batches": EXPECTED_BATCHES,
        "discovered_batches": discovered_names,
        "missing_batches": missing_batches,
        "unexpected_batch_files": unexpected_batch_files,
        "all_batches_passed_individual_validation": bool(batches) and all(
            result["passed"] for result in batches.values()
        ),
        "batch_validation": batches,
        "blind_sheet_columns": blind_headers,
        "global_validation": combined,
        "required_checks": checks,
        "overall_label_distribution": combined["label_distribution"],
        "overall_confidence_distribution": combined["confidence_distribution"],
        "low_confidence_records": combined["confidence_distribution"].get("low", 0),
        "medium_confidence_records": combined["confidence_distribution"].get("medium", 0),
        "high_confidence_records": combined["confidence_distribution"].get("high", 0),
    }
    report["global_validation_passed"] = (
        not missing_batches
        and not unexpected_batch_files
        and report["all_batches_passed_individual_validation"]
        and all(checks.values())
    )

    with OUT_REPORT.open("w", encoding="utf-8", newline="") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    if report["global_validation_passed"]:
        with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows({column: row.get(column, "") for column in COLUMNS} for row in all_rows)

    print(f"Global validation {'PASSED' if report['global_validation_passed'] else 'FAILED'}")
    print(json.dumps(checks, indent=2))
    return 0 if report["global_validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
