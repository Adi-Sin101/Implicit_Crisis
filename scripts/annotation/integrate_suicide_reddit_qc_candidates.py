"""Append the 105 QC-retained Reddit candidates without touching calibration data.

The script intentionally does not generate annotation sheets.  Existing full
sheet generation would overwrite the completed calibration files; the correct
next hand-off is a separate remaining-workload sheet after this append.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.integration.suicide_reddit_qc import (  # noqa: E402
    append_additions, candidate_fingerprint, completed_calibration_ids,
    duplicate_report, prepare_retained,
)
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--qc", default=None)
    parser.add_argument("--calibration-a", default="data/gold/annotation/calibration_150_A.csv")
    parser.add_argument("--calibration-b", default="data/gold/annotation/calibration_150_B.csv")
    parser.add_argument("--report", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = load_config("paths.yaml")
    candidate_path = resolve(args.candidates or paths["gold"]["candidates"])
    qc_path = resolve(args.qc or paths["interim"]["suicide_reddit_hard_negative_qc"])
    report_path = resolve(args.report or paths["interim"]["suicide_reddit_hard_negative_integration_report"])
    existing = read_table(candidate_path)
    calibration_ids = completed_calibration_ids(
        read_table(resolve(args.calibration_a)), read_table(resolve(args.calibration_b))
    )
    if not calibration_ids <= set(existing["id"].astype(str)):
        raise ValueError("Completed calibration IDs are not all present in candidates.csv; integration stopped.")
    before_fingerprint = candidate_fingerprint(existing)
    additions = prepare_retained(read_table(qc_path))
    retained_qc = read_table(qc_path)
    retained_qc = retained_qc[retained_qc["qc_status"].isin({
        "strong_hard_negative_candidate", "borderline_needs_annotator_judgment"
    })]
    duplicates, duplicate_counts = duplicate_report(existing, additions)
    if len(duplicates):
        report = {
            "integration_completed": False,
            "reason": "Duplicate candidates found; no records appended.",
            "candidates_before": len(existing), "annotated_before": len(calibration_ids),
            "qc_retained": len(additions), "duplicate_report": duplicate_counts,
            "duplicate_source_ids": duplicates["src_id"].tolist(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1
    combined = append_additions(existing, additions)
    if candidate_fingerprint(combined.iloc[: len(existing)]) != before_fingerprint:
        raise AssertionError("Existing candidate rows changed; integration stopped.")
    if not args.dry_run:
        write_table(combined, candidate_path)
    report = {
        "integration_completed": not args.dry_run,
        "dry_run": args.dry_run,
        "candidates_before": len(existing),
        "annotated_before": len(calibration_ids),
        "unannotated_before": len(existing) - len(calibration_ids),
        "qc_retained": len(additions),
        "duplicate_report": duplicate_counts,
        "genuinely_new_candidates": len(additions),
        "candidates_after": len(combined),
        "annotated_after": len(calibration_ids),
        "unannotated_after": len(combined) - len(calibration_ids),
        "new_source_distribution": additions["source"].value_counts().to_dict(),
        "new_candidate_reason_distribution": retained_qc["candidate_reason"].value_counts().to_dict(),
        "new_stratum_distribution": additions["stratum"].value_counts().to_dict(),
        "gold_labels_assigned": False,
        "completed_calibration_records_untouched": True,
        "existing_candidate_rows_untouched": True,
        "excluded_qc_records_added": False,
        "next_annotation_integration_point": "Create a separate blind sheet for the 1,837 remaining items; do not regenerate calibration_150_A/B.csv.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
