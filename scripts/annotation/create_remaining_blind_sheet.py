"""Create the blind sheet for candidates not completed in both calibration sheets.

The completed calibration sheets and candidate pool are read-only inputs.  No
gold label is assigned; this script writes only empty annotator-facing fields.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.annotation import sheets  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402

EXPECTED_COMPLETED = 150
EXPECTED_REMAINING = 1837


def nonblank_label_ids(sheet):
    required = {"id", "label"}
    missing = required - set(sheet.columns)
    if missing:
        raise ValueError(f"Calibration sheet missing columns: {sorted(missing)}")
    labels = sheet["label"].fillna("").astype(str).str.strip()
    return set(sheet.loc[labels.ne(""), "id"].astype(str))


def make_remaining_sheet(candidates, calibration_a, calibration_b, seed: int):
    """Return a validated blind sheet; stop on any unexpected annotation state."""
    if candidates["id"].astype(str).duplicated().any():
        raise ValueError("candidates.csv contains duplicate IDs; sheet creation stopped.")
    a_ids, b_ids = nonblank_label_ids(calibration_a), nonblank_label_ids(calibration_b)
    if a_ids != b_ids:
        raise ValueError("Nonblank calibration IDs differ between A and B; sheet creation stopped.")
    completed = a_ids
    if len(completed) != EXPECTED_COMPLETED:
        raise ValueError(f"Expected {EXPECTED_COMPLETED} completed IDs, found {len(completed)}; sheet creation stopped.")
    candidate_ids = set(candidates["id"].astype(str))
    if not completed <= candidate_ids:
        raise ValueError("Completed calibration IDs are absent from candidates.csv; sheet creation stopped.")
    remaining = candidates[~candidates["id"].astype(str).isin(completed)].copy()
    if len(remaining) != EXPECTED_REMAINING:
        raise ValueError(f"Expected {EXPECTED_REMAINING} remaining rows, found {len(remaining)}; sheet creation stopped.")
    blind = sheets.make_sheet(remaining, seed=seed)
    sheets.assert_blind(blind)
    if list(blind.columns) != ["id", "text", "label", "confidence", "notes"]:
        raise AssertionError("Blind sheet has an unexpected schema.")
    if blind["id"].duplicated().any() or set(blind["id"].astype(str)) & completed:
        raise AssertionError("Blind-sheet ID validation failed.")
    if not all(blind[column].fillna("").astype(str).eq("").all() for column in ("label", "confidence", "notes")):
        raise AssertionError("Blind-sheet annotation fields are not empty.")
    return blind, completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--calibration-a", default="data/gold/annotation/calibration_150_A.csv")
    parser.add_argument("--calibration-b", default="data/gold/annotation/calibration_150_B.csv")
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Validate an existing blind sheet and refresh only its report; never rewrite the sheet.",
    )
    args = parser.parse_args()
    paths = load_config("paths.yaml")
    seed = args.seed if args.seed is not None else load_config("candidate_pool.yaml").get("seed", 42)
    candidates = read_table(resolve(args.candidates or paths["gold"]["candidates"]))
    calibration_a = read_table(resolve(args.calibration_a))
    calibration_b = read_table(resolve(args.calibration_b))
    blind, completed = make_remaining_sheet(candidates, calibration_a, calibration_b, seed)
    output = resolve(args.out or paths["gold"]["remaining_blind_sheet"])
    report = resolve(args.report or paths["gold"]["remaining_blind_report"])
    if args.validate_existing:
        if not output.exists():
            raise FileNotFoundError(f"Blind sheet not found for validation: {output}")
        written = read_table(output)
        # Compare the deterministic expected sheet to the existing artefact.
        # Treat empty CSV fields equivalently whether pandas reads them as NaN
        # or as empty strings.
        expected = blind.fillna("").astype(str)
        actual = written.fillna("").astype(str)
        if not actual.equals(expected):
            raise AssertionError("Existing blind sheet differs from deterministic expected output.")
    else:
        write_table(blind, output)
        # Read back so the report attests to the actual file, not merely the frame.
        written = read_table(output)
    if len(written) != EXPECTED_REMAINING or list(written.columns) != list(blind.columns):
        raise AssertionError("Written blind-sheet validation failed.")
    source_text = candidates.set_index(candidates["id"].astype(str))["text"].astype(str)
    written_text = written.set_index(written["id"].astype(str))["text"].astype(str)
    if not written_text.eq(source_text.reindex(written_text.index)).all():
        raise AssertionError("Written blind-sheet text differs from candidates.csv.")
    payload = {
        "source_row_count": len(candidates),
        "completed_calibration_count": len(completed),
        "completed_id_intersection_count": len(completed),
        "blind_row_count": len(written),
        "excluded_count": len(completed),
        "duplicate_id_count": int(written["id"].duplicated().sum()),
        "randomization_seed": seed,
        "output_columns": list(written.columns),
        "completed_ids_present_in_blind_sheet": int(written["id"].astype(str).isin(completed).sum()),
        "text_preserved_exactly": True,
        "empty_annotation_fields": {
            column: bool(written[column].fillna("").astype(str).eq("").all())
            for column in ("label", "confidence", "notes")
        },
        "gold_labels_assigned": False,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    action = "Validated existing blind sheet" if args.validate_existing else "Wrote blind sheet"
    print(f"{action} -> {output.relative_to(resolve('.'))}")
    print(f"Wrote validation report -> {report.relative_to(resolve('.'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
