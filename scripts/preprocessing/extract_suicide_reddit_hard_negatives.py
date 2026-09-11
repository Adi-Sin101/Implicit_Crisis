"""Extract unlabelled hard-negative-like candidates from the archived Reddit source.

This script never writes a project ``label`` and never alters the raw archive,
the existing gold candidates, or annotation sheets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.extraction.suicide_reddit_hard_negatives import extract_candidates  # noqa: E402
from src.data.loaders.suicide_reddit_dataset import inspect_archive, load_records  # noqa: E402
from src.data.pooling.lexicon import load_terms  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402


def _read_if_exists(path: Path):
    return read_table(path) if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=None, help="Override raw RAR archive path.")
    parser.add_argument("--pool", default=None, help="Existing cleaned pool for exact deduplication.")
    parser.add_argument("--existing-candidates", default=None, help="Existing gold candidate file for exact deduplication.")
    parser.add_argument("--out", default=None, help="Output candidate CSV path.")
    parser.add_argument("--report", default=None, help="Output JSON report path.")
    parser.add_argument("--limit", type=int, default=250, help="Maximum selected rows (default: 250).")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    paths = load_config("paths.yaml")
    archive = resolve(args.archive or paths["raw"]["suicide_reddit_dataset"])
    pool = _read_if_exists(resolve(args.pool or paths["interim"]["cleaned_pool"]))
    existing = _read_if_exists(resolve(args.existing_candidates or paths["gold"]["candidates"]))
    records = load_records(archive)
    source_file_audit = inspect_archive(archive)
    candidates, stats = extract_candidates(records, pool, existing, terms=load_terms(), limit=args.limit)

    output = resolve(args.out or paths["interim"]["suicide_reddit_hard_negative_candidates"])
    report = resolve(args.report or paths["interim"]["suicide_reddit_hard_negative_report"])
    write_table(candidates, output)
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "suicide_reddit_dataset",
        "archive": str(archive.relative_to(resolve("."))),
        "selection_is_not_a_gold_label": True,
        "source_file_audit": source_file_audit,
        "stats": stats,
        "candidate_reason_distribution": candidates["candidate_reason"].value_counts().to_dict(),
        "source_label_distribution": candidates["source_label"].value_counts().to_dict(),
    }
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {len(candidates):,} unlabelled candidates -> {output.relative_to(resolve('.'))}")
    print(f"Wrote extraction report -> {report.relative_to(resolve('.'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
