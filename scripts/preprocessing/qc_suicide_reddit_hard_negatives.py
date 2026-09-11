"""Create a separate QC review for the unlabelled Reddit candidate hand-off."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.qc.suicide_reddit_hard_negative_qc import review  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    paths = load_config("paths.yaml")
    source = resolve(args.input or paths["interim"]["suicide_reddit_hard_negative_candidates"])
    out_path = resolve(args.out or paths["interim"]["suicide_reddit_hard_negative_qc"])
    report_path = resolve(args.report or paths["interim"]["suicide_reddit_hard_negative_qc_report"])
    reviewed, counts = review(read_table(source))
    write_table(reviewed, out_path)
    excluded = reviewed[reviewed["qc_status"].isin(["obvious_current_self_crisis", "duplicate_or_near_duplicate"])]
    payload = {
        "input": str(source.relative_to(resolve("."))),
        "qc_statuses_are_not_gold_labels": True,
        "total_reviewed": len(reviewed),
        "qc_status_distribution": counts,
        "candidate_reason_distribution": reviewed["candidate_reason"].value_counts().to_dict(),
        "source_distribution": reviewed["source"].value_counts().to_dict(),
        "source_label_distribution": reviewed["source_label"].value_counts().to_dict(),
        "recommended_exclusions": [
            {
                "source_id": row.source_id,
                "text": row.text,
                "qc_status": row.qc_status,
                "reason": row.qc_explanation,
            }
            for row in excluded.itertuples(index=False)
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote QC rows -> {out_path.relative_to(resolve('.'))}")
    print(f"Wrote QC report -> {report_path.relative_to(resolve('.'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
