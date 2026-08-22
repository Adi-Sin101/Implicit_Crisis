"""Compare model predictions side by side and run the error analysis.

    python scripts/experiments/evaluate_models.py \
        --predictions tfidf=experiments/baselines/tfidf/predictions_test.csv \
                      bert=experiments/bert/predictions_test.csv

Produces the Section 6 comparison table from measured predictions only, plus
the four error-analysis buckets from Step 8. No result is ever filled in from
anywhere but the prediction files supplied.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import error_analysis, metrics  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="NAME=path pairs; each file needs text/label/prediction columns")
    parser.add_argument("--outdir", default="experiments/comparisons")
    parser.add_argument("--error-outdir", default="experiments/error_analysis")
    args = parser.parse_args()

    frames = {}
    for spec in args.predictions:
        if "=" not in spec:
            print(f"Expected NAME=path, got '{spec}'")
            return 1
        name, path = spec.split("=", 1)
        path = resolve(path)
        if not path.exists():
            print(f"Prediction file not found: {path}")
            return 1
        frames[name] = read_table(path)

    names = list(frames)
    reference = frames[names[0]]
    for name, frame in frames.items():
        if len(frame) != len(reference) or not (frame["id"].values == reference["id"].values).all():
            print(f"'{name}' predictions are not aligned with '{names[0]}' - "
                  "all models must be evaluated on the identical test set.")
            return 1

    results = {}
    for name, frame in frames.items():
        results[name] = {
            "overall": metrics.overall_report(frame["label"], frame["prediction"]),
            "slices": metrics.slice_report(frame["label"], frame["prediction"]),
        }
        print(f"\n=== {name} ===")
        print(results[name]["slices"].to_string(index=False))

    table = metrics.comparison_table(results)
    print("\n=== Model comparison ===")
    print(table.to_string(index=False))

    outdir = resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_table(table, outdir / "model_comparison.csv")

    if len(names) >= 2:
        a, b = names[0], names[1]
        frame = error_analysis.build_frame(
            reference["text"], reference["label"],
            {name: frames[name]["prediction"].tolist() for name in names},
        )
        err_dir = resolve(args.error_outdir)
        err_dir.mkdir(parents=True, exist_ok=True)

        print("\n=== Error analysis ===")
        print(error_analysis.summary(frame, a, b).to_string(index=False))
        for bucket, block in error_analysis.disagreements(frame, a, b).items():
            write_table(block, err_dir / f"{bucket}.csv")
        for name in names:
            write_table(error_analysis.missed_implicit(frame, name),
                        err_dir / f"missed_implicit_{name}.csv")
            write_table(error_analysis.hard_negative_false_positives(frame, name),
                        err_dir / f"hard_negative_fp_{name}.csv")
        print(f"\nWritten to {err_dir.relative_to(resolve('.'))}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
