"""Compute inter-annotator agreement over completed annotation files.

    python scripts/annotation/calculate_agreement.py \
        --files A=data/gold/annotation/calibration_150_A.csv \
                B=data/gold/annotation/calibration_150_B.csv

Writes the kappa table, the per-class breakdown and the disagreement matrix to
``annotation/agreement/``. All figures are computed from the supplied files;
nothing is assumed or filled in.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.annotation import agreement  # noqa: E402
from src.utils.config import resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True,
                        help="NAME=path pairs, one per annotator")
    parser.add_argument("--outdir", default="annotation/agreement")
    args = parser.parse_args()

    frames = {}
    for spec in args.files:
        if "=" not in spec:
            print(f"Expected NAME=path, got '{spec}'")
            return 1
        name, path = spec.split("=", 1)
        frames[name] = read_table(resolve(path))

    aligned = agreement.align(frames)
    annotators = list(frames)
    print(f"{len(aligned)} items annotated by all of {annotators}\n")

    outdir = resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    pairwise = agreement.pairwise_cohen(aligned, annotators)
    pairwise["interpretation"] = pairwise["cohens_kappa"].map(agreement.interpret)
    print("Pairwise Cohen's kappa:")
    print(pairwise.to_string(index=False))
    write_table(pairwise, outdir / "pairwise_kappa.csv")

    if len(annotators) > 2:
        matrix = agreement.rating_matrix(aligned, annotators)
        kappa = agreement.fleiss_kappa(matrix)
        print(f"\nFleiss' kappa: {kappa:.3f} ({agreement.interpret(kappa)})")

    per_class = agreement.per_class_agreement(aligned, annotators)
    print("\nPer-class agreement (one-vs-rest):")
    print(per_class.to_string(index=False))
    write_table(per_class, outdir / "per_class_kappa.csv")

    a, b = annotators[0], annotators[1]
    matrix = agreement.disagreement_matrix(aligned, a, b)
    print(f"\nDisagreement matrix ({a} rows x {b} columns):")
    print(matrix.to_string())
    matrix.to_csv(outdir / "disagreement_matrix.csv", encoding="utf-8")

    print(f"\nResults written to {outdir.relative_to(resolve('.'))}/")
    print("If agreement is low on a class, revise annotation/guidelines/ and re-run "
          "calibration before annotating the full batch.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
