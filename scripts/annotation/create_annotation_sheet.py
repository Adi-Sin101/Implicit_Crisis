"""Turn candidates into blind, per-annotator annotation sheets.

    python scripts/annotation/create_annotation_sheet.py --annotators A B
    python scripts/annotation/create_annotation_sheet.py --annotators A B --calibration 150

The sheets contain only id / text / label / confidence / notes. Stratum,
source risk label and lexicon hits are written to a separate key file that
annotators do not open, so their judgement is not anchored by them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.annotation import sheets  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--annotators", nargs="+", default=["A", "B"])
    parser.add_argument("--calibration", type=int, default=0,
                        help="If set, emit a shared calibration subset of this size")
    args = parser.parse_args()

    paths = load_config("paths.yaml")
    seed = load_config("candidate_pool.yaml").get("seed", 42)
    set_seed(seed)

    cand_path = resolve(args.candidates or paths["gold"]["candidates"])
    if not cand_path.exists():
        print(f"Candidates not found at {cand_path}. "
              "Run scripts/annotation/build_gold_candidates.py first.")
        return 1
    candidates = read_table(cand_path)

    outdir = resolve(args.outdir or paths["gold"]["annotation_sheets"])
    outdir.mkdir(parents=True, exist_ok=True)

    # The withheld metadata, kept out of every annotator-facing file.
    write_table(sheets.make_key(candidates), outdir / "candidate_key.csv")

    if args.calibration:
        subset = candidates.sample(min(args.calibration, len(candidates)), random_state=seed)
        sheet = sheets.make_sheet(subset, seed=seed)
        sheets.assert_blind(sheet)
        for annotator in args.annotators:
            path = outdir / f"calibration_{len(subset)}_{annotator}.csv"
            write_table(sheet.copy(), path)
            print(f"calibration sheet -> {path.relative_to(resolve('.'))} ({len(sheet)} items)")
    else:
        sheet = sheets.make_sheet(candidates, seed=seed)
        sheets.assert_blind(sheet)
        for annotator in args.annotators:
            path = outdir / f"annotation_{annotator}.csv"
            write_table(sheet.copy(), path)
            print(f"annotation sheet -> {path.relative_to(resolve('.'))} ({len(sheet)} items)")

    print(f"\nAllowed labels: {sheets.label_options()}")
    print("Guidelines: annotation/guidelines/annotation_guidelines.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
