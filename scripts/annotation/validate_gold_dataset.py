"""Validate an annotated file and (optionally) write the gold splits.

    python scripts/annotation/validate_gold_dataset.py --input data/gold/gold.csv
    python scripts/annotation/validate_gold_dataset.py --input ... --split

Checks label vocabulary, blank labels, duplicate ids and class coverage, then
reports how gold labels relate to the sampling strata - a diagnostic of how
well the routing worked, never a substitute for the human label.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.annotation import validate  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None)
    parser.add_argument("--key", default=None, help="candidate_key.csv, for the strata cross-tab")
    parser.add_argument("--split", action="store_true", help="Write train/val/test splits")
    parser.add_argument("--lenient", action="store_true", help="Do not require all four classes")
    args = parser.parse_args()

    paths = load_config("paths.yaml")
    exp = load_config("experiment.yaml")
    set_seed(exp.get("seed", 42))

    in_path = resolve(args.input or paths["gold"]["gold_final"])
    if not in_path.exists():
        print(f"No gold file at {in_path}.")
        return 1
    gold = read_table(in_path)

    problems = validate.validate(gold, strict=not args.lenient)
    if problems:
        print("Validation FAILED:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"Validation passed: {len(gold):,} annotated items\n")
    print(validate.summarise(gold).to_string(index=False))

    key_path = resolve(args.key) if args.key else Path(
        resolve(paths["gold"]["annotation_sheets"])) / "candidate_key.csv"
    if Path(key_path).exists():
        key = read_table(key_path)
        if "stratum" in key.columns:
            merged = gold.merge(key[["id", "stratum"]], on="id", how="left")
            print("\nAnnotation label x sampling stratum (routing diagnostic only):")
            print(pd.crosstab(merged["stratum"], merged["label"]).to_string())

    if args.split:
        cfg = exp["split"]
        train_val, test = train_test_split(
            gold, test_size=cfg["test_size"], random_state=exp.get("seed", 42),
            stratify=gold["label"])
        train, val = train_test_split(
            train_val, test_size=cfg["val_size"], random_state=exp.get("seed", 42),
            stratify=train_val["label"])
        outdir = resolve(paths["gold"]["splits_dir"])
        for name, frame in (("train", train), ("val", val), ("test", test)):
            path = outdir / f"{name}.csv"
            write_table(frame, path)
            print(f"\n{name}: {len(frame):,} -> {path.relative_to(resolve('.'))}")
        print("\nThe test split must not be inspected or tuned on at any point.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
