"""Draw a stratified candidate sample from the cleaned pool for annotation.

    python scripts/annotation/build_gold_candidates.py

Reads ``data/interim/preprocessing/pool_clean.csv`` (run
``scripts/preprocessing/clean_datasets.py`` first) and writes
``data/gold/candidates/candidates.csv``.

The strata used here are sampling buckets. They select what a human reads;
they are not labels and are never copied into the gold dataset.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pooling import build_pool  # noqa: E402
from src.data.pooling.strata import stratum_report  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild the pool from data/raw/ instead of reading the cached pool")
    args = parser.parse_args()

    config = load_config("candidate_pool.yaml")
    paths = load_config("paths.yaml")
    set_seed(config.get("seed", 42))

    if args.rebuild:
        pool = build_pool.build(config)
    else:
        pool_path = resolve(args.pool or paths["interim"]["cleaned_pool"])
        if not pool_path.exists():
            print(f"Pool not found at {pool_path}.\n"
                  "Run scripts/preprocessing/clean_datasets.py first, or pass --rebuild.")
            return 1
        pool = read_table(pool_path)

    targets = {k: v.get("target_n", 0) for k, v in (config.get("strata") or {}).items()}
    candidates = build_pool.sample_candidates(pool, config)

    out_path = resolve(args.out or paths["gold"]["candidates"])
    write_table(candidates, out_path)
    print(f"Wrote {len(candidates):,} candidates -> {out_path.relative_to(resolve('.'))}\n")

    print(stratum_report(candidates).to_string(index=False))
    for stratum, want in targets.items():
        got = int((candidates["stratum"] == stratum).sum())
        if got < want:
            print(f"  NOTE: {stratum} short of target ({got}/{want}) - pool exhausted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
