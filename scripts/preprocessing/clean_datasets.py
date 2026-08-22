"""Load every external corpus, clean and deduplicate it, and write the pool.

    python scripts/preprocessing/clean_datasets.py

Output: ``data/interim/preprocessing/pool_clean.csv`` plus a stratum report.
Nothing under ``data/raw/`` is written to or modified.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.pooling import build_pool  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import write_table  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="candidate_pool.yaml")
    parser.add_argument("--out", default=None, help="Override the output path")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = load_config("paths.yaml")
    set_seed(config.get("seed", 42))

    print("Building candidate pool from data/raw/ ...")
    pool = build_pool.build(config)

    out_path = resolve(args.out or paths["interim"]["cleaned_pool"])
    write_table(pool, out_path)
    print(f"\nWrote {len(pool):,} rows -> {out_path.relative_to(resolve('.'))}")

    print("\nStratum distribution (sampling buckets, NOT labels):")
    from src.data.pooling.strata import stratum_report

    print(stratum_report(pool).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
