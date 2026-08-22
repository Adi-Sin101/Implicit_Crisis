"""Train and evaluate the TF-IDF + Logistic Regression baseline.

    python scripts/experiments/run_tfidf_baseline.py

Trains on the gold train split, tunes nothing on the test split, and writes
predictions and metrics to ``experiments/baselines/tfidf/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import metrics  # noqa: E402
from src.models.tfidf import baseline  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="experiments/baselines/tfidf")
    parser.add_argument("--eval-split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    exp = load_config("experiment.yaml")
    paths = load_config("paths.yaml")
    set_seed(exp.get("seed", 42))

    splits_dir = resolve(paths["gold"]["splits_dir"])
    train_path = splits_dir / "train.csv"
    eval_path = splits_dir / f"{args.eval_split}.csv"
    if not train_path.exists() or not eval_path.exists():
        print(f"Gold splits not found in {splits_dir}. Run "
              "scripts/annotation/validate_gold_dataset.py --split first.")
        return 1

    train = read_table(train_path)
    evalset = read_table(eval_path)

    pipeline = baseline.build_pipeline(exp)
    pipeline.fit(train["text"], train["label"])
    y_pred = pipeline.predict(evalset["text"])

    overall = metrics.overall_report(evalset["label"], y_pred)
    per_class = metrics.per_class_report(evalset["label"], y_pred)
    slices = metrics.slice_report(evalset["label"], y_pred)
    matrix = metrics.confusion(evalset["label"], y_pred)

    print(json.dumps(overall, indent=2))
    print("\nPer class:\n" + per_class.to_string(index=False))
    print("\nPer evaluation slice (the primary result):\n" + slices.to_string(index=False))
    print("\nConfusion matrix (rows = gold):\n" + matrix.to_string())

    outdir = resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    write_table(per_class, outdir / "per_class.csv")
    write_table(slices, outdir / "slices.csv")
    matrix.to_csv(outdir / "confusion_matrix.csv", encoding="utf-8")
    write_table(
        evalset.assign(prediction=y_pred)[["id", "text", "label", "prediction"]],
        outdir / f"predictions_{args.eval_split}.csv",
    )
    print(f"\nWritten to {outdir.relative_to(resolve('.'))}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
