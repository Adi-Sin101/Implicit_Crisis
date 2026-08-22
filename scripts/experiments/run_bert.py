"""Fine-tune BERT on the gold dataset and evaluate it on the held-out split.

    python scripts/experiments/run_bert.py

Uses the same train/val/test splits and the same seed as the TF-IDF baseline,
so the comparison is controlled. Requires torch + transformers (see
requirements-ml.txt); everything upstream of this script runs without them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation import metrics  # noqa: E402
from src.models.bert import classifier  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import read_table, write_table  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="experiments/bert")
    parser.add_argument("--eval-split", default="test", choices=["val", "test"])
    parser.add_argument("--save-model", default=None,
                        help="Directory for the fine-tuned weights (gitignored)")
    args = parser.parse_args()

    exp = load_config("experiment.yaml")
    paths = load_config("paths.yaml")
    set_seed(exp.get("seed", 42))
    cfg = classifier.BertConfig.from_dict(exp)

    try:
        from transformers import Trainer, TrainingArguments
    except ImportError:
        print("transformers/torch are not installed. "
              "Install them with: pip install -r requirements-ml.txt")
        return 1

    splits_dir = resolve(paths["gold"]["splits_dir"])
    needed = {name: splits_dir / f"{name}.csv" for name in ("train", "val", args.eval_split)}
    missing = [str(p) for p in needed.values() if not p.exists()]
    if missing:
        print("Gold splits not found: " + ", ".join(missing))
        print("Run scripts/annotation/validate_gold_dataset.py --split first.")
        return 1

    train = read_table(needed["train"])
    val = read_table(needed["val"])
    evalset = read_table(needed[args.eval_split])

    tokenizer, model = classifier.load_model(cfg)
    train_ds = classifier.build_dataset(
        train["text"].tolist(), train["label"].tolist(), tokenizer, cfg.max_length)
    val_ds = classifier.build_dataset(
        val["text"].tolist(), val["label"].tolist(), tokenizer, cfg.max_length)

    outdir = resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(outdir / "checkpoints"),
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        seed=cfg.seed,
        logging_steps=50,
        report_to=[],
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()

    if args.save_model:
        model.save_pretrained(resolve(args.save_model))
        tokenizer.save_pretrained(resolve(args.save_model))

    y_pred = classifier.predict(model, tokenizer, evalset["text"].tolist(), cfg)

    overall = metrics.overall_report(evalset["label"], y_pred)
    per_class = metrics.per_class_report(evalset["label"], y_pred)
    slices = metrics.slice_report(evalset["label"], y_pred)
    matrix = metrics.confusion(evalset["label"], y_pred)

    print(json.dumps(overall, indent=2))
    print("\nPer class:\n" + per_class.to_string(index=False))
    print("\nPer evaluation slice (the primary result):\n" + slices.to_string(index=False))
    print("\nConfusion matrix (rows = gold):\n" + matrix.to_string())

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
