"""Fine-tune bert-base-uncased on the frozen gold splits and evaluate on test.

    python scripts/models/train_bert.py

Baseline configuration (the defaults): bert-base-uncased, max_length=256,
learning_rate=2e-5, epochs=3, seed=42. The best checkpoint is selected on
validation macro F1; the test split is touched exactly once, after training,
and never influences model selection.

A plain PyTorch loop is used rather than ``transformers.Trainer`` so that
checkpoint selection, seeding and the epoch history stay explicit and stable
across Transformers major versions.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.bert import pipeline as P  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402
from src.utils.io import write_table  # noqa: E402
from src.utils.seeding import set_seed  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--tokenizer-name", default=None,
                        help="Defaults to --model-name.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--splits-dir", default=None,
                        help="Defaults to gold.splits_dir from configs/paths.yaml.")
    parser.add_argument("--output-dir", default="data/results/bert",
                        help="Where metrics, history and predictions are written.")
    parser.add_argument("--model-dir", default="models/bert",
                        help="Where the best checkpoint is saved (models/bert/best_model).")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-records", type=int, default=None,
                        help="Smoke-test escape hatch; leave unset for the real run.")
    parser.add_argument("--no-strict-sizes", action="store_true",
                        help="Skip the frozen split size check (smoke tests only).")
    return parser.parse_args(argv)


def resolve_device(requested: str):
    import torch

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_environment(device) -> dict:
    import torch
    import transformers

    info = {
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    try:
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except ImportError:  # pragma: no cover - sklearn is a hard dependency
        pass
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


def make_loader(dataset, collator, batch_size: int, shuffle: bool, seed: int, num_workers: int):
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=generator if shuffle else None,
        num_workers=num_workers,
    )


def evaluate(model, loader, device):
    """Return ``(mean loss, logits)`` over a loader without updating the model."""
    import torch

    model.eval()
    losses, logits_all, n_seen = [], [], 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            bs = batch["labels"].shape[0]
            losses.append(float(out.loss.detach()) * bs)
            n_seen += bs
            logits_all.append(out.logits.detach().cpu().numpy())
    return sum(losses) / max(n_seen, 1), np.concatenate(logits_all, axis=0)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()

    try:
        import torch
        from torch.optim import AdamW
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        print(f"torch/transformers are required: {exc}\n"
              "Install with: pip install -r requirements-ml.txt")
        return 1

    set_seed(args.seed)
    torch.use_deterministic_algorithms(False)  # CPU/GPU kernels stay default-fast
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    paths = load_config("paths.yaml")
    splits_dir = resolve(args.splits_dir or paths["gold"]["splits_dir"])
    out_dir = resolve(args.output_dir)
    model_dir = resolve(args.model_dir)
    best_dir = model_dir / "best_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] validating frozen splits in {splits_dir}")
    splits = P.load_splits(splits_dir, strict_sizes=not args.no_strict_sizes)
    print(f"[data] {splits.counts()}")

    train_df = splits.train
    if args.max_train_records:
        train_df = train_df.head(args.max_train_records)

    device = resolve_device(args.device)
    env = describe_environment(device)
    print(f"[env] {env}")

    tokenizer_name = args.tokenizer_name or args.model_name
    label2id, id2label = P.label_mapping()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(P.LABEL_ORDER),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    datasets = {
        "train": P.build_tokenized_dataset(
            train_df["text"], train_df["label"], tokenizer, args.max_length),
        "validation": P.build_tokenized_dataset(
            splits.validation["text"], splits.validation["label"], tokenizer, args.max_length),
        "test": P.build_tokenized_dataset(
            splits.test["text"], splits.test["label"], tokenizer, args.max_length),
    }
    train_loader = make_loader(datasets["train"], collator, args.batch_size, True,
                               args.seed, args.num_workers)
    val_loader = make_loader(datasets["validation"], collator, args.batch_size, False,
                             args.seed, args.num_workers)
    test_loader = make_loader(datasets["test"], collator, args.batch_size, False,
                              args.seed, args.num_workers)

    accum = max(1, args.gradient_accumulation_steps)
    steps_per_epoch = (len(train_loader) + accum - 1) // accum
    total_steps = steps_per_epoch * args.epochs
    no_decay = ("bias", "LayerNorm.weight")
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps)

    run_cfg = P.RunConfig(
        model_name=args.model_name,
        tokenizer_name=tokenizer_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=accum,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        device=str(device),
    )
    config_payload = {
        **run_cfg.to_dict(),
        "effective_batch_size": args.batch_size * accum,
        "steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "scheduler": "linear_with_warmup",
        "splits_dir": str(splits_dir),
        "split_sizes": splits.counts(),
        "feature_columns": list(P.MODEL_INPUT_COLUMNS),
        "excluded_columns": ["confidence", "notes", "source", "stratum",
                             "candidate_reason", "annotation provenance"],
        "environment": env,
    }
    (out_dir / "bert_config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8")

    history: list[dict] = []
    best = {"epoch": None, "macro_f1": -1.0}
    y_val_true = splits.validation["label"].tolist()

    print(f"[train] {args.epochs} epochs x {len(train_loader)} batches "
          f"(bs={args.batch_size}, accum={accum}) on {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_started = time.time()
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            bs = batch["labels"].shape[0]
            running += float(out.loss.detach()) * bs
            seen += bs
            if step % accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 20 == 0 or step == len(train_loader):
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running / max(seen, 1):.4f} "
                      f"({time.time() - epoch_started:.0f}s)", flush=True)

        train_loss = running / max(seen, 1)
        val_loss, val_logits = evaluate(model, val_loader, device)
        val_pred, _ = P.decode_predictions(val_logits)
        val_metrics = P.compute_metrics(y_val_true, val_pred)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "epoch_seconds": time.time() - epoch_started,
        }
        history.append(record)
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_macro_f1={val_metrics['macro_f1']:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f}", flush=True)

        if val_metrics["macro_f1"] > best["macro_f1"]:
            best = {"epoch": epoch, "macro_f1": val_metrics["macro_f1"],
                    "metrics": val_metrics}
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            (best_dir / "selection.json").write_text(
                json.dumps({"selected_epoch": epoch,
                            "selection_metric": "validation_macro_f1",
                            "validation_metrics": val_metrics}, indent=2),
                encoding="utf-8")
            print(f"[select] new best checkpoint (epoch {epoch}, "
                  f"val macro F1 {val_metrics['macro_f1']:.4f})")

        record["is_best_so_far"] = best["epoch"] == epoch

    train_seconds = time.time() - started
    (out_dir / "bert_training_history.json").write_text(
        json.dumps({"history": history,
                    "best_epoch": best["epoch"],
                    "best_val_macro_f1": best["macro_f1"],
                    "selection_metric": "validation_macro_f1",
                    "training_seconds": train_seconds}, indent=2),
        encoding="utf-8")

    # ---- Final test evaluation on the best validation checkpoint -------------
    print(f"[test] loading best checkpoint from epoch {best['epoch']}")
    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    test_loss, test_logits = evaluate(model, test_loader, device)
    y_test_true = splits.test["label"].tolist()
    y_test_pred, test_probs = P.decode_predictions(test_logits)

    test_metrics = P.compute_metrics(y_test_true, y_test_pred)
    report = P.classification_report_dict(y_test_true, y_test_pred)
    matrix = P.confusion_frame(y_test_true, y_test_pred)
    implicit = P.implicit_crisis_analysis(y_test_true, y_test_pred)
    predictions = P.prediction_frame(splits.test, y_test_pred, test_probs)
    errors = P.error_frame(predictions)

    per_class = {
        label: {
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in P.LABEL_ORDER
    }
    metrics_payload = {
        "split": "test",
        "n_examples": len(y_test_true),
        "n_incorrect": int(len(errors)),
        "test_loss": test_loss,
        "overall": test_metrics,
        "per_class": per_class,
        "implicit_crisis_analysis": implicit,
        "confusion_matrix": {"labels": list(P.LABEL_ORDER),
                             "rows_are_true_labels": True,
                             "matrix": matrix.values.tolist()},
        "selected_checkpoint": {"epoch": best["epoch"],
                                "validation_macro_f1": best["macro_f1"],
                                "path": str(best_dir)},
        "label_mapping": dict(P.LABEL2ID),
        "training_seconds": train_seconds,
    }

    (out_dir / "bert_test_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (out_dir / "bert_classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    matrix.to_csv(out_dir / "bert_confusion_matrix.csv", encoding="utf-8")
    write_table(predictions, out_dir / "bert_test_predictions.csv")
    write_table(errors, out_dir / "bert_errors.csv")

    print("\n=== TEST ===")
    print(json.dumps(test_metrics, indent=2))
    print("\nPer class:")
    for label, row in per_class.items():
        print(f"  {label:<16} P={row['precision']:.4f} R={row['recall']:.4f} "
              f"F1={row['f1']:.4f} support={row['support']}")
    print("\nConfusion matrix (rows = gold):\n" + matrix.to_string())
    print("\nImplicit-crisis analysis:\n" + json.dumps(implicit, indent=2))
    print(f"\nErrors: {len(errors)}/{len(y_test_true)}")
    print(f"Results written to {out_dir}")
    print(f"Best checkpoint: {best_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
