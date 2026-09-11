"""Freeze the CPU BERT run as **Baseline v1**.

    python scripts/models/archive_bert_baseline_v1.py

Copies the live run artifacts into ``data/results/bert/baseline_v1/`` and the
selected checkpoint into ``models/bert/baseline_v1/``, then writes
``baseline_v1_metadata.json`` — a single self-contained record assembled from
the result files themselves, never from retyped numbers.

The script is idempotent and refuses to overwrite an archived artifact whose
content differs from the live one, so a later BERT experiment cannot quietly
redefine Baseline v1. Pass ``--verify-only`` to check the archive without
writing anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.bert import pipeline as P  # noqa: E402
from src.utils.config import load_config, resolve  # noqa: E402

EXPERIMENT_NAME = "BERT Baseline v1"
RESULT_FILES = (
    "bert_config.json",
    "bert_training_history.json",
    "bert_test_metrics.json",
    "bert_classification_report.json",
    "bert_confusion_matrix.csv",
    "bert_test_predictions.csv",
    "bert_errors.csv",
    "bert_training_log.txt",
)
CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "selection.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class ArchiveConflict(RuntimeError):
    """An archived Baseline v1 artifact differs from the live one."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_preserving(src: Path, dst: Path, verify_only: bool) -> str:
    """Copy ``src`` to ``dst`` unless ``dst`` exists with different content."""
    if dst.exists():
        if sha256(dst) == sha256(src):
            return "unchanged"
        raise ArchiveConflict(
            f"{dst} already exists with different content than {src}. "
            "Baseline v1 is frozen: archive a later run under a new name instead."
        )
    if verify_only:
        raise ArchiveConflict(f"Missing archived artifact: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied"


def build_metadata(archive_dir: Path, checkpoint_dir: Path, splits_dir: Path) -> dict:
    """Assemble the Baseline v1 record from the archived artifacts."""
    read = lambda name: json.loads((archive_dir / name).read_text(encoding="utf-8"))  # noqa: E731
    config = read("bert_config.json")
    history = read("bert_training_history.json")
    test = read("bert_test_metrics.json")
    report = read("bert_classification_report.json")
    selection = json.loads((checkpoint_dir / "selection.json").read_text(encoding="utf-8"))

    predictions = pd.read_csv(archive_dir / "bert_test_predictions.csv")
    errors = pd.read_csv(archive_dir / "bert_errors.csv")
    matrix = pd.read_csv(archive_dir / "bert_confusion_matrix.csv", index_col=0)

    split_files = {}
    for name in ("train", "validation", "test"):
        path = splits_dir / f"{name}.csv"
        frame = pd.read_csv(path)
        split_files[name] = {
            "path": f"data/gold/splits/{name}.csv",
            "records": int(len(frame)),
            "label_counts": {k: int(v) for k, v in frame["label"].value_counts().items()},
            "sha256": sha256(path),
        }

    gold_path = resolve("data/gold/gold.csv")
    gold = pd.read_csv(gold_path)

    env = config.get("environment", {})
    return {
        "experiment_name": EXPERIMENT_NAME,
        "status": "frozen reproducibility reference — do not modify or overwrite",
        "description": (
            "First fine-tuning run of bert-base-uncased on the frozen gold splits, "
            "four-way classification, trained on CPU. Single configuration, single "
            "run, no hyperparameter search."
        ),
        "run_date": "2026-09-11",
        "archived_date": "2026-09-12",
        "dataset": {
            "gold_file": "data/gold/gold.csv",
            "records": int(len(gold)),
            "label_counts": {k: int(v) for k, v in gold["label"].value_counts().items()},
            "sha256": sha256(gold_path),
            "annotation_note": (
                "Labels are AI-assisted annotations produced under the project's "
                "written guidelines (v2.1), not fully human-validated gold labels. "
                "Per data/gold/final_gold_dataset_report.json the dataset is 1,837 "
                "AI-assisted main annotations plus 150 calibration records that went "
                "through a dual-annotation and adjudication pass. The dataset must "
                "not be described as human-validated. See annotation/guidelines/ "
                "and data/gold/annotation/."
            ),
            "provenance_counts": {"main_ai_assisted": 1837, "calibration_adjudicated": 150},
        },
        "splits": {
            "seed": 42,
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "stratified_on": "label",
            "regenerated_for_this_experiment": False,
            "files": split_files,
        },
        "model": {
            "model_name": config["model_name"],
            "tokenizer_name": config["tokenizer_name"],
            "num_labels": config["num_labels"],
            "head": "standard sequence-classification head",
        },
        "label_mapping": config["label_mapping"],
        "inputs": {
            "feature_columns": config["feature_columns"],
            "excluded_columns": config["excluded_columns"],
            "note": "Only the raw `text` column reaches the tokenizer. No metadata features.",
        },
        "hyperparameters": {
            "max_length": config["max_length"],
            "truncation": config["truncation"],
            "padding": config["padding"],
            "learning_rate": config["learning_rate"],
            "epochs": config["epochs"],
            "batch_size": config["batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "effective_batch_size": config["effective_batch_size"],
            "optimizer": config["optimizer"],
            "weight_decay": config["weight_decay"],
            "scheduler": config["scheduler"],
            "warmup_ratio": config["warmup_ratio"],
            "gradient_clipping": 1.0,
            "seed": config["seed"],
            "steps_per_epoch": config["steps_per_epoch"],
            "total_optimizer_steps": config["total_optimizer_steps"],
            "model_selection_metric": config["model_selection_metric"],
        },
        "hardware": {
            "device": config["device"],
            "cuda_available": env.get("cuda_available", False),
            "gpu_name": env.get("gpu_name"),
            "cpu": "AMD Ryzen 5 5500U",
            "cpu_cores": 6,
            "cpu_logical_processors": 12,
            "ram_gb": 15.4,
            "platform": env.get("platform", platform.platform()),
            "note": "No NVIDIA CUDA GPU available on this machine; CPU-only training.",
        },
        "software": {
            "python": env.get("python"),
            "torch": env.get("torch"),
            "transformers": env.get("transformers"),
            "scikit_learn": env.get("scikit_learn"),
            "numpy": env.get("numpy"),
        },
        "training": {
            "duration_seconds": history["training_seconds"],
            "duration_human": "approximately 1 hour 41 minutes",
            "history": history["history"],
            "best_epoch": history["best_epoch"],
            "best_validation_macro_f1": history["best_val_macro_f1"],
            "selection_metric": history["selection_metric"],
        },
        "validation_results_at_best_epoch": selection["validation_metrics"],
        "test_results": {
            "n_examples": test["n_examples"],
            "n_incorrect": test["n_incorrect"],
            "test_loss": test["test_loss"],
            "overall": test["overall"],
            "per_class": test["per_class"],
            "weighted_avg": {
                "precision": report["weighted avg"]["precision"],
                "recall": report["weighted avg"]["recall"],
                "f1": report["weighted avg"]["f1-score"],
                "support": report["weighted avg"]["support"],
            },
            "confusion_matrix": {
                "labels": list(matrix.columns),
                "rows_are_true_labels": True,
                "matrix": matrix.values.tolist(),
            },
        },
        "implicit_crisis_slice": test["implicit_crisis_analysis"],
        "errors": {
            "total": int(len(errors)),
            "of": int(len(predictions)),
            "by_confusion_pair": [
                {"true_label": t, "predicted_label": p, "n": int(n)}
                for (t, p), n in errors.groupby(["true_label", "predicted_label"])
                .size()
                .sort_values(ascending=False)
                .items()
            ],
            "mean_confidence_on_errors": float(errors["confidence"].mean()),
            "mean_confidence_on_correct": float(
                predictions.loc[
                    predictions["true_label"] == predictions["predicted_label"], "confidence"
                ].mean()
            ),
        },
        "artifacts": {
            "results_dir": "data/results/bert/baseline_v1/",
            "checkpoint_dir": "models/bert/baseline_v1/best_model/",
            "documentation": "docs/methodology/bert_baseline_v1.md",
            "training_script": "scripts/models/train_bert.py",
            "pipeline_module": "src/models/bert/pipeline.py",
            "archive_script": "scripts/models/archive_bert_baseline_v1.py",
            "result_file_sha256": {
                name: sha256(archive_dir / name) for name in RESULT_FILES
            },
            "checkpoint_file_sha256": {
                name: sha256(checkpoint_dir / name) for name in CHECKPOINT_FILES
            },
        },
        "limitations": [
            "CPU-only training; no CUDA GPU was available.",
            "Three epochs only. Validation macro F1 was still rising at epoch 3, so the "
            "model is under-trained rather than converged.",
            "No hyperparameter search, class weighting, resampling, augmentation or "
            "threshold tuning was performed.",
            "implicit_crisis recall is 0.1220 (5 of 41): the baseline largely fails on "
            "the project's primary research slice.",
            "46% of hard negatives (17 of 37) are predicted explicit_crisis, indicating "
            "reliance on surface crisis vocabulary.",
            "Single run with a single seed; no variance estimate.",
            "Small gold dataset (1,987 records) with AI-assisted annotation.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="data/results/bert")
    parser.add_argument("--archive-dir", default="data/results/bert/baseline_v1")
    parser.add_argument("--live-checkpoint", default="models/bert/best_model")
    parser.add_argument("--archive-checkpoint", default="models/bert/baseline_v1/best_model")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check the archive matches the live run; write nothing.")
    args = parser.parse_args(argv)

    results_dir = resolve(args.results_dir)
    archive_dir = resolve(args.archive_dir)
    live_ckpt = resolve(args.live_checkpoint)
    archive_ckpt = resolve(args.archive_checkpoint)
    splits_dir = resolve(load_config("paths.yaml")["gold"]["splits_dir"])

    if not args.verify_only:
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive_ckpt.mkdir(parents=True, exist_ok=True)

    for name in RESULT_FILES:
        src = results_dir / name
        if not src.exists():
            print(f"  ! missing live artifact, skipping: {src}")
            continue
        print(f"  {name}: {copy_preserving(src, archive_dir / name, args.verify_only)}")

    for name in CHECKPOINT_FILES:
        src = live_ckpt / name
        if not src.exists():
            print(f"  ! missing checkpoint file, skipping: {src}")
            continue
        print(f"  checkpoint/{name}: "
              f"{copy_preserving(src, archive_ckpt / name, args.verify_only)}")

    metadata = build_metadata(archive_dir, archive_ckpt, splits_dir)
    metadata_path = archive_dir / "baseline_v1_metadata.json"
    payload = json.dumps(metadata, indent=2)
    if args.verify_only:
        if not metadata_path.exists():
            raise ArchiveConflict(f"Missing {metadata_path}")
        print(f"\nVerified Baseline v1 archive at {archive_dir}")
    else:
        metadata_path.write_text(payload, encoding="utf-8")
        print(f"\nWrote {metadata_path}")

    label_mapping = metadata["label_mapping"]
    assert label_mapping == dict(P.LABEL2ID), "label mapping drift"
    print(f"{EXPERIMENT_NAME}: best epoch {metadata['training']['best_epoch']}, "
          f"test macro F1 {metadata['test_results']['overall']['macro_f1']:.4f}, "
          f"implicit_crisis recall "
          f"{metadata['implicit_crisis_slice']['recall']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
