"""Guards on the frozen BERT Baseline v1 archive.

These tests fail if the archive goes missing, drifts from its recorded hashes,
or stops agreeing with its own predictions file — i.e. if a later BERT
experiment has overwritten the reproducibility reference.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.bert import pipeline as P
from src.utils.config import load_config, resolve

ARCHIVE = resolve("data/results/bert/baseline_v1")
CHECKPOINT = resolve("models/bert/baseline_v1/best_model")
SPLITS_DIR = resolve(load_config("paths.yaml")["gold"]["splits_dir"])
METADATA_PATH = ARCHIVE / "baseline_v1_metadata.json"

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

pytestmark = pytest.mark.skipif(
    not METADATA_PATH.exists(),
    reason="BERT Baseline v1 has not been archived in this checkout",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def metadata() -> dict:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def test_every_archived_result_file_is_present():
    for name in RESULT_FILES:
        assert (ARCHIVE / name).exists(), name


def test_archived_result_files_match_their_recorded_hashes(metadata):
    recorded = metadata["artifacts"]["result_file_sha256"]
    for name, expected in recorded.items():
        assert sha256(ARCHIVE / name) == expected, f"{name} has changed since archiving"


@pytest.mark.skipif(not CHECKPOINT.exists(),
                    reason="Baseline v1 checkpoint weights are not present (gitignored)")
def test_archived_checkpoint_matches_its_recorded_hashes(metadata):
    recorded = metadata["artifacts"]["checkpoint_file_sha256"]
    for name, expected in recorded.items():
        assert sha256(CHECKPOINT / name) == expected, f"checkpoint/{name} has changed"


@pytest.mark.skipif(not CHECKPOINT.exists(),
                    reason="Baseline v1 checkpoint weights are not present (gitignored)")
def test_checkpoint_was_selected_on_validation_macro_f1():
    selection = json.loads((CHECKPOINT / "selection.json").read_text(encoding="utf-8"))
    assert selection["selection_metric"] == "validation_macro_f1"
    assert selection["selected_epoch"] == 3


def test_metadata_pins_the_canonical_splits(metadata):
    files = metadata["splits"]["files"]
    assert metadata["splits"]["seed"] == 42
    assert metadata["splits"]["regenerated_for_this_experiment"] is False
    expected_sizes = {"train": 1391, "validation": 298, "test": 298}
    for name, expected in expected_sizes.items():
        assert files[name]["records"] == expected
        assert sha256(SPLITS_DIR / f"{name}.csv") == files[name]["sha256"], (
            f"{name}.csv has changed since Baseline v1 was archived"
        )


def test_metadata_pins_the_gold_dataset(metadata):
    gold_path = resolve("data/gold/gold.csv")
    assert metadata["dataset"]["records"] == 1987
    assert sha256(gold_path) == metadata["dataset"]["sha256"], (
        "gold.csv has changed since Baseline v1 was archived"
    )


def test_metadata_does_not_claim_human_validated_labels(metadata):
    note = metadata["dataset"]["annotation_note"].lower()
    assert "ai-assisted" in note
    assert "not" in note and "human-validated" in note


def test_metadata_uses_the_fixed_label_mapping(metadata):
    assert metadata["label_mapping"] == dict(P.LABEL2ID)


def test_metadata_records_the_baseline_hyperparameters(metadata):
    hp = metadata["hyperparameters"]
    assert hp["max_length"] == 256
    assert hp["learning_rate"] == 2e-5
    assert hp["epochs"] == 3
    assert hp["batch_size"] == 16
    assert hp["gradient_accumulation_steps"] == 1
    assert hp["optimizer"] == "AdamW"
    assert hp["weight_decay"] == 0.01
    assert hp["warmup_ratio"] == 0.1
    assert hp["gradient_clipping"] == 1.0
    assert hp["seed"] == 42
    assert hp["truncation"] is True
    assert hp["padding"] == "dynamic"
    assert hp["model_selection_metric"] == "validation_macro_f1"


def test_metadata_records_cpu_only_training(metadata):
    hardware = metadata["hardware"]
    assert hardware["device"] == "cpu"
    assert hardware["cuda_available"] is False
    assert hardware["gpu_name"] is None


def test_metadata_uses_only_text_as_a_feature(metadata):
    assert metadata["inputs"]["feature_columns"] == ["text", "label"]
    for excluded in ("confidence", "notes", "source", "stratum"):
        assert excluded in metadata["inputs"]["excluded_columns"]


def test_metadata_test_results_match_the_archived_predictions(metadata):
    predictions = pd.read_csv(ARCHIVE / "bert_test_predictions.csv")
    recomputed = P.compute_metrics(predictions["true_label"], predictions["predicted_label"])
    overall = metadata["test_results"]["overall"]
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"):
        assert recomputed[key] == pytest.approx(overall[key]), key
    assert metadata["test_results"]["n_examples"] == 298
    assert metadata["test_results"]["n_incorrect"] == 90


def test_metadata_implicit_slice_matches_the_archived_predictions(metadata):
    predictions = pd.read_csv(ARCHIVE / "bert_test_predictions.csv")
    recomputed = P.implicit_crisis_analysis(
        predictions["true_label"], predictions["predicted_label"])
    recorded = metadata["implicit_crisis_slice"]
    assert recorded["support"] == recomputed["support"] == 41
    assert recorded["prediction_breakdown"] == recomputed["prediction_breakdown"]
    assert recorded["recall"] == pytest.approx(recomputed["recall"])
    assert recorded["n_predicted_as_slice"] == 8


def test_metadata_best_epoch_matches_the_archived_history(metadata):
    history = json.loads(
        (ARCHIVE / "bert_training_history.json").read_text(encoding="utf-8"))
    best = max(history["history"], key=lambda r: r["val_macro_f1"])
    assert metadata["training"]["best_epoch"] == best["epoch"] == 3
    assert metadata["training"]["best_validation_macro_f1"] == pytest.approx(
        best["val_macro_f1"])
    assert len(metadata["training"]["history"]) == 3


def test_metadata_records_limitations_including_the_implicit_failure(metadata):
    joined = " ".join(metadata["limitations"]).lower()
    assert "cpu" in joined
    assert "implicit_crisis recall" in joined
