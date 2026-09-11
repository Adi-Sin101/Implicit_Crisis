"""Tests for the BERT experiment pipeline (src/models/bert/pipeline.py).

These cover the parts that can go silently wrong — the label mapping, split
validation, prediction decoding and metric computation — without downloading
weights or running a training step.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.models.bert import pipeline as P
from src.utils.config import load_config, resolve
from src.utils.labels import LABELS

SPLITS_DIR = resolve(load_config("paths.yaml")["gold"]["splits_dir"])
RESULTS_DIR = resolve("data/results/bert")


def _frame(labels, n_extra_cols: bool = True) -> pd.DataFrame:
    data = {
        "id": [f"rec-{i}" for i in range(len(labels))],
        "text": [f"example text {i}" for i in range(len(labels))],
        "label": list(labels),
    }
    if n_extra_cols:
        data["confidence"] = [3] * len(labels)
        data["notes"] = [""] * len(labels)
    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# Label mapping
# --------------------------------------------------------------------------- #

def test_label_mapping_is_fixed_and_explicit():
    label2id, id2label = P.label_mapping()
    assert label2id == {
        "explicit_crisis": 0,
        "implicit_crisis": 1,
        "hard_negative": 2,
        "non_crisis": 3,
    }
    assert id2label == {0: "explicit_crisis", 1: "implicit_crisis",
                        2: "hard_negative", 3: "non_crisis"}
    assert P.LABEL_ORDER == tuple(label2id)
    assert set(P.LABEL_ORDER) == set(LABELS)


def test_label_mapping_is_a_copy():
    label2id, _ = P.label_mapping()
    label2id["explicit_crisis"] = 99
    assert P.LABEL2ID["explicit_crisis"] == 0


def test_encode_labels_round_trips():
    labels = ["non_crisis", "implicit_crisis", "explicit_crisis", "hard_negative"]
    ids = P.encode_labels(labels)
    assert list(ids) == [3, 1, 0, 2]
    assert [P.ID2LABEL[i] for i in ids] == labels


# --------------------------------------------------------------------------- #
# Dataset loading and validation
# --------------------------------------------------------------------------- #

def test_load_splits_reads_the_frozen_canonical_split():
    splits = P.load_splits(SPLITS_DIR)
    assert splits.counts() == {"train": 1391, "validation": 298, "test": 298}
    for name in ("train", "validation", "test"):
        frame = getattr(splits, name)
        assert list(P.REQUIRED_COLUMNS) == [c for c in P.REQUIRED_COLUMNS
                                            if c in frame.columns]
        assert set(frame["label"]) == set(P.LABEL_ORDER)


def test_split_ids_do_not_overlap():
    splits = P.load_splits(SPLITS_DIR)
    ids = [set(getattr(splits, n)["id"]) for n in ("train", "validation", "test")]
    assert ids[0] & ids[1] == set()
    assert ids[0] & ids[2] == set()
    assert ids[1] & ids[2] == set()


def test_model_only_consumes_text_and_label():
    assert P.MODEL_INPUT_COLUMNS == ("text", "label")
    assert "confidence" not in P.MODEL_INPUT_COLUMNS
    assert "notes" not in P.MODEL_INPUT_COLUMNS


def test_missing_column_is_rejected():
    frame = _frame(P.LABEL_ORDER).drop(columns=["notes"])
    with pytest.raises(P.SplitValidationError, match="notes"):
        P.validate_split_frame(frame, "train")


def test_blank_label_is_rejected():
    frame = _frame(P.LABEL_ORDER)
    frame.loc[0, "label"] = " "
    with pytest.raises(P.SplitValidationError, match="blank label"):
        P.validate_split_frame(frame, "train")


def test_unknown_label_is_rejected():
    frame = _frame(["explicit_crisis", "maybe_crisis", "hard_negative", "non_crisis"])
    with pytest.raises(P.SplitValidationError, match="unknown label"):
        P.validate_split_frame(frame, "train")


def test_missing_label_class_is_rejected():
    frame = _frame(["explicit_crisis", "non_crisis"])
    with pytest.raises(P.SplitValidationError, match="not represented"):
        P.validate_split_frame(frame, "train")


def test_wrong_split_size_is_rejected():
    frame = _frame(P.LABEL_ORDER)
    with pytest.raises(P.SplitValidationError, match="expected 1391"):
        P.validate_split_frame(frame, "train", expected_size=1391)


def test_missing_split_file_is_rejected():
    with pytest.raises(P.SplitValidationError, match="Missing required split file"):
        P.load_splits(SPLITS_DIR / "no_such_directory")


# --------------------------------------------------------------------------- #
# Tokenizer configuration and dataset construction
# --------------------------------------------------------------------------- #

class _FakeTokenizer:
    """Records the arguments the pipeline passes; emits token ids of one per word."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, texts, truncation=None, max_length=None, **kwargs):
        self.calls.append({"truncation": truncation, "max_length": max_length,
                           "kwargs": kwargs})
        ids = [[101] + [1000 + i for i in range(len(t.split()))][: (max_length or 10) - 2] + [102]
               for t in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]}


def test_tokenizer_is_configured_for_truncation_and_dynamic_padding():
    torch = pytest.importorskip("torch")
    tokenizer = _FakeTokenizer()
    frame = _frame(P.LABEL_ORDER)
    dataset = P.build_tokenized_dataset(frame["text"], frame["label"], tokenizer, 256)

    call = tokenizer.calls[0]
    assert call["truncation"] is True
    assert call["max_length"] == 256
    # Padding is deliberately NOT requested here: the collator pads per batch.
    assert "padding" not in call["kwargs"]

    assert len(dataset) == len(frame)
    item = dataset[0]
    assert set(item) == {"input_ids", "attention_mask", "labels"}
    assert item["labels"] == P.LABEL2ID[frame.loc[0, "label"]]
    assert torch is not None


def test_dataset_without_labels_omits_the_label_key():
    pytest.importorskip("torch")
    frame = _frame(P.LABEL_ORDER)
    dataset = P.build_tokenized_dataset(frame["text"], None, _FakeTokenizer(), 256)
    assert "labels" not in dataset[0]


# --------------------------------------------------------------------------- #
# Prediction decoding
# --------------------------------------------------------------------------- #

def test_decode_predictions_picks_the_argmax_class():
    logits = np.array([
        [5.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0],
        [0.0, 0.0, 0.0, 5.0],
    ])
    preds, probs = P.decode_predictions(logits)
    assert preds == list(P.LABEL_ORDER)
    assert probs.shape == (4, 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert probs.argmax(axis=1).tolist() == [0, 1, 2, 3]


def test_decode_predictions_is_numerically_stable():
    preds, probs = P.decode_predictions(np.array([[1000.0, 999.0, -1000.0, 0.0]]))
    assert preds == ["explicit_crisis"]
    assert np.isfinite(probs).all()


def test_decode_predictions_rejects_wrong_shape():
    with pytest.raises(ValueError):
        P.decode_predictions(np.zeros((3, 5)))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_metrics_on_perfect_predictions():
    y = list(P.LABEL_ORDER) * 3
    m = P.compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["weighted_f1"] == 1.0
    assert m["n"] == 12


def test_metrics_match_a_hand_computed_case():
    y_true = ["explicit_crisis", "explicit_crisis", "implicit_crisis", "non_crisis"]
    y_pred = ["explicit_crisis", "implicit_crisis", "implicit_crisis", "non_crisis"]
    m = P.compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(0.75)
    # explicit: P=1.0 R=0.5 F1=2/3 ; implicit: P=0.5 R=1.0 F1=2/3 ;
    # hard_negative: 0 ; non_crisis: 1.0
    assert m["macro_f1"] == pytest.approx((2 / 3 + 2 / 3 + 0.0 + 1.0) / 4)
    assert m["macro_precision"] == pytest.approx((1.0 + 0.5 + 0.0 + 1.0) / 4)


def test_confusion_frame_orientation():
    y_true = ["explicit_crisis", "implicit_crisis"]
    y_pred = ["non_crisis", "implicit_crisis"]
    matrix = P.confusion_frame(y_true, y_pred)
    assert list(matrix.index) == list(P.LABEL_ORDER)
    assert list(matrix.columns) == list(P.LABEL_ORDER)
    assert matrix.loc["explicit_crisis", "non_crisis"] == 1
    assert matrix.loc["implicit_crisis", "implicit_crisis"] == 1
    assert matrix.to_numpy().sum() == 2


def test_classification_report_covers_every_class():
    y_true = ["explicit_crisis", "implicit_crisis"]
    y_pred = ["explicit_crisis", "implicit_crisis"]
    report = P.classification_report_dict(y_true, y_pred)
    for label in P.LABEL_ORDER:
        assert label in report
        assert {"precision", "recall", "f1-score", "support"} <= set(report[label])


# --------------------------------------------------------------------------- #
# Implicit-crisis slice analysis
# --------------------------------------------------------------------------- #

def test_implicit_crisis_breakdown_accounts_for_every_gold_record():
    y_true = ["implicit_crisis"] * 4 + ["non_crisis"]
    y_pred = ["implicit_crisis", "explicit_crisis", "hard_negative", "non_crisis",
              "implicit_crisis"]
    analysis = P.implicit_crisis_analysis(y_true, y_pred)

    assert analysis["support"] == 4
    assert sum(analysis["prediction_breakdown"].values()) == 4
    assert analysis["prediction_breakdown"] == {
        "explicit_crisis": 1, "implicit_crisis": 1,
        "hard_negative": 1, "non_crisis": 1,
    }
    assert analysis["recall"] == pytest.approx(0.25)
    assert analysis["precision"] == pytest.approx(0.5)  # 1 of 2 predicted implicit
    assert set(analysis["prediction_breakdown_rate"]) == set(P.LABEL_ORDER)


def test_implicit_crisis_analysis_handles_an_empty_slice():
    analysis = P.implicit_crisis_analysis(["non_crisis"], ["non_crisis"])
    assert analysis["support"] == 0
    assert analysis["recall"] == 0.0
    assert sum(analysis["prediction_breakdown"].values()) == 0


# --------------------------------------------------------------------------- #
# Output frames
# --------------------------------------------------------------------------- #

def test_prediction_frame_carries_no_annotation_metadata():
    frame = _frame(P.LABEL_ORDER)
    probs = np.eye(4) * 0.7 + 0.1
    preds = list(P.LABEL_ORDER)
    out = P.prediction_frame(frame, preds, probs)

    assert list(out.columns) == [
        "id", "text", "true_label", "predicted_label", "confidence",
        "prob_explicit_crisis", "prob_implicit_crisis",
        "prob_hard_negative", "prob_non_crisis",
    ]
    assert "confidence" in out.columns  # the model's own confidence, not the annotator's
    assert "notes" not in out.columns
    assert out["confidence"].max() <= 1.0
    assert out.loc[0, "confidence"] == pytest.approx(0.8)


def test_error_frame_keeps_only_incorrect_rows():
    frame = _frame(P.LABEL_ORDER)
    preds = ["explicit_crisis", "non_crisis", "hard_negative", "explicit_crisis"]
    probs = np.full((4, 4), 0.25)
    errors = P.error_frame(P.prediction_frame(frame, preds, probs))

    assert len(errors) == 2
    assert list(errors.columns) == ["id", "text", "true_label",
                                    "predicted_label", "confidence"]
    assert (errors["true_label"] != errors["predicted_label"]).all()


def test_run_config_records_everything_needed_to_reproduce():
    cfg = P.RunConfig(device="cpu").to_dict()
    for key in ("model_name", "tokenizer_name", "max_length", "learning_rate",
                "epochs", "batch_size", "gradient_accumulation_steps", "optimizer",
                "seed", "device", "num_labels", "label_mapping",
                "model_selection_metric"):
        assert key in cfg
    assert cfg["num_labels"] == 4
    assert cfg["label_mapping"] == dict(P.LABEL2ID)
    assert cfg["optimizer"] == "AdamW"


# --------------------------------------------------------------------------- #
# Output structure produced by a completed run
# --------------------------------------------------------------------------- #

EXPECTED_OUTPUTS = (
    "bert_config.json",
    "bert_training_history.json",
    "bert_test_metrics.json",
    "bert_classification_report.json",
    "bert_confusion_matrix.csv",
    "bert_test_predictions.csv",
    "bert_errors.csv",
)


@pytest.mark.skipif(not (RESULTS_DIR / "bert_test_metrics.json").exists(),
                    reason="BERT experiment has not been run in this checkout")
class TestCompletedRunOutputs:
    def test_all_expected_files_exist(self):
        for name in EXPECTED_OUTPUTS:
            assert (RESULTS_DIR / name).exists(), name

    def test_predictions_cover_the_whole_test_split(self):
        preds = pd.read_csv(RESULTS_DIR / "bert_test_predictions.csv")
        test = pd.read_csv(SPLITS_DIR / "test.csv")
        assert len(preds) == len(test) == 298
        assert set(preds["id"]) == set(test["id"])
        assert set(preds["predicted_label"]) <= set(P.LABEL_ORDER)
        assert "notes" not in preds.columns

    def test_errors_are_the_incorrect_predictions(self):
        preds = pd.read_csv(RESULTS_DIR / "bert_test_predictions.csv")
        errors = pd.read_csv(RESULTS_DIR / "bert_errors.csv")
        assert len(errors) == int((preds["true_label"] != preds["predicted_label"]).sum())

    def test_metrics_agree_with_the_saved_predictions(self):
        preds = pd.read_csv(RESULTS_DIR / "bert_test_predictions.csv")
        saved = json.loads((RESULTS_DIR / "bert_test_metrics.json").read_text(encoding="utf-8"))
        recomputed = P.compute_metrics(preds["true_label"], preds["predicted_label"])
        assert recomputed["macro_f1"] == pytest.approx(saved["overall"]["macro_f1"])
        assert recomputed["accuracy"] == pytest.approx(saved["overall"]["accuracy"])
        assert saved["label_mapping"] == dict(P.LABEL2ID)

    def test_best_checkpoint_was_selected_on_validation_macro_f1(self):
        history = json.loads(
            (RESULTS_DIR / "bert_training_history.json").read_text(encoding="utf-8"))
        best = max(history["history"], key=lambda r: r["val_macro_f1"])
        assert history["best_epoch"] == best["epoch"]
        assert history["best_val_macro_f1"] == pytest.approx(best["val_macro_f1"])
        assert history["selection_metric"] == "validation_macro_f1"
