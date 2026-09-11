"""Reusable pieces of the BERT fine-tuning experiment.

The training script (``scripts/models/train_bert.py``) is a thin orchestrator on
top of this module so that dataset loading, the label mapping, tokenisation,
prediction decoding and metric computation can all be unit-tested without
touching a GPU or downloading weights.

Two rules the rest of the project depends on are enforced here:

* the label mapping is **fixed and explicit** (never inferred from the data), and
* only ``text`` reaches the model. ``confidence``, ``notes`` and every piece of
  annotation provenance are read for validation and then dropped.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_table
from src.utils.labels import LABELS

#: Fixed, explicit mapping. Do not generate this dynamically.
LABEL2ID: dict[str, int] = {
    "explicit_crisis": 0,
    "implicit_crisis": 1,
    "hard_negative": 2,
    "non_crisis": 3,
}
ID2LABEL: dict[int, str] = {i: label for label, i in LABEL2ID.items()}

#: Order used for every report, confusion matrix and probability column.
LABEL_ORDER: tuple[str, ...] = tuple(LABEL2ID)

#: Columns each split CSV must carry. Only ``text``/``label`` are used downstream.
REQUIRED_COLUMNS: tuple[str, ...] = ("id", "text", "label", "confidence", "notes")

#: Record counts of the frozen canonical split. Training refuses to start
#: against anything else, so a silently regenerated split cannot go unnoticed.
EXPECTED_SPLIT_SIZES: dict[str, int] = {"train": 1391, "validation": 298, "test": 298}

MODEL_INPUT_COLUMNS: tuple[str, ...] = ("text", "label")


class SplitValidationError(RuntimeError):
    """Raised when the frozen splits are missing, malformed or the wrong size."""


@dataclass
class SplitBundle:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in EXPECTED_SPLIT_SIZES}


def label_mapping() -> tuple[dict[str, int], dict[int, str]]:
    """Return the fixed ``(label2id, id2label)`` pair."""
    return dict(LABEL2ID), dict(ID2LABEL)


def validate_split_frame(
    df: pd.DataFrame,
    name: str,
    expected_size: int | None = None,
    require_all_labels: bool = True,
) -> None:
    """Fail loudly on anything that would quietly corrupt the experiment."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SplitValidationError(f"{name}: missing required columns {missing}")

    if expected_size is not None and len(df) != expected_size:
        raise SplitValidationError(
            f"{name}: expected {expected_size} records, found {len(df)}. "
            "The canonical split must not be regenerated."
        )

    labels = df["label"]
    if labels.isna().any() or (labels.astype(str).str.strip() == "").any():
        raise SplitValidationError(f"{name}: blank label(s) present")

    unknown = sorted(set(labels.astype(str)) - set(LABEL2ID))
    if unknown:
        raise SplitValidationError(f"{name}: unknown label(s) {unknown}")

    if require_all_labels:
        absent = sorted(set(LABEL2ID) - set(labels.astype(str)))
        if absent:
            raise SplitValidationError(f"{name}: label(s) {absent} not represented")

    if df["text"].isna().any() or (df["text"].astype(str).str.strip() == "").any():
        raise SplitValidationError(f"{name}: blank text present")

    if df["id"].duplicated().any():
        raise SplitValidationError(f"{name}: duplicate ids present")


def load_splits(splits_dir: Path | str, strict_sizes: bool = True) -> SplitBundle:
    """Read and validate the three frozen splits. The files are never written to."""
    splits_dir = Path(splits_dir)
    frames: dict[str, pd.DataFrame] = {}
    for name, expected in EXPECTED_SPLIT_SIZES.items():
        path = splits_dir / f"{name}.csv"
        if not path.exists():
            raise SplitValidationError(f"Missing required split file: {path}")
        df = read_table(path)
        validate_split_frame(df, name, expected if strict_sizes else None)
        frames[name] = df
    return SplitBundle(**frames)


def encode_labels(labels) -> np.ndarray:
    """Map label strings to their fixed ids."""
    return np.asarray([LABEL2ID[str(label)] for label in labels], dtype=np.int64)


def decode_predictions(logits) -> tuple[list[str], np.ndarray]:
    """Turn raw logits into ``(predicted label strings, softmax probabilities)``."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != len(LABEL_ORDER):
        raise ValueError(f"Expected logits of shape (n, {len(LABEL_ORDER)}), got {logits.shape}")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    preds = [ID2LABEL[int(i)] for i in probs.argmax(axis=1)]
    return preds, probs


def build_tokenized_dataset(texts, labels, tokenizer, max_length: int):
    """A torch ``Dataset`` of un-padded encodings; padding happens in the collator."""
    import torch

    texts = [str(t) for t in texts]
    encoded = tokenizer(texts, truncation=True, max_length=max_length)
    label_ids = None if labels is None else encode_labels(labels)

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(texts)

        def __getitem__(self, idx: int) -> dict:
            item = {k: v[idx] for k, v in encoded.items()}
            if label_ids is not None:
                item["labels"] = int(label_ids[idx])
            return item

    return _Dataset()


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy plus macro/weighted precision, recall and F1 over the fixed label order."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    labels = list(LABEL_ORDER)
    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "n": int(len(list(y_true))),
    }


def classification_report_dict(y_true, y_pred) -> dict:
    from sklearn.metrics import classification_report

    return classification_report(
        y_true,
        y_pred,
        labels=list(LABEL_ORDER),
        target_names=list(LABEL_ORDER),
        output_dict=True,
        zero_division=0,
    )


def confusion_frame(y_true, y_pred) -> pd.DataFrame:
    """Confusion matrix with gold labels on the rows, in the fixed label order."""
    from sklearn.metrics import confusion_matrix

    labels = list(LABEL_ORDER)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "true_label"
    return frame


def implicit_crisis_analysis(y_true, y_pred, slice_label: str = "implicit_crisis") -> dict:
    """The project's primary slice: how implicit-crisis records are actually classified."""
    from sklearn.metrics import precision_recall_fscore_support

    y_true = pd.Series([str(v) for v in y_true])
    y_pred = pd.Series([str(v) for v in y_pred])
    mask = y_true == slice_label
    support = int(mask.sum())

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[slice_label], average="macro", zero_division=0
    )
    breakdown = {label: int((y_pred[mask] == label).sum()) for label in LABEL_ORDER}
    return {
        "slice": slice_label,
        "support": support,
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "n_predicted_as_slice": int((y_pred == slice_label).sum()),
        "prediction_breakdown": breakdown,
        "prediction_breakdown_rate": {
            label: (count / support if support else 0.0) for label, count in breakdown.items()
        },
    }


def prediction_frame(df: pd.DataFrame, y_pred, probs) -> pd.DataFrame:
    """Per-record predictions. Only id/text/labels/probabilities — no annotation metadata."""
    probs = np.asarray(probs, dtype=float)
    out = pd.DataFrame(
        {
            "id": df["id"].to_numpy(),
            "text": df["text"].to_numpy(),
            "true_label": df["label"].to_numpy(),
            "predicted_label": list(y_pred),
            "confidence": probs.max(axis=1),
        }
    )
    for i, label in enumerate(LABEL_ORDER):
        out[f"prob_{label}"] = probs[:, i]
    return out


def error_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    wrong = predictions[predictions["true_label"] != predictions["predicted_label"]]
    return wrong[["id", "text", "true_label", "predicted_label", "confidence"]].copy()


@dataclass
class RunConfig:
    """Everything needed to reproduce a run; serialised to ``bert_config.json``."""

    model_name: str = "bert-base-uncased"
    tokenizer_name: str = "bert-base-uncased"
    max_length: int = 256
    learning_rate: float = 2e-5
    epochs: int = 3
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    optimizer: str = "AdamW"
    seed: int = 42
    device: str = "cpu"
    num_labels: int = len(LABELS)
    label_mapping: dict = field(default_factory=lambda: dict(LABEL2ID))
    padding: str = "dynamic"
    truncation: bool = True
    model_selection_metric: str = "validation_macro_f1"

    def to_dict(self) -> dict:
        return asdict(self)
