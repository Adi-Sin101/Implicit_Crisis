"""Fine-tuned BERT classifier for the four-way gold task.

Transformers and torch are imported lazily so that the data-preparation and
annotation half of the project runs without a deep-learning stack installed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.labels import LABELS


@dataclass
class BertConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    epochs: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 2
    seed: int = 42

    @classmethod
    def from_dict(cls, config: dict) -> "BertConfig":
        cfg = dict(config.get("bert", {}))
        cfg["seed"] = config.get("seed", 42)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in cfg.items() if k in known})


def label_maps(labels=LABELS):
    label2id = {label: i for i, label in enumerate(labels)}
    return label2id, {i: label for label, i in label2id.items()}


def build_dataset(texts, labels, tokenizer, max_length: int):
    import torch

    label2id, _ = label_maps()
    encodings = tokenizer(
        list(texts), truncation=True, padding="max_length", max_length=max_length
    )

    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(texts)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in encodings.items()}
            if labels is not None:
                item["labels"] = torch.tensor(label2id[labels[idx]])
            return item

    return _Dataset()


def load_model(cfg: BertConfig):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    label2id, id2label = label_maps()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    return tokenizer, model


def predict(model, tokenizer, texts, cfg: BertConfig) -> list[str]:
    import torch

    _, id2label = label_maps()
    model.eval()
    preds: list[str] = []
    with torch.no_grad():
        for start in range(0, len(texts), cfg.batch_size):
            batch = list(texts[start : start + cfg.batch_size])
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            ).to(model.device)
            logits = model(**enc).logits.cpu().numpy()
            preds.extend(id2label[int(i)] for i in np.argmax(logits, axis=1))
    return preds
