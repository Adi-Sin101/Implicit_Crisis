"""Standalone 7-class BERT fine-tuning experiment on the human/LLM-labeled
r/SuicideWatch dataset.

    %run train_human_llm_bert_colab.py
    # or
    !python train_human_llm_bert_colab.py
    # or
    from train_human_llm_bert_colab import run_experiment; run_experiment()

This file is deliberately SELF-CONTAINED. It imports nothing from this
repository (no src.models.bert.*, no src.utils.labels, no scripts.models.*),
so it can be copied into Google Colab on its own. The project's existing
four-class BERT experiment is entirely untouched by this script.

Task
----
    input  : `content`   (the post text, and nothing else)
    target : `severity`  (the dataset's own original labels 0-6)

The five LLM prediction columns (gpt/claude/gemini/llama/mistral_label) and the
provenance columns (url/author/created) are auxiliary: they are never used as
model features and never printed.

Data isolation
--------------
Only the three files under `data/gold/splits/human_llm/` are ever read. The
script never falls back to, searches for, or substitutes any other dataset.
The test split is touched exactly once, after the best validation checkpoint
has been selected.
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# =====================================================================
# CONFIGURATION -- the only section you should need to edit
# =====================================================================

# Leave as None to auto-detect from the repository layout. In Colab set e.g.
#   DATA_DIR   = "/content/Implicit_Crisis/data/gold/splits/human_llm"
#   OUTPUT_DIR = "/content/Implicit_Crisis/data/results/bert/human_llm"
# Setting DATA_DIR alone is enough; OUTPUT_DIR is then derived from it.
DATA_DIR: str | None = None
OUTPUT_DIR: str | None = None

# Model / training hyperparameters (baseline).
MODEL_NAME = "bert-base-uncased"
TOKENIZER_NAME = "bert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

# Column contract of the split files.
TEXT_COLUMN = "content"
LABEL_COLUMN = "severity"

# The frozen split sizes. Training refuses to start against anything else, so a
# silently regenerated or substituted split cannot go unnoticed.
EXPECTED_SPLIT_SIZES = {"train": 819, "validation": 175, "test": 176}
EXPECTED_NUM_LABELS = 7

# Class weighting is intentionally OFF: this is a straightforward fine-tuning
# baseline on the natural (imbalanced) distribution. Do not flip this silently;
# doing so makes the run no longer comparable to the reported baseline.
USE_CLASS_WEIGHTS = False

# =====================================================================
# Dependencies
# =====================================================================

REQUIRED_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
}


def check_dependencies() -> None:
    """Fail early with an actionable pip command rather than mid-training."""
    import importlib

    missing = []
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(pip_name)
    if missing:
        raise SystemExit(
            "Missing required packages: "
            + ", ".join(missing)
            + "\n\nInstall them with:\n    pip install "
            + " ".join(missing)
            + "\n\n(In Colab: !pip install -q " + " ".join(missing) + ")"
        )


check_dependencies()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

# seaborn only makes the PNG prettier; matplotlib alone is a fine fallback.
try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:  # pragma: no cover
    HAS_SEABORN = False


# =====================================================================
# Small utilities
# =====================================================================

class Tee:
    """Mirror stdout into bert_training_log.txt without losing Colab output."""

    def __init__(self, stream, path: Path):
        self.stream = stream
        self.file = open(path, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        return len(data)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def banner(title: str, char: str = "=", width: int = 50) -> None:
    print("\n" + char * width)
    print(title)
    print(char * width)


def set_seed(seed: int) -> None:
    """Seed every RNG we touch, as deterministically as is practical on GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device():
    """GPU when available; otherwise CPU with a loud warning."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    print(
        "\n*** WARNING: CUDA is not available -- falling back to CPU.  ***"
        "\n*** Fine-tuning BERT on CPU is very slow. In Colab choose   ***"
        "\n*** Runtime > Change runtime type > GPU.                    ***\n"
    )
    return torch.device("cpu")


def print_environment(device) -> None:
    banner("HUMAN/LLM SUICIDEWATCH BERT EXPERIMENT")
    print()
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU)"
    print(f"Device:               {device}")
    print(f"GPU:                  {gpu}")
    print(f"CUDA available:       {torch.cuda.is_available()}")
    print(f"PyTorch version:      {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print()
    print("=" * 50)


# =====================================================================
# Paths
# =====================================================================

def resolve_paths() -> tuple[Path, Path]:
    """Honour DATA_DIR/OUTPUT_DIR if set, else derive them from the repo layout."""
    # This file lives at <repo>/experiments/bert_human_llm/<this file>.
    repo_root = Path(__file__).resolve().parents[2]

    data_dir = Path(DATA_DIR) if DATA_DIR else repo_root / "data" / "gold" / "splits" / "human_llm"

    if OUTPUT_DIR:
        out_dir = Path(OUTPUT_DIR)
    elif DATA_DIR:
        # .../data/gold/splits/human_llm -> .../data/results/bert/human_llm,
        # so only one variable has to be set in Colab.
        out_dir = Path(DATA_DIR).resolve().parents[2] / "results" / "bert" / "human_llm"
    else:
        out_dir = repo_root / "data" / "results" / "bert" / "human_llm"

    return data_dir.resolve(), out_dir.resolve()


# =====================================================================
# 1. Data loading
# =====================================================================

def load_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load exactly the three human_llm split files. No fallbacks, ever."""
    paths = {name: data_dir / f"{name}.csv" for name in EXPECTED_SPLIT_SIZES}

    banner("DATASET ISOLATION CHECK", "-")
    print()
    print(f"TRAIN:\n  {paths['train']}")
    print(f"VALIDATION:\n  {paths['validation']}")
    print(f"TEST:\n  {paths['test']}")
    print()

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit(
            "Required split file(s) missing:\n  "
            + "\n  ".join(missing)
            + "\n\nThis script will NOT substitute another dataset. Set DATA_DIR at "
            "the top of this file to the directory holding train.csv / "
            "validation.csv / test.csv of the human_llm split."
        )

    frames = {name: pd.read_csv(p, encoding="utf-8") for name, p in paths.items()}
    print("DATASET ISOLATION: Using ONLY the human_llm split.")
    print()
    return frames


# =====================================================================
# 2. Validation
# =====================================================================

def validate_data(frames: dict[str, pd.DataFrame]) -> None:
    """Fail loudly on anything that would quietly corrupt the experiment.

    Nothing here repairs the data: no rows are dropped, no labels are changed,
    no split is regenerated. A failure means the inputs are wrong, not that the
    script should adapt to them.
    """
    banner("DATASET VALIDATION", "-")
    print()
    errors: list[str] = []

    for name, df in frames.items():
        missing_cols = [c for c in (TEXT_COLUMN, LABEL_COLUMN) if c not in df.columns]
        if missing_cols:
            errors.append(f"{name}: missing required column(s) {missing_cols}")
            print(f"  {name:<11} {len(df):>4} records   FAIL (columns)")
            continue

        problems: list[str] = []

        expected = EXPECTED_SPLIT_SIZES[name]
        if len(df) != expected:
            problems.append(
                f"expected {expected} records, found {len(df)}; "
                "the frozen human_llm split must not be regenerated"
            )

        text = df[TEXT_COLUMN]
        if text.isna().any() or (text.astype(str).str.strip() == "").any():
            problems.append(f"blank/missing '{TEXT_COLUMN}' value(s)")

        labels = df[LABEL_COLUMN]
        if labels.isna().any():
            problems.append(f"missing '{LABEL_COLUMN}' value(s)")
        else:
            try:
                ints = labels.astype(int)
            except (ValueError, TypeError):
                problems.append(f"'{LABEL_COLUMN}' is not integer-valued")
            else:
                bad = sorted(set(ints.unique()) - set(range(EXPECTED_NUM_LABELS)))
                if bad:
                    problems.append(f"label(s) outside 0-{EXPECTED_NUM_LABELS - 1}: {bad}")

        errors.extend(f"{name}: {p}" for p in problems)
        status = "OK" if not problems else "FAIL"
        print(f"  {name:<11} {len(df):>4} records   {status}")

    if not errors:
        train_labels = set(frames["train"][LABEL_COLUMN].astype(int).unique())
        if len(train_labels) != EXPECTED_NUM_LABELS:
            errors.append(
                f"train: expected exactly {EXPECTED_NUM_LABELS} distinct labels, "
                f"found {len(train_labels)}: {sorted(train_labels)}"
            )
        for name in ("validation", "test"):
            unseen = sorted(set(frames[name][LABEL_COLUMN].astype(int).unique()) - train_labels)
            if unseen:
                errors.append(f"{name}: label(s) {unseen} never seen in train")

    if errors:
        raise SystemExit("\nDATASET VALIDATION FAILED:\n  - " + "\n  - ".join(errors))

    print("\n  All checks passed: columns, sizes, labels, no missing values.")
    print()


def report_class_distribution(frames: dict[str, pd.DataFrame], labels: list[int]) -> None:
    """The dataset is imbalanced; make that explicit rather than implicit."""
    banner("CLASS DISTRIBUTION (imbalanced -- reported, not corrected)", "-")
    print()
    header = f"{'Label':>6} | {'Train':>7} | {'Val':>7} | {'Test':>7} | {'Total':>7} | {'Total %':>8}"
    print(header)
    print("-" * len(header))
    total_n = sum(len(df) for df in frames.values())
    for lab in labels:
        counts = {n: int((df[LABEL_COLUMN].astype(int) == lab).sum()) for n, df in frames.items()}
        tot = sum(counts.values())
        print(
            f"{lab:>6} | {counts['train']:>7} | {counts['validation']:>7} | "
            f"{counts['test']:>7} | {tot:>7} | {tot / total_n * 100:>7.2f}%"
        )
    print()
    print(f"  Class weighting enabled: {USE_CLASS_WEIGHTS}  (baseline = plain fine-tuning)")
    print()


# =====================================================================
# 3. Label mapping (derived from the training data, never hard-coded)
# =====================================================================

def create_label_mapping(train_df: pd.DataFrame):
    """Build label2id/id2label from the TRAINING split only."""
    labels = sorted(int(v) for v in train_df[LABEL_COLUMN].astype(int).unique())
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    banner("LABEL MAPPING (derived from train.csv)", "-")
    print()
    print(f"{'Original label':>15} | {'Model ID':>9}")
    print("-" * 27)
    for lab in labels:
        print(f"{lab:>15} | {label2id[lab]:>9}")
    print()
    print(f"  num_labels = {len(labels)}")
    print()
    return label2id, id2label, labels


# =====================================================================
# 4. Token-length analysis
# =====================================================================

def analyze_token_lengths(frames: dict[str, pd.DataFrame], tokenizer) -> dict:
    """Real tokenized lengths -- character counts are not a proxy for these."""
    banner("TOKEN LENGTH ANALYSIS", "-")
    print()
    stats: dict[str, dict] = {}
    for name, df in frames.items():
        texts = df[TEXT_COLUMN].astype(str).tolist()
        # add_special_tokens=True so [CLS]/[SEP] count toward the budget.
        lengths = np.array(
            [len(tokenizer.encode(t, add_special_tokens=True, truncation=False)) for t in texts]
        )
        over = int((lengths > MAX_LENGTH).sum())
        stats[name] = {
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            f"n_over_{MAX_LENGTH}": over,
            f"pct_over_{MAX_LENGTH}": float(over / len(lengths) * 100),
        }
        s = stats[name]
        print(f"  {name}")
        print(f"    max / mean / median : {s['max']} / {s['mean']:.1f} / {s['median']:.1f}")
        print(f"    p95 / p99           : {s['p95']:.1f} / {s['p99']:.1f}")
        print(f"    > {MAX_LENGTH} tokens        : {over} ({s[f'pct_over_{MAX_LENGTH}']:.2f}%)")

    total_over = sum(s[f"n_over_{MAX_LENGTH}"] for s in stats.values())
    print()
    if total_over:
        print(
            f"  TRUNCATION: {total_over} example(s) exceed max_length={MAX_LENGTH} "
            "and will be truncated."
        )
    else:
        print(f"  No truncation: every example fits within max_length={MAX_LENGTH}.")
    print()
    return stats


# =====================================================================
# 5. Dataset / dataloaders
# =====================================================================

class SeverityDataset(Dataset):
    """Un-padded encodings of `content` only; padding happens in the collator."""

    def __init__(self, texts, labels, tokenizer, label2id, max_length):
        self.encodings = tokenizer(
            [str(t) for t in texts], truncation=True, max_length=max_length
        )
        self.label_ids = None if labels is None else [label2id[int(v)] for v in labels]

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        if self.label_ids is not None:
            item["labels"] = self.label_ids[idx]
        return item


def create_datasets(frames, tokenizer, label2id):
    return {
        name: SeverityDataset(
            df[TEXT_COLUMN].tolist(), df[LABEL_COLUMN].tolist(), tokenizer, label2id, MAX_LENGTH
        )
        for name, df in frames.items()
    }


def create_dataloaders(datasets, tokenizer):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)  # dynamic padding
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collator,
            generator=generator,
        ),
        "validation": DataLoader(
            datasets["validation"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
        ),
        "test": DataLoader(
            datasets["test"], batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator
        ),
    }


# =====================================================================
# 6. Model
# =====================================================================

def build_model(label2id: dict, id2label: dict, device):
    """bert-base-uncased with a 7-way head and the original labels attached."""
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        id2label={i: str(lab) for i, lab in id2label.items()},
        label2id={str(lab): i for lab, i in label2id.items()},
    )
    return model.to(device)


# =====================================================================
# 7. Train / evaluate
# =====================================================================

def train_one_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    total, n_batches = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total += loss.item()
        n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    """Return (mean loss, true ids, predicted ids, softmax probabilities)."""
    model.eval()
    total, n_batches = 0.0, 0
    all_logits, all_true = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        if out.loss is not None:
            total += out.loss.item()
            n_batches += 1
        all_logits.append(out.logits.detach().float().cpu().numpy())
        all_true.append(batch["labels"].detach().cpu().numpy())

    logits = np.concatenate(all_logits)
    y_true = np.concatenate(all_true)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    return total / max(n_batches, 1), y_true, y_pred, probs


def compute_metrics(y_true, y_pred, id_order) -> dict:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    macro = precision_recall_fscore_support(
        y_true, y_pred, labels=id_order, average="macro", zero_division=0
    )
    weighted = precision_recall_fscore_support(
        y_true, y_pred, labels=id_order, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "n": int(len(y_true)),
    }


def save_checkpoint(model, tokenizer, best_dir: Path) -> None:
    best_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)


# =====================================================================
# 8. Reporting artifacts
# =====================================================================

def generate_predictions(test_df, y_true_ids, y_pred_ids, probs, id2label, labels):
    """One row per test example: text, gold, prediction, confidence, per-class probs."""
    out = pd.DataFrame(
        {
            TEXT_COLUMN: test_df[TEXT_COLUMN].to_numpy(),
            "true_label": [id2label[int(i)] for i in y_true_ids],
            "predicted_label": [id2label[int(i)] for i in y_pred_ids],
            "confidence": probs.max(axis=1),
        }
    )
    for i, lab in enumerate(labels):
        out[f"prob_{lab}"] = probs[:, i]
    return out


def build_confusion_frame(y_true, y_pred, id_order, labels) -> pd.DataFrame:
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=id_order)
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    frame.index.name = "true_label"
    return frame


def save_confusion_png(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if HAS_SEABORN:
        sns.heatmap(frame, annot=True, fmt="d", cmap="Blues", cbar=True, ax=ax)
    else:
        im = ax.imshow(frame.values, cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set_xticks(range(len(frame.columns)))
        ax.set_xticklabels(frame.columns)
        ax.set_yticks(range(len(frame.index)))
        ax.set_yticklabels(frame.index)
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                ax.text(j, i, int(frame.iat[i, j]), ha="center", va="center")
    ax.set_xlabel("Predicted severity")
    ax.set_ylabel("True severity")
    ax.set_title("BERT 7-class confusion matrix (human/LLM SuicideWatch)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_results(out_dir: Path, artifacts: dict) -> None:
    for name, obj in artifacts.items():
        path = out_dir / name
        if isinstance(obj, pd.DataFrame):
            # Only the confusion matrix carries a meaningful index.
            obj.to_csv(path, index=name.endswith("confusion_matrix.csv"), encoding="utf-8")
        else:
            path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


# =====================================================================
# 9. Orchestration
# =====================================================================

def run_experiment() -> dict:
    """Run the whole experiment end to end and return the test metrics."""
    data_dir, out_dir = resolve_paths()
    out_dir.mkdir(parents=True, exist_ok=True)

    tee = Tee(sys.stdout, out_dir / "bert_training_log.txt")
    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        set_seed(SEED)
        device = resolve_device()
        print_environment(device)

        # --- data -----------------------------------------------------
        frames = load_data(data_dir)
        validate_data(frames)
        label2id, id2label, labels = create_label_mapping(frames["train"])
        id_order = [label2id[lab] for lab in labels]
        report_class_distribution(frames, labels)

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        token_stats = analyze_token_lengths(frames, tokenizer)

        datasets = create_datasets(frames, tokenizer, label2id)
        loaders = create_dataloaders(datasets, tokenizer)

        # --- model ----------------------------------------------------
        model = build_model(label2id, id2label, device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        total_steps = len(loaders["train"]) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * WARMUP_RATIO),
            num_training_steps=total_steps,
        )

        # --- training loop (validation only; test is never touched) ----
        banner("TRAINING")
        print()
        best_dir = out_dir / "best_model"
        history: list[dict] = []
        best_f1, best_epoch = -1.0, -1

        for epoch in range(1, EPOCHS + 1):
            train_loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, device)
            val_loss, v_true, v_pred, _ = evaluate(model, loaders["validation"], device)
            m = compute_metrics(v_true, v_pred, id_order)

            print(f"Epoch {epoch}/{EPOCHS}")
            print(f"  Training loss:           {train_loss:.4f}")
            print(f"  Validation loss:         {val_loss:.4f}")
            print(f"  Validation accuracy:     {m['accuracy']:.4f}")
            print(f"  Validation macro F1:     {m['macro_f1']:.4f}")
            print(f"  Validation weighted F1:  {m['weighted_f1']:.4f}")

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_accuracy": m["accuracy"],
                    "val_macro_precision": m["macro_precision"],
                    "val_macro_recall": m["macro_recall"],
                    "val_macro_f1": m["macro_f1"],
                    "val_weighted_f1": m["weighted_f1"],
                }
            )

            # Checkpoint selection: validation macro F1 only. The test split
            # plays no part in this decision.
            if m["macro_f1"] > best_f1:
                best_f1, best_epoch = m["macro_f1"], epoch
                save_checkpoint(model, tokenizer, best_dir)
                print(f"  -> new best (macro F1 {best_f1:.4f}); checkpoint saved")
            print()

        print("BEST CHECKPOINT")
        print(f"  Best epoch:               {best_epoch}")
        print(f"  Best validation macro F1: {best_f1:.4f}")
        print()

        # --- test: reload the best checkpoint, evaluate exactly once ---
        banner("TEST EVALUATION (single pass, best validation checkpoint)")
        print()
        model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
        _, y_true, y_pred, probs = evaluate(model, loaders["test"], device)
        test_metrics = compute_metrics(y_true, y_pred, id_order)

        from sklearn.metrics import classification_report

        target_names = [str(lab) for lab in labels]
        report = classification_report(
            y_true,
            y_pred,
            labels=id_order,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        confusion = build_confusion_frame(y_true, y_pred, id_order, labels)
        predictions = generate_predictions(frames["test"], y_true, y_pred, probs, id2label, labels)
        errors = predictions[predictions["true_label"] != predictions["predicted_label"]][
            [TEXT_COLUMN, "true_label", "predicted_label", "confidence"]
        ].copy()

        config = {
            "experiment": "bert_human_llm_severity_7class",
            "note": "Standalone experiment. Uses ONLY data/gold/splits/human_llm/*. "
                    "Original severity labels 0-6 preserved; no relabeling.",
            "model_name": MODEL_NAME,
            "tokenizer": TOKENIZER_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "optimizer": "AdamW",
            "lr_schedule": "linear with warmup",
            "class_weights_enabled": USE_CLASS_WEIGHTS,
            "random_seed": SEED,
            "device": str(device),
            "num_labels": len(labels),
            "label_mapping": {str(lab): i for lab, i in label2id.items()},
            "train_size": len(frames["train"]),
            "validation_size": len(frames["validation"]),
            "test_size": len(frames["test"]),
            "dataset_dir": str(data_dir),
            "text_column": TEXT_COLUMN,
            "target_column": LABEL_COLUMN,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "model_selection_metric": "validation_macro_f1",
            "token_length_stats": token_stats,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        save_results(
            out_dir,
            {
                "bert_test_metrics.json": test_metrics,
                "bert_classification_report.json": report,
                "bert_training_history.json": history,
                "bert_config.json": config,
                "bert_confusion_matrix.csv": confusion,
                "bert_test_predictions.csv": predictions,
                "bert_errors.csv": errors,
            },
        )
        save_confusion_png(confusion, out_dir / "confusion_matrix.png")

        print_final_report(
            frames, labels, best_epoch, best_f1, test_metrics, report, confusion, errors, out_dir
        )
        return test_metrics
    finally:
        sys.stdout = original_stdout
        tee.close()


def print_final_report(
    frames, labels, best_epoch, best_f1, test_metrics, report, confusion, errors, out_dir
) -> None:
    banner("FINAL BERT RESULTS")
    print()
    print("Dataset:\nHuman/LLM SuicideWatch dataset")
    print(f"\nTrain:\n{len(frames['train'])}")
    print(f"\nValidation:\n{len(frames['validation'])}")
    print(f"\nTest:\n{len(frames['test'])}")
    print(f"\nNumber of classes:\n{len(labels)}")
    print(f"\nModel:\n{MODEL_NAME}")
    print(f"\nBest epoch:\n{best_epoch}")
    print(f"\nBest validation macro F1:\n{best_f1:.4f}")

    print("\n-------------------------------")
    print("TEST RESULTS")
    print("-------------------------------")
    for key, label in [
        ("accuracy", "Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
        ("macro_f1", "Macro F1"),
        ("weighted_precision", "Weighted Precision"),
        ("weighted_recall", "Weighted Recall"),
        ("weighted_f1", "Weighted F1"),
    ]:
        print(f"\n{label}:\n{test_metrics[key]:.4f}")

    print("\n-------------------------------")
    print("PER-CLASS RESULTS")
    print("-------------------------------\n")
    print(f"{'Class':>6} | {'Precision':>9} | {'Recall':>7} | {'F1':>7} | {'Support':>7}")
    print("-" * 50)
    for lab in labels:
        r = report[str(lab)]
        print(
            f"{lab:>6} | {r['precision']:>9.4f} | {r['recall']:>7.4f} | "
            f"{r['f1-score']:>7.4f} | {int(r['support']):>7}"
        )

    print("\n-------------------------------")
    print("CONFUSION MATRIX")
    print("-------------------------------\n")
    print("(rows = true severity, columns = predicted severity)\n")
    print(confusion.to_string())

    print("\n-------------------------------")
    print("ERRORS")
    print("-------------------------------")
    print(f"\nNumber of test errors:\n{len(errors)}")
    if len(errors):
        print("\nSample of misclassified test examples:\n")
        for _, row in errors.head(5).iterrows():
            text = str(row[TEXT_COLUMN]).replace("\n", " ").replace("\r", " ")
            snippet = text[:160] + ("..." if len(text) > 160 else "")
            print(
                f"  true={row['true_label']}  pred={row['predicted_label']}  "
                f"conf={row['confidence']:.3f}"
            )
            print(f"    {snippet}\n")

    print("=" * 50)
    print(f"\nAll artifacts written to:\n  {out_dir}")
    print(f"Best model saved to:\n  {out_dir / 'best_model'}")
    print()


if __name__ == "__main__":
    run_experiment()
