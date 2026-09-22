"""Create presentation-ready test-set comparisons for the frozen current models.

This script deliberately reuses ``backend.app.FrozenModels`` so its per-text
inference path is the same one exposed by the frontend API. It never fits or
writes a model artifact.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app import LABELS, FrozenModels

TEST_PATH = ROOT_DIR / "data" / "final_datasets" / "splits" / "test.csv"
OUTPUT_DIR = ROOT_DIR / "results" / "model_comparison"
CLASS_IDS = list(LABELS)
CLASS_NAMES = [LABELS[label] for label in CLASS_IDS]
DISPLAY_NAMES = ["No crisis", "Implicit crisis", "Explicit crisis"]
COLORS = {"tfidf": "#355C7D", "bert": "#C06C84"}


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def model_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, predicted, labels=CLASS_IDS, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    indices = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    return float(sum(
        (indices == bin_index).mean()
        * abs(confidence[indices == bin_index].mean() - correct[indices == bin_index].mean())
        for bin_index in range(bins)
        if (indices == bin_index).any()
    ))


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small Markdown table without adding the optional tabulate package."""
    columns = [str(column) for column in frame.columns]
    lines = ["| model | " + " | ".join(columns) + " |", "|---|" + "|".join("---:" for _ in columns) + "|"]
    for index, row in frame.iterrows():
        lines.append("| " + str(index) + " | " + " | ".join(f"{float(value):.4f}" for value in row) + " |")
    return "\n".join(lines)


def plot_overall(metrics: pd.DataFrame) -> None:
    labels = ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1-score\n(macro)"]
    columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    positions = np.arange(len(columns))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5.3))
    for offset, (model, title) in enumerate((("tfidf", "TF-IDF + Logistic Regression"), ("bert", "BERT"))):
        values = metrics.loc[model, columns].to_numpy(dtype=float)
        bars = axis.bar(positions + (offset - 0.5) * width, values, width, label=title, color=COLORS[model])
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=9)
    axis.set(title="Overall performance on the frozen test set", ylabel="Score", xticks=positions, xticklabels=labels, ylim=(0, 1.12))
    axis.yaxis.set_major_formatter("{x:.0%}")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.22)
    save_figure("overall_metrics.png")


def plot_per_class_f1(true: np.ndarray, predictions: dict[str, np.ndarray]) -> None:
    positions = np.arange(len(CLASS_IDS))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8.5, 5.3))
    for offset, (model, title) in enumerate((("tfidf", "TF-IDF + Logistic Regression"), ("bert", "BERT"))):
        _, _, f1, _ = precision_recall_fscore_support(true, predictions[model], labels=CLASS_IDS, zero_division=0)
        bars = axis.bar(positions + (offset - 0.5) * width, f1, width, label=title, color=COLORS[model])
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in f1], padding=3, fontsize=9)
    axis.set(title="Per-class F1 score", ylabel="F1 score", xticks=positions, xticklabels=DISPLAY_NAMES, ylim=(0, 1.12))
    axis.yaxis.set_major_formatter("{x:.0%}")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.22)
    save_figure("per_class_f1.png")


def plot_confusion(true: np.ndarray, predicted: np.ndarray, model: str, title: str, maximum: int) -> None:
    matrix = confusion_matrix(true, predicted, labels=CLASS_IDS)
    fig, axis = plt.subplots(figsize=(6.3, 5.2))
    image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
    for row in range(len(CLASS_IDS)):
        for column in range(len(CLASS_IDS)):
            color = "white" if matrix[row, column] > maximum / 2 else "#172554"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    axis.set(title=title, xlabel="Predicted class", ylabel="Actual class", xticks=range(3), yticks=range(3), xticklabels=DISPLAY_NAMES, yticklabels=DISPLAY_NAMES)
    plt.setp(axis.get_xticklabels(), rotation=15, ha="right")
    fig.colorbar(image, ax=axis, label="Test samples")
    save_figure(f"confusion_matrix_{model}.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(TEST_PATH)
    required = {"content", "label"}
    if missing := required - set(test.columns):
        raise ValueError(f"Current test CSV is missing required columns: {sorted(missing)}")
    if test["content"].isna().any() or test["label"].isna().any():
        raise ValueError("Current test CSV contains missing content or labels")
    true = test["label"].to_numpy(dtype=int)
    if set(true) - set(CLASS_IDS):
        raise ValueError("Current test CSV contains labels outside 0, 1, 2")

    models = FrozenModels()
    rows = []
    for text, label in zip(test["content"].astype(str), true, strict=True):
        tfidf = models.predict_tfidf(text)
        bert = models.predict_bert(text)
        for result in (tfidf, bert):
            if set(result["probabilities"]) != set(CLASS_NAMES) or not np.isclose(sum(result["probabilities"].values()), 1.0):
                raise ValueError("Model returned an invalid probability distribution")
        rows.append({
            "text": text, "true_label": int(label), "true_class_name": LABELS[int(label)],
            "tfidf_prediction": tfidf["label"], "tfidf_class_name": tfidf["class_name"],
            "tfidf_confidence": max(tfidf["probabilities"].values()),
            "bert_prediction": bert["label"], "bert_class_name": bert["class_name"],
            "bert_confidence": max(bert["probabilities"].values()),
        })
    predictions = pd.DataFrame(rows)
    if len(predictions) != len(test) or predictions.isna().any().any():
        raise ValueError("Prediction count or completeness check failed")
    predictions["tfidf_correct"] = predictions["tfidf_prediction"] == predictions["true_label"]
    predictions["bert_correct"] = predictions["bert_prediction"] == predictions["true_label"]
    predictions["models_agree"] = predictions["tfidf_prediction"] == predictions["bert_prediction"]
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    predicted = {name: predictions[f"{name}_prediction"].to_numpy(dtype=int) for name in ("tfidf", "bert")}
    metrics = pd.DataFrame({name: model_metrics(true, values) for name, values in predicted.items()}).T
    metrics.index.name = "model"
    metrics.to_csv(OUTPUT_DIR / "metrics.csv")
    per_class_rows = []
    for model, values in predicted.items():
        precision, recall, f1, support = precision_recall_fscore_support(true, values, labels=CLASS_IDS, zero_division=0)
        per_class_rows.extend(
            {"model": model, "class_name": LABELS[class_id], "precision": p, "recall": r, "f1": score, "support": count}
            for class_id, p, r, score, count in zip(CLASS_IDS, precision, recall, f1, support, strict=True)
        )
    per_class_metrics = pd.DataFrame(per_class_rows)
    per_class_metrics.to_csv(OUTPUT_DIR / "per_class_metrics.csv", index=False)
    plot_overall(metrics)
    plot_per_class_f1(true, predicted)
    maximum = max(confusion_matrix(true, values, labels=CLASS_IDS).max() for values in predicted.values())
    plot_confusion(true, predicted["tfidf"], "tfidf", "TF-IDF + Logistic Regression: confusion matrix", maximum)
    plot_confusion(true, predicted["bert"], "bert", "BERT: confusion matrix", maximum)

    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    box = axis.boxplot([predictions["tfidf_confidence"], predictions["bert_confidence"]], tick_labels=["TF-IDF + LR", "BERT"], patch_artist=True)
    for patch, color in zip(box["boxes"], (COLORS["tfidf"], COLORS["bert"])): patch.set_facecolor(color)
    axis.set(title="Distribution of predicted-class confidence", ylabel="Confidence", ylim=(0, 1.05))
    axis.yaxis.set_major_formatter("{x:.0%}")
    axis.grid(axis="y", alpha=0.22)
    save_figure("confidence_distribution.png")

    fig, axis = plt.subplots(figsize=(9, 5.4))
    groups = [("TF-IDF + LR\nCorrect", "tfidf", True), ("TF-IDF + LR\nIncorrect", "tfidf", False), ("BERT\nCorrect", "bert", True), ("BERT\nIncorrect", "bert", False)]
    values = [predictions.loc[predictions[f"{model}_correct"] == correct, f"{model}_confidence"] for _, model, correct in groups]
    box = axis.boxplot(values, tick_labels=[label for label, _, _ in groups], patch_artist=True)
    for patch, (_, model, _) in zip(box["boxes"], groups): patch.set_facecolor(COLORS[model])
    axis.set(title="Confidence by prediction correctness", ylabel="Confidence", ylim=(0, 1.05))
    axis.yaxis.set_major_formatter("{x:.0%}")
    axis.grid(axis="y", alpha=0.22)
    save_figure("confidence_vs_correctness.png")

    agreement_counts = predictions["models_agree"].value_counts().reindex([True, False], fill_value=0)
    fig, axis = plt.subplots(figsize=(6.6, 5.1))
    bars = axis.bar(["Agreement", "Disagreement"], agreement_counts.to_numpy(), color=["#4C956C", "#E76F51"])
    axis.bar_label(bars, labels=[str(value) for value in agreement_counts], padding=3)
    axis.set(title="Model agreement on the frozen test set", ylabel="Test samples", ylim=(0, max(agreement_counts) * 1.18))
    axis.grid(axis="y", alpha=0.22)
    save_figure("model_agreement.png")

    agreed = predictions[predictions["models_agree"]]
    by_class = agreed["tfidf_class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    fig, axis = plt.subplots(figsize=(7.5, 5.1))
    bars = axis.bar(DISPLAY_NAMES, by_class.to_numpy(), color=["#4C956C", "#6A994E", "#386641"])
    axis.bar_label(bars, labels=[str(value) for value in by_class], padding=3)
    axis.set(title="Agreed predictions by class", ylabel="Agreed test samples", ylim=(0, max(by_class.max(), 1) * 1.2))
    axis.grid(axis="y", alpha=0.22)
    save_figure("agreement_by_class.png")

    calibration = {}
    fig, axis = plt.subplots(figsize=(7, 5.5))
    for model, title in (("tfidf", "TF-IDF + LR"), ("bert", "BERT")):
        confidence = predictions[f"{model}_confidence"].to_numpy()
        correct = predictions[f"{model}_correct"].to_numpy(dtype=int)
        observed, expected = calibration_curve(correct, confidence, n_bins=10, strategy="uniform")
        axis.plot(expected, observed, marker="o", label=title, color=COLORS[model])
        calibration[model] = {"expected_calibration_error": expected_calibration_error(confidence, correct)}
    axis.plot([0, 1], [0, 1], "--", color="#6B7280", label="Perfect calibration")
    axis.set(title="Confidence calibration (predicted class)", xlabel="Mean predicted confidence", ylabel="Observed accuracy", xlim=(0, 1), ylim=(0, 1))
    axis.xaxis.set_major_formatter("{x:.0%}")
    axis.yaxis.set_major_formatter("{x:.0%}")
    axis.legend(frameon=False)
    axis.grid(alpha=0.22)
    save_figure("calibration.png")

    confidence_stats = predictions[["tfidf_confidence", "bert_confidence"]].describe().T[["mean", "std", "min", "50%", "max"]].rename(columns={"50%": "median"})
    agreement_percentage = float(predictions["models_agree"].mean())
    report = [
        "# Current model comparison", "",
        f"- Test CSV: `{TEST_PATH.relative_to(ROOT_DIR)}`", f"- Test samples evaluated: **{len(test)}**", "- Input: `content`; target: `label`", "- Mapping: `0=no_crisis`, `1=implicit_crisis`, `2=explicit_crisis`", "- Inference: the same `backend.app.FrozenModels` methods used by the frontend API.", "",
        "## Overall metrics", "", markdown_table(metrics), "",
        "## Per-class metrics", "", markdown_table(per_class_metrics.set_index(["model", "class_name"])[["precision", "recall", "f1", "support"]]), "",
        "## Confidence statistics", "", markdown_table(confidence_stats), "",
        "## Agreement", "", f"- Agreement: **{int(agreement_counts[True])}/{len(test)} ({agreement_percentage:.1%})**", f"- Disagreement: **{int(agreement_counts[False])}/{len(test)} ({1 - agreement_percentage:.1%})**", "",
        "## Calibration", "", f"- TF-IDF expected calibration error: {calibration['tfidf']['expected_calibration_error']:.4f}", f"- BERT expected calibration error: {calibration['bert']['expected_calibration_error']:.4f}", "",
        "Higher confidence is not interpreted as better predictive performance. Performance, confidence, and calibration are reported separately.",
    ]
    (OUTPUT_DIR / "evaluation_summary.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (OUTPUT_DIR / "metrics.json").open("w", encoding="utf-8") as output:
        json.dump({"overall": metrics.to_dict(orient="index"), "per_class": json.loads(per_class_metrics.to_json(orient="records")), "agreement_percentage": agreement_percentage, "calibration": calibration}, output, indent=2)


if __name__ == "__main__":
    main()
