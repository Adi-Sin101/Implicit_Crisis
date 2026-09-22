"""Render the target-class precision, recall, and F1 comparison from saved metrics."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results" / "model_comparison"


def main() -> None:
    metrics = pd.read_csv(RESULTS_DIR / "per_class_metrics.csv")
    target = metrics.loc[metrics["class_name"] == "implicit_crisis"].set_index("model")
    expected_models = ["tfidf", "bert"]
    if set(target.index) != set(expected_models):
        raise ValueError("Saved metrics must contain implicit_crisis results for TF-IDF and BERT")

    labels = ["Precision", "Recall", "F1-score"]
    columns = ["precision", "recall", "f1"]
    positions = np.arange(len(columns))
    width = 0.36
    colors = {"tfidf": "#355C7D", "bert": "#C06C84"}
    titles = {"tfidf": "TF-IDF + Logistic Regression", "bert": "BERT"}

    fig, axis = plt.subplots(figsize=(9, 5.3))
    for offset, model in enumerate(expected_models):
        values = target.loc[model, columns].to_numpy(dtype=float)
        bars = axis.bar(positions + (offset - 0.5) * width, values, width, label=titles[model], color=colors[model])
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=9)
    axis.set(
        ylabel="Score",
        xticks=positions,
        xticklabels=labels,
        ylim=(0, 1.12),
    )
    axis.set_title("Implicit crisis detection: precision, recall, and F1", pad=22)
    axis.text(0.5, 1.01, f"Frozen test set · support = {int(target.loc['tfidf', 'support'])}", transform=axis.transAxes, ha="center", va="bottom", color="#4B5563", fontsize=10)
    axis.yaxis.set_major_formatter("{x:.0%}")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.22)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "implicit_crisis_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
