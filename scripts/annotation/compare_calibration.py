"""Compare two independent calibration annotation files.

Run from the repository root with::

    python scripts/annotation/compare_calibration.py

The script validates both keyed CSV files, writes a complete comparison report,
the row-level disagreement table, summary CSVs, and report-ready PNG figures.
Existing outputs are never overwritten.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.annotation.agreement import cohens_kappa, interpret  # noqa: E402
from src.utils.io import read_table  # noqa: E402
from src.utils.labels import LABELS  # noqa: E402


REQUIRED_COLUMNS = {"id", "text", "label", "confidence", "notes"}
CONFIDENCES = ("high", "medium", "low")


def available_path(path: Path) -> Path:
    """Return a non-existing path, adding a numeric suffix when necessary."""
    if not path.exists():
        return path
    for number in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{number}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find an unused output name for {path}")


def validate_frame(frame: pd.DataFrame, name: str) -> dict[str, object]:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    ids = frame["id"].astype("string")
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "duplicate_ids": ids[ids.duplicated(keep=False)].drop_duplicates().tolist(),
        "blank_ids": ids[ids.isna() | ids.str.strip().eq("")].tolist(),
        "invalid_labels": sorted(set(frame["label"].dropna()) - set(LABELS)),
        "invalid_confidences": sorted(set(frame["confidence"].dropna()) - set(CONFIDENCES)),
    }


def md_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    headers = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(md_cell(value) for value in row) + " |")
    return "\n".join(lines)


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def make_figures(merged: pd.DataFrame, matrix: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    positions = np.arange(len(LABELS))
    width = 0.38
    distributions = pd.DataFrame({"A": merged["A_label"].value_counts(), "B": merged["B_label"].value_counts()}).reindex(LABELS, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(positions - width / 2, distributions["A"], width, label="Annotator A", color="#176b87")
    ax.bar(positions + width / 2, distributions["B"], width, label="Annotator B", color="#e07a3f")
    for offset, column in [(-width / 2, "A"), (width / 2, "B")]:
        for index, value in enumerate(distributions[column]):
            ax.text(index + offset, value + 0.5, str(value), ha="center", va="bottom")
    ax.set_xticks(positions, LABELS, rotation=20, ha="right")
    ax.set_ylabel("Number of annotations")
    ax.set_title("Label Distribution: Annotator A vs B")
    ax.legend()
    figures["label_distribution"] = available_path(figure_dir / "label_distribution_A_vs_B.png")
    save_figure(figures["label_distribution"])

    status = pd.Series(np.where(merged["A_label"] == merged["B_label"], "Agreement", "Disagreement")).value_counts().reindex(["Agreement", "Disagreement"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(status.index, status.values, color=["#2a9d8f", "#d1495b"])
    for bar, value in zip(bars, status.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(value), ha="center", va="bottom")
    ax.set_ylabel("Number of matched rows")
    ax.set_title("Overall Label Agreement vs Disagreement")
    figures["overall"] = available_path(figure_dir / "overall_agreement.png")
    save_figure(figures["overall"])

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    fig.colorbar(image, ax=ax, label="Number of rows")
    ax.set_xticks(positions, LABELS, rotation=30, ha="right")
    ax.set_yticks(positions, LABELS)
    ax.set_xlabel("Annotator B label")
    ax.set_ylabel("Annotator A label")
    ax.set_title("A vs B Label Confusion Matrix")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            ax.text(column, row, str(int(matrix.iloc[row, column])), ha="center", va="center")
    figures["confusion"] = available_path(figure_dir / "confusion_matrix_A_vs_B.png")
    save_figure(figures["confusion"])

    pair_counts = merged.loc[merged["A_label"] != merged["B_label"]].apply(
        lambda row: " ↔ ".join(sorted((row["A_label"], row["B_label"]))), axis=1
    ).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, max(4, len(pair_counts) * 0.5)))
    bars = ax.barh(pair_counts.index, pair_counts.values, color="#c96b4b")
    for bar, value in zip(bars, pair_counts.values):
        ax.text(value + 0.1, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    ax.set_xlabel("Number of disagreements")
    ax.set_title("Disagreement Pair Distribution")
    figures["pairs"] = available_path(figure_dir / "disagreement_pairs.png")
    save_figure(figures["pairs"])

    confidence_rows = []
    for annotator in ("A", "B"):
        confidence_rows.append(pd.DataFrame({
            "annotator": annotator,
            "confidence": merged[f"{annotator}_confidence"],
            "status": np.where(merged["A_label"] == merged["B_label"], "Agreement", "Disagreement"),
        }))
    confidence_frame = pd.concat(confidence_rows, ignore_index=True)
    confidence_counts = confidence_frame.groupby(["confidence", "status"]).size().unstack(fill_value=0).reindex(CONFIDENCES, fill_value=0).reindex(columns=["Agreement", "Disagreement"], fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 6))
    confidence_counts.plot.bar(ax=ax, color=["#2a9d8f", "#d1495b"])
    ax.set_xlabel("Reported confidence")
    ax.set_ylabel("Annotation count")
    ax.set_title("Agreement and Disagreement by Annotator-Reported Confidence")
    ax.legend(title="Outcome")
    ax.tick_params(axis="x", rotation=0)
    figures["confidence"] = available_path(figure_dir / "confidence_vs_agreement.png")
    save_figure(figures["confidence"])

    per_label = pd.DataFrame(index=LABELS)
    per_label["Agreement"] = [int(((merged["A_label"] == label) & (merged["B_label"] == label)).sum()) for label in LABELS]
    per_label["Disagreement involving label"] = [int(((merged["A_label"] == label) ^ (merged["B_label"] == label)).sum()) for label in LABELS]
    fig, ax = plt.subplots(figsize=(10, 6))
    per_label.plot.bar(ax=ax, color=["#2a9d8f", "#d1495b"])
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of matched rows")
    ax.set_title("Per-Label Agreement and Disagreement Involvement")
    ax.legend(title="Outcome")
    ax.tick_params(axis="x", rotation=20)
    figures["per_label"] = available_path(figure_dir / "per_label_agreement.png")
    save_figure(figures["per_label"])
    return figures


def build_report(a: pd.DataFrame, b: pd.DataFrame, merged: pd.DataFrame, validation: dict[str, object], matrix: pd.DataFrame, per_label: pd.DataFrame, confidence_counts: pd.DataFrame, figures: dict[str, Path], disagreement_file: Path, summary_file: Path, output_dir: Path) -> str:
    matched = len(merged)
    agreements = int((merged["A_label"] == merged["B_label"]).sum())
    disagreements = matched - agreements
    kappa = cohens_kappa(merged["A_label"], merged["B_label"], labels=LABELS)
    pair_counts = merged.loc[merged["A_label"] != merged["B_label"]].apply(lambda row: " ↔ ".join(sorted((row["A_label"], row["B_label"]))), axis=1).value_counts()
    text_mismatches = int((merged["A_text"] != merged["B_text"]).sum())
    order_same = list(a["id"]) == list(b["id"])
    distribution = pd.DataFrame({"A count": a["label"].value_counts(), "B count": b["label"].value_counts()}).reindex(LABELS, fill_value=0).rename_axis("label").reset_index()
    disagreement_view = merged.loc[merged["A_label"] != merged["B_label"], ["id", "A_text", "A_label", "B_label", "A_confidence", "B_confidence"]].rename(columns={"A_text": "text"})
    pair_table = pair_counts.rename("count").rename_axis("label_pair").reset_index()
    per_label_table = per_label.reset_index().rename(columns={"index": "label"})
    confidence_table = confidence_counts.reset_index().rename(columns={"confidence": "confidence"})

    figure_lines = []
    descriptions = {
        "label_distribution": "Compares the number of instances assigned to each of the four labels by annotators A and B.",
        "overall": "Shows the actual number of matching and disagreeing labels across keyed matched rows.",
        "confusion": "Shows the 4x4 count matrix of A labels against B labels, with the actual count in every cell.",
        "pairs": "Ranks only the label pairs that actually disagree, using a symmetric pair name.",
        "confidence": "Separately counts A and B confidence entries by whether the keyed labels agree or disagree.",
        "per_label": "Shows exact same-label agreements and rows where each label appears on one side only.",
    }
    for number, key in enumerate(("label_distribution", "overall", "confusion", "pairs", "confidence", "per_label"), 1):
        relative = figures[key].relative_to(output_dir).as_posix()
        figure_lines.append(f"### Figure {number} - {descriptions[key].split('.')[0]}\n`{relative}`\n\n{descriptions[key]}\n")

    pattern_lines = []
    for pair in ["explicit_crisis ↔ implicit_crisis", "hard_negative ↔ non_crisis", "explicit_crisis ↔ hard_negative", "implicit_crisis ↔ non_crisis"]:
        count = int(pair_counts.get(pair, 0))
        pattern_lines.append(f"- `{pair}`: {count} disagreement(s).")
    low_disagreements = int(((merged["A_label"] != merged["B_label"]) & ((merged["A_confidence"] == "low") | (merged["B_confidence"] == "low"))).sum())
    medium_disagreements = int(((merged["A_label"] != merged["B_label"]) & ((merged["A_confidence"] == "medium") | (merged["B_confidence"] == "medium"))).sum())

    validation_text = (
        f"- A rows: {len(a)}; columns: `{', '.join(a.columns)}`\n"
        f"- B rows: {len(b)}; columns: `{', '.join(b.columns)}`\n"
        f"- A IDs unique: {not validation['A']['duplicate_ids']}; B IDs unique: {not validation['B']['duplicate_ids']}\n"
        f"- Duplicate A IDs: {', '.join(validation['A']['duplicate_ids']) or 'none'}\n"
        f"- Duplicate B IDs: {', '.join(validation['B']['duplicate_ids']) or 'none'}\n"
        f"- IDs only in A: {len(validation['ids_only_a'])}; IDs only in B: {len(validation['ids_only_b'])}\n"
        f"- Matched IDs: {matched}; text mismatches among matched IDs: {text_mismatches}\n"
        f"- Row ordering identical: {order_same}; keyed matching was used regardless of order.\n"
    )
    report = f"""# Calibration 150 A vs B Agreement Report

This report compares independent annotations without changing either source file. All statistics and figures are computed directly from the two input CSV files.

## 1. Dataset validation
{validation_text}

The input columns are `{', '.join(a.columns)}`. IDs are the stable join key. Both files contain {len(a)} and {len(b)} rows respectively; the expected calibration size is 150 per file.

## 2. Overall agreement
- Total matched rows: {matched}
- Matching labels: {agreements}
- Disagreements: {disagreements}
- Overall label agreement: {agreements / matched * 100:.2f}%
- Cohen's kappa: {kappa:.6f} ({interpret(kappa)})

## 3. Label distribution
{markdown_table(distribution)}

## 4. Agreement matrix
Rows are A labels and columns are B labels.

{markdown_table(matrix.rename_axis("A label").reset_index())}

## 5. Disagreement breakdown
{markdown_table(pair_table) if not pair_table.empty else 'No disagreements.'}

The primary observed boundaries are listed directly from the computed pair counts. No disagreement is resolved automatically.
{chr(10).join(pattern_lines)}

## 6. Confidence analysis

Confidence is reported separately by annotator because it measures certainty in each label decision; A and B confidence values are not silently combined. There were {low_disagreements} disagreements with at least one low-confidence annotation and {medium_disagreements} with at least one medium-confidence annotation.

{markdown_table(confidence_table)}

## 7. Disagreement cases

Every keyed row where A's label differs from B's label is in [{disagreement_file.name}]({disagreement_file.name}). The report table includes ID, text, both labels, and both confidence values; the CSV additionally preserves both notes.

{markdown_table(disagreement_view)}

## 8. Annotation guideline issues

These are review signals, not adjudications. The disagreement pairs suggest that calibration should explicitly revisit any nonzero boundary above, especially explicit versus implicit crisis, crisis-level distress versus ordinary sadness or vague hopelessness, and crisis terminology used without the author's own current crisis. Text mismatches ({text_mismatches}) also warrant checking whether both annotators evaluated the same text version.

Review examples involving passive death wishes, historical suicidal behavior, dark humor or sarcasm, creative or fictional statements, third-party crisis references, very short or vague posts, and self-harm without explicit suicidal intent when they appear in the disagreement CSV. The source notes should guide that review, but neither annotator is treated as ground truth here.

## 9. Recommended calibration decisions

1. Manually review every disagreement before producing the gold labels, prioritizing rows where either confidence is `low`, then rows with the most frequent disagreement pairs.
2. Clarify the operational boundary between explicit and implicit crisis, including passive death wishes and historical or non-current statements.
3. Add examples for third-party or educational references, creative or sarcastic language, short vague posts, and self-harm without explicit suicidal intent if those cases occur in the disagreement set.
4. Re-run this script after any guideline revision using fresh independent annotations; do not edit A or B to improve the statistic.

## Figures

{chr(10).join(figure_lines)}

## Reproducibility

- Script: [scripts/annotation/compare_calibration.py](../../../scripts/annotation/compare_calibration.py)
- Inputs: `data/gold/annotation/calibration_150_A.csv` and `data/gold/annotation/calibration_150_B.csv`
- Output directory: `{output_dir.name}/`
- Generated outputs: [{disagreement_file.name}]({disagreement_file.name}), [{summary_file.name}]({summary_file.name}), this report, and the six PNG files under `figures/`.
- Measures: keyed dataset validation, exact label counts, agreement percentage, Cohen's kappa, 4x4 confusion matrix, disagreement-pair counts, per-label counts, and confidence-by-outcome counts.

The comparison is reproducible because no labels are inferred, corrected, sampled, or filled in; every matched row comes from the source CSVs and every figure is generated from the same computed comparison tables used above.
"""
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", default="data/gold/annotation/calibration_150_A.csv")
    parser.add_argument("--b", default="data/gold/annotation/calibration_150_B.csv")
    parser.add_argument("--outdir", default="data/gold/annotation/agreement")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    a_path, b_path = (root / args.a).resolve(), (root / args.b).resolve()
    output_dir = (root / args.outdir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    a, b = read_table(a_path), read_table(b_path)
    a_validation, b_validation = validate_frame(a, "A"), validate_frame(b, "B")
    if a_validation["duplicate_ids"] or b_validation["duplicate_ids"] or a_validation["blank_ids"] or b_validation["blank_ids"]:
        raise ValueError("Duplicate or blank IDs found; refusing to silently ignore them")
    if a_validation["invalid_labels"] or b_validation["invalid_labels"] or a_validation["invalid_confidences"] or b_validation["invalid_confidences"]:
        raise ValueError("Unknown labels or confidence values found")
    a = a.copy(); b = b.copy()
    a["id"], b["id"] = a["id"].astype(str), b["id"].astype(str)
    ids_a, ids_b = set(a["id"]), set(b["id"])
    validation = {"A": a_validation, "B": b_validation, "ids_only_a": sorted(ids_a - ids_b), "ids_only_b": sorted(ids_b - ids_a)}
    merged = a.rename(columns={column: f"A_{column}" for column in a.columns}).merge(
        b.rename(columns={column: f"B_{column}" for column in b.columns}), left_on="A_id", right_on="B_id", how="outer", indicator=True
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("A and B do not contain the same IDs; refusing to silently drop unmatched rows")
    merged = merged.drop(columns=["_merge"]).rename(columns={"A_id": "id"})
    matrix = pd.crosstab(pd.Categorical(merged["A_label"], categories=LABELS), pd.Categorical(merged["B_label"], categories=LABELS), dropna=False).reindex(index=LABELS, columns=LABELS, fill_value=0)
    per_label = pd.DataFrame(index=LABELS)
    per_label["A count"] = [int((merged["A_label"] == label).sum()) for label in LABELS]
    per_label["B count"] = [int((merged["B_label"] == label).sum()) for label in LABELS]
    per_label["same-label agreement"] = [int(((merged["A_label"] == label) & (merged["B_label"] == label)).sum()) for label in LABELS]
    per_label["disagreement involving label"] = [int(((merged["A_label"] == label) ^ (merged["B_label"] == label)).sum()) for label in LABELS]
    confidence_rows = []
    for annotator in ("A", "B"):
        confidence_rows.append(pd.DataFrame({"annotator": annotator, "confidence": merged[f"{annotator}_confidence"], "status": np.where(merged["A_label"] == merged["B_label"], "Agreement", "Disagreement")}))
    confidence_counts = pd.concat(confidence_rows, ignore_index=True).groupby(["confidence", "status"]).size().unstack(fill_value=0).reindex(CONFIDENCES, fill_value=0).reindex(columns=["Agreement", "Disagreement"], fill_value=0)
    disagreement = merged.loc[merged["A_label"] != merged["B_label"], ["id", "A_text", "A_label", "B_label", "A_confidence", "B_confidence", "A_notes", "B_notes"]].rename(columns={"A_text": "text", "A_label": "A label", "B_label": "B label", "A_confidence": "A confidence", "B_confidence": "B confidence", "A_notes": "A notes", "B_notes": "B notes"})
    disagreement_path = available_path(output_dir / "calibration_150_disagreements.csv")
    disagreement.to_csv(disagreement_path, index=False, encoding="utf-8")
    summary = per_label.reset_index().rename(columns={"index": "label"})
    summary["overall_agreement"] = int((merged["A_label"] == merged["B_label"]).sum())
    summary["overall_disagreement"] = int((merged["A_label"] != merged["B_label"]).sum())
    summary["cohens_kappa"] = cohens_kappa(merged["A_label"], merged["B_label"], labels=LABELS)
    summary_path = available_path(output_dir / "calibration_150_agreement_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    figures = make_figures(merged, matrix, output_dir)
    report_path = available_path(output_dir / "calibration_150_agreement_report.md")
    report_path.write_text(build_report(a, b, merged, validation, matrix, per_label, confidence_counts, figures, disagreement_path, summary_path, output_dir), encoding="utf-8")
    print(f"A rows: {len(a)}; B rows: {len(b)}; matched: {len(merged)}")
    print(f"Agreements: {(merged['A_label'] == merged['B_label']).sum()}; disagreements: {len(disagreement)}")
    print(f"Cohen's kappa: {cohens_kappa(merged['A_label'], merged['B_label'], labels=LABELS):.6f}")
    print(f"Report: {report_path}")
    print(f"Disagreements: {disagreement_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())