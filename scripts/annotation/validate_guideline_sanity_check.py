"""Generate and validate a qualitative guideline v2 sanity-check sample."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.labels import LABELS  # noqa: E402


SAMPLE = {
    # Required adjudicated disagreements.
    "cams-24c0c25e6a33": ("explicit vs implicit", "Rules 3-5", "Does self-reference inside a suicide-forum discussion make the crisis explicit?"),
    "cams-687449dcc3c5": ("historical crisis; implicit vs non-crisis", "Rule 6 and Rule 8", "How should past darkest moments be separated from current crisis-level distress?"),
    "irf-5f27a24f37fa": ("explicit vs implicit; second-person/general", "Rules 2-4", "When does second-person language still describe lived personal crisis?"),
    "irf-a11b2a6bb141": ("implicit vs non-crisis; ordinary distress", "Rule 8", "When does persistent loneliness and self-loathing cross the implicit-crisis threshold?"),
    "komati-c0f53895b28d": ("hard-negative vs non-crisis; referenced crisis", "Rule 5", "Are suicide-related community names enough for hard_negative?"),
    "komati-e335a390c977": ("euphemistic language; explicit vs non-crisis", "Rules 1-2 and Rule 4", "How should a first-person euphemism be handled when hyperbole is possible?"),
    "sdcnl-9a7998c1beff": ("denial of suicide; implicit vs non-crisis", "Rules 4 and 7", "Does discussing avoided suicide and self-harm indicate current crisis?"),
    "sdcnl-ca2a4047d113": ("ambiguous/short text; implicit vs non-crisis", "Rule 7", "Is a very short hopeless statement sufficient evidence for implicit crisis?"),
    # Agreed boundary cases selected from the calibration set.
    "sdcnl-a16d8fd4e632": ("passive death wish", "Rules 2-4", "Does fantasizing about death in detail without stated intent count as crisis?"),
    "komati-dea6b540f261": ("dark humor/absurdity; euphemistic language", "Rules 1-2 and Rule 4", "How should an absurd reincarnation joke containing suicide language be labelled?"),
    "cams-17316472e74a7": ("quoted/reference crisis; hard-negative", "Rule 5", "Does a support and helpline post express the author's own crisis?"),
    "komati-5ae5c3543d76": ("other-person crisis; dark humor", "Rule 5", "How should advice to suicidal people with provocative wording be handled?"),
    "irf-7076ccfccc5d": ("historical crisis; ordinary distress", "Rules 4, 6 and 8", "How should long-term depression and past substance use be separated from crisis?"),
    "sdcnl-4f0a4728786d": ("other-person crisis", "Rule 5", "Does describing a sister's suicidal behavior make the author's post hard_negative?"),
    "komati-24bbfb1791ea": ("mixed current/past crisis", "Rules 4 and 6", "Does present last-night language override historical diagnosis and context?"),
    "sdcnl-627cee239252": ("historical suicide/self-harm; mixed temporal context", "Rule 6", "How should past attempts be labelled when suicidal thoughts remain current?"),
    "cams-92e27ba0901a": ("passive death wish; explicit crisis", "Rules 2-4", "Is a stated wish to die with deterrence from harming others explicit crisis?"),
    "komati-0ee2457a06e9": ("passive death wish; implicit crisis", "Rule 2 and Rule 8", "Does being tired of existing without an explicit suicide statement meet implicit crisis?"),
    "irf-579727c6bc55": ("ordinary distress vs implicit crisis", "Rule 8", "How much weight should unbearable pain and social isolation receive?"),
    "cams-f569b3d8ae07": ("ordinary distress vs implicit crisis", "Rule 8", "Does exclusion and feeling worst since medication indicate crisis-level distress?"),
    "cams-727b0d5262d8": ("ordinary distress; historical relationship crisis", "Rules 6 and 8", "Does depression, breakup, and later social recovery remain non_crisis?"),
    "irf-af0c400639da": ("ordinary distress; profanity/intensity", "Rules 1 and 8", "Do failure, loneliness, and strong language exceed ordinary distress?"),
    "irf-52c87b771927": ("euphemistic language; burdensomeness", "Rules 2-4", "Does future-oriented 'off myself' language express personal suicidal intent?"),
    "komati-dc83e85b188c": ("mixed current/past crisis; profanity", "Rules 1, 4 and 6", "How should a recent failed attempt and intense profanity be classified?"),
    "cams-edf86c6a3631": ("mixed current/past crisis; ordinary distress threshold", "Rules 4, 6 and 8", "Does bleak academic context plus self-worth language cross the crisis threshold?"),
    "sdcnl-15809c95cf4b": ("explicit denial; ordinary distress", "Rules 4 and 7", "Does physical and sleep distress count when the author says their mind feels okay?"),
    "irf-22a24aba790f": ("ordinary distress; social anxiety", "Rule 8", "How should severe social anxiety without crisis language be labelled?"),
    "cams-96a1b0c9da4a": ("ordinary distress/loneliness", "Rule 8", "Does social exclusion and loneliness alone support implicit crisis?"),
    "irf-4dc9dfcef82a": ("ordinary distress; depression reference", "Rules 4 and 8", "Does depression terminology plus physical symptoms determine crisis status?"),
    "cams-24583e7bb6a7": ("passive death wish; profanity/intensity", "Rules 1-4", "How should 'do not deserve to live' be distinguished from explicit intent?"),
}

DISAGREEMENTS = {
    "cams-24c0c25e6a33", "cams-687449dcc3c5", "irf-5f27a24f37fa",
    "irf-a11b2a6bb141", "komati-c0f53895b28d", "komati-e335a390c977",
    "sdcnl-9a7998c1beff", "sdcnl-ca2a4047d113",
}

CATEGORY_ORDER = [
    "explicit vs implicit", "implicit vs non-crisis", "hard-negative vs non-crisis",
    "historical crisis", "euphemistic language", "passive death wish", "denial of suicide",
    "profanity", "dark humor", "second-person/general language", "other-person crisis",
    "quoted/reference crisis", "ambiguous/short text", "ordinary distress", "mixed temporal context",
]


def category_tags(reason: str) -> list[str]:
    tags = []
    for category in CATEGORY_ORDER:
        aliases = {
            "explicit vs implicit": ("explicit vs implicit", "implicit vs explicit"),
            "implicit vs non-crisis": ("implicit vs non-crisis", "non-crisis vs implicit"),
            "hard-negative vs non-crisis": ("hard-negative vs non-crisis", "non-crisis vs hard-negative"),
            "historical crisis": ("historical",),
            "euphemistic language": ("euphemistic",),
            "passive death wish": ("passive death wish",),
            "denial of suicide": ("denial",),
            "profanity": ("profanity",),
            "dark humor": ("dark humor", "absurdity"),
            "second-person/general language": ("second-person", "general"),
            "other-person crisis": ("other-person",),
            "quoted/reference crisis": ("quoted", "reference"),
            "ambiguous/short text": ("ambiguous/short",),
            "ordinary distress": ("ordinary distress",),
            "mixed temporal context": ("mixed current/past", "mixed temporal"),
        }
        if any(alias in reason for alias in aliases[category]):
            tags.append(category)
    return tags


def build_sample(root: Path) -> pd.DataFrame:
    a = pd.read_csv(root / "data/gold/annotation/calibration_150_A.csv", encoding="utf-8")
    b = pd.read_csv(root / "data/gold/annotation/calibration_150_B.csv", encoding="utf-8")
    adjudicated = pd.read_csv(root / "data/gold/annotation/agreement/calibration_150_adjudicated.csv", encoding="utf-8")
    merged = a[["id", "text", "label", "confidence"]].rename(columns={"label": "A_label", "confidence": "A_confidence"}).merge(
        b[["id", "label", "confidence"]].rename(columns={"label": "B_label", "confidence": "B_confidence"}), on="id", how="inner"
    ).merge(adjudicated[["id", "final label"]].rename(columns={"final label": "adjudicated_label"}), on="id", how="left")
    if len(merged) != 150 or set(merged["id"]) != set(a["id"]):
        raise ValueError("Calibration sources did not align on all 150 IDs")
    if len(SAMPLE) != 30 or not DISAGREEMENTS.issubset(SAMPLE):
        raise ValueError("Sanity-check sample must contain 30 cases and all disagreements")
    selected = merged[merged["id"].isin(SAMPLE)].copy()
    selected["final_adjudicated_label"] = selected["adjudicated_label"].fillna(selected["A_label"])
    selected["difficulty_reason"] = selected["id"].map(lambda item: SAMPLE[item][0])
    selected["guideline_rule_being_tested"] = selected["id"].map(lambda item: SAMPLE[item][1])
    selected["sanity_check_question"] = selected["id"].map(lambda item: SAMPLE[item][2])
    selected["review_status"] = selected["id"].map(lambda item: "REVIEW REQUIRED - disagreement" if item in DISAGREEMENTS else "REVIEW REQUIRED - boundary example")
    selected["reviewer_notes"] = ""
    output = selected[["id", "text", "A_label", "B_label", "final_adjudicated_label", "A_confidence", "B_confidence", "difficulty_reason", "guideline_rule_being_tested", "sanity_check_question", "review_status", "reviewer_notes"]]
    return output.sort_values("id").reset_index(drop=True)


def make_figures(sample: pd.DataFrame, figure_dir: Path) -> dict[str, Path]:
    figure_dir.mkdir(parents=True, exist_ok=False)
    counts = {category: int(sample["difficulty_reason"].map(lambda reason: category in category_tags(reason)).sum()) for category in CATEGORY_ORDER}
    figure_paths = {}

    fig, ax = plt.subplots(figsize=(12, 7))
    series = pd.Series(counts).sort_values()
    bars = ax.barh(series.index, series.values, color="#356b83")
    for bar, value in zip(bars, series.values):
        ax.text(value + 0.1, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    ax.set_xlabel("Selected cases (a case may count more than once)")
    ax.set_title("Guideline Sanity-Check Coverage by Difficulty Category")
    figure_paths["coverage"] = figure_dir / "difficulty_category_coverage.png"
    fig.savefig(figure_paths["coverage"], dpi=300, bbox_inches="tight"); plt.close(fig)

    labels = list(LABELS)
    distribution = pd.DataFrame({"A": sample["A_label"].value_counts(), "B": sample["B_label"].value_counts(), "Adjudicated": sample["final_adjudicated_label"].value_counts()}).reindex(labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6)); distribution.plot.bar(ax=ax, color=["#176b87", "#e07a3f", "#6b4f8a"])
    ax.set_xlabel("Label"); ax.set_ylabel("Selected cases"); ax.set_title("A vs B vs Adjudicated Labels in Sanity-Check Sample"); ax.legend(title="Source"); ax.tick_params(axis="x", rotation=20)
    figure_paths["distribution"] = figure_dir / "label_distribution_A_B_adjudicated.png"
    fig.savefig(figure_paths["distribution"], dpi=300, bbox_inches="tight"); plt.close(fig)

    status = pd.Series({"A/B agreement": int((sample["A_label"] == sample["B_label"]).sum()), "A/B disagreement": int((sample["A_label"] != sample["B_label"]).sum())})
    fig, ax = plt.subplots(figsize=(7, 5)); bars = ax.bar(status.index, status.values, color=["#2a9d8f", "#d1495b"])
    for bar, value in zip(bars, status.values): ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, str(value), ha="center")
    ax.set_ylabel("Selected cases"); ax.set_title("Agreement Status in Sanity-Check Sample")
    figure_paths["status"] = figure_dir / "agreement_status.png"
    fig.savefig(figure_paths["status"], dpi=300, bbox_inches="tight"); plt.close(fig)

    primary = sample["difficulty_reason"].str.split(";").str[0].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6)); bars = ax.barh(primary.index, primary.values, color="#c96b4b")
    for bar, value in zip(bars, primary.values): ax.text(value + 0.1, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    ax.set_xlabel("Selected cases"); ax.set_title("Primary Boundary-Type Distribution")
    figure_paths["boundary"] = figure_dir / "boundary_type_distribution.png"
    fig.savefig(figure_paths["boundary"], dpi=300, bbox_inches="tight"); plt.close(fig)
    return figure_paths


def build_report(sample: pd.DataFrame, figures: dict[str, Path], output_dir: Path) -> str:
    category_counts = {category: int(sample["difficulty_reason"].map(lambda reason: category in category_tags(reason)).sum()) for category in CATEGORY_ORDER}
    coverage = "\n".join(f"| {category} | {category_counts[category]} |" for category in CATEGORY_ORDER)
    cases = []
    for row in sample.itertuples(index=False):
        agreement = "A/B agreement" if row.A_label == row.B_label else "A/B disagreement"
        clarity = "MINOR CLARIFICATION NEEDED" if row.id in {"cams-687449dcc3c5", "irf-5f27a24f37fa", "irf-a11b2a6bb141", "komati-e335a390c977", "sdcnl-9a7998c1beff", "sdcnl-ca2a4047d113", "sdcnl-a16d8fd4e632", "komati-dea6b540f261", "komati-24bbfb1791ea", "sdcnl-627cee239252", "cams-24583e7bb6a7"} else "CLEAR"
        ambiguity = "Review whether the rule gives enough operational evidence for this boundary." if clarity != "CLEAR" else "No additional ambiguity beyond the case-specific question was observed."
        suggestion = "Add one or two positive/negative examples if reviewers reach different decisions." if clarity != "CLEAR" else "No change proposed."
        cases.append(f"### `{row.id}`\n\n- A label: `{row.A_label}`\n- B label: `{row.B_label}`\n- Adjudicated/final label: `{row.final_adjudicated_label}`\n- Why difficult: {row.difficulty_reason}\n- v2 rule: {row.guideline_rule_being_tested}\n- Sufficiency assessment: **{clarity}**\n- Potential ambiguity: {ambiguity}\n- Suggested clarification, if necessary: {suggestion}\n- Sanity-check question: {row.sanity_check_question}\n")
    figure_lines = "\n".join(f"### {index}. `{path.name}`\n\n`figures/sanity_check/{path.name}`\n" for index, path in enumerate(figures.values(), 1))
    return f"""# Guideline v2 Sanity Check

## 1. Purpose

This review tests whether `annotation_guidelines_v2.md` is sufficiently clear and reproducible before it is frozen for full gold-dataset annotation. It asks whether two reasonable annotators could still assign different labels to difficult cases. This is a qualitative boundary-case review, not a new adjudication or a statistical estimate of future agreement.

## 2. Source Data

The sample was built from:

- `data/gold/annotation/annotation_guidelines_v2.md`
- `data/gold/annotation/calibration_150_A.csv`
- `data/gold/annotation/calibration_150_B.csv`
- `data/gold/annotation/agreement/calibration_150_adjudicated.csv`
- `data/gold/annotation/agreement/calibration_150_disagreements.csv`
- `data/gold/annotation/agreement/calibration_150_adjudication_report.md`
- `scripts/annotation/validate_guideline_sanity_check.py`

## 3. Selection Method

    The sample contains {len(sample)} cases: all 8 previously adjudicated disagreements plus {len(sample) - int((sample['A_label'] != sample['B_label']).sum())} agreed calibration cases selected for observed boundary signals, medium/low confidence, or direct coverage of v2 rules. Selection was based on the actual text and labels, including explicit and euphemistic suicide language, temporal markers, denials, third-party references, second-person language, ordinary distress, absurdity, and short/vague posts. The sample is not statistically representative and must not be used as an agreement estimate.

## 4. Coverage of Difficult Cases

Counts are generated from the sample metadata. A case may belong to multiple categories.

| Category | Selected cases |
| --- | ---: |
{coverage}

## 5. Case-by-Case Review

The CSV contains the original text, A/B labels and confidence, final/adjudicated label, selection reason, tested rule, and a review question for every case.

{chr(10).join(cases)}

## 6. Recurring Ambiguities

Observed ambiguities are concentrated in:

- temporal scope: whether past suicidal behavior or past severe distress should be labelled when current crisis is absent or unclear;
- the implicit-crisis threshold for persistent self-loathing, loneliness, isolation, or unbearable distress;
- whether euphemisms such as `off myself` and `end it all` are literal, hyperbolic, or joking;
- second-person or generalized descriptions that may represent lived experience;
- contextual interpretation of dark or absurd statements containing crisis terminology;
- distinguishing hard negatives from non-crisis when crisis language appears in support, community, or third-party references; and
- very short statements where the available evidence is insufficient.

These are observed review signals, not reasons to optimize labels for higher agreement.

## 7. Guideline Stability Assessment

| Boundary | Assessment | Reason |
| --- | --- | --- |
| Explicit vs implicit crisis | MINOR CLARIFICATION NEEDED | Euphemisms and self-referential wording can still create disagreement about explicitness. |
| Implicit vs non-crisis | MINOR CLARIFICATION NEEDED | The threshold for severe distress versus ordinary loneliness remains judgment-sensitive. |
| Hard-negative vs non-crisis | CLEAR | Third-party, support, and community-reference examples are operationally distinct, though context must be read. |
| Euphemisms and profanity | MINOR CLARIFICATION NEEDED | v2 states the principle, but more paired examples would improve reproducibility. |
| Second-person/general language | MINOR CLARIFICATION NEEDED | Lived-experience inference remains contextual. |
| Short/vague statements | CLEAR | Rule 7 gives a conservative insufficient-evidence direction. |
| Temporal scope | MINOR CLARIFICATION NEEDED | Rule 6 identifies the issue but does not select one explicit research policy for all historical cases. |

## 8. Temporal Scope Decision

The word `current` is important in v2, but the guideline does not fully specify a single operational treatment for historical suicide attempts, past self-harm, past suicidal thoughts, recovery narratives, or descriptions of previous crisis with an explicit denial of current crisis. It says to determine whether the event is historical, current, or the subject of the label, but leaves the final policy open.

**REQUIRES RESEARCH/ADVISOR DECISION:** decide whether the project labels the historical crisis itself, current language only, or a defined combination. That decision should be recorded before full annotation. No silent policy is introduced by this report.

## 9. Proposed Clarifications

Do not edit v2 automatically. Before freezing, consider adding a short temporal decision table stating how to label: current crisis; historical crisis with current residual symptoms; recovery narratives; historical crisis with explicit current denial; and retrospective discussion with no current symptoms. Also consider paired examples for euphemism versus hyperbole and severe loneliness versus ordinary loneliness. These proposals are for review and are not applied here.

## 10. Final Recommendation

Recommendation: **FREEZE AFTER MINOR CLARIFICATION**

The four-label model and most boundaries are usable, and the sample does not indicate a need for major conceptual revision. However, v2 should not be frozen until the temporal policy is chosen by the research team/advisor and a few paired examples are added for the observed euphemism, implicit-threshold, and second-person ambiguities. This recommendation is about reproducibility, not maximizing agreement.

## 11. Figures

All figures were generated with matplotlib at 300 DPI from the selected CSV values.

{figure_lines}

## 12. Traceability and Validation

- Selected cases: {len(sample)}
- A/B disagreements included: {int((sample['A_label'] != sample['B_label']).sum())}
- A/B agreements included: {int((sample['A_label'] == sample['B_label']).sum())}
- All 8 required disagreement IDs are included.
- Original text and source annotations were not modified.
- No labels were applied to the remaining gold candidates.
- This artifact is a review sample, not a replacement guideline or gold dataset.

Generation command:

```powershell
python scripts/annotation/validate_guideline_sanity_check.py
```
"""


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "data/gold/annotation/agreement"
    sample_path = output_dir / "guideline_sanity_check_20_30.csv"
    report_path = output_dir / "guideline_sanity_check_report.md"
    figure_dir = output_dir / "figures/sanity_check"
    if sample_path.exists() or report_path.exists() or figure_dir.exists():
        raise FileExistsError("Sanity-check output already exists; refusing to overwrite it")
    sample = build_sample(root)
    sample.to_csv(sample_path, index=False, encoding="utf-8")
    figures = make_figures(sample, figure_dir)
    report_path.write_text(build_report(sample, figures, output_dir), encoding="utf-8")
    print(f"Created {len(sample)}-case sample with {(sample['A_label'] != sample['B_label']).sum()} disagreements")
    print(f"Final labels: {sample['final_adjudicated_label'].value_counts().reindex(LABELS, fill_value=0).to_dict()}")
    print(f"CSV: {sample_path}")
    print(f"Report: {report_path}")
    print(f"Figures: {len(figures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())