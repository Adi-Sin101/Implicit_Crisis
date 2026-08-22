# Experiments

This directory holds **measured results only**. Nothing here is filled in ahead
of an actual run, and no expected or illustrative figure appears anywhere in
this repository.

| Subdirectory | Written by | Content |
|---|---|---|
| `baselines/tfidf/` | `scripts/experiments/run_tfidf_baseline.py` | overall.json, per_class.csv, slices.csv, confusion_matrix.csv, predictions |
| `bert/` | `scripts/experiments/run_bert.py` | the same set for the fine-tuned transformer |
| `comparisons/` | `scripts/experiments/evaluate_models.py` | model_comparison.csv |
| `error_analysis/` | `scripts/experiments/evaluate_models.py` | disagreement buckets, missed implicit cases, hard-negative false positives |

Metric tables are small and are committed. Model checkpoints and raw prediction
dumps are gitignored (`experiments/**/checkpoints/`, `experiments/**/predictions/`),
since predictions contain verbatim corpus text.

The headline table is per-slice, not pooled:

| Model | Explicit | Implicit | Hard negative | Overall |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | not yet run | not yet run | not yet run | not yet run |
| BERT (fine-tuned) | not yet run | not yet run | not yet run | not yet run |
