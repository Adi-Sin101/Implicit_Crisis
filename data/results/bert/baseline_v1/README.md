# BERT Baseline v1 — frozen

These files are the archived record of the first (CPU) BERT fine-tuning run.
**Do not modify, overwrite or regenerate them.** A later BERT experiment must be
written to a new directory (e.g. `data/results/bert/v2_.../`) and compared
against this one.

- `baseline_v1_metadata.json` — the complete self-contained record: dataset,
  splits (with SHA-256), label mapping, hyperparameters, hardware, software
  versions, training history, test results, per-class results, confusion matrix,
  implicit-crisis slice, error breakdown, limitations.
- `bert_config.json`, `bert_training_history.json`, `bert_test_metrics.json`,
  `bert_classification_report.json`, `bert_confusion_matrix.csv`,
  `bert_test_predictions.csv`, `bert_errors.csv`, `bert_training_log.txt` —
  byte-identical copies of the run's own outputs.

The checkpoint lives in `models/bert/baseline_v1/best_model/` (gitignored).

Verify the archive: `python scripts/models/archive_bert_baseline_v1.py --verify-only`

Documentation: `docs/methodology/bert_baseline_v1.md`

Note on labels: the underlying gold labels are AI-assisted annotations, not
human-validated gold labels.
