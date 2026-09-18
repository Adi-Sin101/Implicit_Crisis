# TF-IDF + Logistic Regression Baseline

Sparse-feature baseline for three-class crisis severity classification.

## Purpose

Establish a reproducible, leakage-free classical baseline that the BERT
experiment in `ipnyb/bert_crisis_classifier (1).ipynb` can be compared against
directly. Comparability is the point: dataset, input column, label mapping,
split, seed and selection metric are all copied from that notebook rather than
chosen independently. The BERT notebook is read-only reference here — nothing
in this experiment trains, modifies or overwrites a BERT model.

## Dataset

```
data/final_datasets/merged_severity_dataset_not_keyword_keyword.csv
```

1,294 rows, 10 columns. No missing `content` or `severity`, no duplicate
content, so no rows are dropped.

## Input column

`content` — the only model input. Nothing is concatenated to it.

Excluded from features (labels, annotations, metadata):
`severity`, `gpt_label`, `claude_label`, `gemini_label`, `llama_label`,
`mistral_label`, `url`, `author`, `created`.

## Label mapping

```
severity 0     ->  0 = Non-Crisis        (351)
severity 1     ->  1 = Implicit Crisis   (324)
severity 2-6   ->  2 = Explicit Crisis   (619)
```

## Split

Stratified 70 / 15 / 15, `random_state = 42`, via the same two-stage
`train_test_split` the BERT notebook uses.

| Split | n | Non-Crisis | Implicit Crisis | Explicit Crisis |
|---|---|---|---|---|
| Train | 905 | 245 | 227 | 433 |
| Validation | 194 | 53 | 48 | 93 |
| Test | 195 | 53 | 49 | 93 |

The TF-IDF vectorizer is fitted on the **training texts only**; validation and
test are transform-only. The test set is evaluated once, after the
configuration is frozen on validation macro F1.

## How to run

From the repository root:

```bash
# dataset audit only
python experiments/tfidf_baseline/audit_dataset.py

# full experiment (audit -> split -> search -> freeze -> test -> figures -> artifacts)
python experiments/tfidf_baseline/run_tfidf_baseline.py

# verification suite (25 tests)
python -m pytest tests/test_tfidf_baseline.py -v
```

Requires `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib`.

## Headline results

Frozen model on the held-out test set (n = 195):

| Metric | Value |
|---|---|
| Accuracy | 0.7128 |
| Macro F1 | **0.6962** |
| Weighted F1 | 0.7117 |
| Implicit Crisis P / R / F1 | 0.6316 / 0.7347 / **0.6792** |

Selected configuration: TF-IDF `char_wb` (3,5), `min_df=2`, `max_df=0.9`,
`sublinear_tf=True`, `max_features=50000` (17,708 features fitted) with
LogisticRegression `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`,
`max_iter=5000`. Validation macro F1 0.8054.

Note the validation-to-test gap (0.8054 → 0.6962). The test figure is the
honest one; see the Limitations section of the full write-up.

## Output files

### `results/`

| File | Contents |
|---|---|
| `dataset_audit.csv` | Row/column counts, missing values, duplicates, length stats |
| `missing_values_per_column.csv` | Missing count per column |
| `text_statistics.csv` | Character and word length distribution |
| `class_distribution.csv` | Three-class counts and proportions |
| `duplicate_report.csv` | Duplicate rows, duplicate content, rows removed/remaining |
| `data_split.csv` | Per-row `split` assignment — reuse this for the BERT comparison |
| `split_distribution.csv` | Class counts per split |
| `validation_results.csv` | All 19 configurations with validation metrics |
| `test_results.csv` | Overall test metrics |
| `classification_report.csv` | Per-class precision / recall / F1 / support |
| `implicit_crisis_metrics.csv` | Implicit Crisis metrics and hit/miss/false-alarm counts |
| `confusion_matrix.csv` | Raw counts, rows = actual |
| `normalized_confusion_matrix.csv` | Row-normalised proportions |
| `top_features.csv` | Top 20 features per class by coefficient |
| `model_comparison_ready.csv` | One row, TF-IDF only — no BERT numbers |

### `figures/`

`class_distribution.png`, `text_length_distribution.png`,
`validation_hyperparameters.png`, `per_class_metrics.png`,
`confusion_matrix.png`, `normalized_confusion_matrix.png`, `top_features.png`

All figures come from the executed run. There are no training-loss or
epoch curves — TF-IDF with Logistic Regression has no such training trajectory.

## Model location

```
models/tfidf_logistic_regression/
├── tfidf_vectorizer.joblib
├── logistic_regression.joblib
└── config.json
```

No label encoder is saved: labels are already integers 0/1/2, so there is
nothing to encode. `config.json` records the dataset, mapping, split, exact
hyperparameters and test metrics, and is duplicated at `config.json` in this
directory.

Both artifacts are reloaded at the end of every run and checked to reproduce
the test feature matrix, the test predictions and valid `predict_proba` output.

## Results location

- Machine-readable: `experiments/tfidf_baseline/results/`
- Figures: `experiments/tfidf_baseline/figures/`
- Full write-up: [`docs/TFIDF_BASELINE.md`](../../docs/TFIDF_BASELINE.md)
- Tests: `tests/test_tfidf_baseline.py`
