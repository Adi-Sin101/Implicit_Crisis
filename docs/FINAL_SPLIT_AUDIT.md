# Final Split Audit — TF-IDF + Logistic Regression

Audit of the frozen dataset split and the frozen final TF-IDF model configuration.

Every figure below was produced by `scripts/data/export_official_splits.py`, which
reads the official split and verifies the exported files against it. Nothing here
was typed by hand or estimated. Reproduce with:

```bash
python scripts/data/export_official_splits.py
```

**Audit date:** 2026-09-19

---

## Source Split

```
experiments/tfidf_baseline/results/data_split.csv
```

This is the authoritative split. It was not regenerated, reshuffled, resampled or
re-partitioned during this audit.

## Exported Files

```
data/final_datasets/splits/train.csv
data/final_datasets/splits/validation.csv
data/final_datasets/splits/test.csv
```

None of the three existed before this audit, so all three were **created** directly
from the source by filtering its `split` column. Rows were copied unchanged and in
source order.

> **Relocation note.** These files were originally exported to `data/splits/` and
> were later moved to `data/final_datasets/splits/`. That was a path change only:
> all four files (including this directory's `README.md`) were verified
> byte-for-byte identical before and after the move by SHA-256, the old
> `data/splits/` directory was removed so no duplicate copy remains, and every
> figure in this audit was re-verified against the source at the new location. No
> data, label, membership or ordering changed.

> **Naming note.** The source `split` column uses the value **`val`**, not
> `validation`. The exported file is named `validation.csv` as required by the
> task, and it contains exactly the rows whose source split value is `val`. No
> value was renamed inside the data.

## Row Counts

| Split | Rows | Expected | Status |
|---|---|---|---|
| Train | 905 | 905 | OK |
| Validation | 194 | 194 | OK |
| Test | 195 | 195 | OK |
| **Total** | **1294** | **1294** | **OK** |

The three exported files reconstruct the source row set exactly (1,294 rows, same
`row_index` set, no row added or lost).

## Columns

The source and all three exported files carry the same seven columns:

| Column | Type | Role |
|---|---|---|
| `row_index` | int64 | Unique row identifier, 0–1293 — used as the identity key |
| `url` | str | Reddit permalink (metadata, not a model input) |
| `content` | str | **The classification text column** |
| `severity` | int64 | Original label source, values 0–6 |
| `label` | int64 | Collapsed 3-class label, values 0/1/2 |
| `class_name` | str | Human-readable class name |
| `split` | str | `train` / `val` / `test` |

Missing values: **zero in every column**, in the source and in all exported files.

## Class Mapping

```
severity 0     → 0 = Non-Crisis
severity 1     → 1 = Implicit Crisis
severity 2–6   → 2 = Explicit Crisis
```

Verified against the actual data, for every severity value present:

| severity | → label | Class | Rows |
|---|---|---|---|
| 0 | 0 | Non-Crisis | 351 |
| 1 | 1 | Implicit Crisis | 324 |
| 2 | 2 | Explicit Crisis | 302 |
| 3 | 2 | Explicit Crisis | 72 |
| 4 | 2 | Explicit Crisis | 37 |
| 5 | 2 | Explicit Crisis | 50 |
| 6 | 2 | Explicit Crisis | 158 |

Every severity value maps to exactly one label, and `class_name` agrees with
`label` on every row. The data contains **exactly three classes**. There is no
fourth class and **no `hard_negative` class** — none exists in this dataset and
none was introduced.

**LABEL MAPPING: PASS**

## Class Distributions

| Split | Non-Crisis | Implicit Crisis | Explicit Crisis | Total |
|---|---|---|---|---|
| Train | 245 | 227 | 433 | 905 |
| Validation | 53 | 48 | 93 | 194 |
| Test | 53 | 49 | 93 | 195 |
| **All** | **351** | **324** | **619** | **1294** |

Raw severity counts per split:

| Split | sev 0 | sev 1 | sev 2 | sev 3 | sev 4 | sev 5 | sev 6 |
|---|---|---|---|---|---|---|---|
| Train | 245 | 227 | 211 | 53 | 31 | 31 | 107 |
| Validation | 53 | 48 | 50 | 13 | 1 | 7 | 22 |
| Test | 53 | 49 | 41 | 6 | 5 | 12 | 29 |

The split was stratified on the **3-class** label, not on raw severity, so the
three-class shares track closely across splits while the raw severity counts
within Explicit Crisis vary (for example severity 4: 31 train / 1 validation /
5 test). This is expected and is not a defect.

## Text Column

The classification text column `content` is present in the source and in all three
exported files. Content was compared per `row_index` by SHA-256 digest between the
source and each exported file: **identical in every row**. No cleaning,
normalisation, rewriting or preprocessing was applied. These are split files only.

**TEXT COLUMN: PASS**

## Sample Identity

Each exported file was verified against the rows the official source assigns to
that split. Identity was checked on three axes: the `row_index` set, the
per-row `content` digest, and the per-row `severity` and `label` values.

```
TRAIN IDENTITY: PASS
VALIDATION IDENTITY: PASS
TEST IDENTITY: PASS
```

| Split | Rows | Same IDs | Same content | Same labels | Verdict |
|---|---|---|---|---|---|
| Train | 905 | yes | yes | yes | PASS |
| Validation | 194 | yes | yes | yes | PASS |
| Test | 195 | yes | yes | yes | PASS |

## Duplicate Checks

In the source:

| Check | Count |
|---|---|
| Duplicate full rows | 0 |
| Duplicate `content` | 0 |
| Duplicate `row_index` | 0 |

Within each exported file: **0 duplicate rows and 0 duplicate content values**.

Cross-split overlap, checked both in the source and between the exported files, on
both `row_index` and `content`:

| Pair | row_index overlap | content overlap |
|---|---|---|
| Train ∩ Validation | 0 | 0 |
| Train ∩ Test | 0 | 0 |
| Validation ∩ Test | 0 | 0 |

```
OVERLAP: 0
DUPLICATES: 0
```

No duplicates were removed, because none exist.

## Split Integrity

```
The split was NOT regenerated.
The three split files were derived directly from the existing official data_split.csv.
```

`scripts/data/export_official_splits.py` does not import `train_test_split` and
contains no sampling, shuffling or membership-assignment code. It filters the
source's existing `split` column and writes the resulting rows. If the three files
already exist on a later run, they are verified against the source rather than
overwritten, and any difference is reported instead of silently corrected.

## Final TF-IDF Configuration

The complete frozen configuration, also recorded machine-readably in
`configs/tfidf_final_frozen.json`:

```text
TF-IDF:
  analyzer      = char_wb
  ngram_range   = (3,5)
  min_df        = 2
  max_df        = 0.9
  sublinear_tf  = true
  max_features  = 50000
  lowercase     = true
  strip_accents = unicode
  fitted on     = training split only (17,708 features)

Logistic Regression:
  C             = 1.0
  class_weight  = balanced
  solver        = lbfgs
  max_iter      = 5000
  random_state  = 42

Model selection metric:
  validation macro F1
```

**Analyzer: `char_wb`**
**N-gram range: `(3,5)`**

Every value above was cross-checked against
`models/tfidf_logistic_regression/config.json` and matches it exactly.

## Final TF-IDF Result

```text
Validation Macro F1: 0.8054
Test Accuracy:       0.7128
Test Macro Precision: 0.7007
Test Macro Recall:   0.6979
Test Macro F1:       0.6962
Test Weighted F1:    0.7117
```

Cross-checked against the on-disk evidence and matching exactly:

- `results/ngram_ablation/results.csv` — the `(3,5)` row gives validation macro F1
  0.8054, and it is the argmax across the six tested ranges.
- `results/ngram_ablation/final_test_results.csv` — selected range `(3,5)`, and all
  five test figures above.
- `models/tfidf_logistic_regression/config.json` — the same test metrics.

## Selection Evidence

The n-gram ablation tested six ranges under `analyzer=char_wb`, with every other
TF-IDF and Logistic Regression parameter held fixed, scoring on validation only:

| n-gram | Validation Macro F1 |
|---|---|
| **`(3,5)`** | **0.8054** ← selected |
| `(2,4)` | 0.7986 |
| `(4,6)` | 0.7981 |
| `(3,6)` | 0.7944 |
| `(4,7)` | 0.7902 |
| `(5,7)` | 0.7862 |

`(3,5)` had the highest validation macro F1; no alternative beat it. The full
evidence remains in `results/ngram_ablation/` (results table, report, run log,
config and final-test files) and must be preserved as the justification for this
selection.

## Frozen Status

```text
The (3,5) TF-IDF configuration is frozen as the final TF-IDF model configuration.

No further n-gram search or TF-IDF model-selection experiment should be performed
unless explicitly requested.
```

The official `data_split.csv` is the frozen dataset split. The three files under
`data/final_datasets/splits/` are explicit copies of that split and carry no independent
authority — if they ever disagree with the source, the source wins.

During this audit: no model was retrained, no vectorizer or model artifact was
replaced, no test metric was recomputed, no hyperparameter search was rerun, and
the n-gram ablation was not repeated. The BERT pipeline was not inspected,
modified, configured or run.

## Problems Discovered

```text
No split-integrity problems were discovered.
```

One naming mismatch is worth recording, though it is a difference between the task
wording and the data rather than a defect: the source `split` column uses `val`,
while the task described the value as `validation`. The export filters on the
actual value `val` and writes it to the required filename `validation.csv`. The
data itself was not altered.

## Verification Summary

```text
Source rows = 1294

Train = 905
Validation = 194
Test = 195

Train identity = PASS
Validation identity = PASS
Test identity = PASS

Overlap = 0
Duplicates = 0

Label mapping = PASS
Text column = PASS

Split regenerated = NO

Final TF-IDF ngram = (3,5)
Final TF-IDF configuration = FROZEN

TF-IDF model modified = NO
TF-IDF results modified = NO
BERT modified = NO
```
