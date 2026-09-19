# Final Test Evaluation — N-gram Ablation

> **Selection happened first.** The configuration below was chosen using **validation** macro F1 across the six n-gram ranges, as recorded in `results.csv` and `RESULTS.md`. Only after that selection was the held-out test set used, once, for this single configuration. No test-set metric influenced the choice of n-gram range, and the other five configurations were never evaluated on test.

- **Run (UTC):** 2026-09-18T20:43:46.513484+00:00
- **Git commit:** `23e97044752bb1b71a68820b94e00ba1865a7705`
- **Selected configuration:** `char_wb` `(3,5)`, 17,708 features
- **Selected on:** validation macro F1 = 0.8054
- **Test set size:** 195

## Overall test metrics

| Metric | Value |
|---|---|
| Accuracy | 0.7128 |
| Macro Precision | 0.7007 |
| Macro Recall | 0.6979 |
| **Macro F1** | 0.6962 |
| Weighted Precision | 0.7155 |
| Weighted Recall | 0.7128 |
| Weighted F1 | 0.7117 |

## Per-class test metrics

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-Crisis | 0.7045 | 0.5849 | 0.6392 | 53 |
| Implicit Crisis | 0.6316 | 0.7347 | 0.6792 | 49 |
| Explicit Crisis | 0.7660 | 0.7742 | 0.7701 | 93 |

Hard Negative metrics are not reported: the class does not exist in this dataset. See `RESULTS.md` §6.

## Confusion matrix

Rows = actual, columns = predicted.

| | Non-Crisis | Implicit Crisis | Explicit Crisis |
|---|---|---|---|
| **Non-Crisis** | 31 | 9 | 13 |
| **Implicit Crisis** | 4 | 36 | 9 |
| **Explicit Crisis** | 9 | 12 | 72 |

## Validation vs test for the selected configuration

| Metric | Validation | Test | Δ |
|---|---|---|---|
| Macro F1 | 0.8054 | 0.6962 | -0.1092 |
| Macro Precision | 0.8090 | 0.7007 | -0.1083 |
| Macro Recall | 0.8114 | 0.6979 | -0.1135 |
| Accuracy | 0.8093 | 0.7128 | -0.0965 |

A drop from validation to test is expected: the configuration was selected on validation, which biases that estimate upward. The test column is the unbiased figure.

## Reproduction

```bash
python experiments/ngram_ablation.py
```