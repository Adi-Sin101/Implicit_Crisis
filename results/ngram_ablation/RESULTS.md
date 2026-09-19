# Character N-gram Ablation — Results

All numbers in this report were produced by `experiments/ngram_ablation.py` and read directly from `results.csv`. Nothing was typed by hand.

- **Run (UTC):** 2026-09-18T20:43:46.513484+00:00
- **Git commit:** `23e97044752bb1b71a68820b94e00ba1865a7705` (branch `main`)
- **Working tree dirty at run time:** True
- **scikit-learn:** 1.9.0  **Python:** 3.13.7

## 1. Objective

Determine whether changing the character n-gram range of the TF-IDF vectorizer improves the crisis-severity classifier, with the n-gram range as the *only* variable. Specifically: does widening or shifting the range beyond the current `(3,5)` baseline help?

## 2. Dataset and split

- **Dataset:** `data/final_datasets/merged_severity_dataset_not_keyword_keyword.csv`
- **Input column:** `content` (only input)
- **Labels:** from `severity`, 0 → Non-Crisis, 1 → Implicit Crisis, 2–6 → Explicit Crisis
- **Split source:** `experiments/tfidf_baseline/results/data_split.csv` — the existing split was *loaded*, not recomputed.
- **Sizes:** train 905, validation 194, test 195
- **Seed:** 42

Split disjointness in text space was asserted at load time. For every configuration the vectorizer was fitted on training texts only; validation and test were transform-only.

## 3. Baseline configuration

`char_wb` `(3,5)` — validation macro F1 **0.8054**, 17,708 features.

This reproduces the previously recorded baseline value of 0.8054, which confirms the ablation harness matches the original experiment.

## 4. Configurations tested

Six, differing only in `ngram_range`:

`(2,4)`  `(3,5)` ← baseline  `(3,6)`  `(4,6)`  `(4,7)`  `(5,7)`

## 5. Exact parameters (held fixed)

```python
TfidfVectorizer(
    analyzer='char_wb',
    ngram_range=<THE ONLY VARIABLE>,
    min_df=2, max_df=0.9,
    sublinear_tf=True, max_features=50000,
    lowercase=True, strip_accents='unicode',
)
LogisticRegression(
    C=1.0, class_weight='balanced',
    solver='lbfgs', max_iter=5000, random_state=42,
)
```

## 6. Complete results (from `results.csv`)

Sorted by validation macro F1, descending. All values measured.

| n-gram | Features | Val Acc | Val Macro P | Val Macro R | **Val Macro F1** | Implicit F1 | Implicit R | Explicit F1 | Explicit R | Non-Crisis P | Vec s | Fit s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `(3,5)` ←base | 17,708 | 0.8093 | 0.8090 | 0.8114 | **0.8054** | 0.7593 | 0.8542 | 0.8287 | 0.8065 | 0.8913 | 0.81 | 0.40 |
| `(2,4)` | 11,015 | 0.8041 | 0.7984 | 0.8091 | 0.7986 | 0.7890 | 0.8958 | 0.8268 | 0.7957 | 0.8298 | 0.94 | 0.28 |
| `(4,6)` | 20,899 | 0.8041 | 0.8092 | 0.7997 | 0.7981 | 0.7664 | 0.8542 | 0.8280 | 0.8280 | 0.9048 | 0.79 | 0.52 |
| `(3,6)` | 24,359 | 0.7990 | 0.8010 | 0.7988 | 0.7944 | 0.7593 | 0.8542 | 0.8197 | 0.8065 | 0.8864 | 1.02 | 0.50 |
| `(4,7)` | 25,894 | 0.7990 | 0.8004 | 0.7934 | 0.7902 | 0.7593 | 0.8542 | 0.8324 | 0.8280 | 0.8810 | 0.81 | 0.55 |
| `(5,7)` | 19,275 | 0.7938 | 0.7866 | 0.7965 | 0.7862 | 0.7890 | 0.8958 | 0.8222 | 0.7957 | 0.8043 | 0.47 | 0.36 |

### Hard Negative metrics — NOT AVAILABLE

The final dataset (merged_severity_dataset_not_keyword_keyword.csv) carries only the `severity` column, which maps to the three classes Non-Crisis / Implicit Crisis / Explicit Crisis. It contains no `hard_negative` annotation. `hard_negative` is a class in the project's older four-class scheme (annotation/guidelines/annotation_guidelines.md: crisis vocabulary present but not the author's own crisis), which the current three-class scheme does not carry. Hard-negative metrics are therefore NOT COMPUTABLE on this dataset and are recorded as NA. Non-Crisis metrics are reported in full and are the closest available quantity, but Non-Crisis is NOT the same class as hard_negative and must not be read as a substitute.

The `hard_negative_f1`, `hard_negative_precision` and `hard_negative_recall` columns in `results.csv` are therefore `NOT_APPLICABLE` with `hard_negative_status = class_not_in_dataset`. (The sentinel is not the string `NA`, which pandas would silently read back as `NaN` and make a non-existent class look like a failed computation.) Non-Crisis precision is shown above as the nearest *available* quantity, clearly labelled as Non-Crisis. It is a different class and is not a stand-in.

## 7. Best validation configuration

```text
Baseline:
char_wb (3,5)
Validation Macro F1 = 0.8054

Best tested configuration:
char_wb (3,5)
Validation Macro F1 = 0.8054

Difference:
+0.0000
```

## 8. Comparison against `(3,5)`

| Metric | `(3,5)` baseline | Best `(3,5)` | Δ |
|---|---|---|---|
| Validation Macro F1 | 0.8054 | 0.8054 | +0.0000 |
| Validation Macro Recall | 0.8114 | 0.8114 | +0.0000 |
| Validation Macro Precision | 0.8090 | 0.8090 | +0.0000 |
| Validation Accuracy | 0.8093 | 0.8093 | +0.0000 |
| Implicit Crisis F1 | 0.7593 | 0.7593 | +0.0000 |
| Implicit Crisis Recall | 0.8542 | 0.8542 | +0.0000 |
| Feature count | 17,708 | 17,708 | +0 |

## 9. Implicit Crisis performance across all configurations

The research focus class, every configuration, measured on validation:

| n-gram | Implicit F1 | Implicit Recall | Implicit Precision |
|---|---|---|---|
| `(2,4)` | 0.7890 | 0.8958 | 0.7049 |
| `(5,7)` | 0.7890 | 0.8958 | 0.7049 |
| `(4,6)` | 0.7664 | 0.8542 | 0.6949 |
| `(3,5)` ←base | 0.7593 | 0.8542 | 0.6833 |
| `(3,6)` | 0.7593 | 0.8542 | 0.6833 |
| `(4,7)` | 0.7593 | 0.8542 | 0.6833 |

Highest Implicit Crisis F1: `(2,4)` at 0.7890. This is **not** the macro-F1 winner (`(3,5)`, implicit F1 0.7593). Selection followed the pre-registered rule, macro F1.

## 10. Feature counts

| n-gram | Features | vs baseline |
|---|---|---|
| `(2,4)` | 11,015 | -6,693 |
| `(3,5)` | 17,708 | +0 |
| `(3,6)` | 24,359 | +6,651 |
| `(4,6)` | 20,899 | +3,191 |
| `(4,7)` | 25,894 | +8,186 |
| `(5,7)` | 19,275 | +1,567 |

Note: `max_features=50,000` caps the vocabulary. Any configuration reporting exactly that many features is truncated, so its feature count is set by the cap rather than by the n-gram range.

## 11. Training time

Wall-clock, single run, same machine, no repeats — indicative only, not a benchmark.

| n-gram | Vectorize (s) | LR fit (s) | Total (s) | LR iters |
|---|---|---|---|---|
| `(2,4)` | 0.94 | 0.28 | 1.23 | 18 |
| `(3,5)` | 0.81 | 0.40 | 1.21 | 20 |
| `(3,6)` | 1.02 | 0.50 | 1.52 | 22 |
| `(4,6)` | 0.79 | 0.52 | 1.30 | 27 |
| `(4,7)` | 0.81 | 0.55 | 1.35 | 25 |
| `(5,7)` | 0.47 | 0.36 | 0.83 | 23 |

## 12. Interpretation

- No tested configuration beat the `(3,5)` baseline on validation macro F1; the baseline remains the best.
- Configurations starting at n>=4 (`(4,6)`, `(4,7)`, `(5,7)`) scored between 0.7862 and 0.7981 validation macro F1, versus 0.8054 for the baseline.
- Feature counts ranged from 11,015 to 25,894; the correlation between feature count and validation macro F1 across the six points is -0.430.
- Implicit Crisis F1 ranged 0.7593 to 0.7890 and Implicit Crisis recall 0.8542 to 0.8958 across the six configurations.
- Total fit time ranged 0.83s to 1.52s.

## 13. Limitations

- **Single validation split, n = 194.** One validation post is worth roughly 0.5 accuracy points, so small differences between configurations are within noise. No cross-validation and no repeated seeds were run, so these differences carry no confidence interval.
- **Selection on a small validation set biases the winner's score upward.** The validation figure for the selected configuration is an optimistic estimate of its true performance; the test figure is not.
- **Only the n-gram range was varied.** `min_df`, `max_df`, `max_features`, `C` and `class_weight` were held at values tuned for `(3,5)`. Another n-gram range might do better under its own tuning; this experiment cannot detect that.
- **The `max_features` cap interacts with the variable under test.** Wider ranges generate more candidate n-grams, so for some configurations the cap, not the range, determines the vocabulary.
- **Timings are single measurements** on one machine with no warm-up or repetition.
- **Hard-negative metrics could not be computed** — the class does not exist in this dataset (§6).
- Results are specific to r/SuicideWatch English posts and this three-class collapse of `severity`.

## 14. Final selected configuration

```text
analyzer    = char_wb
ngram_range = (3,5)
min_df      = 2
max_df      = 0.9
sublinear_tf= True
max_features= 50000
C           = 1.0
class_weight= balanced
solver      = lbfgs
max_iter    = 5000
seed        = 42
```

Selected on **validation macro F1**. Test-set results for this configuration are in `FINAL_TEST.md` and were produced *after* selection.

## 15. Reproduction

```bash
python experiments/ngram_ablation.py
```

Run from the repository root. Deterministic under seed 42; only wall-clock timings vary between runs.
