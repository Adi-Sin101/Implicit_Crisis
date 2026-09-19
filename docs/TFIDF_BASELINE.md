# TF-IDF + Logistic Regression Baseline

> **STATUS: FROZEN.** `analyzer=char_wb` with `ngram_range=(3,5)` is the **final
> TF-IDF model configuration** and must not be changed. The n-gram ablation in
> `results/ngram_ablation/` is the evidence: `(3,5)` had the highest validation
> macro F1 (0.8054) of the six ranges tested. No further n-gram search or TF-IDF
> model-selection experiment should be run unless the project owner asks.
> Machine-readable record: [`configs/tfidf_final_frozen.json`](../configs/tfidf_final_frozen.json).
> Split audit: [`docs/FINAL_SPLIT_AUDIT.md`](FINAL_SPLIT_AUDIT.md).

A sparse-feature baseline for three-class crisis severity classification, built
to be directly comparable with the BERT experiment in
`ipnyb/bert_crisis_classifier (1).ipynb`. Every choice that could affect
comparability — dataset, input column, label mapping, split, seed, selection
metric — is copied from that notebook rather than re-decided here.

All numbers below come from an actual executed run. Reproduce them with the
command in [Reproducibility](#reproducibility).

---

## Dataset

```
data/final_datasets/merged_severity_dataset_not_keyword_keyword.csv
```

This is the project's final dataset and the same file the BERT notebook loads.
No other CSV was read, merged, or compared.

**Audit** (`experiments/tfidf_baseline/results/dataset_audit.csv`):

| Property | Value |
|---|---|
| Rows | 1,294 |
| Columns | 10 |
| Column names | `url`, `author`, `content`, `severity`, `created`, `gpt_label`, `claude_label`, `gemini_label`, `llama_label`, `mistral_label` |
| Missing `content` | 0 |
| Missing `severity` | 0 |
| Empty / whitespace-only `content` | 0 |
| Rows removed by `dropna(content, severity)` | 0 |
| Duplicate full rows | 0 |
| Duplicate `content` values | 0 |
| Unique `content` values | 1,294 |

Missing values exist only in columns the model never sees: `created` (124),
`gpt_label` (133), `claude_label` (125), `gemini_label` (125), `llama_label`
(128), `mistral_label` (124). They are left untouched.

**Text length** (`results/text_statistics.csv`):

| | characters | words |
|---|---|---|
| mean | 304.04 | 59.67 |
| std | 179.87 | 34.90 |
| min | 7 | 2 |
| median | 282 | 55 |
| p95 | 621.3 | 119 |
| max | 898 | 191 |

Per class, mean words: Non-Crisis 46.06, Implicit Crisis 65.71, Explicit Crisis
64.22. Non-Crisis posts are visibly shorter; the two crisis classes are not
separable by length.

---

## Input

```
content
```

The `content` column is the model's only input. Nothing is concatenated to it.

Explicitly **excluded** from the feature space — these are labels, annotations
or metadata, and using any of them would leak the target:

```
severity  gpt_label  claude_label  gemini_label  llama_label
mistral_label  url  author  created
```

The exclusion list is recorded in
`models/tfidf_logistic_regression/config.json` and asserted by
`tests/test_tfidf_baseline.py::test_label_columns_are_excluded_from_features`.

## Original target

```
severity
```

## Mapping

```
severity 0     ->  0 = Non-Crisis
severity 1     ->  1 = Implicit Crisis
severity 2-6   ->  2 = Explicit Crisis
```

Identical to `map_severity_to_3class` in the BERT notebook.

Raw severity counts: 0 → 351, 1 → 324, 2 → 302, 3 → 72, 4 → 37, 5 → 50, 6 → 158.

**Resulting three-class distribution:**

| Label | Class | Count | Share |
|---|---|---|---|
| 0 | Non-Crisis | 351 | 27.13% |
| 1 | Implicit Crisis | 324 | 25.04% |
| 2 | Explicit Crisis | 619 | 47.84% |

Moderate imbalance — Explicit Crisis is roughly twice either other class. No
class is rare enough to be degenerate.

---

## Duplicate handling

Duplicates were checked **before** splitting, because identical texts landing
in different splits would inflate every score.

```
Total rows        : 1294
Duplicate rows    : 0
Duplicate content : 0
Rows removed      : 0
Rows remaining    : 1294
```

No duplicates exist, so no rows were removed. This is worth stating plainly:
it means the leakage-free requirement and exact agreement with the BERT
notebook's split are not in tension here. The BERT notebook does not
deduplicate; because there is nothing to deduplicate, both experiments operate
on the same 1,294 rows and therefore on the same split.

The script still contains the removal path (keep-first, with a printed count)
in case the dataset changes later. Recorded in `results/duplicate_report.csv`.

---

## Split

```
70% train / 15% validation / 15% test
stratified on the 3-class label
random_state = 42
```

Produced by the same two-stage `train_test_split` the BERT notebook uses:
a 0.30 test split, then a 0.50 split of that remainder.

| Split | n | % | Non-Crisis | Implicit Crisis | Explicit Crisis |
|---|---|---|---|---|---|
| Train | 905 | 69.94% | 245 | 227 | 433 |
| Validation | 194 | 14.99% | 53 | 48 | 93 |
| Test | 195 | 15.07% | 53 | 49 | 93 |

Verified: no row overlap and no text overlap between any two splits.

`tests/test_tfidf_baseline.py::test_split_reproduces_bert_notebook_partition`
reruns the notebook's own split calls — passing only `(texts, labels)`, exactly
as the notebook does — and asserts the resulting train/val/test text sets are
identical to this baseline's. The baseline additionally threads a row-index
array through `train_test_split`; that test is what proves the extra array does
not perturb the partition. **The two experiments share the same test set.**

The full assignment is saved to `results/data_split.csv` with columns
`row_index, url, content, severity, label, class_name, split`, so a future BERT
run can be pinned to exactly these rows.

---

## Features

TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`), with
`lowercase=True` and `strip_accents="unicode"` as the only normalisation. No
stemming, no stop-word removal, no keyword rules, no crisis-term substitution,
no external lexicons. The classifier sees the posts as written.

### Leakage control

The vectorizer is fitted on the **training texts only**:

```
train  ->  vectorizer.fit_transform(...)
val    ->  vectorizer.transform(...)
test   ->  vectorizer.transform(...)
```

`fit_transform` is never called on the full dataset. This is verified two ways
in `tests/test_tfidf_baseline.py`:

- `test_vectorizer_vocabulary_comes_from_training_text_only` refits the
  recorded configuration on the training split alone and asserts the vocabulary
  matches the shipped vectorizer's exactly.
- `test_fitting_on_all_data_would_differ` fits the same configuration on all
  1,294 rows and asserts the vocabulary **differs** — which is what gives the
  first test teeth.

## Classifier

`sklearn.linear_model.LogisticRegression`, multinomial, `solver="lbfgs"`,
`max_iter=5000`, `random_state=42`. Every configuration tested converged well
inside the iteration budget (the selected one in 20 iterations), asserted by
`test_every_configuration_converged`.

---

## Hyperparameter search

19 configurations, all scored on the **validation** set only, in two stages.
Full record in `results/validation_results.csv`; chart in
`figures/validation_hyperparameters.png`.

### Stage A — TF-IDF geometry (classifier held at C=1.0, class_weight="balanced")

| Configuration | analyzer | ngram | min_df | max_df | sublinear | max_features | n_features | Val macro F1 |
|---|---|---|---|---|---|---|---|---|
| `char_wb_3_5` | char_wb | (3,5) | 2 | 0.9 | True | 50000 | 17,708 | **0.8054** |
| `char_wb_2_5` | char_wb | (2,5) | 3 | 0.9 | True | 50000 | 14,128 | 0.8037 |
| `word_1_1_mindf2` | word | (1,1) | 2 | 0.9 | True | None | 1,974 | 0.7694 |
| `word_1_1_mindf1` | word | (1,1) | 1 | 1.0 | True | None | 4,318 | 0.7679 |
| `word_1_1_mindf2_nosub` | word | (1,1) | 2 | 0.9 | False | None | 1,974 | 0.7652 |
| `word_1_2_mindf2` | word | (1,2) | 2 | 0.9 | True | None | 7,371 | 0.7430 |
| `word_1_2_max20k` | word | (1,2) | 2 | 0.9 | True | 20000 | 7,371 | 0.7430 |
| `word_1_2_mindf1` | word | (1,2) | 1 | 1.0 | True | None | 30,187 | 0.7292 |
| `word_1_2_mindf3` | word | (1,2) | 3 | 0.9 | True | None | 4,160 | 0.7194 |

Character n-grams were tested for the reason stated in the experiment plan:
implicit crisis language carries abbreviations, informal spelling and unusual
word forms that word-boundary tokenisation fragments. On validation they won by
about 3.6 macro-F1 points. Word bigrams consistently *hurt* — at 905 training
documents most bigrams appear once or twice and add noise rather than signal.
`word_1_2_max20k` scored identically to `word_1_2_mindf2` because the cap of
20,000 never bound (7,371 features).

### Stage B — Logistic Regression on the winning vectorizer (`char_wb_3_5`)

| C | class_weight | Val macro F1 | Val accuracy | Val implicit F1 |
|---|---|---|---|---|
| 1.0 | balanced | **0.8054** | 0.8093 | 0.7593 |
| 5.0 | balanced | 0.7985 | 0.8041 | 0.7778 |
| 0.5 | balanced | 0.7958 | 0.7990 | 0.7636 |
| 10.0 | balanced | 0.7932 | 0.7990 | 0.7664 |
| 0.1 | balanced | 0.7759 | 0.7784 | 0.7568 |
| 5.0 | None | 0.7724 | 0.7835 | 0.7400 |
| 10.0 | None | 0.7679 | 0.7784 | 0.7327 |
| 1.0 | None | 0.7364 | 0.7526 | 0.7312 |
| 0.5 | None | 0.6750 | 0.7113 | 0.7059 |
| 0.1 | None | 0.2833 | 0.5052 | 0.1154 |

### Class imbalance

`class_weight="balanced"` helped at every value of C, and the gap widens as
regularisation strengthens: at C=1.0 it is worth 6.9 macro-F1 points
(0.8054 vs 0.7364), at C=0.1 it is worth 49.2 points (0.7759 vs 0.2833). The
unweighted C=0.1 model collapses onto the majority class — its Implicit Crisis
F1 is 0.1154. Balanced weighting was chosen on **validation** macro F1, before
the test set was touched, and matches the weighted-loss choice the BERT
notebook makes.

The `char_wb_3_5 | C=1.0 | cw=balanced` row appears twice in
`validation_results.csv` and in the chart — once from each stage. Both runs
scored 0.8054, which is a useful determinism check rather than a duplicate
entry.

---

## Best configuration

Selected on the highest **validation macro F1**, then frozen before any test
evaluation.

```json
{
  "tfidf": {
    "analyzer": "char_wb",
    "ngram_range": [3, 5],
    "min_df": 2,
    "max_df": 0.9,
    "sublinear_tf": true,
    "max_features": 50000,
    "lowercase": true,
    "strip_accents": "unicode",
    "n_features_fitted": 17708,
    "fitted_on": "training split only"
  },
  "logistic_regression": {
    "C": 1.0,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "max_iter": 5000,
    "random_state": 42,
    "n_iter_used": 20
  }
}
```

- **Validation macro F1: 0.8054**
- Validation accuracy: 0.8093
- Validation Implicit Crisis F1: 0.7593

---

## Test results

Single evaluation of the frozen model on the held-out 15% test set (n = 195).
No tuning happened after these numbers were seen.

### Overall

| Metric | Value |
|---|---|
| Accuracy | **0.7128** |
| Macro Precision | 0.7007 |
| Macro Recall | 0.6979 |
| **Macro F1** | **0.6962** |
| Weighted Precision | 0.7155 |
| Weighted Recall | 0.7128 |
| Weighted F1 | 0.7117 |

### Per class

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Non-Crisis | 0.7045 | 0.5849 | 0.6392 | 53 |
| Implicit Crisis | 0.6316 | 0.7347 | 0.6792 | 49 |
| Explicit Crisis | 0.7660 | 0.7742 | 0.7701 | 93 |

Performance orders exactly as the difficulty of the classes would suggest:
Explicit Crisis is easiest, Implicit Crisis sits in the middle, and Non-Crisis
has the weakest recall.

---

## Implicit Crisis results

The project's focus class, reported separately so it need not be dug out of the
report above (`results/implicit_crisis_metrics.csv`):

| Metric | Value |
|---|---|
| **Precision** | **0.6316** |
| **Recall** | **0.7347** |
| **F1** | **0.6792** |
| Support | 49 |

Counted from the actual test predictions:

| | Count |
|---|---|
| True Implicit Crisis correctly detected | **36** of 49 |
| Implicit Crisis missed | **13** |
| — misread as Non-Crisis | 4 |
| — misread as Explicit Crisis | 9 |
| Other classes wrongly called Implicit Crisis | **21** |
| — from true Non-Crisis | 9 |
| — from true Explicit Crisis | 12 |

The baseline is tilted toward recall over precision on this class (0.735 vs
0.632), which is the direction a screening tool would usually want: it catches
about three quarters of implicit-crisis posts at the cost of 21 false alarms.
Of the 13 it misses, only 4 are dismissed as Non-Crisis outright; the other 9
are still flagged as a crisis, just at the wrong severity.

---

## Confusion matrix

Counts (`results/confusion_matrix.csv`, `figures/confusion_matrix.png`):

| actual \ predicted | Non-Crisis | Implicit Crisis | Explicit Crisis |
|---|---|---|---|
| **Non-Crisis** | 31 | 9 | 13 |
| **Implicit Crisis** | 4 | 36 | 9 |
| **Explicit Crisis** | 9 | 12 | 72 |

Row-normalised (`results/normalized_confusion_matrix.csv`,
`figures/normalized_confusion_matrix.png`):

| actual \ predicted | Non-Crisis | Implicit Crisis | Explicit Crisis |
|---|---|---|---|
| **Non-Crisis** | 0.5849 | 0.1698 | 0.2453 |
| **Implicit Crisis** | 0.0816 | 0.7347 | 0.1837 |
| **Explicit Crisis** | 0.0968 | 0.1290 | 0.7742 |

### Error patterns actually present

1. **Non-Crisis is the weakest class, and it over-escalates.** Only 58.5% of
   Non-Crisis posts are recognised. 13 of 53 (24.5%) are called Explicit Crisis
   and 9 (17.0%) Implicit Crisis — 41.5% escalated to some crisis label. This
   is the single largest error block in the matrix, and it is the expected
   trade-off from `class_weight="balanced"`: the model is pushed away from the
   majority-adjacent safe prediction.

2. **The Implicit / Explicit boundary is the second source of error**, and it
   is close to symmetric: 9 Implicit posts are read as Explicit, 12 Explicit as
   Implicit. These 21 errors are severity confusions between two classes the
   model agrees are crises — a different and less damaging failure than missing
   a crisis entirely.

3. **Outright misses are rare.** Only 4 Implicit and 9 Explicit Crisis posts
   (13 of 142 true crisis posts, 9.2%) are labelled Non-Crisis.

So the baseline's errors are concentrated in over-flagging rather than
under-flagging. It reads roughly two in five safe posts as some form of crisis,
while letting under one in ten crisis posts through as safe.

---

## Feature analysis

Top 20 features per class by Logistic Regression coefficient, from
`results/top_features.csv` (chart: `figures/top_features.png`). Because the
selected vectorizer is `char_wb`, features are character n-grams; leading and
trailing spaces are part of the n-gram and are shown as written.

**Non-Crisis** — ` you` (0.815), ` you` (0.791), ` yo` (0.787), `you ` (0.674),
`ou ` (0.674), ` you ` (0.656), `talk` (0.613), ` talk` (0.613), ` tal` (0.601),
`alk` (0.574), `talk ` (0.549), `alk ` (0.528), `lk ` (0.513), `tal` (0.505),
` are` (0.432), ` ta` (0.417), `your` (0.417), ` your` (0.401), `ple` (0.371),
`eci` (0.364)

**Implicit Crisis** — `not` (0.832), ` am ` (0.765), `not ` (0.760), ` not`
(0.740), ` am` (0.691), `am ` (0.670), ` wa` (0.658), `ot ` (0.645), ` tir`
(0.636), ` tire` (0.630), `tired` (0.630), ` not ` (0.620), ` wak` (0.616),
`ired` (0.613), `wak` (0.600), `tir` (0.593), `ired ` (0.593), `tire` (0.590),
`ant ` (0.572), `ire` (0.552)

**Explicit Crisis** — ` kil` (0.788), ` kill` (0.788), `kil` (0.740), `kill`
(0.740), `kill ` (0.733), `ill ` (0.732), `ill` (0.730), ` ki` (0.711), `mysel`
(0.684), `myse` (0.684), `yself` (0.684), `mys` (0.684), `ysel` (0.684), `yse`
(0.684), ` mys` (0.679), ` myse` (0.679), `self` (0.641), `elf` (0.619), `uic`
(0.608), `ici` (0.584)

The three groups are qualitatively different. Explicit Crisis is dominated by
fragments of a small number of overt lexical items — `kill`, `myself`, and the
`uic`/`ici` of *suicide*. Non-Crisis leans on second-person and supportive
register (`you`, `your`, `talk`, `are`): posts addressed outward to others.
Implicit Crisis is carried by negation and first-person state description
(`not`, `am`, `tired`, `wak`- as in *wake*) rather than by any crisis term —
which is what makes the class hard, and is consistent with the fact that
character n-grams fragment these forms in ways word tokens do not.

**These are features the linear model weights heavily when separating the
classes in this training split. They describe the classifier's learned decision
boundary. They are not causal indicators of crisis, and they should not be read
as a lexicon, a screening rule, or clinical evidence.** A high coefficient on
`tired` means the token helped this model separate these 905 training posts;
it does not mean the word signals crisis in general.

---

## Limitations

Only what this run actually supports:

1. **Validation overestimates test performance by a wide margin** — 0.8054 vs
   0.6962 macro F1, a gap of 10.9 points. With 194 validation and 195 test
   posts, single-split estimates carry a standard error of roughly ±3 points,
   and the configuration was *selected* on validation, which biases that
   estimate upward. The test figure is the honest one; the validation figure
   should not be quoted as the baseline's performance.

2. **The test set is small (n = 195).** Per-class support is 53 / 49 / 93.
   Differences of a few points between this baseline and any other model on
   this split are within noise. Implicit Crisis F1 of 0.6792 rests on 49 posts.

3. **Character n-grams won on validation but may not be the more robust
   choice.** `char_wb_3_5` beat the best word model by 3.6 validation points,
   yet the word-level models had higher validation Implicit Crisis F1 in
   several configurations. The selection rule was fixed in advance (macro F1 on
   validation), so this was not revisited — but it means the analyzer choice is
   less settled than the macro-F1 ranking alone suggests.

4. **No cross-validation.** A single 70/15/15 split was used, because that is
   what the BERT experiment uses and comparability was the priority. Repeated
   splits or k-fold CV would give tighter estimates.

5. **The model has no notion of context beyond surface strings.** TF-IDF cannot
   represent negation scope, sarcasm, temporal framing ("I used to want to die")
   or narrative about a third party. Some of the 21 Implicit/Explicit
   confusions are plausibly of this kind, though this run did not analyse
   individual errors to confirm it.

6. **Single-source data.** All posts come from r/SuicideWatch. Nothing here
   speaks to performance on other platforms, registers, or populations.

7. **Severity 2–6 are collapsed into one class.** Any structure within Explicit
   Crisis (severity 2 has 302 posts, severity 6 has 158) is invisible to this
   experiment by design.

---

## Reproducibility

Deterministic given the seed: `SEED = 42` fixes `numpy`, the two
`train_test_split` calls, and `LogisticRegression`. Rerunning reproduces every
number in this document.

```bash
# 1. dataset audit
python experiments/tfidf_baseline/audit_dataset.py

# 2. full experiment: split, search, freeze, test, figures, artifacts, reload check
python experiments/tfidf_baseline/run_tfidf_baseline.py

# 3. verification suite (25 tests)
python -m pytest tests/test_tfidf_baseline.py -v
```

Run from the repository root. Requires `scikit-learn`, `pandas`, `numpy`,
`matplotlib`, `joblib`. Verified on Python 3.13.7 with scikit-learn 1.9.0,
pandas 3.0.0, numpy 2.3.5, matplotlib 3.11.1, joblib 1.5.3.

### Artifacts

```
models/tfidf_logistic_regression/
├── tfidf_vectorizer.joblib      # fitted on the 905 training texts only
├── logistic_regression.joblib
└── config.json
```

The run script reloads both artifacts and asserts that the reloaded vectorizer
reproduces the test feature matrix exactly, that the reloaded classifier
reproduces the test predictions exactly, and that `predict_proba` returns valid
3-class distributions. The reload check passed.

### Testing performed

`tests/test_tfidf_baseline.py` — 25 tests, all passing:

- **Dataset (6)** — correct path in config, required columns present, no
  missing `content`/`severity`, severity→3-class mapping correct for every
  severity value present, exactly three classes, no duplicate content.
- **Split (5)** — reproduces the BERT notebook's partition, 70/15/15
  proportions, stratification within 2 points per class per split, no text
  overlap between any pair of splits, seed and stratification recorded.
- **Leakage (3)** — shipped vocabulary equals a train-only refit, differs from
  an all-data fit, label/metadata columns listed as excluded.
- **Artifacts and metrics (8)** — artifacts exist, reloaded artifacts predicting
  from `content` alone reproduce all seven recorded test metrics,
  `predict_proba` well-formed, confusion matrix consistent with the
  classification report, Implicit Crisis counts consistent with the confusion
  matrix, normalised rows sum to 1, comparison file holds only the TF-IDF row.
- **Protocol (3)** — the frozen configuration is the validation macro-F1 winner,
  every configuration converged below `max_iter`, all expected result files and
  figures exist.

---

## Comparability with the BERT experiment

The BERT notebook was read for reference and **not modified**. No BERT model was
trained, replaced, or overwritten by this work, and no BERT numbers appear in
`results/model_comparison_ready.csv`.

| | TF-IDF baseline | BERT notebook |
|---|---|---|
| Dataset | `merged_severity_dataset_not_keyword_keyword.csv` | same |
| Text column | `content` | same |
| Cleaning | `dropna(["content","severity"])` | same |
| Label mapping | 0→0, 1→1, 2–6→2 | same |
| Split | stratified 70/15/15 | same |
| Seed | 42 | same |
| Selection metric | macro F1 on validation | macro F1 on validation |
| Imbalance handling | `class_weight="balanced"` | class-weighted cross-entropy |
| Trained on | train split only | train split only |

The split-equivalence test makes the shared test set a verified fact rather
than an assumption. When the BERT experiment is rerun against
`results/data_split.csv`, the two models will be scored on the same 195 posts
with the same metrics, and `results/model_comparison_ready.csv` can take a
second row.

**Baseline to beat: macro F1 0.6962, Implicit Crisis F1 0.6792.**
