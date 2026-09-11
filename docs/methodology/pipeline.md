# Pipeline

The full path from external corpora to research results, with the code that
implements each step.

```
EXTERNAL DATASETS  (Komati / IRF / SDCNL / CAMS / GoEmotions)
        ↓                                   data/raw/
DATASET-SPECIFIC LOADERS                    src/data/loaders/
        ↓            normalise to: source, src_id, text, src_risk, src_meta
UNIFIED CANDIDATE POOL                      src/data/pooling/build_pool.py
        ↓
CLEANING                                    src/data/cleaning/text_cleaning.py
        ↓
DEDUPLICATION                               src/data/cleaning/dedup.py
        ↓
LEXICAL ROUTING                             src/data/pooling/lexicon.py
        ↓
SAMPLING STRATA                             src/data/pooling/strata.py
        ↓            A_likely_explicit | B_likely_implicit
                     C_hard_negative   | D_likely_noncrisis
STRATIFIED CANDIDATE SAMPLING               scripts/annotation/build_gold_candidates.py
        ↓
ANNOTATION PHASE (FROZEN)                   annotation/guidelines/
        ↓            explicit_crisis | implicit_crisis
                     hard_negative   | non_crisis
INTER-ANNOTATOR AGREEMENT                   src/annotation/agreement.py
        ↓
OPERATIONAL ANNOTATION DATASET (1,987)      data/gold/gold.csv
        ↓
TRAIN / VALIDATION / TEST SPLIT             scripts/annotation/validate_gold_dataset.py --split
        ↓
   ┌────┴────┐
TF-IDF + LR   BERT                          src/models/tfidf/ , src/models/bert/
   └────┬────┘
        ↓
EVALUATION                                  src/evaluation/metrics.py
        ↓            overall + per-slice (explicit / implicit / hard negative)
IMPLICIT-CRISIS ANALYSIS
        ↓
ERROR ANALYSIS                              src/evaluation/error_analysis.py
```

## Two vocabularies, kept apart

This is the single most important invariant in the codebase.

| | Strata | Labels |
|---|---|---|
| Names | `A_likely_explicit`, `B_likely_implicit`, `C_hard_negative`, `D_likely_noncrisis` | `explicit_crisis`, `implicit_crisis`, `hard_negative`, `non_crisis` |
| Assigned by | source signals + a keyword heuristic | a human reading the text |
| Purpose | choosing what gets read | ground truth |
| Defined in | `src/utils/labels.STRATA` | `src/utils/labels.LABELS` |

The naming is deliberately different so that a stratum can never be silently
substituted for a label. Nothing in this codebase maps one onto the other.

Two corollaries:

1. **A source label never becomes a gold label.** `is_suicide=1` in SDCNL,
   `class=suicide` in Komati, and `belong=1` in IRF are carried through as
   `src_risk` for routing and provenance. They are noisy, distantly supervised,
   and defined against different constructs than ours.
2. **The explicit-terminology detector never assigns a gold label.** It cannot
   distinguish "I want to kill myself" from "the article discussed suicide
   prevention" — which is precisely the distinction the `hard_negative` class
   exists to test. Using it as a labeller would build the study's own null
   hypothesis into its ground truth.

## Why stratified sampling

Sampling uniformly from the pool would produce a batch dominated by ordinary
posts, with implicit crisis language and hard negatives too rare to evaluate.
The strata over-sample exactly the two regions the research question lives in:

- **B** — crisis-leaning source signal, no crisis vocabulary → where implicit
  cases concentrate.
- **C** — crisis vocabulary present, non-crisis source signal → where the
  lexical baseline is expected to fail.

Sampling within a stratum is balanced across sources so no single large corpus
dominates, and the resulting batch is shuffled so stratum membership is not
recoverable from row order.

## Evaluation design

Pooled accuracy is not the result. The result is per-slice performance:

| Slice | Read primarily as | Question it answers |
|---|---|---|
| `explicit_crisis` | recall | Does the model catch the easy cases? (both should) |
| `implicit_crisis` | recall, F1 | **The research question.** Does context help where keywords cannot? |
| `hard_negative` | precision | How badly does crisis vocabulary alone trigger a false positive? |

`src/evaluation/metrics.slice_report` produces this table; the pooled figure in
`overall_report` is secondary.

## Reproducibility

- One seed (`configs/experiment.yaml: seed`) drives sampling, splitting and
  training; `src/utils/seeding.set_seed` applies it to `random`, `numpy` and
  `torch`.
- Candidate ids are content-derived SHA-1 prefixes, so re-running the pool
  build yields the same ids for the same text.
- All paths are project-relative and declared in `configs/`; no absolute paths
  appear anywhere in the code.
- Both models train on the identical train split and are evaluated on the
  identical test split. `scripts/experiments/evaluate_models.py` refuses to
  compare prediction files whose ids do not align.
