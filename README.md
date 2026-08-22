# Implicit Crisis Language Detection

Detecting mental-health crisis language expressed **implicitly** — where
explicit suicide-related terminology is absent — and testing whether contextual
transformer representations do better at it than lexical features.

> **Scope.** This is a text-classification research prototype. It is not a
> clinical diagnostic system, it does not determine whether any individual is
> suicidal, and it does not predict suicide attempts. Model outputs are
> classifications of language, not assessments of people.

---

## 1. Project overview

Computational work on distress in social media has largely been organised
around explicit disclosure, and systems trained on that data inherit a lexical
shortcut: terms like *suicide*, *kill myself* and *end my life* become highly
predictive of the positive class. The shortcut fails in two directions —
distress is often expressed with no crisis vocabulary at all, and crisis
vocabulary appears abundantly in text expressing no personal crisis.

This project builds a small, manually annotated evaluation layer over existing
public corpora that separates those cases, and uses it to compare a TF-IDF +
Logistic Regression baseline against a fine-tuned BERT classifier — reported
separately per case type rather than as a single pooled score.

## 2. Research problem

> Can transformer-based NLP models identify crisis-related language expressed
> implicitly, particularly when explicit suicide-related terminology is absent?

The distinction targeted is linguistic, not clinical:

- *"I want to kill myself"* — **explicit**: the crisis meaning is carried by
  the surface lexical items.
- *"Everyone would be better off without me"* — **implicit**: no crisis term
  occurs; the meaning depends on composition in context.

A bag-of-words representation can only encode the first. Whether contextual
representations encode the second is the empirical question.

The gap is an evaluation and annotation gap, not a claim that implicit distress
is unstudied:

- **Labelling provenance.** Several widely used corpora derive labels from
  subreddit membership rather than post content.
- **Explicit dominance.** Where positives come from communities in which
  disclosure is the norm, explicit phrasing is over-represented, and a model
  can score highly while learning little beyond a keyword list.
- **Absent hard negatives.** Negative classes are usually topically unrelated
  to crisis, so the decision boundary is never tested against crisis vocabulary
  used without personal crisis.
- **No four-way organisation.** No surveyed dataset is organised around
  Explicit / Implicit / Hard Negative / Non-Crisis.

Measured on the corpora held in this project: in the released IRF splits, of
the 2,399 posts carrying at least one interpersonal-risk label, **73.2% contain
no explicit crisis marker**; in the SDCNL combined set, **62.8%** of positives
contain no such marker while **18.4%** of negatives do. The material exists;
the labels separating it do not.

## 3. Research objective

1. A manually annotated evaluation layer distinguishing four categories, built
   over existing public corpora. *(primary contribution)*
2. A controlled comparison of TF-IDF + LR against fine-tuned BERT, trained on
   identical data and evaluated on an identical unseen test set.
3. Disaggregated evaluation, reported per slice rather than pooled.
4. A qualitative error analysis of where contextual representation helps and
   which hard negatives induce false positives in each model.

## 4. Gold dataset construction

```
external corpora → candidate pool → cleaning → deduplication
                 → lexical routing → sampling strata → stratified sampling
                 → HUMAN ANNOTATION → agreement → gold dataset
                 → train / val / test
```

**Two vocabularies, deliberately kept apart.** Sampling strata
(`A_likely_explicit`, `B_likely_implicit`, `C_hard_negative`,
`D_likely_noncrisis`) decide *what a human reads*. Gold labels
(`explicit_crisis`, `implicit_crisis`, `hard_negative`, `non_crisis`) come from
*that human*. Nothing in this codebase maps one onto the other:

- A source dataset's label never becomes a gold label.
- The explicit-terminology detector never assigns a gold label — it cannot tell
  "I want to kill myself" from "the article discussed suicide prevention",
  which is exactly what the `hard_negative` class exists to test.

Human annotation is the only source of ground truth. See
[docs/methodology/pipeline.md](docs/methodology/pipeline.md).

## 5. External datasets

None is committed here — `data/raw/` is gitignored. Full detail in
[DATASETS.md](DATASETS.md); licences in [DATA_LICENSES.md](DATA_LICENSES.md).

| Dataset | Role | Size (measured locally) | Labels |
|---|---|---|---|
| **Komati** — Suicide & Depression Detection | bulk candidate text | 232,074 posts | binary, subreddit-derived |
| **IRF** — Interpersonal Risk Factors | conceptual anchor; implicit candidates | 3,522 posts | thwarted belongingness, perceived burdensomeness |
| **SDCNL** | candidate pool; hard negatives | 1,895 posts | binary `is_suicide` |
| **CAMS** | implicit distress via causes | 5,052 rows | six causal categories |
| **GoEmotions** | non-crisis text | 58,009 comments | 27 emotions + neutral |

None of them carries our four labels. They supply candidate text and
conceptual grounding only.

## 6. Annotation strategy

Four labels, exactly one per item:

| Label | Definition |
|---|---|
| `explicit_crisis` | The author's own crisis, stated through overt crisis terminology. |
| `implicit_crisis` | The author's own crisis or crisis-level distress, conveyed **without** such terminology. |
| `hard_negative` | Crisis vocabulary present, but no personal crisis — news, advocacy, third-party concern, fiction, hyperbole, past-tense recovery. |
| `non_crisis` | Neither crisis meaning nor crisis vocabulary — including ordinary sadness, stress, loneliness and frustration. |

The hardest boundary — and the one the guidelines spend the most space on — is
**implicit crisis vs. ordinary negative emotion**. General sadness, stress,
loneliness and frustration are *not* implicit crisis. That label requires
crisis-relevant contextual evidence: self-directed finality, burdensomeness,
escape framing that exceeds the situation, closure behaviour, or total
hopelessness.

Annotators see only `id`, `text`, and the columns they fill in
(`label`, `confidence`, `notes`). Stratum, source label and lexicon hits are
withheld to a separate key file so they cannot anchor the judgement.

Process: 150-item calibration subset → independent annotation → agreement →
guideline revision if needed → full annotation, with a portion double-annotated
so a final agreement figure can be reported.

Guidelines: [annotation/guidelines/annotation_guidelines.md](annotation/guidelines/annotation_guidelines.md).
Agreement: [annotation/agreement/](annotation/agreement/).

## 7. Experimental setup

Both models train on the identical gold train split, tune on the identical
validation split, and are evaluated on an identical held-out test set that is
not touched during training, feature selection or tuning. One seed
(`configs/experiment.yaml`) drives sampling, splitting and training.

The split is stratified by label so all three evaluation slices are adequately
represented in the test set.

## 8. Models

**TF-IDF + Logistic Regression** — word (1–2) and character (3–5) n-grams in a
feature union, class-weighted logistic regression. The lexical control: it can
only see surface forms.
(`src/models/tfidf/baseline.py`)

**BERT (fine-tuned)** — `bert-base-uncased`, four-way sequence classification,
max length 256. Hyperparameters in `configs/experiment.yaml`.
(`src/models/bert/classifier.py`)

## 9. Evaluation

Pooled accuracy is the secondary figure. The primary result is per-slice:

| Reported | Read primarily as | What it answers |
|---|---|---|
| Overall | macro F1, accuracy | Sanity check, not the result. |
| **Explicit crisis** | recall | Does the model catch the easy cases? |
| **Implicit crisis** | recall, F1 | **The research question.** |
| **Hard negative** | precision | How often does crisis vocabulary alone cause a false positive? |

Precision, recall, F1 and confusion matrices are reported overall and per
slice, followed by an error analysis over four buckets: TF-IDF right / BERT
wrong, BERT right / TF-IDF wrong, both wrong, and hard negatives inducing false
positives in each model.

**No results are reported in this repository yet.** The hypothesis under test —
that BERT outperforms the baseline on the implicit slice — is a hypothesis; the
outcome may equally show no advantage. Nothing here is filled in ahead of
measurement.

## 10. Repository structure

```
Implicit_Crisis/
├── data/                       # gitignored except documentation
│   ├── raw/                    # external corpora (see DATASETS.md)
│   │   ├── CAMS/  IRF/  SDCNL/  google-research/  other/
│   ├── interim/                # candidate pool, cleaning artefacts
│   ├── processed/
│   └── gold/                   # candidates → annotation → final
├── src/
│   ├── data/
│   │   ├── loaders/            # one module per corpus, one output schema
│   │   ├── cleaning/           # text cleaning, exact + near dedup
│   │   └── pooling/            # lexicon, strata, pool builder
│   ├── annotation/             # sheets, validation, agreement
│   ├── models/{tfidf,bert}/
│   ├── evaluation/             # metrics, slices, error analysis
│   └── utils/                  # config, io, seeding, label vocabularies
├── scripts/
│   ├── preprocessing/          # clean_datasets.py
│   ├── annotation/             # build candidates, sheets, agreement, validate
│   └── experiments/            # tfidf, bert, evaluate
├── annotation/
│   ├── guidelines/             # the annotation scheme
│   ├── templates/              # sheet schema
│   └── agreement/              # calibration + kappa results
├── configs/                    # paths, candidate pool, experiment
├── docs/
│   ├── project/                # summary
│   ├── datasets/               # dataset setup instructions
│   ├── methodology/            # pipeline, project flow
│   └── reports/                # project report
├── experiments/                # measured results only
├── notebooks/
├── tests/
├── DATASETS.md
├── DATA_LICENSES.md
└── requirements.txt
```

## 11. Installation

```bash
git clone https://github.com/Adi-Sin101/Implicit_Crisis.git
cd Implicit_Crisis

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt          # data + annotation + baseline
pip install -r requirements-ml.txt       # only needed for the BERT experiment
```

Python 3.10 or newer. The BERT dependencies are kept separate so the entire
data-preparation and annotation half of the project runs without a
deep-learning stack.

## 12. Dataset setup

`data/raw/` is empty after cloning. Follow
[docs/datasets/dataset_setup.md](docs/datasets/dataset_setup.md) to place each
corpus at its documented path, then verify:

```bash
python scripts/preprocessing/clean_datasets.py
```

Paths live in `configs/paths.yaml` and are all project-relative. If your data
sits elsewhere, edit that file — not the code.

## 13. Running the pipeline

```bash
# 1. Load, clean and deduplicate every corpus into one candidate pool
python scripts/preprocessing/clean_datasets.py

# 2. Draw a stratified candidate sample for annotation
python scripts/annotation/build_gold_candidates.py

# 3. Generate blind annotation sheets (calibration round first)
python scripts/annotation/create_annotation_sheet.py --annotators A B --calibration 150

#    ... annotators fill in data/gold/annotation/calibration_150_{A,B}.csv ...

# 4. Measure agreement, revise the guidelines if a class is weak
python scripts/annotation/calculate_agreement.py \
    --files A=data/gold/annotation/calibration_150_A.csv \
            B=data/gold/annotation/calibration_150_B.csv

# 5. Full annotation, then validate and split
python scripts/annotation/create_annotation_sheet.py --annotators A B
python scripts/annotation/validate_gold_dataset.py --input data/gold/final/gold.csv --split

# 6. Experiments
python scripts/experiments/run_tfidf_baseline.py
python scripts/experiments/run_bert.py

# 7. Comparison and error analysis
python scripts/experiments/evaluate_models.py \
    --predictions tfidf=experiments/baselines/tfidf/predictions_test.csv \
                  bert=experiments/bert/predictions_test.csv
```

## 14. Reproducibility

- A single seed in `configs/experiment.yaml` drives sampling, splitting and
  training, applied to `random`, `numpy` and `torch`.
- Candidate ids are content-derived SHA-1 prefixes, so rebuilding the pool from
  the same corpora yields the same ids.
- All paths are project-relative and declared in `configs/`. No absolute or
  machine-specific paths appear anywhere in the code.
- Both models use the identical splits; `evaluate_models.py` refuses to compare
  prediction files whose ids do not align.
- Deduplication runs before sampling, so no post can be annotated twice or leak
  across the train/test boundary.

## 15. Licence and dataset restrictions

Code in this repository: **MIT** (see [LICENSE](LICENSE)).

Data is a different matter. Every corpus is third-party, none is redistributed
here, and two of the source repositories (IRF, CAMS) carry no licence file at
all — so redistribution of those is **not assumed to be permitted**. The gold
dataset is an annotation layer over third-party text: the labels are ours, the
text is not. `data/` is gitignored accordingly, and the public release form of
the gold dataset (full text vs. labels-only) is an open question pending
permission from the corpus authors.

Ethical constraints, which apply regardless of licence, are set out in
[DATA_LICENSES.md](DATA_LICENSES.md): no re-identification, no contact with
authors of posts, author identifiers dropped, no new scraping, no clinical use.

## 16. Citation

The project report is at
[docs/reports/project_report.docx](docs/reports/project_report.docx).
Citation details for this work will be added on publication.

Source corpora and prior work:

- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019.
- Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL 2020.
- Garg, M., et al. (2022). *CAMS: An Annotated Corpus for Causal Analysis of Mental Health Issues in Social Media Posts.* LREC 2022.
- Garg, M., Shahbandegan, A., Chadha, A., & Mago, V. (2023). *An Annotated Dataset for Explainable Interpersonal Risk Factors of Mental Disturbance in Social Media Posts.* Findings of ACL 2023.
- Haque, A., Reddi, V., & Giallanza, T. (2021). *Deep Learning for Suicide and Depression Identification with Unsupervised Label Correction.* ICANN 2021.
- Komati, N. *Suicide and Depression Detection* [dataset]. Kaggle.
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12, 2825–2830.

---

If you or someone you know is in crisis, please contact a local emergency
service or a suicide-prevention helpline. This repository is research software
and is not a source of help.
