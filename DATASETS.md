# Datasets

Every corpus this project reads is an **external** resource. None of them is
committed to this repository (`data/raw/` is gitignored); each must be obtained
from its own source and placed at the documented path.

None of these datasets carries our four target labels. They supply **candidate
text** and conceptual grounding. The only labelled resource for the four
categories is the gold dataset produced by human annotation in this project.

Row counts, columns and label distributions below were measured directly from
the copies held locally in `data/raw/`. Where a figure disagrees with the
source paper, or where a fact could not be established from the material at
hand, it is marked **To be verified** rather than guessed.

---

## 1. Komati — Suicide and Depression Detection

| Field | Value |
|---|---|
| Local path | `data/raw/other/Suicide_Detection.csv` |
| Original source | Kaggle: `kaggle.com/datasets/nikhileswarkomati/suicide-watch` |
| Repository | Kaggle dataset (no code repository) |
| Paper | None known — dataset release, not a paper. **To be verified** |
| Files | `Suicide_Detection.csv` (~160 MB) |
| Records | 232,074 rows (measured) |
| Text column | `text` |
| Label column | `class` |
| Label values | `suicide` (116,037) / `non-suicide` (116,037) — exactly balanced (measured) |
| Other columns | `Unnamed: 0` — the original row index |
| Label provenance | Distant supervision: subreddit of origin (r/SuicideWatch vs r/teenagers), not the author's own statement |
| Role here | Primary bulk source of candidate text for all four strata |
| License | **To be verified** — check the Kaggle dataset page for the stated licence |
| Redistribution | **Not redistributed.** Not committed; obtain from Kaggle directly |
| Notes | Positives are explicit-dominant; negatives are topically unrelated rather than hard. Labels are candidate signals only. The exact release version downloaded here is **to be verified** (row counts vary between releases). |

## 2. IRF — Interpersonal Risk Factors

| Field | Value |
|---|---|
| Local path | `data/raw/IRF/` |
| Original source | `github.com/drmuskangarg/Irf` |
| Paper | Garg, M., Shahbandegan, A., Chadha, A., & Mago, V. (2023). *An Annotated Dataset for Explainable Interpersonal Risk Factors of Mental Disturbance in Social Media Posts.* Findings of ACL 2023 |
| Files | `train_data.csv` (1,972), `test_data.csv` (1,057), `val_data.csv` (493) — measured |
| Records | 3,522 across the released splits (measured). The paper reports 5,051 — **to be verified** |
| Text column | `text` |
| Label columns | `belong` (0/1), `burden` (0/1) |
| Explanation columns | `belong_exp`, `burden_exp` — human-written rationale spans |
| Label meaning | `belong` = Thwarted Belongingness; `burden` = Perceived Burdensomeness. Train split: belong 1,063 positive / 909 negative; burden 636 positive / 1,336 negative (measured) |
| Role here | Conceptual anchor for the annotation scheme, and the main source of **implicit** candidates (stratum B). Measured on the released splits: of the 2,399 posts carrying at least one risk-factor flag, 73.2% contain no explicit crisis marker |
| License | **License requires verification** — no licence file in the repository |
| Redistribution | **Not permitted without checking.** Not committed; clone from the source repository. Contact the authors before redistributing derived data |
| Notes | A risk-factor label is **not** a crisis label. IRF labels are never mapped one-to-one onto our categories; they route candidates for human reading only. Also contains `src/` — third-party model code (RNN/transformer trainers, LIME/SHAP notebooks) that this project does not use. |

## 3. SDCNL

| Field | Value |
|---|---|
| Local path | `data/raw/SDCNL/` |
| Original source | `github.com/ayaanzhaque/SDCNL` |
| Paper | Haque, A., Reddi, V., & Giallanza, T. (2021). *Deep Learning for Suicide and Depression Identification with Unsupervised Label Correction.* ICANN 2021 |
| Files | `data/combined-set.csv` (1,895), `data/training-set.csv` (1,516), `data/testing-set.csv` (379) — measured |
| Records | 1,895 posts in the combined set (measured) |
| Text columns | `title`, `selftext` (raw); `title_clean`, `selftext_clean`, `megatext_clean` (pre-cleaned by the authors) |
| Label column | `is_suicide` |
| Label values | `1` (980) / `0` (915) — measured |
| Other columns | `author`, `num_comments`, `url`, `selftext_length`, `title_length` |
| Label meaning | Binary suicide-risk vs depression, produced with unsupervised label correction over subreddit-derived labels |
| Role here | Candidate pool, and the main source of **hard-negative** candidates. Measured: 62.8% of positives contain no explicit crisis marker, while 18.4% of negatives do contain one |
| License | **License requires verification** — see the repository for any stated terms |
| Redistribution | **Not redistributed.** Not committed; clone from the source repository |
| Notes | Small. Draws a suicide-vs-depression distinction, so its labels must be re-read against our scheme rather than mapped across. The `author` and `url` columns are dropped by our loader and never used. Repository also contains third-party model code (`classifiers.py`, `word_embeddings.py`, label-correction scripts, `web-scraper.py`) that this project does not run. |

## 4. CAMS

| Field | Value |
|---|---|
| Local path | `data/raw/CAMS/` |
| Original source | `github.com/drmuskangarg/CAMS` |
| Paper | Garg, M., et al. (2022). *CAMS: An Annotated Corpus for Causal Analysis of Mental Health Issues in Social Media Posts.* LREC 2022. Full author list **to be verified** |
| Files | `dataset (7).csv` (5,052), `CAMS/data/added_CAMS_data.csv` (3,155), `CAMS/data/IntentSDCNL_Training.csv` (1,461), `CAMS/data/IntentSDCNL_Testing.csv` (370) — measured |
| Records | 5,052 in the main file (paper reports 5,051 — **to be verified**) |
| Text column | `text` (main file) / `selftext` (the other three) |
| Label columns | `category` (main) / `cause` (added) / `ANNOTATIONS` (Intent files) |
| Explanation columns | `explanation`, `inference`, `Interpretations` |
| Label values | Six numeric codes. Main file distribution: 5 → 1,408, 4 → 1,344, 0 → 690, 2 → 628, 3 → 623, 1 → 350 (measured) |
| Label meaning | Causal categories — reported by the paper as: no reason, bias/abuse, jobs/careers, medication, relationships, alienation. **The code-to-category mapping is to be verified against the paper before being reported in writing.** |
| Role here | Candidate pool for implicit distress expressed through causes rather than crisis vocabulary |
| License | **License requires verification** — see the repository for any stated terms |
| Redistribution | **Not redistributed.** Not committed; clone from the source repository |
| Notes | Labels a *cause*, not a risk level, and carries no crisis / non-crisis judgement of its own. The `IntentSDCNL_*` files overlap with SDCNL by construction, so deduplication against SDCNL is mandatory — `src/data/cleaning/dedup.py` handles this. `dataset (7).csv` uses a non-UTF-8 encoding in places; the loader falls back to latin-1. |

## 5. GoEmotions

| Field | Value |
|---|---|
| Local path | `data/raw/other/goemotions/data/` |
| Original source | `github.com/google-research/google-research/tree/master/goemotions` |
| Paper | Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions.* ACL 2020 |
| Files | `train.tsv`, `dev.tsv`, `test.tsv`, `emotions.txt`, `ekman_mapping.json`, `sentiment_mapping.json` |
| Records | 58,009 Reddit comments across the curated splits (per the dataset README) |
| Format | Tab-separated, no header: text, comma-separated label ids, comment id |
| Label column | column 2 — indices into `emotions.txt` |
| Label meaning | 27 fine-grained emotions plus `neutral` |
| Role here | Bulk source of ordinary, non-mental-health text for the **non-crisis** stratum (D) |
| License | **Apache License 2.0** (the google-research repository licence) |
| Redistribution | Permitted under Apache-2.0 with attribution. Still not committed here, to keep `data/raw/` uniformly external |
| Notes | Comment-length text with no mental-health framing, so it supplies *easy* rather than hard negatives. The larger `full_dataset/` raw CSVs are downloaded separately per the GoEmotions README and are not required by this project. |

---

## Deduplication between sources

The corpora overlap and must be deduplicated before annotation:

- CAMS `IntentSDCNL_Training.csv` / `IntentSDCNL_Testing.csv` are drawn from SDCNL.
- SDCNL was scraped from the same subreddits that the Komati corpus covers.

`scripts/preprocessing/clean_datasets.py` applies exact deduplication (over
case- and punctuation-normalised text) and optional shingle-based near-duplicate
removal. Without it the same post could be annotated twice, or leak across the
train/test split.

## Role in gold-dataset construction

```
external corpora  →  candidate pool  →  cleaning + deduplication
                  →  lexical routing into sampling strata
                  →  stratified sampling
                  →  HUMAN ANNOTATION      ← the only source of gold labels
                  →  gold dataset          →  train / val / test
```

Source labels (`class`, `is_suicide`, `belong`, `burden`, `category`) are
carried through the pool as `src_risk` / `src_meta` for **routing and
provenance only**. No source label is ever promoted to a gold label, and the
explicit-terminology detector never assigns one either.

## Obtaining the data

See [docs/datasets/dataset_setup.md](docs/datasets/dataset_setup.md) for the
per-dataset download instructions and the expected directory layout.
