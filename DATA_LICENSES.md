# Data licences and redistribution status

None of the external corpora is committed to this repository. `data/raw/` is
gitignored in full, and every dataset must be obtained by the user from its own
source under that source's own terms.

Licence information below was established from the material actually present in
this project. Where a repository carries no licence file and no licence
statement could be found locally, the entry says so explicitly rather than
assuming a permissive default. **An absent licence is not an open licence.**

| Dataset | Source | Licence | Redistribution status | Notes |
|---|---|---|---|---|
| Komati — Suicide and Depression Detection | `kaggle.com/datasets/nikhileswarkomati/suicide-watch` | **To be verified** — check the licence field on the Kaggle dataset page | Not redistributed; obtain from Kaggle under Kaggle's terms | Kaggle accounts accept dataset-specific terms at download time. Reddit content also remains subject to Reddit's terms. |
| IRF — Interpersonal Risk Factors | `github.com/drmuskangarg/Irf` | **Licence requires verification** — no licence file present in the repository | Redistribution **not assumed to be permitted**. Contact the authors before redistributing the data or anything derived from it | Flagged as an open item in the project report. Derived annotations over IRF text may need author permission. |
| SDCNL | `github.com/ayaanzhaque/SDCNL` | **Licence requires verification** — confirm against the repository | Not redistributed; clone from the source repository | Contains scraped Reddit posts including `author` and `url` fields, which this project drops and never uses. |
| CAMS | `github.com/drmuskangarg/CAMS` | **Licence requires verification** — confirm against the repository | Redistribution **not assumed to be permitted**. Contact the authors before redistributing derived data | Overlaps with SDCNL via its `IntentSDCNL_*` files. |
| GoEmotions | `github.com/google-research/google-research/tree/master/goemotions` | **Apache License 2.0** (the google-research repository licence) | Redistribution permitted under Apache-2.0 with attribution and licence notice | The only source here with a clearly established licence. Still kept out of Git for consistency. |

## Status of the gold dataset

The gold dataset produced by this project is an **annotation layer** over text
belonging to the corpora above. Our contribution is the labels, not the text.

Consequently:

- Gold **labels and item ids** are our own work.
- Gold **text** is third-party content and inherits the restrictions of its
  source corpus.
- Because IRF and CAMS carry no established licence, the gold dataset is
  **not released publicly** in a form that reproduces their text until
  permission has been obtained. `data/gold/` is gitignored accordingly.
- A label-only release (ids + labels + source identifiers, no text) is the
  fallback if permission cannot be secured. That decision is open.

## Ethical constraints on use

These apply regardless of licence:

- The data describes real people in distress. It is used for research on
  language, not to identify, profile, re-identify or contact anyone.
- Author identifiers present in the source files (`author`, `url` in SDCNL)
  are dropped by the loaders and are never used as features.
- No new scraping is performed by this project.
- Verbatim posts are not reproduced beyond the minimum needed to illustrate an
  annotation decision, and never with identifying context.
- Nothing produced here is a clinical instrument. Model outputs classify
  language; they say nothing about any individual's risk.

## Open items

1. Confirm the Komati dataset's Kaggle licence and the exact release version used.
2. Establish licensing for IRF, SDCNL, and CAMS by contacting the authors.
3. Decide the gold-dataset release form (full text vs labels-only) once 1 and 2
   are resolved.
