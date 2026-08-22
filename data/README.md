# Data

Everything under this directory is gitignored except these README files. No
corpus text — source or annotated — is committed to this repository.

| Directory | Content | Produced by |
|---|---|---|
| `raw/` | External corpora, unmodified | Downloaded by the user — see [`../docs/datasets/dataset_setup.md`](../docs/datasets/dataset_setup.md) |
| `interim/preprocessing/` | Cleaned, deduplicated candidate pool | `scripts/preprocessing/clean_datasets.py` |
| `interim/candidate_pool/` | Intermediate pooling artefacts | `src/data/pooling/` |
| `processed/` | Model-ready derived data | experiment scripts |
| `gold/candidates/` | Stratified candidate sample | `scripts/annotation/build_gold_candidates.py` |
| `gold/annotation/` | Blind annotation sheets + `candidate_key.csv` | `scripts/annotation/create_annotation_sheet.py` |
| `gold/final/` | Completed gold dataset and its train/val/test splits | `scripts/annotation/validate_gold_dataset.py --split` |

`raw/` is never written to by any script in this project.

## Why none of it is committed

Two of the source repositories carry no licence file, so redistribution of
their text is not assumed to be permitted. The gold dataset is an annotation
layer over that third-party text: the labels are ours, the text is not. See
[`../DATA_LICENSES.md`](../DATA_LICENSES.md).

If the gold set is later released, the likely form is labels-only — item ids,
labels and source identifiers, with no text.
