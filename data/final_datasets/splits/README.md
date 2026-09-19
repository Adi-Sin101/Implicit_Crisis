# Official Dataset Split — Explicit Copies

Location: `data/final_datasets/splits/`
(relocated from `data/splits/` — a path change only; the files were moved
byte-for-byte and the data was not touched.)

These three files are **copies**, not an independent split:

```
train.csv        905 rows
validation.csv   194 rows
test.csv         195 rows
                1294 total
```

## Provenance

They were derived directly from the authoritative split:

```
experiments/tfidf_baseline/results/data_split.csv
```

by filtering its existing `split` column. No sampling, shuffling, reshuffling or
re-partitioning was performed, and no `train_test_split` call was involved.
Membership, text, labels and row order are unchanged from the source.

**The source file is the authority.** If these copies ever disagree with it, the
source wins and these files should be regenerated.

## Naming note

The source `split` column uses the value **`val`**, not `validation`.
`validation.csv` contains exactly the rows whose source split value is `val`. The
value was not renamed inside the data.

## Columns

`row_index`, `url`, `content`, `severity`, `label`, `class_name`, `split`

- `content` is the classification text — copied verbatim, never cleaned or normalised.
- `severity` (0–6) is the original label source.
- `label` (0/1/2) is the collapsed three-class target:
  `severity 0 → 0 Non-Crisis`, `severity 1 → 1 Implicit Crisis`,
  `severity 2–6 → 2 Explicit Crisis`.
- `row_index` is a unique identifier (0–1293) and is the identity key used when
  verifying these files against the source.

## Regenerate / verify

```bash
python scripts/data/export_official_splits.py
```

The script creates the files if absent, and verifies them against the source if
present — it reports differences rather than silently overwriting.

## Status

The split is **frozen**. Verified: 0 duplicates, 0 cross-split overlap, full
identity match with the source. **No new split was generated** — not when these
files were first exported, and not when they were relocated here.
See [`docs/FINAL_SPLIT_AUDIT.md`](../../../docs/FINAL_SPLIT_AUDIT.md).

## Relationship to the frozen TF-IDF model

Separately from this split organisation, the final TF-IDF model is frozen at
`analyzer=char_wb`, `ngram_range=(3,5)` (test macro F1 0.6962). That freeze is
recorded in [`configs/tfidf_final_frozen.json`](../../../configs/tfidf_final_frozen.json)
and is independent of where these split files live — moving them changes no
model, metric or split membership.
