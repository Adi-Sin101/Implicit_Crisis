# Inter-annotator agreement

This directory holds the calibration and agreement record. **No agreement
figure is written here until it has actually been computed** from real
annotation files — there are no placeholder or expected numbers anywhere in
this repository.

## Workflow

```
150-item calibration subset
        ↓
independent annotation by each annotator (no discussion)
        ↓
agreement calculation
        ↓
guideline revision if agreement is weak on any class
        ↓
full annotation under the revised guidelines
        ↓
final agreement reported on the double-annotated portion
```

## Commands

Create the calibration sheets:

```bash
python scripts/annotation/create_annotation_sheet.py --annotators A B --calibration 150
```

Each annotator fills in their own copy in `data/gold/annotation/`. Then:

```bash
python scripts/annotation/calculate_agreement.py \
    --files A=data/gold/annotation/calibration_150_A.csv \
            B=data/gold/annotation/calibration_150_B.csv
```

## What lands here

| File | Content |
|---|---|
| `pairwise_kappa.csv` | Cohen's kappa and raw agreement per annotator pair |
| `per_class_kappa.csv` | one-vs-rest kappa per label |
| `disagreement_matrix.csv` | confusion between two annotators |
| `revision_log.md` | what changed in the guidelines, and why (write by hand) |

Cohen's kappa is used for two annotators, Fleiss' kappa for three or more.
Landis & Koch bands are printed as description only — they are not a pass mark,
and a weak figure is reported rather than chased.

The expected weak point is the `implicit_crisis` / `non_crisis` boundary. If
per-class kappa is low there specifically, revise §2 of the guidelines with
concrete examples drawn from the actual disagreements before annotating more.

## Calibration data

Calibration sheets live in `data/gold/annotation/` (gitignored, since they
contain verbatim corpus text). Only the computed statistics are committed here.
