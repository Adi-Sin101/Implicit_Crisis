# Calibration Disagreement Files Audit

## Scope

Audited all CSV files in `data/gold/annotation/agreement/` whose filenames contain `calibration_150_disagreements` on 2026-09-06. No files were modified or deleted.

## Comparison

| File | Rows | Columns | SHA-256 prefix | Status / purpose |
| --- | ---: | --- | --- | --- |
| `calibration_150_disagreements.csv` | 8 | `id`, `text`, `A label`, `B label`, `A confidence`, `B confidence`, `A notes`, `B notes` | `e411e8457571812e...` | Exact duplicate; intended canonical filename |
| `calibration_150_disagreements_2.csv` | 8 | Same 8 columns | `e411e8457571812e...` | Exact duplicate; collision-safe generated copy |
| `calibration_150_disagreements_3.csv` | 8 | Same 8 columns | `e411e8457571812e...` | Exact duplicate; collision-safe generated copy |

## Findings

- All three files contain exactly 8 rows.
- All three files contain the same 8 disagreement IDs in the same order.
- All three files contain identical A labels, B labels, text, confidence values, and notes.
- All three files have identical column names and ordering.
- The files are byte-for-byte identical, not merely equivalent after parsing.
- No timestamp or other metadata columns are present in the CSV files.
- The filesystem timestamps differ because the files were created during successive comparison-output attempts; these timestamps are not annotation data.

The disagreement cases are:

| ID | A label | B label |
| --- | --- | --- |
| `cams-24c0c25e6a33` | `explicit_crisis` | `implicit_crisis` |
| `cams-687449dcc3c5` | `non_crisis` | `implicit_crisis` |
| `irf-5f27a24f37fa` | `implicit_crisis` | `explicit_crisis` |
| `irf-a11b2a6bb141` | `non_crisis` | `implicit_crisis` |
| `komati-c0f53895b28d` | `hard_negative` | `non_crisis` |
| `komati-e335a390c977` | `explicit_crisis` | `non_crisis` |
| `sdcnl-9a7998c1beff` | `non_crisis` | `implicit_crisis` |
| `sdcnl-ca2a4047d113` | `non_crisis` | `implicit_crisis` |

## Recommendation

Treat `calibration_150_disagreements.csv` as the canonical disagreement file for the current calibration-150 A-versus-B comparison. It has the stable, unsuffixed name expected by the workflow and is byte-for-byte identical to `_2` and `_3`.

The `_2` and `_3` files are redundant output copies, not different disagreement types or versions. They should not be deleted automatically. Keep them until the project owner explicitly decides which redundant files to remove.

## Audit method

The audit used a programmatic comparison of each CSV's file bytes, parsed row count, column list, IDs, labels, and complete DataFrame contents. The source annotation files and all three disagreement files were left unchanged.
