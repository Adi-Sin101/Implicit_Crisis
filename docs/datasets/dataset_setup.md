# Dataset setup

`data/raw/` is gitignored. After cloning this repository it is empty, and the
pipeline will not run until the external corpora are placed at the paths below.
Those paths are the ones declared in [`configs/paths.yaml`](../../configs/paths.yaml);
if you put the data elsewhere, edit that file rather than the code.

Read [`DATA_LICENSES.md`](../../DATA_LICENSES.md) before downloading. Two of
these repositories carry no licence file.

## Expected layout

```
data/raw/
├── CAMS/
│   ├── dataset (7).csv
│   └── CAMS/data/
│       ├── added_CAMS_data.csv
│       ├── IntentSDCNL_Training.csv
│       └── IntentSDCNL_Testing.csv
├── IRF/
│   ├── train_data.csv
│   ├── val_data.csv
│   └── test_data.csv
├── SDCNL/
│   └── data/
│       ├── combined-set.csv
│       ├── training-set.csv
│       └── testing-set.csv
└── other/
    ├── Suicide_Detection.csv
    └── goemotions/data/
        ├── train.tsv
        ├── dev.tsv
        ├── test.tsv
        └── emotions.txt
```

## 1. CAMS

```bash
git clone https://github.com/drmuskangarg/CAMS.git data/raw/CAMS
```

The main file ships as `dataset (7).csv` — keep the name as released, since
`configs/paths.yaml` refers to it.

## 2. IRF

```bash
git clone https://github.com/drmuskangarg/Irf.git data/raw/IRF
```

Note the directory is renamed to uppercase `IRF` here for consistency.

## 3. SDCNL

```bash
git clone https://github.com/ayaanzhaque/SDCNL.git data/raw/SDCNL
```

The upstream ZIP unpacks as `SDCNL-main/`; rename it to `SDCNL/`.

## 4. Komati — Suicide and Depression Detection

Download from Kaggle (account required):

```
https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch
```

Place `Suicide_Detection.csv` at `data/raw/other/Suicide_Detection.csv`
(~160 MB, 232,074 rows). Record which release version you downloaded — row
counts differ between releases.

## 5. GoEmotions

The curated splits live inside the (very large) google-research monorepo. A
sparse clone avoids downloading the whole thing:

```bash
git clone --filter=blob:none --sparse \
    https://github.com/google-research/google-research.git data/raw/google-research
cd data/raw/google-research
git sparse-checkout set goemotions
```

Then copy `goemotions/data/` and `goemotions/README.md` to
`data/raw/other/goemotions/`, or point `configs/paths.yaml` at the clone.

If you already have a full clone with no working tree checked out, the data can
be extracted straight from the object store without touching the checkout:

```bash
cd data/raw/google-research
git archive -o /tmp/ge.tar HEAD:goemotions/ data README.md
tar -xf /tmp/ge.tar -C ../other/goemotions
```

Only `goemotions/` is needed. Nothing else in that repository is used.

## Verifying the setup

```bash
python scripts/preprocessing/clean_datasets.py
```

The script prints per-source row counts as it loads. If a path is wrong you
will get a clear `FileNotFoundError` naming the missing file.
