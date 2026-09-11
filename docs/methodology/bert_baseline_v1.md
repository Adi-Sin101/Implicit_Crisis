# BERT Baseline v1 — frozen reproducibility reference

**Status: frozen. Do not modify, overwrite or re-run into these paths.**

Baseline v1 is the first fine-tuning run of `bert-base-uncased` on this
project's gold splits. It was run on CPU on 2026-09-11 and archived unchanged on
2026-09-12. Every later BERT experiment — different epochs, class weighting, a
GPU run, a different encoder — must be written to a **new** directory and
compared *against* this record, never merged into it.

| Artifact | Location |
| --- | --- |
| Results, metrics, predictions, log | `data/results/bert/baseline_v1/` |
| Machine-readable record of everything below | `data/results/bert/baseline_v1/baseline_v1_metadata.json` |
| Selected checkpoint (epoch 3) | `models/bert/baseline_v1/best_model/` |
| Training script | `scripts/models/train_bert.py` |
| Pipeline module | `src/models/bert/pipeline.py` |
| Archive/verify script | `scripts/models/archive_bert_baseline_v1.py` |

`data/results/bert/*.json|csv|txt` (outside `baseline_v1/`) and
`models/bert/best_model/` are the **live working copies** of the same run. A
later run of `train_bert.py` will overwrite those; the `baseline_v1/` copies are
the authoritative ones. To confirm the archive is still intact:

```bash
python scripts/models/archive_bert_baseline_v1.py --verify-only
```

The script hashes every archived file and refuses to overwrite an archived
artifact whose content differs from the live one, so Baseline v1 cannot be
silently redefined by a future experiment.

## Why "Baseline v1"

It is the reference point, not a result to be improved in place:

- one model, one configuration, one seed, one run;
- no hyperparameter search, no class weighting, no resampling, no augmentation,
  no threshold tuning;
- the canonical frozen split was used as-is, never regenerated;
- the test split was untouched until after checkpoint selection.

Anything that changes any of those is a different experiment and gets its own
version.

## Dataset

`data/gold/gold.csv`, 1,987 records — explicit_crisis 825, implicit_crisis 271,
hard_negative 249, non_crisis 642. Not modified by this experiment.

**Annotation status.** These labels are **AI-assisted annotations** produced
under the project's written guidelines (v2.1). They are *not* fully
human-validated gold labels, and the dataset must not be described as
human-annotated or human-validated. Per
`data/gold/final_gold_dataset_report.json` the composition is 1,837 AI-assisted
main annotations plus 150 calibration records that went through a
dual-annotation and adjudication pass. Every result below inherits whatever
label noise that process left behind.

### Canonical split (seed 42, stratified on label, not regenerated)

| Split | File | Records | explicit | implicit | hard_neg | non_crisis |
| --- | --- | --- | --- | --- | --- | --- |
| train | `data/gold/splits/train.csv` | 1,391 | 577 | 189 | 175 | 450 |
| validation | `data/gold/splits/validation.csv` | 298 | 124 | 41 | 37 | 96 |
| test | `data/gold/splits/test.csv` | 298 | 124 | 41 | 37 | 96 |

SHA-256 of each split file is recorded in `baseline_v1_metadata.json`, so any
future change to the splits is detectable.

### Model input

`text` only. `confidence`, `notes`, `source`, `stratum`, candidate reasons and
all QC/annotation provenance are validated before training and then dropped —
none of them reaches the tokenizer.

### Label mapping (fixed, never inferred)

`explicit_crisis=0`, `implicit_crisis=1`, `hard_negative=2`, `non_crisis=3`.

## Model and training configuration

| Setting | Value |
| --- | --- |
| Model / tokenizer | `bert-base-uncased`, standard sequence-classification head, 4 labels |
| Max length | 256, truncation enabled |
| Padding | dynamic (per-batch, via `DataCollatorWithPadding`) |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size | 16 |
| Gradient accumulation | 1 (effective batch 16) |
| Optimizer | AdamW |
| Weight decay | 0.01 (excluded on bias and LayerNorm) |
| Schedule | linear, 10% warmup, 261 total steps (87 per epoch) |
| Gradient clipping | 1.0 |
| Seed | 42 (Python, `PYTHONHASHSEED`, NumPy, PyTorch, DataLoader generator) |
| Checkpoint selection | validation macro F1, evaluated after every epoch |

A plain PyTorch loop is used instead of `transformers.Trainer` so that seeding,
checkpoint selection and the epoch history are explicit and stable across
Transformers major versions.

## Hardware and software

**CPU-only training.** The machine has no NVIDIA CUDA GPU
(`torch.cuda.is_available() == False`; the only display adapter is AMD Radeon
integrated graphics). This is a property of the run, not a configuration choice:
the training script selects CUDA automatically wherever it is available.

- AMD Ryzen 5 5500U, 6 cores / 12 logical processors, ~15.4 GB RAM
- Windows 11 (10.0.26200)
- Python 3.13.7, torch 2.14.0+cpu, transformers 5.17.0, scikit-learn 1.9.0,
  numpy 2.3.5

Batch size 16 was kept as specified — memory was never the limiting factor on
CPU, only speed (~22 s per optimizer step).

## Training

| Epoch | Train loss | Val loss | Val accuracy | Val macro F1 |
| --- | --- | --- | --- | --- |
| 1 | 1.2142 | 1.0869 | 0.5940 | 0.4101 |
| 2 | 0.8794 | 0.8647 | 0.7013 | 0.5171 |
| 3 | 0.7013 | 0.8338 | 0.7047 | **0.5395** |

Best epoch: **3**, selected on validation macro F1 (never on test).
Total duration: 6,063 s ≈ 1 hour 41 minutes.

## Test results (298 held-out records)

| Metric | Value |
| --- | --- |
| Accuracy | 0.6980 |
| Macro precision | 0.6844 |
| Macro recall | 0.5478 |
| **Macro F1** | **0.5534** |
| Weighted F1 | 0.6569 |
| Errors | 90 / 298 |

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| explicit_crisis | 0.7208 | 0.8952 | 0.7986 | 124 |
| implicit_crisis | 0.6250 | 0.1220 | 0.2041 | 41 |
| hard_negative | 0.7222 | 0.3514 | 0.4727 | 37 |
| non_crisis | 0.6695 | 0.8229 | 0.7383 | 96 |

Confusion matrix (rows = gold, columns = predicted):

| | E | I | H | N |
| --- | --- | --- | --- | --- |
| **gold E** (explicit_crisis) | 111 | 0 | 2 | 11 |
| **gold I** (implicit_crisis) | 15 | 5 | 0 | 21 |
| **gold H** (hard_negative) | 17 | 0 | 13 | 7 |
| **gold N** (non_crisis) | 11 | 3 | 3 | 79 |

## Implicit-crisis slice — the headline weakness

Support 41 · precision 0.6250 · **recall 0.1220** · F1 0.2041. Only 8 test
records in total were predicted `implicit_crisis`, of which 5 were correct.

| Actual implicit_crisis predicted as | n | Rate |
| --- | --- | --- |
| explicit_crisis | 15 | 36.6% |
| implicit_crisis | 5 | 12.2% |
| hard_negative | 0 | 0.0% |
| non_crisis | 21 | 51.2% |

**Baseline v1 does not work on the class this project exists to detect.** It
recovers roughly one implicit-crisis record in eight. Over half of them are
dismissed as `non_crisis` — the most consequential failure direction for a risk
detector — and another third are over-read as `explicit_crisis`. Reporting
"70% accuracy" without this table would misrepresent the model entirely.

The mirror-image failure is on hard negatives: 17 of 37 (46%) are predicted
`explicit_crisis`, consistent with the model keying on surface crisis
vocabulary rather than on intent. The two largest error blocks in the run are
therefore exactly the two failure modes the research question concerns.

Mean softmax confidence is 0.71 on correct predictions and 0.59 on errors: the
model is broadly uncertain rather than confidently wrong, which is what an
under-trained schedule on 1,391 examples looks like.

## Limitations

1. **CPU-only.** No CUDA GPU was available; this bounded how much training was
   practical.
2. **Under-trained.** Validation macro F1 was still rising at epoch 3
   (0.4101 → 0.5171 → 0.5395). Three epochs is a floor, not convergence.
3. **No tuning of any kind** — no learning-rate/epoch search, no class weights
   against the 189-example implicit_crisis training slice, no resampling, no
   augmentation, no decision-threshold adjustment.
4. **implicit_crisis recall 0.1220.** The primary research slice effectively
   fails.
5. **Hard-negative leakage into explicit_crisis** at 46%.
6. **Single seed, single run.** No variance estimate; differences against other
   models of a few F1 points are not separable from run-to-run noise.
7. **Small, AI-assisted dataset.** 1,987 records with AI-assisted labels; the
   ceiling on measurable performance is set partly by label quality.
8. Class imbalance is untouched by design: explicit_crisis is 41.5% of the data
   and implicit_crisis 13.6%.

None of these are defects to be patched inside Baseline v1. They are the agenda
for later, separately versioned experiments.

## Reproducing

```bash
pip install -r requirements-ml.txt
python scripts/models/train_bert.py            # writes the live copies, not baseline_v1/
python scripts/models/archive_bert_baseline_v1.py --verify-only
```

Exact reproduction of the floating-point results also requires the same library
versions and CPU maths; the recorded metrics, not a bit-identical rerun, are the
reference.
