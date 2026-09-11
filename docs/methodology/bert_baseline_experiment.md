# BERT baseline experiment

> This run is now designated **BERT Baseline v1** and is frozen. The
> authoritative record is [`bert_baseline_v1.md`](bert_baseline_v1.md), with
> archived artifacts in `data/results/bert/baseline_v1/` and the checkpoint in
> `models/bert/baseline_v1/`. This file remains as the original run write-up.
>
> Note on labels: the gold labels are AI-assisted annotations, not
> human-validated gold labels.

Fine-tuned `bert-base-uncased` on the frozen gold splits, four-way
classification. This is the **first, unoptimised baseline**: one configuration,
one run, no hyperparameter search.

Run date: 2026-09-11. Reproduce with:

```bash
pip install -r requirements-ml.txt
python scripts/models/train_bert.py
```

## Code

| File | Role |
| --- | --- |
| `src/models/bert/pipeline.py` | Label mapping, split validation, tokenisation, decoding, metrics, slice analysis |
| `scripts/models/train_bert.py` | CLI orchestrator: train → select on validation → evaluate on test → write results |
| `tests/test_bert_pipeline.py` | 26 unit tests + 5 tests over the completed run's outputs |

A plain PyTorch loop is used rather than `transformers.Trainer`, so checkpoint
selection, seeding and the epoch history are explicit and stable across
Transformers major versions (the installed version is 5.x, whose `Trainer` API
differs from the 4.x one assumed by the older `scripts/experiments/run_bert.py`).

## Data

The canonical frozen split is read, never written:

| Split | Records | explicit | implicit | hard_neg | non_crisis |
| --- | --- | --- | --- | --- | --- |
| train | 1,391 | 577 | 189 | 175 | 450 |
| validation | 298 | 124 | 41 | 37 | 96 |
| test | 298 | 124 | 41 | 37 | 96 |

Only `text` reaches the model. `confidence`, `notes`, and every piece of source,
stratum or QC provenance are validated and then dropped. Sizes, columns, label
vocabulary, blank labels/text and duplicate ids are checked before training
starts; a mismatch aborts the run rather than silently retraining on a
regenerated split.

Fixed label mapping (never inferred from the data):
`explicit_crisis=0, implicit_crisis=1, hard_negative=2, non_crisis=3`.

## Configuration

| Setting | Value |
| --- | --- |
| Model / tokenizer | `bert-base-uncased` |
| Max length | 256, truncation on, dynamic per-batch padding |
| Learning rate | 2e-5 |
| Epochs | 3 |
| Batch size | 16 (no gradient accumulation; effective batch 16) |
| Optimizer | AdamW, weight decay 0.01 (not on bias/LayerNorm) |
| Schedule | Linear with 10% warmup, 261 total steps |
| Grad clipping | 1.0 |
| Seed | 42 (Python, NumPy, PyTorch, DataLoader generator) |
| Device | CPU |

**Device note.** The machine has no NVIDIA GPU (AMD Radeon integrated graphics,
`torch.cuda.is_available() == False`), so the run is CPU-only on a Ryzen 5 5500U
(6 cores). Batch size 16 was kept — memory was never the constraint, only speed.
Wall clock: ~32-35 minutes per epoch, 6,063 s (~1h 41m) total including evaluation. The script selects CUDA
automatically when it is available.

Environment: Python 3.13.7, torch 2.14.0+cpu, transformers 5.17.0,
scikit-learn 1.9.0, numpy 2.3.5.

## Training history

| Epoch | Train loss | Val loss | Val acc | Val macro F1 | Val weighted F1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.2142 | 1.0869 | 0.5940 | 0.4101 | 0.5280 |
| 2 | 0.8794 | 0.8647 | 0.7013 | 0.5171 | 0.6394 |
| 3 | 0.7013 | 0.8338 | 0.7047 | **0.5395** | 0.6544 |

Best checkpoint: **epoch 3**, selected on validation macro F1. Validation macro
F1 was still rising at the end of the schedule, so 3 epochs is a floor rather
than a converged optimum — worth noting before any comparison is read as final.
The test split was untouched until after selection.

## Test results (298 records)

Accuracy 0.6980 · macro P 0.6844 · macro R 0.5478 · **macro F1 0.5534** ·
weighted F1 0.6569 · 90 / 298 incorrect.

| Class | Precision | Recall | F1 | Support |
| --- | --- | --- | --- | --- |
| explicit_crisis | 0.7208 | 0.8952 | 0.7986 | 124 |
| implicit_crisis | 0.6250 | 0.1220 | 0.2041 | 41 |
| hard_negative | 0.7222 | 0.3514 | 0.4727 | 37 |
| non_crisis | 0.6695 | 0.8229 | 0.7383 | 96 |

Confusion matrix (rows = gold, columns = predicted):

| | explicit | implicit | hard_neg | non_crisis |
| --- | --- | --- | --- | --- |
| **explicit_crisis** | 111 | 0 | 2 | 11 |
| **implicit_crisis** | 15 | 5 | 0 | 21 |
| **hard_negative** | 17 | 0 | 13 | 7 |
| **non_crisis** | 11 | 3 | 3 | 79 |

## Implicit crisis (primary research slice)

Support 41 · precision 0.6250 · recall 0.1220 · F1 0.2041. Only 8 test records
in total were predicted `implicit_crisis`.

| Actual implicit_crisis predicted as | n | Rate |
| --- | --- | --- |
| explicit_crisis | 15 | 36.6% |
| implicit_crisis | 5 | 12.2% |
| hard_negative | 0 | 0.0% |
| non_crisis | 21 | 51.2% |

The baseline barely learns the class. Half of all implicit-crisis records are
dismissed as `non_crisis` — the false-negative mode that matters most for this
project — and another third are over-read as `explicit_crisis`. Overall accuracy
of 0.70 conceals this entirely, which is exactly why the slice is reported
separately.

The mirror-image failure is on hard negatives: 17 of 37 (46%) are called
`explicit_crisis`, i.e. crisis vocabulary alone triggers the positive class.
The model's two largest error blocks are therefore the two failure modes the
research question is about, not random noise.

Mean softmax confidence is 0.71 on correct predictions and 0.59 on errors — the
model is not confidently wrong so much as broadly uncertain, consistent with an
under-trained 3-epoch schedule on 1,391 examples.

## Outputs

```
data/results/bert/
    bert_config.json              # full configuration + environment + library versions
    bert_training_history.json    # per-epoch losses and validation metrics
    bert_test_metrics.json        # overall, per-class, confusion matrix, implicit analysis
    bert_classification_report.json
    bert_confusion_matrix.csv
    bert_test_predictions.csv     # id, text, true_label, predicted_label, confidence, 4 probs
    bert_errors.csv               # the 90 incorrect predictions
    bert_training_log.txt         # console log of the run

models/bert/best_model/           # gitignored; selection.json records the chosen epoch
```

## Not done (deliberately)

No hyperparameter search, no augmentation, no resampling, no class weighting,
no threshold tuning, no second run. Those are follow-up experiments; this run
establishes the reference point.
