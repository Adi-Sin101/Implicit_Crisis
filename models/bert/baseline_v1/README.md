# BERT Baseline v1 checkpoint — frozen

`best_model/` is the epoch-3 checkpoint selected on validation macro F1 for
BERT Baseline v1 (test macro F1 0.5534, implicit_crisis recall 0.1220).

**Do not replace this with a future checkpoint.** `scripts/models/train_bert.py`
writes to `models/bert/best_model/` by default; that path is the live working
copy and will be overwritten by the next run. This directory is the preserved
one.

Weights are gitignored (the repo-root `/models/` rule). Metadata and results are
committed under `data/results/bert/baseline_v1/`.
