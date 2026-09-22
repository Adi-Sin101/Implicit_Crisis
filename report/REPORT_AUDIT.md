# Report Audit — CrisisSense Final Academic Project Report

## Current source files inspected

- `README.md`
- `New_docs/CRISISSENSE_CURRENT_PROJECT.md`
- `backend/app.py`
- `backend/__init__.py`
- `models/tfidf_logistic_regression/config.json`
- `models/bert_no_severity/config.json`
- `data/final_datasets/splits/README.md`
- `data/final_datasets/splits/{train,validation,test}.csv` (row counts and structure verified; content not reproduced)
- `scripts/evaluate_current_models.py`
- `requirements.txt`, `requirements-ml.txt`
- `tests/test_inference_api.py`
- `New_docs/architecture/*.png` (five architecture diagrams)
- `New_docs/dr_bot_report_final (1).pdf` (structural/style reference only — no technical content reused)

## Result files used

- `results/model_comparison/metrics.json`
- `results/model_comparison/metrics.csv`
- `results/model_comparison/per_class_metrics.csv`
- `results/model_comparison/predictions.csv` (195 rows; used for aggregate error analysis only — no individual message text was reproduced in the report)
- `results/model_comparison/evaluation_summary.md`

## Figures included

All ten current evaluation figures from `results/model_comparison/` and all five architecture diagrams from `New_docs/architecture/`:

`overall_metrics.png`, `per_class_f1.png`, `confidence_distribution.png`, `confusion_matrix_tfidf.png`, `confusion_matrix_bert.png`, `confidence_vs_correctness.png`, `model_agreement.png`, `agreement_by_class.png`, `calibration.png`, `implicit_crisis_metrics.png`, `system-architecture.png`, `artifact-lifecycle.png`, `model-comparison.png`, `dataset-evaluation.png`, `live-prediction-sequence.png`.

All figures were copied as-is into `report/figures/` at their original resolution; none were redrawn or altered.

## Current dataset size

- Total: 1,294 rows (Train 905 / Validation 194 / Test 195).
- Class distribution: No Crisis 351, Implicit Crisis 324, Explicit Crisis 619 (total across all splits).
- All evaluation numbers in the report are computed on the shared 195-row `test.csv`.

## Current model names

- Character-level TF-IDF (`char_wb`, n-grams 3–5) + Logistic Regression (`models/tfidf_logistic_regression/`).
- BERT sequence classifier, `BertForSequenceClassification` (12 layers, hidden 768, 12 heads, 3 labels) (`models/bert_no_severity/`).
- Word2Vec is explicitly documented as **not used** in the current system (Section 6, "Word2Vec Usage" omitted from methodology since it is not part of the current pipeline; the boundary is stated in Chapter 1's scope section and reinforced in Chapter 6).

## Current metrics (verified against `metrics.json` / `per_class_metrics.csv`)

| Metric | TF-IDF + LR | BERT |
|---|---:|---:|
| Accuracy | 71.28% | 76.92% |
| Macro Precision | 70.07% | 75.10% |
| Macro Recall | 69.79% | 74.44% |
| Macro F1 | 69.62% | 74.66% |
| Implicit Crisis F1 | 67.92% | 67.33% |
| Mean confidence | 55.74% | 94.60% |
| ECE | 0.1554 | 0.1767 |
| Agreement | 148/195 (75.9%) | — |

All values above were copied directly from the JSON/CSV source files, not retyped from memory or from README prose alone.

## Historical four-class material — exclusion confirmed

The report explicitly and repeatedly states that CrisisSense is a three-class system and that historical four-class, `hard_negative`-labeled, severity-graded, and Word2Vec-based research material retained elsewhere in the repository (`src/models/bert/`, `scripts/models/`, `annotation/`, older BERT training modules) is out of scope and not used by the current frontend, backend, frozen splits, or frozen models. This boundary statement appears in the Introduction (Scope), Dataset (Label Definitions), and Implementation (current-system boundary) chapters. A post-hoc grep of the compiled report source confirmed no instance of "four-class" or "hard_negative" is used to describe the current system — all instances explicitly exclude that material.

## Result-interpretation guardrails confirmed

- The report does not state that BERT has higher implicit-crisis F1; it explicitly states TF-IDF is marginally ahead on that one class while BERT leads on every other metric.
- The report does not describe CrisisSense as a clinical or diagnostic tool anywhere; the title page, abstract, scope, and limitations chapters all carry an explicit disclaimer.
- Confidence and calibration are presented as distinct from accuracy throughout, with BERT's higher confidence explicitly paired with its higher (worse) ECE.

## PDF compilation

- Toolchain: MiKTeX (installed via `winget install MiKTeX.MiKTeX` during this session) with `pdflatex` + `bibtex`, three `pdflatex` passes plus one `bibtex` pass (standard LaTeX build sequence).
- Final build: 61 pages, all cross-references and citations resolved (zero "undefined reference" or "undefined citation" warnings in the final pass).
- Remaining LaTeX warnings after fixes: 9 minor `Overfull \hbox` warnings, each under 1 inch and confined to inline code/path text in tables and lists; visually inspected via rendered page images and found not to cause any visible clipping, margin overrun, or broken layout in the sampled pages (title page, table of contents, results chapter, confusion-matrix page, confidence-vs-correctness page, limitations chapter, reproducibility appendix).
- No absolute Windows paths appear in any `.tex` source file (verified by grep before compilation).

## Final deliverables

- `report/CrisisSense_Final_Report.pdf` (61 pages)
- `report/main.tex`
- `report/references.bib`
- `report/figures/` (15 current figures)
- `report/sections/` (14 section source files: abstract through appendix)
- `report/REPORT_AUDIT.md` (this file)
