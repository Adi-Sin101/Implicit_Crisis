# CrisisSense — Complete Current Project Documentation

**Crisis Language Detection Using NLP**

**Status:** Frozen three-class inference system with a static frontend, FastAPI API, two saved models, and a common 195-row evaluation set.  
**Last audited:** 2026-09-23  
**Purpose:** The single source of truth for the current deployed CrisisSense project. It separates the active three-class system from older research material retained in the repository.

## 1. What CrisisSense does

CrisisSense classifies crisis-related language in text. It runs two independent models on each submitted text and displays each model’s predicted class and selected-class probability. The models are compared side by side; they are **not an ensemble**.

| ID | Class | Current meaning |
|---:|---|---|
| 0 | `no_crisis` | Text that does not indicate crisis-related language under the current classification scheme. |
| 1 | `implicit_crisis` | Crisis-related meaning expressed indirectly rather than as a direct statement. |
| 2 | `explicit_crisis` | Crisis-related language expressed directly. |

CrisisSense is a three-class crisis-language classifier, not an implicit-only detector and not a clinical diagnostic system.

## 2. Current implementation boundary

The authoritative current runtime paths are:

- `index.html` — current CrisisSense frontend.
- `backend/app.py` — current FastAPI inference API.
- `data/final_datasets/splits/` — current frozen split copies.
- `models/tfidf_logistic_regression/` — current TF-IDF/LR artifacts.
- `models/bert_no_severity/` — current BERT artifact.
- `results/model_comparison/` — current shared-test results and figures.

The repository also retains earlier four-class gold-data, annotation, notebook, and BERT modules. Those use labels such as `hard_negative`, different split sizes, or other artifact paths. They are historical/research materials, **not used by the current frontend, backend, frozen split, or frozen models**.

## 3. Branding and frontend

`index.html` implements the current identity:

- Browser title: `CrisisSense | Crisis Language Detection Using NLP`
- Header: **CrisisSense**
- Subtitle: **Crisis Language Detection Using NLP**
- Live area: “Analyze crisis-related language”
- Information view: “About CrisisSense”

It is a single static HTML page served by FastAPI. Tailwind is loaded from its CDN; no separate local CSS, JavaScript, image, or build asset is required for the current UI.

The live view accepts text and calls `POST /predict`. It displays BERT and TF-IDF/LR predictions, selected-class confidence, and whether their numeric labels agree. The Information button switches to an in-page briefing containing the classes, split counts, model comparison, confidence/agreement caveats, evaluation setup, and three presentation graphs. “Try Live Prediction” returns to the live view without reload or retraining.

## 4. Current file map

| Category | Path | Purpose |
|---|---|---|
| Frontend | `index.html` | CrisisSense interface, live requests, result rendering, Information section. |
| Backend | `backend/app.py` | FastAPI app, frozen model loading, endpoints, and local graph serving. |
| Backend package | `backend/__init__.py` | Backend package marker. |
| Train split | `data/final_datasets/splits/train.csv` | Frozen 905-row training copy. |
| Validation split | `data/final_datasets/splits/validation.csv` | Frozen 194-row validation copy. |
| Test split | `data/final_datasets/splits/test.csv` | Frozen 195-row shared evaluation input. |
| Split notes | `data/final_datasets/splits/README.md` | Provenance and frozen-split verification notes. |
| Split export/audit | `scripts/data/export_official_splits.py` | Verifies/exports explicit copies; does not resample or overwrite an existing differing file. |
| TF-IDF vectorizer | `models/tfidf_logistic_regression/tfidf_vectorizer.joblib` | Frozen fitted TF-IDF vectorizer. |
| LR classifier | `models/tfidf_logistic_regression/logistic_regression.joblib` | Frozen fitted Logistic Regression classifier. |
| TF-IDF config | `models/tfidf_logistic_regression/config.json` | Final data, split, vectorizer, classifier, and test settings. |
| Freeze record | `configs/tfidf_final_frozen.json` | Declares final TF-IDF/LR settings frozen. |
| BERT weights | `models/bert_no_severity/model.safetensors` | Frozen BERT weights, tracked through Git LFS. |
| BERT tokenizer | `models/bert_no_severity/tokenizer.json`, `tokenizer_config.json` | Local tokenizer data/configuration. |
| BERT config | `models/bert_no_severity/config.json` | Saved `BertForSequenceClassification` architecture config with three outputs. |
| Evaluation runner | `scripts/evaluate_current_models.py` | Evaluates current frozen models through the same `FrozenModels` inference path. |
| Results | `results/model_comparison/` | Current metrics, prediction table, summary, and figures. |
| Inference test | `tests/test_inference_api.py` | Checks health, prediction response/probabilities, and blank-input rejection. |
| Dependencies | `requirements.txt`, `requirements-ml.txt` | Core/FastAPI and BERT runtime dependencies. |

## 5. Current dataset

The current model input field is `content`; the target is `label`. The current split CSVs contain:

`row_index`, `url`, `content`, `label`, `class_name`, `split`

| Split | Rows | No crisis (0) | Implicit crisis (1) | Explicit crisis (2) |
|---|---:|---:|---:|---:|
| Train | 905 | 245 | 227 | 433 |
| Validation | 194 | 53 | 48 | 93 |
| Test | 195 | 53 | 49 | 93 |
| **Total** | **1,294** | **351** | **324** | **619** |

The audit found no blank `content` or `label` values in these files. The split README and export script identify them as frozen explicit copies of `experiments/tfidf_baseline/results/data_split.csv`. Its `train`, `val`, and `test` values map to the three exported filenames. The split audit records zero duplicate rows/content and zero cross-split overlap. Both models use the same 195-row `test.csv`.

### Current label construction

The final TF-IDF configuration records the current mapping from original `severity` values:

- severity 0 → label 0 / No Crisis
- severity 1 → label 1 / Implicit Crisis
- severity 2–6 → label 2 / Explicit Crisis

Detailed original annotation provenance for the deployed three-class set is **not verified from the current runtime implementation alone**. Older documents describe a separate four-class annotation design; that must not be read as the current model definition.

## 6. Preprocessing and representation

### TF-IDF + Logistic Regression

For live inference, `backend/app.py` sends submitted text directly to the saved vectorizer through `transform([text])`. No separate runtime call to `src/data/cleaning` or another custom text-normalization function occurs.

| Frozen vectorizer setting | Value |
|---|---|
| Analyzer | `char_wb` (character n-grams within word boundaries) |
| N-gram range | (3, 5) |
| `min_df` | 2 |
| `max_df` | 0.9 |
| `sublinear_tf` | true |
| `max_features` | 50,000 |
| Lowercase | true |
| Accent stripping | unicode |
| Fitted features | 17,708 |
| Fit data | training split only |

The verified live pipeline has no separately implemented stop-word removal, stemming, lemmatization, word-token cleaning, URL removal, or manual punctuation removal. The vectorizer itself lowercases and strips Unicode accents; its `char_wb` analyzer derives character-boundary features.

The vectorizer creates a sparse TF-IDF matrix. The saved Logistic Regression has `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=5000`, and `random_state=42`. Its `predict_proba` returns all class probabilities; `predict` selects the numeric label.

```
Raw text → saved TF-IDF vectorizer → sparse numeric features
         → saved Logistic Regression → probabilities → predicted class
```

### BERT

The API loads `AutoTokenizer` and `AutoModelForSequenceClassification` locally from `models/bert_no_severity/` with `local_files_only=True`. It tokenizes each request with truncation and `max_length=256`, then runs inference under `torch.inference_mode()`. The tokenizer creates the model inputs, including token IDs and attention mask. Softmax converts logits to three probabilities; the largest one is selected.

The saved config identifies `BertForSequenceClassification`, 12 layers, hidden size 768, 12 attention heads, vocabulary size 30,522, and three output labels. Runtime output positions are mapped by `backend/app.py` to `no_crisis`, `implicit_crisis`, and `explicit_crisis`.

```
Raw text → local BERT tokenizer (truncate to 256) → token IDs + attention mask
         → BERT sequence-classification model → logits → softmax
         → three probabilities → predicted class
```

Padding behavior for an individual request is not explicitly set in `backend/app.py`; no additional padding behavior is claimed here.

## 7. Word2Vec usage

**NOT USED IN THE CURRENT PROJECT.**

- **TF-IDF:** current inference uses scikit-learn `TfidfVectorizer` with `char_wb` character n-grams. No Word2Vec model, embedding load, or gensim call exists in the current inference path.
- **Logistic Regression:** receives the sparse TF-IDF matrix, not Word2Vec embeddings.
- **BERT:** uses learned contextual transformer representations inside `BertForSequenceClassification`, not an external Word2Vec model.

Dataset notes mention third-party source repositories that may contain a `word_embeddings.py` file. Those are not executed by current CrisisSense.

## 8. Model comparison

| Property | TF-IDF + Logistic Regression | BERT |
|---|---|---|
| Input | Text | Text |
| Representation | Character-boundary TF-IDF | Contextual transformer representation |
| Classifier | Logistic Regression | BERT sequence-classification head |
| Classes | 3 | 3 |
| Artifact | Two `.joblib` files | Local weights, tokenizer, config |
| Runtime | scikit-learn CPU operations | CUDA if available; otherwise CPU |
| Output | Class + probability map | Class + probability map |
| Relationship | Independent | Independent |

TF-IDF/LR is a feature-based lexical baseline. BERT processes a sequence contextually. Their outputs are displayed together and never combined.

## 9. Backend endpoints and prediction flow

`backend/app.py` loads frozen artifacts once at FastAPI lifespan startup. A failed load is recorded and `/health` and `/predict` return HTTP 503 rather than a fabricated result.

| Endpoint | Purpose |
|---|---|
| `GET /` | Serves `index.html`. |
| `GET /health` | Reports model load status or HTTP 503. |
| `GET /project-info` | Reads current split row counts and returns totals, input/target fields, and class count. |
| `GET /evaluation/{filename}` | Serves only frontend-approved graphs: overall metrics, per-class F1, confidence distribution. |
| `POST /predict` | Accepts `{"text":"..."}`; returns independent TF-IDF and BERT results. Blank text returns HTTP 422. |

Each model object in the response has `label`, `class_name`, and `probabilities` keyed by the three class names.

```
User text
    ↓
index.html → POST /predict
    ↓
FastAPI validates text
    ↓
 ┌──────────────────────┬──────────────────────┐
 ↓                      ↓
TF-IDF + LR             BERT
 ↓                      ↓
class + confidence      class + confidence
 └──────────────────────┴──────────────────────┘
    ↓
frontend displays independent results and agreement
```

## 10. Current evaluation and exact results

`scripts/evaluate_current_models.py` evaluates the two frozen models on `data/final_datasets/splits/test.csv`, using the same `FrozenModels` methods as the API. It writes current outputs into `results/model_comparison/`.

| Metric | TF-IDF + LR | BERT |
|---|---:|---:|
| Accuracy | 71.28% | 76.92% |
| Macro Precision | 70.07% | 75.10% |
| Macro Recall | 69.79% | 74.44% |
| Macro F1 | 69.62% | 74.66% |
| Implicit Crisis F1 | 67.92% | 67.33% |

BERT has higher overall accuracy and macro F1 on this 195-row test set. This does **not** show BERT is better at implicit-crisis detection: implicit-crisis F1 is very close and TF-IDF is slightly higher on this test set.

| Class | TF-IDF F1 | BERT F1 | Test support |
|---|---:|---:|---:|
| `no_crisis` | 63.92% | 72.00% | 53 |
| `implicit_crisis` | 67.92% | 67.33% | 49 |
| `explicit_crisis` | 77.01% | 84.66% | 93 |

### Confidence, agreement, calibration

| Measure | TF-IDF + LR | BERT |
|---|---:|---:|
| Mean selected-class confidence | 55.74% | 94.60% |
| Median selected-class confidence | 52.27% | 99.28% |
| Expected calibration error | 0.1554 | 0.1767 |

Confidence is the probability of the selected class, not accuracy. BERT’s 94.60% mean confidence is not a 94.60% accuracy claim. Performance, confidence, and calibration are reported separately.

The models agree on 148/195 samples (75.9%) and disagree on 47/195 (24.1%). Agreement means both independent models selected the same class; it does not establish correctness.

## 11. Current results, graphs, and confusion matrices

All files below already exist under `results/model_comparison/`.

| File | What it contains |
|---|---|
| `metrics.csv` | Overall accuracy, macro precision, macro recall, macro F1. |
| `metrics.json` | Machine-readable overall/per-class metrics, agreement, calibration. |
| `per_class_metrics.csv` | Precision, recall, F1, support for all classes/models. |
| `predictions.csv` | 195-row ground truth, both predictions, confidences, correctness, agreement. |
| `evaluation_summary.md` | Human-readable metrics/confidence/agreement/ECE summary. |
| `overall_metrics.png` | Accuracy, macro precision, macro recall, macro F1 comparison. |
| `per_class_f1.png` | F1 by no crisis, implicit crisis, explicit crisis. |
| `confidence_distribution.png` | Distribution of selected-class confidence. |
| `confusion_matrix_tfidf.png` | Current TF-IDF/LR confusion matrix. |
| `confusion_matrix_bert.png` | Current BERT confusion matrix. |
| `confidence_vs_correctness.png` | Confidence grouped by model and correctness. |
| `model_agreement.png` | Agreement/disagreement counts. |
| `agreement_by_class.png` | Class distribution among shared predictions. |
| `calibration.png` | Observed accuracy versus confidence plus perfect-calibration line. |
| `implicit_crisis_metrics.png` | Existing current implicit-crisis metric figure. |

The confusion matrices already existed; no graph was generated for this documentation task. Rows are actual labels and columns predicted labels, ordered no crisis, implicit crisis, explicit crisis. Diagonal cells are correct predictions.

| Actual → predicted | No crisis | Implicit crisis | Explicit crisis |
|---|---:|---:|---:|
| TF-IDF actual no crisis | 31 | 9 | 13 |
| TF-IDF actual implicit crisis | 4 | 36 | 9 |
| TF-IDF actual explicit crisis | 9 | 12 | 72 |
| BERT actual no crisis | 36 | 12 | 5 |
| BERT actual implicit crisis | 4 | 34 | 11 |
| BERT actual explicit crisis | 7 | 6 | 80 |

## 12. How to run the current project

Use Python 3.10 or newer. Install the combined runtime requirements and start the single FastAPI process:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-ml.txt
git lfs pull
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/`. There is no separate frontend startup command. Enter non-empty text and select “Analyze with Both Models.” Select “Information” for the project briefing and its frontend graphs.

### Training versus inference

- **Current app/inference:** `index.html` and `backend/app.py`; loads frozen artifacts and predicts only.
- **Frozen-model evaluation:** `scripts/evaluate_current_models.py`; generates evaluation outputs but does not fit/write model artifacts.
- **Frozen TF-IDF training record:** `experiments/tfidf_baseline/run_tfidf_baseline.py`, with settings recorded in `models/tfidf_logistic_regression/config.json` and `configs/tfidf_final_frozen.json`. Do not rerun casually.
- **Older BERT training modules:** `src/models/bert/` and `scripts/models/` belong to separate four-class research paths, not the current runtime application.

## 13. GitHub/LFS and limitations

Current inference artifacts are tracked through Git/Git LFS:

- `models/tfidf_logistic_regression/tfidf_vectorizer.joblib`
- `models/tfidf_logistic_regression/logistic_regression.joblib`
- model files under `models/bert_no_severity/`

Run `git lfs pull` after cloning before starting the API. This document contains no credentials or secrets.

Verified interpretation limits:

- Results describe this frozen 195-sample test set, not universal performance.
- The test set includes 49 implicit-crisis examples.
- Higher BERT confidence is not equivalent to higher accuracy; ECE is reported separately.
- The models disagree on 24.1% of test samples.
- The system classifies text; it does not diagnose people or determine suicide risk.

## 14. Current project history

Verified milestones: frozen three-class split copies; frozen TF-IDF/LR artifacts; frozen BERT artifact; common-test evaluation; metrics, confidence, agreement, calibration, confusion matrices and graphs; CrisisSense frontend; FastAPI `/predict`; Information section; and Git LFS publication of current inference artifacts.

# Final Current-Project Audit

- [x] Current frontend identified
- [x] Current backend identified
- [x] Current dataset identified
- [x] Current models identified
- [x] Current model paths documented
- [x] Dataset split sizes verified
- [x] Class mapping verified
- [x] Preprocessing audited
- [x] TF-IDF pipeline audited
- [x] Logistic Regression pipeline audited
- [x] BERT pipeline audited
- [x] Word2Vec usage audited
- [x] Evaluation metrics verified
- [x] Confidence results verified
- [x] Agreement results verified
- [x] Calibration results verified
- [x] Current graphs identified
- [x] Confusion matrices checked
- [x] Frontend flow documented
- [x] Backend flow documented
- [x] `/predict` documented
- [x] Current file paths documented
- [x] Current GitHub artifacts documented

