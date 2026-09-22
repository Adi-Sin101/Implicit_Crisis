# Current model comparison

- Test CSV: `data\final_datasets\splits\test.csv`
- Test samples evaluated: **195**
- Input: `content`; target: `label`
- Mapping: `0=no_crisis`, `1=implicit_crisis`, `2=explicit_crisis`
- Inference: the same `backend.app.FrozenModels` methods used by the frontend API.

## Overall metrics

| model | accuracy | precision_macro | recall_macro | f1_macro |
|---|---:|---:|---:|---:|
| tfidf | 0.7128 | 0.7007 | 0.6979 | 0.6962 |
| bert | 0.7692 | 0.7510 | 0.7444 | 0.7466 |

## Per-class metrics

| model | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ('tfidf', 'no_crisis') | 0.7045 | 0.5849 | 0.6392 | 53.0000 |
| ('tfidf', 'implicit_crisis') | 0.6316 | 0.7347 | 0.6792 | 49.0000 |
| ('tfidf', 'explicit_crisis') | 0.7660 | 0.7742 | 0.7701 | 93.0000 |
| ('bert', 'no_crisis') | 0.7660 | 0.6792 | 0.7200 | 53.0000 |
| ('bert', 'implicit_crisis') | 0.6538 | 0.6939 | 0.6733 | 49.0000 |
| ('bert', 'explicit_crisis') | 0.8333 | 0.8602 | 0.8466 | 93.0000 |

## Confidence statistics

| model | mean | std | min | median | max |
|---|---:|---:|---:|---:|---:|
| tfidf_confidence | 0.5574 | 0.1363 | 0.3436 | 0.5227 | 0.9593 |
| bert_confidence | 0.9460 | 0.1014 | 0.5355 | 0.9928 | 0.9975 |

## Agreement

- Agreement: **148/195 (75.9%)**
- Disagreement: **47/195 (24.1%)**

## Calibration

- TF-IDF expected calibration error: 0.1554
- BERT expected calibration error: 0.1767

Higher confidence is not interpreted as better predictive performance. Performance, confidence, and calibration are reported separately.
