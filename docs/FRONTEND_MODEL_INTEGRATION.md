# Frontend model integration

The existing static HTML interface in `index.html` is served by a small FastAPI
application. A submitted post is sent to `POST /predict`; the API runs the two
frozen models independently and returns both results for the interface to
display. There is no ensemble or runtime dataset access.

## Models and labels

- TF-IDF + Logistic Regression: `models/tfidf_logistic_regression/tfidf_vectorizer.joblib` and `models/tfidf_logistic_regression/logistic_regression.joblib`
- BERT: `models/bert_no_severity/`
- `0` = `no_crisis`, `1` = `implicit_crisis`, `2` = `explicit_crisis`

Both artifacts are inference-only. The API loads them once at startup and never
fits, retrains, or writes to a model artifact.

## Dependencies and run

Runtime dependencies are listed in `requirements.txt` and
`requirements-ml.txt`. They were already available in the tested environment,
so no installation command was run during this integration.

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open the frontend at `http://127.0.0.1:8000/`. It is served by the same process
as the API, so no cross-origin configuration is required.

## API

- Health check: `GET http://127.0.0.1:8000/health`
- Prediction: `POST http://127.0.0.1:8000/predict`

```json
{"text":"I feel exhausted and cannot see a way forward."}
```

The response contains the original text and independent `tfidf` and `bert`
objects, each with `label`, `class_name`, and probabilities keyed by the three
class names.

## Troubleshooting

- A `503` response means an artifact could not load; inspect the server log and
  confirm the frozen model files are present.
- A blank `text` request returns a validation error; enter a non-empty post.
- If the browser cannot reach the service, start the command above and use the
  displayed `127.0.0.1:8000` URL.
