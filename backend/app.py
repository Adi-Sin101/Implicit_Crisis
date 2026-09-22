"""Serve the existing UI and inference for the two frozen model artifacts."""
from __future__ import annotations

import logging
import csv
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
TFIDF_VECTORIZER_PATH = ROOT_DIR / "models" / "tfidf_logistic_regression" / "tfidf_vectorizer.joblib"
TFIDF_CLASSIFIER_PATH = ROOT_DIR / "models" / "tfidf_logistic_regression" / "logistic_regression.joblib"
BERT_MODEL_PATH = ROOT_DIR / "models" / "bert_no_severity"
LABELS = {0: "no_crisis", 1: "implicit_crisis", 2: "explicit_crisis"}
MAX_LENGTH = 256  # Matches the current BERT training notebook.
logger = logging.getLogger(__name__)
EVALUATION_GRAPHS = {
    "overall_metrics.png",
    "per_class_f1.png",
    "confidence_distribution.png",
}


class PredictionRequest(BaseModel):
    text: str = Field(..., description="Text to classify")

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value


def _prediction(label: int, probabilities: dict[str, float]) -> dict[str, Any]:
    return {"label": label, "class_name": LABELS[label], "probabilities": probabilities}


class FrozenModels:
    """Loads immutable artifacts once and exposes prediction-only operations."""

    def __init__(self) -> None:
        self.vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
        self.classifier = joblib.load(TFIDF_CLASSIFIER_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH, local_files_only=True)
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            BERT_MODEL_PATH, local_files_only=True
        ).to(self.device)
        if self.device.type == "cpu":
            self.bert.float()
        self.bert.eval()

    def predict_tfidf(self, text: str) -> dict[str, Any]:
        features = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(features)[0]
        classes = [int(value) for value in self.classifier.classes_]
        probability_map = {LABELS[label]: float(probability) for label, probability in zip(classes, probabilities)}
        label = int(self.classifier.predict(features)[0])
        return _prediction(label, probability_map)

    def predict_bert(self, text: str) -> dict[str, Any]:
        encoded = self.tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.bert(**encoded).logits, dim=-1)[0].cpu().tolist()
        label = max(range(len(probabilities)), key=probabilities.__getitem__)
        probability_map = {LABELS[index]: float(probability) for index, probability in enumerate(probabilities)}
        return _prediction(label, probability_map)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.models = FrozenModels()
        app.state.load_error = None
        logger.info("Frozen TF-IDF and BERT artifacts loaded on %s", app.state.models.device)
    except Exception as error:
        app.state.models = None
        app.state.load_error = str(error)
        logger.exception("Unable to load frozen model artifacts")
    yield


app = FastAPI(title="Implicit Crisis Inference API", version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(ROOT_DIR / "index.html")


@app.get("/evaluation/{filename}", include_in_schema=False)
def evaluation_graph(filename: str) -> FileResponse:
    """Expose only the three vetted graphs used by the information view."""
    if filename not in EVALUATION_GRAPHS:
        raise HTTPException(status_code=404, detail="Evaluation graph not found")
    return FileResponse(ROOT_DIR / "results" / "model_comparison" / filename)


@app.get("/project-info")
def project_info() -> dict[str, object]:
    """Read current frozen-split counts for the UI; no model inference occurs here."""
    split_dir = ROOT_DIR / "data" / "final_datasets" / "splits"
    counts = {}
    for split in ("train", "validation", "test"):
        with (split_dir / f"{split}.csv").open(encoding="utf-8", newline="") as file:
            counts[split] = sum(1 for _ in csv.DictReader(file))
    return {"splits": counts, "total": sum(counts.values()), "input_field": "content", "target_field": "label", "classes": 3}


@app.get("/health")
def health() -> dict[str, str]:
    if app.state.models is None:
        raise HTTPException(status_code=503, detail="Required model artifacts failed to load")
    return {"status": "ok", "tfidf": "loaded", "bert": "loaded"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    models: FrozenModels | None = app.state.models
    if models is None:
        raise HTTPException(status_code=503, detail="Prediction service is unavailable")
    try:
        return {"text": request.text, "tfidf": models.predict_tfidf(request.text), "bert": models.predict_bert(request.text)}
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction could not be completed") from None
