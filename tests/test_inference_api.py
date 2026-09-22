"""Smoke tests for inference-only loading of the two shipped model artifacts."""
from fastapi.testclient import TestClient

from backend.app import LABELS, app


def test_health_and_prediction_contract():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        response = client.post("/predict", json={"text": "I feel exhausted and cannot see a way forward."})
        assert response.status_code == 200
        payload = response.json()
        for name in ("tfidf", "bert"):
            result = payload[name]
            assert result["label"] in LABELS
            assert result["class_name"] == LABELS[result["label"]]
            assert set(result["probabilities"]) == set(LABELS.values())
            assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


def test_blank_text_is_rejected():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
