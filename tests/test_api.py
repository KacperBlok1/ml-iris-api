from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_valid_sample():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in ["setosa", "versicolor", "virginica"]
    probs = body["probabilities"]
    assert all(name in probs for name in ["setosa", "versicolor", "virginica"])
    assert 0.99 <= sum(probs.values()) <= 1.01


def test_predict_invalid_input():
    payload = {
        "sepal_length": -1.0,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422  # validation error from Pydantic
