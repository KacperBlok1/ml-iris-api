from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "iris_rf.joblib"

iris_dataset = load_iris()
target_names = iris_dataset.target_names

app = FastAPI(title="Iris ML API")  # <- musi być na poziomie modułu


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class PredictionResponse(BaseModel):
    predicted_class: Literal["setosa", "versicolor", "virginica"]
    probabilities: dict[str, float]


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run 'python train_model.py' first."
        )
    return load(MODEL_PATH)


model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    try:
        X = [[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]]
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    predicted_class = target_names[pred]
    probs = {
        target_names[i]: float(proba[i])
        for i in range(len(target_names))
    }

    return PredictionResponse(
        predicted_class=predicted_class,
        probabilities=probs,
    )
