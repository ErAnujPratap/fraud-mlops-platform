"""FastAPI model server with Prometheus metrics."""
import os
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.data.make_dataset import FEATURES

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))

app = FastAPI(title="Fraud Detection API", version="0.1.0")
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

PREDICTIONS = Counter("predictions_total", "Predictions served", ["label"])
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
SCORES = Histogram("fraud_score", "Distribution of fraud scores",
                   buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])


class Transaction(BaseModel):
    features: dict[str, float] = Field(..., description="Time, Amount, V1..V28")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(tx: Transaction):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    missing = set(FEATURES) - set(tx.features)
    if missing:
        raise HTTPException(422, f"Missing features: {sorted(missing)}")
    start = time.perf_counter()
    score = float(model.predict_proba(pd.DataFrame([tx.features])[FEATURES])[0, 1])
    LATENCY.observe(time.perf_counter() - start)
    SCORES.observe(score)
    label = "fraud" if score >= THRESHOLD else "legit"
    PREDICTIONS.labels(label).inc()
    return {"fraud_score": round(score, 4), "label": label}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
