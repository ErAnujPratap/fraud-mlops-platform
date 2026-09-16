from fastapi.testclient import TestClient
from src.data.make_dataset import FEATURES
from src.serving.app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_predict():
    r = client.post("/predict", json={"features": {f: 0.0 for f in FEATURES}})
    assert r.status_code == 200
    assert r.json()["label"] in {"fraud", "legit"}


def test_missing_features():
    assert client.post("/predict", json={"features": {"Time": 1.0}}).status_code == 422
