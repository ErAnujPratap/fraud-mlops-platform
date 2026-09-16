"""Train XGBoost, track with MLflow, and promote only if it beats the current model."""
import json
from pathlib import Path
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.make_dataset import FEATURES, OUT as DATA

MODEL_DIR = Path("models")
PARAMS = {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.1, "eval_metric": "aucpr"}


def main() -> None:
    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["Class"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    params = {**PARAMS, "scale_pos_weight": float((y_tr == 0).sum() / (y_tr == 1).sum())}

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run():
        model = XGBClassifier(**params).fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "pr_auc": average_precision_score(y_te, proba),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
        }
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model", registered_model_name="fraud-xgb")
        print(json.dumps(metrics, indent=2))

        # Model gate: promote only if PR-AUC improves
        MODEL_DIR.mkdir(exist_ok=True)
        current = MODEL_DIR / "metrics.json"
        best = json.loads(current.read_text())["pr_auc"] if current.exists() else 0.0
        if metrics["pr_auc"] > best:
            joblib.dump(model, MODEL_DIR / "model.joblib")
            current.write_text(json.dumps(metrics, indent=2))
            mlflow.set_tag("promoted", "true")
            print(f"PROMOTED: pr_auc {best:.4f} -> {metrics['pr_auc']:.4f}")
        else:
            mlflow.set_tag("promoted", "false")
            print(f"NOT promoted: {metrics['pr_auc']:.4f} <= current {best:.4f}")


if __name__ == "__main__":
    main()
