# Fraud MLOps Platform

End-to-end MLOps pipeline for payment fraud detection.

## Architecture (v0.1)
```
data (Kaggle/synthetic) -> validate -> train (XGBoost) -> MLflow tracking + registry
                                           |
                                      model gate (PR-AUC)
                                           |
                              FastAPI  ->  Docker  ->  Trivy scan
                                  |
                          /metrics -> Prometheus -> Grafana
```

## Quick start
```bash
make setup && source .venv/bin/activate
make train          # validates data, trains, logs to MLflow, applies model gate
mlflow ui           # http://localhost:5000
make test
make serve          # http://localhost:8000/docs
```

Real data: download `creditcard.csv` from Kaggle into `data/`. Without it, a synthetic dataset with the same schema is used.

## Docker stack
```bash
make up   # API :8000, MLflow :5000, Prometheus :9090, Grafana :3000
```

## Roadmap
- [x] v0.1 Training, MLflow tracking, model gate, FastAPI, Docker, CI + Trivy
- [ ] v0.2 DVC data versioning with MinIO
- [ ] v0.3 Prefect/Kubeflow pipeline
- [ ] v0.4 Helm chart + ArgoCD on kind, KServe
- [ ] v0.5 Evidently drift detection -> auto retrain
- [ ] v0.6 cosign-signed models, Vault secrets, SBOM
- [ ] v0.7 LLMOps module (RAG + Langfuse)
