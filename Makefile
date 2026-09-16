.PHONY: setup data train serve test docker up down scan

setup:
	python -m venv .venv && .venv/bin/pip install -r requirements.txt
data:
	python -m src.data.make_dataset
train: data
	python -m src.training.train
serve:
	uvicorn src.serving.app:app --reload --port 8000
test:
	pytest -q
docker:
	docker build -f deploy/docker/Dockerfile -t fraud-api:local .
up:
	docker compose up -d --build
down:
	docker compose down
scan:
	trivy image --severity HIGH,CRITICAL fraud-api:local
