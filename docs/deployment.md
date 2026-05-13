# Deployment Runbook

## Overview
This document covers the end-to-end deployment of the 6 ONNX models for the Industrial Defect Detection System.

## Deploying

1. Ensure the `models_onnx/` directory is populated with all 6 categories (run `scripts/export_all.py` to generate).
2. Start the API using Docker Compose:
   ```bash
   docker-compose up -d
   ```
3. The API will be available at `http://localhost:8000`.

## Endpoints

- `GET /health` : Healthcheck. Returns `{"status": "healthy"}` if models are loaded.
- `GET /metrics` : Prometheus metrics (placeholder).
- `POST /predict` : Single image prediction.
  - Form Data: `category` (string), `file` (image).
- `POST /predict_batch` : Batch prediction.
  - Form Data: `category` (string), `files` (list of images).

## Debugging

- **Models not loaded**: Check the Docker logs using `docker-compose logs -f defect-api`. Ensure the `.onnx` files exist in `models_onnx/`.
- **Latency issues**: Check the `GET /metrics` endpoint or the locust report for latency statistics. CPU inference might spike under heavy load.

## Rollback

To rollback to a previous model version:
1. Re-export the previous checkpoint using `python -m src.models.export_onnx`.
2. Replace the `.onnx` file in `models_onnx/`.
3. Restart the container: `docker-compose restart defect-api`.
