# ML Model Serving Infrastructure

Production-grade ML model serving system using FastAPI, Docker, and Kubernetes.

## Architecture

- **API**: FastAPI (Python)
- **Container**: Docker
- **Orchestration**: Kubernetes
- **Monitoring**: Prometheus + Grafana
- **Model**: scikit-learn RandomForest (Iris dataset)

## Quick Start

### Local Development
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd src
uvicorn app.main:app --reload
```

### Docker
```bash
docker build -f docker/Dockerfile -t ml-model-serving:latest .
docker run -p 8000:8000 ml-model-serving:latest
```

### Kubernetes
```bash
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml
```

## API Endpoints

- `POST /predict` - Make prediction
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics
- `GET /model/info` - Model information

## Example Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Monitoring

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Testing
```bash
pytest tests/ -v
```