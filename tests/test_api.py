"""
Unit tests for the API.
"""

import pytest
from fastapi.testclient import TestClient
from src.app.main import app

client = TestClient(app)

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_readiness():
    """Test readiness check endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] == True

def test_prediction_valid():
    """Test valid prediction."""
    response = client.post("/predict", json={
        "features": [5.1, 3.5, 1.4, 0.2]
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1, 2]

def test_prediction_invalid_feature_count():
    """Test prediction with wrong number of features."""
    response = client.post("/predict", json={
        "features": [5.1, 3.5, 1.4]  # Only 3 features, need 4
    })
    assert response.status_code == 400

def test_prediction_invalid_range():
    """Test prediction with out-of-range features."""
    response = client.post("/predict", json={
        "features": [5.1, 3.5, 1.4, 100]  # 100 is out of range
    })
    assert response.status_code == 400

def test_model_info():
    """Test model info endpoint."""
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["model_version"] == "1.0.0"

def test_metrics():
    """Test metrics endpoint."""
    # Make a prediction first
    client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    
    # Get metrics
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "model_predictions_total" in response.text