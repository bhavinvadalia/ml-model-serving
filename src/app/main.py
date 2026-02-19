"""
FastAPI application for model serving.
Handles predictions, health checks, and metrics.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest
import time
import logging
from typing import List
from .model import get_model

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ML Model Serving API",
    description="Production-grade API for serving ML predictions",
    version="1.0.0"
)

# Prometheus metrics
request_count = Counter(
    'model_predictions_total',
    'Total number of predictions made',
    ['model_version']
)

prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Time taken to make a prediction in seconds',
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    features: List[float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "features": [5.1, 3.5, 1.4, 0.2]
            }
        }

class PredictionResponse(BaseModel):
    """Response body for prediction endpoint."""
    prediction: int
    confidence: float
    model_version: str

# ============ PREDICTION ENDPOINT ============

@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Make a prediction on input features.
    
    Expected input: 4 features for Iris dataset
    Returns: predicted class (0, 1, or 2)
    """
    start_time = time.time()
    
    try:
        # Input validation
        if len(request.features) != 4:
            raise HTTPException(
                status_code=400,
                detail="Expected exactly 4 features"
            )
        
        # Check feature ranges (iris dataset constraints)
        if not all(0 <= f <= 10 for f in request.features):
            raise HTTPException(
                status_code=400,
                detail="Features should be between 0 and 10"
            )
        
        # Get model and make prediction
        model = get_model()
        prediction = model.predict(request.features)
        
        # Record metrics
        model_version = model.model_version
        prediction_latency.observe(time.time() - start_time)
        request_count.labels(model_version=model_version).inc()
        
        # Return response
        return PredictionResponse(
            prediction=prediction,
            confidence=0.95,  # For demo purposes
            model_version=model_version
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# ============ HEALTH CHECK ENDPOINTS ============

@app.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    Used by Kubernetes liveness probe.
    """
    try:
        # Verify model is loaded
        model = get_model()
        if model.model is None:
            return {"status": "unhealthy", "reason": "model not loaded"}
        
        return {
            "status": "healthy",
            "model_version": model.model_version
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "reason": str(e)}

@app.get("/ready")
async def readiness_check():
    """
    Readiness probe endpoint.
    Used by Kubernetes to determine if pod can receive traffic.
    """
    try:
        model = get_model()
        if model.model is None:
            return {"ready": False}
        
        # Could add database connection checks here
        return {"ready": True}
    except Exception:
        return {"ready": False}

# ============ METRICS ENDPOINT ============

@app.get("/metrics")
async def metrics():
    """
    Expose Prometheus metrics.
    Prometheus scrapes this endpoint every 15 seconds.
    """
    return PlainTextResponse(generate_latest())

# ============ MODEL INFO ENDPOINTS ============

@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    model = get_model()
    return {
        "model_type": "RandomForestClassifier",
        "model_version": model.model_version,
        "n_estimators": 100,
        "classes": [0, 1, 2],
        "feature_count": 4,
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    }

@app.get("/model/features")
async def feature_importance():
    """Get feature importance from the model."""
    model = get_model()
    importance = model.get_feature_importance()
    return {
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "importance": importance
    }

# ============ ROOT ENDPOINT ============

@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "message": "ML Model Serving API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

# ============ STARTUP/SHUTDOWN EVENTS ============

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    logger.info("Starting up... Loading model")
    get_model()
    logger.info("Startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down...")
