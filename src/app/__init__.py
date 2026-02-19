"""
ML Model Serving Package
"""

from .main import app
from .model import get_model

__all__ = ["app", "get_model"]
