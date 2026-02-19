from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.model_version = "1.0.0"
        self.load_model()
    
    def load_model(self):
        """Load pre-trained model (or train if not exists)."""
        try:
            # For simplicity, train a model on iris dataset
            logger.info("Loading Iris dataset...")
            iris = load_iris()
            X = iris.data
            y = iris.target
            
            logger.info("Training RandomForest model...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
            
            logger.info(f"Model loaded successfully. Version: {self.model_version}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict(self, features: list) -> int:
        """
        Make prediction on input features.
        
        Args:
            features: List of 4 floats (iris features)
        
        Returns:
            prediction: Integer class (0, 1, or 2)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        prediction = self.model.predict([features])[0]
        return int(prediction)
    
    def get_feature_importance(self):
        """Return feature importance for debugging."""
        return self.model.feature_importances_.tolist()

# Global model instance
model_manager = None

def get_model():
    """Get or initialize model manager."""
    global model_manager
    if model_manager is None:
        model_manager = ModelManager()
    return model_manager
