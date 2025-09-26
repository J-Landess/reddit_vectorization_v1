"""
Machine learning classifier for medical category classification using embeddings.
"""
import logging
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
from .base import ClassificationAnalyzer, MEDICAL_CATEGORIES

logger = logging.getLogger(__name__)

class MLClassifier(ClassificationAnalyzer):
    """Machine learning classifier using embeddings for medical category classification."""
    
    def __init__(self, model_type: str = 'random_forest', model_path: str = None):
        """
        Initialize the ML classifier.
        
        Args:
            model_type: Type of ML model ('random_forest', 'logistic_regression', 'svm')
            model_path: Path to saved model (if None, will train new model)
        """
        super().__init__()
        self.model_type = model_type
        self.model_path = model_path or f'models/medical_classifier_{model_type}.joblib'
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Load model if it exists
        self._load_model()
    
    def _load_model(self):
        """Load pre-trained model if it exists."""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.is_trained = True
                logger.info(f"Loaded pre-trained {self.model_type} model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {self.model_path}: {e}")
                self._initialize_model()
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on model_type."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def train(self, embeddings: List[List[float]], labels: List[str], 
              test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train the classifier on embeddings and labels.
        
        Args:
            embeddings: List of embedding vectors
            labels: List of category labels
            test_size: Fraction of data to use for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with training metrics
        """
        if not embeddings or not labels:
            raise ValueError("Embeddings and labels cannot be empty")
        
        if len(embeddings) != len(labels):
            raise ValueError("Embeddings and labels must have the same length")
        
        # Convert to numpy arrays
        X = np.array(embeddings)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Save model
        self._save_model()
        self.is_trained = True
        
        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
        logger.info(f"Model training completed. Accuracy: {accuracy:.3f}")
        return metrics
    
    def classify(self, text: str, embedding: List[float] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify text using the trained ML model.
        
        Args:
            text: The text to classify (not used, embedding is used instead)
            embedding: Embedding vector for the text
            
        Returns:
            Tuple of (predicted_category, confidence_score, category_probabilities)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning default classification")
            return 'medical_insurance', 0.0, {cat: 0.25 for cat in self.categories}
        
        if embedding is None:
            logger.warning("No embedding provided, returning default classification")
            return 'medical_insurance', 0.0, {cat: 0.25 for cat in self.categories}
        
        try:
            # Prepare embedding
            X = np.array([embedding]).reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Get prediction and probabilities
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Map probabilities to category names
            category_probs = {}
            for i, category in enumerate(self.categories):
                if i < len(probabilities):
                    category_probs[category] = float(probabilities[i])
                else:
                    category_probs[category] = 0.0
            
            # Get confidence (max probability)
            confidence = max(category_probs.values())
            
            return prediction, confidence, category_probs
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return 'medical_insurance', 0.0, {cat: 0.25 for cat in self.categories}
    
    def _save_model(self):
        """Save the trained model and scaler."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'categories': self.categories
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available (for tree-based models).
        
        Returns:
            Dictionary mapping feature index to importance score
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
        
        importance = self.model.feature_importances_
        return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
    
    def predict_proba_batch(self, embeddings: List[List[float]]) -> List[Dict[str, float]]:
        """
        Get probability predictions for a batch of embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            List of probability dictionaries
        """
        if not self.is_trained:
            return [{cat: 0.25 for cat in self.categories} for _ in embeddings]
        
        try:
            X = np.array(embeddings)
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)
            
            results = []
            for prob_row in probabilities:
                category_probs = {}
                for i, category in enumerate(self.categories):
                    if i < len(prob_row):
                        category_probs[category] = float(prob_row[i])
                    else:
                        category_probs[category] = 0.0
                results.append(category_probs)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            return [{cat: 0.25 for cat in self.categories} for _ in embeddings]
