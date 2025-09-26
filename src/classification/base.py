"""
Base class for medical category classification analyzers.
"""
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Medical categories
MEDICAL_CATEGORIES = {
    'medical_insurance': 'Medical Insurance',
    'medical_provider': 'Medical Provider', 
    'medical_broker': 'Medical Broker',
    'employer': 'Employer',
    'policy_changes': 'Policy Changes/Healthcare Legislation/Regulation'
}

class ClassificationAnalyzer(ABC):
    """Abstract base class for medical category classification analyzers."""

    def __init__(self):
        """Initialize the classifier."""
        self.categories = list(MEDICAL_CATEGORIES.keys())
        self.category_labels = list(MEDICAL_CATEGORIES.values())
        
    @abstractmethod
    def classify(self, text: str, embedding: List[float] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify text into one of the medical categories.
        
        Args:
            text: The text to classify
            embedding: Optional embedding vector for the text
            
        Returns:
            Tuple of (predicted_category, confidence_score, category_probabilities)
            - predicted_category: one of the MEDICAL_CATEGORIES keys
            - confidence_score: float in [0, 1] 
            - category_probabilities: dict mapping category -> probability
        """
        raise NotImplementedError
    
    def classify_batch(self, texts: List[str], embeddings: List[List[float]] = None) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Classify a batch of texts.
        
        Args:
            texts: List of texts to classify
            embeddings: Optional list of embedding vectors
            
        Returns:
            List of classification results
        """
        if embeddings is None:
            embeddings = [None] * len(texts)
            
        results = []
        for text, embedding in zip(texts, embeddings):
            try:
                result = self.classify(text, embedding)
                results.append(result)
            except Exception as e:
                logger.error(f"Error classifying text: {e}")
                # Return default classification
                results.append(('medical_insurance', 0.0, {cat: 0.25 for cat in self.categories}))
        
        return results
    
    def get_category_label(self, category: str) -> str:
        """Get human-readable label for a category."""
        return MEDICAL_CATEGORIES.get(category, category)
    
    def validate_category(self, category: str) -> bool:
        """Validate if category is valid."""
        return category in self.categories
