"""
Hybrid classifier combining rule-based and ML approaches for medical category classification.
"""
import logging
from typing import Tuple, Dict, List
from .base import ClassificationAnalyzer, MEDICAL_CATEGORIES
from .rule_based_classifier import RuleBasedClassifier
from .ml_classifier import MLClassifier

logger = logging.getLogger(__name__)

class HybridClassifier(ClassificationAnalyzer):
    """
    Hybrid classifier that combines rule-based and ML approaches.
    
    Uses rule-based classification for high-confidence cases and falls back to ML
    for ambiguous cases. Can also use ensemble voting for maximum accuracy.
    """
    
    def __init__(self, ml_model_type: str = 'random_forest', 
                 rule_confidence_threshold: float = 0.7,
                 ensemble_mode: bool = False):
        """
        Initialize the hybrid classifier.
        
        Args:
            ml_model_type: Type of ML model to use
            rule_confidence_threshold: Minimum confidence for rule-based classification
            ensemble_mode: If True, use ensemble voting instead of fallback
        """
        super().__init__()
        self.rule_classifier = RuleBasedClassifier()
        self.ml_classifier = MLClassifier(model_type=ml_model_type)
        self.rule_confidence_threshold = rule_confidence_threshold
        self.ensemble_mode = ensemble_mode
        
        logger.info(f"Initialized hybrid classifier (rule_threshold={rule_confidence_threshold}, ensemble={ensemble_mode})")
    
    def classify(self, text: str, embedding: List[float] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify text using hybrid approach.
        
        Args:
            text: The text to classify
            embedding: Optional embedding vector for the text
            
        Returns:
            Tuple of (predicted_category, confidence_score, category_probabilities)
        """
        if not text or not text.strip():
            return 'medical_insurance', 0.0, {cat: 0.25 for cat in self.categories}
        
        if self.ensemble_mode:
            return self._ensemble_classify(text, embedding)
        else:
            return self._fallback_classify(text, embedding)
    
    def _fallback_classify(self, text: str, embedding: List[float] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Use rule-based classification if confident, otherwise use ML.
        
        Args:
            text: The text to classify
            embedding: Optional embedding vector for the text
            
        Returns:
            Classification result
        """
        # Try rule-based classification first
        rule_category, rule_confidence, rule_probs = self.rule_classifier.classify(text, embedding)
        
        # If rule-based is confident enough, use it
        if rule_confidence >= self.rule_confidence_threshold:
            logger.debug(f"Using rule-based classification: {rule_category} (confidence: {rule_confidence:.3f})")
            return rule_category, rule_confidence, rule_probs
        
        # Otherwise, use ML classification
        if embedding is not None and self.ml_classifier.is_trained:
            ml_category, ml_confidence, ml_probs = self.ml_classifier.classify(text, embedding)
            logger.debug(f"Using ML classification: {ml_category} (confidence: {ml_confidence:.3f})")
            return ml_category, ml_confidence, ml_probs
        else:
            # Fallback to rule-based even if not confident
            logger.debug(f"Fallback to rule-based classification: {rule_category} (confidence: {rule_confidence:.3f})")
            return rule_category, rule_confidence, rule_probs
    
    def _ensemble_classify(self, text: str, embedding: List[float] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Use ensemble voting between rule-based and ML classifiers.
        
        Args:
            text: The text to classify
            embedding: Optional embedding vector for the text
            
        Returns:
            Classification result
        """
        # Get predictions from both classifiers
        rule_category, rule_confidence, rule_probs = self.rule_classifier.classify(text, embedding)
        
        if embedding is not None and self.ml_classifier.is_trained:
            ml_category, ml_confidence, ml_probs = self.ml_classifier.classify(text, embedding)
        else:
            # If ML not available, use rule-based
            return rule_category, rule_confidence, rule_probs
        
        # Weighted ensemble (can be adjusted based on performance)
        rule_weight = 0.4  # Rule-based weight
        ml_weight = 0.6    # ML weight
        
        # Combine probabilities
        ensemble_probs = {}
        for category in self.categories:
            ensemble_probs[category] = (
                rule_probs[category] * rule_weight + 
                ml_probs[category] * ml_weight
            )
        
        # Find best category
        predicted_category = max(ensemble_probs, key=ensemble_probs.get)
        confidence = ensemble_probs[predicted_category]
        
        # Normalize probabilities
        total_prob = sum(ensemble_probs.values())
        if total_prob > 0:
            ensemble_probs = {cat: prob / total_prob for cat, prob in ensemble_probs.items()}
        
        logger.debug(f"Ensemble classification: {predicted_category} (confidence: {confidence:.3f})")
        return predicted_category, confidence, ensemble_probs
    
    def train_ml_component(self, embeddings: List[List[float]], labels: List[str], **kwargs) -> Dict[str, float]:
        """
        Train the ML component of the hybrid classifier.
        
        Args:
            embeddings: List of embedding vectors
            labels: List of category labels
            **kwargs: Additional arguments for ML training
            
        Returns:
            Training metrics
        """
        return self.ml_classifier.train(embeddings, labels, **kwargs)
    
    def get_classification_breakdown(self, text: str, embedding: List[float] = None) -> Dict[str, any]:
        """
        Get detailed breakdown of classification process.
        
        Args:
            text: The text to classify
            embedding: Optional embedding vector for the text
            
        Returns:
            Dictionary with detailed classification information
        """
        # Rule-based results
        rule_category, rule_confidence, rule_probs = self.rule_classifier.classify(text, embedding)
        rule_matches = self.rule_classifier.get_keyword_matches(text, rule_category)
        
        # ML results (if available)
        ml_results = None
        if embedding is not None and self.ml_classifier.is_trained:
            ml_category, ml_confidence, ml_probs = self.ml_classifier.classify(text, embedding)
            ml_results = {
                'category': ml_category,
                'confidence': ml_confidence,
                'probabilities': ml_probs
            }
        
        # Final hybrid result
        final_category, final_confidence, final_probs = self.classify(text, embedding)
        
        return {
            'text': text,
            'final_result': {
                'category': final_category,
                'confidence': final_confidence,
                'probabilities': final_probs
            },
            'rule_based': {
                'category': rule_category,
                'confidence': rule_confidence,
                'probabilities': rule_probs,
                'keyword_matches': rule_matches
            },
            'ml_based': ml_results,
            'method_used': 'ensemble' if self.ensemble_mode else 'fallback'
        }
    
    def update_rule_confidence_threshold(self, threshold: float):
        """Update the rule-based confidence threshold."""
        self.rule_confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Updated rule confidence threshold to {self.rule_confidence_threshold}")
    
    def set_ensemble_weights(self, rule_weight: float, ml_weight: float):
        """
        Set ensemble weights for voting (only used in ensemble mode).
        
        Args:
            rule_weight: Weight for rule-based classifier
            ml_weight: Weight for ML classifier
        """
        if abs(rule_weight + ml_weight - 1.0) > 0.01:
            logger.warning("Weights should sum to 1.0, normalizing...")
            total = rule_weight + ml_weight
            rule_weight /= total
            ml_weight /= total
        
        self.rule_weight = rule_weight
        self.ml_weight = ml_weight
        logger.info(f"Updated ensemble weights: rule={rule_weight:.2f}, ml={ml_weight:.2f}")
