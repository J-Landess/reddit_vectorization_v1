"""
Embedding similarity classifier for medical category classification.
Uses cosine similarity between text embeddings and category prototypes.
"""
import logging
import numpy as np
from typing import Tuple, Dict, List, Any
from sklearn.metrics.pairwise import cosine_similarity
from .base import ClassificationAnalyzer, MEDICAL_CATEGORIES

logger = logging.getLogger(__name__)

class EmbeddingSimilarityClassifier(ClassificationAnalyzer):
    """
    Classifier that uses embedding similarity to category prototypes.
    No training required - works immediately with existing embeddings.
    """
    
    def __init__(self, category_prototypes: Dict[str, np.ndarray] = None):
        """
        Initialize the embedding similarity classifier.
        
        Args:
            category_prototypes: Pre-computed category prototype embeddings
        """
        super().__init__()
        self.category_prototypes = category_prototypes or {}
        self.is_initialized = len(self.category_prototypes) > 0
        
        if not self.is_initialized:
            logger.warning("No category prototypes provided. Use build_prototypes() to initialize.")
    
    def build_prototypes(self, data: List[Dict[str, Any]], 
                        min_samples_per_category: int = 5) -> Dict[str, np.ndarray]:
        """
        Build category prototypes from existing data.
        
        Args:
            data: List of data items with embeddings and categories
            min_samples_per_category: Minimum samples needed per category
            
        Returns:
            Dictionary mapping category to prototype embedding
        """
        logger.info("Building category prototypes from existing data...")
        
        # Group data by category
        category_embeddings = {category: [] for category in self.categories}
        
        for item in data:
            category = item.get('category')
            embedding = item.get('embedding')
            
            if category in self.categories and embedding and len(embedding) > 0:
                category_embeddings[category].append(embedding)
        
        # Create prototypes (mean embeddings)
        prototypes = {}
        for category, embeddings in category_embeddings.items():
            if len(embeddings) >= min_samples_per_category:
                prototypes[category] = np.mean(embeddings, axis=0)
                logger.info(f"Created prototype for {category}: {len(embeddings)} samples")
            else:
                logger.warning(f"Insufficient samples for {category}: {len(embeddings)} < {min_samples_per_category}")
                # Use zero vector as fallback
                if embeddings:
                    prototypes[category] = np.zeros_like(embeddings[0])
                else:
                    prototypes[category] = np.zeros(384)  # Default embedding size
        
        self.category_prototypes = prototypes
        self.is_initialized = True
        
        logger.info(f"Built prototypes for {len(prototypes)} categories")
        return prototypes
    
    def classify(self, text: str, embedding: List[float] = None) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify text using embedding similarity.
        
        Args:
            text: The text to classify (not used, embedding is used instead)
            embedding: Embedding vector for the text
            
        Returns:
            Tuple of (predicted_category, confidence_score, category_probabilities)
        """
        if not self.is_initialized:
            logger.warning("Classifier not initialized. Returning default classification.")
            return 'medical_insurance', 0.0, {cat: 0.25 for cat in self.categories}
        
        if embedding is None:
            logger.warning("No embedding provided. Returning default classification.")
            return 'medical_insurance', 0.0, {cat: 0.25 for cat in self.categories}
        
        try:
            # Convert embedding to numpy array
            text_embedding = np.array(embedding).reshape(1, -1)
            
            # Calculate similarities to all category prototypes
            similarities = {}
            for category, prototype in self.category_prototypes.items():
                if prototype is not None and len(prototype) > 0:
                    prototype_reshaped = prototype.reshape(1, -1)
                    similarity = cosine_similarity(text_embedding, prototype_reshaped)[0][0]
                    similarities[category] = float(similarity)
                else:
                    similarities[category] = 0.0
            
            # Find best category
            predicted_category = max(similarities, key=similarities.get)
            confidence = similarities[predicted_category]
            
            # Normalize similarities to probabilities
            total_similarity = sum(similarities.values())
            if total_similarity > 0:
                probabilities = {cat: sim / total_similarity for cat, sim in similarities.items()}
            else:
                probabilities = {cat: 0.25 for cat in self.categories}
            
            return predicted_category, confidence, probabilities
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return 'medical_insurance', 0.0, {cat: 0.25 for cat in self.categories}
    
    def get_category_similarities(self, embedding: List[float]) -> Dict[str, float]:
        """
        Get similarity scores to all categories.
        
        Args:
            embedding: Embedding vector for the text
            
        Returns:
            Dictionary mapping category to similarity score
        """
        if not self.is_initialized or embedding is None:
            return {cat: 0.0 for cat in self.categories}
        
        try:
            text_embedding = np.array(embedding).reshape(1, -1)
            similarities = {}
            
            for category, prototype in self.category_prototypes.items():
                if prototype is not None and len(prototype) > 0:
                    prototype_reshaped = prototype.reshape(1, -1)
                    similarity = cosine_similarity(text_embedding, prototype_reshaped)[0][0]
                    similarities[category] = float(similarity)
                else:
                    similarities[category] = 0.0
            
            return similarities
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return {cat: 0.0 for cat in self.categories}
    
    def update_prototype(self, category: str, new_embedding: List[float], 
                        learning_rate: float = 0.1):
        """
        Update a category prototype with new embedding (online learning).
        
        Args:
            category: Category to update
            new_embedding: New embedding to incorporate
            learning_rate: Learning rate for prototype update
        """
        if category not in self.categories:
            logger.warning(f"Unknown category: {category}")
            return
        
        if not self.is_initialized:
            self.category_prototypes = {cat: np.zeros(384) for cat in self.categories}
            self.is_initialized = True
        
        try:
            new_embedding = np.array(new_embedding)
            current_prototype = self.category_prototypes.get(category, np.zeros_like(new_embedding))
            
            # Update prototype using exponential moving average
            updated_prototype = (1 - learning_rate) * current_prototype + learning_rate * new_embedding
            self.category_prototypes[category] = updated_prototype
            
            logger.info(f"Updated prototype for {category} with learning rate {learning_rate}")
            
        except Exception as e:
            logger.error(f"Error updating prototype for {category}: {e}")
    
    def get_prototype_info(self) -> Dict[str, Dict[str, any]]:
        """
        Get information about category prototypes.
        
        Returns:
            Dictionary with prototype information
        """
        info = {}
        for category, prototype in self.category_prototypes.items():
            info[category] = {
                'dimension': len(prototype) if prototype is not None else 0,
                'norm': float(np.linalg.norm(prototype)) if prototype is not None else 0.0,
                'is_zero': np.allclose(prototype, 0) if prototype is not None else True
            }
        return info
