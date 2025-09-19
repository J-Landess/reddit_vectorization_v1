"""
Embedding generation using Sentence Transformers.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for Reddit text data using Sentence Transformers."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', batch_size: int = 32):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the Sentence Transformer model
            batch_size: Batch size for embedding generation
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        logger.info(f"Initializing embedding generator with model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the Sentence Transformer model."""
        try:
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info(f"Successfully loaded model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        try:
            # Generate embeddings in batches to manage memory
            embeddings = []
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                embeddings.append(batch_embeddings)
                
                if i % (self.batch_size * 10) == 0:
                    logger.info(f"Processed {min(i + self.batch_size, len(texts))}/{len(texts)} texts")
            
            # Concatenate all embeddings
            all_embeddings = np.vstack(embeddings)
            
            logger.info(f"Generated embeddings with shape: {all_embeddings.shape}")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_embeddings_for_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for preprocessed Reddit data.
        
        Args:
            data: List of preprocessed data dictionaries
            
        Returns:
            List of data dictionaries with embeddings added
        """
        if not data:
            return []
        
        # Extract cleaned texts
        texts = [item['cleaned_text'] for item in data]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Add embeddings to data
        for i, item in enumerate(data):
            item['embedding'] = embeddings[i].tolist()
            item['embedding_dim'] = len(embeddings[i])
        
        logger.info(f"Added embeddings to {len(data)} items")
        return data
    
    def get_embedding_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_items(self, query_embedding: List[float], 
                          data: List[Dict[str, Any]], 
                          top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find most similar items to a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            data: List of data dictionaries with embeddings
            top_k: Number of most similar items to return
            
        Returns:
            List of most similar items with similarity scores
        """
        similarities = []
        
        for item in data:
            if 'embedding' in item:
                similarity = self.get_embedding_similarity(query_embedding, item['embedding'])
                similarities.append((item, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k items
        return similarities[:top_k]
    
    def get_embedding_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the generated embeddings.
        
        Args:
            data: List of data dictionaries with embeddings
            
        Returns:
            Dictionary of embedding statistics
        """
        if not data or 'embedding' not in data[0]:
            return {}
        
        embeddings = [item['embedding'] for item in data]
        embeddings_array = np.array(embeddings)
        
        stats = {
            'total_embeddings': len(embeddings),
            'embedding_dimension': len(embeddings[0]) if embeddings else 0,
            'mean_embedding_norm': float(np.mean([np.linalg.norm(emb) for emb in embeddings])),
            'std_embedding_norm': float(np.std([np.linalg.norm(emb) for emb in embeddings])),
            'min_embedding_norm': float(np.min([np.linalg.norm(emb) for emb in embeddings])),
            'max_embedding_norm': float(np.max([np.linalg.norm(emb) for emb in embeddings]))
        }
        
        return stats
