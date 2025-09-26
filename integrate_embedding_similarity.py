#!/usr/bin/env python3
"""
Integration script showing how to use embedding similarity classifier in your pipeline.
This replaces the hybrid classifier with embedding similarity for immediate results.
"""
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.classification import EmbeddingSimilarityClassifier, RuleBasedClassifier
from src.database.database_manager import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_embedding_similarity_classifier(db_path: str = './data/reddit_data.db') -> EmbeddingSimilarityClassifier:
    """
    Set up embedding similarity classifier using your existing database data.
    
    Args:
        db_path: Path to your SQLite database
        
    Returns:
        Initialized EmbeddingSimilarityClassifier ready for use
    """
    print("🚀 Setting up Embedding Similarity Classifier...")
    
    try:
        # Get data from database
        db_manager = DatabaseManager(db_path)
        all_data = db_manager.get_all_data(include_embeddings=True)
        db_manager.close()
        
        if not all_data:
            raise ValueError("No data found in database")
        
        # Filter data with embeddings
        valid_data = [item for item in all_data if item.get('embedding')]
        
        if len(valid_data) < 20:
            raise ValueError(f"Not enough data with embeddings: {len(valid_data)} < 20")
        
        print(f"📊 Found {len(valid_data)} items with embeddings")
        
        # Generate rule-based labels for prototype building
        rule_classifier = RuleBasedClassifier()
        print("🏷️  Generating rule-based labels for prototype building...")
        
        for item in valid_data:
            text = item.get('cleaned_text') or item.get('text', '')
            category, _, _ = rule_classifier.classify(text)
            item['category'] = category
        
        # Build prototypes
        embedding_classifier = EmbeddingSimilarityClassifier()
        prototypes = embedding_classifier.build_prototypes(valid_data, min_samples_per_category=3)
        
        print(f"✅ Built prototypes for {len(prototypes)} categories")
        
        # Show prototype statistics
        prototype_info = embedding_classifier.get_prototype_info()
        print("\n📊 Prototype Statistics:")
        for category, info in prototype_info.items():
            print(f"  {category}: {info['dimension']}D, norm={info['norm']:.3f}")
        
        return embedding_classifier
        
    except Exception as e:
        print(f"❌ Error setting up classifier: {e}")
        raise

def classify_new_text(embedding_classifier: EmbeddingSimilarityClassifier, 
                     text: str, 
                     embedding: list = None) -> dict:
    """
    Classify a new text using embedding similarity.
    
    Args:
        embedding_classifier: Initialized classifier
        text: Text to classify
        embedding: Embedding vector (if None, will need to be generated)
        
    Returns:
        Classification results dictionary
    """
    if embedding is None:
        print("⚠️  No embedding provided - you'll need to generate one first")
        return None
    
    # Classify using embedding similarity
    category, confidence, probabilities = embedding_classifier.classify(text, embedding)
    
    # Get detailed similarities
    similarities = embedding_classifier.get_category_similarities(embedding)
    
    return {
        'category': category,
        'category_label': embedding_classifier.get_category_label(category),
        'confidence': confidence,
        'probabilities': probabilities,
        'similarities': similarities
    }

def example_usage():
    """Example of how to use the embedding similarity classifier."""
    print("📝 EMBEDDING SIMILARITY CLASSIFIER - EXAMPLE USAGE")
    print("="*60)
    
    try:
        # 1. Set up the classifier
        classifier = setup_embedding_similarity_classifier()
        
        # 2. Example texts to classify
        example_texts = [
            "I need help choosing a health insurance plan for my family",
            "My doctor recommended a specialist for my condition",
            "I'm an insurance broker and can help you compare plans",
            "Our company is updating our employee benefits package"
        ]
        
        # 3. Mock embeddings (in real usage, you'd generate these)
        mock_embeddings = [
            [0.1] * 384,  # Mock embedding for text 1
            [0.2] * 384,  # Mock embedding for text 2
            [0.3] * 384,  # Mock embedding for text 3
            [0.4] * 384,  # Mock embedding for text 4
        ]
        
        print("\n🔍 Classification Examples:")
        print("-" * 40)
        
        for i, (text, embedding) in enumerate(zip(example_texts, mock_embeddings), 1):
            result = classify_new_text(classifier, text, embedding)
            
            if result:
                print(f"\n{i}. Text: {text}")
                print(f"   Category: {result['category_label']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Top Similarities:")
                sorted_sims = sorted(result['similarities'].items(), key=lambda x: x[1], reverse=True)
                for cat, sim in sorted_sims[:2]:
                    print(f"     - {classifier.get_category_label(cat)}: {sim:.3f}")
        
        print("\n✅ Example completed successfully!")
        print("\nTo use in your pipeline:")
        print("1. Set up classifier once: classifier = setup_embedding_similarity_classifier()")
        print("2. Classify new items: result = classify_new_text(classifier, text, embedding)")
        print("3. Use result['category'] and result['confidence'] in your pipeline")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        logger.exception("Example usage failed")

def main():
    """Main function."""
    example_usage()

if __name__ == "__main__":
    main()
