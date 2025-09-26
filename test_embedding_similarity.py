#!/usr/bin/env python3
"""
Test script for embedding similarity classifier using real database data.
This implements Option 1: Embedding Similarity classification.
"""
import os
import sys
import logging
import random
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.classification import EmbeddingSimilarityClassifier, RuleBasedClassifier
from src.database.database_manager import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_prototypes_from_database(db_path: str = './data/reddit_data.db', 
                                 sample_size: int = 100) -> EmbeddingSimilarityClassifier:
    """
    Build category prototypes from database data using rule-based labels.
    
    Args:
        db_path: Path to the SQLite database
        sample_size: Number of items to sample for prototype building
        
    Returns:
        Initialized EmbeddingSimilarityClassifier
    """
    print(f"🏗️  Building prototypes from database: {db_path}")
    
    try:
        # Get data from database
        db_manager = DatabaseManager(db_path)
        all_data = db_manager.get_all_data(include_embeddings=True)
        db_manager.close()
        
        if not all_data:
            print("❌ No data found in database")
            return None
        
        print(f"📊 Found {len(all_data)} total items in database")
        
        # Filter data with embeddings
        valid_data = [
            item for item in all_data 
            if item.get('embedding') and len(item.get('embedding', [])) > 0
        ]
        
        print(f"🔗 Found {len(valid_data)} items with embeddings")
        
        if len(valid_data) < sample_size:
            print(f"⚠️  Only {len(valid_data)} items available, using all of them")
            sample_data = valid_data
        else:
            sample_data = random.sample(valid_data, sample_size)
        
        print(f"📝 Using {len(sample_data)} items for prototype building")
        
        # Generate rule-based labels
        rule_classifier = RuleBasedClassifier()
        print("🏷️  Generating rule-based labels...")
        
        for item in sample_data:
            text = item.get('cleaned_text') or item.get('text', '')
            category, _, _ = rule_classifier.classify(text)
            item['category'] = category
        
        # Build prototypes
        embedding_classifier = EmbeddingSimilarityClassifier()
        prototypes = embedding_classifier.build_prototypes(sample_data, min_samples_per_category=3)
        
        print(f"✅ Built prototypes for {len(prototypes)} categories")
        
        # Show prototype info
        prototype_info = embedding_classifier.get_prototype_info()
        print("\n📊 Prototype Information:")
        for category, info in prototype_info.items():
            print(f"  {category}: dim={info['dimension']}, norm={info['norm']:.3f}, zero={info['is_zero']}")
        
        return embedding_classifier
        
    except Exception as e:
        print(f"❌ Error building prototypes: {e}")
        logger.exception("Prototype building failed")
        return None

def test_embedding_similarity_classification(embedding_classifier: EmbeddingSimilarityClassifier, 
                                           test_data: list, 
                                           num_tests: int = 10):
    """Test the embedding similarity classifier on real data."""
    print(f"\n🧪 TESTING EMBEDDING SIMILARITY CLASSIFICATION")
    print("="*60)
    
    if not embedding_classifier or not test_data:
        print("❌ No classifier or test data available")
        return
    
    # Sample test data
    test_samples = random.sample(test_data, min(num_tests, len(test_data)))
    
    print(f"Testing on {len(test_samples)} real Reddit items...\n")
    
    correct_predictions = 0
    total_predictions = 0
    
    for i, item in enumerate(test_samples, 1):
        text = item.get('cleaned_text') or item.get('text', '')
        embedding = item.get('embedding')
        
        if not embedding:
            continue
        
        print(f"--- Test {i} ---")
        print(f"Subreddit: r/{item.get('subreddit', 'unknown')}")
        print(f"Type: {item.get('type', 'unknown')}")
        print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        # Get rule-based prediction for comparison
        rule_classifier = RuleBasedClassifier()
        rule_category, rule_confidence, _ = rule_classifier.classify(text)
        
        # Get embedding similarity prediction
        emb_category, emb_confidence, emb_probs = embedding_classifier.classify(text, embedding)
        
        print(f"Rule-based: {rule_classifier.get_category_label(rule_category)} (conf: {rule_confidence:.3f})")
        print(f"Embedding:  {embedding_classifier.get_category_label(emb_category)} (conf: {emb_confidence:.3f})")
        
        # Show similarities to all categories
        similarities = embedding_classifier.get_category_similarities(embedding)
        print("Similarities:")
        for cat, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {embedding_classifier.get_category_label(cat)}: {sim:.3f}")
        
        # Check if predictions match
        if rule_category == emb_category:
            correct_predictions += 1
            print("✅ Predictions match!")
        else:
            print("❌ Predictions differ")
        
        total_predictions += 1
        print()
    
    # Show accuracy
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"📊 Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.3f}")
    
    print()

def compare_classification_methods(embedding_classifier: EmbeddingSimilarityClassifier, 
                                test_data: list, 
                                num_tests: int = 20):
    """Compare embedding similarity vs rule-based classification."""
    print(f"\n🔄 COMPARING CLASSIFICATION METHODS")
    print("="*60)
    
    if not embedding_classifier or not test_data:
        print("❌ No classifier or test data available")
        return
    
    # Sample test data
    test_samples = random.sample(test_data, min(num_tests, len(test_data)))
    
    rule_classifier = RuleBasedClassifier()
    
    # Count predictions by method
    rule_predictions = {}
    embedding_predictions = {}
    agreement_count = 0
    
    for item in test_samples:
        text = item.get('cleaned_text') or item.get('text', '')
        embedding = item.get('embedding')
        
        if not embedding:
            continue
        
        # Rule-based prediction
        rule_category, _, _ = rule_classifier.classify(text)
        rule_predictions[rule_category] = rule_predictions.get(rule_category, 0) + 1
        
        # Embedding similarity prediction
        emb_category, _, _ = embedding_classifier.classify(text, embedding)
        embedding_predictions[emb_category] = embedding_predictions.get(emb_category, 0) + 1
        
        # Check agreement
        if rule_category == emb_category:
            agreement_count += 1
    
    print("📊 Rule-based Predictions:")
    for category, count in sorted(rule_predictions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rule_classifier.get_category_label(category)}: {count}")
    
    print("\n📊 Embedding Similarity Predictions:")
    for category, count in sorted(embedding_predictions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {embedding_classifier.get_category_label(category)}: {count}")
    
    total_comparisons = len([item for item in test_samples if item.get('embedding')])
    agreement_rate = agreement_count / total_comparisons if total_comparisons > 0 else 0
    print(f"\n🤝 Agreement Rate: {agreement_count}/{total_comparisons} = {agreement_rate:.3f}")
    
    print()

def main():
    """Main function to test embedding similarity classification."""
    print("🧪 EMBEDDING SIMILARITY CLASSIFICATION TEST")
    print("="*60)
    print("This implements Option 1: Embedding Similarity classification")
    print("Uses your existing database data and embeddings")
    print("="*60)
    
    # Check if database exists
    db_path = './data/reddit_data.db'
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        print("Please run the main pipeline first to collect some data.")
        return False
    
    try:
        # Build prototypes from database
        embedding_classifier = build_prototypes_from_database(db_path, sample_size=1000)
        
        if not embedding_classifier:
            print("❌ Failed to build prototypes")
            return False
        
        # Get test data
        db_manager = DatabaseManager(db_path)
        test_data = db_manager.get_all_data(include_embeddings=True)
        db_manager.close()
        
        # Filter test data with embeddings
        test_data = [item for item in test_data if item.get('embedding')]
        
        if not test_data:
            print("❌ No test data with embeddings available")
            return False
        
        print(f"📊 Using {len(test_data)} items for testing")
        
        # Test classification
        test_embedding_similarity_classification(embedding_classifier, test_data, num_tests=15)
        
        # Compare methods
        compare_classification_methods(embedding_classifier, test_data, num_tests=30)
        
        print("="*60)
        print("✅ EMBEDDING SIMILARITY TEST COMPLETED!")
        print("="*60)
        print("\nThe embedding similarity classifier is working with your real data.")
        print("You can now use this as your primary classification method.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Embedding similarity test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
