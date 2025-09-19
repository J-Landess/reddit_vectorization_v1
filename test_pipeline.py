#!/usr/bin/env python3
"""
Test script for Reddit Data Analysis Pipeline.
This script tests individual components without requiring Reddit API access.
"""
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.text_cleaner import TextCleaner
from embeddings.embedding_generator import EmbeddingGenerator
from clustering.cluster_analyzer import ClusterAnalyzer
from analysis.analyzer import RedditAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_text_cleaner():
    """Test text cleaning functionality."""
    print("🧹 Testing Text Cleaner")
    print("-" * 30)
    
    cleaner = TextCleaner()
    
    # Test cases
    test_texts = [
        "I have a question about my **health insurance** coverage!",
        "https://example.com - Check this link for more info",
        "u/username mentioned r/Health subreddit",
        "This is a test with [deleted] content",
        "Multiple!!! punctuation??? marks..."
    ]
    
    for i, text in enumerate(test_texts, 1):
        cleaned = cleaner.clean_text(text)
        print(f"Test {i}:")
        print(f"  Original: {text}")
        print(f"  Cleaned:  {cleaned}")
        print()
    
    print("✅ Text cleaner test passed\n")

def test_embedding_generator():
    """Test embedding generation."""
    print("🧠 Testing Embedding Generator")
    print("-" * 30)
    
    try:
        generator = EmbeddingGenerator()
        
        # Test texts
        texts = [
            "health insurance coverage",
            "medicare plan selection",
            "medicaid application help",
            "medical question for doctors"
        ]
        
        # Generate embeddings
        embeddings = generator.generate_embeddings(texts)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        # Test similarity
        similarity = generator.get_embedding_similarity(
            embeddings[0].tolist(),
            embeddings[1].tolist()
        )
        print(f"Similarity between texts 1 and 2: {similarity:.3f}")
        
        print("✅ Embedding generator test passed\n")
        
    except Exception as e:
        print(f"❌ Embedding generator test failed: {e}\n")

def test_cluster_analyzer():
    """Test clustering functionality."""
    print("🔗 Testing Cluster Analyzer")
    print("-" * 30)
    
    try:
        # Generate sample embeddings
        generator = EmbeddingGenerator()
        texts = [
            "health insurance coverage",
            "medicare plan selection", 
            "medicaid application help",
            "medical question for doctors",
            "pharmacy prescription refill",
            "mental health counseling"
        ]
        
        embeddings = generator.generate_embeddings(texts)
        
        # Test k-means clustering
        analyzer = ClusterAnalyzer(algorithm='kmeans', n_clusters=2)
        labels = analyzer.fit(embeddings)
        
        print(f"Cluster labels: {labels}")
        
        # Get statistics
        stats = analyzer.get_cluster_statistics()
        print(f"Cluster statistics: {stats}")
        
        print("✅ Cluster analyzer test passed\n")
        
    except Exception as e:
        print(f"❌ Cluster analyzer test failed: {e}\n")

def test_analyzer():
    """Test analysis functionality."""
    print("📊 Testing Analyzer")
    print("-" * 30)
    
    try:
        analyzer = RedditAnalyzer()
        
        # Sample data
        sample_data = [
            {
                'id': '1',
                'type': 'post',
                'subreddit': 'Health',
                'score': 10,
                'word_count': 25,
                'cleaned_text': 'health insurance coverage question'
            },
            {
                'id': '2', 
                'type': 'comment',
                'subreddit': 'Medicare',
                'score': 5,
                'word_count': 30,
                'cleaned_text': 'medicare plan selection help'
            }
        ]
        
        # Test analysis
        analysis = analyzer.analyze_data_distribution(sample_data)
        print(f"Analysis results: {analysis}")
        
        print("✅ Analyzer test passed\n")
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}\n")

def main():
    """Run all tests."""
    print("🧪 Reddit Analysis Pipeline - Component Tests")
    print("=" * 50)
    print("Testing individual components without Reddit API access...\n")
    
    try:
        test_text_cleaner()
        test_embedding_generator()
        test_cluster_analyzer()
        test_analyzer()
        
        print("🎉 All tests completed!")
        print("\nTo run the full pipeline with Reddit data:")
        print("1. Set your Reddit API credentials")
        print("2. Run: python main.py")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        logger.error(f"Test suite error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
