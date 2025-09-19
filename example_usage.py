#!/usr/bin/env python3
"""
Example usage of individual Reddit analysis components.
This script demonstrates how to use each component separately.
"""
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.reddit_client import RedditClient
from preprocessing.text_cleaner import TextCleaner
from embeddings.embedding_generator import EmbeddingGenerator
from database.database_manager import DatabaseManager
from clustering.cluster_analyzer import ClusterAnalyzer
from analysis.analyzer import RedditAnalyzer
from config import REDDIT_CONFIG, SUBREDDITS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def example_data_collection():
    """Example: Collect data from a single subreddit."""
    print("🔍 Example: Data Collection")
    print("-" * 40)
    
    # Initialize Reddit client
    reddit_client = RedditClient(
        client_id=REDDIT_CONFIG['client_id'],
        client_secret=REDDIT_CONFIG['client_secret'],
        user_agent=REDDIT_CONFIG['user_agent']
    )
    
    # Collect data from one subreddit
    data = reddit_client.collect_subreddit_data('Health', max_posts=10, max_comments_per_post=5)
    
    print(f"Collected {len(data)} items from r/Health")
    if data:
        print(f"Sample item: {data[0]['title'][:50]}..." if data[0].get('title') else f"Sample comment: {data[0]['text'][:50]}...")
    
    return data

def example_text_preprocessing():
    """Example: Preprocess text data."""
    print("\n🧹 Example: Text Preprocessing")
    print("-" * 40)
    
    # Sample Reddit text
    sample_texts = [
        "I have a question about my **health insurance** coverage. Can anyone help?",
        "r/Health - What's the best way to deal with chronic pain?",
        "https://example.com - This is a link to more info",
        "u/username mentioned this subreddit r/AskDocs"
    ]
    
    # Initialize text cleaner
    text_cleaner = TextCleaner()
    
    # Clean each text
    for i, text in enumerate(sample_texts, 1):
        cleaned = text_cleaner.clean_text(text)
        print(f"Original {i}: {text}")
        print(f"Cleaned {i}:  {cleaned}")
        print()

def example_embedding_generation():
    """Example: Generate embeddings for text."""
    print("\n🧠 Example: Embedding Generation")
    print("-" * 40)
    
    # Sample texts
    texts = [
        "I need help with my health insurance",
        "What are the best Medicare plans?",
        "How do I apply for Medicaid?",
        "I have a medical question for doctors"
    ]
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()
    
    # Generate embeddings
    embeddings = embedding_generator.generate_embeddings(texts)
    
    print(f"Generated embeddings with shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Calculate similarity between first two texts
    similarity = embedding_generator.get_embedding_similarity(
        embeddings[0].tolist(), 
        embeddings[1].tolist()
    )
    print(f"Similarity between texts 1 and 2: {similarity:.3f}")

def example_clustering():
    """Example: Perform clustering analysis."""
    print("\n🔗 Example: Clustering Analysis")
    print("-" * 40)
    
    # Sample data (in practice, this would come from Reddit)
    sample_data = [
        {'id': '1', 'text': 'health insurance coverage', 'type': 'post'},
        {'id': '2', 'text': 'medicare plan selection', 'type': 'post'},
        {'id': '3', 'text': 'medicaid application help', 'type': 'comment'},
        {'id': '4', 'text': 'medical question for doctors', 'type': 'post'},
        {'id': '5', 'text': 'pharmacy prescription refill', 'type': 'comment'},
    ]
    
    # Generate embeddings
    embedding_generator = EmbeddingGenerator()
    texts = [item['text'] for item in sample_data]
    embeddings = embedding_generator.generate_embeddings(texts)
    
    # Perform clustering
    cluster_analyzer = ClusterAnalyzer(algorithm='kmeans', n_clusters=2)
    cluster_labels = cluster_analyzer.fit(embeddings)
    
    # Get cluster statistics
    stats = cluster_analyzer.get_cluster_statistics()
    print(f"Clustering results: {stats}")
    
    # Show which items belong to which cluster
    for i, (item, label) in enumerate(zip(sample_data, cluster_labels)):
        print(f"Item {i+1}: '{item['text']}' -> Cluster {label}")

def example_analysis():
    """Example: Generate analysis and visualizations."""
    print("\n📊 Example: Analysis and Visualization")
    print("-" * 40)
    
    # Sample data
    sample_data = [
        {'subreddit': 'Health', 'score': 10, 'word_count': 25, 'type': 'post'},
        {'subreddit': 'Medicare', 'score': 5, 'word_count': 30, 'type': 'post'},
        {'subreddit': 'Health', 'score': 8, 'word_count': 20, 'type': 'comment'},
        {'subreddit': 'Medicaid', 'score': 12, 'word_count': 35, 'type': 'post'},
    ]
    
    # Initialize analyzer
    analyzer = RedditAnalyzer()
    
    # Analyze data distribution
    analysis = analyzer.analyze_data_distribution(sample_data)
    print("Data distribution analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    print("\nNote: Full visualizations would be saved to the outputs/ directory")

def main():
    """Run all examples."""
    print("🚀 Reddit Analysis Pipeline - Component Examples")
    print("=" * 60)
    
    # Check if Reddit credentials are available
    if not REDDIT_CONFIG['client_id'] or not REDDIT_CONFIG['client_secret']:
        print("⚠️  Reddit API credentials not found. Skipping data collection example.")
        print("   Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
        print()
    
    try:
        # Run examples
        if REDDIT_CONFIG['client_id'] and REDDIT_CONFIG['client_secret']:
            example_data_collection()
        
        example_text_preprocessing()
        example_embedding_generation()
        example_clustering()
        example_analysis()
        
        print("\n✅ All examples completed successfully!")
        print("\nTo run the full pipeline, use: python main.py")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        logger.error(f"Error in examples: {e}", exc_info=True)

if __name__ == "__main__":
    main()
