#!/usr/bin/env python3
"""
Test script for topic modeling implementation.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from topic_modeling.topic_analyzer import TopicAnalyzer
from topic_modeling.topic_pipeline import TopicModelingPipeline
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_topic_modeling():
    """Test topic modeling on Reddit healthcare data."""
    print("🧪 Testing Topic Modeling Implementation")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    try:
        df = pd.read_csv('csv_exports/reddit_all_data.csv')
        print(f"✅ Loaded {len(df):,} total records")
    except FileNotFoundError:
        print("❌ CSV file not found. Please run the pipeline first.")
        return
    
    # Get sample of texts for testing
    sample_size = min(2000, len(df))  # Use 2000 samples for testing
    sample_df = df.sample(n=sample_size, random_state=42)
    texts = sample_df['cleaned_text'].fillna('').tolist()
    
    print(f"Testing with {len(texts)} texts")
    
    # Test different algorithms
    algorithms = ['gensim_lda', 'lda', 'nmf']
    
    for algorithm in algorithms:
        print(f"\n🔍 Testing {algorithm.upper()} algorithm...")
        
        try:
            # Initialize analyzer
            analyzer = TopicAnalyzer(algorithm=algorithm, n_topics=10)
            
            # Fit model
            results = analyzer.fit(texts)
            print(f"✅ {algorithm} model fitted successfully")
            print(f"   Results: {results}")
            
            # Get topic words
            topic_words = analyzer.get_topic_words(n_words=5)
            print(f"   Found {len(topic_words)} topics")
            
            # Show first few topics
            for topic_id, words in list(topic_words.items())[:3]:
                word_list = [word for word, weight in words]
                print(f"   Topic {topic_id}: {', '.join(word_list)}")
            
            # Get statistics
            stats = analyzer.get_topic_statistics()
            print(f"   Statistics: {stats}")
            
        except Exception as e:
            print(f"❌ Error with {algorithm}: {e}")
            logger.error(f"Error testing {algorithm}: {e}")
    
    print("\n🎉 Topic modeling test completed!")

def test_topic_pipeline():
    """Test the complete topic modeling pipeline."""
    print("\n🚀 Testing Complete Topic Modeling Pipeline")
    print("=" * 50)
    
    # Load data
    print("📊 Loading data...")
    try:
        df = pd.read_csv('csv_exports/reddit_all_data.csv')
        print(f"✅ Loaded {len(df):,} total records")
    except FileNotFoundError:
        print("❌ CSV file not found. Please run the pipeline first.")
        return
    
    # Convert to data format expected by pipeline
    sample_size = min(1000, len(df))  # Use 1000 samples for pipeline test
    sample_df = df.sample(n=sample_size, random_state=42)
    
    data = []
    for _, row in sample_df.iterrows():
        item = {
            'id': row.get('id', ''),
            'type': row.get('type', ''),
            'text': row.get('text', ''),
            'cleaned_text': row.get('cleaned_text', ''),
            'subreddit': row.get('subreddit', ''),
            'score': row.get('score', 0),
            'author': row.get('author', ''),
            'created_utc': row.get('created_utc', '')
        }
        data.append(item)
    
    print(f"Testing pipeline with {len(data)} items")
    
    # Test pipeline
    try:
        pipeline = TopicModelingPipeline(algorithm='gensim_lda', n_topics=15)
        results = pipeline.run_topic_analysis(data)
        
        print("✅ Pipeline completed successfully")
        print(f"   Algorithm: {results['algorithm']}")
        print(f"   Documents processed: {results['n_documents_processed']}")
        print(f"   Topics found: {results['topic_statistics'].get('n_topics', 'N/A')}")
        
        # Show topic summaries
        topic_summaries = results['topic_summaries']
        print(f"\n📋 Topic Summaries (showing first 5):")
        for topic_id, summary in list(topic_summaries.items())[:5]:
            print(f"   Topic {topic_id}: {summary['top_words']} ({summary['size']} items)")
        
        # Generate report
        report_path = pipeline.generate_topic_report(results)
        print(f"\n📄 Report generated: {report_path}")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        logger.error(f"Pipeline error: {e}")

def test_optimal_topics():
    """Test finding optimal number of topics."""
    print("\n🎯 Testing Optimal Topics Finding")
    print("=" * 50)
    
    # Load data
    try:
        df = pd.read_csv('csv_exports/reddit_all_data.csv')
        sample_df = df.sample(n=min(500, len(df)), random_state=42)
        texts = sample_df['cleaned_text'].fillna('').tolist()
        
        print(f"Testing optimal topics with {len(texts)} texts")
        
        # Test with Gensim LDA
        analyzer = TopicAnalyzer(algorithm='gensim_lda', n_topics=10)
        optimal_analysis = analyzer.find_optimal_topics(texts, topic_range=range(5, 21, 5))
        
        if optimal_analysis:
            print(f"✅ Optimal topics analysis completed")
            print(f"   Optimal topics: {optimal_analysis['optimal_topics']}")
            print(f"   Optimal score: {optimal_analysis['optimal_score']:.4f}")
            print(f"   Metric: {optimal_analysis['metric']}")
        else:
            print("❌ Could not find optimal topics")
            
    except Exception as e:
        print(f"❌ Optimal topics error: {e}")
        logger.error(f"Optimal topics error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Topic Modeling Tests")
    print("=" * 60)
    
    # Run tests
    test_topic_modeling()
    test_topic_pipeline()
    test_optimal_topics()
    
    print("\n🎉 All topic modeling tests completed!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install gensim bertopic umap-learn")
    print("2. Run full analysis: python run_topic_analysis.py")
    print("3. Check outputs in the 'outputs' directory")
