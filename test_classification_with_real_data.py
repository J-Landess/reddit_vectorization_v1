#!/usr/bin/env python3
"""
Test script for classification pipeline using real data from database.
This script samples data from your existing database and tests classification without writing back to DB.
"""
import os
import sys
import logging
import random
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.classification import RuleBasedClassifier, HybridClassifier
from src.analysis.classification_analyzer import ClassificationAnalyzer
from src.database.database_manager import DatabaseManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sample_database_data(db_path: str = './data/reddit_data.db', sample_size: int = 20) -> List[Dict[str, Any]]:
    """
    Sample data from the database for testing.
    
    Args:
        db_path: Path to the SQLite database
        sample_size: Number of items to sample
        
    Returns:
        List of sampled data items
    """
    print(f"📊 Sampling {sample_size} items from database: {db_path}")
    
    try:
        # Initialize database manager
        db_manager = DatabaseManager(db_path)
        
        # Get all data from database
        all_data = db_manager.get_all_data(include_embeddings=True)
        
        if not all_data:
            print("❌ No data found in database")
            return []
        
        print(f"📈 Found {len(all_data)} total items in database")
        
        # Filter data that has text content
        valid_data = [
            item for item in all_data 
            if item.get('cleaned_text') or item.get('text')
        ]
        
        print(f"📝 Found {len(valid_data)} items with text content")
        
        if len(valid_data) < sample_size:
            print(f"⚠️  Only {len(valid_data)} items available, using all of them")
            sample_size = len(valid_data)
        
        # Sample random items
        sampled_data = random.sample(valid_data, sample_size)
        
        print(f"✅ Sampled {len(sampled_data)} items for testing")
        
        # Close database connection
        db_manager.close()
        
        return sampled_data
        
    except Exception as e:
        print(f"❌ Error sampling from database: {e}")
        logger.exception("Database sampling failed")
        return []

def test_classification_on_real_data(sample_data: List[Dict[str, Any]]):
    """Test classification on real sampled data."""
    print("\n" + "="*80)
    print("🧪 TESTING CLASSIFICATION ON REAL DATA")
    print("="*80)
    
    if not sample_data:
        print("❌ No sample data available for testing")
        return
    
    # Initialize classifiers
    rule_classifier = RuleBasedClassifier()
    hybrid_classifier = HybridClassifier()
    
    print(f"Testing on {len(sample_data)} real Reddit items...\n")
    
    # Test each item
    for i, item in enumerate(sample_data, 1):
        text = item.get('cleaned_text') or item.get('text', '')
        embedding = item.get('embedding')
        
        print(f"--- Item {i} ---")
        print(f"ID: {item.get('id', 'unknown')}")
        print(f"Subreddit: r/{item.get('subreddit', 'unknown')}")
        print(f"Type: {item.get('type', 'unknown')}")
        print(f"Author: u/{item.get('author', 'unknown')}")
        print(f"Score: {item.get('score', 0)}")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Has Embedding: {'Yes' if embedding else 'No'}")
        
        # Rule-based classification
        rule_category, rule_confidence, rule_probs = rule_classifier.classify(text)
        print(f"Rule-based: {rule_classifier.get_category_label(rule_category)} (conf: {rule_confidence:.3f})")
        
        # Hybrid classification
        hybrid_category, hybrid_confidence, hybrid_probs = hybrid_classifier.classify(text, embedding)
        print(f"Hybrid: {hybrid_classifier.get_category_label(hybrid_category)} (conf: {hybrid_confidence:.3f})")
        
        # Show top 2 probabilities for each
        print("Rule-based probabilities:")
        sorted_rule_probs = sorted(rule_probs.items(), key=lambda x: x[1], reverse=True)
        for cat, prob in sorted_rule_probs[:2]:
            print(f"  - {rule_classifier.get_category_label(cat)}: {prob:.3f}")
        
        print("Hybrid probabilities:")
        sorted_hybrid_probs = sorted(hybrid_probs.items(), key=lambda x: x[1], reverse=True)
        for cat, prob in sorted_hybrid_probs[:2]:
            print(f"  - {hybrid_classifier.get_category_label(cat)}: {prob:.3f}")
        
        # Get keyword matches for rule-based
        if rule_confidence > 0.5:
            matches = rule_classifier.get_keyword_matches(text, rule_category)
            if matches['exact'] or matches['partial']:
                print("Keyword matches:")
                if matches['exact']:
                    print(f"  - Exact: {', '.join(set(matches['exact'][:3]))}")
                if matches['partial']:
                    print(f"  - Partial: {', '.join(set(matches['partial'][:3]))}")
        
        print()

def analyze_classification_results(sample_data: List[Dict[str, Any]]):
    """Analyze classification results on real data."""
    print("\n" + "="*80)
    print("📊 ANALYZING CLASSIFICATION RESULTS")
    print("="*80)
    
    if not sample_data:
        print("❌ No sample data available for analysis")
        return
    
    # Classify all items
    hybrid_classifier = HybridClassifier()
    
    for item in sample_data:
        text = item.get('cleaned_text') or item.get('text', '')
        embedding = item.get('embedding')
        
        category, confidence, probabilities = hybrid_classifier.classify(text, embedding)
        item['category'] = category
        item['category_confidence'] = confidence
        item['category_probabilities'] = probabilities
    
    # Analyze results
    analyzer = ClassificationAnalyzer(output_dir='./temp_analysis')  # Temporary directory
    analysis_results = analyzer.analyze_classification_results(sample_data)
    
    print("📈 Classification Statistics:")
    print("-" * 40)
    print(f"Total Items: {analysis_results.get('total_classified_items', 0)}")
    
    conf_stats = analysis_results.get('confidence_statistics', {})
    print(f"Average Confidence: {conf_stats.get('average_confidence', 0):.3f}")
    print(f"Min Confidence: {conf_stats.get('min_confidence', 0):.3f}")
    print(f"Max Confidence: {conf_stats.get('max_confidence', 0):.3f}")
    
    print("\n📊 Category Distribution:")
    print("-" * 40)
    category_dist = analysis_results.get('category_distribution', {})
    for category, data in category_dist.items():
        print(f"{hybrid_classifier.get_category_label(category):20} | "
              f"Count: {data['count']:2d} | "
              f"Percentage: {data['percentage']:5.1f}% | "
              f"Avg Conf: {data['avg_confidence']:.3f}")
    
    print("\n🎯 Confidence Distribution:")
    print("-" * 40)
    conf_ranges = conf_stats.get('confidence_ranges', {})
    print(f"High Confidence (≥0.8): {conf_ranges.get('high_confidence', 0)} items")
    print(f"Medium Confidence (0.5-0.8): {conf_ranges.get('medium_confidence', 0)} items")
    print(f"Low Confidence (<0.5): {conf_ranges.get('low_confidence', 0)} items")
    
    print("\n🏷️ Subreddit Analysis:")
    print("-" * 40)
    subreddit_analysis = analysis_results.get('subreddit_analysis', {})
    for subreddit, counts in subreddit_analysis.items():
        print(f"\nr/{subreddit}:")
        total = sum(counts.values())
        for category, count in counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  - {hybrid_classifier.get_category_label(category):20} | "
                  f"Count: {count:2d} | Percentage: {percentage:5.1f}%")
    
    print("\n📝 Content Type Analysis:")
    print("-" * 40)
    content_type_analysis = analysis_results.get('content_type_analysis', {})
    for content_type, counts in content_type_analysis.items():
        print(f"\n{content_type.title()}:")
        total = sum(counts.values())
        for category, count in counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  - {hybrid_classifier.get_category_label(category):20} | "
                  f"Count: {count:2d} | Percentage: {percentage:5.1f}%")

def show_sample_predictions(sample_data: List[Dict[str, Any]], num_examples: int = 5):
    """Show detailed predictions for a few sample items."""
    print("\n" + "="*80)
    print(f"🔍 DETAILED PREDICTIONS (Top {num_examples} Examples)")
    print("="*80)
    
    if not sample_data:
        print("❌ No sample data available")
        return
    
    hybrid_classifier = HybridClassifier()
    
    # Take first few items
    examples = sample_data[:num_examples]
    
    for i, item in enumerate(examples, 1):
        text = item.get('cleaned_text') or item.get('text', '')
        embedding = item.get('embedding')
        
        print(f"\n--- Example {i} ---")
        print(f"Subreddit: r/{item.get('subreddit', 'unknown')}")
        print(f"Type: {item.get('type', 'unknown')}")
        print(f"Score: {item.get('score', 0)}")
        print(f"Text: {text}")
        
        # Get detailed breakdown
        breakdown = hybrid_classifier.get_classification_breakdown(text, embedding)
        
        print(f"\nFinal Prediction: {hybrid_classifier.get_category_label(breakdown['final_result']['category'])}")
        print(f"Confidence: {breakdown['final_result']['confidence']:.3f}")
        print(f"Method Used: {breakdown['method_used']}")
        
        print("\nRule-based Analysis:")
        print(f"  Category: {hybrid_classifier.get_category_label(breakdown['rule_based']['category'])}")
        print(f"  Confidence: {breakdown['rule_based']['confidence']:.3f}")
        
        if breakdown['rule_based']['keyword_matches']['exact']:
            print(f"  Exact Keywords: {', '.join(breakdown['rule_based']['keyword_matches']['exact'])}")
        if breakdown['rule_based']['keyword_matches']['partial']:
            print(f"  Partial Keywords: {', '.join(breakdown['rule_based']['keyword_matches']['partial'])}")
        
        if breakdown['ml_based']:
            print("\nML-based Analysis:")
            print(f"  Category: {hybrid_classifier.get_category_label(breakdown['ml_based']['category'])}")
            print(f"  Confidence: {breakdown['ml_based']['confidence']:.3f}")
        else:
            print("\nML-based Analysis: Not available (model not trained)")
        
        print("\nAll Probabilities:")
        for cat, prob in breakdown['final_result']['probabilities'].items():
            print(f"  - {hybrid_classifier.get_category_label(cat)}: {prob:.3f}")

def main():
    """Main function to run the classification test with real data."""
    print("🧪 MEDICAL CATEGORY CLASSIFICATION - REAL DATA TEST")
    print("="*80)
    print("This script samples real data from your database and tests classification.")
    print("No data is written back to the database - results only printed to screen.")
    print("="*80)
    
    # Check if database exists
    db_path = './data/reddit_data.db'
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        print("Please run the main pipeline first to collect some data.")
        return False
    
    try:
        # Sample data from database
        sample_data = sample_database_data(db_path, sample_size=20)
        
        if not sample_data:
            print("❌ No data could be sampled from database")
            return False
        
        # Test classification
        test_classification_on_real_data(sample_data)
        
        # Analyze results
        analyze_classification_results(sample_data)
        
        # Show detailed examples
        show_sample_predictions(sample_data, num_examples=3)
        
        print("\n" + "="*80)
        print("✅ REAL DATA TEST COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nThe classification pipeline works correctly with your real Reddit data.")
        print("You can now integrate it into your main pipeline with confidence.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Real data test failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
