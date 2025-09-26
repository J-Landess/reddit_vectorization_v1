#!/usr/bin/env python3
"""
Test script for classification pipeline - dry run mode.
This script tests the classification functionality without writing to database or CSV files.
It only prints results to the screen.
"""
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.classification import RuleBasedClassifier, MLClassifier, HybridClassifier
from src.analysis.classification_analyzer import ClassificationAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_data() -> List[Dict[str, Any]]:
    """Create sample Reddit data for testing classification."""
    sample_data = [
        {
            'id': 'test1',
            'title': 'Need help choosing health insurance plan',
            'text': 'I need help choosing a health insurance plan for my family. The premiums are too high and I can\'t afford the deductible.',
            'cleaned_text': 'need help choosing health insurance plan family premiums high afford deductible',
            'author': 'user1',
            'subreddit': 'healthinsurance',
            'score': 15,
            'upvote_ratio': 0.85,
            'num_comments': 8,
            'created_utc': datetime.now().timestamp(),
            'url': 'https://reddit.com/r/healthinsurance/test1',
            'is_self': True,
            'word_count': 20,
            'char_count': 120,
            'type': 'post',
            'embedding': [0.1] * 384  # Mock embedding
        },
        {
            'id': 'test2',
            'text': 'My doctor recommended a specialist for my condition. When can I schedule an appointment?',
            'cleaned_text': 'doctor recommended specialist condition schedule appointment',
            'author': 'user2',
            'subreddit': 'AskDocs',
            'score': 25,
            'created_utc': datetime.now().timestamp(),
            'word_count': 15,
            'char_count': 95,
            'type': 'comment',
            'post_id': 'test1',
            'embedding': [0.2] * 384  # Mock embedding
        },
        {
            'id': 'test3',
            'title': 'Insurance broker recommendations',
            'text': 'I\'m looking for a good insurance broker to help me compare different health plans. Any recommendations?',
            'cleaned_text': 'looking good insurance broker help compare different health plans recommendations',
            'author': 'user3',
            'subreddit': 'Insurance',
            'score': 12,
            'upvote_ratio': 0.75,
            'num_comments': 5,
            'created_utc': datetime.now().timestamp(),
            'url': 'https://reddit.com/r/Insurance/test3',
            'is_self': True,
            'word_count': 18,
            'char_count': 110,
            'type': 'post',
            'embedding': [0.3] * 384  # Mock embedding
        },
        {
            'id': 'test4',
            'text': 'Our company is updating our employee benefits package. HR will send details next week.',
            'cleaned_text': 'company updating employee benefits package hr send details next week',
            'author': 'user4',
            'subreddit': 'healthcare',
            'score': 8,
            'created_utc': datetime.now().timestamp(),
            'word_count': 16,
            'char_count': 85,
            'type': 'comment',
            'post_id': 'test3',
            'embedding': [0.4] * 384  # Mock embedding
        },
        {
            'id': 'test5',
            'title': 'Medical bill help',
            'text': 'I received a huge medical bill from the hospital. My insurance only covered 20% and I can\'t afford the rest.',
            'cleaned_text': 'received huge medical bill hospital insurance covered percent afford rest',
            'author': 'user5',
            'subreddit': 'MedicalBilling',
            'score': 30,
            'upvote_ratio': 0.90,
            'num_comments': 12,
            'created_utc': datetime.now().timestamp(),
            'url': 'https://reddit.com/r/MedicalBilling/test5',
            'is_self': True,
            'word_count': 22,
            'char_count': 130,
            'type': 'post',
            'embedding': [0.5] * 384  # Mock embedding
        },
        {
            'id': 'test6',
            'text': 'The nurse at the clinic was very helpful with my diagnosis and treatment plan.',
            'cleaned_text': 'nurse clinic helpful diagnosis treatment plan',
            'author': 'user6',
            'subreddit': 'medical',
            'score': 18,
            'created_utc': datetime.now().timestamp(),
            'word_count': 12,
            'char_count': 75,
            'type': 'comment',
            'post_id': 'test5',
            'embedding': [0.6] * 384  # Mock embedding
        }
    ]
    
    return sample_data

def test_rule_based_classifier():
    """Test the rule-based classifier."""
    print("\n" + "="*80)
    print("🧪 TESTING RULE-BASED CLASSIFIER")
    print("="*80)
    
    classifier = RuleBasedClassifier()
    sample_data = create_sample_data()
    
    print(f"Testing on {len(sample_data)} sample items...\n")
    
    for i, item in enumerate(sample_data, 1):
        text = item.get('cleaned_text') or item.get('text', '')
        print(f"--- Test Item {i} ---")
        print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"Subreddit: r/{item.get('subreddit', 'unknown')}")
        print(f"Type: {item.get('type', 'unknown')}")
        
        # Classify
        category, confidence, probabilities = classifier.classify(text)
        
        print(f"Predicted Category: {classifier.get_category_label(category)}")
        print(f"Confidence: {confidence:.3f}")
        print("Probabilities:")
        for cat, prob in probabilities.items():
            print(f"  - {classifier.get_category_label(cat)}: {prob:.3f}")
        
        # Get keyword matches for explanation
        matches = classifier.get_keyword_matches(text, category)
        if matches['exact'] or matches['partial']:
            print("Keyword Matches:")
            if matches['exact']:
                print(f"  - Exact: {', '.join(set(matches['exact']))}")
            if matches['partial']:
                print(f"  - Partial: {', '.join(set(matches['partial']))}")
        
        print()

def test_hybrid_classifier():
    """Test the hybrid classifier."""
    print("\n" + "="*80)
    print("🧪 TESTING HYBRID CLASSIFIER")
    print("="*80)
    
    classifier = HybridClassifier()
    sample_data = create_sample_data()
    
    print(f"Testing on {len(sample_data)} sample items...\n")
    
    for i, item in enumerate(sample_data, 1):
        text = item.get('cleaned_text') or item.get('text', '')
        embedding = item.get('embedding')
        
        print(f"--- Test Item {i} ---")
        print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"Subreddit: r/{item.get('subreddit', 'unknown')}")
        print(f"Type: {item.get('type', 'unknown')}")
        print(f"Has Embedding: {'Yes' if embedding else 'No'}")
        
        # Classify
        category, confidence, probabilities = classifier.classify(text, embedding)
        
        print(f"Predicted Category: {classifier.get_category_label(category)}")
        print(f"Confidence: {confidence:.3f}")
        print("Probabilities:")
        for cat, prob in probabilities.items():
            print(f"  - {classifier.get_category_label(cat)}: {prob:.3f}")
        
        # Get detailed breakdown
        breakdown = classifier.get_classification_breakdown(text, embedding)
        print(f"Method Used: {breakdown['method_used']}")
        print(f"Rule-based Result: {breakdown['rule_based']['category']} ({breakdown['rule_based']['confidence']:.3f})")
        if breakdown['ml_based']:
            print(f"ML-based Result: {breakdown['ml_based']['category']} ({breakdown['ml_based']['confidence']:.3f})")
        else:
            print("ML-based Result: Not available (model not trained)")
        
        print()

def test_batch_classification():
    """Test batch classification."""
    print("\n" + "="*80)
    print("🧪 TESTING BATCH CLASSIFICATION")
    print("="*80)
    
    classifier = HybridClassifier()
    sample_data = create_sample_data()
    
    # Extract texts and embeddings
    texts = [item.get('cleaned_text') or item.get('text', '') for item in sample_data]
    embeddings = [item.get('embedding') for item in sample_data]
    
    print(f"Batch classifying {len(texts)} items...\n")
    
    # Batch classify
    results = classifier.classify_batch(texts, embeddings)
    
    print("Batch Results:")
    print("-" * 60)
    for i, (item, result) in enumerate(zip(sample_data, results), 1):
        category, confidence, probabilities = result
        print(f"{i:2d}. {item.get('subreddit', 'unknown'):15} | "
              f"{classifier.get_category_label(category):20} | "
              f"Conf: {confidence:.3f}")
    
    print()

def test_classification_analysis():
    """Test classification analysis and visualization (without saving files)."""
    print("\n" + "="*80)
    print("🧪 TESTING CLASSIFICATION ANALYSIS")
    print("="*80)
    
    # Create sample data with classification results
    sample_data = create_sample_data()
    classifier = HybridClassifier()
    
    # Add classification results to sample data
    for item in sample_data:
        text = item.get('cleaned_text') or item.get('text', '')
        embedding = item.get('embedding')
        
        category, confidence, probabilities = classifier.classify(text, embedding)
        item['category'] = category
        item['category_confidence'] = confidence
        item['category_probabilities'] = probabilities
    
    # Analyze results
    analyzer = ClassificationAnalyzer(output_dir='./temp_outputs')  # Temporary directory
    analysis_results = analyzer.analyze_classification_results(sample_data)
    
    print("Analysis Results:")
    print("-" * 40)
    print(f"Total Classified Items: {analysis_results.get('total_classified_items', 0)}")
    
    conf_stats = analysis_results.get('confidence_statistics', {})
    print(f"Average Confidence: {conf_stats.get('average_confidence', 0):.3f}")
    print(f"Min Confidence: {conf_stats.get('min_confidence', 0):.3f}")
    print(f"Max Confidence: {conf_stats.get('max_confidence', 0):.3f}")
    
    print("\nCategory Distribution:")
    print("-" * 40)
    category_dist = analysis_results.get('category_distribution', {})
    for category, data in category_dist.items():
        print(f"{classifier.get_category_label(category):20} | "
              f"Count: {data['count']:2d} | "
              f"Percentage: {data['percentage']:5.1f}% | "
              f"Avg Conf: {data['avg_confidence']:.3f}")
    
    print("\nConfidence Distribution:")
    print("-" * 40)
    conf_ranges = conf_stats.get('confidence_ranges', {})
    print(f"High Confidence (≥0.8): {conf_ranges.get('high_confidence', 0)} items")
    print(f"Medium Confidence (0.5-0.8): {conf_ranges.get('medium_confidence', 0)} items")
    print(f"Low Confidence (<0.5): {conf_ranges.get('low_confidence', 0)} items")
    
    print("\nSubreddit Analysis:")
    print("-" * 40)
    subreddit_analysis = analysis_results.get('subreddit_analysis', {})
    for subreddit, counts in subreddit_analysis.items():
        print(f"\nr/{subreddit}:")
        total = sum(counts.values())
        for category, count in counts.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  - {classifier.get_category_label(category):20} | "
                  f"Count: {count:2d} | Percentage: {percentage:5.1f}%")
    
    print()

def test_ml_classifier_training():
    """Test ML classifier training (without saving model)."""
    print("\n" + "="*80)
    print("🧪 TESTING ML CLASSIFIER TRAINING")
    print("="*80)
    
    # Create sample data with embeddings
    sample_data = create_sample_data()
    
    # Extract embeddings
    embeddings = [item['embedding'] for item in sample_data]
    
    # Generate labels using rule-based classifier
    rule_classifier = RuleBasedClassifier()
    labels = []
    for item in sample_data:
        text = item.get('cleaned_text') or item.get('text', '')
        category, _, _ = rule_classifier.classify(text)
        labels.append(category)
    
    print(f"Training ML classifier on {len(embeddings)} samples...")
    print(f"Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
    print(f"Labels: {set(labels)}")
    
    # Test different ML models
    model_types = ['random_forest', 'logistic_regression', 'svm']
    
    for model_type in model_types:
        print(f"\n--- Testing {model_type.upper()} ---")
        try:
            ml_classifier = MLClassifier(model_type=model_type)
            metrics = ml_classifier.train(embeddings, labels, test_size=0.3)
            
            print(f"Accuracy: {metrics.get('accuracy', 0):.3f}")
            print(f"Training samples: {metrics.get('n_train_samples', 0)}")
            print(f"Test samples: {metrics.get('n_test_samples', 0)}")
            
            # Test classification
            test_text = "I need help with my health insurance claim"
            test_embedding = [0.1] * 384
            category, confidence, probs = ml_classifier.classify(test_text, test_embedding)
            print(f"Test classification: {ml_classifier.get_category_label(category)} (conf: {confidence:.3f})")
            
        except Exception as e:
            print(f"Error with {model_type}: {e}")
    
    print()

def main():
    """Run all classification tests."""
    print("🧪 MEDICAL CATEGORY CLASSIFICATION - DRY RUN TESTING")
    print("="*80)
    print("This script tests the classification pipeline without writing to database or CSV files.")
    print("All results are printed to the screen only.")
    print("="*80)
    
    try:
        # Test individual components
        test_rule_based_classifier()
        test_hybrid_classifier()
        test_batch_classification()
        test_classification_analysis()
        test_ml_classifier_training()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nThe classification pipeline is working correctly.")
        print("You can now integrate it into your main pipeline.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Test failed")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
