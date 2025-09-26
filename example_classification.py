#!/usr/bin/env python3
"""
Example script demonstrating medical category classification functionality.
"""
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import RedditAnalysisPipeline
from src.classification import RuleBasedClassifier, MLClassifier, HybridClassifier
from src.analysis.classification_analyzer import ClassificationAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rule_based_classification():
    """Test the rule-based classifier with sample texts."""
    print("\n" + "="*60)
    print("TESTING RULE-BASED CLASSIFICATION")
    print("="*60)
    
    classifier = RuleBasedClassifier()
    
    # Sample texts for each category
    test_texts = [
        "I need help choosing a health insurance plan for my family. The premiums are too high!",
        "My doctor recommended a specialist for my condition. When can I schedule an appointment?",
        "I'm an insurance broker and can help you compare different health plans available in your area.",
        "Our company is updating our employee benefits package. HR will send details next week.",
        "The deductible on my insurance is $5000. I can't afford this medical bill.",
        "I work as a nurse at the local hospital. The patient care here is excellent.",
        "As a benefits consultant, I recommend reviewing your coverage options annually.",
        "The employer-sponsored health plan covers 80% of medical expenses for employees."
    ]
    
    for text in test_texts:
        category, confidence, probabilities = classifier.classify(text)
        print(f"\nText: {text[:80]}...")
        print(f"Category: {classifier.get_category_label(category)}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Probabilities: {probabilities}")

def test_hybrid_classification():
    """Test the hybrid classifier."""
    print("\n" + "="*60)
    print("TESTING HYBRID CLASSIFICATION")
    print("="*60)
    
    classifier = HybridClassifier()
    
    test_texts = [
        "I'm looking for affordable health insurance coverage for my small business employees.",
        "The medical provider at the clinic was very helpful with my diagnosis.",
        "Contact our insurance broker for personalized health plan recommendations.",
        "Our employer benefits include comprehensive health coverage and dental insurance."
    ]
    
    for text in test_texts:
        category, confidence, probabilities = classifier.classify(text)
        print(f"\nText: {text}")
        print(f"Category: {classifier.get_category_label(category)}")
        print(f"Confidence: {confidence:.3f}")
        
        # Get detailed breakdown
        breakdown = classifier.get_classification_breakdown(text)
        print(f"Method used: {breakdown['method_used']}")
        print(f"Rule-based: {breakdown['rule_based']['category']} ({breakdown['rule_based']['confidence']:.3f})")
        if breakdown['ml_based']:
            print(f"ML-based: {breakdown['ml_based']['category']} ({breakdown['ml_based']['confidence']:.3f})")

def test_pipeline_integration():
    """Test classification integration in the main pipeline."""
    print("\n" + "="*60)
    print("TESTING PIPELINE INTEGRATION")
    print("="*60)
    
    try:
        # Initialize pipeline
        pipeline = RedditAnalysisPipeline(analyzer_type='vader')
        
        # Check if classification analyzer is initialized
        if pipeline.classification_analyzer:
            print("✓ Classification analyzer initialized successfully")
            
            # Test with sample data
            sample_data = [
                {
                    'id': 'test1',
                    'text': 'I need help with my health insurance claim',
                    'cleaned_text': 'I need help with my health insurance claim',
                    'subreddit': 'healthinsurance',
                    'type': 'post',
                    'embedding': [0.1] * 384  # Mock embedding
                },
                {
                    'id': 'test2', 
                    'text': 'My doctor recommended a specialist',
                    'cleaned_text': 'My doctor recommended a specialist',
                    'subreddit': 'AskDocs',
                    'type': 'comment',
                    'embedding': [0.2] * 384  # Mock embedding
                }
            ]
            
            # Process data through pipeline
            processed_data = pipeline.preprocess_data(sample_data)
            
            print(f"\nProcessed {len(processed_data)} items:")
            for item in processed_data:
                print(f"- {item['text'][:50]}...")
                print(f"  Category: {item.get('category', 'N/A')}")
                print(f"  Confidence: {item.get('category_confidence', 0):.3f}")
                print(f"  Sentiment: {item.get('sentiment', 'N/A')}")
                print()
                
        else:
            print("✗ Classification analyzer not initialized")
            
    except Exception as e:
        print(f"✗ Error testing pipeline integration: {e}")

def test_classification_analysis():
    """Test classification analysis and visualization."""
    print("\n" + "="*60)
    print("TESTING CLASSIFICATION ANALYSIS")
    print("="*60)
    
    # Create sample data with classification results
    sample_data = [
        {
            'id': '1',
            'text': 'Health insurance is too expensive',
            'category': 'medical_insurance',
            'category_confidence': 0.85,
            'subreddit': 'healthinsurance',
            'type': 'post',
            'created_utc': datetime.now().timestamp()
        },
        {
            'id': '2',
            'text': 'My doctor was very helpful',
            'category': 'medical_provider',
            'category_confidence': 0.92,
            'subreddit': 'AskDocs',
            'type': 'comment',
            'created_utc': datetime.now().timestamp()
        },
        {
            'id': '3',
            'text': 'I can help you find the best insurance plan',
            'category': 'medical_broker',
            'category_confidence': 0.78,
            'subreddit': 'Insurance',
            'type': 'post',
            'created_utc': datetime.now().timestamp()
        },
        {
            'id': '4',
            'text': 'Our company benefits include health coverage',
            'category': 'employer',
            'category_confidence': 0.88,
            'subreddit': 'healthcare',
            'type': 'comment',
            'created_utc': datetime.now().timestamp()
        }
    ]
    
    # Initialize analyzer
    analyzer = ClassificationAnalyzer()
    
    # Analyze results
    analysis_results = analyzer.analyze_classification_results(sample_data)
    
    print("Analysis Results:")
    print(f"- Total classified items: {analysis_results.get('total_classified_items', 0)}")
    print(f"- Average confidence: {analysis_results.get('confidence_statistics', {}).get('average_confidence', 0):.3f}")
    
    print("\nCategory Distribution:")
    for category, data in analysis_results.get('category_distribution', {}).items():
        print(f"- {category}: {data['count']} items ({data['percentage']:.1f}%)")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizations = analyzer.create_classification_visualizations(sample_data, analysis_results)
    
    print(f"Created {len(visualizations)} visualizations:")
    for name, path in visualizations.items():
        print(f"- {name}: {path}")
    
    # Export report
    report_path = analyzer.export_classification_report(analysis_results)
    print(f"\nExported report: {report_path}")

def main():
    """Run all classification tests."""
    print("MEDICAL CATEGORY CLASSIFICATION TESTING")
    print("="*60)
    
    try:
        # Test individual components
        test_rule_based_classification()
        test_hybrid_classification()
        test_pipeline_integration()
        test_classification_analysis()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        logger.exception("Test failed")

if __name__ == "__main__":
    main()
