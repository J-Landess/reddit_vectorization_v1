#!/usr/bin/env python3
"""
Simple classification test - just tests the core functionality.
No database, no CSV, just prints results to screen.
"""
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.classification import RuleBasedClassifier, HybridClassifier

def main():
    print("🧪 SIMPLE CLASSIFICATION TEST")
    print("="*50)
    
    # Test texts
    test_texts = [
        "I need help choosing a health insurance plan for my family",
        "My doctor recommended a specialist for my condition", 
        "I'm an insurance broker and can help you compare plans",
        "Our company is updating our employee benefits package",
        "The deductible on my insurance is $5000",
        "I work as a nurse at the local hospital",
        "As a benefits consultant, I recommend reviewing coverage",
        "The employer-sponsored health plan covers 80% of expenses"
    ]
    
    # Test rule-based classifier
    print("\n1. RULE-BASED CLASSIFIER")
    print("-" * 30)
    rule_classifier = RuleBasedClassifier()
    
    for i, text in enumerate(test_texts, 1):
        category, confidence, probs = rule_classifier.classify(text)
        print(f"{i}. {text[:50]}...")
        print(f"   → {rule_classifier.get_category_label(category)} (conf: {confidence:.3f})")
        print()
    
    # Test hybrid classifier
    print("\n2. HYBRID CLASSIFIER")
    print("-" * 30)
    hybrid_classifier = HybridClassifier()
    
    for i, text in enumerate(test_texts, 1):
        # Mock embedding
        embedding = [0.1 * i] * 384
        category, confidence, probs = hybrid_classifier.classify(text, embedding)
        print(f"{i}. {text[:50]}...")
        print(f"   → {hybrid_classifier.get_category_label(category)} (conf: {confidence:.3f})")
        print()
    
    print("✅ Test completed successfully!")

if __name__ == "__main__":
    main()
