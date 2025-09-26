#!/usr/bin/env python3
"""
Quick test with real data from database - just shows a few examples.
"""
import os
import sys
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.classification import HybridClassifier
from src.database.database_manager import DatabaseManager

def main():
    print("🧪 QUICK REAL DATA CLASSIFICATION TEST")
    print("="*50)
    
    # Check database
    db_path = './data/reddit_data.db'
    if not os.path.exists(db_path):
        print(f"❌ Database not found at {db_path}")
        return
    
    try:
        # Get sample data
        db_manager = DatabaseManager(db_path)
        all_data = db_manager.get_all_data(include_embeddings=True)
        db_manager.close()
        
        if not all_data:
            print("❌ No data in database")
            return
        
        # Filter and sample
        valid_data = [item for item in all_data if item.get('cleaned_text') or item.get('text')]
        sample_data = random.sample(valid_data, min(10, len(valid_data)))
        
        print(f"📊 Testing on {len(sample_data)} real items from your database")
        print()
        
        # Test classification
        classifier = HybridClassifier()
        
        for i, item in enumerate(sample_data, 1):
            text = item.get('cleaned_text') or item.get('text', '')
            embedding = item.get('embedding')
            
            category, confidence, probs = classifier.classify(text, embedding)
            
            print(f"{i:2d}. r/{item.get('subreddit', 'unknown'):15} | "
                  f"{classifier.get_category_label(category):20} | "
                  f"Conf: {confidence:.3f}")
            print(f"    {text[:80]}{'...' if len(text) > 80 else ''}")
            print()
        
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
