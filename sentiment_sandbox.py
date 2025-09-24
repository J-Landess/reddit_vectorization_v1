#!/usr/bin/env python3
import os
import sys
import random
from typing import List, Tuple, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment import VaderSentimentAnalyzer, TransformerSentimentAnalyzer
from database.database_manager import DatabaseManager
from config import DATABASE_CONFIG


SAMPLE_POSTS: List[str] = [
    "I absolutely love this new feature! Great job everyone.",
    "This is terrible. The update broke my workflow and I'm frustrated.",
    "It's okay, nothing special but it works as expected.",
    "I'm not sure how I feel about this. Mixed thoughts.",
    "Fantastic performance improvements, the app feels so much faster!",
    "Worst experience ever. Crashed three times in five minutes.",
]


def get_random_reddit_samples(count: int = 10) -> List[Tuple[str, str, str]]:
    """
    Get random Reddit posts/comments from database.
    
    Returns:
        List of (text, type, subreddit) tuples
    """
    samples = []
    
    try:
        with DatabaseManager(DATABASE_CONFIG['path']) as db:
            # Get random posts
            posts_query = '''
                SELECT cleaned_text, title, subreddit FROM posts 
                WHERE cleaned_text IS NOT NULL AND cleaned_text != '' 
                ORDER BY RANDOM() LIMIT ?
            '''
            cursor = db.connection.cursor()
            cursor.execute(posts_query, (count // 2,))
            posts = cursor.fetchall()
            
            for post in posts:
                text = post[0] or post[1] or ""  # Use cleaned_text or title
                if text.strip():
                    samples.append((text, "post", post[2]))
            
            # Get random comments
            comments_query = '''
                SELECT cleaned_text, subreddit FROM comments 
                WHERE cleaned_text IS NOT NULL AND cleaned_text != '' 
                ORDER BY RANDOM() LIMIT ?
            '''
            cursor.execute(comments_query, (count - len(samples),))
            comments = cursor.fetchall()
            
            for comment in comments:
                text = comment[0] or ""
                if text.strip():
                    samples.append((text, "comment", comment[1]))
                    
    except Exception as e:
        print(f"Warning: Could not load Reddit data from database: {e}")
        print("Falling back to hard-coded samples only.")
    
    return samples


def analyze_texts(analyzer_name: str, analyzer, texts: List[Tuple[str, str, str]]) -> None:
    """Analyze texts and print results."""
    print(f"\n=== {analyzer_name} Results ===")
    
    for i, (text, data_type, subreddit) in enumerate(texts, 1):
        # Show full text (no truncation)
        display_text = text
        
        label, conf = analyzer.analyze(text)
        print(f"[{i:2d}] {label:8s}  conf={conf:0.3f}  | {data_type:7s} | r/{subreddit:15s} | {display_text}")


def main() -> int:
    print("\n" + "="*80)
    print("🔍 SENTIMENT ANALYSIS SANDBOX")
    print("="*80)
    
    # Initialize analyzers
    print("\nInitializing analyzers...")
    vader = VaderSentimentAnalyzer()
    transformer = TransformerSentimentAnalyzer()
    print("✅ Analyzers ready")
    
    # Test hard-coded samples
    print(f"\n📝 Testing {len(SAMPLE_POSTS)} hard-coded samples:")
    hardcoded_samples = [(text, "sample", "test") for text in SAMPLE_POSTS]
    
    analyze_texts("VADER", vader, hardcoded_samples)
    analyze_texts("Transformer (distilbert-sst2)", transformer, hardcoded_samples)
    
    # Test random Reddit data
    print(f"\n🔴 Testing random Reddit data:")
    reddit_samples = get_random_reddit_samples(10)
    
    if reddit_samples:
        print(f"Found {len(reddit_samples)} random Reddit samples")
        analyze_texts("VADER", vader, reddit_samples)
        analyze_texts("Transformer (distilbert-sst2)", transformer, reddit_samples)
    else:
        print("No Reddit data available - database may be empty or inaccessible")
    
    print(f"\n" + "="*80)
    print("📋 Summary:")
    print(f"• Hard-coded samples: {len(SAMPLE_POSTS)}")
    print(f"• Reddit samples: {len(reddit_samples)}")
    print(f"• Total analyzed: {len(SAMPLE_POSTS) + len(reddit_samples)}")
    print("• Note: Sandbox mode does not write to DB or CSV")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


