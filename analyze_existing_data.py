#!/usr/bin/env python3
"""
Analyze existing Reddit database data with sentiment analysis.
This script processes existing data without collecting new data from Reddit.
"""
import os
import sys
import sqlite3
from typing import List, Tuple, Optional
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment import VaderSentimentAnalyzer, TransformerSentimentAnalyzer
from config import DATABASE_CONFIG


def get_database_connection():
    """Get database connection."""
    return sqlite3.connect(DATABASE_CONFIG['path'])


def get_posts_without_sentiment(limit: Optional[int] = None) -> List[Tuple[int, str, str]]:
    """Get posts that don't have sentiment analysis yet."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Check if sentiment columns exist
    cursor.execute("PRAGMA table_info(posts)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'sentiment_label' not in columns or 'sentiment_score' not in columns:
        print("Adding sentiment columns to posts table...")
        cursor.execute("ALTER TABLE posts ADD COLUMN sentiment_label TEXT")
        cursor.execute("ALTER TABLE posts ADD COLUMN sentiment_score REAL")
        conn.commit()
    
    # Get posts without sentiment analysis
    query = """
        SELECT id, cleaned_text, title 
        FROM posts 
        WHERE (cleaned_text IS NOT NULL AND cleaned_text != '') 
        AND (sentiment_label IS NULL OR sentiment_score IS NULL)
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    posts = cursor.fetchall()
    conn.close()
    
    return posts


def get_comments_without_sentiment(limit: Optional[int] = None) -> List[Tuple[int, str]]:
    """Get comments that don't have sentiment analysis yet."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    # Check if sentiment columns exist
    cursor.execute("PRAGMA table_info(comments)")
    columns = [column[1] for column in cursor.fetchall()]
    
    if 'sentiment_label' not in columns or 'sentiment_score' not in columns:
        print("Adding sentiment columns to comments table...")
        cursor.execute("ALTER TABLE comments ADD COLUMN sentiment_label TEXT")
        cursor.execute("ALTER TABLE comments ADD COLUMN sentiment_score REAL")
        conn.commit()
    
    # Get comments without sentiment analysis
    query = """
        SELECT id, cleaned_text 
        FROM comments 
        WHERE cleaned_text IS NOT NULL AND cleaned_text != ''
        AND (sentiment_label IS NULL OR sentiment_score IS NULL)
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    comments = cursor.fetchall()
    conn.close()
    
    return comments


def update_post_sentiment(post_id: int, sentiment_label: str, sentiment_score: float):
    """Update post with sentiment analysis results."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE posts 
        SET sentiment_label = ?, sentiment_score = ?
        WHERE id = ?
    """, (sentiment_label, sentiment_score, post_id))
    
    conn.commit()
    conn.close()


def update_comment_sentiment(comment_id: int, sentiment_label: str, sentiment_score: float):
    """Update comment with sentiment analysis results."""
    conn = get_database_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE comments 
        SET sentiment_label = ?, sentiment_score = ?
        WHERE id = ?
    """, (sentiment_label, sentiment_score, comment_id))
    
    conn.commit()
    conn.close()


def analyze_existing_data(analyzer_type: str = 'vader', limit: Optional[int] = None):
    """Analyze existing database data with sentiment analysis."""
    print("=" * 80)
    print("🔍 ANALYZING EXISTING REDDIT DATA")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analyzer: {analyzer_type}")
    if limit:
        print(f"Limit: {limit} records per table")
    print()
    
    # Initialize analyzer
    print("Initializing sentiment analyzer...")
    if analyzer_type.lower() == 'transformer':
        analyzer = TransformerSentimentAnalyzer()
    else:
        analyzer = VaderSentimentAnalyzer()
    print("✅ Analyzer ready")
    
    # Process posts
    print("\n📝 Processing posts...")
    posts = get_posts_without_sentiment(limit)
    print(f"Found {len(posts)} posts to analyze")
    
    posts_processed = 0
    for i, (post_id, cleaned_text, title) in enumerate(posts, 1):
        # Use cleaned_text if available, otherwise use title
        text_to_analyze = cleaned_text if cleaned_text and cleaned_text.strip() else title
        
        if text_to_analyze and text_to_analyze.strip():
            try:
                sentiment_label, sentiment_score = analyzer.analyze(text_to_analyze)
                update_post_sentiment(post_id, sentiment_label, sentiment_score)
                posts_processed += 1
                
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(posts)} posts...")
                    
            except Exception as e:
                print(f"  Error processing post {post_id}: {e}")
    
    print(f"✅ Processed {posts_processed} posts")
    
    # Process comments
    print("\n💬 Processing comments...")
    comments = get_comments_without_sentiment(limit)
    print(f"Found {len(comments)} comments to analyze")
    
    comments_processed = 0
    for i, (comment_id, cleaned_text) in enumerate(comments, 1):
        if cleaned_text and cleaned_text.strip():
            try:
                sentiment_label, sentiment_score = analyzer.analyze(cleaned_text)
                update_comment_sentiment(comment_id, sentiment_label, sentiment_score)
                comments_processed += 1
                
                if i % 100 == 0:
                    print(f"  Processed {i}/{len(comments)} comments...")
                    
            except Exception as e:
                print(f"  Error processing comment {comment_id}: {e}")
    
    print(f"✅ Processed {comments_processed} comments")
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"Posts analyzed: {posts_processed}")
    print(f"Comments analyzed: {comments_processed}")
    print(f"Total analyzed: {posts_processed + comments_processed}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze existing Reddit database data with sentiment analysis')
    parser.add_argument('--analyzer', choices=['vader', 'transformer'], default='vader',
                       help='Sentiment analyzer to use (default: vader)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records to process per table (for testing)')
    
    args = parser.parse_args()
    
    try:
        analyze_existing_data(args.analyzer, args.limit)
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
