#!/usr/bin/env python3
"""
Test script for intelligent healthcare filtering system.
"""
import os
import sys
import logging
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_collection.intelligent_filter import IntelligentHealthcareFilter
from data_collection.reddit_client import RedditClient
from config import REDDIT_CONFIG, SUBREDDITS, INTELLIGENT_FILTERING

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_intelligent_filter():
    """Test the intelligent filtering functionality."""
    logger.info("Testing intelligent healthcare filter...")
    
    # Initialize filter
    filter_obj = IntelligentHealthcareFilter()
    
    # Test data samples
    test_posts = [
        {
            'id': 'test1',
            'title': 'Insurance denied my claim for MRI',
            'text': 'My insurance company denied coverage for an MRI that my doctor ordered. I need help understanding why and how to appeal this decision.',
            'author': 'user1',
            'subreddit': 'healthinsurance',
            'type': 'post'
        },
        {
            'id': 'test2', 
            'title': 'Breaking: New healthcare policy announced',
            'text': 'According to sources, a new healthcare policy will be announced tomorrow. This could affect millions of Americans.',
            'author': 'news_bot',
            'subreddit': 'Health',
            'type': 'post'
        },
        {
            'id': 'test3',
            'title': 'My experience with Medicare coverage',
            'text': 'I wanted to share my personal experience with Medicare coverage. It has been challenging but I found some helpful resources.',
            'author': 'user2',
            'subreddit': 'Medicare',
            'type': 'post'
        },
        {
            'id': 'test4',
            'title': 'What is the best insurance plan?',
            'text': 'I am looking for advice on choosing the best health insurance plan. What should I consider?',
            'author': 'user3',
            'subreddit': 'healthinsurance',
            'type': 'post'
        }
    ]
    
    test_comments = [
        {
            'id': 'comment1',
            'text': 'I had the same issue with my insurance. Here is what I did to appeal...',
            'author': 'user4',
            'post_id': 'test1',
            'type': 'comment'
        },
        {
            'id': 'comment2',
            'text': 'This is a bot message. Please read the community guidelines.',
            'author': 'AutoModerator',
            'post_id': 'test1',
            'type': 'comment'
        },
        {
            'id': 'comment3',
            'text': 'I need help understanding my deductible and copay structure.',
            'author': 'user5',
            'post_id': 'test4',
            'type': 'comment'
        }
    ]
    
    # Test post prioritization
    logger.info("Testing post prioritization...")
    prioritized_posts = filter_obj.prioritize_content(test_posts)
    
    logger.info(f"Original posts: {len(test_posts)}")
    logger.info(f"Prioritized posts: {len(prioritized_posts)}")
    
    for post in prioritized_posts:
        logger.info(f"Post: {post['title'][:50]}... | Relevance: {post.get('relevance_score', 0):.3f} | Healthcare: {post.get('is_healthcare_relevant', False)}")
    
    # Test comment filtering
    logger.info("\nTesting comment filtering...")
    filtered_comments = filter_obj.filter_comments(test_comments)
    
    logger.info(f"Original comments: {len(test_comments)}")
    logger.info(f"Filtered comments: {len(filtered_comments)}")
    
    for comment in filtered_comments:
        logger.info(f"Comment: {comment['text'][:50]}... | Relevance: {comment.get('relevance_score', 0):.3f} | Healthcare: {comment.get('is_healthcare_relevant', False)}")
    
    # Test filtering statistics
    all_data = prioritized_posts + filtered_comments
    stats = filter_obj.get_filtering_stats(all_data)
    logger.info(f"\nFiltering statistics: {stats}")
    
    return True


def test_reddit_client_integration():
    """Test Reddit client with intelligent filtering."""
    logger.info("Testing Reddit client integration...")
    
    # Check if credentials are available
    if not all([REDDIT_CONFIG['client_id'], REDDIT_CONFIG['client_secret']]):
        logger.warning("Reddit credentials not found. Skipping Reddit client test.")
        return False
    
    try:
        # Initialize Reddit client with intelligent filtering
        client = RedditClient(
            client_id=REDDIT_CONFIG['client_id'],
            client_secret=REDDIT_CONFIG['client_secret'],
            user_agent=REDDIT_CONFIG['user_agent'],
            filter_noise=True,
            intelligent_filtering=True
        )
        
        logger.info("Reddit client with intelligent filtering initialized successfully")
        
        # Test collection strategy
        strategy = client.intelligent_filter.get_collection_strategy(target_samples=1000)
        logger.info(f"Collection strategy: {strategy}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing Reddit client: {e}")
        return False


def test_relevance_scoring():
    """Test relevance scoring with various text samples."""
    logger.info("Testing relevance scoring...")
    
    filter_obj = IntelligentHealthcareFilter()
    
    test_cases = [
        ("Insurance denied my MRI claim", "My insurance company denied coverage for an MRI. I need help appealing this decision.", "High relevance - claim denial"),
        ("Breaking news about healthcare", "According to sources, new healthcare legislation will be announced tomorrow.", "Low relevance - news content"),
        ("My experience with Medicare", "I wanted to share my personal experience with Medicare coverage and how it helped me.", "Medium relevance - personal experience"),
        ("What is the best insurance?", "I am looking for advice on choosing health insurance. What should I consider?", "Medium relevance - question seeking help"),
        ("Bot message", "This is an automated message. Please read the community guidelines.", "Low relevance - bot content")
    ]
    
    for title, text, description in test_cases:
        score = filter_obj.calculate_relevance_score(text, title)
        is_quality = filter_obj.is_high_quality_content(text, title)
        logger.info(f"{description}: Score={score:.3f}, Quality={is_quality}")


def main():
    """Run all tests."""
    logger.info("Starting intelligent filtering tests...")
    
    try:
        # Test 1: Intelligent filter functionality
        test_intelligent_filter()
        
        # Test 2: Relevance scoring
        test_relevance_scoring()
        
        # Test 3: Reddit client integration
        test_reddit_client_integration()
        
        logger.info("All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
