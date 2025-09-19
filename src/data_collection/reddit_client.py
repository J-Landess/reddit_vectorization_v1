"""
Reddit API client using PRAW for data collection.
"""
import praw
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class RedditClient:
    """Reddit API client for collecting posts and comments."""
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize Reddit client.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Test connection
        try:
            self.reddit.user.me()
            logger.info("Successfully connected to Reddit API")
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            raise
    
    def get_subreddit_posts(self, subreddit_name: str, limit: int = 100, 
                           time_filter: str = 'month') -> List[Dict[str, Any]]:
        """
        Collect posts from a specific subreddit.
        
        Args:
            subreddit_name: Name of the subreddit (without r/)
            limit: Maximum number of posts to collect
            time_filter: Time filter for posts ('day', 'week', 'month', 'year', 'all')
            
        Returns:
            List of post dictionaries
        """
        posts = []
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            
            for post in subreddit.top(time_filter=time_filter, limit=limit):
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'author': str(post.author) if post.author else '[deleted]',
                    'subreddit': subreddit_name,
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'num_comments': post.num_comments,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'url': post.url,
                    'is_self': post.is_self,
                    'type': 'post'
                }
                posts.append(post_data)
                
        except Exception as e:
            logger.error(f"Error collecting posts from r/{subreddit_name}: {e}")
            
        logger.info(f"Collected {len(posts)} posts from r/{subreddit_name}")
        return posts
    
    def get_post_comments(self, post_id: str, max_comments: int = 50) -> List[Dict[str, Any]]:
        """
        Collect comments from a specific post.
        
        Args:
            post_id: Reddit post ID
            max_comments: Maximum number of comments to collect
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        try:
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Remove "more comments" objects
            
            comment_count = 0
            for comment in submission.comments.list():
                if comment_count >= max_comments:
                    break
                    
                if hasattr(comment, 'body') and comment.body != '[deleted]':
                    comment_data = {
                        'id': comment.id,
                        'text': comment.body,
                        'author': str(comment.author) if comment.author else '[deleted]',
                        'post_id': post_id,
                        'score': comment.score,
                        'created_utc': datetime.fromtimestamp(comment.created_utc),
                        'type': 'comment'
                    }
                    comments.append(comment_data)
                    comment_count += 1
                    
        except Exception as e:
            logger.error(f"Error collecting comments for post {post_id}: {e}")
            
        logger.info(f"Collected {len(comments)} comments for post {post_id}")
        return comments
    
    def collect_subreddit_data(self, subreddit_name: str, max_posts: int = 100, 
                              max_comments_per_post: int = 50) -> List[Dict[str, Any]]:
        """
        Collect both posts and comments from a subreddit.
        
        Args:
            subreddit_name: Name of the subreddit
            max_posts: Maximum number of posts to collect
            max_comments_per_post: Maximum comments per post
            
        Returns:
            Combined list of posts and comments
        """
        all_data = []
        
        # Collect posts
        posts = self.get_subreddit_posts(subreddit_name, max_posts)
        all_data.extend(posts)
        
        # Collect comments for each post
        for post in posts[:max_posts]:  # Limit to avoid too many API calls
            comments = self.get_post_comments(post['id'], max_comments_per_post)
            all_data.extend(comments)
            
            # Rate limiting - be respectful to Reddit's API
            time.sleep(1)
            
        logger.info(f"Total data collected from r/{subreddit_name}: {len(all_data)} items")
        return all_data
    
    def collect_multiple_subreddits(self, subreddit_names: List[str], 
                                   max_posts_per_subreddit: int = 100,
                                   max_comments_per_post: int = 50) -> List[Dict[str, Any]]:
        """
        Collect data from multiple subreddits.
        
        Args:
            subreddit_names: List of subreddit names
            max_posts_per_subreddit: Maximum posts per subreddit
            max_comments_per_post: Maximum comments per post
            
        Returns:
            Combined list of all collected data
        """
        all_data = []
        
        for subreddit_name in subreddit_names:
            logger.info(f"Collecting data from r/{subreddit_name}")
            subreddit_data = self.collect_subreddit_data(
                subreddit_name, max_posts_per_subreddit, max_comments_per_post
            )
            all_data.extend(subreddit_data)
            
            # Rate limiting between subreddits
            time.sleep(2)
            
        logger.info(f"Total data collected from all subreddits: {len(all_data)} items")
        return all_data
