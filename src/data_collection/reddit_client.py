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
    
    def __init__(self, client_id: str, client_secret: str, user_agent: str, 
                 filter_noise: bool = True, intelligent_filtering: bool = True,
                 time_filter: str = 'month'):
        """
        Initialize Reddit client.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: User agent string for API requests
            filter_noise: Whether to filter out bot messages and guidelines
            intelligent_filtering: Whether to use intelligent healthcare filtering
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.filter_noise = filter_noise
        self.intelligent_filtering = intelligent_filtering
        self.time_filter = time_filter
        
        # Initialize intelligent filter if enabled
        if intelligent_filtering:
            try:
                from .intelligent_filter import IntelligentHealthcareFilter
                self.intelligent_filter = IntelligentHealthcareFilter()
                logger.info("Intelligent healthcare filtering enabled")
            except ImportError as e:
                logger.warning(f"Could not import intelligent filter: {e}")
                self.intelligent_filter = None
                self.intelligent_filtering = False
        else:
            self.intelligent_filter = None
        
        # Test connection
        try:
            self.reddit.user.me()
            logger.info("Successfully connected to Reddit API")
            logger.info(f"Noise filtering: {'enabled' if filter_noise else 'disabled'}")
            logger.info(f"Intelligent filtering: {'enabled' if intelligent_filtering else 'disabled'}")
        except Exception as e:
            logger.error(f"Failed to connect to Reddit API: {e}")
            raise
    
    def _is_noise_post(self, title: str, text: str, author: str) -> bool:
        """
        Check if a post is noise (bot posts, guidelines, etc.).
        
        Args:
            title: Post title
            text: Post text
            author: Post author
            
        Returns:
            True if post should be filtered out
        """
        combined_text = f"{title} {text}".lower()
        
        # Filter out bot posts and automated content
        noise_indicators = [
            'automod',
            'moderator',
            'rule violation',
            'removed post',
            'deleted post',
            'bot message',
            'community guidelines',
            'subreddit rules',
            'posting guidelines',
            'medical disclaimer',
            'doctor patient relationship',
            'informal advice',
            'not medical advice',
            'please read',
            'avoid post removal'
        ]
        
        for indicator in noise_indicators:
            if indicator in combined_text:
                return True
        
        # Filter out very short posts
        if len(combined_text.split()) < 5:
            return True
            
        return False

    def get_subreddit_posts(self, subreddit_name: str, limit: int = 100, 
                           time_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Collect posts from a specific subreddit, filtering out noise.
        
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
            effective_time_filter = time_filter or self.time_filter
            
            for post in subreddit.top(time_filter=effective_time_filter, limit=limit):
                author = str(post.author) if post.author else '[deleted]'
                
                # Filter out noise posts if enabled
                if self.filter_noise and self._is_noise_post(post.title, post.selftext, author):
                    continue
                
                post_data = {
                    'id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'author': author,
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
    
    def _is_noise_comment(self, text: str, author: str) -> bool:
        """
        Check if a comment is noise (bot messages, guidelines, etc.).
        
        Args:
            text: Comment text
            author: Comment author
            
        Returns:
            True if comment should be filtered out
        """
        if not text or len(text.strip()) < 10:
            return True
            
        # Filter out bot messages and automated responses
        bot_indicators = [
            'bot message',
            'help make better community',
            'clicking report link',
            'anti vaxxers user breaks',
            'thank submission',
            'please read following carefully',
            'avoid post removal',
            'medical emergency',
            'please note response constitute',
            'doctor patient relationship',
            'subreddit informal',
            'automod',
            'moderator',
            'rule violation',
            'removed comment',
            'deleted comment'
        ]
        
        text_lower = text.lower()
        for indicator in bot_indicators:
            if indicator in text_lower:
                return True
        
        # Filter out very short or repetitive comments
        if len(text.split()) < 3:
            return True
            
        # Filter out comments that are mostly punctuation
        if len([c for c in text if c.isalpha()]) < len(text) * 0.3:
            return True
            
        return False

    def get_post_comments(self, post_id: str, max_comments: int = 50) -> List[Dict[str, Any]]:
        """
        Collect comments from a specific post, filtering out noise.
        
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
                    author = str(comment.author) if comment.author else '[deleted]'
                    
                    # Filter out noise comments if enabled
                    if self.filter_noise and self._is_noise_comment(comment.body, author):
                        continue
                    
                    comment_data = {
                        'id': comment.id,
                        'text': comment.body,
                        'author': author,
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
        posts = self.get_subreddit_posts(subreddit_name, max_posts, None)
        all_data.extend(posts)
        
        # Collect comments for each post
        for post in posts[:max_posts]:  # Limit to avoid too many API calls
            comments = self.get_post_comments(post['id'], max_comments_per_post)
            all_data.extend(comments)
            
            # Rate limiting - optimized for 100 queries/min capacity
            time.sleep(0.6)  # ~100 queries per minute
            
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
            
            # Rate limiting between subreddits - optimized for API capacity
            time.sleep(1)  # Brief pause between subreddits
            
        logger.info(f"Total data collected from all subreddits: {len(all_data)} items")
        return all_data
    
    def collect_with_intelligent_filtering(self, subreddit_names: List[str], 
                                         target_samples: int = 50000,
                                         max_posts_per_subreddit: int = 100,
                                         max_comments_per_post: int = 50) -> List[Dict[str, Any]]:
        """
        Collect data with intelligent healthcare filtering and prioritization.
        
        Args:
            subreddit_names: List of subreddit names
            target_samples: Target number of samples to collect
            max_posts_per_subreddit: Maximum posts per subreddit
            max_comments_per_post: Maximum comments per post
            
        Returns:
            Filtered and prioritized list of healthcare-relevant data
        """
        if not self.intelligent_filtering or not self.intelligent_filter:
            logger.warning("Intelligent filtering not available, falling back to standard collection")
            return self.collect_multiple_subreddits(
                subreddit_names, max_posts_per_subreddit, max_comments_per_post
            )
        
        logger.info(f"Starting intelligent collection targeting {target_samples} samples")
        all_data = []
        
        for subreddit_name in subreddit_names:
            logger.info(f"Collecting data from r/{subreddit_name} with intelligent filtering")
            
            # Collect posts
            posts = self.get_subreddit_posts(subreddit_name, max_posts_per_subreddit)
            
            # Apply intelligent filtering to posts
            if posts:
                filtered_posts = self.intelligent_filter.prioritize_content(posts)
                all_data.extend(filtered_posts)
                
                # Collect and filter comments for each post
                for post in filtered_posts[:max_posts_per_subreddit]:
                    comments = self.get_post_comments(post['id'], max_comments_per_post)
                    
                    if comments:
                        filtered_comments = self.intelligent_filter.filter_comments(comments)
                        all_data.extend(filtered_comments)
                    
                    # Rate limiting
                    time.sleep(0.6)
            
            # Rate limiting between subreddits
            time.sleep(1)
            
            # Check if we've reached target samples
            if len(all_data) >= target_samples:
                logger.info(f"Reached target samples ({len(all_data)}), stopping collection")
                break
        
        # Final filtering and prioritization
        if all_data:
            # Separate posts and comments for final processing
            posts = [item for item in all_data if item.get('type') == 'post']
            comments = [item for item in all_data if item.get('type') == 'comment']
            
            # Apply final prioritization
            final_posts = self.intelligent_filter.prioritize_content(posts)
            final_comments = self.intelligent_filter.filter_comments(comments)
            
            all_data = final_posts + final_comments
            
            # Get filtering statistics
            stats = self.intelligent_filter.get_filtering_stats(all_data)
            logger.info(f"Intelligent filtering stats: {stats}")
        
        logger.info(f"Intelligent collection complete: {len(all_data)} items")
        return all_data
