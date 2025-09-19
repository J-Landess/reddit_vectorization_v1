"""
SQLite database manager for storing Reddit data and embeddings.
"""
import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite database manager for Reddit data storage and retrieval."""
    
    def __init__(self, db_path: str):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.connection = None
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the database with required tables."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            cursor = self.connection.cursor()
            
            # Create posts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    text TEXT,
                    cleaned_text TEXT,
                    author TEXT,
                    subreddit TEXT,
                    score INTEGER,
                    upvote_ratio REAL,
                    num_comments INTEGER,
                    created_utc TIMESTAMP,
                    url TEXT,
                    is_self BOOLEAN,
                    word_count INTEGER,
                    char_count INTEGER,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    cluster_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create comments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    id TEXT PRIMARY KEY,
                    text TEXT,
                    cleaned_text TEXT,
                    author TEXT,
                    post_id TEXT,
                    score INTEGER,
                    created_utc TIMESTAMP,
                    word_count INTEGER,
                    char_count INTEGER,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    cluster_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (post_id) REFERENCES posts (id)
                )
            ''')
            
            # Create clusters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    algorithm TEXT,
                    parameters TEXT,
                    size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_subreddit ON posts(subreddit)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_utc)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_cluster ON posts(cluster_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_post_id ON comments(post_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_created ON comments(created_utc)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_cluster ON comments(cluster_id)')
            
            self.connection.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def insert_posts(self, posts: List[Dict[str, Any]]) -> int:
        """
        Insert posts into the database.
        
        Args:
            posts: List of post dictionaries
            
        Returns:
            Number of posts inserted
        """
        if not posts:
            return 0
        
        cursor = self.connection.cursor()
        inserted_count = 0
        
        try:
            for post in posts:
                cursor.execute('''
                    INSERT OR REPLACE INTO posts (
                        id, title, text, cleaned_text, author, subreddit,
                        score, upvote_ratio, num_comments, created_utc, url,
                        is_self, word_count, char_count, embedding, embedding_dim, cluster_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    post['id'],
                    post.get('title', ''),
                    post.get('text', ''),
                    post.get('cleaned_text', ''),
                    post.get('author', ''),
                    post.get('subreddit', ''),
                    post.get('score', 0),
                    post.get('upvote_ratio', 0.0),
                    post.get('num_comments', 0),
                    post.get('created_utc'),
                    post.get('url', ''),
                    post.get('is_self', False),
                    post.get('word_count', 0),
                    post.get('char_count', 0),
                    json.dumps(post.get('embedding', [])),
                    post.get('embedding_dim', 0),
                    post.get('cluster_id')
                ))
                inserted_count += 1
            
            self.connection.commit()
            logger.info(f"Inserted {inserted_count} posts")
            
        except Exception as e:
            logger.error(f"Error inserting posts: {e}")
            self.connection.rollback()
            raise
        
        return inserted_count
    
    def insert_comments(self, comments: List[Dict[str, Any]]) -> int:
        """
        Insert comments into the database.
        
        Args:
            comments: List of comment dictionaries
            
        Returns:
            Number of comments inserted
        """
        if not comments:
            return 0
        
        cursor = self.connection.cursor()
        inserted_count = 0
        
        try:
            for comment in comments:
                cursor.execute('''
                    INSERT OR REPLACE INTO comments (
                        id, text, cleaned_text, author, post_id, score,
                        created_utc, word_count, char_count, embedding, embedding_dim, cluster_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    comment['id'],
                    comment.get('text', ''),
                    comment.get('cleaned_text', ''),
                    comment.get('author', ''),
                    comment.get('post_id', ''),
                    comment.get('score', 0),
                    comment.get('created_utc'),
                    comment.get('word_count', 0),
                    comment.get('char_count', 0),
                    json.dumps(comment.get('embedding', [])),
                    comment.get('embedding_dim', 0),
                    comment.get('cluster_id')
                ))
                inserted_count += 1
            
            self.connection.commit()
            logger.info(f"Inserted {inserted_count} comments")
            
        except Exception as e:
            logger.error(f"Error inserting comments: {e}")
            self.connection.rollback()
            raise
        
        return inserted_count
    
    def get_all_data(self, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve all posts and comments from the database.
        
        Args:
            include_embeddings: Whether to include embedding data
            
        Returns:
            List of all data dictionaries
        """
        cursor = self.connection.cursor()
        all_data = []
        
        try:
            # Get posts
            posts_query = '''
                SELECT id, title, text, cleaned_text, author, subreddit,
                       score, upvote_ratio, num_comments, created_utc, url,
                       is_self, word_count, char_count, cluster_id
            '''
            if include_embeddings:
                posts_query += ', embedding, embedding_dim'
            
            posts_query += ' FROM posts ORDER BY created_utc DESC'
            
            cursor.execute(posts_query)
            posts = cursor.fetchall()
            
            for post in posts:
                post_dict = dict(post)
                if include_embeddings and 'embedding' in post_dict:
                    post_dict['embedding'] = json.loads(post_dict['embedding'])
                post_dict['type'] = 'post'
                all_data.append(post_dict)
            
            # Get comments
            comments_query = '''
                SELECT id, text, cleaned_text, author, post_id, score,
                       created_utc, word_count, char_count, cluster_id
            '''
            if include_embeddings:
                comments_query += ', embedding, embedding_dim'
            
            comments_query += ' FROM comments ORDER BY created_utc DESC'
            
            cursor.execute(comments_query)
            comments = cursor.fetchall()
            
            for comment in comments:
                comment_dict = dict(comment)
                if include_embeddings and 'embedding' in comment_dict:
                    comment_dict['embedding'] = json.loads(comment_dict['embedding'])
                comment_dict['type'] = 'comment'
                all_data.append(comment_dict)
            
            logger.info(f"Retrieved {len(all_data)} total items from database")
            
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            raise
        
        return all_data
    
    def get_data_by_subreddit(self, subreddit: str, include_embeddings: bool = False) -> List[Dict[str, Any]]:
        """
        Retrieve data from a specific subreddit.
        
        Args:
            subreddit: Name of the subreddit
            include_embeddings: Whether to include embedding data
            
        Returns:
            List of data dictionaries from the subreddit
        """
        cursor = self.connection.cursor()
        data = []
        
        try:
            # Get posts from subreddit
            posts_query = '''
                SELECT id, title, text, cleaned_text, author, subreddit,
                       score, upvote_ratio, num_comments, created_utc, url,
                       is_self, word_count, char_count, cluster_id
            '''
            if include_embeddings:
                posts_query += ', embedding, embedding_dim'
            
            posts_query += ' FROM posts WHERE subreddit = ? ORDER BY created_utc DESC'
            
            cursor.execute(posts_query, (subreddit,))
            posts = cursor.fetchall()
            
            for post in posts:
                post_dict = dict(post)
                if include_embeddings and 'embedding' in post_dict:
                    post_dict['embedding'] = json.loads(post_dict['embedding'])
                post_dict['type'] = 'post'
                data.append(post_dict)
            
            logger.info(f"Retrieved {len(data)} items from r/{subreddit}")
            
        except Exception as e:
            logger.error(f"Error retrieving data for r/{subreddit}: {e}")
            raise
        
        return data
    
    def update_cluster_assignments(self, data: List[Dict[str, Any]]) -> int:
        """
        Update cluster assignments for data items.
        
        Args:
            data: List of data dictionaries with cluster_id
            
        Returns:
            Number of items updated
        """
        cursor = self.connection.cursor()
        updated_count = 0
        
        try:
            for item in data:
                if 'cluster_id' in item and item['cluster_id'] is not None:
                    if item['type'] == 'post':
                        cursor.execute(
                            'UPDATE posts SET cluster_id = ? WHERE id = ?',
                            (item['cluster_id'], item['id'])
                        )
                    else:
                        cursor.execute(
                            'UPDATE comments SET cluster_id = ? WHERE id = ?',
                            (item['cluster_id'], item['id'])
                        )
                    updated_count += 1
            
            self.connection.commit()
            logger.info(f"Updated cluster assignments for {updated_count} items")
            
        except Exception as e:
            logger.error(f"Error updating cluster assignments: {e}")
            self.connection.rollback()
            raise
        
        return updated_count
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the database contents.
        
        Returns:
            Dictionary of database statistics
        """
        cursor = self.connection.cursor()
        stats = {}
        
        try:
            # Count posts and comments
            cursor.execute('SELECT COUNT(*) FROM posts')
            stats['total_posts'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM comments')
            stats['total_comments'] = cursor.fetchone()[0]
            
            # Count by subreddit
            cursor.execute('''
                SELECT subreddit, COUNT(*) as count 
                FROM posts 
                GROUP BY subreddit 
                ORDER BY count DESC
            ''')
            stats['posts_by_subreddit'] = dict(cursor.fetchall())
            
            # Count clustered items
            cursor.execute('SELECT COUNT(*) FROM posts WHERE cluster_id IS NOT NULL')
            stats['clustered_posts'] = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM comments WHERE cluster_id IS NOT NULL')
            stats['clustered_comments'] = cursor.fetchone()[0]
            
            # Date range
            cursor.execute('SELECT MIN(created_utc), MAX(created_utc) FROM posts')
            date_range = cursor.fetchone()
            stats['date_range'] = {
                'earliest': date_range[0],
                'latest': date_range[1]
            }
            
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            raise
        
        return stats
    
    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
