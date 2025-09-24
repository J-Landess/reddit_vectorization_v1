"""
General database query and export utilities for Reddit data.
"""
import sqlite3
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import os

logger = logging.getLogger(__name__)


class RedditDBQueries:
    """General database query utilities for Reddit data analysis."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # Basic queries
    def get_all_posts(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get all posts with optional limit."""
        conn = self.get_connection()
        query = "SELECT * FROM posts ORDER BY created_utc DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_all_comments(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Get all comments with optional limit."""
        conn = self.get_connection()
        query = "SELECT * FROM comments ORDER BY created_utc DESC"
        if limit:
            query += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_posts_by_subreddit(self, subreddit: str) -> pd.DataFrame:
        """Get posts from specific subreddit."""
        conn = self.get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM posts WHERE subreddit = ? ORDER BY created_utc DESC",
            conn, params=(subreddit,)
        )
        conn.close()
        return df
    
    def get_comments_by_post(self, post_id: str) -> pd.DataFrame:
        """Get comments for specific post."""
        conn = self.get_connection()
        df = pd.read_sql_query(
            "SELECT * FROM comments WHERE post_id = ? ORDER BY score DESC",
            conn, params=(post_id,)
        )
        conn.close()
        return df
    
    def get_data_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get all data within date range (inclusive)."""
        conn = self.get_connection()
        query = (
            "SELECT 'post' as type, * FROM posts WHERE created_utc BETWEEN ? AND ? "
            "UNION ALL "
            "SELECT 'comment' as type, * FROM comments WHERE created_utc BETWEEN ? AND ? "
            "ORDER BY created_utc DESC"
        )
        df = pd.read_sql_query(query, conn, params=(start_date, end_date, start_date, end_date))
        conn.close()
        return df
    
    def get_data_by_cluster(self, cluster_id: int) -> pd.DataFrame:
        """Get all data from specific cluster."""
        conn = self.get_connection()
        query = (
            "SELECT 'post' as type, * FROM posts WHERE cluster_id = ? "
            "UNION ALL "
            "SELECT 'comment' as type, * FROM comments WHERE cluster_id = ? "
            "ORDER BY score DESC"
        )
        df = pd.read_sql_query(query, conn, params=(cluster_id, cluster_id))
        conn.close()
        return df
    
    def get_data_by_run(self, run_id: str) -> pd.DataFrame:
        """Get all data from specific run."""
        conn = self.get_connection()
        query = (
            "SELECT 'post' as type, * FROM posts WHERE run_id = ? "
            "UNION ALL "
            "SELECT 'comment' as type, * FROM comments WHERE run_id = ? "
            "ORDER BY created_utc DESC"
        )
        df = pd.read_sql_query(query, conn, params=(run_id, run_id))
        conn.close()
        return df
    
    # Analytics queries
    def get_subreddit_stats(self) -> pd.DataFrame:
        """Get statistics by subreddit."""
        conn = self.get_connection()
        query = (
            "SELECT "
            "    subreddit, "
            "    COUNT(*) as total_items, "
            "    SUM(CASE WHEN type = 'post' THEN 1 ELSE 0 END) as posts, "
            "    SUM(CASE WHEN type = 'comment' THEN 1 ELSE 0 END) as comments, "
            "    AVG(score) as avg_score, "
            "    AVG(word_count) as avg_word_count, "
            "    MIN(created_utc) as earliest, "
            "    MAX(created_utc) as latest "
            "FROM ( "
            "    SELECT 'post' as type, subreddit, score, word_count, created_utc FROM posts "
            "    UNION ALL "
            "    SELECT 'comment' as type, subreddit, score, word_count, created_utc FROM comments "
            ") combined "
            "GROUP BY subreddit "
            "ORDER BY total_items DESC"
        )
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_cluster_stats(self) -> pd.DataFrame:
        """Get statistics by cluster."""
        conn = self.get_connection()
        query = (
            "SELECT "
            "    cluster_id, "
            "    COUNT(*) as size, "
            "    COUNT(DISTINCT subreddit) as unique_subreddits, "
            "    AVG(score) as avg_score, "
            "    AVG(word_count) as avg_word_count "
            "FROM ( "
            "    SELECT cluster_id, subreddit, score, word_count FROM posts WHERE cluster_id IS NOT NULL "
            "    UNION ALL "
            "    SELECT cluster_id, subreddit, score, word_count FROM comments WHERE cluster_id IS NOT NULL "
            ") combined "
            "GROUP BY cluster_id "
            "ORDER BY size DESC"
        )
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_run_stats(self) -> pd.DataFrame:
        """Get statistics by run."""
        conn = self.get_connection()
        query = (
            "SELECT "
            "    run_id, "
            "    COUNT(*) as total_items, "
            "    SUM(CASE WHEN type = 'post' THEN 1 ELSE 0 END) as posts, "
            "    SUM(CASE WHEN type = 'comment' THEN 1 ELSE 0 END) as comments, "
            "    COUNT(DISTINCT subreddit) as unique_subreddits, "
            "    MIN(created_utc) as start_time, "
            "    MAX(created_utc) as end_time "
            "FROM ( "
            "    SELECT 'post' as type, run_id, subreddit, created_utc FROM posts WHERE run_id IS NOT NULL "
            "    UNION ALL "
            "    SELECT 'comment' as type, run_id, subreddit, created_utc FROM comments WHERE run_id IS NOT NULL "
            ") combined "
            "GROUP BY run_id "
            "ORDER BY start_time DESC"
        )
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    # Export utilities
    def export_to_excel(self, output_path: str, sheets: Optional[List[str]] = None) -> None:
        """Export data to Excel with multiple sheets."""
        if sheets is None:
            sheets = ['posts', 'comments', 'subreddit_stats', 'cluster_stats', 'run_stats']
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            if 'posts' in sheets:
                self.get_all_posts().to_excel(writer, sheet_name='posts', index=False)
            if 'comments' in sheets:
                self.get_all_comments().to_excel(writer, sheet_name='comments', index=False)
            if 'subreddit_stats' in sheets:
                self.get_subreddit_stats().to_excel(writer, sheet_name='subreddit_stats', index=False)
            if 'cluster_stats' in sheets:
                self.get_cluster_stats().to_excel(writer, sheet_name='cluster_stats', index=False)
            if 'run_stats' in sheets:
                self.get_run_stats().to_excel(writer, sheet_name='run_stats', index=False)
        logger.info(f"Exported data to Excel: {output_path}")
    
    def export_query_json(self, output_path: str, query: str, params: tuple = ()) -> None:
        """Export arbitrary query results to JSON."""
        conn = self.get_connection()
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        df.to_json(output_path, orient='records', indent=2)
        logger.info(f"Exported query results to JSON: {output_path}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information."""
        conn = self.get_connection()
        cursor = conn.cursor()
        info: Dict[str, Any] = {}
        cursor.execute("SELECT COUNT(*) FROM posts")
        info['total_posts'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM comments")
        info['total_comments'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM runs")
        info['total_runs'] = cursor.fetchone()[0]
        cursor.execute("SELECT MIN(created_utc), MAX(created_utc) FROM posts")
        earliest, latest = cursor.fetchone()
        info['date_range'] = {'earliest': earliest, 'latest': latest}
        cursor.execute("SELECT COUNT(DISTINCT subreddit) FROM posts")
        info['unique_subreddits'] = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM posts WHERE cluster_id IS NOT NULL")
        info['clustered_posts'] = cursor.fetchone()[0]
        conn.close()
        return info
