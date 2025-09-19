"""
Main pipeline orchestrator for Reddit data analysis.
"""
import logging
import os
import sys
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection.reddit_client import RedditClient
from preprocessing.text_cleaner import TextCleaner
from embeddings.embedding_generator import EmbeddingGenerator
from database.database_manager import DatabaseManager
from clustering.cluster_analyzer import ClusterAnalyzer
from analysis.analyzer import RedditAnalyzer
from config import REDDIT_CONFIG, SUBREDDITS, COLLECTION_CONFIG, DATABASE_CONFIG, EMBEDDING_CONFIG, CLUSTERING_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reddit_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class RedditAnalysisPipeline:
    """Main pipeline for Reddit data collection, processing, and analysis."""
    
    def __init__(self):
        """Initialize the pipeline components."""
        self.reddit_client = None
        self.text_cleaner = TextCleaner()
        self.embedding_generator = None
        self.database_manager = None
        self.cluster_analyzer = None
        self.analyzer = RedditAnalyzer()
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        logger.info("Reddit Analysis Pipeline initialized")
    
    def setup_components(self):
        """Set up all pipeline components."""
        try:
            # Initialize Reddit client
            if not all([REDDIT_CONFIG['client_id'], REDDIT_CONFIG['client_secret']]):
                raise ValueError("Reddit API credentials not found. Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables.")
            
            self.reddit_client = RedditClient(
                client_id=REDDIT_CONFIG['client_id'],
                client_secret=REDDIT_CONFIG['client_secret'],
                user_agent=REDDIT_CONFIG['user_agent']
            )
            
            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator(
                model_name=EMBEDDING_CONFIG['model_name'],
                batch_size=EMBEDDING_CONFIG['batch_size']
            )
            
            # Initialize database manager
            self.database_manager = DatabaseManager(DATABASE_CONFIG['path'])
            
            # Initialize cluster analyzer
            self.cluster_analyzer = ClusterAnalyzer(
                algorithm=CLUSTERING_CONFIG['algorithm'],
                min_cluster_size=CLUSTERING_CONFIG['min_cluster_size'],
                min_samples=CLUSTERING_CONFIG['min_samples']
            )
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error setting up pipeline components: {e}")
            raise
    
    def collect_data(self) -> List[Dict[str, Any]]:
        """
        Collect data from all specified subreddits.
        
        Returns:
            List of collected data
        """
        logger.info("Starting data collection from Reddit")
        
        try:
            all_data = self.reddit_client.collect_multiple_subreddits(
                subreddit_names=SUBREDDITS,
                max_posts_per_subreddit=COLLECTION_CONFIG['max_posts_per_subreddit'],
                max_comments_per_post=COLLECTION_CONFIG['max_comments_per_post']
            )
            
            logger.info(f"Collected {len(all_data)} total items from {len(SUBREDDITS)} subreddits")
            return all_data
            
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            raise
    
    def preprocess_data(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preprocess the collected data.
        
        Args:
            raw_data: Raw data from Reddit
            
        Returns:
            Preprocessed data
        """
        logger.info("Starting data preprocessing")
        
        try:
            processed_data = self.text_cleaner.preprocess_reddit_data(raw_data)
            
            # Get preprocessing statistics
            stats = self.text_cleaner.get_text_statistics(processed_data)
            logger.info(f"Preprocessing statistics: {stats}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise
    
    def generate_embeddings(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for the processed data.
        
        Args:
            processed_data: Preprocessed data
            
        Returns:
            Data with embeddings
        """
        logger.info("Starting embedding generation")
        
        try:
            data_with_embeddings = self.embedding_generator.generate_embeddings_for_data(processed_data)
            
            # Get embedding statistics
            stats = self.embedding_generator.get_embedding_statistics(data_with_embeddings)
            logger.info(f"Embedding statistics: {stats}")
            
            return data_with_embeddings
            
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            raise
    
    def store_data(self, data_with_embeddings: List[Dict[str, Any]]) -> None:
        """
        Store data in the database.
        
        Args:
            data_with_embeddings: Data with embeddings
        """
        logger.info("Starting data storage")
        
        try:
            # Separate posts and comments
            posts = [item for item in data_with_embeddings if item['type'] == 'post']
            comments = [item for item in data_with_embeddings if item['type'] == 'comment']
            
            # Store in database
            posts_inserted = self.database_manager.insert_posts(posts)
            comments_inserted = self.database_manager.insert_comments(comments)
            
            logger.info(f"Stored {posts_inserted} posts and {comments_inserted} comments")
            
        except Exception as e:
            logger.error(f"Error during data storage: {e}")
            raise
    
    def perform_clustering(self, data_with_embeddings: List[Dict[str, Any]]) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
        """
        Perform clustering analysis on the data.
        
        Args:
            data_with_embeddings: Data with embeddings
            
        Returns:
            Tuple of cluster labels and cluster summaries
        """
        logger.info("Starting clustering analysis")
        
        try:
            # Extract embeddings
            embeddings = np.array([item['embedding'] for item in data_with_embeddings])
            
            # Perform clustering
            cluster_labels = self.cluster_analyzer.fit(embeddings)
            
            # Add cluster labels to data
            for i, item in enumerate(data_with_embeddings):
                item['cluster_id'] = int(cluster_labels[i])
            
            # Get cluster statistics
            cluster_stats = self.cluster_analyzer.get_cluster_statistics()
            quality_metrics = self.cluster_analyzer.get_cluster_quality_metrics()
            
            logger.info(f"Clustering statistics: {cluster_stats}")
            logger.info(f"Quality metrics: {quality_metrics}")
            
            # Get cluster representatives and summaries
            representatives = self.cluster_analyzer.get_cluster_representatives(data_with_embeddings)
            cluster_summaries = self.cluster_analyzer.get_cluster_summaries(data_with_embeddings, representatives)
            
            # Update database with cluster assignments
            self.database_manager.update_cluster_assignments(data_with_embeddings)
            
            logger.info(f"Generated summaries for {len(cluster_summaries)} clusters")
            
            return cluster_labels, cluster_summaries
            
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise
    
    def generate_analysis(self, data_with_embeddings: List[Dict[str, Any]], 
                         cluster_labels: np.ndarray, 
                         cluster_summaries: Dict[int, Dict[str, Any]]) -> None:
        """
        Generate analysis and visualizations.
        
        Args:
            data_with_embeddings: Data with embeddings
            cluster_labels: Cluster labels
            cluster_summaries: Cluster summaries
        """
        logger.info("Starting analysis and visualization generation")
        
        try:
            # Generate visualizations
            self.analyzer.create_subreddit_visualization(data_with_embeddings)
            self.analyzer.create_cluster_visualization(data_with_embeddings, cluster_labels)
            
            # Generate word clouds for each cluster
            unique_clusters = np.unique(cluster_labels)
            for cluster_id in unique_clusters:
                if cluster_id != -1:  # Skip noise
                    self.analyzer.generate_word_cloud(data_with_embeddings, int(cluster_id))
            
            # Generate overall word cloud
            self.analyzer.generate_word_cloud(data_with_embeddings)
            
            # Create interactive dashboard
            self.analyzer.create_interactive_dashboard(data_with_embeddings, cluster_summaries)
            
            # Export cluster summaries
            self.analyzer.export_cluster_summaries(cluster_summaries)
            
            # Generate analysis report
            cluster_stats = self.cluster_analyzer.get_cluster_statistics()
            report = self.analyzer.generate_analysis_report(data_with_embeddings, cluster_summaries, cluster_stats)
            
            # Save report
            with open('outputs/analysis_report.md', 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info("Analysis and visualization generation completed")
            
        except Exception as e:
            logger.error(f"Error during analysis generation: {e}")
            raise
    
    def run_full_pipeline(self) -> None:
        """Run the complete analysis pipeline."""
        logger.info("Starting full Reddit analysis pipeline")
        
        try:
            # Setup components
            self.setup_components()
            
            # Collect data
            raw_data = self.collect_data()
            
            # Preprocess data
            processed_data = self.preprocess_data(raw_data)
            
            # Generate embeddings
            data_with_embeddings = self.generate_embeddings(processed_data)
            
            # Store data
            self.store_data(data_with_embeddings)
            
            # Perform clustering
            cluster_labels, cluster_summaries = self.perform_clustering(data_with_embeddings)
            
            # Generate analysis
            self.generate_analysis(data_with_embeddings, cluster_labels, cluster_summaries)
            
            # Get final database statistics
            db_stats = self.database_manager.get_database_statistics()
            logger.info(f"Final database statistics: {db_stats}")
            
            logger.info("Reddit analysis pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            # Clean up
            if self.database_manager:
                self.database_manager.close()


def main():
    """Main entry point for the pipeline."""
    pipeline = RedditAnalysisPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
