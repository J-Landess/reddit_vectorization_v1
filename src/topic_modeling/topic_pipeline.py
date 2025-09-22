"""
Topic modeling pipeline for Reddit healthcare data.
"""
import logging
import sys
import os
from typing import List, Dict, Any, Optional
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topic_modeling.topic_analyzer import TopicAnalyzer
from analysis.analyzer import RedditAnalyzer
from database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class TopicModelingPipeline:
    """Pipeline for topic modeling analysis of Reddit healthcare data."""
    
    def __init__(self, algorithm: str = 'gensim_lda', n_topics: int = 20):
        """
        Initialize the topic modeling pipeline.
        
        Args:
            algorithm: Topic modeling algorithm
            n_topics: Number of topics
        """
        self.algorithm = algorithm
        self.n_topics = n_topics
        self.topic_analyzer = TopicAnalyzer(algorithm=algorithm, n_topics=n_topics)
        self.database_manager = None
        
        logger.info(f"Initialized topic modeling pipeline with {algorithm}")
    
    def run_topic_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run complete topic analysis on the data.
        
        Args:
            data: List of Reddit data items
            
        Returns:
            Topic analysis results
        """
        logger.info(f"Running topic analysis on {len(data)} items")
        
        # Extract texts
        texts = []
        for item in data:
            text = item.get('cleaned_text') or item.get('text', '')
            if text and len(text.strip()) > 10:
                texts.append(text)
        
        logger.info(f"Extracted {len(texts)} valid texts for topic modeling")
        
        # Fit topic model
        model_results = self.topic_analyzer.fit(texts)
        
        # Get topic assignments
        topic_labels = self.topic_analyzer.topic_labels
        
        # Add topic assignments to data
        text_idx = 0
        for i, item in enumerate(data):
            text = item.get('cleaned_text') or item.get('text', '')
            if text and len(text.strip()) > 10:
                if text_idx < len(topic_labels):
                    item['topic_id'] = int(topic_labels[text_idx])
                text_idx += 1
            else:
                item['topic_id'] = -1  # No topic assigned
        
        # Get topic statistics
        topic_stats = self.topic_analyzer.get_topic_statistics()
        
        # Get topic summaries
        topic_summaries = self.topic_analyzer.get_topic_summaries(data)
        
        # Get topic words
        topic_words = self.topic_analyzer.get_topic_words(n_words=10)
        
        results = {
            'model_results': model_results,
            'topic_statistics': topic_stats,
            'topic_summaries': topic_summaries,
            'topic_words': topic_words,
            'n_documents_processed': len(texts),
            'algorithm': self.algorithm
        }
        
        logger.info("Topic analysis completed")
        return results
    
    def save_results(self, data: List[Dict[str, Any]], results: Dict[str, Any]):
        """
        Save topic modeling results to database.
        
        Args:
            data: Data with topic assignments
            results: Topic analysis results
        """
        logger.info("Saving topic modeling results to database")
        
        if not self.database_manager:
            self.database_manager = DatabaseManager()
            self.database_manager.create_tables()
        
        # Update database with topic assignments
        posts = [item for item in data if item.get('type') == 'post']
        comments = [item for item in data if item.get('type') == 'comment']
        
        if posts:
            self.database_manager.store_posts(posts)
            logger.info(f"Updated {len(posts)} posts with topic assignments")
        
        if comments:
            self.database_manager.store_comments(comments)
            logger.info(f"Updated {len(comments)} comments with topic assignments")
        
        logger.info("Topic modeling results saved")
    
    def generate_topic_report(self, results: Dict[str, Any], output_dir: str = 'outputs') -> str:
        """
        Generate a comprehensive topic modeling report.
        
        Args:
            results: Topic analysis results
            output_dir: Output directory for the report
            
        Returns:
            Path to the generated report
        """
        logger.info("Generating topic modeling report")
        
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'topic_modeling_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Topic Modeling Analysis Report\n\n")
            f.write(f"**Algorithm:** {results['algorithm']}\n")
            f.write(f"**Documents Processed:** {results['n_documents_processed']}\n\n")
            
            # Model results
            model_results = results['model_results']
            f.write("## Model Performance\n\n")
            for key, value in model_results.items():
                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
            
            # Topic statistics
            topic_stats = results['topic_statistics']
            f.write(f"\n## Topic Statistics\n\n")
            f.write(f"- **Number of Topics:** {topic_stats.get('n_topics', 'N/A')}\n")
            f.write(f"- **Total Documents:** {topic_stats.get('total_documents', 'N/A')}\n")
            f.write(f"- **Largest Topic Size:** {topic_stats.get('largest_topic_size', 'N/A')}\n")
            f.write(f"- **Smallest Topic Size:** {topic_stats.get('smallest_topic_size', 'N/A')}\n")
            f.write(f"- **Average Topic Size:** {topic_stats.get('avg_topic_size', 'N/A'):.1f}\n")
            
            # Topic summaries
            topic_summaries = results['topic_summaries']
            f.write(f"\n## Topic Summaries\n\n")
            
            for topic_id, summary in topic_summaries.items():
                f.write(f"### Topic {topic_id}\n\n")
                f.write(f"- **Size:** {summary['size']} documents\n")
                f.write(f"- **Top Words:** {', '.join(summary['top_words'])}\n")
                f.write(f"- **Top Subreddits:** {', '.join(summary['top_subreddits'])}\n")
                f.write(f"- **Average Score:** {summary['avg_score']:.2f}\n\n")
                
                # Representative documents
                if summary['representatives']:
                    f.write("**Representative Documents:**\n")
                    for i, rep in enumerate(summary['representatives'][:2]):
                        text_preview = rep.get('text', '')[:200] + "..." if len(rep.get('text', '')) > 200 else rep.get('text', '')
                        f.write(f"{i+1}. {text_preview}\n\n")
        
        logger.info(f"Topic modeling report generated: {report_path}")
        return report_path
