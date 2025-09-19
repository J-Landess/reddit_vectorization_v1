"""
Analysis and visualization utilities for Reddit data.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

logger = logging.getLogger(__name__)


class RedditAnalyzer:
    """Analysis and visualization utilities for Reddit data."""
    
    def __init__(self, output_dir: str = './outputs'):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory to save output files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_data_distribution(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze the distribution of data across subreddits and other dimensions.
        
        Args:
            data: List of data dictionaries
            
        Returns:
            Dictionary of distribution analysis
        """
        if not data:
            return {}
        
        df = pd.DataFrame(data)
        
        analysis = {
            'total_items': len(data),
            'posts': len([item for item in data if item.get('type') == 'post']),
            'comments': len([item for item in data if item.get('type') == 'comment']),
            'subreddit_distribution': df['subreddit'].value_counts().to_dict(),
            'author_distribution': df['author'].value_counts().head(10).to_dict(),
            'date_range': {
                'earliest': df['created_utc'].min() if 'created_utc' in df.columns else None,
                'latest': df['created_utc'].max() if 'created_utc' in df.columns else None
            }
        }
        
        # Word count statistics
        if 'word_count' in df.columns:
            analysis['word_count_stats'] = {
                'mean': float(df['word_count'].mean()),
                'median': float(df['word_count'].median()),
                'std': float(df['word_count'].std()),
                'min': int(df['word_count'].min()),
                'max': int(df['word_count'].max())
            }
        
        # Score statistics
        if 'score' in df.columns:
            analysis['score_stats'] = {
                'mean': float(df['score'].mean()),
                'median': float(df['score'].median()),
                'std': float(df['score'].std()),
                'min': int(df['score'].min()),
                'max': int(df['score'].max())
            }
        
        logger.info(f"Analyzed distribution for {len(data)} items")
        return analysis
    
    def create_subreddit_visualization(self, data: List[Dict[str, Any]], 
                                     save_path: Optional[str] = None) -> None:
        """
        Create visualizations for subreddit distribution.
        
        Args:
            data: List of data dictionaries
            save_path: Path to save the plot
        """
        if not data:
            logger.warning("No data provided for visualization")
            return
        
        df = pd.DataFrame(data)
        subreddit_counts = df['subreddit'].value_counts()
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Reddit Data Distribution Analysis', fontsize=16)
        
        # Subreddit distribution (bar plot)
        subreddit_counts.plot(kind='bar', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('Posts/Comments by Subreddit')
        axes[0, 0].set_xlabel('Subreddit')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Subreddit distribution (pie chart)
        axes[0, 1].pie(subreddit_counts.values, labels=subreddit_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Subreddit Distribution')
        
        # Word count distribution
        if 'word_count' in df.columns:
            df['word_count'].hist(bins=30, ax=axes[1, 0], color='lightgreen', alpha=0.7)
            axes[1, 0].set_title('Word Count Distribution')
            axes[1, 0].set_xlabel('Word Count')
            axes[1, 0].set_ylabel('Frequency')
        
        # Score distribution
        if 'score' in df.columns:
            df['score'].hist(bins=30, ax=axes[1, 1], color='salmon', alpha=0.7)
            axes[1, 1].set_title('Score Distribution')
            axes[1, 1].set_xlabel('Score')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved subreddit visualization to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'subreddit_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            logger.info("Saved subreddit visualization")
        
        plt.close()
    
    def create_cluster_visualization(self, data: List[Dict[str, Any]], 
                                   cluster_labels: np.ndarray,
                                   save_path: Optional[str] = None) -> None:
        """
        Create visualizations for clustering results.
        
        Args:
            data: List of data dictionaries
            cluster_labels: Cluster labels array
            save_path: Path to save the plot
        """
        if not data or len(cluster_labels) == 0:
            logger.warning("No data or cluster labels provided for visualization")
            return
        
        df = pd.DataFrame(data)
        df['cluster'] = cluster_labels
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clustering Analysis', fontsize=16)
        
        # Cluster size distribution
        cluster_counts = df['cluster'].value_counts().sort_index()
        cluster_counts.plot(kind='bar', ax=axes[0, 0], color='lightcoral')
        axes[0, 0].set_title('Cluster Size Distribution')
        axes[0, 0].set_xlabel('Cluster ID')
        axes[0, 0].set_ylabel('Count')
        
        # Subreddit distribution by cluster
        if len(df) > 0:
            cluster_subreddit = pd.crosstab(df['cluster'], df['subreddit'])
            cluster_subreddit.plot(kind='bar', stacked=True, ax=axes[0, 1])
            axes[0, 1].set_title('Subreddit Distribution by Cluster')
            axes[0, 1].set_xlabel('Cluster ID')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Word count by cluster
        if 'word_count' in df.columns:
            df.boxplot(column='word_count', by='cluster', ax=axes[1, 0])
            axes[1, 0].set_title('Word Count by Cluster')
            axes[1, 0].set_xlabel('Cluster ID')
            axes[1, 0].set_ylabel('Word Count')
        
        # Score by cluster
        if 'score' in df.columns:
            df.boxplot(column='score', by='cluster', ax=axes[1, 1])
            axes[1, 1].set_title('Score by Cluster')
            axes[1, 1].set_xlabel('Cluster ID')
            axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster visualization to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'cluster_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            logger.info("Saved cluster visualization")
        
        plt.close()
    
    def generate_word_cloud(self, data: List[Dict[str, Any]], 
                          cluster_id: Optional[int] = None,
                          save_path: Optional[str] = None) -> None:
        """
        Generate word cloud for text data.
        
        Args:
            data: List of data dictionaries
            cluster_id: Specific cluster ID (None for all data)
            save_path: Path to save the word cloud
        """
        if not data:
            logger.warning("No data provided for word cloud")
            return
        
        # Filter data by cluster if specified
        if cluster_id is not None:
            filtered_data = [item for item in data if item.get('cluster_id') == cluster_id]
            title_suffix = f" (Cluster {cluster_id})"
        else:
            filtered_data = data
            title_suffix = " (All Data)"
        
        if not filtered_data:
            logger.warning(f"No data found for cluster {cluster_id}")
            return
        
        # Combine all text
        all_text = ' '.join([item.get('cleaned_text', '') for item in filtered_data])
        
        if not all_text.strip():
            logger.warning("No text content found for word cloud")
            return
        
        # Generate word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            colormap='viridis'
        ).generate(all_text)
        
        # Create plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud{title_suffix}', fontsize=16)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved word cloud to {save_path}")
        else:
            filename = f'wordcloud_cluster_{cluster_id}.png' if cluster_id is not None else 'wordcloud_all.png'
            plt.savefig(os.path.join(self.output_dir, filename), 
                       dpi=300, bbox_inches='tight')
            logger.info(f"Saved word cloud: {filename}")
        
        plt.close()
    
    def create_interactive_dashboard(self, data: List[Dict[str, Any]], 
                                   cluster_summaries: Dict[int, Dict[str, Any]],
                                   save_path: Optional[str] = None) -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            data: List of data dictionaries
            cluster_summaries: Cluster summaries
            save_path: Path to save the HTML dashboard
        """
        if not data:
            logger.warning("No data provided for dashboard")
            return
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Subreddit Distribution', 'Cluster Analysis', 
                          'Word Count vs Score', 'Timeline'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Subreddit distribution
        subreddit_counts = df['subreddit'].value_counts()
        fig.add_trace(
            go.Bar(x=subreddit_counts.index, y=subreddit_counts.values, 
                  name="Posts/Comments", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Cluster analysis
        if 'cluster_id' in df.columns:
            cluster_counts = df['cluster_id'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=cluster_counts.index.astype(str), y=cluster_counts.values,
                      name="Cluster Size", marker_color='lightcoral'),
                row=1, col=2
            )
        
        # Word count vs Score scatter plot
        if 'word_count' in df.columns and 'score' in df.columns:
            fig.add_trace(
                go.Scatter(x=df['word_count'], y=df['score'], mode='markers',
                          name="Word Count vs Score", marker=dict(size=6, opacity=0.6)),
                row=2, col=1
            )
        
        # Timeline (if created_utc is available)
        if 'created_utc' in df.columns:
            df['date'] = pd.to_datetime(df['created_utc']).dt.date
            daily_counts = df['date'].value_counts().sort_index()
            fig.add_trace(
                go.Scatter(x=daily_counts.index, y=daily_counts.values, 
                          mode='lines+markers', name="Daily Activity"),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Reddit Data Analysis Dashboard",
            showlegend=True,
            height=800
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Subreddit", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Cluster ID", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Word Count", row=2, col=1)
        fig.update_yaxes(title_text="Score", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive dashboard to {save_path}")
        else:
            dashboard_path = os.path.join(self.output_dir, 'reddit_dashboard.html')
            fig.write_html(dashboard_path)
            logger.info(f"Saved interactive dashboard to {dashboard_path}")
    
    def export_cluster_summaries(self, cluster_summaries: Dict[int, Dict[str, Any]], 
                               save_path: Optional[str] = None) -> None:
        """
        Export cluster summaries to a readable format.
        
        Args:
            cluster_summaries: Cluster summaries dictionary
            save_path: Path to save the summary file
        """
        if not cluster_summaries:
            logger.warning("No cluster summaries provided for export")
            return
        
        output_lines = []
        output_lines.append("# Reddit Data Cluster Analysis Summary\n")
        output_lines.append(f"Generated on: {pd.Timestamp.now()}\n")
        output_lines.append(f"Total clusters: {len(cluster_summaries)}\n\n")
        
        for cluster_id, summary in cluster_summaries.items():
            output_lines.append(f"## Cluster {cluster_id}\n")
            output_lines.append(f"- **Size**: {summary['size']} items\n")
            output_lines.append(f"- **Top Subreddits**: {', '.join(summary['top_subreddits'])}\n")
            output_lines.append(f"- **Common Words**: {', '.join(summary['common_words'])}\n")
            output_lines.append(f"- **Average Score**: {summary['avg_score']:.2f}\n")
            
            # Representative posts
            if 'representatives' in summary and summary['representatives']:
                output_lines.append("- **Representative Posts/Comments**:\n")
                for i, rep in enumerate(summary['representatives'][:3], 1):
                    text_preview = rep.get('cleaned_text', '')[:100] + "..." if len(rep.get('cleaned_text', '')) > 100 else rep.get('cleaned_text', '')
                    output_lines.append(f"  {i}. {text_preview}\n")
            
            output_lines.append("\n---\n\n")
        
        # Write to file
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.writelines(output_lines)
            logger.info(f"Exported cluster summaries to {save_path}")
        else:
            summary_path = os.path.join(self.output_dir, 'cluster_summaries.md')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.writelines(output_lines)
            logger.info(f"Exported cluster summaries to {summary_path}")
    
    def generate_analysis_report(self, data: List[Dict[str, Any]], 
                               cluster_summaries: Dict[int, Dict[str, Any]],
                               cluster_stats: Dict[str, Any]) -> str:
        """
        Generate a comprehensive analysis report.
        
        Args:
            data: List of data dictionaries
            cluster_summaries: Cluster summaries
            cluster_stats: Clustering statistics
            
        Returns:
            Analysis report as string
        """
        report_lines = []
        report_lines.append("# Reddit Data Analysis Report\n")
        report_lines.append(f"Generated on: {pd.Timestamp.now()}\n\n")
        
        # Data overview
        analysis = self.analyze_data_distribution(data)
        report_lines.append("## Data Overview\n")
        report_lines.append(f"- **Total Items**: {analysis['total_items']}\n")
        report_lines.append(f"- **Posts**: {analysis['posts']}\n")
        report_lines.append(f"- **Comments**: {analysis['comments']}\n")
        report_lines.append(f"- **Subreddits**: {len(analysis['subreddit_distribution'])}\n\n")
        
        # Top subreddits
        report_lines.append("### Top Subreddits\n")
        for subreddit, count in list(analysis['subreddit_distribution'].items())[:5]:
            report_lines.append(f"- r/{subreddit}: {count} items\n")
        
        # Clustering results
        if cluster_stats:
            report_lines.append("\n## Clustering Results\n")
            report_lines.append(f"- **Algorithm**: {cluster_stats.get('algorithm', 'Unknown')}\n")
            report_lines.append(f"- **Number of Clusters**: {cluster_stats.get('n_clusters', 0)}\n")
            report_lines.append(f"- **Noise Points**: {cluster_stats.get('n_noise', 0)}\n")
            report_lines.append(f"- **Average Cluster Size**: {cluster_stats.get('avg_cluster_size', 0):.1f}\n\n")
        
        # Cluster summaries
        if cluster_summaries:
            report_lines.append("## Cluster Analysis\n")
            for cluster_id, summary in list(cluster_summaries.items())[:5]:  # Show top 5 clusters
                report_lines.append(f"### Cluster {cluster_id}\n")
                report_lines.append(f"- **Size**: {summary['size']} items\n")
                report_lines.append(f"- **Top Subreddits**: {', '.join(summary['top_subreddits'])}\n")
                report_lines.append(f"- **Common Words**: {', '.join(summary['common_words'])}\n\n")
        
        return ''.join(report_lines)
