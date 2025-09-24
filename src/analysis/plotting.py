"""
Reusable plotting utilities for Reddit data using matplotlib and seaborn.
"""
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")


class RedditPlotter:
    """Convenience plotting class with timestamped file outputs."""

    def __init__(self, output_dir: str = './outputs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _timestamped_path(self, base_name: str) -> str:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        return os.path.join(self.output_dir, f"{base_name}_{ts}.png")

    def _save_fig(self, path: str) -> None:
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

    # Subreddit counts (bar)
    def plot_subreddit_counts(self, df: pd.DataFrame, base_name: str = 'subreddit_counts', save_path: Optional[str] = None) -> str:
        counts = df['subreddit'].value_counts().sort_values(ascending=False).head(30)
        plt.figure(figsize=(12, 6))
        sns.barplot(x=counts.index, y=counts.values, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title('Items by Subreddit')
        plt.xlabel('Subreddit')
        plt.ylabel('Count')
        target = save_path or self._timestamped_path(base_name)
        self._save_fig(target)
        return target

    # Daily activity (time series)
    def plot_daily_activity(self, df: pd.DataFrame, base_name: str = 'daily_activity', save_path: Optional[str] = None) -> str:
        if 'created_utc' not in df.columns:
            raise ValueError('created_utc column is required')
        tmp = df.copy()
        tmp['date'] = pd.to_datetime(tmp['created_utc']).dt.date
        daily = tmp['date'].value_counts().sort_index()
        plt.figure(figsize=(12, 4))
        sns.lineplot(x=list(daily.index), y=list(daily.values))
        plt.xticks(rotation=45, ha='right')
        plt.title('Daily Activity')
        plt.xlabel('Date')
        plt.ylabel('Items')
        target = save_path or self._timestamped_path(base_name)
        self._save_fig(target)
        return target

    # Score distribution
    def plot_score_distribution(self, df: pd.DataFrame, base_name: str = 'score_distribution', save_path: Optional[str] = None) -> str:
        if 'score' not in df.columns:
            raise ValueError('score column is required')
        plt.figure(figsize=(8, 5))
        sns.histplot(df['score'], bins=40, kde=True, color='salmon')
        plt.title('Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        target = save_path or self._timestamped_path(base_name)
        self._save_fig(target)
        return target

    # Word count distribution
    def plot_wordcount_distribution(self, df: pd.DataFrame, base_name: str = 'wordcount_distribution', save_path: Optional[str] = None) -> str:
        if 'word_count' not in df.columns:
            raise ValueError('word_count column is required')
        plt.figure(figsize=(8, 5))
        sns.histplot(df['word_count'], bins=40, kde=True, color='lightgreen')
        plt.title('Word Count Distribution')
        plt.xlabel('Word Count')
        plt.ylabel('Frequency')
        target = save_path or self._timestamped_path(base_name)
        self._save_fig(target)
        return target

    # Cluster size distribution
    def plot_cluster_sizes(self, df: pd.DataFrame, base_name: str = 'cluster_sizes', save_path: Optional[str] = None) -> str:
        if 'cluster_id' not in df.columns:
            raise ValueError('cluster_id column is required')
        counts = df['cluster_id'].value_counts().sort_index()
        plt.figure(figsize=(10, 5))
        sns.barplot(x=counts.index.astype(str), y=counts.values, color='mediumpurple')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster ID')
        plt.ylabel('Count')
        target = save_path or self._timestamped_path(base_name)
        self._save_fig(target)
        return target

    # Score vs word count by cluster
    def plot_score_vs_wordcount(self, df: pd.DataFrame, base_name: str = 'score_vs_wordcount', save_path: Optional[str] = None) -> str:
        if not {'score', 'word_count'}.issubset(df.columns):
            raise ValueError('score and word_count columns are required')
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x='word_count', y='score', hue=df.get('cluster_id', None))
        plt.title('Score vs Word Count')
        plt.xlabel('Word Count')
        plt.ylabel('Score')
        target = save_path or self._timestamped_path(base_name)
        self._save_fig(target)
        return target

    # Correlation heatmap
    def plot_correlation_heatmap(self, df: pd.DataFrame, base_name: str = 'correlation_heatmap', save_path: Optional[str] = None) -> str:
        cols = [c for c in ['score', 'word_count', 'char_count'] if c in df.columns]
        if not cols:
            raise ValueError('No numeric columns available for correlation heatmap')
        plt.figure(figsize=(6, 5))
        sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Heatmap')
        target = save_path or self._timestamped_path(base_name)
        self._save_fig(target)
        return target

    # Generate a suite of plots
    def generate_all(self, df: pd.DataFrame, prefix: str = 'analysis') -> dict:
        results = {}
        results['subreddit_counts'] = self.plot_subreddit_counts(df, base_name=f'{prefix}_subreddit_counts')
        if 'created_utc' in df.columns:
            results['daily_activity'] = self.plot_daily_activity(df, base_name=f'{prefix}_daily_activity')
        if 'score' in df.columns:
            results['score_distribution'] = self.plot_score_distribution(df, base_name=f'{prefix}_score_distribution')
        if 'word_count' in df.columns:
            results['wordcount_distribution'] = self.plot_wordcount_distribution(df, base_name=f'{prefix}_wordcount_distribution')
        if 'cluster_id' in df.columns:
            results['cluster_sizes'] = self.plot_cluster_sizes(df, base_name=f'{prefix}_cluster_sizes')
            results['score_vs_wordcount'] = self.plot_score_vs_wordcount(df, base_name=f'{prefix}_score_vs_wordcount')
        results['correlation_heatmap'] = self.plot_correlation_heatmap(df, base_name=f'{prefix}_correlation_heatmap')
        return results
