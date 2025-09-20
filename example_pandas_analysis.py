#!/usr/bin/env python3
"""
Example pandas analysis of Reddit data exported to CSV.
This script demonstrates how to load and analyze the exported data.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_reddit_data(csv_dir: str = './csv_exports'):
    """
    Load Reddit data from CSV files.
    
    Args:
        csv_dir: Directory containing CSV files
        
    Returns:
        Dictionary of DataFrames
    """
    data = {}
    
    # Load posts
    posts_file = os.path.join(csv_dir, 'reddit_posts.csv')
    if os.path.exists(posts_file):
        data['posts'] = pd.read_csv(posts_file)
        print(f"Loaded {len(data['posts'])} posts")
    
    # Load comments
    comments_file = os.path.join(csv_dir, 'reddit_comments.csv')
    if os.path.exists(comments_file):
        data['comments'] = pd.read_csv(comments_file)
        print(f"Loaded {len(data['comments'])} comments")
    
    # Load combined data
    all_file = os.path.join(csv_dir, 'reddit_all_data.csv')
    if os.path.exists(all_file):
        data['all'] = pd.read_csv(all_file)
        print(f"Loaded {len(data['all'])} total items")
    
    return data

def basic_analysis(df: pd.DataFrame):
    """Perform basic analysis on the data."""
    print("\n📊 Basic Data Analysis")
    print("=" * 30)
    
    # Data info
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Subreddit distribution
    if 'subreddit' in df.columns:
        print(f"\nSubreddit distribution:")
        print(df['subreddit'].value_counts())
    
    # Score statistics
    if 'score' in df.columns:
        print(f"\nScore statistics:")
        print(df['score'].describe())
    
    # Word count statistics
    if 'word_count' in df.columns:
        print(f"\nWord count statistics:")
        print(df['word_count'].describe())
    
    # Cluster distribution
    if 'cluster_id' in df.columns:
        print(f"\nCluster distribution:")
        print(df['cluster_id'].value_counts().sort_index())

def create_visualizations(df: pd.DataFrame, output_dir: str = './pandas_analysis'):
    """Create visualizations of the data."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📈 Creating visualizations in {output_dir}")
    
    # Set up plotting
    plt.style.use('seaborn-v0_8')
    sns.set_palette('husl')
    
    # Subreddit distribution
    if 'subreddit' in df.columns:
        plt.figure(figsize=(12, 6))
        subreddit_counts = df['subreddit'].value_counts()
        subreddit_counts.plot(kind='bar')
        plt.title('Posts/Comments by Subreddit')
        plt.xlabel('Subreddit')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'subreddit_distribution.png'), dpi=300)
        plt.close()
        print("✅ Created subreddit distribution plot")
    
    # Score analysis
    if 'score' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Score histogram
        df['score'].hist(bins=30, ax=axes[0, 0])
        axes[0, 0].set_title('Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        
        # Score by subreddit
        if 'subreddit' in df.columns:
            df.boxplot(column='score', by='subreddit', ax=axes[0, 1])
            axes[0, 1].set_title('Score by Subreddit')
            axes[0, 1].set_xlabel('Subreddit')
            axes[0, 1].set_ylabel('Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Word count histogram
        if 'word_count' in df.columns:
            df['word_count'].hist(bins=30, ax=axes[1, 0])
            axes[1, 0].set_title('Word Count Distribution')
            axes[1, 0].set_xlabel('Word Count')
            axes[1, 0].set_ylabel('Frequency')
        
        # Score vs Word Count scatter
        if 'word_count' in df.columns:
            axes[1, 1].scatter(df['word_count'], df['score'], alpha=0.6)
            axes[1, 1].set_title('Score vs Word Count')
            axes[1, 1].set_xlabel('Word Count')
            axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_analysis.png'), dpi=300)
        plt.close()
        print("✅ Created score analysis plots")
    
    # Cluster analysis
    if 'cluster_id' in df.columns:
        plt.figure(figsize=(12, 6))
        cluster_counts = df['cluster_id'].value_counts().sort_index()
        cluster_counts.plot(kind='bar')
        plt.title('Items per Cluster')
        plt.xlabel('Cluster ID')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=300)
        plt.close()
        print("✅ Created cluster distribution plot")
    
    # Time series analysis
    if 'created_utc' in df.columns:
        df['created_utc'] = pd.to_datetime(df['created_utc'])
        df['date'] = df['created_utc'].dt.date
        
        daily_counts = df.groupby('date').size()
        plt.figure(figsize=(12, 6))
        daily_counts.plot(kind='line')
        plt.title('Daily Activity')
        plt.xlabel('Date')
        plt.ylabel('Number of Posts/Comments')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'daily_activity.png'), dpi=300)
        plt.close()
        print("✅ Created daily activity plot")

def analyze_clusters(df: pd.DataFrame):
    """Analyze cluster content."""
    if 'cluster_id' not in df.columns:
        print("No cluster data available")
        return
    
    print("\n🔍 Cluster Analysis")
    print("=" * 30)
    
    for cluster_id in sorted(df['cluster_id'].dropna().unique()):
        cluster_data = df[df['cluster_id'] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_data)} items):")
        
        # Top subreddits in cluster
        if 'subreddit' in cluster_data.columns:
            top_subreddits = cluster_data['subreddit'].value_counts().head(3)
            print(f"  Top subreddits: {dict(top_subreddits)}")
        
        # Average score
        if 'score' in cluster_data.columns:
            avg_score = cluster_data['score'].mean()
            print(f"  Average score: {avg_score:.2f}")
        
        # Sample text
        if 'cleaned_text' in cluster_data.columns:
            sample_texts = cluster_data['cleaned_text'].dropna().head(2)
            print(f"  Sample texts:")
            for i, text in enumerate(sample_texts, 1):
                preview = text[:100] + "..." if len(text) > 100 else text
                print(f"    {i}. {preview}")

def main():
    """Main analysis function."""
    print("🐼 Reddit Data Analysis with Pandas")
    print("=" * 40)
    
    # Load data
    data = load_reddit_data()
    
    if not data:
        print("❌ No CSV data found. Run the pipeline first or check the csv_exports directory.")
        return
    
    # Analyze combined data
    if 'all' in data:
        df = data['all']
        basic_analysis(df)
        create_visualizations(df)
        analyze_clusters(df)
    
    # Analyze posts separately
    if 'posts' in data:
        print(f"\n📝 Posts Analysis ({len(data['posts'])} posts)")
        print("-" * 30)
        basic_analysis(data['posts'])
    
    # Analyze comments separately
    if 'comments' in data:
        print(f"\n💬 Comments Analysis ({len(data['comments'])} comments)")
        print("-" * 30)
        basic_analysis(data['comments'])
    
    print(f"\n🎉 Analysis complete! Check the 'pandas_analysis' directory for visualizations.")

if __name__ == "__main__":
    main()
