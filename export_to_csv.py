#!/usr/bin/env python3
"""
Export Reddit data from SQLite database to CSV files for pandas analysis.
"""
import os
import sys
import pandas as pd
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from database.database_manager import DatabaseManager
from config import DATABASE_CONFIG

def export_database_to_csv(db_path: str, output_dir: str = './csv_exports'):
    """
    Export all data from the SQLite database to CSV files.
    
    Args:
        db_path: Path to the SQLite database
        output_dir: Directory to save CSV files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize database manager
    with DatabaseManager(db_path) as db:
        print(f"📊 Exporting data from {db_path}")
        
        # Get all data
        all_data = db.get_all_data(include_embeddings=False)
        
        if not all_data:
            print("❌ No data found in database")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Separate posts and comments
        posts_df = df[df['type'] == 'post'].copy()
        comments_df = df[df['type'] == 'comment'].copy()
        
        # Clean up the data for CSV export
        def clean_dataframe(df):
            """Clean DataFrame for CSV export."""
            # Convert datetime objects to strings
            if 'created_utc' in df.columns:
                df['created_utc'] = df['created_utc'].astype(str)
            
            # Convert boolean columns
            if 'is_self' in df.columns:
                df['is_self'] = df['is_self'].fillna(False).astype(int)
            
            # Handle numeric columns with NaN values
            numeric_columns = ['score', 'upvote_ratio', 'num_comments', 'word_count', 'char_count', 'cluster_id']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Ensure sentiment columns exist in export even if missing
            for col in ['sentiment', 'confidence']:
                if col not in df.columns:
                    df[col] = '' if col == 'sentiment' else 0.0

            # Fill remaining NaN values
            df = df.fillna('')
            
            return df
        
        # Clean and export posts (overwrite canonical filenames)
        if not posts_df.empty:
            posts_clean = clean_dataframe(posts_df)
            posts_file = os.path.join(output_dir, 'reddit_posts.csv')
            posts_clean.to_csv(posts_file, index=False, encoding='utf-8')
            print(f"✅ Exported {len(posts_clean)} posts to {posts_file}")
        
        # Clean and export comments (overwrite canonical filenames)
        if not comments_df.empty:
            comments_clean = clean_dataframe(comments_df)
            comments_file = os.path.join(output_dir, 'reddit_comments.csv')
            comments_clean.to_csv(comments_file, index=False, encoding='utf-8')
            print(f"✅ Exported {len(comments_clean)} comments to {comments_file}")
        
        # Export combined data (overwrite canonical filename)
        all_clean = clean_dataframe(df)
        combined_file = os.path.join(output_dir, 'reddit_all_data.csv')
        all_clean.to_csv(combined_file, index=False, encoding='utf-8')
        print(f"✅ Exported {len(all_clean)} total items to {combined_file}")
        
        # Export cluster summaries
        cluster_summary_file = os.path.join(output_dir, 'cluster_summary.csv')
        if 'cluster_id' in df.columns:
            cluster_summary = df.groupby('cluster_id').agg({
                'id': 'count',
                'subreddit': lambda x: ', '.join(x.value_counts().head(3).index),
                'score': 'mean',
                'word_count': 'mean',
                'type': lambda x: x.value_counts().to_dict()
            }).round(2)
            cluster_summary.columns = ['count', 'top_subreddits', 'avg_score', 'avg_word_count', 'type_distribution']
            cluster_summary.to_csv(cluster_summary_file)
            print(f"✅ Exported cluster summary to {cluster_summary_file}")
        
        # Export subreddit analysis
        subreddit_file = os.path.join(output_dir, 'subreddit_analysis.csv')
        if 'subreddit' in df.columns:
            subreddit_analysis = df.groupby('subreddit').agg({
                'id': 'count',
                'score': ['mean', 'std', 'min', 'max'],
                'word_count': ['mean', 'std', 'min', 'max'],
                'type': lambda x: x.value_counts().to_dict()
            }).round(2)
            subreddit_analysis.columns = [
                'total_items', 'avg_score', 'std_score', 'min_score', 'max_score',
                'avg_words', 'std_words', 'min_words', 'max_words', 'type_distribution'
            ]
            subreddit_analysis.to_csv(subreddit_file)
            print(f"✅ Exported subreddit analysis to {subreddit_file}")
        
        # Print summary statistics
        print(f"\n📈 Data Summary:")
        print(f"   Total items: {len(df)}")
        print(f"   Posts: {len(posts_df)}")
        print(f"   Comments: {len(comments_df)}")
        print(f"   Subreddits: {df['subreddit'].nunique() if 'subreddit' in df.columns else 'N/A'}")
        print(f"   Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
        
        if 'cluster_id' in df.columns:
            print(f"   Clusters: {df['cluster_id'].nunique()}")
            print(f"   Clustered items: {len(df[df['cluster_id'].notna()])}")

def create_analysis_notebook(output_dir: str = './csv_exports'):
    """
    Create a Jupyter notebook template for data analysis.
    
    Args:
        output_dir: Directory containing CSV files
    """
    notebook_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Reddit Data Analysis with Pandas\\n",
    "\\n",
    "This notebook provides a template for analyzing Reddit data exported from the pipeline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "from datetime import datetime\\n",
    "\\n",
    "# Set up plotting\\n",
    "plt.style.use('seaborn-v0_8')\\n",
    "sns.set_palette('husl')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the data\\n",
    "posts_df = pd.read_csv('reddit_posts.csv')\\n",
    "comments_df = pd.read_csv('reddit_comments.csv')\\n",
    "all_data_df = pd.read_csv('reddit_all_data.csv')\\n",
    "\\n",
    "print(f'Loaded {len(posts_df)} posts and {len(comments_df)} comments')\\n",
    "print(f'Total items: {len(all_data_df)}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic data exploration\\n",
    "print('Data Overview:')\\n",
    "print(all_data_df.info())\\n",
    "print('\\nFirst few rows:')\\n",
    "print(all_data_df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Subreddit analysis\\n",
    "subreddit_counts = all_data_df['subreddit'].value_counts()\\n",
    "print('Posts/Comments by Subreddit:')\\n",
    "print(subreddit_counts)\\n",
    "\\n",
    "plt.figure(figsize=(12, 6))\\n",
    "subreddit_counts.plot(kind='bar')\\n",
    "plt.title('Posts/Comments by Subreddit')\\n",
    "plt.xlabel('Subreddit')\\n",
    "plt.ylabel('Count')\\n",
    "plt.xticks(rotation=45)\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Score analysis\\n",
    "plt.figure(figsize=(15, 5))\\n",
    "\\n",
    "plt.subplot(1, 3, 1)\\n",
    "all_data_df['score'].hist(bins=30)\\n",
    "plt.title('Score Distribution')\\n",
    "plt.xlabel('Score')\\n",
    "plt.ylabel('Frequency')\\n",
    "\\n",
    "plt.subplot(1, 3, 2)\\n",
    "all_data_df.boxplot(column='score', by='subreddit')\\n",
    "plt.title('Score by Subreddit')\\n",
    "plt.xticks(rotation=45)\\n",
    "\\n",
    "plt.subplot(1, 3, 3)\\n",
    "all_data_df['word_count'].hist(bins=30)\\n",
    "plt.title('Word Count Distribution')\\n",
    "plt.xlabel('Word Count')\\n",
    "plt.ylabel('Frequency')\\n",
    "\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cluster analysis (if available)\\n",
    "if 'cluster_id' in all_data_df.columns:\\n",
    "    cluster_counts = all_data_df['cluster_id'].value_counts().sort_index()\\n",
    "    print('Cluster Distribution:')\\n",
    "    print(cluster_counts)\\n",
    "    \\n",
    "    plt.figure(figsize=(12, 6))\\n",
    "    cluster_counts.plot(kind='bar')\\n",
    "    plt.title('Items per Cluster')\\n",
    "    plt.xlabel('Cluster ID')\\n",
    "    plt.ylabel('Count')\\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Time series analysis\\n",
    "all_data_df['created_utc'] = pd.to_datetime(all_data_df['created_utc'])\\n",
    "all_data_df['date'] = all_data_df['created_utc'].dt.date\\n",
    "\\n",
    "daily_counts = all_data_df.groupby('date').size()\\n",
    "plt.figure(figsize=(12, 6))\\n",
    "daily_counts.plot(kind='line')\\n",
    "plt.title('Daily Activity')\\n",
    "plt.xlabel('Date')\\n",
    "plt.ylabel('Number of Posts/Comments')\\n",
    "plt.xticks(rotation=45)\\n",
    "plt.tight_layout()\\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''
    
    notebook_file = os.path.join(output_dir, 'reddit_analysis_template.ipynb')
    with open(notebook_file, 'w') as f:
        f.write(notebook_content)
    
    print(f"✅ Created analysis notebook template: {notebook_file}")

def main():
    """Main function for CSV export."""
    parser = argparse.ArgumentParser(description='Export Reddit data to CSV files')
    parser.add_argument('--db-path', default=DATABASE_CONFIG['path'], 
                       help='Path to SQLite database')
    parser.add_argument('--output-dir', default='./csv_exports',
                       help='Output directory for CSV files')
    parser.add_argument('--create-notebook', action='store_true',
                       help='Create Jupyter notebook template')
    parser.add_argument('--prune-old', action='store_true',
                       help='Delete old timestamped CSVs and keep canonical files only')
    
    args = parser.parse_args()
    
    print("🔄 Reddit Data CSV Export Tool")
    print("=" * 40)
    
    if not os.path.exists(args.db_path):
        print(f"❌ Database not found: {args.db_path}")
        print("Run the pipeline first to collect data.")
        return
    
    try:
        export_database_to_csv(args.db_path, args.output_dir)

        # Optional prune: remove old timestamped CSVs lacking sentiment or duplicates
        if args.prune_old:
            removed = 0
            for name in os.listdir(args.output_dir):
                if not name.endswith('.csv'):
                    continue
                if name.startswith('reddit_posts_') or name.startswith('reddit_comments_') or name.startswith('reddit_all_data_'):
                    try:
                        os.remove(os.path.join(args.output_dir, name))
                        removed += 1
                    except Exception:
                        pass
            if removed:
                print(f"🧹 Pruned {removed} old timestamped CSV(s)")
        
        if args.create_notebook:
            create_analysis_notebook(args.output_dir)
        
        print(f"\n🎉 Export completed! Check the '{args.output_dir}' directory.")
        print("\nNext steps:")
        print("1. Open the CSV files in pandas: pd.read_csv('reddit_posts.csv')")
        print("2. Use the Jupyter notebook template for analysis")
        print("3. Explore the data with your favorite tools!")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
