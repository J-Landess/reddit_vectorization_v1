#!/usr/bin/env python3
"""
CLI to generate standard plots from the Reddit SQLite database.

Usage examples:
  python tools/plots_cli.py --db ./data/reddit_data.db --all
  python tools/plots_cli.py --db ./data/reddit_data.db --run-id 20250923_114819
  python tools/plots_cli.py --db ./data/reddit_data.db --subreddit Health --prefix health
  python tools/plots_cli.py --db ./data/reddit_data.db --convex-hull --embeddings-csv ./csv_exports/reddit_all_data_*.csv
"""
import argparse
import glob
import os
import sys
from typing import List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.db_queries import RedditDBQueries
from analysis.plotting import RedditPlotter
import pandas as pd


def load_df(db_path: str, run_id: str = None, subreddit: str = None) -> pd.DataFrame:
    db = RedditDBQueries(db_path)
    if run_id:
        df = db.get_data_by_run(run_id)
    elif subreddit:
        df = db.get_posts_by_subreddit(subreddit)
    else:
        df = db.get_all_posts()
    return df


def maybe_load_embeddings_from_csv(pattern: str) -> pd.DataFrame:
    paths: List[str] = sorted(glob.glob(pattern))
    if not paths:
        return None
    # Pick the latest file
    latest = paths[-1]
    try:
        return pd.read_csv(latest)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Generate plots for Reddit data')
    parser.add_argument('--db', required=True, help='Path to SQLite database')
    parser.add_argument('--output', default='./outputs', help='Output directory for plots')
    parser.add_argument('--run-id', help='Filter by run_id')
    parser.add_argument('--subreddit', help='Filter by subreddit')
    parser.add_argument('--prefix', default='reddit', help='Filename prefix for outputs')
    parser.add_argument('--all', action='store_true', help='Generate all standard plots')
    parser.add_argument('--convex-hull', action='store_true', help='Generate embeddings convex hull plot (requires embeddings)')
    parser.add_argument('--embeddings-csv', help='CSV path pattern to load embeddings (from combined export)')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = load_df(args.db, args.run_id, args.subreddit)
    plotter = RedditPlotter(args.output)

    results = {}
    if args.all:
        results.update(plotter.generate_all(df, prefix=args.prefix))

    if args.convex_hull:
        emb_df = None
        if args.embeddings_csv:
            emb_df = maybe_load_embeddings_from_csv(args.embeddings_csv)
        if emb_df is None:
            print('Convex hull skipped: embeddings CSV not provided/found. Use --embeddings-csv with a CSV containing embeddings.')
        else:
            # Expect a column 'embedding' with JSON-like list strings; parse if present
            if 'embedding' not in emb_df.columns:
                print('Convex hull skipped: embedding column not found in CSV.')
            else:
                import json
                import numpy as np
                try:
                    embeddings = np.vstack(emb_df['embedding'].apply(lambda s: np.array(json.loads(s), dtype=float)))
                    labels = emb_df.get('cluster_id', None)
                    path = plotter.plot_embeddings_convex_hull(embeddings, labels)
                    results['embeddings_convex_hull'] = path
                except Exception as e:
                    print(f'Convex hull failed: {e}')

    if results:
        print('Generated plots:')
        for k, v in results.items():
            print(f'  - {k}: {v}')
    else:
        print('No plots generated. Consider using --all and/or --convex-hull with --embeddings-csv.')


if __name__ == '__main__':
    main()
