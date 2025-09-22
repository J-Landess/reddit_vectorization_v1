#!/usr/bin/env python3
"""
Script to collect data in multiple runs until reaching 50k samples.
"""
import os
import sys
import time
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import RedditAnalysisPipeline
from config import INTELLIGENT_FILTERING

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_current_sample_count():
    """Get current sample count from database."""
    import sqlite3
    
    db_path = 'data/reddit_data.db'
    if not os.path.exists(db_path):
        return 0
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM posts')
    posts_count = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM comments')
    comments_count = cursor.fetchone()[0]
    
    conn.close()
    return posts_count + comments_count


def collect_to_target(target_samples=50000, max_runs=20):
    """
    Run multiple collection cycles until reaching target samples.
    
    Args:
        target_samples: Target number of samples
        max_runs: Maximum number of runs to prevent infinite loops
    """
    logger.info(f"Starting collection to reach {target_samples:,} samples")
    
    initial_count = get_current_sample_count()
    logger.info(f"Current samples in database: {initial_count:,}")
    
    if initial_count >= target_samples:
        logger.info(f"Already have {initial_count:,} samples! Target reached.")
        return
    
    runs_completed = 0
    current_count = initial_count
    
    while current_count < target_samples and runs_completed < max_runs:
        runs_completed += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"RUN {runs_completed}/{max_runs}")
        logger.info(f"Current samples: {current_count:,}")
        logger.info(f"Target samples: {target_samples:,}")
        logger.info(f"Remaining: {target_samples - current_count:,}")
        logger.info(f"{'='*60}")
        
        try:
            # Run the pipeline
            pipeline = RedditAnalysisPipeline()
            pipeline.run_full_pipeline()
            
            # Check new count
            new_count = get_current_sample_count()
            samples_added = new_count - current_count
            
            logger.info(f"Run {runs_completed} completed!")
            logger.info(f"Samples added this run: {samples_added:,}")
            logger.info(f"Total samples now: {new_count:,}")
            
            current_count = new_count
            
            # Check if we've reached the target
            if current_count >= target_samples:
                logger.info(f"🎉 TARGET REACHED! {current_count:,} samples collected!")
                break
            
            # Wait between runs to be respectful to Reddit API
            if runs_completed < max_runs:
                wait_time = 300  # 5 minutes
                logger.info(f"Waiting {wait_time} seconds before next run...")
                time.sleep(wait_time)
                
        except Exception as e:
            logger.error(f"Error in run {runs_completed}: {e}")
            logger.info("Continuing with next run...")
            time.sleep(60)  # Wait 1 minute before retry
    
    # Final summary
    final_count = get_current_sample_count()
    total_added = final_count - initial_count
    
    logger.info(f"\n{'='*60}")
    logger.info(f"COLLECTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Initial samples: {initial_count:,}")
    logger.info(f"Final samples: {final_count:,}")
    logger.info(f"Total added: {total_added:,}")
    logger.info(f"Runs completed: {runs_completed}")
    logger.info(f"Target reached: {'Yes' if final_count >= target_samples else 'No'}")
    
    if final_count < target_samples:
        logger.warning(f"Target not reached. Consider:")
        logger.warning(f"1. Increasing collection limits in config.py")
        logger.warning(f"2. Running more cycles")
        logger.warning(f"3. Adjusting intelligent filtering parameters")


def main():
    """Main entry point."""
    target = INTELLIGENT_FILTERING.get('target_samples', 50000)
    max_runs = 20  # Adjust as needed
    
    logger.info(f"Starting collection to {target:,} samples (max {max_runs} runs)")
    
    try:
        collect_to_target(target, max_runs)
    except KeyboardInterrupt:
        logger.info("\nCollection interrupted by user")
        current_count = get_current_sample_count()
        logger.info(f"Current sample count: {current_count:,}")
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        raise


if __name__ == "__main__":
    main()
