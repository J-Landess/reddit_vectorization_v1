#!/usr/bin/env python3
"""
Reddit Data Analysis Pipeline
Main entry point for the Reddit data collection and analysis system.
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from pipeline import RedditAnalysisPipeline

def setup_logging(log_level: str = 'INFO'):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/reddit_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def check_environment():
    """Check if required environment variables are set."""
    required_vars = ['REDDIT_CLIENT_ID', 'REDDIT_CLIENT_SECRET']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your environment or create a .env file.")
        print("Example .env file:")
        print("REDDIT_CLIENT_ID=your_client_id_here")
        print("REDDIT_CLIENT_SECRET=your_client_secret_here")
        print("REDDIT_USER_AGENT=RedditAnalysisBot/1.0")
        return False
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Reddit Data Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run full pipeline
  python main.py --log-level DEBUG        # Run with debug logging
  python main.py --help                   # Show this help message

Environment Variables Required:
  REDDIT_CLIENT_ID     - Your Reddit API client ID
  REDDIT_CLIENT_SECRET - Your Reddit API client secret
  REDDIT_USER_AGENT    - User agent string (optional, defaults to RedditAnalysisBot/1.0)

Optional Environment Variables:
  MAX_POSTS_PER_SUBREDDIT - Maximum posts per subreddit (default: 1000)
  MAX_COMMENTS_PER_POST   - Maximum comments per post (default: 50)
  EMBEDDING_MODEL         - Sentence transformer model (default: all-MiniLM-L6-v2)
  CLUSTERING_ALGORITHM    - Clustering algorithm (default: hdbscan)
        """
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set the logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Reddit Analysis Pipeline v1.0'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print banner
    print("=" * 60)
    print("🔍 Reddit Data Analysis Pipeline")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log level: {args.log_level}")
    print()
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    try:
        # Initialize and run pipeline
        logger.info("Initializing Reddit Analysis Pipeline")
        pipeline = RedditAnalysisPipeline()
        
        logger.info("Starting pipeline execution")
        pipeline.run_full_pipeline()
        
        print("\n✅ Pipeline completed successfully!")
        print(f"Check the 'outputs' directory for results and visualizations.")
        print(f"Check the 'logs' directory for detailed logs.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
