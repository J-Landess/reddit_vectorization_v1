#!/usr/bin/env python3
"""
Test script for historical tracking system.
"""
import os
import sys
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from tracking.historical_tracker import HistoricalTracker
from config import HISTORICAL_TRACKING

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_historical_tracker():
    """Test the historical tracking functionality."""
    logger.info("Testing historical tracking system...")
    
    # Initialize tracker
    tracker = HistoricalTracker()
    
    # Test configuration
    test_config = {
        'intelligent_filtering': True,
        'target_samples': 1000,
        'subreddits': ['healthinsurance', 'Medicare'],
        'test_run': True
    }
    
    # Start a test run
    run_id = tracker.start_run(test_config)
    logger.info(f"Started test run: {run_id}")
    
    # Test database backup (create a dummy file)
    dummy_db_path = 'test_database.db'
    with open(dummy_db_path, 'w') as f:
        f.write("dummy database content")
    
    backup_path = tracker.backup_database(dummy_db_path)
    if backup_path:
        logger.info(f"Database backup test successful: {backup_path}")
    
    # Clean up dummy file
    if os.path.exists(dummy_db_path):
        os.remove(dummy_db_path)
    
    # Test output preservation (create dummy outputs)
    os.makedirs('test_outputs', exist_ok=True)
    test_files = [
        'test_outputs/test_visualization.png',
        'test_outputs/test_report.html',
        'test_outputs/test_data.csv'
    ]
    
    for file_path in test_files:
        with open(file_path, 'w') as f:
            f.write(f"Test content for {file_path}")
    
    preserved_files = tracker.preserve_outputs('test_outputs')
    logger.info(f"Preserved {len(preserved_files)} test files")
    
    # Test run report creation
    test_stats = {
        'intelligent_filtering_enabled': True,
        'target_samples': 1000,
        'subreddits': ['healthinsurance', 'Medicare'],
        'total_items': 500,
        'posts_collected': 200,
        'comments_collected': 300,
        'filtering_effectiveness': '75%',
        'clustering_algorithm': 'hdbscan',
        'n_clusters': 3,
        'n_noise': 50,
        'silhouette_score': 0.45
    }
    
    report_path = tracker.create_run_report(test_stats)
    logger.info(f"Run report created: {report_path}")
    
    # Test run summary update
    tracker.update_run_summary(test_stats)
    logger.info("Run summary updated")
    
    # Test historical runs listing
    runs = tracker.get_historical_runs()
    logger.info(f"Found {len(runs)} historical runs")
    
    # Test comparison report (if multiple runs)
    if len(runs) > 1:
        comparison_path = tracker.create_comparison_report()
        if comparison_path:
            logger.info(f"Comparison report created: {comparison_path}")
    
    # Clean up test outputs
    import shutil
    if os.path.exists('test_outputs'):
        shutil.rmtree('test_outputs')
    
    logger.info("Historical tracking test completed successfully!")
    return True


def test_configuration():
    """Test historical tracking configuration."""
    logger.info("Testing historical tracking configuration...")
    
    # Check if configuration is loaded correctly
    assert HISTORICAL_TRACKING['enabled'] == True, "Historical tracking should be enabled"
    assert HISTORICAL_TRACKING['preserve_all_runs'] == True, "Preserve all runs should be enabled"
    assert HISTORICAL_TRACKING['timestamp_outputs'] == True, "Timestamp outputs should be enabled"
    assert HISTORICAL_TRACKING['backup_database'] == True, "Backup database should be enabled"
    assert HISTORICAL_TRACKING['create_run_summary'] == True, "Create run summary should be enabled"
    
    logger.info("Configuration test passed!")
    return True


def main():
    """Run all historical tracking tests."""
    logger.info("Starting historical tracking system tests...")
    
    try:
        # Test configuration
        test_configuration()
        
        # Test historical tracker
        test_historical_tracker()
        
        logger.info("All historical tracking tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"Historical tracking test failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
