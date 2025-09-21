"""
Historical tracking system for preserving all pipeline runs and outputs.
"""
import os
import shutil
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List
import logging
import glob

logger = logging.getLogger(__name__)


class HistoricalTracker:
    """Tracks and preserves all pipeline runs with timestamps."""
    
    def __init__(self, base_dir: str = '.'):
        """
        Initialize historical tracker.
        
        Args:
            base_dir: Base directory for the project
        """
        self.base_dir = base_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(base_dir, 'historical_runs', f'run_{self.run_id}')
        
        # Create run directory
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Subdirectories for different outputs
        self.dirs = {
            'outputs': os.path.join(self.run_dir, 'outputs'),
            'logs': os.path.join(self.run_dir, 'logs'),
            'csv_exports': os.path.join(self.run_dir, 'csv_exports'),
            'database_backups': os.path.join(self.run_dir, 'database_backups'),
            'visualizations': os.path.join(self.run_dir, 'visualizations'),
            'reports': os.path.join(self.run_dir, 'reports')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"Historical tracker initialized for run {self.run_id}")
    
    def start_run(self, config: Dict[str, Any]) -> str:
        """
        Start a new historical run.
        
        Args:
            config: Pipeline configuration
            
        Returns:
            Run ID
        """
        # Save configuration
        config_file = os.path.join(self.run_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        # Create run summary
        summary = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'config': config,
            'status': 'started'
        }
        
        summary_file = os.path.join(self.run_dir, 'run_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Started historical run {self.run_id}")
        return self.run_id
    
    def backup_database(self, db_path: str) -> str:
        """
        Backup the current database.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            Path to backup file
        """
        if not os.path.exists(db_path):
            logger.warning(f"Database file not found: {db_path}")
            return None
        
        backup_path = os.path.join(
            self.dirs['database_backups'], 
            f'reddit_data_backup_{self.run_id}.db'
        )
        
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    
    def preserve_outputs(self, source_dir: str) -> Dict[str, str]:
        """
        Preserve all output files with timestamps.
        
        Args:
            source_dir: Source directory containing outputs
            
        Returns:
            Dictionary mapping original paths to preserved paths
        """
        preserved_files = {}
        
        if not os.path.exists(source_dir):
            logger.warning(f"Source directory not found: {source_dir}")
            return preserved_files
        
        # Preserve different types of outputs
        output_mappings = {
            'outputs': ['*.png', '*.html', '*.md'],
            'logs': ['*.log'],
            'csv_exports': ['*.csv']
        }
        
        for output_type, patterns in output_mappings.items():
            target_dir = self.dirs[output_type]
            
            for pattern in patterns:
                files = glob.glob(os.path.join(source_dir, pattern))
                
                for file_path in files:
                    if os.path.isfile(file_path):
                        filename = os.path.basename(file_path)
                        # Add timestamp to filename
                        name, ext = os.path.splitext(filename)
                        timestamped_name = f"{name}_{self.run_id}{ext}"
                        target_path = os.path.join(target_dir, timestamped_name)
                        
                        shutil.copy2(file_path, target_path)
                        preserved_files[file_path] = target_path
        
        logger.info(f"Preserved {len(preserved_files)} output files")
        return preserved_files
    
    def create_run_report(self, pipeline_stats: Dict[str, Any]) -> str:
        """
        Create a comprehensive run report.
        
        Args:
            pipeline_stats: Statistics from the pipeline run
            
        Returns:
            Path to the report file
        """
        report_path = os.path.join(self.dirs['reports'], f'run_report_{self.run_id}.md')
        
        with open(report_path, 'w') as f:
            f.write(f"# Reddit Analysis Pipeline Run Report\n\n")
            f.write(f"**Run ID:** {self.run_id}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Intelligent Filtering:** {pipeline_stats.get('intelligent_filtering_enabled', 'Unknown')}\n")
            f.write(f"- **Target Samples:** {pipeline_stats.get('target_samples', 'Unknown')}\n")
            f.write(f"- **Subreddits:** {len(pipeline_stats.get('subreddits', []))}\n\n")
            
            f.write("## Collection Results\n\n")
            f.write(f"- **Total Items Collected:** {pipeline_stats.get('total_items', 'Unknown')}\n")
            f.write(f"- **Posts:** {pipeline_stats.get('posts_collected', 'Unknown')}\n")
            f.write(f"- **Comments:** {pipeline_stats.get('comments_collected', 'Unknown')}\n")
            f.write(f"- **Filtering Effectiveness:** {pipeline_stats.get('filtering_effectiveness', 'Unknown')}\n\n")
            
            f.write("## Clustering Results\n\n")
            f.write(f"- **Algorithm:** {pipeline_stats.get('clustering_algorithm', 'Unknown')}\n")
            f.write(f"- **Number of Clusters:** {pipeline_stats.get('n_clusters', 'Unknown')}\n")
            f.write(f"- **Noise Points:** {pipeline_stats.get('n_noise', 'Unknown')}\n")
            f.write(f"- **Silhouette Score:** {pipeline_stats.get('silhouette_score', 'Unknown')}\n\n")
            
            f.write("## Files Generated\n\n")
            f.write(f"- **Database Backup:** `database_backups/reddit_data_backup_{self.run_id}.db`\n")
            f.write(f"- **CSV Exports:** `csv_exports/`\n")
            f.write(f"- **Visualizations:** `visualizations/`\n")
            f.write(f"- **Logs:** `logs/`\n")
            f.write(f"- **Configuration:** `config.json`\n\n")
            
            f.write("## Next Steps\n\n")
            f.write("1. Review the visualizations in the `visualizations/` directory\n")
            f.write("2. Analyze the CSV exports in the `csv_exports/` directory\n")
            f.write("3. Compare with previous runs in the `historical_runs/` directory\n")
            f.write("4. Use the database backup for further analysis\n")
        
        logger.info(f"Run report created: {report_path}")
        return report_path
    
    def update_run_summary(self, stats: Dict[str, Any]) -> None:
        """
        Update the run summary with final statistics.
        
        Args:
            stats: Final pipeline statistics
        """
        summary_file = os.path.join(self.run_dir, 'run_summary.json')
        
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {'run_id': self.run_id}
        
        summary.update({
            'end_time': datetime.now().isoformat(),
            'status': 'completed',
            'final_stats': stats
        })
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Run summary updated: {summary_file}")
    
    def get_historical_runs(self) -> List[Dict[str, Any]]:
        """
        Get list of all historical runs.
        
        Returns:
            List of run information dictionaries
        """
        historical_dir = os.path.join(self.base_dir, 'historical_runs')
        
        if not os.path.exists(historical_dir):
            return []
        
        runs = []
        for run_dir in os.listdir(historical_dir):
            if run_dir.startswith('run_'):
                run_path = os.path.join(historical_dir, run_dir)
                summary_file = os.path.join(run_path, 'run_summary.json')
                
                if os.path.exists(summary_file):
                    with open(summary_file, 'r') as f:
                        run_info = json.load(f)
                        runs.append(run_info)
        
        # Sort by start time
        runs.sort(key=lambda x: x.get('start_time', ''), reverse=True)
        return runs
    
    def create_comparison_report(self) -> str:
        """
        Create a comparison report across all historical runs.
        
        Returns:
            Path to the comparison report
        """
        runs = self.get_historical_runs()
        
        if len(runs) < 2:
            logger.warning("Need at least 2 runs to create comparison report")
            return None
        
        comparison_path = os.path.join(
            self.base_dir, 'historical_runs', 'comparison_report.md'
        )
        
        with open(comparison_path, 'w') as f:
            f.write("# Reddit Analysis Pipeline - Historical Comparison\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Runs:** {len(runs)}\n\n")
            
            f.write("## Run Summary\n\n")
            f.write("| Run ID | Start Time | Status | Total Items | Clusters | Filtering % |\n")
            f.write("|--------|------------|--------|-------------|----------|-------------|\n")
            
            for run in runs:
                stats = run.get('final_stats', {})
                f.write(f"| {run.get('run_id', 'Unknown')} | ")
                f.write(f"{run.get('start_time', 'Unknown')[:19]} | ")
                f.write(f"{run.get('status', 'Unknown')} | ")
                f.write(f"{stats.get('total_items', 'Unknown')} | ")
                f.write(f"{stats.get('n_clusters', 'Unknown')} | ")
                f.write(f"{stats.get('filtering_effectiveness', 'Unknown')} |\n")
            
            f.write("\n## Trend Analysis\n\n")
            f.write("### Data Collection Growth\n")
            f.write("- Track how your dataset grows over time\n")
            f.write("- Monitor filtering effectiveness improvements\n")
            f.write("- Analyze cluster evolution\n\n")
            
            f.write("### Recommendations\n")
            f.write("1. **Consistent Collection:** Run the pipeline regularly to build a comprehensive dataset\n")
            f.write("2. **Filter Tuning:** Adjust filtering parameters based on effectiveness trends\n")
            f.write("3. **Cluster Analysis:** Compare cluster themes across runs to identify patterns\n")
            f.write("4. **Quality Metrics:** Monitor data quality and relevance over time\n")
        
        logger.info(f"Comparison report created: {comparison_path}")
        return comparison_path
