"""
Classification analysis and visualization tools for medical category classification.
"""
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ClassificationAnalyzer:
    """Analyzer for medical category classification results."""
    
    def __init__(self, output_dir: str = './outputs'):
        """
        Initialize the classification analyzer.
        
        Args:
            output_dir: Directory to save analysis outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Medical categories
        self.categories = ['medical_insurance', 'medical_provider', 'medical_broker', 'employer', 'policy_changes']
        self.category_labels = {
            'medical_insurance': 'Medical Insurance',
            'medical_provider': 'Medical Provider',
            'medical_broker': 'Medical Broker',
            'employer': 'Employer',
            'policy_changes': 'Policy Changes/Healthcare Legislation/Regulation'
        }
    
    def analyze_classification_results(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze classification results and generate statistics.
        
        Args:
            data: List of data with classification results
            
        Returns:
            Dictionary with analysis results
        """
        if not data:
            return {}
        
        # Filter data with classification results
        classified_data = [item for item in data if item.get('category')]
        
        if not classified_data:
            logger.warning("No classified data found")
            return {}
        
        # Basic statistics
        total_items = len(classified_data)
        category_counts = Counter([item['category'] for item in classified_data])
        
        # Confidence statistics
        confidences = [item.get('category_confidence', 0) for item in classified_data]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Category distribution
        category_distribution = {
            category: {
                'count': category_counts.get(category, 0),
                'percentage': (category_counts.get(category, 0) / total_items) * 100,
                'avg_confidence': np.mean([
                    item.get('category_confidence', 0) 
                    for item in classified_data 
                    if item.get('category') == category
                ]) if category_counts.get(category, 0) > 0 else 0
            }
            for category in self.categories
        }
        
        # Confidence distribution
        confidence_ranges = {
            'high_confidence': len([c for c in confidences if c >= 0.8]),
            'medium_confidence': len([c for c in confidences if 0.5 <= c < 0.8]),
            'low_confidence': len([c for c in confidences if c < 0.5])
        }
        
        # By subreddit analysis
        subreddit_analysis = {}
        for item in classified_data:
            subreddit = item.get('subreddit', 'unknown')
            if subreddit not in subreddit_analysis:
                subreddit_analysis[subreddit] = Counter()
            subreddit_analysis[subreddit][item['category']] += 1
        
        # By content type analysis
        content_type_analysis = {
            'posts': Counter([item['category'] for item in classified_data if item.get('type') == 'post']),
            'comments': Counter([item['category'] for item in classified_data if item.get('type') == 'comment'])
        }
        
        results = {
            'total_classified_items': total_items,
            'category_distribution': category_distribution,
            'confidence_statistics': {
                'average_confidence': avg_confidence,
                'confidence_ranges': confidence_ranges,
                'min_confidence': min(confidences) if confidences else 0,
                'max_confidence': max(confidences) if confidences else 0
            },
            'subreddit_analysis': subreddit_analysis,
            'content_type_analysis': content_type_analysis
        }
        
        logger.info(f"Classification analysis completed for {total_items} items")
        return results
    
    def create_classification_visualizations(self, data: List[Dict[str, Any]], 
                                           analysis_results: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Create visualizations for classification results.
        
        Args:
            data: List of data with classification results
            analysis_results: Pre-computed analysis results
            
        Returns:
            Dictionary mapping visualization name to file path
        """
        if analysis_results is None:
            analysis_results = self.analyze_classification_results(data)
        
        if not analysis_results:
            logger.warning("No analysis results to visualize")
            return {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        visualizations = {}
        
        try:
            # 1. Category Distribution Pie Chart
            self._create_category_pie_chart(analysis_results, timestamp)
            visualizations['category_pie'] = f'{self.output_dir}/classification_category_distribution_{timestamp}.png'
            
            # 2. Category Confidence Box Plot
            self._create_confidence_box_plot(data, timestamp)
            visualizations['confidence_box'] = f'{self.output_dir}/classification_confidence_box_{timestamp}.png'
            
            # 3. Subreddit Category Heatmap
            self._create_subreddit_heatmap(analysis_results, timestamp)
            visualizations['subreddit_heatmap'] = f'{self.output_dir}/classification_subreddit_heatmap_{timestamp}.png'
            
            # 4. Confidence Distribution Histogram
            self._create_confidence_histogram(data, timestamp)
            visualizations['confidence_histogram'] = f'{self.output_dir}/classification_confidence_histogram_{timestamp}.png'
            
            # 5. Category Trends Over Time (if timestamps available)
            self._create_temporal_analysis(data, timestamp)
            visualizations['temporal_analysis'] = f'{self.output_dir}/classification_temporal_{timestamp}.png'
            
            logger.info(f"Created {len(visualizations)} classification visualizations")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
        
        return visualizations
    
    def _create_category_pie_chart(self, analysis_results: Dict[str, Any], timestamp: str):
        """Create pie chart for category distribution."""
        category_dist = analysis_results.get('category_distribution', {})
        
        labels = []
        sizes = []
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        for category in self.categories:
            if category_dist.get(category, {}).get('count', 0) > 0:
                labels.append(self.category_labels[category])
                sizes.append(category_dist[category]['count'])
        
        if not sizes:
            return
        
        plt.figure(figsize=(10, 8))
        wedges, texts, autotexts = plt.pie(sizes, labels=labels, colors=colors[:len(sizes)], 
                                          autopct='%1.1f%%', startangle=90)
        
        plt.title('Medical Category Distribution', fontsize=16, fontweight='bold')
        plt.axis('equal')
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/classification_category_distribution_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confidence_box_plot(self, data: List[Dict[str, Any]], timestamp: str):
        """Create box plot for confidence by category."""
        classified_data = [item for item in data if item.get('category') and item.get('category_confidence')]
        
        if not classified_data:
            return
        
        # Prepare data for box plot
        plot_data = []
        categories = []
        
        for category in self.categories:
            category_confidences = [
                item['category_confidence'] 
                for item in classified_data 
                if item.get('category') == category
            ]
            if category_confidences:
                plot_data.append(category_confidences)
                categories.append(self.category_labels[category])
        
        if not plot_data:
            return
        
        plt.figure(figsize=(12, 8))
        box_plot = plt.boxplot(plot_data, labels=categories, patch_artist=True)
        
        # Color the boxes
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        for patch, color in zip(box_plot['boxes'], colors[:len(categories)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        plt.title('Classification Confidence by Category', fontsize=16, fontweight='bold')
        plt.xlabel('Medical Category', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/classification_confidence_box_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_subreddit_heatmap(self, analysis_results: Dict[str, Any], timestamp: str):
        """Create heatmap for subreddit vs category distribution."""
        subreddit_analysis = analysis_results.get('subreddit_analysis', {})
        
        if not subreddit_analysis:
            return
        
        # Create DataFrame for heatmap
        heatmap_data = []
        subreddits = list(subreddit_analysis.keys())
        
        for subreddit in subreddits:
            row = []
            total = sum(subreddit_analysis[subreddit].values())
            for category in self.categories:
                count = subreddit_analysis[subreddit].get(category, 0)
                percentage = (count / total) * 100 if total > 0 else 0
                row.append(percentage)
            heatmap_data.append(row)
        
        df = pd.DataFrame(heatmap_data, 
                         index=subreddits,
                         columns=[self.category_labels[cat] for cat in self.categories])
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Percentage'})
        
        plt.title('Category Distribution by Subreddit', fontsize=16, fontweight='bold')
        plt.xlabel('Medical Category', fontsize=12)
        plt.ylabel('Subreddit', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/classification_subreddit_heatmap_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confidence_histogram(self, data: List[Dict[str, Any]], timestamp: str):
        """Create histogram for confidence distribution."""
        classified_data = [item for item in data if item.get('category_confidence')]
        
        if not classified_data:
            return
        
        confidences = [item['category_confidence'] for item in classified_data]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        
        plt.title('Distribution of Classification Confidence Scores', fontsize=16, fontweight='bold')
        plt.xlabel('Confidence Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_conf = np.mean(confidences)
        plt.axvline(mean_conf, color='red', linestyle='--', 
                   label=f'Mean: {mean_conf:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/classification_confidence_histogram_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_analysis(self, data: List[Dict[str, Any]], timestamp: str):
        """Create temporal analysis of categories over time."""
        classified_data = [item for item in data if item.get('category') and item.get('created_utc')]
        
        if not classified_data:
            return
        
        # Convert timestamps and group by day
        df = pd.DataFrame(classified_data)
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
        df['date'] = df['created_utc'].dt.date
        
        # Count categories by date
        daily_counts = df.groupby(['date', 'category']).size().unstack(fill_value=0)
        
        if daily_counts.empty:
            return
        
        plt.figure(figsize=(14, 8))
        
        for category in self.categories:
            if category in daily_counts.columns:
                plt.plot(daily_counts.index, daily_counts[category], 
                        label=self.category_labels[category], marker='o', linewidth=2)
        
        plt.title('Category Trends Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Items', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/classification_temporal_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_classification_report(self, analysis_results: Dict[str, Any], 
                                   output_file: str = None) -> str:
        """
        Export detailed classification analysis report.
        
        Args:
            analysis_results: Analysis results dictionary
            output_file: Optional output file path
            
        Returns:
            Path to the exported report file
        """
        if not analysis_results:
            logger.warning("No analysis results to export")
            return ""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if output_file is None:
            output_file = f'{self.output_dir}/classification_report_{timestamp}.md'
        
        with open(output_file, 'w') as f:
            f.write("# Medical Category Classification Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Classified Items**: {analysis_results.get('total_classified_items', 0)}\n")
            
            conf_stats = analysis_results.get('confidence_statistics', {})
            f.write(f"- **Average Confidence**: {conf_stats.get('average_confidence', 0):.3f}\n")
            f.write(f"- **Min Confidence**: {conf_stats.get('min_confidence', 0):.3f}\n")
            f.write(f"- **Max Confidence**: {conf_stats.get('max_confidence', 0):.3f}\n\n")
            
            # Category distribution
            f.write("## Category Distribution\n\n")
            category_dist = analysis_results.get('category_distribution', {})
            
            f.write("| Category | Count | Percentage | Avg Confidence |\n")
            f.write("|----------|-------|------------|----------------|\n")
            
            for category in self.categories:
                cat_data = category_dist.get(category, {})
                f.write(f"| {self.category_labels[category]} | "
                       f"{cat_data.get('count', 0)} | "
                       f"{cat_data.get('percentage', 0):.1f}% | "
                       f"{cat_data.get('avg_confidence', 0):.3f} |\n")
            
            f.write("\n")
            
            # Confidence ranges
            f.write("## Confidence Distribution\n\n")
            conf_ranges = conf_stats.get('confidence_ranges', {})
            f.write(f"- **High Confidence (≥0.8)**: {conf_ranges.get('high_confidence', 0)} items\n")
            f.write(f"- **Medium Confidence (0.5-0.8)**: {conf_ranges.get('medium_confidence', 0)} items\n")
            f.write(f"- **Low Confidence (<0.5)**: {conf_ranges.get('low_confidence', 0)} items\n\n")
            
            # Subreddit analysis
            f.write("## Subreddit Analysis\n\n")
            subreddit_analysis = analysis_results.get('subreddit_analysis', {})
            
            for subreddit, counts in subreddit_analysis.items():
                f.write(f"### r/{subreddit}\n")
                total = sum(counts.values())
                for category in self.categories:
                    count = counts.get(category, 0)
                    percentage = (count / total) * 100 if total > 0 else 0
                    f.write(f"- {self.category_labels[category]}: {count} ({percentage:.1f}%)\n")
                f.write("\n")
        
        logger.info(f"Classification report exported to {output_file}")
        return output_file
