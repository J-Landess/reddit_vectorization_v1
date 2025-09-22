#!/usr/bin/env python3
"""
Run comprehensive topic modeling analysis on Reddit healthcare data.
"""
import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from topic_modeling.topic_pipeline import TopicModelingPipeline
from topic_modeling.topic_analyzer import TopicAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_comprehensive_topic_analysis():
    """Run comprehensive topic modeling analysis."""
    print("🚀 REDDIT HEALTHCARE TOPIC MODELING ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("📊 Loading Reddit healthcare data...")
    try:
        df = pd.read_csv('csv_exports/reddit_all_data.csv')
        print(f"✅ Loaded {len(df):,} total records")
    except FileNotFoundError:
        print("❌ CSV file not found. Please run the main pipeline first.")
        return
    
    # Convert to data format expected by pipeline
    data = []
    for _, row in df.iterrows():
        item = {
            'id': row.get('id', ''),
            'type': row.get('type', ''),
            'text': row.get('text', ''),
            'cleaned_text': row.get('cleaned_text', ''),
            'subreddit': row.get('subreddit', ''),
            'score': row.get('score', 0),
            'author': row.get('author', ''),
            'created_utc': row.get('created_utc', ''),
            'title': row.get('title', ''),
            'word_count': row.get('word_count', 0),
            'char_count': row.get('char_count', 0)
        }
        data.append(item)
    
    print(f"📝 Processing {len(data):,} items for topic modeling")
    
    # Test different algorithms
    algorithms = ['gensim_lda', 'lda', 'nmf']
    results = {}
    
    for algorithm in algorithms:
        print(f"\n🔍 Running {algorithm.upper()} analysis...")
        
        try:
            # Use a sample for faster processing
            sample_size = min(5000, len(data))
            sample_data = data[:sample_size]
            
            pipeline = TopicModelingPipeline(algorithm=algorithm, n_topics=20)
            result = pipeline.run_topic_analysis(sample_data)
            
            results[algorithm] = result
            
            print(f"✅ {algorithm.upper()} completed successfully")
            print(f"   Documents processed: {result['n_documents_processed']}")
            print(f"   Topics found: {result['topic_statistics'].get('n_topics', 'N/A')}")
            
            # Show top topics
            topic_summaries = result['topic_summaries']
            print(f"   Top 5 topics:")
            for topic_id, summary in list(topic_summaries.items())[:5]:
                print(f"     Topic {topic_id}: {summary['top_words']} ({summary['size']} items)")
            
        except Exception as e:
            print(f"❌ Error with {algorithm}: {e}")
            logger.error(f"Error with {algorithm}: {e}")
    
    # Generate comprehensive report
    print(f"\n📄 Generating comprehensive report...")
    
    # Use the best performing algorithm (gensim_lda)
    if 'gensim_lda' in results:
        best_result = results['gensim_lda']
        
        # Generate detailed report
        report_path = generate_comprehensive_report(best_result, data[:5000])
        print(f"✅ Comprehensive report generated: {report_path}")
        
        # Generate topic visualizations
        generate_topic_visualizations(best_result)
        
        # Export topic data
        export_topic_data(best_result, data[:5000])
    
    print(f"\n🎉 Topic modeling analysis completed!")
    print(f"📁 Check the 'outputs' directory for all results")

def generate_comprehensive_report(results, data):
    """Generate a comprehensive topic modeling report."""
    report_path = 'outputs/topic_modeling_comprehensive_report.md'
    os.makedirs('outputs', exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("# Reddit Healthcare Topic Modeling Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Algorithm:** {results['algorithm']}\n")
        f.write(f"**Documents Processed:** {results['n_documents_processed']:,}\n\n")
        
        # Model performance
        model_results = results['model_results']
        f.write("## Model Performance\n\n")
        for key, value in model_results.items():
            f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
        
        # Topic statistics
        topic_stats = results['topic_statistics']
        f.write(f"\n## Topic Statistics\n\n")
        f.write(f"- **Number of Topics:** {topic_stats.get('n_topics', 'N/A')}\n")
        f.write(f"- **Total Documents:** {topic_stats.get('total_documents', 'N/A'):,}\n")
        f.write(f"- **Largest Topic Size:** {topic_stats.get('largest_topic_size', 'N/A')}\n")
        f.write(f"- **Smallest Topic Size:** {topic_stats.get('smallest_topic_size', 'N/A')}\n")
        f.write(f"- **Average Topic Size:** {topic_stats.get('avg_topic_size', 'N/A'):.1f}\n")
        
        # Topic distribution
        f.write(f"\n## Topic Distribution\n\n")
        topic_dist = topic_stats.get('topic_distribution', {})
        f.write("| Topic ID | Size | Percentage |\n")
        f.write("|----------|------|------------|\n")
        total_docs = topic_stats.get('total_documents', 1)
        for topic_id, size in sorted(topic_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (size / total_docs) * 100
            f.write(f"| {topic_id} | {size} | {percentage:.1f}% |\n")
        
        # Detailed topic summaries
        topic_summaries = results['topic_summaries']
        f.write(f"\n## Detailed Topic Analysis\n\n")
        
        for topic_id, summary in sorted(topic_summaries.items(), key=lambda x: x[1]['size'], reverse=True):
            f.write(f"### Topic {topic_id}\n\n")
            f.write(f"- **Size:** {summary['size']} documents ({summary['size']/total_docs*100:.1f}%)\n")
            f.write(f"- **Top Words:** {', '.join([str(word) for word in summary['top_words']])}\n")
            f.write(f"- **Top Subreddits:** {', '.join([str(subr) for subr in summary['top_subreddits']])}\n")
            f.write(f"- **Average Score:** {summary['avg_score']:.2f}\n\n")
            
            # Representative documents
            if summary['representatives']:
                f.write("**Representative Documents:**\n")
                for i, rep in enumerate(summary['representatives'][:3]):
                    # Try cleaned_text first, then text, then title
                    text = rep.get('cleaned_text', '') or rep.get('text', '') or rep.get('title', '')
                    if isinstance(text, (int, float)) and np.isnan(text):
                        text = 'No text available'
                    elif not isinstance(text, str):
                        text = str(text)
                    text_preview = text[:300] + "..." if len(text) > 300 else text
                    f.write(f"{i+1}. **{rep.get('subreddit', 'Unknown')}** - {text_preview}\n\n")
        
        # Healthcare insights
        f.write(f"\n## Healthcare Insights\n\n")
        f.write("### Key Themes Identified:\n")
        
        # Analyze topics for healthcare themes
        healthcare_themes = analyze_healthcare_themes(topic_summaries)
        for theme, topics in healthcare_themes.items():
            f.write(f"- **{theme}:** {', '.join([f'Topic {t}' for t in topics])}\n")
        
        f.write(f"\n### Recommendations:\n")
        f.write("1. **Focus on High-Impact Topics:** Prioritize topics with high document counts and engagement\n")
        f.write("2. **Subreddit-Specific Analysis:** Analyze topic distribution across different healthcare subreddits\n")
        f.write("3. **Temporal Analysis:** Track how topics evolve over time\n")
        f.write("4. **Sentiment Analysis:** Add sentiment analysis to understand emotional context\n")
        f.write("5. **Actionable Insights:** Use topic themes to inform healthcare policy and practice\n")
    
    return report_path

def analyze_healthcare_themes(topic_summaries):
    """Analyze topics to identify healthcare themes."""
    themes = {
        'Mental Health': [],
        'Insurance & Billing': [],
        'Medical Conditions': [],
        'Medication & Treatment': [],
        'Healthcare Access': [],
        'General Discussion': []
    }
    
    for topic_id, summary in topic_summaries.items():
        # Convert top_words to strings and handle numpy types
        top_words_list = summary.get('top_words', [])
        if isinstance(top_words_list, list) and len(top_words_list) > 0:
            # Convert each word to string, handling numpy types
            top_words = ' '.join([str(word) for word in top_words_list]).lower()
        else:
            top_words = ''
        
        if any(word in top_words for word in ['depression', 'anxiety', 'mental', 'therapy', 'therapist', 'suicide', 'bipolar']):
            themes['Mental Health'].append(topic_id)
        elif any(word in top_words for word in ['insurance', 'billing', 'claim', 'medicare', 'medicaid', 'coverage', 'premium']):
            themes['Insurance & Billing'].append(topic_id)
        elif any(word in top_words for word in ['pain', 'symptom', 'diagnosis', 'condition', 'chronic', 'illness']):
            themes['Medical Conditions'].append(topic_id)
        elif any(word in top_words for word in ['medication', 'drug', 'prescription', 'treatment', 'therapy', 'medication']):
            themes['Medication & Treatment'].append(topic_id)
        elif any(word in top_words for word in ['access', 'afford', 'cost', 'expensive', 'free', 'available']):
            themes['Healthcare Access'].append(topic_id)
        else:
            themes['General Discussion'].append(topic_id)
    
    return themes

def generate_topic_visualizations(results):
    """Generate topic visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        topic_stats = results['topic_statistics']
        topic_dist = topic_stats.get('topic_distribution', {})
        
        # Topic size distribution
        plt.figure(figsize=(12, 8))
        topics = list(topic_dist.keys())
        sizes = list(topic_dist.values())
        
        plt.subplot(2, 2, 1)
        plt.bar(topics, sizes)
        plt.title('Topic Size Distribution')
        plt.xlabel('Topic ID')
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=45)
        
        # Top 10 topics
        plt.subplot(2, 2, 2)
        sorted_topics = sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        top_topics = [str(t[0]) for t in sorted_topics]
        top_sizes = [t[1] for t in sorted_topics]
        plt.bar(top_topics, top_sizes)
        plt.title('Top 10 Topics by Size')
        plt.xlabel('Topic ID')
        plt.ylabel('Number of Documents')
        plt.xticks(rotation=45)
        
        # Topic size histogram
        plt.subplot(2, 2, 3)
        plt.hist(sizes, bins=20, alpha=0.7)
        plt.title('Topic Size Distribution (Histogram)')
        plt.xlabel('Number of Documents')
        plt.ylabel('Frequency')
        
        # Cumulative distribution
        plt.subplot(2, 2, 4)
        sorted_sizes = sorted(sizes, reverse=True)
        cumulative = [sum(sorted_sizes[:i+1]) for i in range(len(sorted_sizes))]
        plt.plot(range(len(cumulative)), cumulative)
        plt.title('Cumulative Topic Size Distribution')
        plt.xlabel('Topic Rank')
        plt.ylabel('Cumulative Documents')
        
        plt.tight_layout()
        plt.savefig('outputs/topic_visualizations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Topic visualizations generated: outputs/topic_visualizations.png")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping visualizations")
    except Exception as e:
        print(f"⚠️  Error generating visualizations: {e}")

def export_topic_data(results, data):
    """Export topic modeling results to CSV."""
    try:
        # Create topic assignments dataframe
        topic_assignments = []
        for item in data:
            topic_assignments.append({
                'id': item.get('id', ''),
                'type': item.get('type', ''),
                'subreddit': item.get('subreddit', ''),
                'topic_id': item.get('topic_id', -1),
                'text': item.get('text', '')[:500],  # Truncate for CSV
                'score': item.get('score', 0),
                'author': item.get('author', ''),
                'created_utc': item.get('created_utc', '')
            })
        
        df_assignments = pd.DataFrame(topic_assignments)
        df_assignments.to_csv('outputs/topic_assignments.csv', index=False)
        
        # Create topic summaries dataframe
        topic_summaries = results['topic_summaries']
        topic_summary_data = []
        for topic_id, summary in topic_summaries.items():
            topic_summary_data.append({
                'topic_id': topic_id,
                'size': summary['size'],
                'top_words': ', '.join(summary['top_words']),
                'top_subreddits': ', '.join(summary['top_subreddits']),
                'avg_score': summary['avg_score']
            })
        
        df_summaries = pd.DataFrame(topic_summary_data)
        df_summaries.to_csv('outputs/topic_summaries.csv', index=False)
        
        print("✅ Topic data exported:")
        print("   - outputs/topic_assignments.csv")
        print("   - outputs/topic_summaries.csv")
        
    except Exception as e:
        print(f"⚠️  Error exporting topic data: {e}")

if __name__ == "__main__":
    run_comprehensive_topic_analysis()
