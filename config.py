"""
Configuration settings for Reddit data analysis pipeline.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Reddit API Configuration
REDDIT_CONFIG = {
    'client_id': os.getenv('REDDIT_CLIENT_ID'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
    'user_agent': os.getenv('REDDIT_USER_AGENT', 'RedditAnalysisBot/1.0')
}

# Target subreddits - Full healthcare spectrum
SUBREDDITS = [
    'healthinsurance',
    'Medicare',
    'Medicaid',
    'medicalproviders',
    'AskDocs',
    'Health',
    'ChronicIllness',
    'PatientExperience',
    'MedicalBilling',
    'Pharmacy',
    'MentalHealth',
    'medical',
    'Obamacare'
]

# Collection settings - SCALED UP for 50k samples
COLLECTION_CONFIG = {
    'max_posts_per_subreddit': int(os.getenv('MAX_POSTS_PER_SUBREDDIT', 100)),  # Increased from 20 to 100
    'max_comments_per_post': int(os.getenv('MAX_COMMENTS_PER_POST', 200)),      # Increased from 50 to 200
    'collection_limit': int(os.getenv('COLLECTION_LIMIT', 100)),                # Increased from 20 to 100
    'filter_noise': os.getenv('FILTER_NOISE', 'true').lower() == 'true'        # Filter out bot messages and guidelines
}

# Database configuration
DATABASE_CONFIG = {
    'path': os.getenv('DATABASE_PATH', './data/reddit_data.db')
}

# Embedding configuration
EMBEDDING_CONFIG = {
    'model_name': os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
    'batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', 32))
}

# Clustering configuration
CLUSTERING_CONFIG = {
    'algorithm': os.getenv('CLUSTERING_ALGORITHM', 'hdbscan'),
    'min_cluster_size': int(os.getenv('MIN_CLUSTER_SIZE', 5)),
    'min_samples': int(os.getenv('MIN_SAMPLES', 3))
}

# Intelligent filtering configuration - ADJUSTED for 50k target
INTELLIGENT_FILTERING = {
    'enabled': os.getenv('INTELLIGENT_FILTERING', 'true').lower() == 'true',
    'target_samples': int(os.getenv('TARGET_SAMPLES', 50000)),  # Target 50k samples
    'min_relevance_score': float(os.getenv('MIN_RELEVANCE_SCORE', 0.05)),  # Lowered from 0.1 to 0.05 for more samples
    'prioritize_healthcare': os.getenv('PRIORITIZE_HEALTHCARE', 'true').lower() == 'true'
}

# Historical tracking configuration
HISTORICAL_TRACKING = {
    'enabled': os.getenv('HISTORICAL_TRACKING', 'true').lower() == 'true',
    'preserve_all_runs': os.getenv('PRESERVE_ALL_RUNS', 'true').lower() == 'true',
    'timestamp_outputs': os.getenv('TIMESTAMP_OUTPUTS', 'true').lower() == 'true',
    'backup_database': os.getenv('BACKUP_DATABASE', 'true').lower() == 'true',
    'create_run_summary': os.getenv('CREATE_RUN_SUMMARY', 'true').lower() == 'true'
}