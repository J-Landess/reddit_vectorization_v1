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

# Collection settings - Balanced approach for comprehensive analysis
COLLECTION_CONFIG = {
    'max_posts_per_subreddit': int(os.getenv('MAX_POSTS_PER_SUBREDDIT', 20)),  # Reasonable for 13 subreddits
    'max_comments_per_post': int(os.getenv('MAX_COMMENTS_PER_POST', 50)),      # Good sample size
    'collection_limit': int(os.getenv('COLLECTION_LIMIT', 20)),                # Manageable limit
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
