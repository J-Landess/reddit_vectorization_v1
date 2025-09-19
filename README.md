# Reddit Data Analysis Pipeline

A comprehensive Python pipeline for collecting, processing, and analyzing Reddit data with embeddings and clustering. This tool is specifically designed for healthcare-related subreddits but can be easily adapted for other domains.

## 🚀 Features

- **Reddit API Integration**: Collect posts and comments from multiple subreddits using PRAW
- **Text Preprocessing**: Clean and preprocess text data (remove URLs, stopwords, etc.)
- **Embedding Generation**: Generate semantic embeddings using Sentence Transformers
- **Database Storage**: Store data and embeddings in SQLite database
- **Clustering Analysis**: Group similar posts using k-means or HDBSCAN
- **Visualization**: Generate comprehensive visualizations and interactive dashboards
- **Scalable Design**: Memory-efficient processing for large datasets

## 📋 Target Subreddits

The pipeline is configured to analyze these healthcare-related subreddits:
- r/healthinsurance
- r/Medicare
- r/Medicaid
- r/medicalproviders
- r/AskDocs
- r/Health
- r/ChronicIllness
- r/PatientExperience
- r/MedicalBilling
- r/Pharmacy
- r/MentalHealth
- r/medical
- r/Obamacare

## 🛠️ Installation

1. **Clone or download the project**:
   ```bash
   cd /path/to/your/projects
   # The project is already in your reddit directory
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Reddit API credentials**:
   Create a `.env` file in the project root:
   ```bash
   REDDIT_CLIENT_ID=your_client_id_here
   REDDIT_CLIENT_SECRET=your_client_secret_here
   REDDIT_USER_AGENT=RedditAnalysisBot/1.0
   ```

   Or set environment variables:
   ```bash
   export REDDIT_CLIENT_ID="your_client_id_here"
   export REDDIT_CLIENT_SECRET="your_client_secret_here"
   export REDDIT_USER_AGENT="RedditAnalysisBot/1.0"
   ```

## 🚀 Usage

### Basic Usage

Run the complete pipeline:
```bash
python main.py
```

### Advanced Usage

Run with debug logging:
```bash
python main.py --log-level DEBUG
```

### Configuration

You can customize the pipeline behavior by setting environment variables:

```bash
# Collection settings
export MAX_POSTS_PER_SUBREDDIT=500
export MAX_COMMENTS_PER_POST=25

# Embedding settings
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export EMBEDDING_BATCH_SIZE=32

# Clustering settings
export CLUSTERING_ALGORITHM="hdbscan"
export MIN_CLUSTER_SIZE=5
export MIN_SAMPLES=3

# Database settings
export DATABASE_PATH="./data/reddit_data.db"
```

## 📁 Project Structure

```
reddit/
├── main.py                 # Main entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── src/                  # Source code
│   ├── data_collection/  # Reddit API client
│   ├── preprocessing/    # Text cleaning utilities
│   ├── embeddings/       # Embedding generation
│   ├── database/         # SQLite database manager
│   ├── clustering/       # Clustering algorithms
│   ├── analysis/         # Analysis and visualization
│   └── pipeline.py       # Main pipeline orchestrator
├── data/                 # Database storage
├── outputs/              # Generated visualizations and reports
└── logs/                 # Log files
```

## 🔧 Components

### 1. Data Collection (`src/data_collection/`)
- **RedditClient**: Handles Reddit API interactions using PRAW
- Collects posts and comments from specified subreddits
- Implements rate limiting and error handling

### 2. Text Preprocessing (`src/preprocessing/`)
- **TextCleaner**: Cleans and preprocesses Reddit text
- Removes URLs, Reddit formatting, stopwords
- Handles special characters and punctuation

### 3. Embedding Generation (`src/embeddings/`)
- **EmbeddingGenerator**: Generates semantic embeddings
- Uses Sentence Transformers (default: all-MiniLM-L6-v2)
- Batch processing for memory efficiency

### 4. Database Management (`src/database/`)
- **DatabaseManager**: SQLite database operations
- Stores posts, comments, and embeddings
- Supports cluster assignments and metadata

### 5. Clustering Analysis (`src/clustering/`)
- **ClusterAnalyzer**: Clustering algorithms
- Supports k-means and HDBSCAN
- Quality metrics and cluster summaries

### 6. Analysis & Visualization (`src/analysis/`)
- **RedditAnalyzer**: Analysis and visualization tools
- Generates plots, word clouds, and interactive dashboards
- Exports cluster summaries and reports

## 📊 Outputs

The pipeline generates several outputs in the `outputs/` directory:

- **Visualizations**:
  - `subreddit_analysis.png`: Subreddit distribution plots
  - `cluster_analysis.png`: Clustering results visualization
  - `wordcloud_*.png`: Word clouds for each cluster
  - `reddit_dashboard.html`: Interactive dashboard

- **Reports**:
  - `cluster_summaries.md`: Detailed cluster analysis
  - `analysis_report.md`: Comprehensive analysis report

- **Database**:
  - `data/reddit_data.db`: SQLite database with all data

## ⚙️ Configuration Options

### Collection Settings
- `MAX_POSTS_PER_SUBREDDIT`: Maximum posts to collect per subreddit
- `MAX_COMMENTS_PER_POST`: Maximum comments per post
- `COLLECTION_LIMIT`: Overall collection limit

### Embedding Settings
- `EMBEDDING_MODEL`: Sentence Transformer model name
- `EMBEDDING_BATCH_SIZE`: Batch size for embedding generation

### Clustering Settings
- `CLUSTERING_ALGORITHM`: Algorithm to use ('hdbscan' or 'kmeans')
- `MIN_CLUSTER_SIZE`: Minimum cluster size for HDBSCAN
- `MIN_SAMPLES`: Minimum samples for HDBSCAN
- `N_CLUSTERS`: Number of clusters for k-means

## 🔍 Understanding the Results

### Cluster Analysis
Each cluster represents a group of similar posts/comments based on semantic similarity. The analysis includes:

- **Cluster Size**: Number of items in each cluster
- **Top Subreddits**: Most common subreddits in the cluster
- **Common Words**: Most frequent words in the cluster
- **Representative Posts**: Top-scoring posts that represent the cluster

### Quality Metrics
- **Silhouette Score**: Measures cluster quality (-1 to 1, higher is better)
- **Calinski-Harabasz Score**: Measures cluster separation (higher is better)

## 🚨 Troubleshooting

### Common Issues

1. **Reddit API Credentials**:
   - Ensure your Reddit API credentials are correctly set
   - Check that your Reddit app has the necessary permissions

2. **Memory Issues**:
   - Reduce `EMBEDDING_BATCH_SIZE` for large datasets
   - Process data in smaller chunks

3. **Rate Limiting**:
   - The pipeline includes built-in rate limiting
   - If you encounter rate limits, the pipeline will retry automatically

4. **Dependencies**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Some packages may require additional system dependencies

### Logs
Check the `logs/` directory for detailed execution logs. Logs include:
- Data collection progress
- Processing statistics
- Error messages and stack traces

## 🤝 Contributing

This is a modular design that can be easily extended:

1. **Add new subreddits**: Modify `SUBREDDITS` in `config.py`
2. **Custom preprocessing**: Extend `TextCleaner` class
3. **Different embeddings**: Change `EMBEDDING_MODEL` in configuration
4. **New clustering algorithms**: Add to `ClusterAnalyzer` class
5. **Additional visualizations**: Extend `RedditAnalyzer` class

## 📝 License

This project is for educational and research purposes. Please respect Reddit's API terms of service and the subreddit communities you're analyzing.

## 🆘 Support

For issues or questions:
1. Check the logs in the `logs/` directory
2. Verify your Reddit API credentials
3. Ensure all dependencies are installed
4. Check the configuration settings

---

**Happy analyzing! 🔍📊**
