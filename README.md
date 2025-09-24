# Reddit Data Analysis Pipeline

A comprehensive Python pipeline for collecting, processing, and analyzing Reddit data with embeddings, clustering, and topic modeling. Designed for healthcare-related subreddits with intelligent filtering, historical run tracking, and CSV exports.

## 🚀 Features

- **Reddit API Integration**: Collect posts and comments from multiple subreddits using PRAW
- **Intelligent Filtering**: Heuristics + keywords to prioritize healthcare experiences and de-noise content
- **Text Preprocessing**: Clean and preprocess text data (URLs, stopwords, punctuation)
- **Embeddings**: Sentence Transformers (default: `all-MiniLM-L6-v2`)
- **SQLite Storage**: Persist raw/clean text, metadata, embeddings, clusters
- **Clustering**: HDBSCAN or k-means; representative items per cluster
- **Topic Modeling**: Gensim LDA, scikit-learn LDA, and NMF (+ BERTopic optional)
- **CSV Exports**: Posts/comments/combined data and topic outputs to `csv_exports/` and `outputs/`
- **Historical Tracking**: Time-stamped run directories with DB backups and reports
- **Visualizations**: Word clouds, cluster plots, dashboards, and analysis reports
- **Scalable Design**: Batching, sampling, and resumable collection utilities

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

2. **(Recommended) Create a conda environment**:
   ```bash
   conda create -n reddit-analysis python=3.9 -y
   conda activate reddit-analysis
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Reddit API credentials**:
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

Run the complete pipeline (collection → preprocessing → embeddings → clustering → exports/plots):
```bash
python main.py
```

### Advanced Usage

Run with debug logging:
```bash
python main.py --log-level DEBUG
```

### Sentiment Analyzer Selection

You can choose the sentiment analyzer:

```bash
# Use default VADER
python main.py --analyzer vader

# Use Hugging Face transformer (distilbert-sst2)
python main.py --analyzer transformer
```

Environment variable alternative:

```bash
export SENTIMENT_ANALYZER=transformer
python main.py
```

### Focused utilities

- Export the database to CSVs:
```bash
python export_to_csv.py
```

- Sandbox test both sentiment analyzers on sample texts (no DB writes):
```bash
python sentiment_sandbox.py
```

- Evaluate analyzers on SST-2 or fallback mini dataset:
```bash
python evaluate_analyzers.py
```

- Analyze convexity and cluster separability (saves `convexity_analysis.png`):
```bash
python analyze_convexity.py
```

- Run topic modeling (Gensim LDA, sklearn LDA, NMF) and generate a report/visuals:
```bash
python run_topic_analysis.py
```

- Collect repeatedly until target sample count (uses intelligent filtering and historical tracking):
```bash
python collect_to_50k.py
```

### Configuration

You can customize the pipeline behavior by setting environment variables:

You can customize behavior with environment variables (see `config.py` for defaults):

```bash
# Collection
export MAX_POSTS_PER_SUBREDDIT=100
export MAX_COMMENTS_PER_POST=200
export COLLECTION_LIMIT=100
export FILTER_NOISE=true

# Intelligent filtering
export INTELLIGENT_FILTERING=true
export TARGET_SAMPLES=50000
export MIN_RELEVANCE_SCORE=0.05
export PRIORITIZE_HEALTHCARE=true

# Historical tracking
export HISTORICAL_TRACKING=true
export PRESERVE_ALL_RUNS=true
export TIMESTAMP_OUTPUTS=true
export BACKUP_DATABASE=true
export CREATE_RUN_SUMMARY=true

# Embeddings
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export EMBEDDING_BATCH_SIZE=32

# Clustering
export CLUSTERING_ALGORITHM="hdbscan"   # or "kmeans"
export MIN_CLUSTER_SIZE=25
export MIN_SAMPLES=5
export N_CLUSTERS=20                     # for k-means

# Database
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
│   ├── topic_modeling/   # Topic modeling (LDA, NMF, BERTopic)
│   ├── tracking/         # Historical run tracking utilities
│   └── pipeline.py       # Main pipeline orchestrator
├── data/                 # Database storage
├── outputs/              # Generated visualizations and reports
├── csv_exports/          # CSV exports from DB (posts, comments, combined)
├── historical_runs/      # Time-stamped preserved runs (DB backups, reports)
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

### 7. Topic Modeling (`src/topic_modeling/`)
- **TopicAnalyzer**: Gensim LDA, sklearn LDA, NMF (BERTopic optional)
- Preprocessing, topic extraction, coherence support (for Gensim LDA)
- `run_topic_analysis.py` generates `outputs/topic_modeling_comprehensive_report.md`

### 8. Historical Tracking (`src/tracking/`)
- **HistoricalTracker**: Creates `historical_runs/run_<timestamp>/` with:
  - DB backups, preserved PNG/HTML/CSV, run report and comparison report

## 📊 Outputs

The pipeline generates several outputs in the `outputs/` directory:

- **Visualizations**:
  - `subreddit_analysis.png`: Subreddit distribution plots
  - `cluster_analysis.png`: Clustering results visualization
  - `wordcloud_*.png`: Word clouds for each cluster
  - `reddit_dashboard.html`: Interactive dashboard
  - `topic_visualizations.png`: Topic modeling plots

- **Reports**:
  - `cluster_summaries.md`: Detailed cluster analysis
  - `analysis_report.md`: Comprehensive analysis report
  - `topic_modeling_comprehensive_report.md`: Topic modeling results
  - `historical_runs/comparison_report.md`: Cross-run comparison

- **Database**:
  - `data/reddit_data.db`: SQLite database with all data

- **CSV Exports**:
  - `csv_exports/reddit_posts.csv`, `reddit_comments.csv`, `reddit_all_data.csv`
  - `outputs/topic_assignments.csv`, `outputs/topic_summaries.csv`

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
 - Convexity analysis (see `analyze_convexity.py`) includes separation ratio and hull plots

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

5. **CSV/Text Type Errors**:
   - If you see `float object is not subscriptable` during topic export, ensure you’re on the latest code. We normalize non-string/NaN text before slicing in `run_topic_analysis.py`.

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
6. **Topic modeling**: Adjust algorithms/params in `src/topic_modeling/`
7. **Historical tracking**: Tune preservation in `config.py` → `HISTORICAL_TRACKING`

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
