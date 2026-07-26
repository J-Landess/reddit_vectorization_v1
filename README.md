# Reddit Data Analysis Pipeline

Python tooling to collect, process, and analyze Reddit data (embeddings, clustering, topic modeling, optional sentiment and medical-category classification). Aimed at healthcare-related subreddits.

## Installation

This package is **not** published as a verified PyPI distribution in this repo’s workflow. Install from source:

```bash
git clone https://github.com/justinlandess/reddit.git
cd reddit
pip install -e .
# or: pip install -r requirements.txt
```

`pyproject.toml` names the project `reddit-analysis-pipeline` and registers a `reddit-pipeline` console script pointing at `pipeline:main` (package root is `src/`). Prefer running from the repo root with the scripts below if the console entry point is awkward in your environment.

## Features

- Reddit API collection via PRAW
- Filtering / preprocessing, Sentence Transformers embeddings
- SQLite storage, HDBSCAN / k-means clustering
- Topic modeling (Gensim LDA, sklearn LDA/NMF; BERTopic optional)
- Sentiment (VADER / transformer) and **classification** utilities (`src/classification/`)
- CSV exports, historical run tracking, visualizations

## Setup

Create a `.env` in the repo root:

```bash
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_USER_AGENT=RedditAnalysisBot/1.0
```

## Usage

Run from the **repository root** so `src/` is on the path (scripts append `src` themselves).

### Main entry points

```bash
python main.py
python main.py --analyzer transformer
python main.py --log-level DEBUG
```

### Python API (from repo / after editable install)

Packages live under `src/` as top-level modules (`pipeline`, `classification`, …), not `reddit_analysis_pipeline`:

```python
# From repo root scripts (same pattern as main.py):
from pipeline import RedditAnalysisPipeline

pipeline = RedditAnalysisPipeline(analyzer_type='vader')
pipeline.run_full_pipeline()
```

Some examples use `from src.pipeline import ...` after adding the repo to `sys.path` — either style works if you run from the repo root as those scripts do.

### Utilities that exist

```bash
python export_to_csv.py
python evaluate_analyzers.py
python run_topic_analysis.py
python collect_to_50k.py
python analyze_existing_data.py --analyzer vader
python analyze_existing_data.py --analyzer transformer --limit 500
python example_classification.py
python example_usage.py
python example_pandas_analysis.py
python organize_outputs.py
```

See **[CLASSIFICATION_GUIDE.md](CLASSIFICATION_GUIDE.md)** for medical-category classification (`RuleBasedClassifier`, `MLClassifier`, `HybridClassifier`, embedding-similarity helpers).

### Scripts referenced elsewhere but not present

These are **not** in the repo (do not expect them to run):

- `sentiment_sandbox.py`
- `analyze_convexity.py` (a `convexity_analysis.png` artifact may still exist)

## Configuration

Environment variables are documented in `config.py` (collection limits, subreddits, embeddings, clustering, historical tracking, sentiment, classification). Examples:

```bash
export MAX_POSTS_PER_SUBREDDIT=100
export SUBREDDITS="healthinsurance,Medicare,Medicaid"
export CLUSTERING_ALGORITHM=hdbscan
export SENTIMENT_ANALYZER=vader
export DATABASE_PATH=./data/reddit_data.db
```

## Project structure

```
reddit/
├── main.py
├── config.py
├── pyproject.toml
├── requirements.txt
├── CLASSIFICATION_GUIDE.md
├── example_classification.py
├── export_to_csv.py
├── run_topic_analysis.py
├── collect_to_50k.py
├── analyze_existing_data.py
└── src/
    ├── pipeline.py
    ├── data_collection/
    ├── preprocessing/
    ├── embeddings/
    ├── database/
    ├── clustering/
    ├── analysis/
    ├── sentiment/
    ├── classification/
    ├── topic_modeling/
    └── tracking/
```

## License

For educational and research use. Respect Reddit’s API terms and community norms.
