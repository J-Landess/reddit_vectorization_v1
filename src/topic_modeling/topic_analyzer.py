"""
Topic modeling analysis for Reddit healthcare data using multiple algorithms.
"""
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import re

# Topic modeling libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
# coherence_score is not available in sklearn.metrics, using gensim instead
import gensim
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel
from bertopic import BERTopic
from umap import UMAP

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)


class TopicAnalyzer:
    """Advanced topic modeling for Reddit healthcare data."""
    
    def __init__(self, algorithm: str = 'lda', n_topics: int = 20, random_state: int = 42):
        """
        Initialize the topic analyzer.
        
        Args:
            algorithm: Topic modeling algorithm ('lda', 'nmf', 'bertopic', 'gensim_lda')
            n_topics: Number of topics to discover
            random_state: Random state for reproducibility
        """
        self.algorithm = algorithm.lower()
        self.n_topics = n_topics
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.topic_labels = None
        self.topic_probabilities = None
        self.vocabulary = None
        
        # Initialize NLTK data
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass
        
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Healthcare-specific stop words
        self.healthcare_stopwords = {
            'insurance', 'health', 'medical', 'doctor', 'hospital', 'patient',
            'treatment', 'medication', 'care', 'healthcare', 'would', 'could',
            'should', 'like', 'think', 'know', 'get', 'go', 'see', 'want',
            'need', 'help', 'time', 'year', 'day', 'work', 'good', 'bad',
            'also', 'really', 'much', 'even', 'still', 'way', 'make', 'take'
        }
        
        logger.info(f"Initialized topic analyzer with {algorithm}, {n_topics} topics")
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """
        Preprocess texts for topic modeling.
        
        Args:
            texts: List of text documents
            
        Returns:
            Preprocessed texts
        """
        logger.info(f"Preprocessing {len(texts)} texts for topic modeling")
        
        processed_texts = []
        
        for text in texts:
            if not text or not isinstance(text, str):
                processed_texts.append("")
                continue
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs, mentions, and special characters
            text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            text = re.sub(r'@\w+|#\w+', '', text)
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize and lemmatize
            tokens = word_tokenize(text)
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words 
                     and token not in self.healthcare_stopwords
                     and len(token) > 2]
            
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
        
        logger.info("Text preprocessing completed")
        return processed_texts
    
    def fit_lda(self, texts: List[str]) -> Dict[str, Any]:
        """
        Fit Latent Dirichlet Allocation (LDA) model.
        
        Args:
            texts: Preprocessed texts
            
        Returns:
            Model results
        """
        logger.info("Fitting LDA model")
        
        # Vectorize texts
        self.vectorizer = CountVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit LDA model
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=100,
            learning_method='batch'
        )
        
        self.model.fit(doc_term_matrix)
        
        # Get topic assignments
        topic_probs = self.model.transform(doc_term_matrix)
        self.topic_labels = np.argmax(topic_probs, axis=1)
        self.topic_probabilities = topic_probs
        
        # Calculate perplexity
        perplexity = self.model.perplexity(doc_term_matrix)
        
        return {
            'model_type': 'sklearn_lda',
            'perplexity': perplexity,
            'n_topics': self.n_topics,
            'n_documents': len(texts)
        }
    
    def fit_nmf(self, texts: List[str]) -> Dict[str, Any]:
        """
        Fit Non-negative Matrix Factorization (NMF) model.
        
        Args:
            texts: Preprocessed texts
            
        Returns:
            Model results
        """
        logger.info("Fitting NMF model")
        
        # Vectorize texts with TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        
        # Fit NMF model
        self.model = NMF(
            n_components=self.n_topics,
            random_state=self.random_state,
            max_iter=200
        )
        
        topic_weights = self.model.fit_transform(doc_term_matrix)
        
        # Get topic assignments
        self.topic_labels = np.argmax(topic_weights, axis=1)
        self.topic_probabilities = topic_weights
        
        # Calculate reconstruction error
        reconstruction_error = self.model.reconstruction_err_
        
        return {
            'model_type': 'nmf',
            'reconstruction_error': reconstruction_error,
            'n_topics': self.n_topics,
            'n_documents': len(texts)
        }
    
    def fit_gensim_lda(self, texts: List[str]) -> Dict[str, Any]:
        """
        Fit Gensim LDA model with coherence calculation.
        
        Args:
            texts: Preprocessed texts
            
        Returns:
            Model results
        """
        logger.info("Fitting Gensim LDA model")
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts if text.strip()]
        
        # Create dictionary and corpus
        dictionary = corpora.Dictionary(tokenized_texts)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Fit LDA model
        self.model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=self.n_topics,
            random_state=self.random_state,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Calculate coherence
        coherence_model = CoherenceModel(
            model=self.model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        # Get topic assignments
        topic_assignments = []
        topic_probs = []
        
        for doc in corpus:
            doc_topics = self.model.get_document_topics(doc, minimum_probability=0)
            doc_topics_dict = dict(doc_topics)
            
            # Get dominant topic
            dominant_topic = max(doc_topics_dict, key=doc_topics_dict.get)
            topic_assignments.append(dominant_topic)
            
            # Get probability distribution
            probs = [doc_topics_dict.get(i, 0) for i in range(self.n_topics)]
            topic_probs.append(probs)
        
        self.topic_labels = np.array(topic_assignments)
        self.topic_probabilities = np.array(topic_probs)
        self.vocabulary = dictionary
        
        return {
            'model_type': 'gensim_lda',
            'coherence_score': coherence_score,
            'perplexity': self.model.log_perplexity(corpus),
            'n_topics': self.n_topics,
            'n_documents': len(texts)
        }
    
    def fit_bertopic(self, texts: List[str]) -> Dict[str, Any]:
        """
        Fit BERTopic model for advanced topic modeling.
        
        Args:
            texts: Raw texts (BERTopic handles preprocessing)
            
        Returns:
            Model results
        """
        logger.info("Fitting BERTopic model")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and text.strip()]
        
        # Configure UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=self.random_state
        )
        
        # Fit BERTopic model
        self.model = BERTopic(
            umap_model=umap_model,
            nr_topics=self.n_topics,
            calculate_probabilities=True,
            verbose=True
        )
        
        topics, probabilities = self.model.fit_transform(valid_texts)
        
        self.topic_labels = np.array(topics)
        self.topic_probabilities = probabilities
        
        # Get topic info
        topic_info = self.model.get_topic_info()
        
        return {
            'model_type': 'bertopic',
            'n_topics_found': len(topic_info),
            'n_topics_requested': self.n_topics,
            'n_documents': len(valid_texts),
            'topic_info': topic_info
        }
    
    def fit(self, texts: List[str]) -> Dict[str, Any]:
        """
        Fit the topic model to the texts.
        
        Args:
            texts: List of text documents
            
        Returns:
            Model fitting results
        """
        if not texts or len(texts) == 0:
            logger.warning("No texts provided for topic modeling")
            return {}
        
        logger.info(f"Fitting {self.algorithm} model on {len(texts)} documents")
        
        try:
            if self.algorithm == 'lda':
                processed_texts = self.preprocess_texts(texts)
                return self.fit_lda(processed_texts)
            
            elif self.algorithm == 'nmf':
                processed_texts = self.preprocess_texts(texts)
                return self.fit_nmf(processed_texts)
            
            elif self.algorithm == 'gensim_lda':
                processed_texts = self.preprocess_texts(texts)
                return self.fit_gensim_lda(processed_texts)
            
            elif self.algorithm == 'bertopic':
                return self.fit_bertopic(texts)
            
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        except Exception as e:
            logger.error(f"Error fitting topic model: {e}")
            raise
    
    def get_topic_words(self, n_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top words for each topic.
        
        Args:
            n_words: Number of top words per topic
            
        Returns:
            Dictionary mapping topic_id to list of (word, weight) tuples
        """
        if self.model is None:
            return {}
        
        topic_words = {}
        
        try:
            if self.algorithm in ['lda', 'nmf']:
                feature_names = self.vectorizer.get_feature_names_out()
                
                for topic_idx, topic in enumerate(self.model.components_):
                    top_words_idx = topic.argsort()[-n_words:][::-1]
                    top_words = [(feature_names[i], topic[i]) for i in top_words_idx]
                    topic_words[topic_idx] = top_words
            
            elif self.algorithm == 'gensim_lda':
                for topic_idx in range(self.n_topics):
                    topic_terms = self.model.show_topic(topic_idx, topn=n_words)
                    topic_words[topic_idx] = [(term, prob) for prob, term in topic_terms]
            
            elif self.algorithm == 'bertopic':
                for topic_idx in range(len(self.model.get_topic_info())):
                    if topic_idx == -1:  # Skip outlier topic
                        continue
                    topic_terms = self.model.get_topic(topic_idx)
                    if topic_terms:
                        topic_words[topic_idx] = topic_terms[:n_words]
        
        except Exception as e:
            logger.error(f"Error getting topic words: {e}")
        
        return topic_words
    
    def get_topic_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the topic modeling results.
        
        Returns:
            Dictionary of topic statistics
        """
        if self.topic_labels is None:
            return {}
        
        unique_topics, counts = np.unique(self.topic_labels, return_counts=True)
        
        stats = {
            'n_topics': len(unique_topics),
            'topic_distribution': dict(zip(unique_topics, counts)),
            'largest_topic_size': int(np.max(counts)),
            'smallest_topic_size': int(np.min(counts)),
            'avg_topic_size': float(np.mean(counts)),
            'total_documents': len(self.topic_labels)
        }
        
        return stats
    
    def get_topic_summaries(self, data: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Generate summaries for each topic.
        
        Args:
            data: List of data dictionaries with metadata
            
        Returns:
            Dictionary mapping topic_id to topic summary
        """
        if self.topic_labels is None:
            return {}
        
        topic_words = self.get_topic_words(n_words=10)
        summaries = {}
        
        unique_topics = np.unique(self.topic_labels)
        
        for topic_id in unique_topics:
            if topic_id == -1:  # Skip outlier topic in BERTopic
                continue
            
            # Get documents in this topic
            topic_mask = self.topic_labels == topic_id
            topic_indices = np.where(topic_mask)[0]
            topic_items = [data[i] for i in topic_indices if i < len(data)]
            
            if not topic_items:
                continue
            
            # Calculate topic statistics
            subreddits = [item.get('subreddit', '') for item in topic_items]
            subreddit_counts = Counter(subreddits)
            
            scores = [item.get('score', 0) for item in topic_items if 'score' in item]
            avg_score = np.mean(scores) if scores else 0
            
            # Get top words for this topic
            top_words = topic_words.get(topic_id, [])
            if self.algorithm == 'gensim_lda':
                # For Gensim LDA, tuples are (probability, word)
                word_list = [word for prob, word in top_words[:5]]
            else:
                # For other algorithms, tuples are (word, weight)
                word_list = [word for word, weight in top_words[:5]]
            
            # Get representative documents
            if hasattr(self, 'topic_probabilities') and self.topic_probabilities is not None:
                topic_probs = self.topic_probabilities[topic_mask, topic_id] if topic_id < self.topic_probabilities.shape[1] else []
                if len(topic_probs) > 0:
                    top_doc_indices = np.argsort(topic_probs)[-3:][::-1]
                    representatives = [topic_items[i] for i in top_doc_indices if i < len(topic_items)]
                else:
                    representatives = topic_items[:3]
            else:
                representatives = topic_items[:3]
            
            summary = {
                'topic_id': int(topic_id),
                'size': len(topic_items),
                'top_words': word_list,
                'subreddit_distribution': dict(subreddit_counts),
                'top_subreddits': [subr for subr, count in subreddit_counts.most_common(3)],
                'avg_score': float(avg_score),
                'representatives': representatives[:3]
            }
            
            summaries[topic_id] = summary
        
        logger.info(f"Generated summaries for {len(summaries)} topics")
        return summaries
    
    def find_optimal_topics(self, texts: List[str], topic_range: range = range(5, 31, 5)) -> Dict[str, Any]:
        """
        Find optimal number of topics using coherence or perplexity.
        
        Args:
            texts: List of text documents
            topic_range: Range of topic numbers to test
            
        Returns:
            Dictionary with optimal topic analysis
        """
        logger.info(f"Finding optimal number of topics in range {list(topic_range)}")
        
        processed_texts = self.preprocess_texts(texts)
        scores = []
        topic_counts = []
        
        for n_topics in topic_range:
            try:
                # Create temporary model
                temp_analyzer = TopicAnalyzer(
                    algorithm=self.algorithm,
                    n_topics=n_topics,
                    random_state=self.random_state
                )
                
                result = temp_analyzer.fit(texts)
                
                if self.algorithm == 'gensim_lda':
                    score = result.get('coherence_score', 0)
                elif self.algorithm in ['lda']:
                    score = -result.get('perplexity', float('inf'))  # Lower perplexity is better
                elif self.algorithm == 'nmf':
                    score = -result.get('reconstruction_error', float('inf'))
                else:
                    score = 0
                
                scores.append(score)
                topic_counts.append(n_topics)
                
                logger.info(f"Topics: {n_topics}, Score: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Error testing {n_topics} topics: {e}")
                continue
        
        if not scores:
            return {}
        
        # Find optimal number of topics
        optimal_idx = np.argmax(scores)
        optimal_topics = topic_counts[optimal_idx]
        
        analysis = {
            'topic_counts': topic_counts,
            'scores': scores,
            'optimal_topics': optimal_topics,
            'optimal_score': scores[optimal_idx],
            'metric': 'coherence' if self.algorithm == 'gensim_lda' else 'perplexity'
        }
        
        logger.info(f"Optimal number of topics: {optimal_topics}")
        return analysis
