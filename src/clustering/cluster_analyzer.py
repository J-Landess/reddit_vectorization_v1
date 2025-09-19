"""
Clustering analysis for Reddit data using k-means and HDBSCAN.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import hdbscan
from collections import Counter

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """Clustering analysis for Reddit data embeddings."""
    
    def __init__(self, algorithm: str = 'hdbscan', min_cluster_size: int = 5, 
                 min_samples: int = 3, n_clusters: int = 10):
        """
        Initialize the cluster analyzer.
        
        Args:
            algorithm: Clustering algorithm ('hdbscan' or 'kmeans')
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for HDBSCAN
            n_clusters: Number of clusters for k-means
        """
        self.algorithm = algorithm.lower()
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_clusters = n_clusters
        self.model = None
        self.labels_ = None
        self.embeddings_ = None
        
        logger.info(f"Initialized cluster analyzer with {algorithm}")
    
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit the clustering model to the embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Cluster labels
        """
        if embeddings is None or len(embeddings) == 0:
            logger.warning("No embeddings provided for clustering")
            return np.array([])
        
        self.embeddings_ = embeddings
        logger.info(f"Fitting clustering model on {len(embeddings)} embeddings")
        
        try:
            if self.algorithm == 'hdbscan':
                self.model = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric='euclidean'
                )
                self.labels_ = self.model.fit_predict(embeddings)
                
            elif self.algorithm == 'kmeans':
                self.model = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    n_init=10
                )
                self.labels_ = self.model.fit_predict(embeddings)
                
            else:
                raise ValueError(f"Unsupported algorithm: {self.algorithm}")
            
            # Count clusters
            unique_labels = np.unique(self.labels_)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = list(self.labels_).count(-1)
            
            logger.info(f"Clustering completed: {n_clusters} clusters, {n_noise} noise points")
            
            return self.labels_
            
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            raise
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the clustering results.
        
        Returns:
            Dictionary of clustering statistics
        """
        if self.labels_ is None:
            return {}
        
        unique_labels = np.unique(self.labels_)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(self.labels_).count(-1)
        
        # Cluster sizes
        cluster_sizes = Counter(self.labels_)
        cluster_sizes_dict = {str(k): v for k, v in cluster_sizes.items()}
        
        stats = {
            'algorithm': self.algorithm,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'total_points': len(self.labels_),
            'cluster_sizes': cluster_sizes_dict,
            'min_cluster_size': min(cluster_sizes.values()) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes.values()) if cluster_sizes else 0,
            'avg_cluster_size': np.mean(list(cluster_sizes.values())) if cluster_sizes else 0
        }
        
        # Add algorithm-specific parameters
        if self.algorithm == 'hdbscan':
            stats['min_cluster_size_param'] = self.min_cluster_size
            stats['min_samples_param'] = self.min_samples
        elif self.algorithm == 'kmeans':
            stats['n_clusters_param'] = self.n_clusters
        
        return stats
    
    def get_cluster_quality_metrics(self) -> Dict[str, float]:
        """
        Calculate clustering quality metrics.
        
        Returns:
            Dictionary of quality metrics
        """
        if self.labels_ is None or self.embeddings_ is None:
            return {}
        
        # Remove noise points for metric calculation
        valid_mask = self.labels_ != -1
        if not np.any(valid_mask):
            return {}
        
        valid_labels = self.labels_[valid_mask]
        valid_embeddings = self.embeddings_[valid_mask]
        
        if len(np.unique(valid_labels)) < 2:
            return {}
        
        try:
            # Silhouette score
            silhouette = silhouette_score(valid_embeddings, valid_labels)
            
            # Calinski-Harabasz score
            calinski_harabasz = calinski_harabasz_score(valid_embeddings, valid_labels)
            
            metrics = {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz
            }
            
            logger.info(f"Clustering quality metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {}
    
    def get_cluster_representatives(self, data: List[Dict[str, Any]], 
                                  top_k: int = 5) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get representative items for each cluster.
        
        Args:
            data: List of data dictionaries with embeddings
            top_k: Number of representative items per cluster
            
        Returns:
            Dictionary mapping cluster_id to list of representative items
        """
        if self.labels_ is None or not data:
            return {}
        
        representatives = {}
        unique_labels = np.unique(self.labels_)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise
                continue
            
            # Get items in this cluster
            cluster_mask = self.labels_ == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_items = [data[i] for i in cluster_indices]
            
            if not cluster_items:
                continue
            
            # Sort by score (if available) or by word count
            cluster_items.sort(
                key=lambda x: x.get('score', 0) if 'score' in x else x.get('word_count', 0),
                reverse=True
            )
            
            # Take top_k representatives
            representatives[cluster_id] = cluster_items[:top_k]
        
        logger.info(f"Found representatives for {len(representatives)} clusters")
        return representatives
    
    def get_cluster_summaries(self, data: List[Dict[str, Any]], 
                            representatives: Dict[int, List[Dict[str, Any]]]) -> Dict[int, Dict[str, Any]]:
        """
        Generate summaries for each cluster.
        
        Args:
            data: List of data dictionaries
            representatives: Cluster representatives
            
        Returns:
            Dictionary mapping cluster_id to cluster summary
        """
        if self.labels_ is None:
            return {}
        
        summaries = {}
        unique_labels = np.unique(self.labels_)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise
                continue
            
            # Get items in this cluster
            cluster_mask = self.labels_ == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_items = [data[i] for i in cluster_indices]
            
            if not cluster_items:
                continue
            
            # Calculate cluster statistics
            subreddits = [item.get('subreddit', '') for item in cluster_items]
            subreddit_counts = Counter(subreddits)
            
            # Get common words (simple approach)
            all_text = ' '.join([item.get('cleaned_text', '') for item in cluster_items])
            words = all_text.split()
            word_counts = Counter(words)
            common_words = [word for word, count in word_counts.most_common(10) if len(word) > 3]
            
            # Calculate average scores
            scores = [item.get('score', 0) for item in cluster_items if 'score' in item]
            avg_score = np.mean(scores) if scores else 0
            
            summary = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_items),
                'subreddit_distribution': dict(subreddit_counts),
                'top_subreddits': [subr for subr, count in subreddit_counts.most_common(3)],
                'common_words': common_words[:5],
                'avg_score': float(avg_score),
                'representatives': representatives.get(cluster_id, [])[:3]
            }
            
            summaries[cluster_id] = summary
        
        logger.info(f"Generated summaries for {len(summaries)} clusters")
        return summaries
    
    def find_optimal_clusters(self, embeddings: np.ndarray, 
                            max_clusters: int = 20) -> Dict[str, Any]:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.
        
        Args:
            embeddings: Numpy array of embeddings
            max_clusters: Maximum number of clusters to test
            
        Returns:
            Dictionary with optimal cluster analysis
        """
        if len(embeddings) < 2:
            return {}
        
        logger.info(f"Finding optimal number of clusters (max: {max_clusters})")
        
        # Test different numbers of clusters
        cluster_range = range(2, min(max_clusters + 1, len(embeddings) // 2))
        inertias = []
        silhouette_scores = []
        
        for n_clusters in cluster_range:
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(embeddings)
                
                inertias.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(embeddings, labels))
                
            except Exception as e:
                logger.warning(f"Error testing {n_clusters} clusters: {e}")
                continue
        
        if not inertias:
            return {}
        
        # Find optimal number of clusters
        optimal_k = cluster_range[np.argmax(silhouette_scores)]
        
        analysis = {
            'cluster_range': list(cluster_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k,
            'max_silhouette': max(silhouette_scores)
        }
        
        logger.info(f"Optimal number of clusters: {optimal_k}")
        return analysis
