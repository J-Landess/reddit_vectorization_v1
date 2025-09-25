"""
Sentiment analysis package providing a common interface and implementations.
"""
import sys
import os

# Add the project root to Python path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sentiment.base import SentimentAnalyzer
from src.sentiment.vader_analyzer import VaderSentimentAnalyzer
from src.sentiment.transformer_analyzer import TransformerSentimentAnalyzer

__all__ = [
    "SentimentAnalyzer",
    "VaderSentimentAnalyzer",
    "TransformerSentimentAnalyzer",
]


