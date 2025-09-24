from abc import ABC, abstractmethod
from typing import Tuple


class SentimentAnalyzer(ABC):
    """Abstract base class for sentiment analyzers."""

    @abstractmethod
    def analyze(self, text: str) -> Tuple[str, float]:
        """
        Analyze text and return a tuple of (sentiment_label, confidence_score).

        sentiment_label: one of {"positive", "negative", "neutral"}
        confidence_score: float in [0, 1]
        """
        raise NotImplementedError


