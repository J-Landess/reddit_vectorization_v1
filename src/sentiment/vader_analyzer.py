from typing import Tuple

from .base import SentimentAnalyzer


class VaderSentimentAnalyzer(SentimentAnalyzer):
    """VADER sentiment analyzer using nltk's SentimentIntensityAnalyzer."""

    def __init__(self):
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "nltk is required for VADER analyzer. Please install nltk and download vader_lexicon."
            ) from e

        # Lazy ensure resource
        import nltk  # type: ignore
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)

        self._sia = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Tuple[str, float]:
        if not text:
            return "neutral", 0.0
        scores = self._sia.polarity_scores(text)
        compound = scores.get('compound', 0.0)
        # Map compound to label and confidence
        if compound >= 0.05:
            label = "positive"
            confidence = compound
        elif compound <= -0.05:
            label = "negative"
            confidence = -compound
        else:
            label = "neutral"
            confidence = 1.0 - abs(compound)
        # Normalize confidence to [0,1]
        confidence = max(0.0, min(1.0, confidence))
        return label, confidence


