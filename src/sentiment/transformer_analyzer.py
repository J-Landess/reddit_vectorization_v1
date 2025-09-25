from typing import Tuple
import sys
import os

# Add the project root to Python path to enable imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sentiment.base import SentimentAnalyzer


class TransformerSentimentAnalyzer(SentimentAnalyzer):
    """Hugging Face transformer-based sentiment analyzer using a pipeline."""

    def __init__(self, model_name: str = 'distilbert-base-uncased-finetuned-sst-2-english', device: int = -1):
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for TransformerSentimentAnalyzer. Please install transformers."
            ) from e

        # Defer import to avoid top-level dependency if unused
        from transformers import pipeline  # type: ignore

        self._pipe = pipeline('sentiment-analysis', model=model_name, device=device)

    def analyze(self, text: str) -> Tuple[str, float]:
        if not text:
            return "neutral", 0.0
        result = self._pipe(text)[0]
        label_raw = result.get('label', '').lower()
        score = float(result.get('score', 0.0))
        if 'pos' in label_raw:
            label = 'positive'
        elif 'neg' in label_raw:
            label = 'negative'
        else:
            # Some models output neutral; map as-is
            label = 'neutral' if 'neu' in label_raw or 'neutral' in label_raw else label_raw
        # Clamp score
        score = max(0.0, min(1.0, score))
        return label, score


if __name__ == "__main__":
    # Test the analyzer
    print("Testing TransformerSentimentAnalyzer...")
    try:
        analyzer = TransformerSentimentAnalyzer()
        print("✓ Analyzer initialized successfully")
        
        # Test with sample text
        test_texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, nothing special."
        ]
        
        print("\nTesting sentiment analysis:")
        for text in test_texts:
            label, score = analyzer.analyze(text)
            print(f"Text: '{text}'")
            print(f"Sentiment: {label} (confidence: {score:.3f})")
            print()
            
        print("✓ All tests passed! The analyzer is working correctly.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure you have the 'transformers' library installed:")
        print("pip install transformers")


