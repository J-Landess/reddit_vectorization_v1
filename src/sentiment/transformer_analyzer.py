from typing import Tuple

from .base import SentimentAnalyzer


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


