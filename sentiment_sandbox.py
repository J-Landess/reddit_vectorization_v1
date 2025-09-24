#!/usr/bin/env python3
import os
import sys
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment import VaderSentimentAnalyzer, TransformerSentimentAnalyzer


SAMPLE_POSTS: List[str] = [
    "I absolutely love this new feature! Great job everyone.",
    "This is terrible. The update broke my workflow and I'm frustrated.",
    "It's okay, nothing special but it works as expected.",
    "I'm not sure how I feel about this. Mixed thoughts.",
    "Fantastic performance improvements, the app feels so much faster!",
    "Worst experience ever. Crashed three times in five minutes.",
]


def main() -> int:
    print("\n== Sentiment Sandbox ==\n")

    vader = VaderSentimentAnalyzer()
    transformer = TransformerSentimentAnalyzer()

    print("Testing VADER:")
    for i, text in enumerate(SAMPLE_POSTS, 1):
        label, conf = vader.analyze(text)
        print(f"[{i}] {label:8s}  conf={conf:0.3f}  | {text}")

    print("\nTesting Transformer (distilbert-sst2):")
    for i, text in enumerate(SAMPLE_POSTS, 1):
        label, conf = transformer.analyze(text)
        print(f"[{i}] {label:8s}  conf={conf:0.3f}  | {text}")

    print("\nNote: Sandbox mode does not write to DB or CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


