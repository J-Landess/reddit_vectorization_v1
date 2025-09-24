#!/usr/bin/env python3
"""
Evaluate VADER vs Transformer sentiment analyzers.

- If datasets package is available, use SST-2 validation split.
- Otherwise, fall back to a small built-in labeled sample.
"""
import os
import sys
import time
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sentiment import VaderSentimentAnalyzer, TransformerSentimentAnalyzer


def load_sst2() -> Tuple[List[str], List[str]]:
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset('glue', 'sst2', split='validation')
        texts = [x['sentence'] for x in ds]
        labels = ['positive' if int(x['label']) == 1 else 'negative' for x in ds]
        return texts, labels
    except Exception:
        # Fallback tiny labeled set
        data = [
            ("I loved the movie, it was fantastic!", 'positive'),
            ("Absolutely terrible film, waste of time.", 'negative'),
            ("The plot was engaging and the actors were great.", 'positive'),
            ("I didn't like it at all.", 'negative'),
            ("Not bad, but not great either.", 'negative'),  # force binary for SST-2 style
            ("One of the best experiences this year!", 'positive'),
        ]
        texts, labels = zip(*data)
        return list(texts), list(labels)


def evaluate(name: str, analyzer, texts: List[str], true_labels: List[str]) -> None:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    start = time.time()
    preds: List[str] = []
    for t in texts:
        label, _ = analyzer.analyze(t)
        # Map neutral to negative for SST-2-style binary
        if label == 'neutral':
            label = 'negative'
        preds.append(label)
    elapsed = time.time() - start

    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='binary', pos_label='positive')
    speed = len(texts) / max(1e-6, elapsed)

    print(f"\n{name} Results:")
    print(f"  Accuracy : {acc:0.4f}")
    print(f"  Precision: {precision:0.4f}")
    print(f"  Recall   : {recall:0.4f}")
    print(f"  F1       : {f1:0.4f}")
    print(f"  Runtime  : {elapsed:0.2f}s for {len(texts)} samples ({speed:0.1f} samples/s)")


def main() -> int:
    print("\n== Evaluating Sentiment Analyzers ==\n")
    texts, labels = load_sst2()
    print(f"Loaded {len(texts)} samples.")

    vader = VaderSentimentAnalyzer()
    transformer = TransformerSentimentAnalyzer()

    evaluate("VADER", vader, texts, labels)
    evaluate("Transformer (distilbert-sst2)", transformer, texts, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


