# Medical Category Classification Guide

This guide explains the new medical category classification feature added to the Reddit analysis pipeline.

## 🎯 Overview

The classification system automatically categorizes Reddit posts and comments into four medical-related categories:

- **Medical Insurance**: Health insurance plans, coverage, premiums, deductibles
- **Medical Provider**: Doctors, nurses, clinics, hospitals, medical care
- **Medical Broker**: Insurance brokers, agents, consultants, plan comparisons
- **Employer**: Company benefits, HR, workplace health, employee coverage

## 🏗️ Architecture

The classification system uses a **hybrid approach** combining multiple methods:

### 1. Rule-Based Classifier
- Uses medical-specific keywords and patterns
- High accuracy for obvious cases
- Fast and interpretable
- No training data required

### 2. Machine Learning Classifier
- Uses your existing embeddings as features
- Supports Random Forest, Logistic Regression, and SVM
- Can be trained on your data
- Handles ambiguous cases better

### 3. Hybrid Classifier (Recommended)
- Combines rule-based and ML approaches
- Uses rule-based for high-confidence cases
- Falls back to ML for ambiguous cases
- Can use ensemble voting for maximum accuracy

## 🚀 Quick Start

### Basic Usage

```python
from src.pipeline import RedditAnalysisPipeline

# Initialize pipeline with classification enabled
pipeline = RedditAnalysisPipeline(analyzer_type='vader')
pipeline.setup_components()

# Run the full pipeline (includes classification)
pipeline.run_full_analysis()
```

### Test the Classification

```bash
# Run the example script
python example_classification.py
```

## ⚙️ Configuration

Configure classification behavior using environment variables or by editing `config.py`:

```python
CLASSIFICATION_CONFIG = {
    'enabled': True,                    # Enable/disable classification
    'classifier_type': 'hybrid',       # 'rule_based', 'ml', 'hybrid'
    'ml_model_type': 'random_forest',  # 'random_forest', 'logistic_regression', 'svm'
    'rule_confidence_threshold': 0.7,  # Threshold for rule-based classification
    'ensemble_mode': False,            # Use ensemble voting
    'auto_train_ml': False            # Auto-train ML component
}
```

### Environment Variables

```bash
export CLASSIFICATION_ENABLED=true
export CLASSIFIER_TYPE=hybrid
export ML_MODEL_TYPE=random_forest
export RULE_CONFIDENCE_THRESHOLD=0.7
export ENSEMBLE_MODE=false
export AUTO_TRAIN_ML=false
```

## 📊 Database Schema

The classification system adds three new columns to your database:

- `category`: The predicted medical category
- `category_confidence`: Confidence score (0-1)
- `category_probabilities`: JSON with probabilities for all categories

## 🔧 Advanced Usage

### Training the ML Component

```python
# Train on existing data
pipeline.train_classification_model()

# Train on specific data
pipeline.train_classification_model(your_data)
```

### Using Individual Classifiers

```python
from src.classification import RuleBasedClassifier, MLClassifier, HybridClassifier

# Rule-based only
rule_classifier = RuleBasedClassifier()
category, confidence, probs = rule_classifier.classify("I need health insurance")

# ML only (requires training)
ml_classifier = MLClassifier(model_type='random_forest')
ml_classifier.train(embeddings, labels)
category, confidence, probs = ml_classifier.classify("I need health insurance", embedding)

# Hybrid (recommended)
hybrid_classifier = HybridClassifier()
category, confidence, probs = hybrid_classifier.classify("I need health insurance", embedding)
```

### Customizing Keywords

You can extend the rule-based classifier by modifying the keyword patterns in `src/classification/rule_based_classifier.py`:

```python
# Add new keywords for a category
self.insurance_keywords['exact'].append('new_keyword')
```

## 📈 Analysis and Visualization

The system includes comprehensive analysis tools:

### Classification Analysis

```python
from src.analysis.classification_analyzer import ClassificationAnalyzer

analyzer = ClassificationAnalyzer()
results = analyzer.analyze_classification_results(data)
visualizations = analyzer.create_classification_visualizations(data)
report = analyzer.export_classification_report(results)
```

### Generated Visualizations

1. **Category Distribution Pie Chart**: Shows overall category distribution
2. **Confidence Box Plot**: Confidence scores by category
3. **Subreddit Heatmap**: Category distribution by subreddit
4. **Confidence Histogram**: Overall confidence distribution
5. **Temporal Analysis**: Category trends over time

## 🎯 Performance Tips

### For High Accuracy
- Use the hybrid classifier with ensemble mode
- Train the ML component on your specific data
- Adjust rule confidence threshold based on your data

### For Speed
- Use rule-based classifier only
- Increase rule confidence threshold
- Disable classification for real-time processing

### For Interpretability
- Use rule-based classifier
- Check keyword matches for explanations
- Use the classification breakdown feature

## 🔍 Debugging and Monitoring

### Check Classification Results

```python
# Get detailed breakdown
breakdown = hybrid_classifier.get_classification_breakdown(text, embedding)
print(f"Method used: {breakdown['method_used']}")
print(f"Rule-based: {breakdown['rule_based']['category']}")
print(f"ML-based: {breakdown['ml_based']['category']}")
```

### Monitor Performance

```python
# Get feature importance (for tree-based models)
importance = ml_classifier.get_feature_importance()

# Get training metrics
metrics = pipeline.train_classification_model()
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

## 📝 Example Outputs

### Classification Results

```json
{
  "category": "medical_insurance",
  "category_confidence": 0.85,
  "category_probabilities": {
    "medical_insurance": 0.85,
    "medical_provider": 0.08,
    "medical_broker": 0.05,
    "employer": 0.02
  }
}
```

### Analysis Report

The system generates detailed Markdown reports with:
- Summary statistics
- Category distribution tables
- Confidence analysis
- Subreddit breakdowns
- Performance metrics

## 🚨 Troubleshooting

### Common Issues

1. **Low Classification Accuracy**
   - Train the ML component on your data
   - Adjust rule confidence threshold
   - Check keyword patterns

2. **Slow Performance**
   - Use rule-based classifier only
   - Reduce batch sizes
   - Disable classification for testing

3. **Memory Issues**
   - Reduce embedding batch size
   - Use smaller ML models
   - Process data in smaller chunks

### Getting Help

- Check the logs in `logs/reddit_analysis.log`
- Run `python example_classification.py` to test
- Use the classification breakdown for debugging

## 🔮 Future Enhancements

Planned improvements include:

- **Transformer-based Classification**: Fine-tuned models for medical text
- **Active Learning**: Improve models with user feedback
- **Multi-label Classification**: Support multiple categories per item
- **Custom Categories**: User-defined classification categories
- **Real-time Classification**: Stream processing capabilities

## 📚 API Reference

### ClassificationAnalyzer (Base Class)
- `classify(text, embedding)`: Classify a single text
- `classify_batch(texts, embeddings)`: Classify multiple texts
- `get_category_label(category)`: Get human-readable label

### RuleBasedClassifier
- `classify(text, embedding)`: Rule-based classification
- `get_keyword_matches(text, category)`: Get matched keywords

### MLClassifier
- `classify(text, embedding)`: ML-based classification
- `train(embeddings, labels)`: Train the model
- `get_feature_importance()`: Get feature importance

### HybridClassifier
- `classify(text, embedding)`: Hybrid classification
- `train_ml_component(embeddings, labels)`: Train ML component
- `get_classification_breakdown(text, embedding)`: Get detailed breakdown

### ClassificationAnalyzer (Analysis)
- `analyze_classification_results(data)`: Analyze results
- `create_classification_visualizations(data)`: Create visualizations
- `export_classification_report(results)`: Export report
