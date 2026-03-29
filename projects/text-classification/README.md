# Text Classification

An end-to-end NLP pipeline for multi-class text classification tasks such as sentiment analysis, spam detection, and topic labeling.

## Features

- Data loading and preprocessing (tokenization, stopword removal, TF-IDF)
- Classical ML baselines: Logistic Regression and Naive Bayes
- Evaluation with accuracy, precision, recall, and F1 scores
- Inference script for single-text and batch predictions

## Project Structure

```
text-classification/
├── data/                   # Place raw datasets here (CSV/JSON)
├── src/
│   ├── preprocess.py       # Text cleaning and feature extraction
│   ├── train.py            # Model training entry point
│   ├── evaluate.py         # Evaluation utilities
│   └── predict.py          # Inference for new inputs
├── tests/
│   └── test_preprocess.py  # Unit tests for preprocessing
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Train a Logistic Regression model (default)
python src/train.py --data data/dataset.csv --label_col label --text_col text

# Train a Naive Bayes model
python src/train.py --data data/dataset.csv --label_col label --text_col text --classifier naive_bayes

# Evaluate on a held-out test set
python src/evaluate.py --model models/classifier.pkl --data data/test.csv

# Run inference
python src/predict.py --model models/classifier.pkl --text "This is a great product!"
```

## Dataset Format

The training script expects a CSV file with at least two columns:

| text | label |
|---|---|
| I love this product | positive |
| Terrible experience | negative |

## Metrics

After training, the following metrics are reported on the validation set:

- **Accuracy**
- **Macro F1**
- **Classification report** (per-class precision/recall/F1)
