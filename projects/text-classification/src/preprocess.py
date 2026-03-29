"""Text preprocessing utilities for the text classification pipeline."""

import re
import string
from typing import List

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data on first use
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split cleaned text into word tokens."""
    return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Filter out common English stop words."""
    return [t for t in tokens if t not in _STOP_WORDS]


def stem_tokens(tokens: List[str]) -> List[str]:
    """Apply Porter stemming to each token."""
    return [_STEMMER.stem(t) for t in tokens]


def preprocess(text: str, stem: bool = False) -> str:
    """Full preprocessing pipeline: clean → tokenize → remove stopwords → (stem) → rejoin."""
    tokens = tokenize(clean_text(text))
    tokens = remove_stopwords(tokens)
    if stem:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


def build_tfidf_vectorizer(max_features: int = 10_000) -> TfidfVectorizer:
    """Return a configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def load_dataset(csv_path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """Load and validate a CSV dataset."""
    df = pd.read_csv(csv_path)
    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {csv_path}")
    df = df.dropna(subset=[text_col, label_col])
    return df
