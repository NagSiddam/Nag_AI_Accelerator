"""Unit tests for text preprocessing utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import clean_text, remove_stopwords, stem_tokens, tokenize, preprocess


def test_clean_text_lowercases():
    assert clean_text("Hello World") == "hello world"


def test_clean_text_removes_digits():
    assert "1" not in clean_text("abc 123 def")


def test_clean_text_removes_punctuation():
    assert "!" not in clean_text("wow!")


def test_tokenize_splits_on_whitespace():
    assert tokenize("hello world") == ["hello", "world"]


def test_remove_stopwords_filters_common_words():
    tokens = ["this", "is", "a", "great", "product"]
    result = remove_stopwords(tokens)
    assert "this" not in result
    assert "great" in result


def test_stem_tokens_reduces_words():
    tokens = ["running", "runs"]
    stemmed = stem_tokens(tokens)
    # "running" and "runs" should stem to the same root ("run")
    assert len(set(stemmed)) == 1


def test_preprocess_returns_string():
    result = preprocess("The quick brown fox jumps over the lazy dog!")
    assert isinstance(result, str)
    assert len(result) > 0


def test_preprocess_stem_option():
    result_default = preprocess("running quickly", stem=False)
    result_stemmed = preprocess("running quickly", stem=True)
    # Stemming should produce shorter or equal length output
    assert len(result_stemmed) <= len(result_default)
