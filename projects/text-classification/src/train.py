"""Model training entry point for text classification."""

import argparse
import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_tfidf_vectorizer, load_dataset, preprocess

_CLASSIFIERS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "naive_bayes": MultinomialNB(),
}


def build_pipeline(classifier: str = "logistic_regression") -> Pipeline:
    """Construct a TF-IDF + classifier pipeline.

    Args:
        classifier: One of "logistic_regression" or "naive_bayes".
    """
    if classifier not in _CLASSIFIERS:
        raise ValueError(f"Unknown classifier '{classifier}'. Choose from: {list(_CLASSIFIERS)}")
    return Pipeline(
        [
            ("tfidf", build_tfidf_vectorizer()),
            ("clf", _CLASSIFIERS[classifier]),
        ]
    )


def train(data_path: str, text_col: str, label_col: str, output_dir: str, classifier: str) -> None:
    df = load_dataset(data_path, text_col, label_col)
    df["processed"] = df[text_col].apply(preprocess)

    X_train, X_val, y_train, y_val = train_test_split(
        df["processed"], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )

    pipeline = build_pipeline(classifier)
    pipeline.fit(X_train, y_train)

    val_acc = pipeline.score(X_val, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "classifier.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classifier")
    parser.add_argument("--data", required=True, help="Path to training CSV")
    parser.add_argument("--text_col", default="text", help="Column name for text")
    parser.add_argument("--label_col", default="label", help="Column name for label")
    parser.add_argument("--output_dir", default="models", help="Directory to save model")
    parser.add_argument(
        "--classifier",
        default="logistic_regression",
        choices=list(_CLASSIFIERS),
        help="Classifier to use",
    )
    args = parser.parse_args()
    train(args.data, args.text_col, args.label_col, args.output_dir, args.classifier)
