"""Evaluation utilities for the text classification pipeline."""

import argparse
import joblib

from sklearn.metrics import classification_report

from preprocess import load_dataset, preprocess


def evaluate(model_path: str, data_path: str, text_col: str, label_col: str) -> None:
    pipeline = joblib.load(model_path)
    df = load_dataset(data_path, text_col, label_col)
    df["processed"] = df[text_col].apply(preprocess)

    y_pred = pipeline.predict(df["processed"])
    print(classification_report(df[label_col], y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a text classifier")
    parser.add_argument("--model", required=True, help="Path to saved model (.pkl)")
    parser.add_argument("--data", required=True, help="Path to evaluation CSV")
    parser.add_argument("--text_col", default="text", help="Column name for text")
    parser.add_argument("--label_col", default="label", help="Column name for label")
    args = parser.parse_args()
    evaluate(args.model, args.data, args.text_col, args.label_col)
