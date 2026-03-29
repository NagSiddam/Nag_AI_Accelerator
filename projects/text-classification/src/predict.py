"""Inference script for the text classification pipeline."""

import argparse
import joblib

from preprocess import preprocess


def predict(model_path: str, text: str) -> str:
    pipeline = joblib.load(model_path)
    processed = preprocess(text)
    label = pipeline.predict([processed])[0]
    proba = pipeline.predict_proba([processed])[0].max()
    print(f"Prediction: {label}  (confidence: {proba:.2%})")
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a text classifier")
    parser.add_argument("--model", required=True, help="Path to saved model (.pkl)")
    parser.add_argument("--text", required=True, help="Input text to classify")
    args = parser.parse_args()
    predict(args.model, args.text)
