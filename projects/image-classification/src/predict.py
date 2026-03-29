"""Inference script for image classification."""

import argparse

import torch
from PIL import Image

from dataset import VAL_TRANSFORMS
from model import build_model, load_checkpoint


def predict(model_path: str, image_path: str) -> str:
    checkpoint = torch.load(model_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    num_classes = len(class_names)

    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(image_path).convert("RGB")
    tensor = VAL_TRANSFORMS(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    label = class_names[pred_idx]
    confidence = probs[pred_idx].item()
    print(f"Prediction: {label}  (confidence: {confidence:.2%})")
    return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an image")
    parser.add_argument("--model", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    predict(args.model, args.image)
