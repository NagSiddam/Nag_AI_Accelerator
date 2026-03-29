"""Training entry point for image classification."""

import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset import get_dataloaders
from model import build_model


def train(
    data_dir: str,
    backbone: str,
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    train_loader, val_loader, class_names = get_dataloaders(data_dir, batch_size=batch_size)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    model = build_model(num_classes=num_classes, backbone=backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.fc.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_val_acc = 0.0
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- Training phase ---
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # --- Validation phase ---
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = correct / len(val_loader.dataset)
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(output_dir, "best_model.pth")
            torch.save(
                {"model_state_dict": model.state_dict(), "class_names": class_names},
                checkpoint_path,
            )
            print(f"  ✓ Saved best model (val_acc={best_val_acc:.4f}) to {checkpoint_path}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument("--data_dir", default="data", help="Root data directory")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", default="models")
    args = parser.parse_args()
    train(args.data_dir, args.backbone, args.epochs, args.batch_size, args.lr, args.output_dir)
