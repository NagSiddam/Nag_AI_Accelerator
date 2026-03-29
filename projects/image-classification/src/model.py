"""Model construction with transfer learning for image classification."""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int, backbone: str = "resnet18", pretrained: bool = True) -> nn.Module:
    """Load a pretrained backbone and replace the final classification layer.

    Args:
        num_classes: Number of output classes.
        backbone: Torchvision model name ("resnet18" or "resnet50").
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        PyTorch model with a new fully-connected head.
    """
    weights_map = {
        "resnet18": models.ResNet18_Weights.DEFAULT if pretrained else None,
        "resnet50": models.ResNet50_Weights.DEFAULT if pretrained else None,
    }
    if backbone not in weights_map:
        raise ValueError(f"Unsupported backbone '{backbone}'. Choose from: {list(weights_map)}")

    model_fn = getattr(models, backbone)
    model = model_fn(weights=weights_map[backbone])

    # Freeze backbone parameters for initial fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load model weights from a checkpoint file."""
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    return model
