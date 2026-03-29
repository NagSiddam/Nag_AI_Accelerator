"""Unit tests for image classification dataset utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from torchvision import transforms

from dataset import TRAIN_TRANSFORMS, VAL_TRANSFORMS


def test_train_transforms_output_shape():
    """Train transforms should produce a (3, 224, 224) tensor."""
    from PIL import Image
    import numpy as np

    img = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype="uint8"))
    tensor = TRAIN_TRANSFORMS(img)
    assert tensor.shape == (3, 224, 224)


def test_val_transforms_output_shape():
    """Val transforms should produce a (3, 224, 224) tensor."""
    from PIL import Image
    import numpy as np

    img = Image.fromarray(np.random.randint(0, 256, (300, 400, 3), dtype="uint8"))
    tensor = VAL_TRANSFORMS(img)
    assert tensor.shape == (3, 224, 224)


def test_val_transforms_normalized():
    """Normalized output should not be in [0, 255] range."""
    from PIL import Image
    import numpy as np

    img = Image.fromarray(np.ones((300, 300, 3), dtype="uint8") * 255)
    tensor = VAL_TRANSFORMS(img)
    assert tensor.min() < 5.0  # normalized values are much smaller than 255
