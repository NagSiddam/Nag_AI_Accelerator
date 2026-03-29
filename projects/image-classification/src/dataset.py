"""Dataset and data augmentation utilities for image classification."""

from pathlib import Path
from typing import Tuple

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ImageNet normalization statistics
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

TRAIN_TRANSFORMS = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)

VAL_TRANSFORMS = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]
)


def get_dataloaders(
    data_dir: str, batch_size: int = 32, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, list]:
    """Create train and validation DataLoaders from an ImageFolder directory layout."""
    data_path = Path(data_dir)

    train_dataset = datasets.ImageFolder(data_path / "train", transform=TRAIN_TRANSFORMS)
    val_dataset = datasets.ImageFolder(data_path / "val", transform=VAL_TRANSFORMS)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    class_names = train_dataset.classes
    return train_loader, val_loader, class_names
