# Image Classification

Transfer-learning template for multi-class image recognition using PyTorch and torchvision pretrained models.

## Features

- Data loading with `ImageFolder` convention (one folder per class)
- Augmentation pipeline for training and validation
- Transfer learning from ResNet-18/50 (easily swappable)
- Training loop with early stopping and learning-rate scheduling
- Model checkpoint saving and loading
- Inference on single images or a directory

## Project Structure

```
image-classification/
├── data/
│   ├── train/          # Training images (one sub-folder per class)
│   └── val/            # Validation images (one sub-folder per class)
├── src/
│   ├── dataset.py      # Dataset and augmentation utilities
│   ├── model.py        # Model construction with transfer learning
│   ├── train.py        # Training entry point
│   └── predict.py      # Inference for new images
├── tests/
│   └── test_dataset.py # Unit tests for dataset utilities
├── requirements.txt
└── README.md
```

## Data Layout

```
data/
├── train/
│   ├── cat/
│   │   ├── img001.jpg
│   │   └── ...
│   └── dog/
│       ├── img001.jpg
│       └── ...
└── val/
    ├── cat/
    └── dog/
```

## Quick Start

```bash
pip install -r requirements.txt

# Train with default settings (ResNet-18, 10 epochs)
python src/train.py --data_dir data --epochs 10 --output_dir models

# Predict a single image
python src/predict.py --model models/best_model.pth --image path/to/image.jpg
```
