"""Shared helper utilities for all Nag AI Accelerator projects."""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across numpy, random, and (optionally) torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(data: Dict[str, Any], path: str) -> None:
    """Serialize a dictionary to a JSON file, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
