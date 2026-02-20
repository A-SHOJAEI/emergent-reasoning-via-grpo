"""Configuration loading and environment setup."""

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs(config: dict) -> None:
    """Create output directories."""
    for key in ["output_dir", "results_dir", "logs_dir"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
