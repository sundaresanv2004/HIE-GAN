import os
import yaml
from pathlib import Path


def load_yaml(path):
    """Load YAML config file"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_configs(config_dir="src/configs"):
    """Load all configuration files"""
    config_dir = Path(config_dir)

    dataset_cfg = load_yaml(config_dir / "dataset.yaml")
    model_cfg = load_yaml(config_dir / "model.yaml")
    train_cfg = load_yaml(config_dir / "train.yaml")

    # Ensure directories exist
    os.makedirs(train_cfg["logging"]["log_dir"], exist_ok=True)
    os.makedirs(train_cfg["checkpoints"]["dir"], exist_ok=True)

    return dataset_cfg, model_cfg, train_cfg
