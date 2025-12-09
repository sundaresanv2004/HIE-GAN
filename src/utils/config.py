import os
import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_configs():
    dataset_cfg = load_yaml("src/configs/dataset.yaml")
    model_cfg = load_yaml("src/configs/model.yaml")
    train_cfg = load_yaml("src/configs/train.yaml")

    # Ensure directories exist
    os.makedirs(train_cfg["logging"]["log_dir"], exist_ok=True)
    os.makedirs(train_cfg["checkpoints"]["dir"], exist_ok=True)

    return dataset_cfg, model_cfg, train_cfg
