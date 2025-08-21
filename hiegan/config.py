from dataclasses import dataclass, field
import torch


@dataclass
class Config:
    # Paths
    DATASET_ROOT: str = "../data/ShapeNetCore.v2"
    CHECKPOINT_DIR: str = "../outputs/checkpoints"
    RENDERS_DIR: str = "../outputs/renders"

    # Data
    IMAGE_SIZE: int = 224
    MULTI_VIEW: bool = False  # False = single-view (random pick per object)
    VIEWS_PER_ITEM: int = 1

    # Training
    BATCH_SIZE: int = 2
    NUM_WORKERS: int = 2
    EPOCHS: int = 100
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    LATENT_DIM: int = 512
    SEED: int = 42

    # Device
    DEVICE: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    ))

