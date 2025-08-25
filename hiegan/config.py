from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class Config:
    PROJECT_ROOT: Path = Path(__file__).parent.parent

    # Paths
    DATASET_ROOT: str = str(PROJECT_ROOT / "dataset/ShapeNetCore.v2")
    CHECKPOINT_DIR: str = str(PROJECT_ROOT / "outputs/checkpoints")
    RENDERS_DIR: str = str(PROJECT_ROOT / "outputs/renders")

    # Data
    IMAGE_SIZE: int = 224
    MULTI_VIEW: bool = False  # False = single-view (random pick per object)
    VIEWS_PER_ITEM: int = 1

    # Training
    BATCH_SIZE: int = 4
    NUM_WORKERS: int = 0
    EPOCHS: int = 100
    LR: float = 1e-4
    WEIGHT_DECAY: float = 1e-4
    LATENT_DIM: int = 512
    SEED: int = 42

    # Model Architecture
    HIDDEN_DIM: int = 512
    NUM_ATTENTION_HEADS: int = 8
    NUM_TRANSFORMER_LAYERS: int = 6

    # Loss weights
    CHAMFER_WEIGHT: float = 1.0
    ADVERSARIAL_WEIGHT: float = 0.1
    IMPLICIT_WEIGHT: float = 0.5
    EXPLICIT_WEIGHT: float = 0.5

    # Device
    DEVICE: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    ))
