#!/usr/bin/env python3
"""
HIE-GAN Phase 1 Training Entry Point
Handles all CLI argument parsing and delegates to trainer
"""
import argparse
import sys
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="HIE-GAN Phase 1 Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ===== Training Modes =====
    mode_group = parser.add_argument_group("Training Modes")
    mode_group.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "resume", "scratch"],
        help="'train': auto-resume if checkpoint exists | "
             "'resume': force resume from checkpoint | "
             "'scratch': start fresh training"
    )

    # ===== Config Files =====
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config-dir", type=str, default="src/configs",
        help="Directory containing YAML config files"
    )
    config_group.add_argument(
        "--dataset-config", type=str, default=None,
        help="Override dataset config file path"
    )
    config_group.add_argument(
        "--model-config", type=str, default=None,
        help="Override model config file path"
    )
    config_group.add_argument(
        "--train-config", type=str, default=None,
        help="Override train config file path"
    )

    # ===== Training Hyperparameters =====
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs from config"
    )
    train_group.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config"
    )
    train_group.add_argument(
        "--lr", "--learning-rate", type=float, default=None,
        dest="lr", help="Override learning rate from config"
    )
    train_group.add_argument(
        "--weight-decay", type=float, default=None,
        help="Weight decay for optimizer"
    )
    train_group.add_argument(
        "--grad-clip", type=float, default=None,
        help="Gradient clipping max norm (disabled if not set)"
    )

    # ===== Device and Performance =====
    device_group = parser.add_argument_group("Device and Performance")
    device_group.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cuda', 'cpu', 'cuda:0', 'cuda:1', etc."
    )
    device_group.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of DataLoader workers"
    )
    device_group.add_argument(
        "--pin-memory", action="store_true", default=None,
        help="Enable pin_memory for DataLoader"
    )
    device_group.add_argument(
        "--mixed-precision", action="store_true",
        help="Enable mixed precision training (FP16)"
    )
    device_group.add_argument(
        "--compile", action="store_true",
        help="Use torch.compile for model optimization (PyTorch 2.0+)"
    )

    # ===== Checkpointing =====
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--checkpoint", "--resume-from", type=str, default=None,
        dest="checkpoint", help="Specific checkpoint path to load"
    )
    ckpt_group.add_argument(
        "--save-every", type=int, default=None,
        help="Save checkpoint every N epochs"
    )
    ckpt_group.add_argument(
        "--keep-last", type=int, default=3,
        help="Keep only last N checkpoints"
    )
    ckpt_group.add_argument(
        "--no-save", action="store_true",
        help="Disable checkpoint saving (for testing)"
    )

    # ===== Logging =====
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--exp-name", type=str, default=None,
        help="Experiment name for organizing outputs"
    )
    log_group.add_argument(
        "--log-dir", type=str, default=None,
        help="Override log directory"
    )
    log_group.add_argument(
        "--log-every", type=int, default=None,
        help="Log every N steps"
    )
    log_group.add_argument(
        "--no-tqdm", action="store_true",
        help="Disable tqdm progress bars"
    )
    log_group.add_argument(
        "--quiet", action="store_true",
        help="Minimal console output"
    )

    # ===== Data Options =====
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument(
        "--data-root", type=str, default=None,
        help="Override dataset root directory"
    )
    data_group.add_argument(
        "--num-samples", type=int, default=None,
        help="Limit dataset to N samples (for testing)"
    )

    # ===== Debugging and Testing =====
    debug_group = parser.add_argument_group("Debugging and Testing")
    debug_group.add_argument(
        "--debug", action="store_true",
        help="Debug mode: small dataset, verbose logging, 5 batches per epoch"
    )
    debug_group.add_argument(
        "--dry-run", action="store_true",
        help="Dry run: initialize everything but don't train"
    )
    debug_group.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility"
    )
    debug_group.add_argument(
        "--deterministic", action="store_true",
        help="Enable deterministic mode (slower but reproducible)"
    )

    # ===== Validation =====
    val_group = parser.add_argument_group("Validation")
    val_group.add_argument(
        "--val-split", type=float, default=None,
        help="Validation split ratio (0.0-1.0)"
    )
    val_group.add_argument(
        "--val-every", type=int, default=None,
        help="Run validation every N epochs"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Import here to avoid issues
    from trainer.train import Trainer

    # Parse arguments
    args = parse_args()

    # Initialize trainer with args
    trainer = Trainer(args)

    # Start training
    trainer.run()
