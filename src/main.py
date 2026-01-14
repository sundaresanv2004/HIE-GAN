#!/usr/bin/env python3
"""
HIE-GAN Phase 1 Training Entry Point
Handles all CLI argument parsing and delegates to trainer
"""
import argparse
import sys
import torch
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
        choices=["train", "resume", "scratch", "generate", "plot"],
        help="Execution mode: train/resume/scratch (training), generate (inference), plot (visualization)"
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
        dest="checkpoint", help="Specific checkpoint path (file or output directory) to load"
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
        help="Experiment name for organizing output"
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
        help="Limit dataset to N samples (for testing/generation)"
    )
    data_group.add_argument(
        "--inspect-data", action="store_true",
        help="Inspect sample data during loading"
    )
    data_group.add_argument(
        "--image", type=str, default=None,
        help="Path to single image for 'generate' mode"
    )
    data_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Explicit output directory for generation results"
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

    # ===== Post-Training Automation =====
    post_group = parser.add_argument_group("Post-Training Automation")
    post_group.add_argument(
        "--generate-graph", action="store_true", default=True,
        help="Generate training graphs (Loss vs Epoch) after training"
    )
    post_group.add_argument(
        "--generate-model", action="store_true", default=True,
        help="Generate 3D models for classes after training"
    )

    return parser.parse_args()


if __name__ == "__main__":
    from trainer.train import Trainer
    from utils.plotter import plot_training_logs
    from inference.generate import generate_batch, generate_single, load_model

    args = parse_args()

    if args.mode == "plot":
        # Plotting Mode
        # Determine CSV path
        if args.log_dir:
            path = Path(args.log_dir)
            if path.is_file():
                if path.suffix == ".log":
                     # Likely user pointed to text log, try to find csv sibling
                     csv_candidate = path.with_suffix(".csv")
                     if csv_candidate.exists():
                         print(f"ℹ Auto-correcting {path.name} -> {csv_candidate.name}")
                         csv_path = csv_candidate
                     else:
                         csv_path = path # Let it fail or user named it weirdly
                else:
                    csv_path = path
            elif path.is_dir():
                # Look for common csv names
                candidates = ["phase1_train.csv", "training_log.csv"]
                found = False
                for c in candidates:
                    p = path / c
                    if p.exists():
                        csv_path = p
                        found = True
                        break
                if not found:
                    print(f"❌ Could not find CSV log in {path}")
                    print(f"   Checked: {candidates}")
                    sys.exit(1)
            else:
                print(f"❌ Path not found: {path}")
                sys.exit(1)
        else:
            # Default location
            csv_path = Path("output/latest/phase1_train.csv")
            if not csv_path.exists():
                 csv_path = Path("output/latest/training_log.csv")
        
        if csv_path.exists():
             print(f"Plotting log: {csv_path}")
             plot_training_logs(csv_path)
        else:
             print(f"❌ File not found: {csv_path}")
             print("Usage: python src/main.py --mode plot --log-dir path/to/phase1_train.csv")

    elif args.mode == "generate":
        # Generation Mode
        if not args.checkpoint:
            print("❌ --checkpoint is required for generation--")
            sys.exit(1)
            
        print(f"Loading model from {args.checkpoint}...")
        
        # Setup device for generation
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)

        # Load model components
        # We need config dir. 
        # Note: generate.py's load_model expects these params
        encoder, explicit, model_cfg = load_model(args.config_dir, args.checkpoint, device)
        
        # Determine base directory from checkpoint path
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.parent.name == "checkpoints":
             base_exp_dir = ckpt_path.parent.parent
        else:
             base_exp_dir = ckpt_path.parent

        if args.output_dir:
            out_dir = Path(args.output_dir)
        elif args.exp_name:
            # Treat exp_name as subdirectory name within the experiment folder
            out_dir = base_exp_dir / args.exp_name
        else:
            # Default
            out_dir = base_exp_dir / "generated_samples"
        
        print(f"Output directory: {out_dir}")
        
        if args.image:
            # Single Image
            print(f"Generating single sample from {args.image}...")
            path = generate_single(args.image, out_dir, encoder, explicit, device, model_cfg)
            print(f"✓ Saved: {path}")
        else:
            # Batch
            print(f"Generating batch samples...")
            generate_batch(args.config_dir, out_dir, encoder, explicit, device, model_cfg, num_samples=args.num_samples or 20)

    else:
        # Training Modes (train, resume, scratch)
        trainer = Trainer(args)
        trainer.run()
