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
        choices=["train", "test", "resume", "scratch", "generate", "plot", "dataset-check"],
        help="Execution mode: train/resume/scratch (training), test (evaluation), generate (inference), plot (visualization), dataset-check (dataset validation)"
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
        "--output-root", type=str, default="output",
        help="Base directory for experiment outputs (default: output)"
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
    data_group.add_argument(
        "--split", type=str, default=None,
        choices=["train", "val", "test", "all"],
        help="Dataset split to check (for dataset-check mode). 'all' checks all splits"
    )
    data_group.add_argument(
        "--detailed", action="store_true",
        help="Show detailed dataset statistics (for dataset-check mode)"
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
    val_group.add_argument(
        "--no-validation", action="store_true",
        help="Disable validation during training"
    )

    # ===== Post-Training Automation =====
    post_group = parser.add_argument_group("Post-Training Automation")
    post_group.add_argument(
        "--no-plot", action="store_true",
        help="Disable automatic graph plotting after training"
    )
    post_group.add_argument(
        "--no-test", action="store_true",
        help="Skip automatic testing after training"
    )

    return parser.parse_args()


def resolve_checkpoint_path(checkpoint_arg):
    """
    Smart checkpoint path resolution.
    
    Supports:
    1. Direct file: /path/to/checkpoint_best.pth
    2. Output directory: /path/to/exp/DD-MM-YYYY/HH-MM-SS/ (auto-finds best)
    3. Checkpoints directory: /path/to/exp/DD-MM-YYYY/HH-MM-SS/checkpoints/
    
    Returns: Path to checkpoint file
    """
    ckpt_path = Path(checkpoint_arg)
    
    # Case 1: Direct file
    if ckpt_path.is_file():
        return ckpt_path
    
    # Case 2: Directory - find best checkpoint
    if ckpt_path.is_dir():
        checkpoints_dir = ckpt_path / "checkpoints"
        
        # Edge case: user provided checkpoints/ dir directly
        if not checkpoints_dir.exists() and ckpt_path.name == "checkpoints":
            checkpoints_dir = ckpt_path
        
        if not checkpoints_dir.exists():
            raise FileNotFoundError(f"No 'checkpoints' folder in: {ckpt_path}")
        
        # Priority: best > latest > most recent epoch
        for ckpt_name in ["checkpoint_best.pth", "checkpoint_latest.pth"]:
            ckpt_file = checkpoints_dir / ckpt_name
            if ckpt_file.exists():
                print(f"🏆 Found checkpoint: {ckpt_file}")
                return ckpt_file
        
        # Fallback: most recent epoch
        epoch_ckpts = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pth"))
        if epoch_ckpts:
            print(f"📝 Using most recent: {epoch_ckpts[-1]}")
            return epoch_ckpts[-1]
        
        raise FileNotFoundError(f"No checkpoints found in: {checkpoints_dir}")
    
    raise FileNotFoundError(f"Path does not exist: {ckpt_path}")


def resolve_log_path(log_arg):
    """
    Smart log path resolution.
    
    Supports:
    1. Direct CSV file: /path/to/training.csv
    2. Log file: /path/to/training.log (converts to .csv)
    3. Output directory: /path/to/exp/DD-MM-YYYY/HH-MM-SS/ (auto-finds CSV)
    
    Returns: Path to CSV log file
    """
    path = Path(log_arg)
    
    # Case 1: Direct file
    if path.is_file():
        if path.suffix == ".log":
            # Try to find CSV sibling
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                print(f"ℹ️  Auto-corrected .log → .csv")
                return csv_path
        return path
    
    # Case 2: Directory - find CSV log
    if path.is_dir():
        # Common CSV names
        for csv_name in ["training.csv", "phase1_train.csv", "training_log.csv"]:
            csv_path = path / csv_name
            if csv_path.exists():
                print(f"📊 Found log: {csv_path}")
                return csv_path
        
        raise FileNotFoundError(f"No CSV log found in: {path}")
    
    raise FileNotFoundError(f"Path does not exist: {path}")


if __name__ == "__main__":
    from trainer.train import Trainer
    from utils.plotter import plot_training_logs
    from inference.generate import generate_batch, generate_single, load_model

    args = parse_args()

    if args.mode == "dataset-check":
        # Dataset Checking Mode
        from scripts.check_dataset_structure import check_dataset_structure
        
        # Determine root directory
        if args.data_root:
            root_dir = Path(args.data_root)
        else:
            # Use default from configs
            root_dir = Path("data/ShapeNetCore_V5")
        
        if not root_dir.exists():
            print(f"❌ Dataset directory not found: {root_dir}")
            sys.exit(1)
        
        # Determine which splits to check
        splits_to_check = []
        if args.split == "all" or args.split is None:
            # Check all available splits
            for split in ["train", "val", "test"]:
                split_dir = root_dir / split
                if split_dir.exists():
                    splits_to_check.append((split, split_dir))
                elif split_dir.parent.is_dir():
                    # Maybe root is the split itself (no train/val/test subdirs)
                    if not splits_to_check:  # Only add root once
                        splits_to_check.append(("root", root_dir))
                        break
        else:
            # Check specific split
            split_dir = root_dir / args.split
            if split_dir.exists():
                splits_to_check.append((args.split, split_dir))
            else:
                print(f"❌ Split directory not found: {split_dir}")
                sys.exit(1)
        
        # Check each split
        all_stats = {}
        for split_name, split_path in splits_to_check:
            print(f"\n{'='*80}")
            print(f"Checking: {split_name.upper()} split")
            print(f"{'='*80}")
            stats = check_dataset_structure(str(split_path), create_report=False)
            if stats:
                all_stats[split_name] = stats
        
        # Print overall summary if multiple splits
        if len(all_stats) > 1:
            print(f"\n\n{'='*80}")
            print("📊 OVERALL DATASET SUMMARY")
            print(f"{'='*80}")
            
            for split_name, stats in all_stats.items():
                total_valid = sum(s["valid"] for s in stats.values())
                total_objects = sum(s["total"] for s in stats.values())
                print(f"\n{split_name.upper()}: {total_valid}/{total_objects} valid objects")
                
                if args.detailed:
                    for class_id, class_stats in stats.items():
                        print(f"  {class_stats['name']:10} - {class_stats['valid']:4}/{class_stats['total']:4} objects")
        
        print(f"\n{'='*80}")
        print("✅ Dataset structure check complete!")
        print(f"{'='*80}\n")

    elif args.mode == "plot":
        # Plotting Mode with smart path resolution
        if not args.log_dir:
            print("❌ --log-dir is required for plotting")
            print("Usage: python src/main.py --mode plot --log-dir output/exp/DATE/TIME")
            sys.exit(1)
        
        try:
            csv_path = resolve_log_path(args.log_dir)
            print(f"📈 Plotting training logs from: {csv_path}")
            plot_training_logs(csv_path)
            print("✅ Graphs generated successfully")
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

    elif args.mode == "generate":
        # Generation Mode with smart path resolution
        if not args.checkpoint:
            print("❌ --checkpoint is required for generation")
            print("Usage: python src/main.py --mode generate --checkpoint output/exp/DATE/TIME")
            sys.exit(1)
        
        try:
            # Resolve checkpoint path
            ckpt_path = resolve_checkpoint_path(args.checkpoint)
            print(f"🎨 Loading model from: {ckpt_path}")
            
            # Setup device
            if args.device == "auto":
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                device = torch.device(args.device)
            
            # Load model
            encoder, explicit, model_cfg = load_model(args.config_dir, str(ckpt_path), device)
            
            # Determine output directory
            if ckpt_path.parent.name == "checkpoints":
                base_exp_dir = ckpt_path.parent.parent
            else:
                base_exp_dir = ckpt_path.parent
            
            if args.output_dir:
                out_dir = Path(args.output_dir)
            elif args.exp_name:
                out_dir = base_exp_dir / args.exp_name
            else:
                out_dir = base_exp_dir / "generated_samples"
            
            print(f"📁 Output directory: {out_dir}")
            
            # Generate
            if args.image:
                print(f"Generating from image: {args.image}")
                path = generate_single(args.image, out_dir, encoder, explicit, device, model_cfg)
                print(f"✅ Saved: {path}")
            else:
                print(f"Generating batch samples...")
                generate_batch(args.config_dir, out_dir, encoder, explicit, device, model_cfg, num_samples=args.num_samples or 20)
                print("✅ Batch generation complete")
                
        except FileNotFoundError as e:
            print(f"❌ {e}")
            sys.exit(1)

    else:
        # Training Modes (train, resume, scratch) or Test Mode
        trainer = Trainer(args)
        if args.mode == "test":
             trainer.evaluate_test()
        else:
             trainer.run()
