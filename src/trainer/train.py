import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm  # Changed for Colab compatibility
from pathlib import Path
import random
import numpy as np
import datetime
import json

from utils.config import load_configs
from utils.logger import setup_logger, CSVLogger, MetricsLogger
from dataloader.dataset import ShapeNetDataset
from models.feature_extractor import FeatureExtractor
from models.explicit_branch import ExplicitDeformer
from models.init_mesh import create_sphere_mesh
from losses.chamfer import chamfer_loss
from losses.regularizers import smoothness_loss, edge_length_loss


class Trainer:
    """Main trainer class for HIE-GAN Phase 1"""

    def __init__(self, args):
        """Initialize trainer with parsed arguments"""
        self.args = args

        # Set random seeds for reproducibility
        if args.seed is not None:
            self._set_seed(args.seed)

        # Load configurations
        self.dataset_cfg, self.model_cfg, self.train_cfg = self._load_configs()

        # Apply CLI overrides to configs
        self._apply_overrides()

        # Setup directories
        self.exp_dir, self.ckpt_dir = self._setup_directories()

        # Setup logging
        self.logger, self.csv_logger, self.metrics_logger = self._setup_logging()

        # Setup device
        self.device = self._setup_device()

        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None

        # Log configuration
        self._log_config()

        # Initialize components (lazy loading)
        self.train_loader = None
        self.val_loader = None
        self.encoder = None
        self.explicit = None
        self.optimizer = None
        self.start_epoch = 0
        self.best_loss = float("inf")

    def _set_seed(self, seed):
        """Set random seed for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if self.args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _load_configs(self):
        """Load YAML configurations"""
        return load_configs(config_dir=self.args.config_dir)

    def _apply_overrides(self):
        """Apply CLI argument overrides to configs with type safety"""
        if self.args.epochs:
            self.train_cfg["epochs"] = int(self.args.epochs)
        if self.args.batch_size:
            self.train_cfg["batch_size"] = int(self.args.batch_size)
        if self.args.lr:
            self.train_cfg["optimizer"]["lr"] = float(self.args.lr)
        else:
            # Ensure lr from YAML is float
            self.train_cfg["optimizer"]["lr"] = float(self.train_cfg["optimizer"]["lr"])

        if self.args.num_workers is not None:
            self.train_cfg["num_workers"] = int(self.args.num_workers)
        if self.args.save_every:
            self.train_cfg["checkpoints"]["save_every"] = int(self.args.save_every)
        if self.args.log_every:
            self.train_cfg["logging"]["log_every_n_steps"] = int(self.args.log_every)
        if self.args.data_root:
            self.dataset_cfg["root_dir"] = self.args.data_root
        if self.args.log_dir:
            self.train_cfg["logging"]["log_dir"] = self.args.log_dir

    def _setup_directories(self):
        """Setup experiment and checkpoint directories"""
        # Get current timestamp
        now = datetime.datetime.now()
        date_str = now.strftime("%d-%m-%Y")
        time_str = now.strftime("%H-%M-%S")
        
        # Base output directory: output/dd-mm-yyyy/timestamp
        base_dir = Path("output") / date_str / time_str
        
        if self.args.exp_name:
             base_dir = base_dir / self.args.exp_name

        # Update log dir in config so other components know where to log
        self.train_cfg["logging"]["log_dir"] = str(base_dir)
        self.train_cfg["checkpoints"]["dir"] = str(base_dir / "checkpoints")

        exp_dir = base_dir
        ckpt_dir = base_dir / "checkpoints"

        exp_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        return exp_dir, ckpt_dir

    def _setup_logging(self):
        """Setup all loggers"""
        logger = setup_logger(
            self.exp_dir,
            self.train_cfg["logging"]["log_filename"],
            quiet=self.args.quiet
        )
        csv_logger = CSVLogger(
            self.exp_dir,
            self.train_cfg["logging"]["csv_filename"]
        )
        metrics_logger = MetricsLogger(
            self.exp_dir,
            "metrics.json"
        )
        return logger, csv_logger, metrics_logger

    def _setup_device(self):
        """Setup compute device"""
        if self.args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                if not self.args.quiet:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"✓ Auto-selected GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                device = torch.device("cpu")
                if not self.args.quiet:
                    print("⚠ No GPU available, using CPU")
        else:
            device = torch.device(self.args.device)
            if not self.args.quiet:
                if device.type == "cuda":
                    gpu_name = torch.cuda.get_device_name(device)
                    print(f"✓ Using GPU: {gpu_name}")
                else:
                    print(f"✓ Using device: {device}")

        return device

    def _log_config(self):
        """Log training configuration"""
        if self.args.quiet:
            return

        self.logger.info("=" * 70)
        self.logger.info("HIE-GAN Phase 1 Training")
        self.logger.info("=" * 70)
        self.logger.info(f"Mode: {self.args.mode}")
        self.logger.info(f"Experiment: {self.args.exp_name or 'default'}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.train_cfg['epochs']}")
        self.logger.info(f"Batch Size: {self.train_cfg['batch_size']}")
        self.logger.info(f"Learning Rate: {self.train_cfg['optimizer']['lr']}")
        self.logger.info(f"Workers: {self.train_cfg['num_workers']}")

        if self.args.mixed_precision:
            self.logger.info("✓ Mixed precision enabled")
        if self.args.grad_clip:
            self.logger.info(f"✓ Gradient clipping: {self.args.grad_clip}")
        if self.args.compile:
            self.logger.info("✓ Model compilation enabled")
        if self.args.debug:
            self.logger.info("⚠ DEBUG MODE ACTIVE")
        if self.args.seed is not None:
            self.logger.info(f"✓ Random seed: {self.args.seed}")

        self.logger.info("=" * 70)

    def _build_dataset(self):
        """Build train and validation datasets with detailed logging"""
        self.logger.info("=" * 70)
        self.logger.info("Loading Dataset")
        self.logger.info("=" * 70)
        self.logger.info(f"Dataset root: {self.dataset_cfg['root_dir']}")
        self.logger.info(f"Classes: {', '.join(self.dataset_cfg['classes'])}")

        dataset = ShapeNetDataset(
            root_dir=self.dataset_cfg["root_dir"],
            classes=self.dataset_cfg["classes"],
            pc_filename=self.dataset_cfg["pointcloud"]["filename"],
            image_size=self.dataset_cfg["image"]["size"],
        )

        total_objects = len(dataset)
        self.logger.info(f"Total objects found: {total_objects}")

        # Limit dataset size for debugging/testing
        if self.args.debug:
            dataset.object_paths = dataset.object_paths[:32]
            self.logger.info(f"⚠ DEBUG: Limited to {len(dataset)} samples")
        elif self.args.num_samples:
            dataset.object_paths = dataset.object_paths[:self.args.num_samples]
            self.logger.info(f"⚠ Limited to {len(dataset)} samples")

        # Calculate batches
        batch_size = self.train_cfg["batch_size"]
        num_batches = len(dataset) // batch_size
        if len(dataset) % batch_size != 0:
            num_batches += 1

        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Total batches per epoch: {num_batches}")
        self.logger.info(f"Samples per epoch: {len(dataset)}")

        # Split into train/val if needed
        if self.args.val_split and self.args.val_split > 0:
            val_size = int(len(dataset) * self.args.val_split)
            train_size = len(dataset) - val_size
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.args.seed or 42)
            )
            self.logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
            train_batches = len(train_dataset) // batch_size
            val_batches = len(val_dataset) // batch_size
            self.logger.info(f"Train batches: {train_batches} | Val batches: {val_batches}")
        else:
            train_dataset = dataset
            val_dataset = None
            self.logger.info("No validation split")

        # Warning if very few batches
        if num_batches < 10:
            self.logger.warning(f"⚠ Only {num_batches} batches per epoch!")
            self.logger.warning(f"⚠ Consider decreasing batch size or adding more data")

        # Create dataloaders
        if self.args.pin_memory is not None:
            pin_memory = self.args.pin_memory
        else:
            pin_memory = (self.device.type == "cuda")

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.train_cfg["num_workers"],
            pin_memory=pin_memory,
            persistent_workers=(self.train_cfg["num_workers"] > 0),
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.train_cfg["num_workers"],
                pin_memory=pin_memory,
                persistent_workers=(self.train_cfg["num_workers"] > 0),
            )

        self.logger.info("=" * 70)

    def _build_models(self):
        """Build and initialize models"""
        self.logger.info("Initializing models...")

        # Feature extractor
        self.encoder = FeatureExtractor(
            out_dim=self.model_cfg["feature_extractor"]["out_dim"]
        ).to(self.device)

        # Explicit branch
        V0, E0 = create_sphere_mesh(
            self.model_cfg["explicit_branch"]["init_mesh"]["subdivisions"]
        )
        V0, E0 = V0.to(self.device), E0.to(self.device)

        self.explicit = ExplicitDeformer(
            init_mesh=(V0, E0),
            feature_dim=self.model_cfg["feature_extractor"]["out_dim"],
            hidden_dims=self.model_cfg["explicit_branch"]["gcn_hidden_dims"],
        ).to(self.device)

        # Count parameters
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        explicit_params = sum(p.numel() for p in self.explicit.parameters())
        total_params = encoder_params + explicit_params

        self.logger.info(f"Encoder parameters: {encoder_params:,}")
        self.logger.info(f"Explicit branch parameters: {explicit_params:,}")
        self.logger.info(f"Total parameters: {total_params:,}")

        # Compile models if requested (PyTorch 2.0+)
        if self.args.compile:
            try:
                self.logger.info("Compiling models with torch.compile...")
                self.encoder = torch.compile(self.encoder)
                self.explicit = torch.compile(self.explicit)
                self.logger.info("✓ Model compilation successful")
            except Exception as e:
                self.logger.warning(f"⚠ Model compilation failed: {e}")
                self.logger.warning("Continuing without compilation")

    def _build_optimizer(self):
        """Build optimizer with type safety"""
        lr = float(self.train_cfg["optimizer"]["lr"])
        weight_decay = float(self.args.weight_decay) if self.args.weight_decay else 0.0

        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.explicit.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )

        if weight_decay > 0:
            self.logger.info(f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")
        else:
            self.logger.info(f"Optimizer: Adam (lr={lr})")

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint from path"""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.explicit.load_state_dict(checkpoint["explicit_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        self.logger.info(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        self.logger.info(f"  Best loss: {self.best_loss:.6f}")

    def _handle_checkpoint_loading(self):
        """Handle checkpoint loading based on mode"""
        if self.args.checkpoint:
            # Force load specific checkpoint
            ckpt_path = Path(self.args.checkpoint)
            if ckpt_path.is_dir():
                # If directory provided, look for latest checkpoint
                ckpt_path = ckpt_path / "checkpoints" / "checkpoint_latest.pth"
                if not ckpt_path.exists():
                     # Try looking directly in the dir
                     ckpt_path = Path(self.args.checkpoint) / "checkpoint_latest.pth"
            
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
                
            self._load_checkpoint(ckpt_path)

        elif self.args.mode == "resume":
            # Force resume from latest
            latest_ckpt = self.ckpt_dir / "checkpoint_latest.pth"
            if latest_ckpt.exists():
                self._load_checkpoint(latest_ckpt)
            else:
                self.logger.error(f"❌ No checkpoint found at {latest_ckpt}")
                raise FileNotFoundError(f"Cannot resume: {latest_ckpt} not found")

        elif self.args.mode == "train":
            # Auto-resume if checkpoint exists
            latest_ckpt = self.ckpt_dir / "checkpoint_latest.pth"
            if latest_ckpt.exists():
                self.logger.info(f"Auto-resuming from: {latest_ckpt}")
                self._load_checkpoint(latest_ckpt)
            else:
                self.logger.info("Starting fresh training")

        elif self.args.mode == "scratch":
            self.logger.info("Starting fresh training (ignoring checkpoints)")

    def _save_checkpoint(self, epoch, loss, is_best=False):
        """Save checkpoint"""
        if self.args.no_save:
            return

        checkpoint = {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "explicit_state_dict": self.explicit.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "loss": loss,
            "best_loss": self.best_loss,
        }

        # Save epoch checkpoint
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, ckpt_path)

        # Save latest
        latest_path = self.ckpt_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.ckpt_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"  💾 Best checkpoint saved (loss: {loss:.6f})")

        # Save metadata
        metadata = {
            "last_epoch": epoch,
            "best_epoch": -1,  # You might want to track this in self
            "last_loss": loss,
            "best_loss": self.best_loss,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # If this is best, update best epoch
        if is_best:
            metadata["best_epoch"] = epoch
            
        # Try to read existing metadata to preserve history if needed, 
        # but for now we overwrite with current state
        meta_path = self.exp_dir / "training_metadata.json"
        
        # If exists, read to keep best_epoch if we are not currently best
        if meta_path.exists() and not is_best:
            try:
                with open(meta_path, 'r') as f:
                    old_meta = json.load(f)
                    metadata["best_epoch"] = old_meta.get("best_epoch", -1)
            except:
                pass

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        # Rotate old checkpoints
        all_checkpoints = sorted(self.ckpt_dir.glob("checkpoint_epoch_*.pth"))
        if len(all_checkpoints) > self.args.keep_last:
            for old_ckpt in all_checkpoints[:-self.args.keep_last]:
                old_ckpt.unlink()
                if not self.args.quiet:
                    self.logger.info(f"  🗑️  Removed old checkpoint: {old_ckpt.name}")

    def train_epoch(self, epoch):
        """Train single epoch - Colab optimized"""
        self.encoder.train()
        self.explicit.train()

        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        # Setup progress bar (Colab compatible)
        if not self.args.no_tqdm:
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.train_cfg['epochs']}",
                total=num_batches,
                leave=False,  # Don't leave bar after completion (fixes mess)
                ncols=100,
                disable=self.args.quiet,
                position=0  # Single position for Colab
            )
        else:
            pbar = self.train_loader

        for step, (img, gt_pc) in enumerate(pbar):
            # Move to device
            img = img.to(self.device, non_blocking=True)
            gt_pc = gt_pc.to(self.device, non_blocking=True)


            # Forward pass with mixed precision
            if self.args.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    feat = self.encoder(img)
                    pred_pc = self.explicit(feat)
                    
                    # Losses
                    cham_loss = chamfer_loss(pred_pc, gt_pc)
                    
                    # Get mesh structure for regularizers
                    # explicit.E is the edge index buffer
                    pred_mesh = (pred_pc, self.explicit.E)
                    
                    smooth_loss = smoothness_loss(pred_mesh, penalty=0.1)
                    edge_loss = edge_length_loss(pred_mesh, target_length=0.05, penalty=0.1)
                    
                    loss = cham_loss + smooth_loss + edge_loss

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()

                if self.args.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.explicit.parameters()),
                        self.args.grad_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                feat = self.encoder(img)
                pred_pc = self.explicit(feat)
                
                # Losses
                cham_loss = chamfer_loss(pred_pc, gt_pc)
                
                # Regularizers
                pred_mesh = (pred_pc, self.explicit.E)
                smooth_loss = smoothness_loss(pred_mesh, penalty=0.1)
                edge_loss = edge_length_loss(pred_mesh, target_length=0.05, penalty=0.1)
                
                loss = cham_loss + smooth_loss + edge_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                if self.args.grad_clip:
                    nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.explicit.parameters()),
                        self.args.grad_clip
                    )

                self.optimizer.step()

            # Logging
            loss_val = loss.item()
            epoch_loss += loss_val

            # Update progress bar
            if not self.args.no_tqdm and not self.args.quiet:
                pbar.set_postfix({"loss": f"{loss_val:.6f}"})

            # Periodic logging
            log_every = self.train_cfg["logging"]["log_every_n_steps"]
            if step % log_every == 0 and step > 0:
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.train_cfg['epochs']} | "
                    f"Step {step}/{num_batches} | Loss: {loss_val:.6f}"
                )
                self.csv_logger.write(epoch, step, loss_val)
                self.metrics_logger.log_step(epoch, step, {"loss": loss_val})

            # Debug mode: only 5 batches
            if self.args.debug and step >= 4:
                break

        avg_loss = epoch_loss / (step + 1)
        return avg_loss

    def run(self):
        """Main training loop"""
        # Dry run check
        if self.args.dry_run:
            self.logger.info("🔍 Dry run mode - initializing only")
            self._build_dataset()
            self._build_models()
            self._build_optimizer()
            self.logger.info("✓ Dry run complete - all components initialized successfully")
            return

        # Build all components
        self._build_dataset()
        self._build_models()
        self._build_optimizer()
        self._handle_checkpoint_loading()

        # Training loop
        self.logger.info("=" * 70)
        self.logger.info("Starting training...")
        self.logger.info("=" * 70)

        try:
            for epoch in range(self.start_epoch, self.train_cfg["epochs"]):
                # Train epoch
                avg_loss = self.train_epoch(epoch)

                # Epoch summary
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.train_cfg['epochs']} complete | "
                    f"Avg Loss: {avg_loss:.6f}"
                )
                self.metrics_logger.log_epoch(epoch, {"avg_loss": avg_loss})

                # Check if best
                is_best = avg_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_loss

                # Save checkpoint
                save_every = self.train_cfg["checkpoints"].get("save_every", 1)
                if (epoch + 1) % save_every == 0 or is_best:
                    self._save_checkpoint(epoch, avg_loss, is_best)
                    if not is_best:
                        self.logger.info(f"  💾 Checkpoint saved at epoch {epoch + 1}")

        except KeyboardInterrupt:
            self.logger.info("\n⚠ Training interrupted by user")
            self.logger.info("Saving emergency checkpoint...")
            self._save_checkpoint(epoch, avg_loss, False)
            self.logger.info("✓ Emergency checkpoint saved")

        except Exception as e:
            self.logger.error(f"❌ Training failed with error: {e}")
            self.logger.info("Saving emergency checkpoint...")
            try:
                self._save_checkpoint(epoch, avg_loss, False)
                self.logger.info("✓ Emergency checkpoint saved")
            except:
                pass
            raise

        self.logger.info("=" * 70)
        self.logger.info("Training finished!")
        self.logger.info(f"Best loss: {self.best_loss:.6f}")
        self.logger.info(f"Checkpoints saved to: {self.ckpt_dir}")
        self.logger.info(f"Logs saved to: {self.exp_dir}")
        self.logger.info("=" * 70)


# Backward compatibility wrapper
def train_phase1(args):
    """Wrapper function for backward compatibility"""
    trainer = Trainer(args)
    trainer.run()
