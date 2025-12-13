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

from utils.checkpoint import CheckpointManager
from utils.setup import EnvironmentSetup
from utils.plotter import plot_training_logs
from inference.generate import generate_batch


class Trainer:
    """
    Main trainer class for HIE-GAN Phase 1.
    
    This class handles the training loop, validation, logging, and model management.
    It uses helper classes for checkpointing and environment setup to keep the
    core logic clean.
    """

    def __init__(self, args):
        """
        Initialize trainer with parsed arguments.
        
        Args:
            args: Parsed command line arguments containing training options.
        """
        self.args = args

        # Set random seeds for reproducibility
        if args.seed is not None:
            EnvironmentSetup.set_seed(args.seed, args.deterministic)

        # Load configurations
        self.dataset_cfg, self.model_cfg, self.train_cfg = self._load_configs()

        # Apply CLI overrides to configs
        self._apply_overrides()

        # Setup directories
        self.exp_dir, self.ckpt_dir = EnvironmentSetup.setup_directories(args, self.train_cfg)

        # Setup logging
        self.logger, self.csv_logger, self.metrics_logger = self._setup_logging()

        # Setup device
        self.device = EnvironmentSetup.setup_device(self.args.device, self.args.quiet)

        # Setup mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
        
        # Initialize Checkpoint Manager
        self.ckpt_manager = CheckpointManager(
            self.ckpt_dir, 
            self.logger, 
            keep_last=self.args.keep_last
        )

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

    def _handle_checkpoint_loading(self):
        """Handle checkpoint loading based on mode"""
        checkpoint = None
        
        if self.args.checkpoint:
            # Force load specific checkpoint
            ckpt_path = Path(self.args.checkpoint)
            if ckpt_path.is_dir():
                ckpt_path = ckpt_path / "checkpoints" / "checkpoint_latest.pth"
                if not ckpt_path.exists():
                     ckpt_path = Path(self.args.checkpoint) / "checkpoint_latest.pth"
            
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
            checkpoint = self.ckpt_manager.load(ckpt_path, self.device)

        elif self.args.mode == "resume":
            # Force resume from latest
            latest_ckpt = self.ckpt_dir / "checkpoint_latest.pth"
            if latest_ckpt.exists():
                checkpoint = self.ckpt_manager.load(latest_ckpt, self.device)
            else:
                self.logger.error(f"❌ No checkpoint found at {latest_ckpt}")
                raise FileNotFoundError(f"Cannot resume: {latest_ckpt} not found")

        elif self.args.mode == "train":
            # Auto-resume if checkpoint exists
            latest_ckpt = self.ckpt_dir / "checkpoint_latest.pth"
            if latest_ckpt.exists():
                self.logger.info(f"Auto-resuming from: {latest_ckpt}")
                checkpoint = self.ckpt_manager.load(latest_ckpt, self.device)
            else:
                self.logger.info("Starting fresh training")

        elif self.args.mode == "scratch":
            self.logger.info("Starting fresh training (ignoring checkpoints)")
            
        # Apply checkpoint state if loaded
        if checkpoint:
            self._apply_checkpoint(checkpoint)

    def _apply_checkpoint(self, checkpoint):
        """Applies loaded checkpoint state to model and optimizer"""
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.explicit.load_state_dict(checkpoint["explicit_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        self.logger.info(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        self.logger.info(f"  Best loss: {self.best_loss:.6f}")

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
                leave=False,
                ncols=100,
                disable=self.args.quiet,
                position=0
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
                    if not self.args.no_save:
                        state = {
                            "epoch": epoch,
                            "encoder_state_dict": self.encoder.state_dict(),
                            "explicit_state_dict": self.explicit.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                            "loss": avg_loss,
                            "best_loss": self.best_loss,
                        }
                        self.ckpt_manager.save(state, epoch, avg_loss, is_best, self.best_loss)
                    
                    if not is_best and not self.args.no_save:
                        self.logger.info(f"  💾 Checkpoint saved at epoch {epoch + 1}")

        except KeyboardInterrupt:
            self.logger.info("\n⚠ Training interrupted by user")
            self.logger.info("Saving emergency checkpoint...")
            if not self.args.no_save:
                state = {
                    "epoch": epoch,
                    "encoder_state_dict": self.encoder.state_dict(),
                    "explicit_state_dict": self.explicit.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                    "loss": avg_loss,
                    "best_loss": self.best_loss,
                }
                self.ckpt_manager.save(state, epoch, avg_loss, False, self.best_loss)
            self.logger.info("✓ Emergency checkpoint saved")

        except Exception as e:
            self.logger.error(f"❌ Training failed with error: {e}")
            self.logger.info("Saving emergency checkpoint...")
            try:
                if not self.args.no_save:
                    state = {
                        "epoch": epoch,
                        "encoder_state_dict": self.encoder.state_dict(),
                        "explicit_state_dict": self.explicit.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
                        "loss": avg_loss,
                        "best_loss": self.best_loss,
                    }
                    self.ckpt_manager.save(state, epoch, avg_loss, False, self.best_loss)
                self.logger.info("✓ Emergency checkpoint saved")
            except:
                pass
            raise

        self.logger.info("=" * 70)
        self.logger.info("Training finished!")
        self.logger.info(f"Best loss: {self.best_loss:.6f}")
        self.logger.info(f"Checkpoints saved to: {self.ckpt_dir}")
        self.logger.info(f"Logs saved to: {self.exp_dir}")

        # Post-Training Hooks
        if self.args.generate_graph:
            self.logger.info("Generating training graphs...")
            try:
                plot_training_logs(self.csv_logger.path, self.exp_dir / "graphs")
            except Exception as e:
                self.logger.error(f"Failed to generate graphs: {e}")

        if self.args.generate_model:
            self.logger.info("Generating sample 3D models...")
            try:
                # Reload best model for generation
                best_ckpt = self.ckpt_dir / "checkpoint_best.pth"
                if best_ckpt.exists():
                    checkpoint = torch.load(best_ckpt, map_location=self.device)
                    self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
                    self.explicit.load_state_dict(checkpoint["explicit_state_dict"])
                    
                generate_batch(
                    self.args.config_dir, 
                    self.exp_dir / "generated_samples",
                    self.encoder,
                    self.explicit,
                    self.device,
                    self.model_cfg,
                    num_samples=10 # Default to 10 per run
                )
            except Exception as e:
                self.logger.error(f"Failed to generate models: {e}")

        self.logger.info("=" * 70)

# Backward compatibility wrapper (kept if needed, but class is main)
def train_phase1(args):
    """Wrapper function for backward compatibility"""
    trainer = Trainer(args)
    trainer.run()
