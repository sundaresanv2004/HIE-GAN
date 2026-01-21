import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
import random
import numpy as np
import trimesh

from utils.config import load_configs
from utils.logger import setup_logger, CSVLogger, MetricsLogger
from dataloader.dataset import DatasetLoader, ShapeNetDataset
from models.feature_extractor import FeatureExtractor
from models.explicit_branch import ExplicitDeformer
from models.implicit_branch import ImplicitDecoder
from models.fusion_module import FusionModule
from models.hie_gan import HIEGANModel
from models.init_mesh import create_sphere_mesh
from losses.chamfer import chamfer_loss
from losses.sdf_loss import SDFLoss
from losses.regularizers import smoothness_loss, edge_length_loss
from utils.mesh_ops import generate_mesh_from_sdf

from utils.checkpoint import CheckpointManager
from utils.setup import EnvironmentSetup
from utils.plotter import plot_training_graphs
from inference.generate import generate_batch


class Trainer:
    """
    Main trainer class for HIE-GAN Phase 3 (Fusion Module).
    """

    def __init__(self, args):
        self.args = args

        # Set random seeds
        if args.seed is not None:
            EnvironmentSetup.set_seed(args.seed, args.deterministic)

        # Load configurations
        self.dataset_cfg, self.model_cfg, self.train_cfg = self._load_configs()
        self._apply_overrides()

        # Setup directories
        self.exp_dir, self.ckpt_dir = EnvironmentSetup.setup_directories(args, self.train_cfg)

        # Setup logging
        self.logger, self.csv_logger, self.metrics_logger = self._setup_logging()

        # Setup device
        self.device = EnvironmentSetup.setup_device(self.args.device, self.args.quiet)

        # Setup mixed precision
        self.use_amp = args.mixed_precision or self.train_cfg.get("mixed_precision", False)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            self.logger.info("⚡ Mixed Precision Enabled (FP16)")
        
        # Initialize Checkpoint Manager
        self.ckpt_manager = CheckpointManager(
            self.ckpt_dir, 
            self.logger, 
            keep_last=self.args.keep_last
        )

        self._log_config()

        # Components
        self.dataset_loader = None
        self.train_loader = None
        self.val_loader = None
        self.encoder = None
        self.explicit = None
        self.implicit = None
        self.fusion = None
        self.model = None  # HIEGANModel wrapper
        self.optimizer = None
        self.sdf_loss_fn = None
        
        self.start_epoch = 0
        self.best_loss = float("inf")

    def _load_configs(self):
        return load_configs(config_dir=self.args.config_dir)

    def _apply_overrides(self):
        """Apply command-line overrides to configuration"""
        if self.args.data_root:
            self.dataset_cfg["root_dir"] = self.args.data_root
        
        if self.args.epochs:
            self.train_cfg["epochs"] = int(self.args.epochs)
        if self.args.batch_size:
            self.train_cfg["batch_size"] = int(self.args.batch_size)
        if self.args.lr:
            self.train_cfg["optimizer"]["lr"] = float(self.args.lr)
        if self.args.num_workers is not None:
            self.train_cfg["num_workers"] = int(self.args.num_workers)

    def _setup_logging(self):
        logger = setup_logger(self.exp_dir, self.train_cfg["logging"]["log_filename"], quiet=self.args.quiet)
        csv_logger = CSVLogger(self.exp_dir, self.train_cfg["logging"]["csv_filename"])
        metrics_logger = MetricsLogger(self.exp_dir, "metrics.json")
        return logger, csv_logger, metrics_logger

    def _log_config(self):
        """Log configuration summary"""
        if self.args.quiet:
            return
        
        self.logger.info("=" * 70)
        self.logger.info("HIE-GAN Phase 3 Training (Fusion Module)")
        self.logger.info("=" * 70)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.train_cfg['epochs']}")
        self.logger.info(f"Batch Size: {self.train_cfg['batch_size']}")

    def _build_dataset(self):
        self.dataset_loader = DatasetLoader(
            self.dataset_cfg, 
            self.train_cfg, 
            self.args, 
            self.logger
        )
        self.logger.info("📊 Loading Dataset")
        self.logger.info("=" * 70)
        
        self.train_loader, self.val_loader, self.test_loader = self.dataset_loader.load(ShapeNetDataset)

    def _build_models(self):
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
            use_layer_norm=self.model_cfg["explicit_branch"].get("use_layer_norm", True)
        ).to(self.device)
        
        # Implicit branch
        imp_cfg = self.model_cfg["implicit_branch"]
        self.implicit = ImplicitDecoder(
            feature_dim=self.model_cfg["feature_extractor"]["out_dim"],
            hidden_dim=imp_cfg["hidden_dim"],
            num_layers=imp_cfg["num_layers"],
            skip_connection_at=imp_cfg.get("skip_connection_at", []),
            use_positional_encoding=imp_cfg.get("use_positional_encoding", True),
            pos_enc_levels=imp_cfg.get("pos_enc_levels", 6)
        ).to(self.device)

        # Fusion Module (Phase 3)
        self.fusion = FusionModule(step_size=1.0).to(self.device)

        # Losses
        self.sdf_loss_fn = SDFLoss(clamp_dist=0.1, weight=1.0).to(self.device)

        # Build Unified Model
        self.model = HIEGANModel(self.encoder, self.explicit, self.implicit, self.fusion).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")

        # Compile models if requested (PyTorch 2.0+)
        use_compile = self.args.compile or self.train_cfg.get("compile", False)
        if use_compile:
             self.logger.info("🚀 Compiling models (torch.compile)...")
             # Compile the whole model
             self.model = torch.compile(self.model)
        
        # Parallelism (Multi-GPU)
        if torch.cuda.device_count() > 1:
            self.logger.info(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = nn.DataParallel(self.model)

    def _build_optimizer(self):
        """Initialize optimizer for all model components"""
        lr = float(self.train_cfg["optimizer"]["lr"])
        weight_decay = float(self.args.weight_decay) if self.args.weight_decay else 0.0

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.logger.info(f"Optimizer: Adam (lr={lr})")

    def _handle_checkpoint_loading(self):
        """
        Handle checkpoint loading with smart path resolution.
        
        Supports two modes:
        1. Direct file path: /path/to/checkpoint_best.pth
        2. Output directory: /path/to/output/exp_name/DD-MM-YYYY/HH-MM-SS/
           - Automatically loads checkpoint_best.pth from checkpoints/ subfolder
        """
        checkpoint = None
        
        if self.args.checkpoint:
            ckpt_path = Path(self.args.checkpoint)
            
            # Case 1: Direct file path
            if ckpt_path.is_file():
                self.logger.info(f"📂 Loading checkpoint from file: {ckpt_path}")
                checkpoint = self.ckpt_manager.load(ckpt_path, self.device)
            
            # Case 2: Directory path - auto-detect best checkpoint
            elif ckpt_path.is_dir():
                # Check if this is the experiment root or the timestamped subdirectory
                checkpoints_dir = ckpt_path / "checkpoints"
                
                if not checkpoints_dir.exists():
                    # Maybe user provided the checkpoints directory directly
                    if ckpt_path.name == "checkpoints":
                        checkpoints_dir = ckpt_path
                    else:
                        self.logger.error(f"❌ No 'checkpoints' folder found in: {ckpt_path}")
                        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoints_dir}")
                
                # Priority: best > latest > most recent epoch
                best_ckpt = checkpoints_dir / "checkpoint_best.pth"
                latest_ckpt = checkpoints_dir / "checkpoint_latest.pth"
                
                if best_ckpt.exists():
                    self.logger.info(f"🏆 Loading best checkpoint from: {best_ckpt}")
                    checkpoint = self.ckpt_manager.load(best_ckpt, self.device)
                elif latest_ckpt.exists():
                    self.logger.info(f"⏰ Loading latest checkpoint from: {latest_ckpt}")
                    checkpoint = self.ckpt_manager.load(latest_ckpt, self.device)
                else:
                    # Find most recent epoch checkpoint
                    epoch_ckpts = sorted(checkpoints_dir.glob("checkpoint_epoch_*.pth"))
                    if epoch_ckpts:
                        most_recent = epoch_ckpts[-1]
                        self.logger.info(f"📝 Loading most recent checkpoint: {most_recent}")
                        checkpoint = self.ckpt_manager.load(most_recent, self.device)
                    else:
                        self.logger.error(f"❌ No checkpoint files found in: {checkpoints_dir}")
                        raise FileNotFoundError(f"No checkpoints in: {checkpoints_dir}")
            else:
                self.logger.error(f"❌ Checkpoint path does not exist: {ckpt_path}")
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        elif self.args.mode == "resume":
            # Auto-resume from current experiment directory
            latest_ckpt = self.ckpt_dir / "checkpoint_latest.pth"
            if latest_ckpt.exists():
                self.logger.info(f"🔄 Auto-resuming from: {latest_ckpt}")
                checkpoint = self.ckpt_manager.load(latest_ckpt, self.device)
        
        if checkpoint:
            self._apply_checkpoint(checkpoint)


    def _apply_checkpoint(self, checkpoint):
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.explicit.load_state_dict(checkpoint["explicit_state_dict"])
        
        if "implicit_state_dict" in checkpoint:
            self.implicit.load_state_dict(checkpoint["implicit_state_dict"])
        
        if "fusion_state_dict" in checkpoint:
            self.fusion.load_state_dict(checkpoint["fusion_state_dict"])
            
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        self.logger.info(f"✓ Resumed from epoch {self.start_epoch}")

    def train_epoch(self, epoch):
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        if not self.args.no_tqdm:
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        else:
            pbar = self.train_loader

        for step, batch in enumerate(pbar):
            img, gt_pc, query_pts, query_sdf_gt = batch
            
            img = img.to(self.device, non_blocking=True)
            gt_pc = gt_pc.to(self.device, non_blocking=True)
            query_pts = query_pts.to(self.device, non_blocking=True)
            query_sdf_gt = query_sdf_gt.to(self.device, non_blocking=True)

            # Mixed Precision
            with torch.amp.autocast('cuda', enabled=(self.scaler is not None)):
                # Unified Forward Pass
                pred_pc_fused, pred_sdf, pred_pc_exp, feat = self.model(img, query_pts)
                
                loss_sdf = self.sdf_loss_fn(pred_sdf, query_sdf_gt)

                # 5. Losses
                # Retrieve explicit edges (E from canonical sphere)
                # Handle DataParallel unwrapping to access attributes
                raw_model = self.model.module if hasattr(self.model, "module") else self.model
                
                pred_mesh_exp = (pred_pc_exp, raw_model.explicit.E)
                loss_smooth = smoothness_loss(pred_mesh_exp) * 0.1
                loss_edge = edge_length_loss(pred_mesh_exp) * 0.1
                loss_cham_coarse = chamfer_loss(pred_pc_exp, gt_pc)
                loss_cham_fused = chamfer_loss(pred_pc_fused, gt_pc)
                
                total_loss = loss_cham_fused + loss_sdf + loss_cham_coarse + loss_smooth + loss_edge

            # Backprop
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                if self.args.grad_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(all_params, self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

            loss_val = total_loss.item()
            epoch_loss += loss_val
            
            # Log to CSV every step
            self.csv_logger.write(epoch, step, loss_val)
            
            # Log to metrics JSON every N steps (configurable)
            log_every_n = self.train_cfg.get("logging", {}).get("log_every_n_steps", 10)
            if step % log_every_n == 0:
                self.metrics_logger.log_step(epoch, step, {
                    "loss": loss_val,
                    "chamfer_fused": loss_cham_fused.item(),
                    "chamfer_coarse": loss_cham_coarse.item(),
                    "sdf": loss_sdf.item(),
                    "smooth": loss_smooth.item(),
                    "edge": loss_edge.item()
                })
            
            if not self.args.no_tqdm and not self.args.quiet:
                pbar.set_postfix({
                    "loss": f"{loss_val:.4f}", 
                    "cham_f": f"{loss_cham_fused.item():.4f}",
                    "sdf": f"{loss_sdf.item():.4f}"
                })

        return epoch_loss / num_batches

    def validate(self, epoch):
        """Run validation loop and correct saving of metrics/samples"""
        self.logger.info("Running validation...")
        self.model.eval()
        
        val_loss = 0.0
        
        # Save output directory for this epoch
        val_out_dir = self.exp_dir / "val_outputs" / f"epoch_{epoch+1}"
        val_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sphere for export
        temp_sphere = trimesh.creation.icosphere(subdivisions=self.model_cfg["explicit_branch"]["init_mesh"]["subdivisions"])
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                # Just take first batch for visualization to save time? 
                # Or validate on full set for metrics.
                # Let's compute loss on full set and visualize first batch.
                
                img, gt_pc, query_pts, query_sdf_gt = batch
                
                img = img.to(self.device, non_blocking=True)
                gt_pc = gt_pc.to(self.device, non_blocking=True)
                query_pts = query_pts.to(self.device, non_blocking=True)
                query_sdf_gt = query_sdf_gt.to(self.device, non_blocking=True)
                
                # Unified Forward Pass
                # Fusion requires gradients for SDF normals, even in validation if that's how it works
                # But typically validation is strictly no_grad unless specific parts need it.
                # Since FusionModule does `autograd.grad`, we generally need `grad` enabled for INPUTS at least.
                # However, `autograd.grad` works inside no_grad block IF inputs have requires_grad=True?
                # Actually, `torch.set_grad_enabled(True)` is safest for Fusion logic.
                
                with torch.set_grad_enabled(True):
                     pred_pc_fused, pred_sdf, pred_pc_exp, feat = self.model(img, query_pts)
                
                loss_sdf = self.sdf_loss_fn(pred_sdf, query_sdf_gt)
                loss_cham_fused = chamfer_loss(pred_pc_fused, gt_pc)
                
                total_loss = loss_cham_fused + loss_sdf
                val_loss += total_loss.item()
                
                # Visualize first batch (limit to 5 samples)
                if i == 0:
                    for j in range(min(5, img.shape[0])):
                        # Save Explicit
                        v_exp = pred_pc_exp[j].cpu().numpy()
                        mesh_exp = trimesh.Trimesh(vertices=v_exp, faces=temp_sphere.faces)
                        mesh_exp.export(val_out_dir / f"sample_{j}_explicit.obj")
                        
                        # Save Fused
                        v_fused = pred_pc_fused[j].cpu().numpy()
                        mesh_fused = trimesh.Trimesh(vertices=v_fused, faces=temp_sphere.faces)
                        mesh_fused.export(val_out_dir / f"sample_{j}_fused.obj")
                        
                        # Save Implicit (Marching Cubes) - expensive, do only 1 or 2
                        if j < 2:
                            # Access implicit model from wrapper
                            raw_model = self.model.module if hasattr(self.model, "module") else self.model
                            mesh_imp = generate_mesh_from_sdf(raw_model.implicit, feat[j:j+1], resolution=64, device=self.device)
                            if mesh_imp:
                                mesh_imp.export(val_out_dir / f"sample_{j}_implicit.obj")
                                
        avg_val_loss = val_loss / len(self.val_loader)
        self.logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        self.metrics_logger.log_epoch(epoch, {"val_loss": avg_val_loss})
        return avg_val_loss

    def evaluate_test(self):
        """Run evaluation on the test set after training completes"""
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            self.logger.warning("⚠️  No test loader available. Skipping test evaluation.")
            return
        
        self.logger.info("Running evaluation on TEST set...")


    def run(self):
        """Main training loop with feature toggles"""
        if self.args.dry_run:
            self.logger.info("🔍 Dry Run Mode")
            self._build_dataset()
            self._build_models()
            self._build_optimizer()
            self.logger.info("✅ Dry run complete - all systems initialized successfully")
            return

        self._build_dataset()
        self._build_models()
        self._build_optimizer()
        self._handle_checkpoint_loading()

        self.logger.info("Starting training...")
        total_start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.train_cfg["epochs"]):
                epoch_start_time = time.time()
                avg_loss = self.train_epoch(epoch)
                epoch_duration = time.time() - epoch_start_time
                
                self.logger.info(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f} | Time: {epoch_duration:.2f}s")
                
                # Validation (can be disabled with --no-validation)
                avg_val_loss = None
                
                if not self.args.no_validation:
                    val_every = self.train_cfg.get("validation", {}).get("val_every", 1)
                    if (epoch + 1) % val_every == 0:
                        avg_val_loss = self.validate(epoch)
                
                # Log epoch metrics to JSON (with both train and val loss if available)
                epoch_metrics = {
                    "train_loss": avg_loss,
                    "epoch_time": epoch_duration
                }
                if avg_val_loss is not None:
                    epoch_metrics["val_loss"] = avg_val_loss
                self.metrics_logger.log_epoch(epoch, epoch_metrics)
                
                # Use train loss as fallback for best model tracking if no validation
                if avg_val_loss is None:
                    avg_val_loss = avg_loss

                # Track best model based on validation (or train) loss
                is_best = avg_val_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_val_loss
                
                # Save checkpoints
                save_every = self.train_cfg["checkpoints"].get("save_every", 1)
                if (epoch + 1) % save_every == 0 or is_best:
                    if not self.args.no_save:
                        self.save_checkpoint(epoch, avg_val_loss, is_best)

        except KeyboardInterrupt:
            self.logger.warning("⚠️ Training interrupted!")
            self.save_checkpoint(epoch, avg_loss, False)
            
        # Post-training automation
        total_duration = time.time() - total_start_time
        self.logger.info("\n" + "="*70)
        self.logger.info(f"🎉 Training Complete! Total Time: {total_duration:.2f}s ({total_duration/60:.2f} min)")
        self.logger.info("="*70)
        
        # Generate training graphs (can be disabled with --no-plot)
        if not self.args.no_plot:
            self.logger.info("📊 Generating training graphs...")
            success = plot_training_graphs(self.exp_dir)
            if success:
                self.logger.info("✅ Training graphs saved")
            else:
                self.logger.warning("⚠️ Graph generation completed with warnings")
            
        # Automatic testing (can be disabled with --no-test)
        if not self.args.no_test:
            self.evaluate_test()

    def save_checkpoint(self, epoch, loss, is_best):
        # Unwrap model to save state dicts compatible with previous structure
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
        state = {
            "epoch": epoch,
            "encoder_state_dict": raw_model.encoder.state_dict(),
            "explicit_state_dict": raw_model.explicit.state_dict(),
            "implicit_state_dict": raw_model.implicit.state_dict(),
            "fusion_state_dict": raw_model.fusion.state_dict(), 
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "loss": loss,
            "best_loss": self.best_loss,
        }
        self.ckpt_manager.save(state, epoch, loss, is_best, self.best_loss)

    def evaluate_test(self):
        """Run evaluation on the test set"""
        self.logger.info("="*30)
        self.logger.info("🧪 Running TEST Evaluation")
        self.logger.info("="*30)
        
        # 1. Ensure models are loaded (handled by _build_models or _handle_checkpoint_loading)
        if self.args.mode == "test":
             # If invoked directly, we must load the best checkpoint
             self._build_dataset()
             self._build_models()
             self._handle_checkpoint_loading()
             # If the loader wasn't built by _build_dataset (e.g. test mode logic difference?), ensure it is
             if not hasattr(self, 'test_loader') or self.test_loader is None:
                 self.train_loader, self.val_loader, self.test_loader = self.dataset_loader.load(ShapeNetDataset)

        if not hasattr(self, 'test_loader') or self.test_loader is None:
             self.logger.warning("⚠️ No test loader available. Skipping test.")
             return

        self.model.eval()

        test_loss = 0.0
        output_dir = self.exp_dir / "test_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Access unwrapped model for geometry info
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        temp_sphere = trimesh.creation.icosphere(subdivisions=self.model_cfg["explicit_branch"]["init_mesh"]["subdivisions"])

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                img, gt_pc, query_pts, query_sdf_gt = batch
                
                img = img.to(self.device, non_blocking=True)
                gt_pc = gt_pc.to(self.device, non_blocking=True)
                query_pts = query_pts.to(self.device, non_blocking=True)
                query_sdf_gt = query_sdf_gt.to(self.device, non_blocking=True)
                
                with torch.set_grad_enabled(True):
                     pred_pc_fused, pred_sdf, pred_pc_exp, feat = self.model(img, query_pts)
                
                loss_sdf = self.sdf_loss_fn(pred_sdf, query_sdf_gt)
                loss_cham_fused = chamfer_loss(pred_pc_fused, gt_pc)
                
                total_loss = loss_cham_fused + loss_sdf
                test_loss += total_loss.item()
                
                # Save first 10 batches of samples (or limit)
                if i < 10:
                    for j in range(min(4, img.shape[0])):
                        sample_idx = i * self.test_loader.batch_size + j
                        
                        # Save Fused
                        v_fused = pred_pc_fused[j].cpu().numpy()
                        mesh_fused = trimesh.Trimesh(vertices=v_fused, faces=temp_sphere.faces)
                        mesh_fused.export(output_dir / f"test_{sample_idx}_fused.obj")
                        
        avg_test_loss = test_loss / len(self.test_loader)
        self.logger.info(f"✅ Test Evaluation Complete | Avg Loss: {avg_test_loss:.6f}")
        self.metrics_logger.log_epoch("test", {"test_loss": avg_test_loss})
