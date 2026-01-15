import os
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
from models.init_mesh import create_sphere_mesh
from losses.chamfer import chamfer_loss
from losses.sdf_loss import SDFLoss
from losses.regularizers import smoothness_loss, edge_length_loss
from utils.mesh_ops import generate_mesh_from_sdf

from utils.checkpoint import CheckpointManager
from utils.setup import EnvironmentSetup
from utils.plotter import plot_training_logs
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
        self.scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
        
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
        self.optimizer = None
        self.sdf_loss_fn = None
        
        self.start_epoch = 0
        self.best_loss = float("inf")

    def _load_configs(self):
        return load_configs(config_dir=self.args.config_dir)

    def _apply_overrides(self):
        # Dataset path override
        if self.args.data_root:
            self.dataset_cfg["root_dir"] = self.args.data_root
        
        # Training config overrides
        if self.args.epochs:
            self.train_cfg["epochs"] = int(self.args.epochs)
        if self.args.batch_size:
            self.train_cfg["batch_size"] = int(self.args.batch_size)
        if self.args.lr:
            self.train_cfg["optimizer"]["lr"] = float(self.args.lr)
        else:
            self.train_cfg["optimizer"]["lr"] = float(self.train_cfg["optimizer"]["lr"])
        if self.args.num_workers is not None:
            self.train_cfg["num_workers"] = int(self.args.num_workers)

    def _setup_logging(self):
        logger = setup_logger(self.exp_dir, self.train_cfg["logging"]["log_filename"], quiet=self.args.quiet)
        csv_logger = CSVLogger(self.exp_dir, self.train_cfg["logging"]["csv_filename"])
        metrics_logger = MetricsLogger(self.exp_dir, "metrics.json")
        return logger, csv_logger, metrics_logger

    def _log_config(self):
        if self.args.quiet: return
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

        # Count parameters
        total_params = sum(p.numel() for p in self.encoder.parameters()) + \
                       sum(p.numel() for p in self.explicit.parameters()) + \
                       sum(p.numel() for p in self.implicit.parameters()) + \
                       sum(p.numel() for p in self.fusion.parameters())

        self.logger.info(f"Total parameters: {total_params:,}")

        if self.args.compile:
             self.encoder = torch.compile(self.encoder)
             self.explicit = torch.compile(self.explicit)
             self.implicit = torch.compile(self.implicit)
             # Fusion usually doesn't need compile as it's just ops, but we can try
             self.fusion = torch.compile(self.fusion)

    def _build_optimizer(self):
        lr = float(self.train_cfg["optimizer"]["lr"])
        weight_decay = float(self.args.weight_decay) if self.args.weight_decay else 0.0

        print_params = list(self.encoder.parameters()) + \
                       list(self.explicit.parameters()) + \
                       list(self.implicit.parameters()) + \
                       list(self.fusion.parameters())

        self.optimizer = torch.optim.Adam(
            print_params,
            lr=lr,
            weight_decay=weight_decay
        )
        self.logger.info(f"Optimizer: Adam (lr={lr})")

    def _handle_checkpoint_loading(self):
        """Handle checkpoint loading"""
        checkpoint = None
        if self.args.checkpoint:
            # Load specific
            ckpt_path = Path(self.args.checkpoint)
            if ckpt_path.is_dir(): ckpt_path = ckpt_path / "checkpoint_latest.pth"
            checkpoint = self.ckpt_manager.load(ckpt_path, self.device)
        elif self.args.mode == "resume":
            latest_ckpt = self.ckpt_dir / "checkpoint_latest.pth"
            if latest_ckpt.exists():
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
        self.encoder.train()
        self.explicit.train()
        self.implicit.train()
        self.fusion.train()

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
            with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
                # 1. Feature Extraction
                feat = self.encoder(img) # (B, C)
                
                # 2. Explicit Branch
                pred_pc_exp = self.explicit(feat)
                
                # 3. Implicit Branch (SDF Loss)
                pred_sdf = self.implicit(feat, query_pts)
                loss_sdf = self.sdf_loss_fn(pred_sdf, query_sdf_gt)

                # 4. Fusion Module
                pred_pc_fused = self.fusion(pred_pc_exp, self.implicit, feat)
                
                # 5. Losses
                pred_mesh_exp = (pred_pc_exp, self.explicit.E)
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
                    torch.nn.utils.clip_grad_norm_(self.optimizer, self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                if self.args.grad_clip:
                     torch.nn.utils.clip_grad_norm_(self.optimizer, self.args.grad_clip)
                self.optimizer.step()

            loss_val = total_loss.item()
            epoch_loss += loss_val
            
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
        self.encoder.eval()
        self.explicit.eval()
        self.implicit.eval()
        self.fusion.eval()
        
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
                
                feat = self.encoder(img)
                pred_pc_exp = self.explicit(feat)
                pred_sdf = self.implicit(feat, query_pts)
                
                # Fusion requires gradients for SDF normals, so we must enable grad temporarily
                with torch.enable_grad():
                     pred_pc_fused = self.fusion(pred_pc_exp, self.implicit, feat)
                
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
                            mesh_imp = generate_mesh_from_sdf(self.implicit, feat[j:j+1], resolution=64, device=self.device)
                            if mesh_imp:
                                mesh_imp.export(val_out_dir / f"sample_{j}_implicit.obj")
                                
        avg_val_loss = val_loss / len(self.val_loader)
        self.logger.info(f"Validation Loss: {avg_val_loss:.6f}")
        self.metrics_logger.log_epoch(epoch, {"val_loss": avg_val_loss})
        return avg_val_loss

    def evaluate_test(self):
        """Run evaluation on the test set after training completes"""
        self.logger.info("Running evaluation on TEST set...")
        
        # Ensure test loader exists
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            # Try to load it if not available
             dataset_cfg, _, _ = load_configs(self.args.config_dir)
             if self.args.dataset_config: dataset_cfg = load_yaml(self.args.dataset_config)
             
             # Manually trigger test loader loading if dataset class supports it
             # Current implementation of DatasetLoader.load needs update to return test_loader
             # checking if we can get it from self.dataset_loader logic
             pass
             
        # For now, let's assume if it wasn't loaded in .load(), we can't easily get it without refactor.
        # But wait, we saw "Found explicit test directory" in logs.
        # Let's update DatasetLoader.load to return it first, or access it from here.
        # Actually, let's check self.dataset_loader.load return.
        pass


    def run(self):
        if self.args.dry_run:
            self.logger.info("🔍 Dry Run Mode")
            self._build_dataset()
            self._build_models()
            self._build_optimizer()
            # Test validation output logic in dry run
            self.train_loader, self.val_loader, self.test_loader = self.dataset_loader.load(ShapeNetDataset)
            self.validate(epoch=-1) # Dummy validation
            return

        self._build_dataset()
        self._build_models()
        self._build_optimizer()
        self._handle_checkpoint_loading()

        self.logger.info("Starting training...")
        
        try:
            for epoch in range(self.start_epoch, self.train_cfg["epochs"]):
                avg_loss = self.train_epoch(epoch)
                
                self.logger.info(f"Epoch {epoch+1} | Train Loss: {avg_loss:.6f}")
                self.metrics_logger.log_epoch(epoch, {"train_loss": avg_loss})
                
                # Validate every N epochs
                val_every = self.train_cfg.get("logging", {}).get("val_every", 1)
                
                avg_val_loss = avg_loss # Fallback if not validating
                
                if (epoch + 1) % val_every == 0:
                     avg_val_loss = self.validate(epoch)

                is_best = avg_val_loss < self.best_loss
                if is_best: self.best_loss = avg_val_loss
                
                save_every = self.train_cfg["checkpoints"].get("save_every", 1)
                if (epoch + 1) % save_every == 0 or is_best:
                    if not self.args.no_save:
                        self.save_checkpoint(epoch, avg_val_loss, is_best)


        except KeyboardInterrupt:
            self.logger.warning("Interrupted!")
            self.save_checkpoint(epoch, avg_loss, False)
            
        # Automatic Graph Generation
        if self.args.generate_graph:
            self.logger.info("📊 Generating training graphs...")
            from utils.plotter import plot_training_graphs
            success = plot_training_graphs(self.exp_dir)
            if success:
                self.logger.info("✅ Training graphs generated successfully")
            
        # Automatic Testing after training
        if not self.args.no_test:
            self.evaluate_test()

    def save_checkpoint(self, epoch, loss, is_best):
        state = {
            "epoch": epoch,
            "encoder_state_dict": self.encoder.state_dict(),
            "explicit_state_dict": self.explicit.state_dict(),
            "implicit_state_dict": self.implicit.state_dict(),
            "fusion_state_dict": self.fusion.state_dict(), # Phase 3
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

        self.encoder.eval()
        self.explicit.eval()
        self.implicit.eval()
        self.fusion.eval()

        test_loss = 0.0
        output_dir = self.exp_dir / "test_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temp_sphere = trimesh.creation.icosphere(subdivisions=self.model_cfg["explicit_branch"]["init_mesh"]["subdivisions"])

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader, desc="Testing")):
                img, gt_pc, query_pts, query_sdf_gt = batch
                
                img = img.to(self.device, non_blocking=True)
                gt_pc = gt_pc.to(self.device, non_blocking=True)
                query_pts = query_pts.to(self.device, non_blocking=True)
                query_sdf_gt = query_sdf_gt.to(self.device, non_blocking=True)
                
                feat = self.encoder(img)
                pred_pc_exp = self.explicit(feat)
                pred_sdf = self.implicit(feat, query_pts)
                
                with torch.enable_grad():
                     pred_pc_fused = self.fusion(pred_pc_exp, self.implicit, feat)
                
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
