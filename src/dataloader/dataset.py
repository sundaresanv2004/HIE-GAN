"""
Dataset loading utilities with detailed statistics and visualization
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, ConcatDataset
from pathlib import Path
import trimesh

class DatasetLoader:
    """Handles dataset loading with detailed logging and statistics"""

    def __init__(self, dataset_cfg, train_cfg, args, logger):
        self.dataset_cfg = dataset_cfg
        self.train_cfg = train_cfg
        self.args = args
        self.logger = logger

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset_stats = {}

    def load(self, dataset_class):
        """Load dataset with comprehensive logging"""
        self.logger.info("=" * 70)
        self.logger.info("📊 Loading Dataset")
        self.logger.info("=" * 70)

        # Determine if we have explicit train/val/test folders
        root_dir = Path(self.dataset_cfg["root_dir"])
        
        # Validate root directory exists
        if not root_dir.exists():
            raise FileNotFoundError(
                f"Dataset root directory not found: {root_dir}\n"
                f"Please check your --data-root path or configs/dataset.yaml"
            )
        
        self.logger.info(f"📁 Dataset root: {root_dir}")
        
        train_dir = root_dir / "train"
        val_dir = root_dir / "val"
        
        explicit_splits = train_dir.exists() and val_dir.exists()
        test_dataset = None
        
        if explicit_splits:
            self.logger.info("✓ Found explicit train/val directories")
            train_dataset = self._create_dataset(dataset_class, root_dir=train_dir, split="train")
            val_dataset = self._create_dataset(dataset_class, root_dir=val_dir, split="val")
            
            # Only check for test split if user hasn't disabled testing
            if not self.args.no_test:
                test_dir = root_dir / "test"
                if test_dir.exists():
                    self.logger.info("✓ Found explicit test directory")
                    test_dataset = self._create_dataset(dataset_class, root_dir=test_dir, split="test")
                else:
                    self.logger.info("ℹ No explicit test directory found (this is OK with --no-test flag)")
            else:
                self.logger.info("⏭️  Skipping test directory (--no-test flag set)")
        else:
            self.logger.info("⚠ No explicit splits found, loading from root and splitting randomly")
            full_dataset = self._create_dataset(dataset_class, root_dir=root_dir, split="all")
            
            # Validate dataset is not empty
            if len(full_dataset) == 0:
                raise ValueError(
                    f"No objects found in dataset!\n"
                    f"Root directory: {root_dir}\n\n"
                    f"Expected structure:\n"
                    f"  {root_dir}/\n"
                    f"    <class_id>/\n"
                    f"      <object_id>/\n"
                    f"        images/\n"
                    f"          *.png or *.jpg\n"
                    f"        model_normalized.ply\n\n"
                    f"OR with explicit splits:\n"
                    f"  {root_dir}/\n"
                    f"    train/\n"
                    f"      <class_id>/<object_id>/...\n"
                    f"    val/\n"
                    f"      <class_id>/<object_id>/...\n\n"
                    f"Classes configured: {self.dataset_cfg['classes']}"
                )
            
            train_dataset, val_dataset = self._split_dataset(full_dataset)
        
        # Validate train dataset is not empty
        if len(train_dataset) == 0:
            raise ValueError(
                f"Train dataset is empty!\n"
                f"Please check your data path and structure.\n"
                f"Root: {root_dir}\n"
                f"Classes: {self.dataset_cfg['classes']}"
            )
        
        # Analyze datasets
        self._analyze_dataset(train_dataset, "Train")
        if val_dataset:
            self._analyze_dataset(val_dataset, "Val")
        if test_dataset:
            self._analyze_dataset(test_dataset, "Test")

        # Create dataloaders
        self._create_dataloaders(train_dataset, val_dataset, test_dataset)

        # Log summary
        self._log_summary()

        self.logger.info("=" * 70)

        if test_dataset:
             return self.train_loader, self.val_loader, self.test_loader
        else:
             return self.train_loader, self.val_loader, None

    def _create_dataset(self, dataset_class, root_dir, split):
        """Create dataset instance"""
        self.logger.info(f"📁 Loading {split} dataset from: {root_dir}")
        
        dataset = dataset_class(
            root_dir=root_dir,
            classes=self.dataset_cfg["classes"],
            pc_filename=self.dataset_cfg["pointcloud"]["filename"],
            image_size=self.dataset_cfg["image"]["size"],
            num_sdf_samples=self.dataset_cfg.get("sdf", {}).get("num_samples", 2048),
        )
        return dataset

    def _analyze_dataset(self, dataset, name):
        """Analyze dataset and gather statistics"""
        total_objects = len(dataset)
        self.logger.info(f"📊 {name} objects: {total_objects}")
        
        self.dataset_stats[f'{name}_size'] = total_objects

        # Limit dataset if requested
        if self.args.debug:
            dataset.object_paths = dataset.object_paths[:32]
            self.logger.info(f"⚠️  DEBUG: Limited {name} to {len(dataset)} samples")
        elif self.args.num_samples:
              # For explicit splits, we might want to limit total, but here we limit per split roughly
            dataset.object_paths = dataset.object_paths[:self.args.num_samples]
            self.logger.info(f"⚠️  Limited {name} to {len(dataset)} samples")

        # Sample one batch for analysis on Train only
        if name == "Train" and self.args.inspect_data:
            self._inspect_sample(dataset)

    def _inspect_sample(self, dataset):
        """Inspect a single sample from dataset"""
        self.logger.info("🔍 Inspecting sample data...")

        try:
            # Unpack assuming 5 items now
            sample = dataset[0]
            if len(sample) == 5:
                img, pc, query_pts, query_sdf, class_idx = sample
                self.logger.info(f"   Image: {img.shape} {img.dtype}")
                self.logger.info(f"   PC: {pc.shape} {pc.dtype}")
                self.logger.info(f"   Query Pts: {query_pts.shape} {query_pts.dtype}")
                self.logger.info(f"   Query SDF: {query_sdf.shape} {query_sdf.dtype}")
                self.logger.info(f"   Class Index: {class_idx}")
                self.logger.info(f"   SDF Range: {query_sdf.min():.4f} to {query_sdf.max():.4f}")
            elif len(sample) == 4:
                # Fallback for old style if needed, though we just changed getitem
                 img, pc, query_pts, query_sdf = sample
                 self.logger.info(f"   Got 4 items (Legacy)")
            else:
                self.logger.info(f"   Got {len(sample)} items in sample")

        except Exception as e:
            self.logger.warning(f"⚠️  Failed to inspect sample: {e}")

    def _split_dataset(self, dataset):
        """Split dataset into train/val (Legacy fallback)"""
        if self.args.val_split and self.args.val_split > 0:
            val_size = int(len(dataset) * self.args.val_split)
            train_size = len(dataset) - val_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.args.seed or 42)
            )
            return train_dataset, val_dataset
        else:
            return dataset, None

    def _create_dataloaders(self, train_dataset, val_dataset, test_dataset=None):
        """Create train, validation, and test dataloaders"""
        batch_size = self.train_cfg["batch_size"]
        num_workers = self.train_cfg["num_workers"]
        
        pin_memory = self.args.pin_memory if self.args.pin_memory is not None else torch.cuda.is_available()

        self.logger.info(f"🔢 Batch size: {batch_size}")
        self.logger.info(f"👷 Workers: {num_workers}")
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
            )
            self.dataset_stats['val_batches'] = len(self.val_loader)
            
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False, 
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
            )
            self.dataset_stats['test_batches'] = len(self.test_loader)

    def _log_summary(self):
        """Log dataset summary"""
        self.logger.info("📋 Dataset Summary:")
        if self.train_loader:
            self.logger.info(f"   Train batches: {len(self.train_loader)}")
        if self.val_loader:
             self.logger.info(f"   Val batches: {len(self.val_loader)}")
        if self.test_loader is not None:
             self.logger.info(f"   Test batches: {len(self.test_loader)}")


class ShapeNetDataset(torch.utils.data.Dataset):
    """
    ShapeNet Dataset for Phase 1 & 2
    Phase 2 Adds: SDF sampling
    """


    def __init__(self, root_dir, classes, pc_filename="model_normalized.ply", image_size=224, num_points=2500, num_sdf_samples=2048):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.pc_filename = pc_filename
        self.image_size = image_size
        self.num_points = num_points
        self.num_sdf_samples = num_sdf_samples
        self.object_paths = [] # List of tuples (path, class_idx)

        # Collecting all object paths
        if not self.root_dir.exists():
            print(f"⚠️  Dataset root not found: {self.root_dir}")
            return

        for i, class_id in enumerate(self.classes):
            class_dir = self.root_dir / class_id
            if not class_dir.exists():
                print(f"⚠️  Class directory not found: {class_dir} (class_id: {class_id})")
                continue
            
            objects_found = 0
            for obj_name in os.listdir(class_dir):
                obj_path = class_dir / obj_name
                if obj_path.is_dir():
                    has_images = (obj_path / "images").exists()
                    has_pc = (obj_path / self.pc_filename).exists()
                    
                    if has_images and has_pc:
                        self.object_paths.append((str(obj_path), i))
                        objects_found += 1
                    elif not has_images:
                        print(f"⚠️  Missing 'images' folder in: {obj_path.name}")
                    elif not has_pc:
                        print(f"⚠️  Missing '{self.pc_filename}' in: {obj_path.name}")
            
            print(f"✓ Found {objects_found} objects for class {class_id}")

        # Standard ImageNet normalization
        from torchvision import transforms
        from PIL import Image
        self.transform = transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.object_paths)

    def __getitem__(self, idx):
        obj_path_str, class_idx = self.object_paths[idx]
        obj_path = Path(obj_path_str)

        # 1. Load Image
        image_tensor = self._load_image(obj_path)

        # 2. Load Mesh & Point Cloud
        pc_path = obj_path / self.pc_filename
        gt_pc, mesh = self._load_mesh_and_pc(pc_path)

        # 3. Compute SDF Samples (Phase 2)
        query_points, query_sdf = self._sample_sdf(mesh)

        return image_tensor, gt_pc, query_points, query_sdf, class_idx

    def _load_image(self, obj_path):
        img_dir = obj_path / "images"
        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        
        if not image_files:
            return torch.zeros(3, self.image_size, self.image_size)
            
        choice_idx = np.random.randint(len(image_files))
        img_path = image_files[choice_idx]
        
        from PIL import Image
        try:
            image = Image.open(img_path).convert("RGB")
            return self.transform(image)
        except Exception:
            return torch.zeros(3, self.image_size, self.image_size)

    def _load_mesh_and_pc(self, pc_path):
        try:
            mesh = trimesh.load(pc_path, process=False) # Keep original
            
            # Sample surface points for Chamfer Loss
            if hasattr(mesh, 'faces') and len(mesh.faces) > 0:
                points, _ = trimesh.sample.sample_surface(mesh, self.num_points)
            else:
                 # Fallback if just point cloud
                if hasattr(mesh, 'vertices'):
                     points = np.array(mesh.vertices)
                else:
                     points = np.zeros((self.num_points, 3))
                     
            # Resample
            if len(points) != self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=True)
                points = points[indices]
                
            return torch.from_numpy(points).float(), mesh
            
        except Exception:
            return torch.zeros(self.num_points, 3), None

    def _sample_sdf(self, mesh):
        """
        Sample points near surface and compute SDF.
        Returns:
            points: (N_samples, 3)
            sdf: (N_samples, 1)
        """
        if mesh is None or not hasattr(mesh, 'faces') or len(mesh.faces) == 0:
            # Fallback if no valid mesh for SDF
            return torch.zeros(self.num_sdf_samples, 3), torch.zeros(self.num_sdf_samples, 1)

        # 1. Surface points (for near-surface sampling)
        surface_points, _ = trimesh.sample.sample_surface(mesh, self.num_sdf_samples)
        
        # 2. Add noise to surface points to get near-surface points
        # Strategy: 90% near surface, 10% uniform
        n_near = int(0.9 * self.num_sdf_samples)
        n_uniform = self.num_sdf_samples - n_near
        
        # Gaussian noise sigma=0.01 for near surface
        near_points = surface_points[:n_near] + np.random.normal(0, 0.01, (n_near, 3))
        
        # Uniform points in [-0.5, 0.5] (assuming normalized mesh)
        uniform_points = np.random.uniform(-0.5, 0.5, (n_uniform, 3))
        
        query_points = np.vstack([near_points, uniform_points])
        
        # 3. Compute Signed Distance
        # trimesh.proximity.signed_distance returns - inside, + outside?
        # Check docs or convention. Usually Trimesh: positive outside, negative inside.
        sdf = trimesh.proximity.signed_distance(mesh, query_points)
        
        # Convert to tensor
        # Shape (N, 1)
        query_points = torch.from_numpy(query_points).float()
        sdf = torch.from_numpy(sdf).float().unsqueeze(1)
        
        return query_points, sdf
