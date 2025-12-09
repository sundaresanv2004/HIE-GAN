"""
Dataset loading utilities with detailed statistics and visualization
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from pathlib import Path


class DatasetLoader:
    """Handles dataset loading with detailed logging and statistics"""

    def __init__(self, dataset_cfg, train_cfg, args, logger):
        self.dataset_cfg = dataset_cfg
        self.train_cfg = train_cfg
        self.args = args
        self.logger = logger

        self.train_loader = None
        self.val_loader = None
        self.dataset_stats = {}

    def load(self, dataset_class):
        """Load dataset with comprehensive logging"""
        self.logger.info("=" * 70)
        self.logger.info("📊 Loading Dataset")
        self.logger.info("=" * 70)

        # Create dataset
        dataset = self._create_dataset(dataset_class)

        # Analyze dataset
        self._analyze_dataset(dataset)

        # Split train/val
        train_dataset, val_dataset = self._split_dataset(dataset)

        # Create dataloaders
        self._create_dataloaders(train_dataset, val_dataset)

        # Log summary
        self._log_summary()

        self.logger.info("=" * 70)

        return self.train_loader, self.val_loader

    def _create_dataset(self, dataset_class):
        """Create dataset instance"""
        self.logger.info(f"📁 Dataset root: {self.dataset_cfg['root_dir']}")
        self.logger.info(f"📦 Classes: {', '.join(self.dataset_cfg['classes'])}")
        self.logger.info(f"🖼️  Image size: {self.dataset_cfg['image']['size']}x{self.dataset_cfg['image']['size']}")
        self.logger.info(f"☁️  Point cloud file: {self.dataset_cfg['pointcloud']['filename']}")

        dataset = dataset_class(
            root_dir=self.dataset_cfg["root_dir"],
            classes=self.dataset_cfg["classes"],
            pc_filename=self.dataset_cfg["pointcloud"]["filename"],
            image_size=self.dataset_cfg["image"]["size"],
        )

        return dataset

    def _analyze_dataset(self, dataset):
        """Analyze dataset and gather statistics"""
        total_objects = len(dataset)
        self.logger.info(f"📊 Total objects found: {total_objects}")

        # Store stats
        self.dataset_stats['total_objects'] = total_objects
        self.dataset_stats['classes'] = self.dataset_cfg['classes']

        # Count objects per class
        class_counts = {}
        for obj_path in dataset.object_paths:
            class_name = Path(obj_path).parent.name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        self.logger.info("📈 Objects per class:")
        for cls, count in sorted(class_counts.items()):
            self.logger.info(f"   {cls}: {count}")

        self.dataset_stats['class_counts'] = class_counts

        # Limit dataset if requested
        if self.args.debug:
            dataset.object_paths = dataset.object_paths[:32]
            self.logger.info(f"⚠️  DEBUG: Limited to {len(dataset)} samples")
        elif self.args.num_samples:
            dataset.object_paths = dataset.object_paths[:self.args.num_samples]
            self.logger.info(f"⚠️  Limited to {len(dataset)} samples")

        self.dataset_stats['used_objects'] = len(dataset)

        # Sample one batch for analysis
        if self.args.inspect_data:
            self._inspect_sample(dataset)

    def _inspect_sample(self, dataset):
        """Inspect a single sample from dataset"""
        self.logger.info("🔍 Inspecting sample data...")

        try:
            img, pc = dataset[0]

            self.logger.info(f"   Image shape: {img.shape}")
            self.logger.info(f"   Image dtype: {img.dtype}")
            self.logger.info(f"   Image range: [{img.min():.3f}, {img.max():.3f}]")
            self.logger.info(f"   Point cloud shape: {pc.shape}")
            self.logger.info(f"   Point cloud dtype: {pc.dtype}")
            self.logger.info(f"   PC range: X=[{pc[:, 0].min():.3f}, {pc[:, 0].max():.3f}], "
                             f"Y=[{pc[:, 1].min():.3f}, {pc[:, 1].max():.3f}], "
                             f"Z=[{pc[:, 2].min():.3f}, {pc[:, 2].max():.3f}]")

            self.dataset_stats['sample_info'] = {
                'image_shape': tuple(img.shape),
                'pc_shape': tuple(pc.shape),
                'pc_num_points': pc.shape[0],
            }

        except Exception as e:
            self.logger.warning(f"⚠️  Failed to inspect sample: {e}")

    def _split_dataset(self, dataset):
        """Split dataset into train/val"""
        if self.args.val_split and self.args.val_split > 0:
            val_size = int(len(dataset) * self.args.val_split)
            train_size = len(dataset) - val_size

            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(self.args.seed or 42)
            )

            self.logger.info(f"📊 Split - Train: {len(train_dataset)} | Val: {len(val_dataset)} "
                             f"(ratio: {self.args.val_split:.1%})")

            self.dataset_stats['train_size'] = len(train_dataset)
            self.dataset_stats['val_size'] = len(val_dataset)
        else:
            train_dataset = dataset
            val_dataset = None
            self.logger.info("📊 No validation split")
            self.dataset_stats['train_size'] = len(train_dataset)
            self.dataset_stats['val_size'] = 0

        return train_dataset, val_dataset

    def _create_dataloaders(self, train_dataset, val_dataset):
        """Create train and validation dataloaders"""
        batch_size = self.train_cfg["batch_size"]
        num_workers = self.train_cfg["num_workers"]

        # Determine pin_memory
        if self.args.pin_memory is not None:
            pin_memory = self.args.pin_memory
        else:
            pin_memory = torch.cuda.is_available()

        # Calculate batches
        train_batches = len(train_dataset) // batch_size
        if len(train_dataset) % batch_size != 0:
            train_batches += 1

        self.logger.info(f"🔢 Batch size: {batch_size}")
        self.logger.info(f"👷 Workers: {num_workers}")
        self.logger.info(f"📦 Train batches per epoch: {train_batches}")
        self.logger.info(f"📊 Train samples per epoch: {len(train_dataset)}")

        self.dataset_stats['batch_size'] = batch_size
        self.dataset_stats['train_batches'] = train_batches

        # Warning if very few batches
        if train_batches < 10:
            self.logger.warning(f"⚠️  Only {train_batches} batches per epoch!")
            self.logger.warning(f"⚠️  Consider:")
            self.logger.warning(f"    - Decreasing batch size (current: {batch_size})")
            self.logger.warning(f"    - Adding more data classes")
            self.logger.warning(f"    - Increasing dataset size")

        # Create train loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            drop_last=False,
        )

        # Create val loader if needed
        if val_dataset:
            val_batches = len(val_dataset) // batch_size
            if len(val_dataset) % batch_size != 0:
                val_batches += 1

            self.logger.info(f"📦 Val batches per epoch: {val_batches}")
            self.dataset_stats['val_batches'] = val_batches

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
                drop_last=False,
            )

    def _log_summary(self):
        """Log dataset summary"""
        self.logger.info("📋 Dataset Summary:")
        self.logger.info(f"   Total objects: {self.dataset_stats['total_objects']}")
        self.logger.info(f"   Used objects: {self.dataset_stats['used_objects']}")
        self.logger.info(f"   Train samples: {self.dataset_stats['train_size']}")
        self.logger.info(f"   Val samples: {self.dataset_stats['val_size']}")
        self.logger.info(f"   Batch size: {self.dataset_stats['batch_size']}")
        self.logger.info(f"   Train batches: {self.dataset_stats['train_batches']}")

        if 'sample_info' in self.dataset_stats:
            self.logger.info(f"   Image shape: {self.dataset_stats['sample_info']['image_shape']}")
            self.logger.info(f"   Points per cloud: {self.dataset_stats['sample_info']['pc_num_points']}")

    def get_stats(self):
        """Return dataset statistics"""
        return self.dataset_stats


class ShapeNetDataset(torch.utils.data.Dataset):
    """
    ShapeNet Dataset for Phase 1 Training
    Loads images and corresponding point clouds
    """

    def __init__(self, root_dir, classes, pc_filename="model_normalized.ply", image_size=224, num_points=2500):
        """
        Args:
            root_dir (str): Path to ShapeNet dataset root
            classes (list): List of class IDs to load
            pc_filename (str): Name of the point cloud file in each object folder
            image_size (int): Target image size
            num_points (int): Number of points to sample from ground truth
        """
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.pc_filename = pc_filename
        self.image_size = image_size
        self.num_points = num_points
        self.object_paths = []

        # Collecting all object paths
        if not self.root_dir.exists():
            print(f"Dataset root not found: {self.root_dir}")
            return

        for class_id in self.classes:
            class_dir = self.root_dir / class_id
            if not class_dir.exists():
                continue
            
            # Each subdirectory is an object
            for obj_name in os.listdir(class_dir):
                obj_path = class_dir / obj_name
                if obj_path.is_dir():
                    # Check if required files exist
                    if (obj_path / "images").exists() and (obj_path / self.pc_filename).exists():
                        self.object_paths.append(str(obj_path))

        # Standard ImageNet normalization
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.object_paths)

    def __getitem__(self, idx):
        obj_path = Path(self.object_paths[idx])

        # 1. Load Image
        # Randomly select one image from the 'images' folder
        img_dir = obj_path / "images"
        image_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        
        if not image_files:
            # Fallback if no images (shouldn't happen due to init check)
            image_tensor = torch.zeros(3, self.image_size, self.image_size)
        else:
            choice_idx = np.random.randint(len(image_files))
            img_path = image_files[choice_idx]
            
            from PIL import Image
            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = self.transform(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                image_tensor = torch.zeros(3, self.image_size, self.image_size)

        # 2. Load Point Cloud
        pc_path = obj_path / self.pc_filename
        try:
            # Use trimesh for robust loading if available, else numpy fallback
            import trimesh
            pc_mesh = trimesh.load(pc_path)
            
            # If it's a mesh, sample points
            if hasattr(pc_mesh, 'vertices'):
                 # If it is a mesh (faces), sample surface
                if hasattr(pc_mesh, 'faces') and len(pc_mesh.faces) > 0:
                    points, _ = trimesh.sample.sample_surface(pc_mesh, self.num_points)
                else:
                    # Just vertices
                    points = np.array(pc_mesh.vertices)
            else:
                # Fallback
                points = np.zeros((self.num_points, 3))

            # Resample if we have vertices but not enough or too many and didn't sample surface
            if len(points) != self.num_points:
                indices = np.random.choice(len(points), self.num_points, replace=True)
                points = points[indices]

            gt_pc = torch.from_numpy(points).float()
            
        except Exception as e:
            # print(f"Error loading PC {pc_path}: {e}")
            gt_pc = torch.zeros(self.num_points, 3)

        return image_tensor, gt_pc
