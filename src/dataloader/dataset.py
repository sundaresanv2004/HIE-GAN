import os
import random
from PIL import Image
import open3d as o3d
import torch
from torch.utils.data import Dataset


class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # scan all object folders across classes
        self.object_paths = []
        for cls in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path): continue
            for obj in os.listdir(cls_path):
                obj_path = os.path.join(cls_path, obj)
                if os.path.isdir(obj_path):
                    self.object_paths.append(obj_path)

    def __len__(self):
        return len(self.object_paths)

    def __getitem__(self, idx):
        obj_path = self.object_paths[idx]

        # -------- image ----------
        img_dir = os.path.join(obj_path, "images")
        img_files = os.listdir(img_dir)
        img_file = random.choice(img_files)
        img = Image.open(os.path.join(img_dir, img_file)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # -------- GT POINT CLOUD ----------
        ply_file = os.path.join(obj_path, "model_normalized.ply")
        cloud = o3d.io.read_point_cloud(ply_file)
        pc = torch.tensor(np.asarray(cloud.points), dtype=torch.float32)

        return img, pc
