import os
import random
import numpy as np
import torch
from PIL import Image
import open3d as o3d
from torch.utils.data import Dataset
from torchvision import transforms


class ShapeNetDataset(Dataset):
    def __init__(self, root_dir, classes, pc_filename, image_size):
        self.root_dir = root_dir
        self.pc_filename = pc_filename
        self.image_size = image_size

        # torchvision transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

        # gather all object paths from selected classes
        self.object_paths = []
        for cls in classes:
            cpath = os.path.join(root_dir, cls)
            if not os.path.exists(cpath):
                continue
            for obj in os.listdir(cpath):
                objpath = os.path.join(cpath, obj)
                if os.path.isdir(objpath):
                    self.object_paths.append(objpath)

    def __len__(self):
        return len(self.object_paths)

    def __getitem__(self, idx):
        obj_path = self.object_paths[idx]

        # load random single image
        img_dir = os.path.join(obj_path, "images")
        img_file = random.choice(os.listdir(img_dir))
        img = Image.open(os.path.join(img_dir, img_file)).convert("RGB")
        img = self.transform(img)

        # load GT point cloud
        ply_file = os.path.join(obj_path, self.pc_filename)
        cloud = o3d.io.read_point_cloud(ply_file)
        gt = torch.tensor(np.asarray(cloud.points), dtype=torch.float32)

        return img, gt
