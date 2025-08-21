import os
import random
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _list_dir_safe(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return [d for d in os.listdir(path) if not d.startswith(".")]

class ShapeNetMVDataset(Dataset):
    """
    Expects:
        dataset_root/class_id/object_id/
            images/   -> PNG images (multi-view)
            mesh.obj  -> GT mesh
    Returns:
        (imgs_tensor, mesh_path)
        imgs_tensor shape: (V, 3, H, W) where V=1 for single-view
    """

    def __init__(self, dataset_root: str, image_size: int = 224, multi_view: bool = False, transform=None):
        self.dataset_root = dataset_root
        self.multi_view = multi_view
        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.items: List[dict] = []
        for class_id in _list_dir_safe(dataset_root):
            cpath = os.path.join(dataset_root, class_id)
            for obj_id in _list_dir_safe(cpath):
                obj_path = os.path.join(cpath, obj_id)
                img_dir = os.path.join(obj_path, "images")
                mesh_path = os.path.join(obj_path, "model_normalized.obj")

                if not os.path.isfile(mesh_path) or not os.path.isdir(img_dir):
                    print("not exist", mesh_path)
                    continue

                image_files = sorted([
                    os.path.join(img_dir, f) for f in os.listdir(img_dir)
                    if f.lower().endswith(".png")
                ])
                if len(image_files) == 0:
                    continue
                self.items.append({
                    "images": image_files,
                    "mesh": mesh_path
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        entry = self.items[idx]
        imgs = entry["images"]
        mesh_path = entry["mesh"]

        if self.multi_view:
            pil_imgs = [Image.open(p).convert("RGB") for p in imgs]
        else:
            pil_imgs = [Image.open(random.choice(imgs)).convert("RGB")]

        tens = [self.transform(im) for im in pil_imgs]
        imgs_tensor = torch.stack(tens, dim=0)  # (V, 3, H, W)
        return imgs_tensor, mesh_path
