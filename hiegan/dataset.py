import os
import random
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _list_dir_safe(path: str) -> List[str]:
    """Safely list directory contents, filtering out hidden files"""
    if not os.path.isdir(path):
        return []
    try:
        return [d for d in os.listdir(path) if not d.startswith(".")]
    except OSError:
        return []


class ShapeNetMVDataset(Dataset):
    def __init__(self, dataset_root: str, image_size: int = 224, multi_view: bool = False, transform=None):
        self.dataset_root = dataset_root
        self.multi_view = multi_view
        self.image_size = image_size

        # Store transform parameters instead of the transform object
        self.transform = transform or T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.items: List[dict] = []
        self._build_dataset()
        print(f"Dataset loaded: {len(self.items)} samples")

    def _build_dataset(self):
        """Build dataset with proper error handling"""
        for class_id in _list_dir_safe(self.dataset_root):
            class_path = os.path.join(self.dataset_root, class_id)

            for obj_id in _list_dir_safe(class_path):
                obj_path = os.path.join(class_path, obj_id)
                img_dir = os.path.join(obj_path, "images")
                mesh_path = os.path.join(obj_path, "model_normalized.obj")

                # Validate paths exist
                if not os.path.isfile(mesh_path) or not os.path.isdir(img_dir):
                    continue

                # Get valid image files
                try:
                    image_files = sorted([
                        os.path.join(img_dir, f) for f in os.listdir(img_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    ])
                except OSError:
                    continue

                if len(image_files) == 0:
                    continue

                self.items.append({
                    "class_id": class_id,
                    "obj_id": obj_id,
                    "images": image_files,
                    "mesh": mesh_path
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, dict]:
        entry = self.items[idx]
        imgs = entry["images"]
        mesh_path = entry["mesh"]

        try:
            if self.multi_view:
                pil_imgs = [Image.open(p).convert("RGB") for p in imgs]
                tensors = [self.transform(im) for im in pil_imgs]
                imgs_tensor = torch.stack(tensors, dim=0)
            else:
                # Single view - random selection
                pil_img = Image.open(random.choice(imgs)).convert("RGB")
                imgs_tensor = self.transform(pil_img).unsqueeze(0)

            # Return metadata for debugging
            metadata = {
                "class_id": entry["class_id"],
                "obj_id": entry["obj_id"],
                "num_views": len(imgs)
            }

            return imgs_tensor, mesh_path, metadata

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            raise e

    def __getstate__(self):
        """Custom pickling method"""
        state = self.__dict__.copy()
        # Don't pickle the transform - recreate it in __setstate__
        state['transform'] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling method"""
        self.__dict__.update(state)
        # Recreate the transform
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
