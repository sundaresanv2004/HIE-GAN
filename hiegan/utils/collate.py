from typing import List, Tuple
import torch
from pytorch3d.structures import Meshes
from .mesh_utils import load_and_normalize_mesh


class MeshCollator:
    """
    Class-based collator that is robust to pickling errors in multiprocessing.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

    def __call__(self, batch: List[Tuple[torch.Tensor, str, dict]]) -> Tuple[torch.Tensor, Meshes, List[dict]]:
        """
        Processes a batch of samples.
        batch: List of (imgs_tensor, mesh_path, metadata) tuples
        """
        all_imgs = []
        all_meshes_verts = []
        all_meshes_faces = []
        all_metadata = []

        for imgs_tensor, mesh_path, metadata in batch:
            try:
                all_imgs.append(imgs_tensor)

                # Load mesh on CPU first to avoid memory issues
                mesh = load_and_normalize_mesh(mesh_path, device="cpu")
                all_meshes_verts.append(mesh.verts_list()[0])
                all_meshes_faces.append(mesh.faces_list()[0])
                all_metadata.append(metadata)

            except Exception as e:
                print(f"Error processing sample in batch: {e}")
                continue

        if not all_imgs:
            raise ValueError("No valid samples in batch")

        # Combine images: (B*V, C, H, W) where V is views per item
        batched_imgs = torch.cat(all_imgs, dim=0).to(self.device)

        # Create batched meshes
        batched_meshes = Meshes(
            verts=all_meshes_verts,
            faces=all_meshes_faces
        ).to(self.device)

        return batched_imgs, batched_meshes, all_metadata
