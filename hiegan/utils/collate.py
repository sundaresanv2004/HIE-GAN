from typing import List, Tuple
import torch
from pytorch3d.structures import Meshes
from .mesh_utils import load_and_normalize_mesh


class MeshCollator:
    """
    Class-based collator that is robust to pickling errors in multiprocessing.
    """

    def __init__(self, neural_device: str = "cpu", mesh_device: str = "cpu"):
        self.neural_device = neural_device  # For neural network operations
        self.mesh_device = mesh_device  # For PyTorch3D operations

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

                # Load mesh on mesh_device (CPU for PyTorch3D compatibility)
                mesh = load_and_normalize_mesh(mesh_path, device=self.mesh_device)
                all_meshes_verts.append(mesh.verts_list()[0])
                all_meshes_faces.append(mesh.faces_list()[0])
                all_metadata.append(metadata)

            except Exception as e:
                print(f"Error processing sample in batch: {e}")
                continue

        if not all_imgs:
            raise ValueError("No valid samples in batch")

        # Images go to neural device (MPS/CUDA for neural networks)
        batched_imgs = torch.cat(all_imgs, dim=0).to(self.neural_device)

        # Meshes stay on mesh device (CPU for PyTorch3D compatibility)
        batched_meshes = Meshes(
            verts=all_meshes_verts,
            faces=all_meshes_faces
        ).to(self.mesh_device)

        return batched_imgs, batched_meshes, all_metadata
