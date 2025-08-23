from typing import List, Tuple, Dict
import torch
from pytorch3d.structures import Meshes
from .mesh_utils import load_and_normalize_mesh


def mesh_collate_fn(batch: List[Tuple[torch.Tensor, str]], device: str = "cpu") -> Tuple[torch.Tensor, Meshes]:
    """
    Custom collate function to load meshes in parallel with DataLoader workers.

    Args:
        batch (List[Tuple[torch.Tensor, str]]): A list of samples from the Dataset,
            where each sample is a tuple of (image_tensor, mesh_path).
        device (str): The device to move the final tensors to.

    Returns:
        Tuple[torch.Tensor, Meshes]: A tuple containing:
            - A batched tensor of images of shape (B * V, 3, H, W).
            - A PyTorch3D Meshes object representing the batch of loaded meshes.
    """
    all_imgs = []
    all_meshes_verts = []
    all_meshes_faces = []

    for imgs_tensor, mesh_path in batch:
        all_imgs.append(imgs_tensor)

        # 1. Load the mesh to the CPU within the worker process.
        mesh = load_and_normalize_mesh(mesh_path, device="cpu")

        all_meshes_verts.append(mesh.verts_list()[0])
        all_meshes_faces.append(mesh.faces_list()[0])

    # Batched tensors are created on the CPU
    batched_imgs = torch.cat(all_imgs, dim=0)
    batched_meshes_cpu = Meshes(
        verts=all_meshes_verts,
        faces=all_meshes_faces
    )

    # 2. Move the entire batch to the target device in one go.
    # This happens in the main process after the workers are done.
    return batched_imgs.to(device), batched_meshes_cpu.to(device)
