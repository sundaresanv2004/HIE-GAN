from typing import Tuple
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes


def load_and_normalize_mesh(mesh_path: str, device: str = "cuda") -> Meshes:
    """
    Loads a single OBJ mesh and normalizes it to fit inside the unit sphere.
    Returns a PyTorch3D Meshes object on the target device.
    """
    try:
        meshes = load_objs_as_meshes([mesh_path], device=device)
        verts = meshes.verts_list()[0]  # (V, 3)
        faces = meshes.faces_list()[0]  # (F, 3)

        # Normalize: center at origin, scale to unit sphere
        center = verts.mean(dim=0, keepdim=True)
        v_centered = verts - center
        scale = v_centered.norm(dim=1).max().clamp(min=1e-6)  # max radius
        v_norm = v_centered / scale

        mesh_norm = Meshes(verts=[v_norm.to(device)], faces=[faces.to(device)])
        return mesh_norm

    except Exception as e:
        print(f"Error loading mesh {mesh_path}: {e}")
        raise e


@torch.no_grad()
def mesh_to_pointcloud(mesh: Meshes, num_samples: int = 2048) -> torch.Tensor:
    """
    Samples points on the mesh surface.
    Returns: (B, num_samples, 3) points
    """
    pts = sample_points_from_meshes(mesh, num_samples=num_samples)
    return pts


def sdf_to_occupancy(sdf_grid: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    """
    Convert SDF grid to occupancy grid by thresholding.
    sdf_grid: (B, D, H, W) where value<0 = inside
    Returns occupancy: (B, D, H, W) in {0,1}
    """
    return (sdf_grid <= threshold).to(sdf_grid.dtype)
