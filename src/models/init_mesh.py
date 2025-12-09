import trimesh
import torch


def create_sphere_mesh(subdivisions=2):
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
    V = torch.tensor(sphere.vertices, dtype=torch.float32)
    E = torch.tensor(sphere.edges_unique.T, dtype=torch.long)
    return V, E
