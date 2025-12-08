import trimesh
import torch


def create_sphere_mesh(num_subdivisions=2):
    """
    num_subdivisions controls mesh resolution:
        0 → coarse sphere
        1 → ~162 vertices
        2 → ~642 vertices   (recommended)
        3 → ~2562 vertices (slow)
    """

    sphere = trimesh.creation.icosphere(subdivisions=num_subdivisions)

    # vertices → torch tensor (N, 3)
    V = torch.tensor(sphere.vertices, dtype=torch.float32)  # (N,3)

    # edges: list of (u,v) pairs
    edges = sphere.edges_unique
    edges = torch.tensor(edges.T, dtype=torch.long)  # (2, E)
    # NOTE: transposed because torch_geometric requires shape (2, num_edges)

    return V, edges
