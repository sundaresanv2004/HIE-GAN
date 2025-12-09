import torch


def smoothness_loss(pred_mesh, penalty=0.1):
    """
    Encourages neighboring faces to have similar normals.
    (Laplacian smoothing approximation)
    
    Args:
        pred_mesh (tuple): (V, E) - vertices and edges
        penalty (float): Weight for the loss
    """
    V, E = pred_mesh
    # V: (B, N, 3)
    # E: (2, K) - edge indices
    
    # Gather neighbor vertices
    v1 = V[:, E[0, :], :]  # (B, K, 3)
    v2 = V[:, E[1, :], :]  # (B, K, 3)
    
    # Minimize distance between connected vertices
    loss = torch.mean((v1 - v2) ** 2)
    return loss * penalty


def edge_length_loss(pred_mesh, target_length=0.1, penalty=0.1):
    """
    Penalizes edges that deviate from target length.
    
    Args:
        pred_mesh (tuple): (V, E)
        target_length (float): Desired edge length
        penalty (float): Weight for the loss
    """
    V, E = pred_mesh
    
    v1 = V[:, E[0, :], :]
    v2 = V[:, E[1, :], :]
    
    edge_lengths = torch.norm(v1 - v2, dim=-1)  # (B, K)
    
    loss = torch.mean((edge_lengths - target_length) ** 2)
    
    return loss * penalty
