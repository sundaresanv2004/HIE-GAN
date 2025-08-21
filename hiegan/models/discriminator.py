import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudDiscriminator(nn.Module):
    """
    Discriminator for 3D point clouds sampled from predicted or GT meshes.
    Input: point cloud (B, N, 3)
    Output: probability (B, 1)
    """
    def __init__(self, hidden_dim=256, num_layers=4):
        super().__init__()
        layers = []
        input_dim = 3
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, pc):
        """
        pc: (B, N, 3) point cloud
        returns: (B, 1) probability of real/fake
        """
        # Simple global max-pooling over points (like PointNet)
        x = self.mlp(pc)  # (B, N, 1)
        x, _ = torch.max(x, dim=1)  # (B, 1)
        prob = torch.sigmoid(x)
        return prob
