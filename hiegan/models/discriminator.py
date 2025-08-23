import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudDiscriminator(nn.Module):
    """
    Discriminator for 3D point clouds.
    Input: point cloud (B, N, 3)
    Output: A single logit (B, 1) - NO SIGMOID HERE
    """
    def __init__(self, num_points=2048, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, hidden_dim, 1)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)

    def forward(self, pc):
        """
        pc: (B, N, 3) point cloud
        returns: (B, 1) logit
        """
        pc = pc.transpose(1, 2)  # (B, 3, N)
        x = F.relu(self.conv1(pc))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, dim=2)  # Global max pooling -> (B, hidden_dim)
        x = F.relu(self.fc1(x))
        logit = self.fc2(x) # Final logit output
        return logit
