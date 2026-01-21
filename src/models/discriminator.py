import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    """
    Geometry-aware Discriminator for 3D Point Clouds.
    Based on a simplified PointNet architecture.
    """
    def __init__(self, input_dim=3, feature_dim=64):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Args:
            x: Input point cloud of shape (B, N, 3) or (B, 3, N)
        Returns:
            validity: Realism score (B, 1)
        """
        # Ensure input is (B, C, N)
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(1, 2)
            
        B = x.size(0)
        
        # PointNet Encoder
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        
        # Global Max Pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(B, -1)
        
        # MLP Classifier
        x = F.leaky_relu(self.bn4(self.fc1(x)), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn5(self.fc2(x)), 0.2)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
