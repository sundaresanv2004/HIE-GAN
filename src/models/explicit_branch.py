import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class ExplicitDeformer(nn.Module):
    def __init__(self, init_mesh, feature_dim=256, hidden_dims=[128, 64]):
        super().__init__()
        V0, E0 = init_mesh

        # Store as parameter (trainable) and buffer (non-trainable)
        self.register_parameter("V", nn.Parameter(V0))
        self.register_buffer("E", E0)

        self.conv1 = GCNConv(3 + feature_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], 3)
        self.relu = nn.ReLU()

    def forward(self, img_feat):
        """
        Args:
            img_feat: (B, F) image features
        Returns:
            deformed_vertices: (B, N, 3)
        """
        B, F = img_feat.shape
        N = self.V.shape[0]

        # Broadcast template vertices to batch
        V = self.V.unsqueeze(0).expand(B, -1, -1)  # (B, N, 3)

        # Broadcast image features to each vertex
        Fv = img_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, F)

        # Concatenate vertex positions with features
        X = torch.cat([V, Fv], dim=-1)  # (B, N, 3+F)
        X = X.reshape(B * N, 3 + F)  # Flatten for GCN

        # Expand edge indices for batch processing
        E = self.E

        # GCN layers with activations
        X = self.relu(self.conv1(X, E))
        X = self.relu(self.conv2(X, E))
        dV = self.conv3(X, E)  # Predict vertex displacement

        # Reshape back to batch format
        dV = dV.reshape(B, N, 3)

        # Return deformed vertices
        return V + dV
