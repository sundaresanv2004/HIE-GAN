import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class ExplicitDeformer(nn.Module):
    def __init__(self, init_mesh, feature_dim=256, hidden_dims=[128, 64]):
        super().__init__()
        V0, E0 = init_mesh
        self.V = nn.Parameter(V0)
        self.E = E0
        self.conv1 = GCNConv(3 + feature_dim, hidden_dims[0])
        self.conv2 = GCNConv(hidden_dims[0], hidden_dims[1])
        self.conv3 = GCNConv(hidden_dims[1], 3)

    def forward(self, img_feat):
        B, F = img_feat.shape
        V = self.V.unsqueeze(0).repeat(B, 1, 1)
        Fv = img_feat.unsqueeze(1).repeat(1, V.shape[1], 1)
        X = torch.cat([V, Fv], dim=-1).reshape(-1, 3 + F)
        E = self.E
        X = self.conv1(X, E)
        X = self.conv2(X, E)
        dV = self.conv3(X, E).reshape(B, V.shape[1], 3)
        return V + dV
