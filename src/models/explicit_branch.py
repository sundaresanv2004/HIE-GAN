import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class ExplicitDeformer(nn.Module):
    def __init__(self, init_mesh, feature_dim=256):
        super().__init__()

        V0, E0 = init_mesh

        self.V = nn.Parameter(V0)  # trainable vertices
        self.E0 = E0  # (2, E) edge indices

        self.conv1 = GCNConv(3 + feature_dim, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 3)

    def forward(self, img_feat):
        B, F = img_feat.shape
        Nv = self.V.size(0)

        # expand initial mesh
        V = self.V.unsqueeze(0).repeat(B, 1, 1)  # (B,Nv,3)
        F = img_feat.unsqueeze(1).repeat(1, Nv, 1)  # (B,Nv,256)
        X = torch.cat([V, F], dim=-1)  # (B,Nv,259)

        # graph convolution: flatten for pyg
        X = X.reshape(-1, X.size(-1))  # (B*Nv,259)

        E = self.E0  # (2,E)

        X = self.conv1(X, E)
        X = self.conv2(X, E)
        dV = self.conv3(X, E)
        dV = dV.reshape(B, Nv, 3)

        return V + dV
