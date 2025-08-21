# hiegan/models/explicit_gcn.py
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class ExplicitGCN(nn.Module):
    """
    Graph convolution network for explicit mesh deformation.
    Input: template mesh vertices and latent vector.
    Output: deformed mesh vertices.
    """
    def __init__(self, latent_dim=512, hidden_dim=128, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.input_fc = nn.Linear(latent_dim + 3, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.output_fc = nn.Linear(hidden_dim, 3)  # predict new vertex positions

    def forward(self, vertex_xyz, latent, edge_index):
        """
        vertex_xyz: (V, 3) template mesh vertices
        latent: (B, latent_dim)
        edge_index: PyG edge_index for mesh graph
        returns: deformed vertices (B, V, 3)
        """
        B = latent.shape[0]
        V = vertex_xyz.shape[0]

        # Expand latent to all vertices
        latent_exp = latent.unsqueeze(1).expand(B, V, latent.shape[-1])
        x = torch.cat([vertex_xyz.unsqueeze(0).expand(B, V, 3), latent_exp], dim=-1)

        # Flatten batch for GCNConv
        x = x.view(B*V, -1)
        x = self.input_fc(x).relu()

        for conv in self.convs:
            x = conv(x, edge_index.repeat(1, B))  # repeat edges for batch
            x = x.relu()

        x = self.output_fc(x)
        x = x.view(B, V, 3)
        return x
