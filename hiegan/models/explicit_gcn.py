import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class ExplicitGCN(nn.Module):
    """
    Graph convolution network for explicit mesh deformation.
    It predicts vertex displacements for a template mesh.
    """

    def __init__(self, latent_dim=512, hidden_dim=256, num_layers=5):
        super().__init__()
        self.num_layers = num_layers

        # Initial feature projection for each vertex (pos + latent)
        self.input_fc = nn.Linear(latent_dim + 3, hidden_dim)

        # GCN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer predicts a 3D displacement vector for each vertex
        self.output_fc = nn.Linear(hidden_dim, 3)

    def forward(self, vertex_xyz, latent, edge_index):
        """
        Processes a batch of latent vectors to deform a single template mesh.

        Args:
            vertex_xyz (torch.Tensor): Template mesh vertices of shape (V, 3).
            latent (torch.Tensor): Batch of latent vectors of shape (B, latent_dim).
            edge_index (torch.Tensor): PyG edge_index for the mesh graph of shape (2, E).

        Returns:
            torch.Tensor: Predicted vertex displacements of shape (B, V, 3).
        """
        B = latent.shape[0]
        V = vertex_xyz.shape[0]

        # Expand latent vector for each vertex and concatenate with vertex positions
        latent_exp = latent.unsqueeze(1).expand(B, V, -1)  # (B, V, latent_dim)
        vertex_xyz_exp = vertex_xyz.unsqueeze(0).expand(B, V, -1)  # (B, V, 3)

        # Initial features for all vertices in the batch
        x = torch.cat([vertex_xyz_exp, latent_exp], dim=-1)  # (B, V, 3 + latent_dim)

        # GCNs are slow with loops, but this is the simplest correct way
        # without complex batching. We flatten for the GCN layers.
        x = x.view(B * V, -1)  # (B*V, 3 + latent_dim)
        x = F.relu(self.input_fc(x))

        # Create a batched edge_index for PyG
        # This correctly offsets indices for each graph in the batch
        edge_index_batched = edge_index.repeat(1, B) + torch.arange(B, device=edge_index.device).repeat_interleave(
            edge_index.shape[1]) * V

        for conv in self.convs:
            x = conv(x, edge_index_batched)
            x = F.relu(x)

        # Predict final displacements
        displacements = self.output_fc(x)

        return displacements.view(B, V, 3)  # Reshape back to (B, V, 3)
