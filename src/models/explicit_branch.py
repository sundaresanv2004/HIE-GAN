import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class ExplicitDeformer(nn.Module):
    def __init__(self, init_mesh, feature_dim=256, hidden_dims=[128, 64], use_layer_norm=True):
        super().__init__()
        V0, E0 = init_mesh

        # Store as parameter (trainable) and buffer (non-trainable)
        self.register_parameter("V", nn.Parameter(V0))
        self.register_buffer("E", E0)

        layers = []
        norms = []
        
        # Input layer
        input_dim = 3 + feature_dim
        
        # Hidden layers
        for h_dim in hidden_dims:
            layers.append(GCNConv(input_dim, h_dim))
            
            if use_layer_norm:
                 norms.append(nn.LayerNorm(h_dim))
            else:
                 norms.append(nn.Identity())
                 
            input_dim = h_dim
            
        self.gcn_layers = nn.ModuleList(layers)
        self.norm_layers = nn.ModuleList(norms)
        
        # Output layer (always outputs 3 for displacement)
        self.final_layer = GCNConv(input_dim, 3)
        
        self.relu = nn.ReLU()
        self.use_layer_norm = use_layer_norm

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
        num_edges = self.E.shape[1]
        E = self.E.repeat(1, B)
        # Create offsets for each batch item: [0, 0, ..., N, N, ...]
        offset = torch.arange(B, device=self.E.device).repeat_interleave(num_edges) * N
        E = E + offset.unsqueeze(0)

        # GCN layers with activations and norms
        for layer, norm in zip(self.gcn_layers, self.norm_layers):
            X = layer(X, E)
            X = norm(X)
            X = self.relu(X)
            
        # Final layer
        dV = self.final_layer(X, E)  # Predict vertex displacement

        # Reshape back to batch format
        dV = dV.reshape(B, N, 3)

        # Return deformed vertices
        return V + dV
