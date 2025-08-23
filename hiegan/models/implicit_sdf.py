import torch
import torch.nn as nn

class ImplicitSDF(nn.Module):
    """
    MLP predicting SDF values for 3D points given a latent vector.
    """
    def __init__(self, latent_dim=512, hidden_dim=256, num_layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        layers = [nn.Linear(latent_dim + 3, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))  # output SDF
        self.mlp = nn.Sequential(*layers)

    def forward(self, latent, xyz):
        """
        latent: (B, latent_dim)
        xyz: (B, N, 3)  points to evaluate SDF
        returns: (B, N, 1)
        """
        B, N, _ = xyz.shape
        latent_exp = latent.unsqueeze(1).expand(B, N, self.latent_dim)
        inp = torch.cat([latent_exp, xyz], dim=-1)
        sdf = self.mlp(inp)
        return sdf
