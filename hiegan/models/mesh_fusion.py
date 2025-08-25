import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshFusion(nn.Module):
    """
    Fusion module to combine implicit and explicit representations
    """

    def __init__(self, latent_dim=512, fusion_dim=256):
        super().__init__()

        # Learnable fusion weights
        self.implicit_weight_net = nn.Sequential(
            nn.Linear(latent_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )

        self.explicit_weight_net = nn.Sequential(
            nn.Linear(latent_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1),
            nn.Sigmoid()
        )

        # Refinement network
        self.refinement = nn.Sequential(
            nn.Linear(latent_dim + 3, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 3)  # Output 3D coordinates
        )

    def forward(self, latent, implicit_sdf, explicit_vertices, sample_points):
        """
        Fuse implicit and explicit representations

        Args:
            latent: (B, latent_dim) - encoded features
            implicit_sdf: (B, N, 1) - SDF values at sample points
            explicit_vertices: (B, V, 3) - deformed mesh vertices
            sample_points: (B, N, 3) - query points for fusion

        Returns:
            fused_points: (B, N, 3) - final 3D coordinates
        """
        B, N, _ = sample_points.shape

        # Compute adaptive fusion weights
        implicit_weight = self.implicit_weight_net(latent).unsqueeze(1)  # (B, 1, 1)
        explicit_weight = self.explicit_weight_net(latent).unsqueeze(1)  # (B, 1, 1)

        # Normalize weights
        total_weight = implicit_weight + explicit_weight + 1e-8
        implicit_weight = implicit_weight / total_weight
        explicit_weight = explicit_weight / total_weight

        # Convert SDF to approximate surface points
        # For points near surface (|SDF| < threshold), move them to surface
        surface_mask = (implicit_sdf.abs() < 0.1).float()
        implicit_surface_points = sample_points - implicit_sdf * surface_mask

        # Find nearest explicit vertices for each sample point
        # Simple approach: use mean of explicit vertices as reference
        explicit_reference = explicit_vertices.mean(dim=1, keepdim=True).expand(B, N, 3)

        # Weighted fusion
        fused_base = (implicit_weight * implicit_surface_points +
                      explicit_weight * explicit_reference)

        # Refinement network for final adjustment
        latent_expanded = latent.unsqueeze(1).expand(B, N, -1)
        refinement_input = torch.cat([latent_expanded, fused_base], dim=-1)
        refinement_offset = self.refinement(refinement_input)

        fused_points = fused_base + 0.1 * refinement_offset  # Small refinement

        return fused_points, implicit_weight, explicit_weight
