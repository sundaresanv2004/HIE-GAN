# hiegan/models/generator.py
import torch
import torch.nn as nn
from hiegan.models.encoder_vit import ViTEncoder
from hiegan.models.implicit_sdf import ImplicitSDF
from hiegan.models.explicit_gcn import ExplicitGCN
from hiegan.utils.mesh_utils import mesh_to_pointcloud, sdf_to_occupancy

class HIEGenerator(nn.Module):
    """
    Hybrid Implicit-Explicit GAN Generator
    Input: Image(s)
    Output: Predicted 3D mesh (implicit + explicit fusion)
    """
    def __init__(self, latent_dim=512, device="cuda"):
        super().__init__()
        self.device = device

        # Encoder
        self.encoder = ViTEncoder(latent_dim=latent_dim)

        # Branches
        self.implicit = ImplicitSDF(latent_dim=latent_dim)
        self.explicit = ExplicitGCN(latent_dim=latent_dim)

        # Fusion (optional, simple averaging for now)
        # Can be replaced with learned fusion later
        self.fusion_weight = 0.5

    def forward(self, imgs, template_mesh_vertices=None, template_mesh_edges=None, sample_xyz=None):
        """
        imgs: (B, 3, H, W)
        template_mesh_vertices: (V,3) for explicit branch
        template_mesh_edges: edge_index (2,E) for GCN
        sample_xyz: (B, N, 3) points for implicit SDF evaluation

        Returns:
            pred_mesh: dict with keys:
                'implicit_sdf': (B, N, 1)
                'explicit_vertices': (B, V, 3)
                'fused_vertices': (B, V, 3) (optional)
        """
        B = imgs.shape[0]

        # --- Encode image to latent vector ---
        latent = self.encoder(imgs)  # (B, latent_dim)

        # --- Implicit branch ---
        implicit_sdf = None
        if sample_xyz is not None:
            implicit_sdf = self.implicit(latent, sample_xyz)  # (B, N, 1)

        # --- Explicit branch ---
        explicit_vertices = None
        fused_vertices = None
        if template_mesh_vertices is not None and template_mesh_edges is not None:
            explicit_vertices = self.explicit(template_mesh_vertices, latent, template_mesh_edges)  # (B, V, 3)
            # Simple fusion: weighted average between template & predicted explicit
            fused_vertices = template_mesh_vertices.unsqueeze(0).expand(B, -1, -1) * (1 - self.fusion_weight) + \
                             explicit_vertices * self.fusion_weight

        return {
            "latent": latent,
            "implicit_sdf": implicit_sdf,
            "explicit_vertices": explicit_vertices,
            "fused_vertices": fused_vertices
        }
