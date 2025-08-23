import torch
import torch.nn as nn
from .encoder_vit import ViTEncoder
from .implicit_sdf import ImplicitSDF
from .explicit_gcn import ExplicitGCN


class HIEGenerator(nn.Module):
    """
    Hybrid Implicit-Explicit GAN Generator
    Input: Image(s)
    Output: Predicted 3D mesh (implicit + explicit fusion)
    """

    def __init__(self, latent_dim=512):
        super().__init__()

        # Encoder
        self.encoder = ViTEncoder(latent_dim=latent_dim)

        # Decoders (Branches)
        self.implicit_decoder = ImplicitSDF(latent_dim=latent_dim)
        self.explicit_decoder = ExplicitGCN(latent_dim=latent_dim)

    def forward(self, imgs, template_mesh_vertices=None, template_mesh_edges=None, sample_xyz=None):
        """
        Args:
            imgs (torch.Tensor): Input images (B, 3, H, W)
            template_mesh_vertices (torch.Tensor): Template vertices (V,3) for explicit branch
            template_mesh_edges (torch.Tensor): Template edge_index (2,E) for GCN
            sample_xyz (torch.Tensor): Query points (B, N, 3) for implicit SDF evaluation

        Returns:
            dict: A dictionary containing model outputs.
        """
        # --- Encode image to latent vector ---
        latent = self.encoder(imgs)  # (B, latent_dim)

        outputs = {"latent": latent}

        # --- Implicit branch ---
        if sample_xyz is not None:
            implicit_sdf = self.implicit_decoder(latent, sample_xyz)  # (B, N, 1)
            outputs["implicit_sdf"] = implicit_sdf

        # --- Explicit branch ---
        if template_mesh_vertices is not None and template_mesh_edges is not None:
            # Predict displacements from the template
            displacements = self.explicit_decoder(template_mesh_vertices, latent, template_mesh_edges)  # (B, V, 3)

            # Deform the mesh by adding the predicted displacements
            deformed_vertices = template_mesh_vertices.unsqueeze(0) + displacements  # (B, V, 3)

            outputs["explicit_displacements"] = displacements
            outputs["deformed_vertices"] = deformed_vertices

        return outputs
