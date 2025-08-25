import torch
import torch.nn as nn
from .encoder_vit import ViTEncoder
from .implicit_sdf import ImplicitSDF
from .explicit_gcn import ExplicitGCN
from .mesh_fusion import MeshFusion


# Update the HIEGenerator class
class HIEGenerator(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()

        # Encoder
        self.encoder = ViTEncoder(latent_dim=latent_dim)

        # Decoders (Branches)
        self.implicit_decoder = ImplicitSDF(latent_dim=latent_dim)
        self.explicit_decoder = ExplicitGCN(latent_dim=latent_dim)

        # Fusion module
        self.fusion = MeshFusion(latent_dim=latent_dim)

    def forward(self, imgs, template_mesh_vertices=None, template_mesh_edges=None,
                sample_xyz=None, enable_fusion=True):

        # Encode image to latent vector
        latent = self.encoder(imgs)
        outputs = {"latent": latent}

        # Implicit branch
        if sample_xyz is not None:
            implicit_sdf = self.implicit_decoder(latent, sample_xyz)
            outputs["implicit_sdf"] = implicit_sdf

        # Explicit branch
        if template_mesh_vertices is not None and template_mesh_edges is not None:
            displacements = self.explicit_decoder(template_mesh_vertices, latent, template_mesh_edges)
            deformed_vertices = template_mesh_vertices.unsqueeze(0) + displacements
            outputs["explicit_displacements"] = displacements
            outputs["deformed_vertices"] = deformed_vertices

        # Fusion (if both branches are active)
        if (enable_fusion and 'implicit_sdf' in outputs and
                'deformed_vertices' in outputs and sample_xyz is not None):
            fused_points, imp_weight, exp_weight = self.fusion(
                latent, implicit_sdf, deformed_vertices, sample_xyz
            )
            outputs["fused_points"] = fused_points
            outputs["fusion_weights"] = {"implicit": imp_weight, "explicit": exp_weight}

        return outputs
