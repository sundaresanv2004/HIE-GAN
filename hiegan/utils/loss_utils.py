import torch
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes


class HIEGANLoss(nn.Module):
    """
    Combined loss function for HIE-GAN
    """

    def __init__(self,
                 chamfer_weight=1.0,
                 sdf_weight=0.5,
                 explicit_weight=0.5,
                 fusion_weight=1.0):
        super().__init__()

        self.chamfer_weight = chamfer_weight
        self.sdf_weight = sdf_weight
        self.explicit_weight = explicit_weight
        self.fusion_weight = fusion_weight

    def chamfer_loss(self, pred_points, gt_points):
        """Compute Chamfer distance"""
        loss, _ = chamfer_distance(pred_points, gt_points)
        return loss

    def sdf_loss(self, pred_sdf, gt_sdf):
        """L1 loss for SDF values"""
        return F.l1_loss(pred_sdf, gt_sdf)

    def explicit_loss(self, pred_vertices, gt_mesh):
        """Loss for explicit mesh vertices"""
        gt_points = sample_points_from_meshes(gt_mesh, num_samples=2048)
        pred_points = pred_vertices  # Assuming same number of points

        # If different sizes, sample from predicted vertices
        if pred_vertices.shape[1] != gt_points.shape[1]:
            indices = torch.randint(0, pred_vertices.shape[1],
                                    (gt_points.shape[1],), device=pred_vertices.device)
            pred_points = pred_vertices[:, indices, :]

        return self.chamfer_loss(pred_points, gt_points)

    def forward(self, outputs, gt_mesh, gt_sdf=None):
        """
        Compute total loss

        Args:
            outputs: Dictionary from HIE generator
            gt_mesh: Ground truth mesh
            gt_sdf: Ground truth SDF values (optional)
        """
        total_loss = 0.0
        loss_dict = {}

        # Chamfer loss for final fused output (if available)
        if 'fused_points' in outputs:
            gt_points = sample_points_from_meshes(gt_mesh, num_samples=outputs['fused_points'].shape[1])
            chamfer_loss = self.chamfer_loss(outputs['fused_points'], gt_points)
            total_loss += self.fusion_weight * chamfer_loss
            loss_dict['fusion_chamfer'] = chamfer_loss

        # SDF loss (if GT SDF available)
        if 'implicit_sdf' in outputs and gt_sdf is not None:
            sdf_loss = self.sdf_loss(outputs['implicit_sdf'], gt_sdf)
            total_loss += self.sdf_weight * sdf_loss
            loss_dict['sdf_loss'] = sdf_loss

        # Explicit mesh loss
        if 'deformed_vertices' in outputs:
            explicit_loss = self.explicit_loss(outputs['deformed_vertices'], gt_mesh)
            total_loss += self.explicit_weight * explicit_loss
            loss_dict['explicit_loss'] = explicit_loss

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
