import torch
import torch.nn.functional as F
from pytorch3d.loss import chamfer_distance
from pytorch3d.structures import Meshes
from .mesh_utils import mesh_to_pointcloud


class HieGanLoss(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def get_generator_loss(self, gen_outputs, gt_meshes, disc_fake_logits):
        deformed_verts = gen_outputs["deformed_vertices"]
        B = deformed_verts.shape[0]

        # 1. Chamfer Reconstruction Loss
        # Assumes gt_meshes provides the correct face topology for the template
        faces = gt_meshes.faces_list()[0].expand(B, -1, -1)
        pred_meshes = Meshes(verts=deformed_verts, faces=faces)

        pred_points = mesh_to_pointcloud(pred_meshes, self.cfg.NUM_POINTS_SAMPLED)
        gt_points = mesh_to_pointcloud(gt_meshes, self.cfg.NUM_POINTS_SAMPLED)
        loss_chamfer, _ = chamfer_distance(pred_points, gt_points)

        # 2. Adversarial Loss (numerically stable version)
        loss_adv = F.binary_cross_entropy_with_logits(
            disc_fake_logits, torch.ones_like(disc_fake_logits)
        )

        total_loss = self.cfg.W_CHAMFER * loss_chamfer + self.cfg.W_ADV * loss_adv

        return {
            "g_loss": total_loss,
            "chamfer": loss_chamfer,
            "g_adv": loss_adv,
        }

    def get_discriminator_loss(self, disc_real_logits, disc_fake_logits):
        loss_real = F.binary_cross_entropy_with_logits(
            disc_real_logits, torch.ones_like(disc_real_logits)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            disc_fake_logits, torch.zeros_like(disc_fake_logits)
        )
        total_loss = (loss_real + loss_fake) / 2
        return {"d_loss": total_loss}
