import torch
import torch.nn as nn

class SDFLoss(nn.Module):
    """
    SDF Regression Loss for HIE-GAN (Phase 2).
    
    Computes the difference between predicted SDF values and ground truth SDF values.
    Typically uses L1 loss (Mean Absolute Error).
    
    Optionally clamps the ground truth distance to focus learning near the surface.
    """
    def __init__(self, clamp_dist=0.1, weight=1.0):
        """
        Args:
            clamp_dist (float): Maximum absolute distance to consider (truncation).
                                If None, no clamping is applied.
            weight (float): Loss scaling factor.
        """
        super().__init__()
        self.clamp_dist = clamp_dist
        self.weight = weight
        self.l1_loss = nn.L1Loss()
        
    def forward(self, pred_sdf, gt_sdf):
        """
        Args:
            pred_sdf: (B, N, 1) Predicted SDF values
            gt_sdf: (B, N, 1) Ground truth SDF values
            
        Returns:
            loss: scalar tensor
        """
        if self.clamp_dist is not None:
            # Clamping focus learning on the near-surface field
            # We clamp both prediction and GT for stability, or just GT depending on strategy.
            # Standard DeepSDF clamps the GT values. 
            gt_sdf_clamped = torch.clamp(gt_sdf, -self.clamp_dist, self.clamp_dist)
            pred_sdf_clamped = torch.clamp(pred_sdf, -self.clamp_dist, self.clamp_dist)
            
            loss = self.l1_loss(pred_sdf_clamped, gt_sdf_clamped)
        else:
            loss = self.l1_loss(pred_sdf, gt_sdf)
            
        return loss * self.weight
