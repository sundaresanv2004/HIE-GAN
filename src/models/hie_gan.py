import torch
import torch.nn as nn

class HIEGANModel(nn.Module):
    """
    Unified HIE-GAN Model Wrapper.
    Wraps Encoder, Explicit, Implicit, and Fusion modules into a single nn.Module.
    This enables usage of nn.DataParallel for Multi-GPU training.
    """
    def __init__(self, encoder, explicit, implicit, fusion):
        super().__init__()
        self.encoder = encoder
        self.explicit = explicit
        self.implicit = implicit
        self.fusion = fusion

    def forward(self, img, query_pts):
        """
        Forward pass for the entire pipeline.
        
        Args:
            img: (B, C, H, W) Input images
            query_pts: (B, N, 3) Query points for SDF
            
        Returns:
            pred_pc_fused: (B, N_pc, 3) Fused point cloud
            pred_sdf: (B, N_sdf, 1) Predicted SDF values
            pred_pc_exp: (B, N_pc, 3) Explicit branch point cloud
            feat: (B, Dim) Global feature vector (returned for loss computation if needed)
        """
        if self.training:
            # Training Mode: Everything needs gradients
            feat = self.encoder(img) # (B, C)
            pred_pc_exp = self.explicit(feat)
            pred_sdf = self.implicit(feat, query_pts)
            pred_pc_fused = self.fusion(pred_pc_exp, self.implicit, feat)
            return pred_pc_fused, pred_sdf, pred_pc_exp, feat
        
        else:
            # Inference/Validation Mode: Optimize Memory
            # 1. Run backbone without gradients (saves huge memory)
            with torch.no_grad():
                feat = self.encoder(img)
                pred_pc_exp = self.explicit(feat)
                pred_sdf = self.implicit(feat, query_pts)
            
            # 2. Run Fusion with gradients enabled (required for autograd.grad internal logic)
            # The context manager ensures Fusion can compute gradients of SDF w.r.t inputs,
            # but assumes inputs (feat/pred_pc_exp) are treated as constants/leafs or appropriately detached.
            with torch.enable_grad():
                 pred_pc_fused = self.fusion(pred_pc_exp, self.implicit, feat)
            
            return pred_pc_fused, pred_sdf, pred_pc_exp, feat
