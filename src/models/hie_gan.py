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
        # 1. Feature Extraction
        feat = self.encoder(img) # (B, C)
        
        # 2. Explicit Branch
        pred_pc_exp = self.explicit(feat)
        
        # 3. Implicit Branch
        pred_sdf = self.implicit(feat, query_pts)

        # 4. Fusion Module
        # Note: Fusion module internally computes gradients for SDF
        # We pass the implicit module ITSELF to fusion.
        # However, inside DataParallel, passing sub-modules can be tricky if they are not on same device.
        # But since 'implicit' is a submodule of THIS module, it is already on the correct device replica.
        pred_pc_fused = self.fusion(pred_pc_exp, self.implicit, feat)
        
        return pred_pc_fused, pred_sdf, pred_pc_exp, feat
