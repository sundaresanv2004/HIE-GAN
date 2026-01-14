import torch
import torch.nn as nn
import torch.autograd as autograd

class FusionModule(nn.Module):
    """
    Fusion Module for HIE-GAN (Phase 3).
    
    Refines the Explicit Mesh using the implicit field's gradient.
    Moves vertices towards the zero-level set of the predicted SDF.
    
    Formula:
    V_new = V_old - alpha * SDF(V_old) * sign(SDF(V_old)) * Normal(V_old)
    OR simpler: V_new = V_old - alpha * SDF(V_old) * Gradient(SDF)
    """
    def __init__(self, step_size=1.0):
        super().__init__()
        # Alpha can be fixed or learnable. Starting with fixed.
        self.step_size = step_size
        
        # Optional: Learnable step size per vertex or global
        # self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, explicit_verts, implicit_decoder, global_features):
        """
        Args:
            explicit_verts: (B, N, 3) Vertices from explicit branch
            implicit_decoder: Trained ImplicitDecoder model
            global_features: (B, C) Global image features
            
        Returns:
            fused_verts: (B, N, 3) Refined vertices
        """
        B, N, _ = explicit_verts.shape
        
        # We need gradients w.r.t input vertices
        # Clone and enable grad
        verts_in = explicit_verts.clone().detach().requires_grad_(True)
        
        # 1. Query SDF at explicit vertex locations
        # implicit_decoder(features, points) -> (B, N, 1)
        sdf_pred = implicit_decoder(global_features, verts_in)
        
        # 2. Compute Gradient (Normal direction)
        # d_SDF / d_Position
        # sum() is needed because autograd.grad works on scalar outputs
        d_output = torch.ones_like(sdf_pred, requires_grad=False, device=explicit_verts.device)
        
        gradients = autograd.grad(
            outputs=sdf_pred,
            inputs=verts_in,
            grad_outputs=d_output,
            create_graph=True, # Enable higher-order derivatives if needed (e.g. for regularization)
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Normalize gradients to get true normals (SDF gradients should be unit length, but network might not enforce it perfectly)
        # Adding epsilon for stability
        normals = torch.nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        
        # 3. Refine Vertices
        # Move against the gradient direction by the SDF distance
        # V_new = V - SDF * Normal
        # If SDF is positive (outside), we move -Normal (inwards)
        # If SDF is negative (inside), we move -Normal * (-dist) = +Normal (outwards)
        
        displacement = -self.step_size * sdf_pred * normals
        
        # Apply displacement to ORIGINAL tensor (to keep computational graph from explicit branch if needed later, 
        # though usually we detach, but here we want end-to-end fusion)
        fused_verts = explicit_verts + displacement
        
        return fused_verts
