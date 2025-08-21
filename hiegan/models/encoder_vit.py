import torch
import torch.nn as nn
import timm

class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder to extract image features into a latent vector.
    """
    def __init__(self, latent_dim=512, pretrained=True):
        super().__init__()
        # Use timm ViT backbone
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        # Remove classifier head
        self.backbone.head = nn.Identity()
        # Project to latent_dim
        self.fc = nn.Linear(self.backbone.num_features, latent_dim)

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: (B, latent_dim)
        """
        feat = self.backbone(x)
        latent = self.fc(feat)
        return latent
