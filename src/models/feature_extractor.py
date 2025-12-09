import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        # Use ImageNet pretrained weights for faster convergence
        backbone = models.resnet18(weights="DEFAULT")
        backbone.fc = nn.Linear(backbone.fc.in_features, out_dim)
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
