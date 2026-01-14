import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=6, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        # Create freq bands: 2^0, 2^1, ..., 2^(L-1)
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)

    def forward(self, x):
        """
        x: (B, ..., 3)
        Returns: (B, ..., 3 + 3 * 2 * num_freqs) if include_input
        """
        if self.include_input:
            outputs = [x]
        else:
            outputs = []
            
        for freq in self.freq_bands:
            # Move freqs to same device as x
            freq = freq.to(x.device)
            outputs.append(torch.sin(x * freq * np.pi))
            outputs.append(torch.cos(x * freq * np.pi))
            
        return torch.cat(outputs, dim=-1)

class ImplicitDecoder(nn.Module):
    """
    Advanced Implicit Decoder for HIE-GAN (Phase 3 Optimized).
    
    Improvements:
    - Positional Encoding (High freq details)
    - Skip Connections (Better gradients)
    - Weight Normalization (Stability)
    - Softplus (Smooth gradients for Fusion)
    """
    def __init__(self, feature_dim=256, hidden_dim=256, num_layers=6, 
                 skip_connection_at=[3], use_positional_encoding=True, pos_enc_levels=6):
        super().__init__()
        
        self.skip_connection_at = skip_connection_at
        
        # 1. Positional Encoding
        if use_positional_encoding:
            self.pos_enc = PositionalEncoding(num_freqs=pos_enc_levels, include_input=True)
            # Input dim = 3 + 3 * 2 * L
            coord_dim = 3 + 3 * 2 * pos_enc_levels
        else:
            self.pos_enc = None
            coord_dim = 3
            
        self.input_dim = feature_dim + coord_dim

        # 2. MLP Layers
        layers = []
        
        # First layer
        # weight_norm for training stability
        self.layer0 = nn.utils.weight_norm(nn.Linear(self.input_dim, hidden_dim))
        
        # Hidden layers
        for i in range(1, num_layers - 1):
            if i in self.skip_connection_at:
                # If skip, input is hidden + original input
                layers.append(nn.utils.weight_norm(nn.Linear(hidden_dim + self.input_dim, hidden_dim)))
            else:
                layers.append(nn.utils.weight_norm(nn.Linear(hidden_dim, hidden_dim)))
                
        self.hidden_layers = nn.ModuleList(layers)
        
        # Output layer (SDF)
        self.last_layer = nn.Linear(hidden_dim, 1) # simple linear here usually fine
        
        # 3. Activation
        # Softplus with beta=100 behaves like ReLU but is smooth (C1 continuous)
        # Crucial for stable gradients in Fusion Module
        self.act = nn.Softplus(beta=100)

    def forward(self, features, query_points):
        """
        Args:
            features: (B, C) Global image features
            query_points: (B, N, 3)
        """
        B, N, _ = query_points.shape
        
        # Expand features
        features_exp = features.unsqueeze(1).expand(-1, N, -1)
        
        # Embed points
        if self.pos_enc:
            points_enc = self.pos_enc(query_points) # (B, N, coord_dim)
        else:
            points_enc = query_points
            
        # Concatenate input
        # (B, N, C + coord_dim)
        inputs = torch.cat([features_exp, points_enc], dim=-1)
        
        x = inputs
        
        # First layer
        x = self.act(self.layer0(x))
        
        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            layer_idx = i + 1 # since layer0 is outside
            
            if layer_idx in self.skip_connection_at:
                # Concatenate skip
                x = torch.cat([x, inputs], dim=-1)
                
            x = self.act(layer(x))
            
        # Output SDF
        # We might want to initialize bias to e.g. 0.5 to start with spheres? 
        # But standard init is ok usually.
        sdf = self.last_layer(x)
        
        return sdf
