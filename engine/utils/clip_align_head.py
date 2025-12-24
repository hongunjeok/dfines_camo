import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectFeatureAlignHead(nn.Module):
    def __init__(self, in_dim=256, out_dim=512):
        """
        Projects object features to CLIP embedding space.

        Args:
            in_dim (int): Input dimension of object features (e.g., from encoder).
            out_dim (int): Output dimension matching CLIP text embedding (e.g., 512).
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        """
        Forward pass to get L2-normalized feature aligned with CLIP text space.

        Args:
            x (Tensor): Input tensor of shape [N, in_dim]

        Returns:
            Tensor: L2-normalized output of shape [N, out_dim]
        """
        x = self.proj(x)
        return F.normalize(x, dim=-1)