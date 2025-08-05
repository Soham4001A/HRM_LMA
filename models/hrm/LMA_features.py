# LMA_features.py
"""
LMA_features.py
Author  : Soham Sane (adapted for HRM by Gemini)
Purpose : Provides the Latent Meta Attention (LMA) initial transformation
          module, ported directly from the PointNet LMA vs MHA repository.
          This module transforms an input tensor into a latent representation
          while preserving the sequence length.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LMAInitialTransform(nn.Module):
    """
    Implements the LMA pre-processing pipeline exactly as described in the
    PointNet LMA implementation. It mixes features across the embedding
    dimension before projecting into a latent space.

    The sequence length of the tensor is preserved.

    Pipeline:
    1.  Split embedding dim `d0` into `H` chunks.
    2.  Stack these chunks along the sequence dimension `L`, creating a
        tensor of shape (B, H*L, d0/H).
    3.  Reshape the tensor back to (B, L, d0), effectively mixing features.
    4.  Project the mixed tensor into the new latent dimension `d_new`.
    5.  Add a residual connection from a direct projection of the original input.
    """
    def __init__(self, d0: int, n_heads: int, d_new: int = None):
        """
        Args:
            d0 (int): Original embedding dimension.
            n_heads (int): Number of heads for the split-stack feature mixing.
            d_new (int): Target latent embedding dimension. Defaults to d0 // 2.
        """
        super().__init__()
        if d0 % n_heads != 0:
            raise ValueError(f"d0 ({d0}) must be divisible by n_heads ({n_heads})")

        self.n_heads = n_heads
        self.d0 = d0
        self.d_new = d_new if d_new is not None else d0 // 2

        # This single projection layer is used for both the main and residual paths.
        self.proj = nn.Linear(self.d0, self.d_new, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the LMA transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d0).

        Returns:
            torch.Tensor: Transformed latent tensor of shape (B, L, d_new).
        """
        B, L, _ = x.shape

        # --- Main LMA Path (Feature Mixing + Projection) ---
        chunks = torch.chunk(x, self.n_heads, dim=-1)
        stacked = torch.cat(chunks, dim=1)  # Shape: (B, H*L, d0/H)
        reshaped = stacked.view(B, L, self.d0)
        y = F.relu(self.proj(reshaped))

        # --- Residual Path (Direct Projection) ---
        x_proj = F.relu(self.proj(x))

        # --- Final Output ---
        return y + x_proj
