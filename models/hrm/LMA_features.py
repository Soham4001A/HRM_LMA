# LMA_features.py
"""
LMA_features.py
Author  : Soham Sane (adapted for HRM by Gemini)
Purpose : Provides building blocks for Latent Meta Attention (LMA).
          This includes the initial transformation pipeline and a
          self-contained LMA Transformer block that can replace a
          standard Multi-Head Attention block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import rms_norm, SwiGLU, Attention, CosSin

# =========================================================================
# Latent Meta Attention initial transform (split-stack-re-chunk-embed)
# =========================================================================
class LMAInitialTransform(nn.Module):
    """
    Implements the LMA pre-processing pipeline to transform an input tensor
    into a latent space representation with reduced dimensionality.

    The pipeline consists of four main steps:
        1. Split embedding dim into H chunks       -> (B, L, d0/H) * H
        2. Stack along sequence dim                -> (B, H*L, d0/H)
        3. Re-chunk to a new sequence length L'    -> (B, L', d_chunk)
        4. Project to the final latent dim d'      -> (B, L', d')
    """
    def __init__(self, d0: int, n_heads: int, d_new: int):
        """
        Args:
            d0 (int): Original embedding dimension.
            n_heads (int): Number of heads to split the embedding dimension into.
            d_new (int): Target latent embedding dimension.
        """
        super().__init__()
        if d0 % n_heads != 0:
            raise ValueError(f"Original dimension d0 ({d0}) must be divisible by n_heads ({n_heads})")

        self.d0 = d0
        self.n_heads = n_heads
        self.d_new = d_new
        self.d_head = d0 // n_heads

        # This projection will map the re-chunked tensor to the latent dimension d_new
        self.proj_latent = nn.Linear(self.d_head, self.d_new, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Applies the LMA transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, d0).

        Returns:
            torch.Tensor: Transformed latent tensor of shape (B, L * H, d_new).
                          Note: The new sequence length L' is L * H.
        """
        B, L, d0 = x.shape
        H = self.n_heads

        # 1. & 2. Split embedding dim and stack along sequence dim
        # Reshape to (B, L, H, d_head) -> (B, H, L, d_head) -> (B, H*L, d_head)
        x_stacked = x.view(B, L, H, self.d_head).transpose(1, 2).reshape(B, H * L, self.d_head)

        # 3. & 4. Re-chunking is implicit in the stacking. Now, project to d_new.
        # The new sequence length L' is H*L. The new feature dim is d_head.
        # We project this d_head to d_new.
        z = self.proj_latent(x_stacked)

        return z

# =========================================================================
# LMA Transformer Block
# =========================================================================
class LMATransformerBlock(nn.Module):
    """
    A self-contained Transformer block that uses Latent Meta Attention.
    It's designed as a drop-in replacement for a standard MHA block, maintaining
    the input and output tensor dimensions.

    Internal Pipeline:
    1.  Transform input `x` into a latent representation `z` using `LMAInitialTransform`.
    2.  Apply standard Multi-Head Attention within the latent space.
    3.  Apply a Feed-Forward Network (FFN) within the latent space.
    4.  Project the result back to the original input dimension `d0`.
    5.  Apply the final residual connection with the original input `x`.
    """
    def __init__(self, hidden_size: int, lma_heads: int, num_heads: int, expansion: float, norm_eps: float):
        """
        Args:
            hidden_size (int): The input and output dimension of the block (d0).
            lma_heads (int): The number of heads for the LMA transformation (H).
            num_heads (int): The number of heads for the attention mechanism in the latent space.
            expansion (float): The expansion factor for the FFN.
            norm_eps (float): Epsilon for RMS Normalization.
        """
        super().__init__()
        # The latent dimension d_new is chosen to be the head dimension of the LMA transform
        self.d_new = hidden_size // lma_heads

        self.lma_transform = LMAInitialTransform(d0=hidden_size, n_heads=lma_heads, d_new=self.d_new)

        # Attention and MLP operate in the latent space (dim=d_new)
        self.self_attn = Attention(
            hidden_size=self.d_new,
            head_dim=self.d_new // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=self.d_new,
            expansion=expansion,
        )

        # Projection to map the latent output back to the original dimension
        self.final_proj = nn.Linear(self.d_new, hidden_size, bias=False)
        self.norm_eps = norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Note: The `cos_sin` for RoPE is passed but not used here, as RoPE is typically
        applied to the original sequence length, which is altered by LMA. For simplicity,
        we omit positional encodings in the latent space. A more advanced implementation
        could re-calculate positional encodings for the new sequence length.
        """
        # --- LMA Path ---
        # 1. Transform to latent space
        z = self.lma_transform(hidden_states)

        # 2. Apply Attention in latent space (Post-Norm style)
        z_attn = self.self_attn(cos_sin=None, hidden_states=z) # RoPE disabled
        z = rms_norm(z + z_attn, variance_epsilon=self.norm_eps)

        # 3. Apply MLP in latent space
        z_mlp = self.mlp(z)
        z = rms_norm(z + z_mlp, variance_epsilon=self.norm_eps)

        # 4. Project back to original dimension
        output = self.final_proj(z)

        # The LMA transformation changes the sequence length (L -> H*L).
        # We must reshape the output back to the original sequence length L
        # to apply the residual connection.
        B, L_orig, d_orig = hidden_states.shape
        H = self.lma_transform.n_heads
        output_reshaped = output.view(B, H, L_orig, d_orig).transpose(1, 2).reshape(B, L_orig, d_orig * H)
        # This seems wrong, the final projection should handle the dimension restoration.
        # Let's re-think the projection. The output from the latent space is (B, L*H, d_new).
        # final_proj takes it to (B, L*H, d0). How to get back to (B, L, d0)?
        # Let's adjust the final projection to do this.
        # Let's simplify. We project (B, L*H, d_new) to (B, L, d_orig). This requires flattening.
        # Let's adjust self.final_proj
        d_latent_flat = (L_orig * H) * self.d_new
        d_orig_flat = L_orig * d_orig
        
        # This dynamic flattening is not clean. Let's stick to the Post-Norm structure
        # of the original block. `hidden_states + self.final_proj(z)` is the goal.
        # The shape mismatch `z` (B, L*H, d_new) and `hidden_states` (B, L, d0) is the problem.
        
        # Let's follow the PointNet implementation logic more closely.
        # The transform is applied, then attention is applied. The residual is on the latent var.
        # Let's simplify the LMA block to not be a drop-in replacement but a component
        # that changes the tensor shape, and the main model must handle it.
        # The user's request was to make it a block replacement.
        # Let's make a strong assumption: the projection back must work.
        # The most direct way is to average or pool over the H dimension.
        
        # Let's try this:
        output_reshaped = output.view(B, H, L_orig, self.d_new).mean(dim=1) # Average over H meta-heads
        projected_output = self.final_proj(output_reshaped)

        # 5. Final residual connection
        final_hidden_states = rms_norm(hidden_states + projected_output, variance_epsilon=self.norm_eps)

        return final_hidden_states
