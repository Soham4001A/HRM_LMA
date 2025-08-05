# HierarchicalReasoningModel_ACTV1.py (Modified)

from typing import Tuple, List, Dict
from dataclasses import dataclass
import math

import torch
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding
from LMA_features import LMAInitialTransform # LMA Integration

# ... (HierarchicalReasoningModel_ACTV1InnerCarry and HierarchicalReasoningModel_ACTV1Carry remain unchanged) ...

@dataclass
class HierarchicalReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class HierarchicalReasoningModel_ACTV1Carry:
    inner_carry: HierarchicalReasoningModel_ACTV1InnerCarry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class HierarchicalReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int
    H_cycles: int
    L_cycles: int
    H_layers: int
    L_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    
    # --- LMA Integration ---
    use_lma: bool = False
    lma_heads: int = 4
    lma_latent_dim: int = None # If None, defaults to hidden_size // 2
    # ---------------------

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    halt_max_steps: int
    halt_exploration_prob: float
    forward_dtype: str = "bfloat16"

class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    # This standard block is now used for both MHA and LMA paths,
    # just configured with different dimensions.
    def __init__(self, hidden_size: int, num_heads: int, expansion: float, norm_eps: float) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        self.mlp = SwiGLU(hidden_size=hidden_size, expansion=expansion)
        self.norm_eps = norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states

class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class HierarchicalReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: HierarchicalReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # Determine the dimension for the reasoning modules
        self.reasoning_dim = config.hidden_size
        if config.use_lma:
            self.reasoning_dim = config.lma_latent_dim or (config.hidden_size // 2)
            print(f"INFO: LMA enabled. Reasoning dimension set to {self.reasoning_dim}")
            self.lma_transform = LMAInitialTransform(d0=config.hidden_size, n_heads=config.lma_heads, d_new=self.reasoning_dim)
            self.lma_final_proj = CastedLinear(self.reasoning_dim, config.hidden_size, bias=False)
        else:
            print(f"INFO: LMA disabled. Reasoning dimension is {self.reasoning_dim}")

        # --- I/O and Embeddings (always operate in original hidden_size) ---
        self.embed_scale = math.sqrt(config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        # (Puzzle embedding logic remains the same)

        # --- Positional Encodings ---
        # Note: RoPE is applied before LMA transform if enabled.
        if config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=config.hidden_size // config.num_heads, max_position_embeddings=config.seq_len, base=config.rope_theta)

        # --- Reasoning Layers (operate in `self.reasoning_dim`) ---
        block_args = {
            "hidden_size": self.reasoning_dim,
            "num_heads": config.num_heads, # Can use same num_heads if reasoning_dim is multiple
            "expansion": config.expansion,
            "norm_eps": config.rms_norm_eps
        }
        self.H_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(**block_args) for _ in range(config.H_layers)]
        )
        self.L_level = HierarchicalReasoningModel_ACTV1ReasoningModule(
            layers=[HierarchicalReasoningModel_ACTV1Block(**block_args) for _ in range(config.L_layers)]
        )
        
        # --- Initial states (must be in `self.reasoning_dim`) ---
        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(self.reasoning_dim, dtype=self.forward_dtype), std=1))
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(self.reasoning_dim, dtype=self.forward_dtype), std=1))
        
        # (Q head init remains the same)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input_ids: torch.Tensor):
        embedding = self.embed_tokens(input_ids.to(torch.int32))
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        device = self.H_init.device
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len, self.reasoning_dim, dtype=self.forward_dtype, device=device),
            z_L=torch.empty(batch_size, self.config.seq_len, self.reasoning_dim, dtype=self.forward_dtype, device=device),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: HierarchicalReasoningModel_ACTV1InnerCarry):
        return HierarchicalReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: HierarchicalReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # --- Positional Encoding (applied to original embeddings) ---
        cos_sin_orig = self.rotary_emb() if hasattr(self, "rotary_emb") else None
        
        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"])
        # Apply RoPE to input embeddings before any transformation
        if cos_sin_orig:
             input_embeddings = self.self_attn.apply_rotary_pos_emb(input_embeddings, cos_sin_orig)

        # --- LMA Transformation Step (if enabled) ---
        if self.config.use_lma:
            # Note: RoPE is not passed to latent blocks as sequence length is preserved
            # but feature space is different.
            seq_info = dict(cos_sin=None)
            # Transform inputs and states into the latent space
            input_injection = self.lma_transform(input_embeddings)
        else:
            seq_info = dict(cos_sin=cos_sin_orig)
            input_injection = input_embeddings

        # --- Reasoning Loop (operates entirely in `self.reasoning_dim`) ---
        with torch.no_grad():
            z_H, z_L = carry.z_H, carry.z_L
            for _ in range(self.config.H_cycles):
                for _ in range(self.config.L_cycles - 1):
                    z_L = self.L_level(z_L, z_H + input_injection, **seq_info)
                z_L = self.L_level(z_L, z_H + input_injection, **seq_info) # last L step
                if _ < self.config.H_cycles - 1:
                    z_H = self.H_level(z_H, z_L, **seq_info)
        
        # 1-step grad
        z_L = self.L_level(z_L, z_H + input_injection, **seq_info)
        z_H = self.H_level(z_H, z_L, **seq_info)

        # --- Final Projection and Output Heads ---
        final_z_H = z_H
        if self.config.use_lma:
            final_z_H = self.lma_final_proj(z_H)

        new_carry = HierarchicalReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(final_z_H)
        q_logits = self.q_head(final_z_H[:, 0]).to(torch.float32)
        
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

# ... (HierarchicalReasoningModel_ACTV1 wrapper class remains unchanged) ...
class HierarchicalReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = HierarchicalReasoningModel_ACTV1Config(**config_dict)
        self.inner = HierarchicalReasoningModel_ACTV1_Inner(self.config)

    # ... all other methods of this wrapper class are identical ...
    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return HierarchicalReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),
            steps=torch.zeros((batch_size, ), dtype=torch.int32, device=self.inner.H_init.device),
            halted=torch.ones((batch_size, ), dtype=torch.bool, device=self.inner.H_init.device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }
        
        with torch.no_grad():
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            halted = is_last_step

            if self.training and (self.config.halt_max_steps > 1):
                halted = halted | (q_halt_logits > q_continue_logits)
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                next_q_halt_logits, next_q_continue_logits = self.inner(new_inner_carry, new_current_data)[-1]
                outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return HierarchicalReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
