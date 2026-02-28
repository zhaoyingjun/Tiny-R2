import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect
from muon import Muon
import config

from hyper_connections.hyper_connections import get_init_and_expand_reduce_stream_functions
from value_residual import ValueResidualState


# =============================================================================
# Full Attention (Causal Self-Attention)
# =============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Value residual connections
        self.v_residual = config.v_residual
        if self.v_residual:
            self.lamb1 = nn.Parameter(torch.tensor(0.5))
            self.lamb2 = nn.Parameter(torch.tensor(0.5))
        else:
            self.lamb1 = 1.0
            self.lamb2 = 0.0

        # Flash attention check
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            bias = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("bias", bias.view(1, 1, config.block_size, config.block_size))

    def forward(self, x, vrl_state=None):
        B, T, C = x.size()

        # Compute Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply value residual if enabled
        if self.v_residual:
            if vrl_state is None:
                raise ValueError("v_residual requires vrl_state")
            v = vrl_state.mix(v, self.lamb1, self.lamb2)

        # Attention computation
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reshape and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# =============================================================================
# Rotary Positional Embeddings (RoPE)
# =============================================================================

class RoPE(nn.Module):
    def __init__(self, d, base=100_000_000_000, device=config.device):
        super().__init__()
        self.base = base
        self.d = d
        self.device = device
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x):
        if self.cos_cached is not None:
            return
        
        head_dim = x.shape[-1]
        theta = 1.0 / (self.base ** (torch.arange(0, head_dim, 2, device=self.device).float() / self.d))
        seq_idx = torch.arange(x.shape[0], device=self.device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

        cos_cache = torch.cos(idx_theta)
        sin_cache = torch.sin(idx_theta)
        
        self.cos_cached = torch.cat([cos_cache, cos_cache], dim=-1).unsqueeze(0).unsqueeze(0)
        self.sin_cached = torch.cat([sin_cache, sin_cache], dim=-1).unsqueeze(0).unsqueeze(0)

    def _neg_half(self, x):
        head_dim = x.shape[-1]
        d_2 = head_dim // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

    def forward(self, x):
        if self.cos_cached is None or self.cos_cached.shape[2] != x.shape[1]:
            self._build_cache(x)

        x_rope = x.clone()
        neg_half_x = self._neg_half(x_rope)
        x_out = (x_rope * self.cos_cached[:, :, :x.shape[1], :]) + \
                (neg_half_x * self.sin_cached[:, :, :x.shape[1], :])
        return x_out


def precompute_freqs_cis(dim, end, device, theta=10000.0):
    """Precompute frequency cis for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, y: torch.Tensor, freqs_cis) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors."""
    cos_freqs, sin_freqs = freqs_cis
    seq_len = x.shape[-2]

    cos_seq = cos_freqs[:seq_len].unsqueeze(0).unsqueeze(0)
    sin_seq = sin_freqs[:seq_len].unsqueeze(0).unsqueeze(0)

    x_real, x_imag = x.chunk(2, dim=-1)
    y_real, y_imag = y.chunk(2, dim=-1)

    x_rotated_real = x_real * cos_seq - x_imag * sin_seq
    x_rotated_imag = x_real * sin_seq + x_imag * cos_seq
    y_rotated_real = y_real * cos_seq - y_imag * sin_seq
    y_rotated_imag = y_real * sin_seq + y_imag * cos_seq

    x_rotated = torch.cat([x_rotated_real, x_rotated_imag], dim=-1)
    y_rotated = torch.cat([y_rotated_real, y_rotated_imag], dim=-1)
    
    return x_rotated.type_as(x), y_rotated.type_as(y)


# =============================================================================
# MLA-NSA Hybrid Attention
# =============================================================================

class Attn(nn.Module):
    """
    Native Sparse Attention with Multi-headed Latent Attention integration.
    Combines MLA's compression techniques with NSA's natural sparsity.
    Supports configurable branches via config.
    """
    
    def __init__(self,atten_mode):
        super().__init__()
        self.device = config.device
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.ctx_len = config.ctx_len
        self.rms_norm_eps = config.rms_norm_eps
        self.atten_mode=atten_mode
        

        # Branch configuration (add to your config.py):
        # config.nsa_use_branch1 = True  # Coarse-grained compression (MLA)
        # config.nsa_use_branch2 = True  # Token selection (NSA)
        # config.nsa_use_branch3 = True  # Sliding window (NSA)
        self.use_branch1 = getattr(config, 'nsa_use_branch1', True)
        self.use_branch2 = getattr(config, 'nsa_use_branch2', True)
        self.use_branch3 = getattr(config, 'nsa_use_branch3', True)
        
        # Validate at least one branch is enabled
        if not any([self.use_branch1, self.use_branch2, self.use_branch3]):
            raise ValueError("At least one NSA branch must be enabled!")

        # MLA parameters
        self.v_head_dim = 32
        self.kv_lora_rank = 32
        self.q_lora_rank = 3 * self.kv_lora_rank
        self.rope_head_dim = 64
        self.nope_head_dim = 32
        self.value_dim = self.n_head * self.v_head_dim
        self.nope_dim = self.n_head * self.nope_head_dim
        self.rope_dim = self.n_head * self.rope_head_dim

        # NSA parameters
        self.block_size = config.block_size
        self.num_blocks = self.ctx_len // self.block_size
        self.window_size = config.window_size
        self.num_tokens_to_keep = config.num_tokens_to_keep
        
        if self.atten_mode=="SWA":
          self.use_branch1,self.use_branch2,self.use_branch3=[1,0,1]
        elif self.atten_mode=="NSA":
          self.use_branch1,self.use_branch2,self.use_branch3=[1,1,0]
        else:
          self.use_branch1,self.use_branch2,self.use_branch3=[1,1,1]
        

        # === Branch 1: Coarse-grained compression (MLA) ===
        if self.use_branch1:
            self.compress_q_linear = nn.Linear(self.n_embd, self.q_lora_rank, bias=False)
            self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=self.rms_norm_eps)
            self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_dim, bias=False)
            self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_dim, bias=False)

            self.compress_kv_linear = nn.Linear(self.n_embd, self.kv_lora_rank, bias=False)
            self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=self.rms_norm_eps)
            self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_dim, bias=False)
            self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_dim, bias=False)
            self.k_rope_linear = nn.Linear(self.n_embd, self.rope_head_dim, bias=False)

        # === Branch 2: Token Selection (NSA) ===
        if self.use_branch2:
            self.importance_scorer = nn.Linear(self.n_embd, 1, bias=False)
            self.selection_k = nn.Linear(self.n_embd, self.n_head * (self.rope_head_dim + self.nope_head_dim), bias=False)
            self.selection_v = nn.Linear(self.n_embd, self.value_dim, bias=False)

        # === Branch 3: Sliding Window (NSA) ===
        if self.use_branch3:
            self.window_k = nn.Linear(self.n_embd, self.n_head * (self.rope_head_dim + self.nope_head_dim), bias=False)
            self.window_v = nn.Linear(self.n_embd, self.value_dim, bias=False)

        # Token Compression (NSA) - needed for branch1 if used
        if self.use_branch1:
            self.block_compressor = nn.Sequential(
                nn.Linear(self.block_size * self.n_embd, 4 * self.n_embd, bias=False),
                nn.GELU(),
                nn.Linear(4 * self.n_embd, self.n_embd, bias=False)
            )
            self.intra_block_pos_encoding = nn.Parameter(torch.randn(1, self.block_size, self.n_embd))

        # Gated Multi-Branch Integration - adjust gate size based on active branches
        num_active_branches = sum([self.use_branch1, self.use_branch2, self.use_branch3])
        self.branch_gate = nn.Linear(self.n_embd, num_active_branches, bias=False)

        # Output projection
        self.proj = nn.Linear(self.value_dim, self.n_embd, bias=False)
        self.res_dropout = nn.Dropout(p=self.dropout)

        # Caching for inference
        self.k_cache = None
        self.v_cache = None
        self.cache_filled = 0

        # RoPE
        self.rope = RoPE(self.rope_head_dim, device=self.device)
        self.freqs_cis = precompute_freqs_cis(self.rope_head_dim, self.ctx_len, self.device)

    def _compress_tokens(self, x):
        """Token compression mechanism from NSA."""
        B, T, C = x.size()
        
        padded_len = ((T + self.block_size - 1) // self.block_size) * self.block_size
        if padded_len > T:
            padding = torch.zeros(B, padded_len - T, C, device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=1)
        else:
            x_padded = x

        blocks = x_padded.view(B, -1, self.block_size, C)
        pos_encoded_blocks = blocks + self.intra_block_pos_encoding
        blocks_flat = pos_encoded_blocks.view(B, -1, self.block_size * C)
        compressed_blocks = self.block_compressor(blocks_flat)

        return compressed_blocks

    def _select_important_tokens(self, x, importance_scores):
        """Select most important tokens based on scores."""
        B, T, _ = x.size()
        
        _, indices = torch.topk(
            importance_scores.squeeze(-1),
            min(self.num_tokens_to_keep, T),
            dim=1
        )
        indices, _ = torch.sort(indices, dim=1)
        
        batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, indices.size(1))
        selected_tokens = x[batch_indices, indices]

        return selected_tokens, indices

    def _get_sliding_window_tokens(self, x, current_pos=None):
        """Extract tokens within sliding window."""
        if self.training or current_pos is None:
            return x
        else:
            B, T, _ = x.size()
            window_start = max(0, current_pos - self.window_size // 2)
            window_end = min(T, window_start + self.window_size)
            return x[:, window_start:window_end]

    def _prepare_queries(self, x):
        """Prepare queries using MLA approach."""
        B, T, _ = x.size()
        
        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope = self.decompress_q_nope(norm_q)
        query_rope = self.decompress_q_rope(norm_q)

        query_nope = query_nope.view(B, T, self.n_head, self.nope_head_dim).transpose(1, 2)
        query_rope = query_rope.view(B, T, self.n_head, self.rope_head_dim).transpose(1, 2)

        q_rope, _ = apply_rope(query_rope, query_rope, self.freqs_cis)

        q_recombined = torch.empty(
            (B, self.n_head, T, self.rope_head_dim + self.nope_head_dim),
            device=x.device, dtype=x.dtype
        )
        q_recombined[:, :, :, :self.nope_head_dim] = query_nope
        q_recombined[:, :, :, self.nope_head_dim:] = q_rope

        return q_recombined

    def _branch1_compression(self, x):
        """Coarse-grained compression branch."""
        B, T, _ = x.size()
        
        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope = self.decompress_k_nope(norm_kv)
        value = self.decompress_v_linear(norm_kv)
        key_rope = self.k_rope_linear(x)

        key_nope = key_nope.view(B, T, self.n_head, self.nope_head_dim).transpose(1, 2)
        key_rope = key_rope.view(B, T, 1, self.rope_head_dim).transpose(1, 2)
        value = value.view(B, T, self.n_head, self.v_head_dim).transpose(1, 2)

        key_rope = key_rope / self.n_head
        _, k_rope = apply_rope(key_rope, key_rope, self.freqs_cis)

        k_recombined = torch.empty(
            (B, self.n_head, T, self.rope_head_dim + self.nope_head_dim),
            device=x.device, dtype=x.dtype
        )
        k_recombined[:, :, :, :self.nope_head_dim] = key_nope
        k_recombined[:, :, :, self.nope_head_dim:] = k_rope

        return k_recombined, value

    def _branch2_selection(self, x):
        """Token selection branch."""
        importance_scores = self.importance_scorer(x)
        selected_tokens, _ = self._select_important_tokens(x, importance_scores)
        
        B, S, _ = selected_tokens.size()
        k_selected = self.selection_k(selected_tokens)
        v_selected = self.selection_v(selected_tokens)

        k_selected = k_selected.view(B, S, self.n_head, -1).transpose(1, 2)
        v_selected = v_selected.view(B, S, self.n_head, self.v_head_dim).transpose(1, 2)

        k_selected_rope = k_selected[:, :, :, self.nope_head_dim:]
        k_selected_nope = k_selected[:, :, :, :self.nope_head_dim]
        _, k_selected_rope = apply_rope(k_selected_rope, k_selected_rope, self.freqs_cis)

        k_selected[:, :, :, self.nope_head_dim:] = k_selected_rope
        k_selected[:, :, :, :self.nope_head_dim] = k_selected_nope

        return k_selected, v_selected

    def _branch3_window(self, x):
        """Sliding window branch."""
        window_tokens = self._get_sliding_window_tokens(x)
        B, W, _ = window_tokens.size()

        k_window = self.window_k(window_tokens)
        v_window = self.window_v(window_tokens)

        k_window = k_window.view(B, W, self.n_head, -1).transpose(1, 2)
        v_window = v_window.view(B, W, self.n_head, self.v_head_dim).transpose(1, 2)

        k_window_rope = k_window[:, :, :, self.nope_head_dim:]
        k_window_nope = k_window[:, :, :, :self.nope_head_dim]
        _, k_window_rope = apply_rope(k_window_rope, k_window_rope, self.freqs_cis)

        k_window[:, :, :, self.nope_head_dim:] = k_window_rope
        k_window[:, :, :, :self.nope_head_dim] = k_window_nope

        return k_window, v_window

    def forward(self, x):
        B, T, C = x.size()

        # Prepare queries (always needed)
        q_recombined = self._prepare_queries(x)
        
        # Compute branch gates
        branch_weights = F.softmax(self.branch_gate(x).mean(dim=1), dim=-1)  # [B, num_active_branches]

        # Collect outputs from active branches
        branch_outputs = []
        branch_idx = 0

        # Branch 1: Compression
        if self.use_branch1:
            k_recombined_1, value_1 = self._branch1_compression(x)
            
            if self.training:
                self.cache_filled = 0
                output_1 = F.scaled_dot_product_attention(
                    q_recombined, k_recombined_1, value_1,
                    is_causal=True, dropout_p=self.dropout
                )
            else:
                # Update cache
                if self.k_cache is None or self.k_cache.size(0) != B:
                    self.k_cache = torch.zeros(
                        B, self.n_head, self.ctx_len, self.rope_head_dim + self.nope_head_dim,
                        device=self.device, dtype=x.dtype
                    )
                    self.v_cache = torch.zeros(
                        B, self.n_head, self.ctx_len, self.v_head_dim,
                        device=self.device, dtype=x.dtype
                    )
                    self.cache_filled = 0

                new_cache_filled = min(self.cache_filled + T, self.ctx_len)
                k_to_cache = k_recombined_1[:, :, :new_cache_filled - self.cache_filled]
                v_to_cache = value_1[:, :, :new_cache_filled - self.cache_filled]

                self.k_cache[:, :, self.cache_filled:new_cache_filled] = k_to_cache
                self.v_cache[:, :, self.cache_filled:new_cache_filled] = v_to_cache
                self.cache_filled = new_cache_filled

                k1 = self.k_cache[:, :, :self.cache_filled]
                v1 = self.v_cache[:, :, :self.cache_filled]
                
                output_1 = F.scaled_dot_product_attention(
                    q_recombined, k1, v1, is_causal=True, dropout_p=0
                )
            
            branch_outputs.append(output_1 * branch_weights[:, branch_idx:branch_idx+1].view(B, 1, 1, 1))
            branch_idx += 1

        # Branch 2: Selection
        if self.use_branch2:
            k_selected, v_selected = self._branch2_selection(x)
            output_2 = F.scaled_dot_product_attention(
                q_recombined, k_selected, v_selected,
                is_causal=False, dropout_p=self.dropout if self.training else 0
            )
            branch_outputs.append(output_2 * branch_weights[:, branch_idx:branch_idx+1].view(B, 1, 1, 1))
            branch_idx += 1

        # Branch 3: Window
        if self.use_branch3:
            k_window, v_window = self._branch3_window(x)
            output_3 = F.scaled_dot_product_attention(
                q_recombined, k_window, v_window,
                is_causal=True, dropout_p=self.dropout if self.training else 0
            )
            branch_outputs.append(output_3 * branch_weights[:, branch_idx:branch_idx+1].view(B, 1, 1, 1))
            branch_idx += 1

        # Sum all branch outputs
        blended_output = sum(branch_outputs)

        # Final processing
        output = blended_output.transpose(1, 2).contiguous().view(B, T, self.value_dim)
        output = self.proj(output)
        output = self.res_dropout(output)

        return output


# =============================================================================
# MLP
# =============================================================================

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        n_embd = config.n_embd
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU squared, not GELU
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


# =============================================================================
# DeepSeek MoE (DS-MoE)
# =============================================================================

class UnitCenteredNoise(nn.Module):
    def __init__(self, scaling=0.02):
        super().__init__()
        self.scaling = scaling
        self.base = 1 - (scaling * 0.5)

    def forward(self, x):
        if self.training:
            noise = torch.rand(x.size(), device=x.device, dtype=x.dtype)
            noise_centered = (noise * self.scaling) + self.base
            return x * noise_centered
        return x


def moe_load_balance_loss(router_weights, num_experts, shared_expert=True, eps=1e-9):
    """Compute load balancing loss for MoE routing."""
    if shared_expert:
        router_weights = router_weights[:, 1:]
        num_experts = num_experts - 1

    load = router_weights.sum(dim=0)
    load = load / (load.sum() + eps)
    ideal = torch.full_like(load, 1.0 / num_experts)
    loss = num_experts * torch.sum((load - ideal) ** 2)
    
    return loss


class DSMoE(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.hidden_dim = config.n_embd * 2
        self.num_experts = config.n_experts
        self.num_exp = config.num_exp
        self.moe_scaling = config.init_moe_scaling
        
        self.experts = nn.ModuleList([MLP() for _ in range(self.num_experts)])
        self.gate = nn.Sequential(
            nn.Linear(config.n_embd, self.num_experts - 1, bias=False),
            UnitCenteredNoise(scaling=0.02),
            nn.Softmax(dim=-1)
        )
        
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts - 1), requires_grad=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)

        gate_val_continuous = self.gate(x_flat)
        biased_gate_vals = gate_val_continuous + self.expert_bias

        gate_vals, gate_val_indices = torch.topk(biased_gate_vals, self.num_exp - 1, dim=-1)
        gate_vals = gate_vals / gate_vals.sum(dim=-1, keepdim=True)

        # Prepend shared expert
        shared_expert_weight = torch.ones_like(gate_vals[:, :1]) / self.num_exp
        gate_vals = torch.cat([shared_expert_weight, gate_vals * (self.num_exp - 1) / self.num_exp], dim=-1)
        gate_val_indices = torch.cat([
            torch.zeros_like(gate_val_indices[:, :1]),
            gate_val_indices + 1
        ], dim=-1)

        # Process all experts
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=0)

        # Create routing weights matrix
        router_weights = torch.zeros(x_flat.size(0), self.num_experts, device=x.device)
        for i in range(self.num_exp):
            idx = gate_val_indices[:, i:i+1]
            val = gate_vals[:, i:i+1]
            router_weights.scatter_add_(1, idx, val)

        # Apply routing weights
        weighted_outputs = expert_outputs * router_weights.transpose(0, 1).unsqueeze(-1)
        output = weighted_outputs.sum(dim=0)

        return output.reshape(B, T, C), router_weights


# =============================================================================
# Attention Branch Wrapper
# =============================================================================

class AttnBranch(nn.Module):
    def __init__(self, norm, attn):
        super().__init__()
        self.norm = norm
        self.attn = attn

    def forward(self, x, vrl_state=None):
        x = self.norm(x)
        return self.attn(x)


# =============================================================================
# Transformer Block
# =============================================================================

class Block(nn.Module):
    def __init__(self, index, init_hc):
        super().__init__()
        n_embd = config.n_embd

        # Select attention type
        self.atten_types = config.attention_types[index % len(config.attention_types)]
        if self.atten_types == "FULL":
            self.attn = CausalSelfAttention(config)
        elif self.atten_types == "Spares":
            self.atten_mode=config.attention_mode[index % len(config.attention_mode)]
            self.attn = Attn(self.atten_mode)
        else:
            raise ValueError(f"Invalid Attention type: {self.atte_types}")

        # Select FFN type
        self.ffn_type = config.types[index % len(config.types)]
        if self.ffn_type == "mlp":
            self.ffn = MLP()
        elif self.ffn_type == "moe":
            self.ffn = DSMoE(index)
        else:
            raise ValueError(f"Invalid layer type: {self.ffn_type}")

        # Normalization
        self.rm1 = nn.RMSNorm(n_embd)
        self.rm2 = nn.RMSNorm(n_embd)

        # Attention branch wrapper
        self.attn_branch = AttnBranch(self.rm1, self.attn)

        # Hyper-connection kwargs
        hc_kwargs = dict(
            mhc=config.mhc,
            sinkhorn_iters=config.sinkhorn_iters,
            sinkhorn_tau=config.sinkhorn_tau,
            mhc_h_res_proj=config.mhc_h_res_proj,
            ns_steps=config.ns_steps,
            ns_eps=config.ns_eps,
            ns_coeffs=config.ns_coeffs,
        )

        # Initialize hyper-connections
        self.hc_attn = init_hc(
            dim=config.n_embd,
            branch=self.attn_branch,
            layer_index=index * 2,
            **hc_kwargs,
        )

        self.hc_mlp = init_hc(
            dim=config.n_embd,
            branch=nn.Sequential(self.rm2, self.ffn),
            layer_index=index * 2 + 1,
            **hc_kwargs,
        )

    def forward(self, x, vrl_state=None):
        if config.hc:
            x = self.hc_attn(x, vrl_state=vrl_state)
            
            if self.ffn_type == "moe":
                x_ffn, router_weights = self.hc_mlp(self.rm2(x))
                return x_ffn, router_weights
            else:
                x_ffn = self.hc_mlp(self.rm2(x))
                return x_ffn, None
        else:
            x = x + self.attn(self.rm1(x))
            
            if self.ffn_type == "moe":
                x_ffn, router_weights = self.ffn(self.rm2(x))
                return x + x_ffn, router_weights
            else:
                x_ffn = self.ffn(self.rm2(x))
                return x + x_ffn, None


# =============================================================================
# Transformer Model
# =============================================================================

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.init_hc, self.expand_stream, self.reduce_stream = get_init_and_expand_reduce_stream_functions(
            config.hc_num_streams,
            num_fracs=config.hc_num_fracs,
            disable=config.hc_disable,
        )

        # Embeddings
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.ctx_len, config.n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(i, self.init_hc) for i in range(config.n_layer)])
        
        # Final normalization and head
        self.rm_f = nn.RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.token_embedding_table.weight = self.lm_head.weight
        
        # Initialization
        self.apply(self._init_weights)
        self.total_params = sum(p.numel() for p in self.parameters())
        
        # Value residual state
        self.vrl_state = ValueResidualState() if config.v_residual else None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Embeddings
        tok_emb = self.token_embedding_table(idx).clone()
        pos_emb = self.position_embedding_table(torch.arange(T, device=config.device))
        x = tok_emb + pos_emb
        
        # Expand stream for hyper-connections
        x = self.expand_stream(x)
        
        # Reset value residual state
        vrl_state = self.vrl_state
        if vrl_state is not None:
            vrl_state.reset()

        # Collect router weights from MoE layers
        all_router_weights = []
        
        for block in self.blocks:
            x, router_weights = block(x, vrl_state=vrl_state)
            if router_weights is not None:
                all_router_weights.append(router_weights)

        # Final processing
        x = self.rm_f(x)
        x = self.reduce_stream(x)
        logits = self.lm_head(x)

        # Compute loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, all_router_weights

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,top_p=None, tiktoken_vocab_size=None):
        """
        Generates sequences of tokens autoregressively.
        """
        if temperature <= 0:
            print("Warning: Temperature <= 0. Using a very small value (1e-6) instead.")
            temperature = 1e-6

        model_vocab_size = config.vocab_size
        use_vocab_mask = False
        effective_vocab_size = model_vocab_size
        
        if tiktoken_vocab_size is not None:
            if tiktoken_vocab_size < model_vocab_size:
                print(f"generate(): Masking logits for indices >= {tiktoken_vocab_size}")
                use_vocab_mask = True
                effective_vocab_size = tiktoken_vocab_size
            elif tiktoken_vocab_size > model_vocab_size:
                print(f"Warning: tiktoken_vocab_size > model_vocab_size. Masking ineffective.")

        for _ in range(max_new_tokens):
            start_pos = max(0, idx.size(1) - config.ctx_len)
            idx_cond = idx[:, start_pos:]

            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature

            # Apply vocabulary masking
            if use_vocab_mask:
                logits[:, tiktoken_vocab_size:] = -float('Inf')

            # Apply top-k filtering
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                top_k_values, _ = torch.topk(logits, k=k, dim=-1)
                kth_logit_value = top_k_values[:, [-1]]
                logits[logits < kth_logit_value] = -float('Inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            
            # Top-p nucleus sampling
            if top_p is not None and top_p > 0:
               sorted_probs, sorted_indices = torch.sort(probs, descending=True)
               cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # remove tokens with cumulative prob above threshold
               sorted_indices_to_remove = cumulative_probs > top_p
               sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
               sorted_indices_to_remove[..., 0] = 0

               sorted_probs[sorted_indices_to_remove] = 0
               probs = torch.zeros_like(probs).scatter(-1, sorted_indices, sorted_probs)

               probs = probs / probs.sum(dim=-1, keepdim=True)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        # Calculate KV cache size
        total_size_gb = 0
        if hasattr(self, 'blocks') and self.blocks is not None:
            for block in self.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, 'k_cache') and block.attn.k_cache is not None:
                    size_bytes = block.attn.k_cache.numel() * block.attn.k_cache.element_size()
                    total_size_gb += size_bytes / (1024 ** 3)
                if hasattr(block, 'attn') and hasattr(block.attn, 'v_cache') and block.attn.v_cache is not None:
                    size_bytes = block.attn.v_cache.numel() * block.attn.v_cache.element_size()
                    total_size_gb += size_bytes / (1024 ** 3)

        return idx, total_size_gb

    def configure_optimizers(self, weight_decay, learning_rate, device):
        """
        Configures optimizers: Muon for >=2D parameters in blocks, AdamW for others.
        """
        muon_params = []
        adamw_params = []

        muon_exclude_patterns = [
            'attn.intra_block_pos_encoding',
            'attn.importance_scorer.weight',
            'attn.importance_scorer.bias',
            'attn.block_compressor',
        ]

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            is_excluded = any(pattern in name for pattern in muon_exclude_patterns)

            if 'blocks' in name and param.ndim >= 2 and not is_excluded:
                muon_params.append(param)
            else:
                adamw_params.append(param)

        num_muon_params = sum(p.numel() for p in muon_params)
        num_adamw_params = sum(p.numel() for p in adamw_params)
        print(f"num Muon parameters: {num_muon_params:,}")
        print(f"num AdamW parameters: {num_adamw_params:,}")

        if not muon_params:
            print("\n*** WARNING: Muon parameter list is EMPTY! Proceeding with only AdamW. ***")
            return [torch.optim.AdamW(adamw_params, lr=learning_rate, betas=(0.90, 0.95), weight_decay=weight_decay)]

        return [
            Muon(muon_params, lr=0.02, momentum=0.95),
            torch.optim.AdamW(adamw_params, lr=learning_rate, betas=(0.90, 0.95), weight_decay=weight_decay)
        ]

    def update_expert_biases(self, all_router_weights, update_rate):
        """Update expert biases based on load balancing statistics."""
        with torch.no_grad():
            j = 0
            for block in self.blocks:
                if isinstance(block.ffn, DSMoE):
                    router_weights = all_router_weights[j]
                    j += 1

                    c_i = router_weights[:, 1:].sum(dim=0)
                    total_routed_tokens = c_i.sum()
                    c_i_bar = total_routed_tokens / (block.ffn.num_experts - 1)
                    e_i = c_i - c_i_bar

                    block.ffn.expert_bias.add_(update_rate * torch.sign(e_i))
