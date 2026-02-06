# Tiny-R2
A better combination: DSA, mHC, and DSMoE  similar to DeepSeek R2.
https://wandb.ai/yingjun-xuda/Tiny-R2/reports/train-loss-26-01-20-21-52-23---VmlldzoxNTY4NzA2Nw

## 概述

本文档详细描述了一个融合多种前沿技术的Tiny-R2模型架构，根据Deepseek最新的论文，集成DSA、mHC、DSMoE的结构，在优化器上采用Muon和AdamW混合优化器架构。该模型集成了以下前沿创新：

- **MLA-NSA 混合注意力**: 结合Multi-head Latent Attention的压缩技术和Native Sparse Attention的稀疏性
- **Hyper-connections**: 多头流处理机制，支持HC、mHC等选择
- **DSMoE**: DeepSpeek混合专家层，支持Deepseek类的moe层构建
- **Value Residual Learning**: 跨层值向量残差学习
- **RoPE**: 旋转位置编码
- - **Muon优化器**: 采用Muon优化节约HBM、加速收敛速度

---

## 1. 整体架构

```
Input Tokens (Embedding) ──┬──> Positional Embedding
                           │
                           ▼
                         [ + ]
                           │
                           ▼
              Expand Stream (Hyper-connection)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐       ┌─────────┐
   │ Block 1 │  -->  │ Block 2 │  -->  │ Block N │
   │Attn+FFN │       │Attn+MoE │       │Attn+FFN │
   └─────────┘       └─────────┘       └─────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           │
              Value Residual State (VRL)
                           │
                           ▼
                       RMSNorm
                           │
              Reduce Stream (Hyper-connection)
                           │
                           ▼
                       LM Head
                           │
                           ▼
                    Logits / Next Token
```



### 1.1 核心组件

| 组件 | 描述 | 配置参数 |
|------|------|----------|
| Token Embedding | 词嵌入层 | `vocab_size`, `n_embd` |
| Positional Embedding | 位置编码 | `ctx_len` |
| Expand Stream | 超连接流扩展 | `hc_num_streams` |
| Transformer Blocks | 变换器块堆叠 | `n_layer` |
| RMSNorm | 均方根归一化 | `rms_norm_eps` |
| Reduce Stream | 超连接流归约 | - |
| LM Head | 语言模型头 | 权重共享 |

---

## 2. Block 详细结构

每个Transformer Block包含两个主要分支：Attention分支和FFN分支。

```
                    x (Input)
                      │
                      ▼
                  RMSNorm ────────────┐
                      │               │
                      ▼               │ Residual
              ┌───────────────┐       │ Connection
              │  Attention    │       │
              │ (Full/MLA-NSA)│       │
              └───────────────┘       │
                      │               │
                      ▼               │
           Hyper-connection (hc_attn) │
                      │               │
                      ▼               │
                  RMSNorm ────────────┤
                      │               │
                      ▼               │
              ┌───────────────┐       │
              │  FFN / MoE    │       │
              │ (MLP/DSMoE)   │       │
              └───────────────┘       │
                      │               │
                      ▼               │
           Hyper-connection (hc_mlp)  │
                      │               │
                      └───────────────┘
                      │
                      ▼
            Output + router_weights
```

### 2.1 Attention 类型

Block支持两种Attention类型，通过配置动态选择：

#### 2.1.1 CausalSelfAttention (全注意力)

标准的多头自注意力机制，支持Flash Attention优化。

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)  # QKV投影
        self.c_proj = nn.Linear(n_embd, n_embd)      # 输出投影
        self.flash = hasattr(F, "scaled_dot_product_attention")
```

**关键特性**:
- Flash Attention支持（如果可用）
- Value Residual Learning集成
- 因果掩码确保自回归特性

#### 2.1.2 Attn (MLA-NSA 混合注意力)

结合MLA的压缩技术和NSA的稀疏性的混合注意力机制。

---

## 3. MLA-NSA 混合注意力架构

这是模型的核心创新，结合了三种注意力分支：

### 3.1 架构概览

```
                              x (Input)
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│  Branch 1     │      │  Branch 2     │      │  Branch 3     │
│ Coarse-grained│      │ Token Select  │      │ Sliding Window│
│ Compression   │      │ (NSA)         │      │ (SWA)         │
│ (MLA)         │      │               │      │               │
└───────────────┘      └───────────────┘      └───────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
   output_1                 output_2                 output_3
        │                        │                        │
        └────────────────────────┼────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Blend with gate weights │
                    │  out = w1*o1 + w2*o2 +   │
                    │        w3*o3             │
                    └────────────┬────────────┘
                                 │
                                 ▼
                            proj (Linear)
                                 │
                                 ▼
                          res_dropout
                                 │
                                 ▼
                              Output
```

### 3.2 Branch 1: 粗粒度压缩 (MLA风格)

```python
# Query压缩路径
compressed_q = compress_q_linear(x)      # n_embd → q_lora_rank
norm_q = q_norm(compressed_q)
query_nope = decompress_q_nope(norm_q)   # → nope_dim
query_rope = decompress_q_rope(norm_q)   # → rope_dim

# KV压缩路径
compressed_kv = compress_kv_linear(x)    # n_embd → kv_lora_rank
norm_kv = kv_norm(compressed_kv)
key_nope = decompress_k_nope(norm_kv)
value = decompress_v_linear(norm_kv)
key_rope = k_rope_linear(x)

# 应用RoPE并重组
q_rope, _ = apply_rope(query_rope, query_rope, freqs_cis)
_, k_rope = apply_rope(key_rope, key_rope, freqs_cis)

# 重组为完整Q/K
q_recombined = [query_nope | q_rope]
k_recombined = [key_nope | k_rope]
```

**关键参数**:
| 参数 | 值 | 描述 |
|------|-----|------|
| `v_head_dim` | 32 | Value头维度 |
| `kv_lora_rank` | 32 | KV压缩秩 |
| `q_lora_rank` | 96 | Query压缩秩 (3*kv_lora_rank) |
| `rope_head_dim` | 64 | RoPE维度 |
| `nope_head_dim` | 32 | 非RoPE维度 |

### 3.3 Branch 2: Token选择 (NSA)

```python
# 计算重要性分数
importance_scores = importance_scorer(x)  # Linear → score

# 选择Top-k重要token
_, indices = torch.topk(importance_scores, num_tokens_to_keep)
indices = torch.sort(indices)  # 保持序列连续性

# 获取选中token的独立KV
k_selected = selection_k(selected_tokens)
v_selected = selection_v(selected_tokens)

# 应用RoPE
k_selected_rope = apply_rope(k_selected[:,:,:,nope_head_dim:])
```

### 3.4 Branch 3: 滑动窗口 (SWA)

```python
# 提取滑动窗口内的token
window_tokens = x[:, window_start:window_end]

# 独立KV投影
k_window = window_k(window_tokens)
v_window = window_v(window_tokens)
```

### 3.5 门控融合

```python
# 计算分支权重
branch_weights = softmax(branch_gate(x).mean(dim=1))  # [B, 3]

# 融合三个分支输出
blended_output = (
    output_1 * branch_weights[:, 0].view(B, 1, 1, 1) +
    output_2 * branch_weights[:, 1].view(B, 1, 1, 1) +
    output_3 * branch_weights[:, 2].view(B, 1, 1, 1)
)
```

### 3.6 KV缓存 (推理模式)

```python
# 初始化缓存
self.k_cache = torch.zeros(B, n_head, ctx_len, rope_head_dim + nope_head_dim)
self.v_cache = torch.zeros(B, n_head, ctx_len, v_head_dim)

# 更新缓存
self.k_cache[:, :, cache_filled:new_filled] = k_recombined
self.v_cache[:, :, cache_filled:new_filled] = value
```

---

## 4. DSMoE (DeepSpeeK 混合专家)

### 4.1 架构

```
x [batch, seq, n_embd]
           │
           ▼
      Flatten [batch*seq, n_embd]
           │
           ▼
┌─────────────────────┐
│ gate (Linear +      │
│ UnitCenteredNoise)  │
└─────────────────────┘
           │
           ▼
      softmax → gate_val_continuous
           │
           ▼
      topk (num_exp-1) + shared expert
           │
           ▼
      Normalize gate_vals
           │
    ┌──────┴──────┬────────┬────────┐
    ▼             ▼        ▼        ▼
Expert 0     Expert 1  Expert 2  Expert 3
(Shared)      (MLP)     (MLP)     (MLP)
    │             │        │        │
    └─────────────┴────────┴────────┘
                  │
                  ▼
        Weighted Sum
    router_weights * outputs
                  │
                  ▼
      Output [batch, seq, n_embd]
```

### 4.2 关键实现

```python
class DSMoE(nn.Module):
    def __init__(self, index):
        self.experts = nn.ModuleList([MLP() for _ in range(n_experts)])
        self.gate = nn.Sequential(
            nn.Linear(n_embd, n_experts - 1),  # 排除共享专家
            UnitCenteredNoise(scaling=0.02),
            nn.Softmax(dim=-1)
        )
        self.expert_bias = nn.Parameter(torch.zeros(n_experts - 1), 
                                       requires_grad=False)
    
    def forward(self, x):
        # 门控计算
        gate_val_continuous = self.gate(x_flat)
        
        # 应用expert bias
        biased_gate_vals = gate_val_continuous + self.expert_bias
        
        # Top-k选择
        gate_vals, indices = torch.topk(biased_gate_vals, num_exp - 1)
        
        # 添加共享专家
        shared_weight = torch.ones_like(gate_vals[:, :1]) / num_exp
        gate_vals = torch.cat([shared_weight, gate_vals * (num_exp - 1) / num_exp])
        
        # 加权聚合
        output = sum(expert(x) * weight for expert, weight in zip(experts, gate_vals))
```

### 4.3 Expert Bias更新

```python
def update_expert_biases(all_router_weights, update_rate):
    for router_weights in all_router_weights:
        c_i = router_weights[:, 1:].sum(dim=0)  # 专家负载
        c_i_bar = c_i.sum() / (num_experts - 1)  # 平均负载
        e_i = c_i - c_i_bar  # 负载偏差
        expert_bias.add_(update_rate * torch.sign(e_i))
```

---

## 5. Hyper-connections (超连接,支持mHC)

### 5.1 原理

Hyper-connections实现多头流处理，通过Sinkhorn算法进行归一化。

```
Input Stream
     │
     ├───> Head 1 ────┐
     │                │
     ├───> Head 2 ────┤
     │                ├──> Branch Processing
     ├───> Head 3 ────┤    (Attn or MLP/MoE)
     │                │
     └───> Head 4 ────┘
                          │
                          ▼
              Sinkhorn Normalization
                   (mhc, sinkhorn_iters, tau)
                          │
                          ▼
                    Output Stream
```

### 5.2 关键参数

| 参数 | 描述 |
|------|------|
| `mhc` | 多头连接数 |
| `sinkhorn_iters` | Sinkhorn迭代次数 |
| `sinkhorn_tau` | Sinkhorn温度参数 |
| `hc_num_streams` | 流数量 |
| `hc_num_fracs` | 分数数量 |

---

## 6. Value Residual Learning (VRL)

### 6.1 原理

跨层传递Value向量，通过可学习参数混合当前层和前层的Value。

```
Layer i-1                    Layer i
   │                            │
   ▼                            ▼
v_prev ──────────────────> [ mix ]
                              │
                         v_current
                              │
                         v_mix = lamb1 * v_current + lamb2 * v_prev
```

### 6.2 实现

```python
class ValueResidualState:
    def __init__(self):
        self.v_prev = None
    
    def mix(self, v_current, lamb1, lamb2):
        if self.v_prev is None:
            self.v_prev = v_current
            return v_current
        v_mix = lamb1 * v_current + lamb2 * self.v_prev
        self.v_prev = v_mix
        return v_mix
    
    def reset(self):
        self.v_prev = None
```

### 6.3 参数

```python
if v_residual:
    self.lamb1 = nn.Parameter(torch.tensor(0.5))
    self.lamb2 = nn.Parameter(torch.tensor(0.5))
else:
    self.lamb1 = 1.0
    self.lamb2 = 0.0
```

---

## 7. RoPE (旋转位置编码)

### 7.1 实现

```python
class RoPE(nn.Module):
    def __init__(self, d, base=100_000_000_000):
        self.base = base
        
    def _build_cache(self, x):
        theta = 1 / (self.base ** (torch.arange(0, head_dim, 2).float() / d))
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        self.cos_cached = torch.cos(idx_theta)
        self.sin_cached = torch.sin(idx_theta)
```

### 7.2 应用

```python
def apply_rope(x, y, freqs_cis):
    cos_freqs, sin_freqs = freqs_cis
    x_real, x_imag = x.chunk(2, dim=-1)
    y_real, y_imag = y.chunk(2, dim=-1)
    
    x_rotated_real = x_real * cos_seq - x_imag * sin_seq
    x_rotated_imag = x_real * sin_seq + x_imag * cos_seq
    
    x_rotated = torch.cat([x_rotated_real, x_rotated_imag], dim=-1)
    return x_rotated, y_rotated
```

---

## 8. 训练流程

```
Input idx [B, T]
       │
       ▼
Token + Pos Embedding
       │
       ▼
expand_stream()
       │
       ├──> vrl_state.reset()
       │
       ▼
┌─────────────────────────────────────┐
│ For each Block (n_layer iterations) │
│                                     │
│  ┌───────────────────────────────┐  │
│  │ hc_attn (Hyper-connection)    │  │
│  │ RMSNorm                       │  │
│  │ FFN/MoE                       │  │
│  │ └──> router_weights (if MoE)  │  │
│  │     └──> all_router_weights[] │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
       │
       ▼
rm_f (RMSNorm)
       │
       ▼
reduce_stream()
       │
       ▼
lm_head (Linear)
       │
       ▼
CrossEntropy Loss

Returns: logits, loss, all_router_weights
```

---

## 9. 推理流程 (Generate)

```
idx [B, T] + max_new_tokens
       │
       ▼
┌─────────────────────────────────────────────┐
│ For _ in range(max_new_tokens)              │
│                                             │
│  Crop to ctx_len                            │
│       │                                     │
│       ▼                                     │
│  Forward Pass                               │
│       │                                     │
│       ▼                                     │
│  logits[:, -1, :]  (取最后一个token)         │
│       │                                     │
│       ▼                                     │
│  Temperature Scaling                        │
│       │                                     │
│       ├──> Top-k Filtering (if specified)   │
│       │                                     │
│       ▼                                     │
│  Softmax + Multinomial Sampling             │
│       │                                     │
│       ▼                                     │
│  cat(idx, idx_next)  (拼接新token)           │
│                                             │
└─────────────────────────────────────────────┘
       │
       ├──> Calculate KV Cache Size
       │
       ▼
Generated Sequence [B, T + max_new_tokens]
       │
       ▼
total_kv_cache_size_gb

Returns: idx, total_kv_cache_size_gb
```

---

## 10. 关键配置参数

| 类别 | 参数 | 默认值 | 描述 |
|------|------|--------|------|
| **模型** | `n_embd` | - | 嵌入维度 |
| | `n_head` | - | 注意力头数 |
| | `n_layer` | - | 层数 |
| | `ctx_len` | - | 上下文长度 |
| | `vocab_size` | - | 词表大小 |
| | `dropout` | - | Dropout率 |
| **Attention** | `v_head_dim` | 32 | Value头维度 |
| | `kv_lora_rank` | 32 | KV压缩秩 |
| | `q_lora_rank` | 96 | Query压缩秩 |
| | `rope_head_dim` | 64 | RoPE维度 |
| | `nope_head_dim` | 32 | 非RoPE维度 |
| **NSA** | `block_size` | - | Token块大小 |
| | `window_size` | - | 滑动窗口大小 |
| | `num_tokens_to_keep` | - | 保留的细粒度token数 |
| **MoE** | `n_experts` | - | 专家总数 |
| | `num_exp` | - | 每token激活的专家数 |
| **Hyper-conn** | `mhc` | - | 选择HC或mHC |
| | `sinkhorn_iters` | - | Sinkhorn迭代次数 |
| | `sinkhorn_tau` | - | Sinkhorn温度 |
| | `hc_num_streams` | - | 流数量 |
| | `hc_num_fracs` | - | 分数数量 |
| | `hc_disable` | - | 是否禁用 |
| **其他** | `v_residual` | - | 是否使用VRL |
| | `bias` | - | 是否使用偏置 |

---

## 11. 优化器配置

模型使用混合优化器策略：

| 参数类型 | 优化器 | 学习率 |
|----------|--------|--------|
| `blocks`内 ≥2D参数 | Muon | 0.02 |
| 其他参数 | AdamW | `learning_rate` |

### 11.1 Muon排除的参数

- `attn.intra_block_pos_encoding`
- `attn.importance_scorer.weight/bias`
- `attn.block_compressor`

### 11.2 AdamW配置

```python
torch.optim.AdamW(
    params, 
    lr=learning_rate, 
    betas=(0.90, 0.95), 
    weight_decay=weight_decay
)
```

---

## 12. 架构图文件

本文档配套的架构图文件：

1. **model_architecture.png** - 完整模型架构图
2. **model_components.png** - 组件详细说明
3. **model_dataflow.png** - 训练/推理数据流

---

## 13. 参考文献

- **MLA**: DeepSeek-V2 Technical Report
- **NSA**: Native Sparse Attention
- **Hyper-connections**: Hyper-connections
- **RoPE**: RoFormer: Enhanced Transformer with Rotary Position Embedding
- **MoE**: Switch Transformers: Scaling to Trillion Parameter Models
