# Tiny-R2: A Hybrid Architecture Combining SWA, CSA, HCA, mHC and DSMoE Under the DeepSeek V4 Design Paradigm




<img width="1536" height="1024" alt="模型结构" src="https://github.com/user-attachments/assets/2a7c317f-5b61-4abf-8fdf-56dc4e613002" />
<img width="833" height="441" alt="benchimark" src="https://github.com/user-attachments/assets/bae360e3-b48a-4f01-94ce-07e3c0fc7d3b" />
<img width="2131" height="891" alt="loss" src="https://github.com/user-attachments/assets/7821bf23-1e81-406b-a867-2396bbe0da63" />











# Tiny-R2 模型架构与训练流程文档


---

## 📋 目录

1. [项目概述](#项目概述)
2. [模型架构总览](#模型架构总览)
3. [核心组件详解](#核心组件详解)
   - 3.1 [注意力机制](#31-注意力机制)
   - 3.2 [HCA-CSA 混合注意力](#32-HCA-nsa-混合注意力)
   - 3.3 [前馈网络与 MoE](#33-前馈网络与-moe)
   - 3.4 [Hyper-Connections](#34-hyper-connections)
4. [训练流程](#训练流程)
5. [关键技术特性](#关键技术特性)
6. [附录：图表索引](#附录图表索引)

---

## 项目概述

Tiny-R2 是一个紧凑型但功能强大的语言模型，结合了多种先进的深度学习技术：

- **稀疏注意力机制** (HCA-CSA Hybrid Attention)
- **专家混合模型** (DeepSeek MoE)
- **超连接技术** (Hyper-Connections)
- **双优化器策略** (Muon + AdamW)

---

## 模型架构总览


<img width="1536" height="1024" alt="模型结构" src="https://github.com/user-attachments/assets/2a7c317f-5b61-4abf-8fdf-56dc4e613002" />


---

## 快速启动
### 2.1 安装依赖
```
pip install tiktoken datasets transformers huggingface_hub
pip install git+https://github.com/KellerJordan/Muon

```

### 2.2 启动训练

```
python train.py --n_layer 6 --n_embd 768 --hc 'True' --mhc 'True' --n_experts 32  --max_iters 10000 --attention_types 'Sparse' --batch_size 8 --ctx_len 2048 --hf_dataset 'karpathy/climbmix-400b-shuffle' --resume True --save_best_only True
```
### 2.3 验证模型训练效果，PPL
```
python evaluate.py --checkpoint checkpoints/best_model_step_4720.pt
```

## 核心组件详解

### 3.1 注意力机制

Tiny-R2 支持两种注意力类型，通过配置 `attention_types` 灵活切换：

#### 3.1.1 CausalSelfAttention (Full Attention)

标准的因果自注意力机制：

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        # Projections: Q, K, V from single linear
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        # Value residual connections
        self.v_residual = config.v_residual
        self.lamb1 = nn.Parameter(torch.tensor(0.5))
        self.lamb2 = nn.Parameter(torch.tensor(0.5))
        
        # Flash Attention support
        self.flash = hasattr(F, "scaled_dot_product_attention")
```

**关键特性：**
- 使用 Flash Attention 加速（如果可用）
- 支持 Value Residual Connections
- 标准的因果掩码

#### 3.1.2 HCA-NSA Hybrid Attention

结合 Multi-head Latent Attention (HCA) 和 Native Sparse Attention (NSA) 的混合注意力机制。

**三种运行模式：**

| 模式 | 分支配置 | 说明 |
|------|----------|------|
| `HCA` | [1, 0, 0] | 启用所有三个分支 |
| `SWA` | [0, 0, 1] | 压缩分支 + 滑动窗口分支 |
| `CSA` | [0, 1, 0] | 压缩分支 + 选择分支 |

---

### 3.2 HCA-NSA 混合注意力

HCA-NSA 是 Tiny-R2 的核心创新之一，通过三个并行分支实现高效的稀疏注意力计算。

#### 架构流程

```
                       Input x
                          ↓
┌──────────────────────────────────────────────────────────────┐
│ Query Preparation (HCA style)                                │
│   compress_q → q_norm → decompress_q → RoPE → Query         │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────┬──────────────────┬──────────────────────┐
│   Branch 1       │   Branch 2       │   Branch 3           │
│   Compression    │   Selection      │   Sliding Window     │
│   (HCA)          │   (CSA)          │   (SWA)              │
├──────────────────┼──────────────────┼──────────────────────┤
│ compress_kv      │ importance_score │ window_k/v           │
│ kv_norm          │ topk selection   │ sliding_window       │
│ decompress_k/v   │ selection_k/v    │ RoPE                 │
│ k_rope           │ RoPE             │                      │
│ K/V Recombine    │ K/V Selected     │ K/V Window           │
└──────────────────┴──────────────────┴──────────────────────┘
    ↓                    ↓                    ↓
┌──────────────────────────────────────────────────────────────┐
│ Attention Computation                                        │
│   Attention 1: (Q @ K1.T) @ V1                               │
│   Attention 2: (Q @ K2.T) @ V2                               │
│   Attention 3: (Q @ K3.T) @ V3                               │
└──────────────────────────────────────────────────────────────┘
    ↓
branch_gate (Linear + Softmax) → Weighted Sum
    ↓
proj (Linear) → res_dropout → Output
```

#### 关键参数

```python
# HCA 参数
self.v_head_dim = 32
self.kv_lora_rank = 32
self.q_lora_rank = 3 * self.kv_lora_rank
self.rope_head_dim = 64
self.nope_head_dim = 32

# NSA 参数
self.block_size = config.block_size      # Token压缩块大小
self.window_size = config.window_size    # 滑动窗口大小
self.num_tokens_to_keep = config.num_tokens_to_keep  # 选择保留的token数
```

---

### 3.3 前馈网络与 MoE

#### 3.3.1 MLP

标准的前馈网络，使用 ReLU² 激活函数：

```python
class MLP(nn.Module):
    def __init__(self):
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()  # ReLU squared
        x = self.c_proj(x)
        return x
```

#### 3.3.2 DSMoE (DeepSeek Mixture of Experts)

DeepSeek 风格的专家混合模型：

```
Input x [B, T, C]
    ↓
Gate Network (Linear + UnitCenteredNoise)
    ↓
Softmax → Top-k Selection
    ↓
┌─────────────────────────────────────────────────────────────┐
│                    Expert Networks                          │
│  ┌──────────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Shared Exp 0 │  │ Expert 1 │  │ Expert 2 │  │  ...   │ │
│  │ (Always On)  │  │ (Top-k)  │  │ (Top-k)  │  │ (Top-k)│ │
│  └──────────────┘  └──────────┘  └──────────┘  └────────┘ │
└─────────────────────────────────────────────────────────────┘
    ↓
Weighted Sum of Expert Outputs
    ↓
Output [B, T, C]
```

**关键特性：**

| 特性 | 说明 |
|------|------|
| Shared Expert | 始终激活的共享专家，提供稳定性 |
| Routed Experts | Top-k 选择的路由专家 |
| Load Balance Loss | 防止专家崩溃的负载均衡损失 |
| Expert Bias | 可学习的专家偏置，用于路由优化 |
| UnitCenteredNoise | 训练时添加噪声以增加探索 |

**Load Balance Loss 计算：**

```python
def moe_load_balance_loss(router_weights, num_experts):
    load = router_weights.sum(dim=0)
    load = load / load.sum()
    ideal = torch.full_like(load, 1.0 / num_experts)
    loss = num_experts * torch.sum((load - ideal) ** 2)
    return loss
```

---

### 3.4 Hyper-Connections

Hyper-Connections 是 Tiny-R2 的另一大创新，通过多流路由机制增强信息流动。

**核心概念：**

```python
# 初始化 Hyper-Connections
self.init_hc, self.expand_stream, self.reduce_stream = \
    get_init_and_expand_reduce_stream_functions(
        config.hc_num_streams,
        num_fracs=config.hc_num_fracs,
        disable=config.hc_disable,
    )

# 在每个 Block 中使用
self.hc_attn = init_hc(
    dim=config.n_embd,
    branch=self.attn_branch,
    layer_index=index * 2,
    mhc=config.mhc,
    sinkhorn_iters=config.sinkhorn_iters,
    sinkhorn_tau=config.sinkhorn_tau,
)
```

**关键参数：**

| 参数 | 说明 |
|------|------|
| `hc_num_streams` | 超连接流数量 |
| `hc_num_fracs` | 分段数量 |
| `mhc` | 多超连接配置 |
| `sinkhorn_iters` | Sinkhorn 算法迭代次数 |
| `sinkhorn_tau` | Sinkhorn 温度参数 |

---

## 训练流程

### 4.1 初始化阶段

```
Parse Arguments → Update Config → Init WandB → Setup Distributed → Setup AMP
```

### 4.2 数据准备

```
Load HF Dataset (flytech/python-codes-25k)
    ↓
Init GPT2 Tokenizer
    ↓
Create TokenBuffer
```

**TokenBuffer 功能：**
- 流式读取 HuggingFace 数据集
- 动态填充 token buffer
- 生成连续的 token batch

### 4.3 模型初始化

```
Create Transformer
    ↓
Configure Optimizers (Muon + AdamW)
    ↓
Create LR Scheduler (Warmup + Cosine)
```

### 4.4 训练循环

```
For iter in range(max_iters):
    │
    ├── For step in grad_accum_steps:
    │       ├── Get Batch (TokenBuffer)
    │       ├── Forward Pass (model)
    │       ├── Backward Pass (scaler.scale)
    │       └── Collect Router Weights
    │
    ├── Gradient Clipping (clip_grad_norm_)
    ├── Optimizer Steps (Muon + AdamW)
    ├── Update Scaler (scaler.update)
    ├── LR Scheduler Step
    ├── Update Expert Biases (load balancing)
    └── Log Metrics (WandB)
```

### 4.5 评估与保存

```
If iter % eval_interval == 0:
    ├── Estimate Loss (eval mode)
    ├── Save Checkpoint (if val_loss < 5.27)
    └── Log to WandB
```

### 4.6 优化器配置

Tiny-R2 使用双优化器策略：

```python
def configure_optimizers(self, weight_decay, learning_rate, device):
    muon_params = []    # ≥2D parameters in blocks
    adamw_params = []   # Other parameters
    
    for name, param in self.named_parameters():
        if 'blocks' in name and param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    return [
        Muon(muon_params, lr=0.02, momentum=0.95),
        torch.optim.AdamW(adamw_params, lr=learning_rate, 
                          betas=(0.90, 0.95), weight_decay=weight_decay)
    ]
```

---

## 关键技术特性

### 5.1 注意力机制对比

| 特性 | CausalSelfAttention | HCA-NSA Hybrid |
|------|---------------------|----------------|
| 计算复杂度 | O(n²) | O(n) ~ O(n log n) |
| 内存使用 | 高 | 低 |
| 适用场景 | 短序列 | 长序列 |
| 分支数量 | 1 | 3 (可配置) |

### 5.2 FFN 类型对比

| 特性 | MLP | DSMoE |
|------|-----|-------|
| 参数量 | 固定 | 共享 + 路由 |
| 计算量 | 固定 | 稀疏激活 |
| 表达能力 | 标准 | 更强 |
| 训练稳定性 | 高 | 需要负载均衡 |

### 5.3 核心配置参数

```python
# 模型架构
n_embd = 512        # 嵌入维度
n_head = 8          # 注意力头数
n_layer = 8         # 层数
n_experts = 8       # 专家数量
num_exp = 2         # 每token激活的专家数

# 注意力配置
attention_types = ["FULL", "Spares", ...]  # 每层注意力类型
attention_mode = ["FULL", "SWA", "NSA"]    # 稀疏注意力模式

# Hyper-Connections
hc = True           # 启用超连接
hc_num_streams = 4  # 流数量

# 训练
batch_size = 32
ctx_len = 512       # 上下文长度
lr = 1e-3
warmup_iters = 1000
max_iters = 100000
```

---

## 附录：图表索引

本文档配套图表保存在 `/mnt/okcomputer/output/` 目录：

| 文件名 | 说明 |
|--------|------|
| `model_architecture.png` | 模型整体架构图 |
| `HCA_nsa_attention.png` | HCA-NSA 混合注意力详细结构图 |
| `dsmoe_architecture.png` | DSMoE 专家混合结构图 |
| `training_pipeline.png` | 完整训练流程图 |
| `tinyr2_overview.png` | Tiny-R2 综合概览图 |

---

## 参考资料

- [Tiny-R2 GitHub Repository](https://github.com/zhaoyingjun/Tiny-R2)
- [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434)
- [Native Sparse Attention (NSA)](https://arxiv.org/abs/2502.04543)
- [Hyper-Connections Paper](https://arxiv.org/abs/2409.19607)

---


