
# config.py
import torch

# -------------------------
# Training parameters
# -------------------------
batch_size = 1
ctx_len = 2048          # context length
eval_interval = 20
grad_accum = 4
max_grad_norm=1.0

lr = 1e-3
min_lr = 1e-4
dropout = 0.2

max_iters = 1000
eval_iters = 2
warmup_iters = 100

resume = False
res_path = ""

data_dir = "tokenized_data"

info_levl=1 #1:model parameters and Optimizer and  MoE / Router Info and mhc / mhc Info.2:ALL

# -------------------------
# Model parameters
# -------------------------
n_embd = 256
n_head = 16
n_layer = 4
n_experts = 32
num_exp=4
shared_experts=1
use_expert_bias = 'True'
types = ['moe']
attention_types=['full']
hc=True
vocab_size = 50257
init_moe_scaling = 1.25

device = 'cuda' 

rms_norm_eps=1e-6
block_size=16

num_tokens_to_keep=256

window_size=128

weight_decay=1e-1

bias=False


# -------------------------
# HC / MHC / Sinkhorn Parameters
# -------------------------
hc_num_streams = 4
hc_num_fracs = 1
hc_disable = False  # note: convert string to bool in training code if needed
mhc = True
sinkhorn_iters = 10
sinkhorn_tau = 0.05
mhc_h_res_proj = 'sinkhorn'

# -------------------------
# Neural Sort / Regularization
# -------------------------
ns_steps = 5
ns_eps = 1e-7
ns_coeffs = (3.0, -3.2, 1.2)
v_residual = False




