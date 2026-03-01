
# config.py
import torch

# -------------------------
# Training parameters
# -------------------------
batch_size = 1
ctx_len = 1024          # context length
eval_interval = 20
grad_accum = 8
max_grad_norm=1.0

lr = 1e-5
min_lr = 2e-7
dropout = 0.2

max_iters = 1000
eval_iters = 2
warmup_iters = 500

resume = True
res_path = ""

data_dir = "tokenized_data"

info_levl=1 #1:model parameters and Optimizer and  MoE / Router Info and mhc / mhc Info.2:ALL

# -------------------------
# Model parameters
# -------------------------
n_embd = 768
n_head = 16
n_layer = 12
n_experts = 8
num_exp=1
shared_experts=1
use_expert_bias = 'True'
types = ['moe']
attention_types=['Spares']
hc=True
vocab_size = 151643
init_moe_scaling = 1.25

device = 'cuda' 

rms_norm_eps=1e-6
block_size=16

num_tokens_to_keep=128

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
# NSA Branch Configuration
nsa_use_branch1 = 1   # Coarse-grained compression (MLA)
nsa_use_branch2 = 1   # Token selection (NSA)
nsa_use_branch3 = 1   # Sliding window (NSA)
attention_mode=['SWA','NSA','DSA']




