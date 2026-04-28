
# config.py
import torch

# -------------------------
# Training parameters
# -------------------------
batch_size = 8
ctx_len = 2048          # context length
eval_interval = 20
grad_accum = 16
max_grad_norm=1.0

lr = 1.5e-4
min_lr = 2e-7
dropout = 0.05

max_iters = 10000
eval_iters = 20
warmup_iters = 200

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
n_experts = 32
num_exp=2
shared_experts=1
use_expert_bias = 'True'
types = ['moe']
attention_types=['Sparse']
hc=True
vocab_size = 32768
init_moe_scaling = 1.25

device = 'cuda' 

rms_norm_eps=1e-6
block_size=16

num_tokens_to_keep=128

window_size=128

weight_decay=1e-2

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
cosine_decay = True

# -------------------------
# Neural Sort / Regularization
# -------------------------
ns_steps = 5
ns_eps = 1e-7
ns_coeffs = (3.0, -3.2, 1.2)
v_residual = True
# NSA Branch Configuration
nsa_use_branch1 = 1   # Coarse-grained compression (MLA)
nsa_use_branch2 = 1   # Token selection (DSA)
nsa_use_branch3 = 1   # Sliding window (SWA)
attention_mode=['SWA','SWA','DSA','MLA','DSA','MLA','MLA','DSA','DSA','DSA','DSA','MLA']




