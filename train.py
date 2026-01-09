# Training Loop (Optimized with MoE / Router Loss and CUDA-safe batches)

import os
import math
import time
import glob
import torch
import string
import random
import pickle
import argparse
import numpy as np
from contextlib import nullcontext

import torch.amp as amp
import torch.distributed as dist

import model
from model import Transformer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import plot
from plot import plot_loss

from muon import Muon
import config

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=config.batch_size)
parser.add_argument('--ctx_len', type=int, default=config.ctx_len)
parser.add_argument('--lr', type=float, default=config.lr)
parser.add_argument('--max_iters', type=int, default=config.max_iters)
parser.add_argument('--eval_iters', type=int, default=config.eval_interval)
parser.add_argument('--warmup_iters', type=int, default=config.warmup_iters)
parser.add_argument('--data_dir', type=str, default=config.data_dir)
parser.add_argument('--n_embd', type=int, default=config.n_embd)
parser.add_argument('--n_head', type=int, default=config.n_head)
parser.add_argument('--n_layer', type=int, default=config.n_layer)
parser.add_argument('--n_experts', type=int, default=config.n_experts)


args = parser.parse_args()
config.batch_size=args.batch_size
config.ctx_len=args.ctx_len
config.lr=args.lr
config.max_iters=args.max_iters
config.eval_interval=args.eval_iters
config.warmup_iters=args.warmup_iters
config.data_dir=args.data_dir
config.n_embd=args.n_embd
config.n_head=args.n_head
config.n_layer=args.n_layer
config.n_experts=args.n_experts





# ---------------- Arguments ----------------
batch_size = config.batch_size
block_size = config.ctx_len
eval_interval = config.eval_interval
grad_accum_steps = config.grad_accum

lr = 3 * config.lr    
min_lr = config.min_lr
dropout = config.dropout
max_iters = config.max_iters
eval_iters = config.eval_iters
warmup_iters = config.warmup_iters
resume = config.resume
resume_checkpoint = config.res_path
data_dir = config.data_dir
device = config.device
weight_decay = config.weight_decay
max_grad_norm=config.max_grad_norm
n_layer=config.n_layer
n_head=config.n_head
n_embd=config.n_embd
batch_size=config.batch_size
block_size=config.block_size
learning_rate=config.lr
max_iters=config.max_iters
hc_num_streams=config.hc_num_streams
hc_num_fracs= config.hc_num_fracs
            
mhc=config.mhc,
sinkhorn_iters=config.sinkhorn_iters
sinkhorn_tau=config.sinkhorn_tau
mhc_h_res_proj=config.mhc_h_res_proj
ns_steps=config.ns_steps
ns_eps=config.ns_eps
ns_coeffs=config.ns_coeffs

         
           

import wandb

# --- Initialize WandB ---
# wandb logging
wandb_log = True
wandb_project = "Tiny-R2"
wandb_run_name = "baseline"
wandb_log_layer_stats = True
wandb_log_layer_cosine = True



if wandb_log :

    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config={
            "dataset": data_dir,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "batch_size": batch_size,
            "block_size": block_size,
            "learning_rate": learning_rate,
            "max_iters": max_iters,
            "hc_num_streams": hc_num_streams,
            "hc_num_fracs": hc_num_fracs,
           
            "mhc": mhc,
            "sinkhorn_iters": sinkhorn_iters,
            "sinkhorn_tau": sinkhorn_tau,
            "mhc_h_res_proj": mhc_h_res_proj,
            "ns_steps": ns_steps,
            "ns_eps": ns_eps,
            "ns_coeffs": ns_coeffs,
           
        },
    )



# ---------------- Distributed init ----------------
distributed_initialized = False
if 'cuda' in device:
    try:
        backend = 'nccl' if dist.is_nccl_available() else 'gloo'
        init_url = "tcp://localhost:12355"
        dist.init_process_group(backend=backend, init_method=init_url, world_size=1, rank=0)
        distributed_initialized = True
    except Exception as e:
        print(f"Distributed init failed: {e}")

# ---------------- Mixed precision ----------------
ctx = nullcontext() if device=='cpu' else torch.amp.autocast(device_type=device, dtype=torch.float16)
scaler = amp.GradScaler(enabled=('cuda' in device))
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# ---------------- Data batch function ----------------
def get_batch(split):
    split_filenames = glob.glob(os.path.join("data", data_dir, f"{split}_*.bin"))
    if not split_filenames:
        raise FileNotFoundError(f"No {split} shard files in {data_dir}")
    shard_file = np.random.choice(split_filenames)
    try:
        data = np.memmap(shard_file, dtype=np.uint16, mode='r', offset=1024)
    except Exception as e:
        print(f"Error reading shard {shard_file}: {e}")
        return get_batch(split)

    num_tokens = len(data)
    if num_tokens <= block_size + 1:
        return get_batch(split)

    ix = torch.randint(0, num_tokens - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])

    # 🔹 Safety clamp to avoid CUDA index errors
    vocab_size = config.vocab_size
    x = torch.clamp(x, 0, vocab_size-1)
    y = torch.clamp(y, 0, vocab_size-1)

    if (y.max() >= vocab_size) or (x.max() >= vocab_size):
        print(f"WARNING: token id exceeds vocab_size ({vocab_size}), clamped")

    x, y = (x.to(device, non_blocking=True), y.to(device, non_blocking=True)) if device=='cuda' else (x.to(device), y.to(device))
    return x, y

# ---------------- Vocab ----------------
meta_path = f"data/{data_dir}/meta.pkl"
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    model.config.vocab_size = meta['vocab_size']
else:
    print(f"meta.pkl not found, using default vocab_size={model.config.get('vocab_size',256)}")

# ---------------- Loss estimation ----------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss,_ = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# ---------------- Model / Optimizer / Scheduler ----------------
start_iter = 0
scheduler = None

if resume:
    checkpoint = torch.load(args.res_path, map_location=device)
    model.config.update(checkpoint.get('config', {}))
    model = Transformer().to(device)
    state_dict = checkpoint['model']
    # unwrap compiled keys if needed
    new_state = {k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k:v for k,v in state_dict.items()}
    model.load_state_dict(new_state)
    optimizers = model.configure_optimizers(weight_decay, lr, device)
    adamw_optimizer = optimizers[-1]
    scheduler = SequentialLR(adamw_optimizer, schedulers=[
        LinearLR(adamw_optimizer,start_factor=1e-3,total_iters=warmup_iters),
        CosineAnnealingLR(adamw_optimizer,T_max=max_iters-warmup_iters,eta_min=min_lr)
    ], milestones=[warmup_iters])
    start_iter = checkpoint.get('iter',0)+1
else:
    model = Transformer().to(device)
    optimizers = model.configure_optimizers(weight_decay, lr, device)
    if len(optimizers)==1:
        adamw_optimizer = optimizers[0]
    elif len(optimizers)==2:
        adamw_optimizer = optimizers[1]
    scheduler = SequentialLR(adamw_optimizer, schedulers=[
        LinearLR(adamw_optimizer,start_factor=1e-3,total_iters=warmup_iters),
        CosineAnnealingLR(adamw_optimizer,T_max=max_iters-warmup_iters,eta_min=min_lr)
    ], milestones=[warmup_iters])

# Compile model
if 'cuda' in device:
    try:
        model = torch.compile(model, fullgraph=False, dynamic=False)
    except:
        pass



# ---------------- Detailed Model & Optimizer Summary with init_hc ----------------
def print_detailed_summary(model, optimizers):
    print("\n================ Model & Optimizer Summary ================\n")
    
  
    # 1️⃣ 总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params/1e6:.3f} M\n")
    if config.info_levl==2:
    # 2️⃣ 按层参数
     print("--- Layer-wise Parameters ---")
     for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50} | shape={tuple(param.shape)} | params={param.numel()}")

    # 3️⃣ Transformer / Attention info
    if hasattr(model, 'config'):
        print("\n--- Transformer / Attention Info ---")
        n_layer = config.n_layer
        n_head = config.n_head
        n_embd = config.n_embd
        print(f"Number of layers: {n_layer}")
        print(f"Attention heads: {n_head}")
        print(f"Embedding / hidden size: {n_embd}")
        attn_type_list = getattr(model, 'attn_type', ['SelfAttention']*n_layer)
        for i, t in enumerate(attn_type_list):
            print(f"Layer {i} attention type: {t}")

    # 4️⃣ MoE / Router info
    print("\n--- MoE / Router Info ---")
    n_experts = config.n_experts
    active_experts=config.num_exp
    shared_experts=config.shared_experts
    use_bias = config.use_expert_bias
    print(f"Number of experts: {n_experts}")
    print(f"Number of active_expets: {active_experts}")
    print(f"Number of shared_experts: {shared_experts}")
    print(f"Expert bias used: {use_bias}")

    # 5️⃣ init_hc / mhc info
    print("\n--- mhc / mhc Info ---")
    if hasattr(model, 'expand_stream'):
        expand_stream = getattr(model, 'expand_stream')
        print(f"expand_stream type: {type(expand_stream)}")
        
    if hasattr(model, 'reduce_stream'):
        reduce_stream = getattr(model, 'reduce_stream')
        print(f"reduce_stream type: {type(reduce_stream)}")
      
    # 6️⃣ Optimizer info (区分 AdamW 和 Muon)
    print("\n--- Optimizer Info ---")
    for i, opt in enumerate(optimizers):
        opt_type = type(opt).__name__
        for j, pg in enumerate(opt.param_groups):
            lr = pg.get('lr', 'N/A')
            wd = pg.get('weight_decay', 'N/A')
            print(f"Optimizer {i} ({opt_type}) - param_group {j}: lr={lr}, weight_decay={wd}")
        if opt_type.lower() == 'muon':
            moe_params = getattr(opt, 'num_experts', 'N/A')
            print(f"  -> Muon optimizer handles {moe_params} experts")

    print("\n===========================================================\n")
    return total_params




# ---------------- Training Loop ----------------
train_losses_history = []
val_losses_history = []

time_s = time.time()
prev_time = time_s
#print(model.config.__dict__)
total_params = print_detailed_summary(model, optimizers)
try:
    for iter in range(start_iter, max_iters + 1):

        step_start = time.time()

        # ---- Eval ----
        if (iter % eval_interval == 0 or (iter < 100 and iter % 10 == 0) or iter == max_iters) and iter > 0:
            losses = estimate_loss()
            val_losses_history.append(losses['val'])
            print(f"[Eval] Step {iter}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")

            model_state_to_save = model.state_dict()
            if hasattr(model, '_orig_mod'):
                 model_state_to_save = model._orig_mod.state_dict()
            optimizer_states_to_save = [opt.state_dict() for opt in optimizers]

            checkpoint = {
                'model': model_state_to_save,
                'optimizer_states':optimizer_states_to_save,
                'scheduler': scheduler.state_dict() if scheduler else None,
                'iter': iter,
                'train_losses_history': train_losses_history,
                'val_losses_history': val_losses_history,
                'config': model.config
            }
            if losses['val'] < 10.28:
                torch.save(checkpoint,f'/content/NanoPoor/src/checkpoints/check_{iter}.pt')
    
        
        
        
        # ---- Training Step ----
        if iter == max_iters: break

        loss_accum = 0.0
        all_router_weights = []

        for micro_step in range(grad_accum_steps):
            xb, yb = get_batch('train')
            with ctx:
                logits, loss, rw = model(xb, yb)
                # 🔹 Add lb_loss from MoE / router
                #if hasattr(model,'lb_loss'):
                 #   loss = ce_loss
                #else:
                    #loss = ce_loss
                loss = loss / grad_accum_steps

            if rw: 
              all_router_weights.extend(rw)
            scaler.scale(loss).backward()
            loss_accum += loss.item() * grad_accum_steps

        train_losses_history.append(loss_accum / grad_accum_steps)

        for opt in optimizers:
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        for opt in optimizers:
            scaler.step(opt)
            opt.zero_grad(set_to_none=True)

        scaler.update()
        if scheduler: scheduler.step()

        if all_router_weights and hasattr(model,'update_expert_biases'):
            model.update_expert_biases(all_router_weights,1e-3)

        # ---- Step Timing & TPS ----
        step_end = time.time()
        step_time_ms = (step_end - step_start) * 1000.0
        tokens_processed = batch_size * block_size * grad_accum_steps
        tps = tokens_processed / (step_end - step_start)
        print(f"[Step {iter}] trian_loss={loss_accum/grad_accum_steps:.4f}, step_time={step_time_ms:.2f} ms, TPS={tps:.2f} tokens/sec")



        





finally:
    if distributed_initialized and dist.is_initialized():
        dist.destroy_process_group()
    print("Training finished or interrupted.")
