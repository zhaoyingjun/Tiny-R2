# train_hf_codes_moe.py

import os
import time
import torch
import argparse
from contextlib import nullcontext

import torch.amp as amp
import torch.distributed as dist

from datasets import load_dataset
from transformers import AutoTokenizer

import model
from model import Transformer
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import config
import wandb

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=config.batch_size)
parser.add_argument('--ctx_len', type=int, default=config.ctx_len)
parser.add_argument('--lr', type=float, default=config.lr)
parser.add_argument('--max_iters', type=int, default=config.max_iters)
parser.add_argument('--eval_iters', type=int, default=config.eval_interval)
parser.add_argument('--warmup_iters', type=int, default=config.warmup_iters)




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
parser.add_argument('--hc', type=str, default=config.hc)
parser.add_argument('--mhc', type=str, default=config.mhc)
parser.add_argument('--attention_types',  nargs="+", type=str, default=config.attention_types)


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
config.hc=args.hc
config.mhc=args.mhc
config.attention_types=args.attention_types





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






# HF dataset
parser.add_argument('--hf_dataset', type=str, default="flytech/python-codes-25k")
parser.add_argument('--hf_split', type=str, default="train")
parser.add_argument('--hf_text_key', type=str, default="text")

args = parser.parse_args()

# ---------------- Config ----------------
batch_size = args.batch_size
block_size = args.ctx_len
grad_accum_steps = config.grad_accum
device = config.device

max_iters = args.max_iters
eval_interval = args.eval_iters
warmup_iters = args.warmup_iters

learning_rate = args.lr
min_lr = config.min_lr
weight_decay = config.weight_decay
max_grad_norm = config.max_grad_norm

# ---------------- WandB ----------------
wandb.init(
    project="Tiny-R2-code",
    name="flytech-python-codes-moe",
    config=vars(args)
)

# ---------------- Distributed (optional) ----------------
distributed_initialized = False
if 'cuda' in device:
    try:
        dist.init_process_group(
            backend='nccl',
            init_method="tcp://localhost:12355",
            world_size=1,
            rank=0
        )
        distributed_initialized = True
    except Exception as e:
        print("Distributed init failed:", e)

# ---------------- Mixed Precision ----------------
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(
    device_type="cuda", dtype=torch.float16
)
scaler = amp.GradScaler(enabled=('cuda' in device))

# ---------------- Tokenizer ----------------
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
config.vocab_size = tokenizer.vocab_size

# ---------------- Dataset ----------------
print(f"Loading HF dataset: {args.hf_dataset}")
raw_ds = load_dataset(args.hf_dataset, split=args.hf_split)
print(raw_ds)

# ---------------- TokenBuffer ----------------
class TokenBuffer:
    """
    Streaming token buffer for causal LM training.
    Packs variable-length code samples into contiguous token stream.
    """
    def __init__(self, dataset, tokenizer, text_key):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.iterator = iter(dataset)
        self.buffer = []

    def refill(self, min_tokens):
        while len(self.buffer) < min_tokens:
            try:
                ex = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
                ex = next(self.iterator)

            text = ex.get(self.text_key, "")
            if not text:
                continue

            ids = self.tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]

            if ids:
                self.buffer.extend(ids)

    def get_batch(self, batch_size, block_size, device):
        needed = batch_size * (block_size + 1)
        self.refill(needed)

        xs, ys = [], []
        for _ in range(batch_size):
            x = self.buffer[:block_size]
            y = self.buffer[1:block_size+1]
            self.buffer = self.buffer[block_size:]
            xs.append(torch.tensor(x, dtype=torch.long))
            ys.append(torch.tensor(y, dtype=torch.long))

        return (
            torch.stack(xs).to(device, non_blocking=True),
            torch.stack(ys).to(device, non_blocking=True)
        )

train_buffer = TokenBuffer(raw_ds, tokenizer, args.hf_text_key)

# ---------------- Model / Optimizer / Scheduler ----------------
model = Transformer().to(device)

optimizers = model.configure_optimizers(weight_decay, learning_rate, device)
adamw = optimizers[-1]

scheduler = SequentialLR(
    adamw,
    schedulers=[
        LinearLR(adamw, start_factor=1e-3, total_iters=warmup_iters),
        CosineAnnealingLR(
            adamw,
            T_max=max_iters - warmup_iters,
            eta_min=min_lr
        )
    ],
    milestones=[warmup_iters]
)

if 'cuda' in device:
    try:
        model = torch.compile(model)
    except:
        pass

# ---------------- Eval ----------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    total = 0.0
    for _ in range(eval_interval):
        xb, yb = train_buffer.get_batch(batch_size, block_size, device)
        with ctx:
            _, loss, _ = model(xb, yb)
        total += loss.item()
    model.train()
    return total / eval_interval




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






total_params = print_detailed_summary(model, optimizers)

# ---------------- Training Loop ----------------
print("🚀 Start training")
train_losses_history = []
val_losses_history = []
for it in range(max_iters):
    step_start = time.time()
    loss_accum = 0.0
    all_router_weights = []

    for _ in range(grad_accum_steps):
        xb, yb = train_buffer.get_batch(batch_size, block_size, device)
        with ctx:
            _, loss, rw = model(xb, yb)
            loss = loss / grad_accum_steps

        if rw:
            all_router_weights.extend(rw)

        scaler.scale(loss).backward()
        loss_accum += loss.item()
    
    train_losses_history.append(loss_accum)

    # grad clip
    for opt in optimizers:
        scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # optimizer step
    for opt in optimizers:
        scaler.step(opt)
        opt.zero_grad(set_to_none=True)

    scaler.update()
    scheduler.step()

    # router bias update
    if all_router_weights and hasattr(model, "update_expert_biases"):
        model.update_expert_biases(all_router_weights, 1e-3)

    # logging
    step_time = time.time() - step_start
    tps = batch_size * block_size * grad_accum_steps / step_time

    if it % eval_interval == 0:
        val_loss = estimate_loss()
        print(f"[{it}] train_loss={loss_accum:.4f} val_loss={val_loss:.4f} step_time={step_time:.2f}, TPS={tps:.2f} tokens/sec")
        model_state_to_save = model.state_dict()
        val_losses_history.append(val_loss)
        if hasattr(model, '_orig_mod'):
          model_state_to_save = model._orig_mod.state_dict()
        optimizer_states_to_save = [opt.state_dict() for opt in optimizers]
        checkpoint = {
                'model': model_state_to_save,
                'optimizer_states':optimizer_states_to_save,
                'scheduler': scheduler.state_dict() if scheduler else None,
                'iter': it,
                'train_losses_history': train_losses_history,
                'val_losses_history': val_losses_history,
                'config': model.config
            }
        if val_loss < 5.27:
            torch.save(checkpoint,f'checkpoints/check_{it}.pt')
    
        wandb.log({"eval/loss": val_loss}, step=it)

    wandb.log({
        "train/loss": loss_accum,
        "perf/tokens_per_sec": tps,
        "perf/iter_time_ms": step_time * 1000,
        "perf/max_mem_allocated_mb":
            torch.cuda.max_memory_allocated() / 1e6
            if device == "cuda" else 0
    }, step=it)

print("✅ Training finished.")
