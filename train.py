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


# =============================================================================
# Argument Parsing
# =============================================================================

def parse_arguments():
    """Parse and setup command line arguments."""
    parser = argparse.ArgumentParser(description="Train Transformer with MoE on HuggingFace datasets")
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=config.batch_size)
    parser.add_argument('--ctx_len', type=int, default=config.ctx_len)
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument('--max_iters', type=int, default=config.max_iters)
    parser.add_argument('--eval_iters', type=int, default=config.eval_interval)
    parser.add_argument('--warmup_iters', type=int, default=config.warmup_iters)
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default=config.data_dir)
    
    # Model architecture
    parser.add_argument('--n_embd', type=int, default=config.n_embd)
    parser.add_argument('--n_head', type=int, default=config.n_head)
    parser.add_argument('--n_layer', type=int, default=config.n_layer)
    parser.add_argument('--n_experts', type=int, default=config.n_experts)
    
    # Hyper-connection settings
    parser.add_argument('--hc', type=str, default=config.hc)
    parser.add_argument('--mhc', type=str, default=config.mhc)
    
    # Attention settings
    parser.add_argument('--attention_types', nargs="+", type=str, default=config.attention_types)
    parser.add_argument('--attention_mode', nargs="+", type=str, default=config.attention_mode)
    
    # HuggingFace dataset settings
    parser.add_argument('--hf_dataset', type=str, default="flytech/python-codes-25k")
    parser.add_argument('--hf_split', type=str, default="train")
    parser.add_argument('--hf_text_key', type=str, default="text")
    
    return parser.parse_args()


def update_config_from_args(args):
    """Update global config with parsed arguments."""
    config_attrs = [
        'batch_size', 'ctx_len', 'lr', 'max_iters', 'eval_iters',
        'warmup_iters', 'data_dir', 'n_embd', 'n_head', 'n_layer',
        'n_experts', 'hc', 'mhc', 'attention_types','attention_mode'
    ]
    
    for attr in config_attrs:
        setattr(config, attr, getattr(args, attr))
    
    # Special handling for attention_mode mapping
    config.attention_mode = args.attention_mode


# =============================================================================
# Token Buffer for Streaming Data
# =============================================================================

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

    def _refill(self, min_tokens):
        """Refill buffer with tokens until min_tokens is reached."""
        while len(self.buffer) < min_tokens:
            try:
                example = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)
                example = next(self.iterator)

            text = example.get(self.text_key, "")
            if not text:
                continue

            token_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            if token_ids:
                self.buffer.extend(token_ids)

    def get_batch(self, batch_size, block_size, device):
        """Get a batch of token sequences."""
        needed = batch_size * (block_size + 1)
        self._refill(needed)

        xs, ys = [], []
        for _ in range(batch_size):
            x = self.buffer[:block_size]
            y = self.buffer[1:block_size + 1]
            self.buffer = self.buffer[block_size:]
            xs.append(torch.tensor(x, dtype=torch.long))
            ys.append(torch.tensor(y, dtype=torch.long))

        return (
            torch.stack(xs).to(device, non_blocking=True),
            torch.stack(ys).to(device, non_blocking=True)
        )


# =============================================================================
# Model Summary
# =============================================================================

def print_detailed_summary(model, optimizers):
    """Print detailed model and optimizer summary."""
    print("\n" + "=" * 60)
    print("Model & Optimizer Summary")
    print("=" * 60 + "\n")
    
    # Total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.3f} M\n")
    
    # Layer-wise parameters (if detailed info enabled)
    if getattr(config, 'info_level', 0) == 2:
        print("--- Layer-wise Parameters ---")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name:50} | shape={tuple(param.shape)} | params={param.numel()}")
        print()
    
    # Transformer architecture info
    print("--- Transformer Architecture ---")
    print(f"Number of layers: {config.n_layer}")
    print(f"Attention heads: {config.n_head}")
    print(f"Embedding size: {config.n_embd}")
    
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            atten_type = getattr(block, 'atten_types', 'unknown')
            atten_mode=getattr(block, 'atten_mode', 'MLA')
            ffn_type = getattr(block, 'ffn_type', 'unknown')
            print(f"Layer {i}: atten={atten_type}, atten_mode={atten_mode},ffn={ffn_type}")
    print()
    
    # MoE configuration
    print("--- MoE Configuration ---")
    print(f"Number of experts: {config.n_experts}")
    print(f"Active experts per token: {getattr(config, 'num_exp', 'N/A')}")
    print(f"Shared experts: {getattr(config, 'shared_experts', 'N/A')}")
    print(f"Expert bias enabled: {getattr(config, 'use_expert_bias', False)}")
    print()
    
    # Hyper-connections info
    print("--- Hyper-connections ---")
    if hasattr(model, 'expand_stream'):
        print(f"Expand stream: {type(model.expand_stream).__name__}")
    if hasattr(model, 'reduce_stream'):
        print(f"Reduce stream: {type(model.reduce_stream).__name__}")
    print()
    
    # Optimizer info
    print("--- Optimizers ---")
    for i, opt in enumerate(optimizers):
        opt_type = type(opt).__name__
        for j, pg in enumerate(opt.param_groups):
            lr = pg.get('lr', 'N/A')
            wd = pg.get('weight_decay', 'N/A')
            print(f"Optimizer {i} ({opt_type}) group {j}: lr={lr}, wd={wd}")
        
        if opt_type.lower() == 'muon':
            num_experts = getattr(opt, 'num_experts', 'N/A')
            print(f"  -> Muon handles {num_experts} experts")
    
    print("\n" + "=" * 60 + "\n")
    return total_params


# =============================================================================
# Training Utilities
# =============================================================================

def setup_distributed(device):
    """Initialize distributed training if available."""
    if 'cuda' not in device:
        return False
    
    try:
        dist.init_process_group(
            backend='nccl',
            init_method="tcp://localhost:12355",
            world_size=1,
            rank=0
        )
        return True
    except Exception as e:
        print(f"Distributed init failed: {e}")
        return False


def create_scheduler(optimizer, warmup_iters, max_iters, min_lr):
    """Create learning rate scheduler with warmup and cosine annealing."""
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters),
            CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters, eta_min=min_lr)
        ],
        milestones=[warmup_iters]
    )


@torch.no_grad()
def estimate_loss(model, train_buffer, batch_size, block_size, device, ctx, eval_iters):
    """Estimate validation loss."""
    model.eval()
    total_loss = 0.0
    
    for _ in range(eval_iters):
        xb, yb = train_buffer.get_batch(batch_size, block_size, device)
        with ctx:
            _, loss, _ = model(xb, yb)
        total_loss += loss.item()
    
    model.train()
    return total_loss / eval_iters


def save_checkpoint(model, optimizers, scheduler, iteration, train_losses, val_losses, filepath):
    """Save training checkpoint."""
    model_state = model.state_dict()
    
    # Handle compiled model
    if hasattr(model, '_orig_mod'):
        model_state = model._orig_mod.state_dict()
    
    checkpoint = {
        'model': model_state,
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scheduler': scheduler.state_dict() if scheduler else None,
        'iter': iteration,
        'train_losses_history': train_losses,
        'val_losses_history': val_losses,
    }
    
    torch.save(checkpoint, filepath)
    return checkpoint


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    """Main training function."""
    
    # Update config with CLI arguments
    update_config_from_args(args)
    
    # Training configuration
    batch_size = config.batch_size
    block_size = config.ctx_len
    grad_accum_steps = config.grad_accum
    device = config.device
    
    max_iters = args.max_iters
    eval_interval = args.eval_iters
    warmup_iters = args.warmup_iters
    
    learning_rate = args.lr
    min_lr = config.min_lr
    weight_decay = config.weight_decay
    max_grad_norm = config.max_grad_norm
    
    # Initialize WandB
    wandb.init(
        project="Tiny-R2-code",
        name="flytech-python-codes-moe",
        config=vars(args)
    )
    
    # Setup distributed training
    distributed_initialized = setup_distributed(device)
    
    # Setup mixed precision context
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(
        device_type="cuda", dtype=torch.float16
    )
    scaler = amp.GradScaler(enabled=('cuda' in device))
    
    # Initialize tokenizer and update vocab size
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    
    # Load dataset
    print(f"Loading HF dataset: {args.hf_dataset}")
    raw_dataset = load_dataset(args.hf_dataset, split=args.hf_split)
    print(raw_dataset)
    
    # Create token buffer
    train_buffer = TokenBuffer(raw_dataset, tokenizer, args.hf_text_key)
    
    # Initialize model
    model = Transformer().to(device)
    
    # Setup optimizers and scheduler
    optimizers = model.configure_optimizers(weight_decay, learning_rate, device)
    adamw_optimizer = optimizers[-1]  # AdamW is always the last optimizer
    scheduler = create_scheduler(adamw_optimizer, warmup_iters, max_iters, min_lr)
    
    # Compile model if on CUDA
    if 'cuda' in device:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    
    # Print model summary
    total_params = print_detailed_summary(model, optimizers)
    
    # Setup checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training state
    train_losses_history = []
    val_losses_history = []
    
    print("🚀 Starting training")
    
    # Main training loop
    for iteration in range(max_iters):
        step_start = time.time()
        loss_accum = 0.0
        all_router_weights = []
        
        # Gradient accumulation steps
        for _ in range(grad_accum_steps):
            xb, yb = train_buffer.get_batch(batch_size, block_size, device)
            
            with ctx:
                _, loss, router_weights = model(xb, yb)
                loss = loss / grad_accum_steps
            
            if router_weights:
                all_router_weights.extend(router_weights)
            
            scaler.scale(loss).backward()
            loss_accum += loss.item()
        
        train_losses_history.append(loss_accum)
        
        # Gradient clipping
        for optimizer in optimizers:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        # Optimizer steps
        for optimizer in optimizers:
            scaler.step(optimizer)
            optimizer.zero_grad(set_to_none=True)
        
        scaler.update()
        scheduler.step()
        
        # Update expert biases for load balancing
        if all_router_weights and hasattr(model, "update_expert_biases"):
            model.update_expert_biases(all_router_weights, update_rate=1e-3)
        
        # Logging metrics
        step_time = time.time() - step_start
        tokens_per_sec = batch_size * block_size * grad_accum_steps / step_time
        
        # Evaluation and checkpointing
        if iteration % eval_interval == 0:
            val_loss = estimate_loss(
                model, train_buffer, batch_size, block_size, 
                device, ctx, config.eval_iters
            )
            
            print(f"[{iteration}] "
                  f"train_loss={loss_accum:.4f} "
                  f"val_loss={val_loss:.4f} "
                  f"step_time={step_time:.2f}s "
                  f"TPS={tokens_per_sec:.2f}")
            
            val_losses_history.append(val_loss)
            
            # Save checkpoint if validation loss is good
            if val_loss < 5.27:
                checkpoint_path = f"checkpoints/check_{iteration}.pt"
                save_checkpoint(
                    model, optimizers, scheduler, iteration,
                    train_losses_history, val_losses_history, checkpoint_path
                )
            
            wandb.log({"eval/loss": val_loss}, step=iteration)
        
        # Log training metrics
        wandb.log({
            "train/loss": loss_accum,
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/iter_time_ms": step_time * 1000,
            "perf/max_mem_allocated_mb": (
                torch.cuda.max_memory_allocated() / 1e6 
                if device == "cuda" else 0
            ),
            "train/learning_rate": scheduler.get_last_lr()[0] if scheduler else learning_rate,
        }, step=iteration)
    
    print("✅ Training finished")
    wandb.finish()
    
    return model, train_losses_history, val_losses_history


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    args = parse_arguments()
    train(args)
