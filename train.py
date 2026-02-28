import os
import time
import torch
import argparse
import glob
import re
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
    # Tokenizer settings
    parser.add_argument('--tokenizer_name', type=str, default="gpt-2")
    
    # Data and model paths
    parser.add_argument('--data_dir', type=str, default=config.data_dir)
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints")
    
    # Resume training
    parser.add_argument('--resume', type=str2bool, default=False, 
                       help="Resume from latest checkpoint")
    parser.add_argument('--resume_path', type=str, default=None,
                       help="Specific checkpoint path to resume from (optional)")
    
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
    
    # WandB settings
    parser.add_argument('--wandb_project', type=str, default="Tiny-R2-openweb")
    parser.add_argument('--wandb_run_id', type=str, default=None,
                       help="WandB run ID to resume logging")
    
    # Checkpoint saving settings
    parser.add_argument('--save_best_only', type=str2bool, default=True,
                       help="Only save the best model checkpoint")
    parser.add_argument('--val_loss_threshold', type=float, default=float('inf'),
                       help="Only save models with val_loss below this threshold")
    
    return parser.parse_args()


def str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def update_config_from_args(args):
    """Update global config with parsed arguments."""
    config_attrs = [
        'batch_size', 'ctx_len', 'lr', 'max_iters', 'eval_iters',
        'warmup_iters', 'data_dir', 'n_embd', 'n_head', 'n_layer',
        'n_experts', 'hc', 'mhc', 'attention_types', 'attention_mode',
        'checkpoint_dir', 'save_best_only', 'val_loss_threshold'
    ]
    
    for attr in config_attrs:
        setattr(config, attr, getattr(args, attr))


# =============================================================================
# Checkpoint Management
# =============================================================================

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint by iteration number."""
    checkpoint_pattern = os.path.join(checkpoint_dir, "best_model_step_*.pt")
    checkpoints = glob.glob(checkpoint_pattern)
    
    if not checkpoints:
        # Fallback to old naming pattern
        checkpoint_pattern = os.path.join(checkpoint_dir, "best_model.pt")
        checkpoints = glob.glob(checkpoint_pattern)
        if checkpoints:
            return checkpoints[0], 0  # Unknown step, return 0
        return None, 0
    
    # Extract iteration numbers and find max
    latest = None
    max_step = -1
    
    for ckpt_path in checkpoints:
        # Extract number from filename (best_model_step_123.pt -> 123)
        match = re.search(r'best_model_step_(\d+)\.pt$', ckpt_path)
        if match:
            step_num = int(match.group(1))
            if step_num > max_step:
                max_step = step_num
                latest = ckpt_path
    
    return latest, max_step if latest else None, 0


def load_checkpoint(model, optimizers, scheduler, checkpoint_path, device):
    """Load checkpoint and restore model, optimizer, and scheduler states."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model'])
    
    # Load optimizer states
    if 'optimizer_states' in checkpoint and checkpoint['optimizer_states']:
        for opt, state in zip(optimizers, checkpoint['optimizer_states']):
            opt.load_state_dict(state)
    
    # Load scheduler state
    if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler']:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    # Get training history
    start_iter = checkpoint.get('iter', 0) + 1  # Resume from next iteration
    train_losses = checkpoint.get('train_losses_history', [])
    val_losses = checkpoint.get('val_losses_history', [])
    
    # Get best val loss from checkpoint
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    # Get WandB run ID if available
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    
    print(f"Resumed from iteration {start_iter - 1}")
    print(f"Best val loss so far: {best_val_loss:.4f}")
    print(f"Training history: {len(train_losses)} train steps, {len(val_losses)} eval steps")
    
    return start_iter, train_losses, val_losses, best_val_loss, wandb_run_id


def save_checkpoint(model, optimizers, scheduler, iteration, train_losses, val_losses, 
                   filepath, best_val_loss, wandb_run_id=None):
    """Save training checkpoint with all states."""
    model_state = model.state_dict()
    
    # Handle compiled model (torch.compile)
    if hasattr(model, '_orig_mod'):
        model_state = model._orig_mod.state_dict()
    
    checkpoint = {
        'model': model_state,
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scheduler': scheduler.state_dict() if scheduler else None,
        'iter': iteration,
        'train_losses_history': train_losses,
        'val_losses_history': val_losses,
        'best_val_loss': best_val_loss,
        'current_val_loss': val_losses[-1] if val_losses else None,
        'config': {
            'n_layer': config.n_layer,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_experts': config.n_experts,
            'batch_size': config.batch_size,
            'ctx_len': config.ctx_len,
            'lr': config.lr,
        },
        'wandb_run_id': wandb_run_id,
    }
    
    # Atomic save (save to temp then rename)
    temp_path = filepath + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, filepath)
    
    return checkpoint


def cleanup_old_checkpoints(checkpoint_dir, keep_only_best=True):
    """Remove old checkpoint files, keeping only the best one."""
    if not keep_only_best:
        return
    
    # Find all checkpoint files except best_model_step_*.pt
    patterns_to_remove = [
        os.path.join(checkpoint_dir, "check_*.pt"),
        os.path.join(checkpoint_dir, "final_model.pt"),
        os.path.join(checkpoint_dir, "best_model.pt"),  # Old naming
    ]
    
    removed = []
    for pattern in patterns_to_remove:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
                removed.append(os.path.basename(filepath))
            except Exception as e:
                print(f"Warning: Could not remove {filepath}: {e}")
    
    if removed:
        print(f"Cleaned up old checkpoints: {', '.join(removed)}")


# =============================================================================
# Token Buffer with State Saving
# =============================================================================

class TokenBuffer:
    """
    Streaming token buffer for causal LM training.
    Packs variable-length code samples into contiguous token stream.
    Supports saving/loading state for resume training.
    """
    
    def __init__(self, dataset, tokenizer, text_key, buffer_state=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.iterator = iter(dataset)
        self.buffer = []
        
        # Restore state if provided
        if buffer_state:
            self._restore_state(buffer_state)
    
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
    
    def get_state(self):
        """Get current buffer state for checkpointing."""
        return {
            'buffer': self.buffer.copy(),
            'dataset_index': getattr(self.iterator, '_index', 0)  # Approximate position
        }
    
    def _restore_state(self, state):
        """Restore buffer from saved state."""
        self.buffer = state.get('buffer', [])
        # Fast-forward dataset iterator (approximate)
        dataset_index = state.get('dataset_index', 0)
        for _ in range(dataset_index):
            try:
                next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataset)

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

def print_detailed_summary(model, optimizers, start_iter=0, max_iters=0, best_val_loss=float('inf')):
    """Print detailed model and optimizer summary."""
    print("\n" + "=" * 60)
    print("Model & Optimizer Summary")
    if start_iter > 0:
        print(f"Resuming from iteration {start_iter}/{max_iters}")
        print(f"Best validation loss so far: {best_val_loss:.4f}")
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
            atten_mode = getattr(block, 'atten_mode', 'MLA')
            ffn_type = getattr(block, 'ffn_type', 'unknown')
            print(f"Layer {i}: atten={atten_type}, atten_mode={atten_mode}, ffn={ffn_type}")
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


def create_scheduler(optimizer, warmup_iters, max_iters, min_lr, last_epoch=-1):
    """Create learning rate scheduler with warmup and cosine annealing."""
    return SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_iters),
            CosineAnnealingLR(optimizer, T_max=max_iters - warmup_iters, eta_min=min_lr)
        ],
        milestones=[warmup_iters],
        last_epoch=last_epoch
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


# =============================================================================
# Main Training Loop
# =============================================================================

def train(args):
    """Main training function with resume support."""
    
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
    
    checkpoint_dir = config.checkpoint_dir
    save_best_only = config.save_best_only
    val_loss_threshold = config.val_loss_threshold
    
    # Setup checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Determine checkpoint to load
    checkpoint_path = None
    start_iter = 0
    train_losses_history = []
    val_losses_history = []
    best_val_loss = float('inf')
    wandb_run_id = args.wandb_run_id
    
    if args.resume:
        if args.resume_path:
            # Use specified checkpoint
            checkpoint_path = args.resume_path
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Specified checkpoint not found: {checkpoint_path}")
                checkpoint_path = None
        else:
            # Find latest checkpoint
            result = find_latest_checkpoint(checkpoint_dir)
            if result[0]:
                checkpoint_path, start_iter = result[0], result[1]
                print(f"Found latest checkpoint: {checkpoint_path} (step {start_iter})")
            else:
                print("No checkpoint found, starting from scratch")
    
    # Initialize WandB (resume if we have a run_id)
    wandb.init(
        project=args.wandb_project,
        name="openwebtext",
        config=vars(args),
        resume="must" if wandb_run_id else False,
        id=wandb_run_id
    )
    # Store current run ID for future resumes
    current_wandb_id = wandb.run.id
    
    # Setup distributed training
    distributed_initialized = setup_distributed(device)
    
    # Setup mixed precision context
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(
        device_type="cuda", dtype=torch.float16
    )
    scaler = amp.GradScaler(enabled=('cuda' in device))
    
    # Initialize tokenizer and update vocab size
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    special_tokens = {

    "additional_special_tokens": [
        "<|assistant|>",
        "<|user|>"
    ]
    }

    num_added = tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token = tokenizer.eos_token
    config.vocab_size = tokenizer.vocab_size
    
    # Load dataset
    print(f"Loading HF dataset: {args.hf_dataset}")
    raw_dataset = load_dataset(args.hf_dataset, split=args.hf_split)
    print(raw_dataset)
    
    # Create token buffer (restore state if resuming)
    buffer_state = None  # Could be loaded from checkpoint in future
    train_buffer = TokenBuffer(raw_dataset, tokenizer, args.hf_text_key, buffer_state)
    
    # Initialize model
    model = Transformer().to(device)
    
    # Setup optimizers and scheduler
    optimizers = model.configure_optimizers(weight_decay, learning_rate, device)
    adamw_optimizer = optimizers[-1]  # AdamW is always the last optimizer
    
    # Create scheduler (will be updated if resuming)
    scheduler = create_scheduler(adamw_optimizer, warmup_iters, max_iters, min_lr)
    
    # Load checkpoint if resuming
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_iter, train_losses_history, val_losses_history, best_val_loss, loaded_wandb_id = \
            load_checkpoint(model, optimizers, scheduler, checkpoint_path, device)
        
        # Use loaded WandB ID if available and not overridden
        if loaded_wandb_id and not args.wandb_run_id:
            current_wandb_id = loaded_wandb_id
            print(f"Restored WandB run ID: {current_wandb_id}")
        
        # Clean up old checkpoints if resuming
        cleanup_old_checkpoints(checkpoint_dir, keep_only_best=save_best_only)
    
    # Compile model if on CUDA (do this after loading checkpoint)
    if 'cuda' in device:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"Model compilation failed: {e}")
    
    # Print model summary
    total_params = print_detailed_summary(model, optimizers, start_iter, max_iters, best_val_loss)
    
    # Training state
    if start_iter>0:
      start_iter=start_iter+1000
    last_saved_step = start_iter
    
    print("🚀 Starting training" if start_iter == 0 else f"🚀 Resuming training from iteration {start_iter}")
    
    # Main training loop
    for iteration in range(start_iter, max_iters):
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
        val_loss = None
        if iteration % eval_interval == 0 or iteration == max_iters - 1:
            val_loss = estimate_loss(
                model, train_buffer, batch_size, block_size, 
                device, ctx, config.eval_iters
            )
            
            print(f"[{iteration}] "
                  f"train_loss={loss_accum:.4f} "
                  f"val_loss={val_loss:.4f} "
                  f"best_val_loss={best_val_loss:.4f} "
                  f"step_time={step_time:.2f}s "
                  f"TPS={tokens_per_sec:.2f} "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")
            
            val_losses_history.append(val_loss)
            
            # Check if this is the best model
            is_best = val_loss < best_val_loss
            below_threshold = val_loss < val_loss_threshold
            
            if is_best:
                best_val_loss = val_loss
                print(f"  -> New best model! (val_loss={val_loss:.4f})")
                
                # Save best model with step number in filename
                best_path = os.path.join(checkpoint_dir, f"best_model_step_{iteration}.pt")
                save_checkpoint(
                    model, optimizers, scheduler, iteration,
                    train_losses_history, val_losses_history, best_path, 
                    best_val_loss, current_wandb_id
                )
                print(f"  -> Saved to: {best_path}")
                
                # Clean up old checkpoints (keep only best)
                if save_best_only:
                    cleanup_old_checkpoints(checkpoint_dir, keep_only_best=True)
                    # Rename current to ensure it's the only one
                    # (cleanup removes old ones, but we might have multiple best_model_step_*.pt)
                    for old_file in glob.glob(os.path.join(checkpoint_dir, "best_model_step_*.pt")):
                        if old_file != best_path:
                            try:
                                os.remove(old_file)
                                print(f"  -> Removed old: {os.path.basename(old_file)}")
                            except Exception as e:
                                print(f"  -> Warning: Could not remove {old_file}: {e}")
            
            # Also save if below threshold but not best (optional behavior)
            elif below_threshold and not save_best_only:
                threshold_path = os.path.join(checkpoint_dir, f"model_step_{iteration}_loss_{val_loss:.4f}.pt")
                save_checkpoint(
                    model, optimizers, scheduler, iteration,
                    train_losses_history, val_losses_history, threshold_path,
                    best_val_loss, current_wandb_id
                )
                print(f"  -> Saved threshold model: {threshold_path}")
            
            # Log to WandB
            wandb.log({
                "eval/loss": val_loss,
                "eval/best_loss": best_val_loss,
                "eval/is_new_best": is_best,
            }, step=iteration)
        
        # Log training metrics (every step)
        log_dict = {
            "train/loss": loss_accum,
            "perf/tokens_per_sec": tokens_per_sec,
            "perf/iter_time_ms": step_time * 1000,
            "train/learning_rate": scheduler.get_last_lr()[0],
            "train/iteration": iteration,
        }
        
        # Add memory stats if CUDA
        if device == "cuda":
            log_dict["perf/max_mem_allocated_mb"] = torch.cuda.max_memory_allocated() / 1e6
            log_dict["perf/max_mem_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
        
        wandb.log(log_dict, step=iteration)
    
    # Final checkpoint (only if it's the best)
    final_val_loss = estimate_loss(model, train_buffer, batch_size, block_size, device, ctx, config.eval_iters)
    if final_val_loss <= best_val_loss:
        final_path = os.path.join(checkpoint_dir, f"best_model_step_{max_iters-1}_final.pt")
        save_checkpoint(
            model, optimizers, scheduler, max_iters - 1,
            train_losses_history, val_losses_history, final_path,
            best_val_loss, current_wandb_id
        )
        print(f"Final model saved (best): {final_path}")
    
    print("✅ Training finished")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {os.path.join(checkpoint_dir, 'best_model_step_*.pt')}")
    wandb.finish()
    
    return model, train_losses_history, val_losses_history


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    args = parse_arguments()
    train(args)
