#!/usr/bin/env python3
"""
Tiny-R2 OPD 训练 - 多数据集支持版
✅ 新增：支持 AIME25、AMC23、DAPO、MATH-500、MMLU-Pro、MedQA、MedXpertQA-Text、Minerva
✅ 新增：智能词汇表维度对齐（支持检查点词汇表 < 当前模型）
✅ 新增：优雅关闭分布式环境
✅ 修复：优化器状态维度不匹配问题（重新初始化优化器）
✅ 修复：教师/学生词汇表不匹配 RuntimeError
"""

import os
import sys
import random
import argparse
import glob
import re
from typing import Optional, List, Dict, Any
from contextlib import nullcontext

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

# ====================== 导入Tiny-R2核心模块 ======================
try:
    import model
    from model import Transformer
    import config
    print("✅ 成功导入 Tiny-R2 核心模块 (model, config)")
except ImportError as e:
    print(f"❌ 无法导入 Tiny-R2 模块: {e}")
    sys.exit(1)

# ====================== 工具函数 ======================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# ====================== 数据集配置注册表 ======================
DATASET_CONFIGS = {
    "gpqa_diamond": {
        "hf_path": "Idavidrein/gpqa",
        "hf_subset": "gpqa_diamond",
        "split": "train",
        "instruction_key": "Question",
        "response_key": "Correct Answer",
        "system_prompt": None
    },
    "aime25": {
        "hf_path": "math-ai/aime25",
        "hf_subset": None,
        "split": "train",
        "instruction_key": "problem",
        "response_key": "solution",
        "system_prompt": "请解决以下数学竞赛题，并给出详细解答过程。"
    },
    "amc23": {
        "hf_path": "math-ai/amc23",
        "hf_subset": None,
        "split": "train",
        "instruction_key": "problem",
        "response_key": "solution",
        "system_prompt": "请解决以下数学竞赛题，并给出详细解答过程。"
    },
    "dapo": {
        "hf_path": "BytedTsinghua-SIA/DAPO-Math-17k",
        "hf_subset": None,
        "split": "train",
        "instruction_key": "instruction",
        "response_key": "response",
        "system_prompt": None
    },
    "math500": {
        "hf_path": "math-ai/math500",
        "hf_subset": None,
        "split": "test",
        "instruction_key": "problem",
        "response_key": "solution",
        "system_prompt": "请解决以下数学题，并给出详细解答过程。"
    },
    "mmlu_pro": {
        "hf_path": "TIGER-Lab/MMLU-Pro",
        "hf_subset": None,
        "split": "test",
        "instruction_key": "question",
        "response_key": "answer",
        "system_prompt": "请回答以下多项选择题，只需给出正确选项。",
        "preprocess": lambda x: f"{x['question']}\n选项：\n" + "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(x['options'])])
    },
    "medqa": {
        "hf_path": "bigbio/med_qa",
        "hf_subset": "medqa_usmle_hf",
        "split": "train",
        "instruction_key": "question",
        "response_key": "answer",
        "system_prompt": "请回答以下医学问题。"
    },
    "medxpertqa_text": {
        "hf_path": "OctoMed/MedXpertQA-Text",
        "hf_subset": None,
        "split": "train",
        "instruction_key": "question",
        "response_key": "answer",
        "system_prompt": "请回答以下医学专业问题。"
    },
    "minerva": {
        "hf_path": "math-ai/minervamath",
        "hf_subset": None,
        "split": "train",
        "instruction_key": "question",
        "response_key": "answer",
        "system_prompt": "请解决以下问题并给出详细步骤。"
    }
}

# ====================== 检查点管理函数 ======================
def find_latest_checkpoint(checkpoint_dir):
    """
    查找最新的检查点
    优先级：best_model_step_*.pt (按step排序) > best_model.pt
    """
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
    # 1. 查找 best_model_step_*.pt
    step_pattern = os.path.join(checkpoint_dir, "best_model_step_*.pt")
    step_checkpoints = glob.glob(step_pattern)
    
    if step_checkpoints:
        latest_ckpt = None
        max_step = -1
        for ckpt_path in step_checkpoints:
            match = re.search(r'best_model_step_(\d+)\.pt$', ckpt_path)
            if match:
                step = int(match.group(1))
                if step > max_step:
                    max_step = step
                    latest_ckpt = ckpt_path
        if latest_ckpt:
            return latest_ckpt, max_step
    
    # 2. 查找 best_model.pt
    best_pattern = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_pattern):
        return best_pattern, 0
    
    return None, 0

def load_checkpoint(model, optimizers, scheduler, checkpoint_path, device):
    """
    🔥 智能加载检查点（终极版）
    - 自动处理词汇表大小不匹配（支持检查点 < 当前模型）
    - 对于不匹配的层，智能复制前 N 个 token
    - 优化器状态重新初始化（避免维度不匹配）
    返回：start_iter, best_val_loss
    """
    print(f"📂 加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 加载模型权重
    model_state = checkpoint.get('model', checkpoint)
    # 处理 torch.compile 模型
    is_compiled_model = hasattr(model, '_orig_mod')
    is_compiled_ckpt = any(k.startswith("_orig_mod.") for k in model_state.keys())
    
    if is_compiled_model and not is_compiled_ckpt:
        model_state = {f"_orig_mod.{k}": v for k, v in model_state.items()}
    elif not is_compiled_model and is_compiled_ckpt:
        model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    
    # 🔥 核心修复：智能词汇表对齐
    current_sd = model.state_dict()
    filtered_state = {}
    
    for key, ckpt_tensor in model_state.items():
        if key not in current_sd:
            continue
            
        current_tensor = current_sd[key]
        
        # 维度完全匹配 -> 直接加载
        if ckpt_tensor.shape == current_tensor.shape:
            filtered_state[key] = ckpt_tensor
        
        # 🔥 词汇表维度不匹配（检查点 < 当前模型）-> 智能复制
        elif key in ["token_embedding_table.weight", "lm_head.weight"]:
            ckpt_vocab, hidden_dim = ckpt_tensor.shape
            curr_vocab, _ = current_tensor.shape
            
            if ckpt_vocab < curr_vocab:
                print(f"⚠️  词汇表对齐: {key} | 检查点 {ckpt_vocab} -> 当前 {curr_vocab}")
                # 创建新张量，保持当前初始化
                new_tensor = current_tensor.clone()
                # 复制前 ckpt_vocab 个 token
                new_tensor[:ckpt_vocab, :] = ckpt_tensor
                filtered_state[key] = new_tensor
            else:
                print(f"⚠️  跳过 {key}: 检查点词汇表更大 ({ckpt_vocab} > {curr_vocab})")
        
        # 其他层维度不匹配 -> 跳过
        else:
            print(f"⚠️  跳过 {key}: 维度不匹配 {ckpt_tensor.shape} vs {current_tensor.shape}")
    
    # 加载
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    
    # 打印信息
    if missing_keys:
        print(f"ℹ️  缺失键（可能是新初始化的层）: {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"ℹ️  多余键（来自检查点）: {unexpected_keys[:5]}...")
    print(f"✅ 模型权重加载成功！")
    
    # 🔥 关键修复：不加载优化器状态，让它重新初始化
    # 这样可以避免维度不匹配的问题
    print("ℹ️  优化器状态将重新初始化（避免维度不匹配）")
    
    # 加载调度器状态
    if 'scheduler' in checkpoint and scheduler:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("✅ 调度器状态加载成功")
        except Exception as e:
            print(f"⚠️  调度器状态加载失败: {e}")
    
    # 加载训练状态
    start_iter = checkpoint.get('iter', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"✅ 检查点加载完成 | 恢复迭代: {start_iter} | 最优val_loss: {best_val_loss:.4f}")
    return start_iter, best_val_loss

# ====================== 参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Tiny-R2 OPD Train - Multi-Dataset Support")

    # 训练超参
    parser.add_argument('--batch_size', type=int, default=getattr(config, 'batch_size', 2))
    parser.add_argument('--ctx_len', type=int, default=getattr(config, 'ctx_len', 2048))
    parser.add_argument('--lr', type=float, default=getattr(config, 'lr', 2e-5))
    parser.add_argument('--max_iters', type=int, default=getattr(config, 'max_iters', 10000))
    parser.add_argument('--warmup_iters', type=int, default=getattr(config, 'warmup_iters', 100))
    parser.add_argument('--grad_accum_steps', type=int, default=getattr(config, 'grad_accum', 1))
    parser.add_argument('--weight_decay', type=float, default=getattr(config, 'weight_decay', 0.01))
    parser.add_argument('--max_grad_norm', type=float, default=getattr(config, 'max_grad_norm', 1.0))
    parser.add_argument('--min_lr', type=float, default=getattr(config, 'min_lr', 1e-6))

    # 模型结构
    parser.add_argument('--n_embd', type=int, default=getattr(config, 'n_embd', 768))
    parser.add_argument('--n_head', type=int, default=getattr(config, 'n_head', 16))
    parser.add_argument('--n_layer', type=int, default=getattr(config, 'n_layer', 6))
    parser.add_argument('--n_experts', type=int, default=getattr(config, 'n_experts', 32))
    
    # 数据集配置（核心新增）
    parser.add_argument("--dataset", type=str, default="gpqa_diamond", 
                        choices=list(DATASET_CONFIGS.keys()),
                        help=f"选择训练数据集: {', '.join(DATASET_CONFIGS.keys())}")
    parser.add_argument("--val_size", type=float, default=0.2,
                        help="验证集比例 (0-1)")
    parser.add_argument("--hf_cache_dir", type=str, default=None,
                        help="HuggingFace 数据集缓存目录")

    # 模型路径
    parser.add_argument("--student_ckpt", type=str, default=None, help="指定检查点路径 (可选，不指定则自动查找)")
    parser.add_argument("--hf_teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3.5-9B")

    # OPD损失
    parser.add_argument("--opd_loss_type", type=str, default="reverse_kl")
    parser.add_argument("--opd_chunk_size", type=int, default=512)
    parser.add_argument("--opd_beta", type=float, default=0.5)

    # 保存
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--opd_checkpoint_dir", type=str, default="opd_checkpoints")
    parser.add_argument("--val_freq", type=int, default=50)

    # 硬件
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def update_config_from_args(args):
    config_attrs = ['batch_size', 'ctx_len', 'lr', 'max_iters', 'device']
    for attr in config_attrs:
        if hasattr(args, attr):
            setattr(config, attr, getattr(args, attr))

# ====================== 通用数据集类 ======================
class UnifiedDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, ctx_len: int, dataset_config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.dataset = hf_dataset
        self.config = dataset_config
        print(f"📊 数据集加载完成 | 样本数: {len(self.dataset)}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        
        # 获取指令和回复
        instruction_key = self.config["instruction_key"]
        response_key = self.config["response_key"]
        
        # 应用自定义预处理（如果有）
        if "preprocess" in self.config:
            instruction = self.config["preprocess"](item)
        else:
            instruction = item[instruction_key]
        
        response = item[response_key]
        
        # 构建消息列表
        messages = []
        
        # 添加系统提示（如果有）
        if self.config.get("system_prompt"):
            messages.append({"role": "system", "content": self.config["system_prompt"]})
        
        # 添加用户和助手消息
        messages.extend([
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ])
        
        # 应用聊天模板
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        tokenized = self.tokenizer(
            full_text, truncation=True, max_length=self.ctx_len,
            padding=False, return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0)

        # 构建 mask（仅对回复部分计算损失）
        prompt_messages = messages[:-1]  # 去掉最后的 assistant 回复
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_len = len(self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0])

        labels = input_ids.clone()
        labels[:prompt_len] = -100
        response_mask = torch.zeros_like(input_ids)
        response_mask[prompt_len:] = 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask
        }

# ====================== 批次填充 ======================
def collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    response_mask = [item["response_mask"] for item in batch]

    max_len = max(len(x) for x in input_ids)
    padded_input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    padded_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for i in range(len(batch)):
        padded_input_ids[i, :len(input_ids[i])] = input_ids[i]
        padded_labels[i, :len(labels[i])] = labels[i]
        padded_mask[i, :len(response_mask[i])] = response_mask[i]

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "response_mask": padded_mask
    }

# ====================== 修复版 OPD 损失函数 ======================
def chunked_divergence_loss(
    student_logits, teacher_logits, loss_type="reverse_kl", chunk_size=512, response_mask=None, beta=0.5
):
    B, L, V_s = student_logits.shape
    _, _, V_t = teacher_logits.shape

    # 词汇表维度对齐
    min_vocab = min(V_s, V_t)
    if V_s != V_t:
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]

    # dtype 统一
    teacher_logits = teacher_logits.to(student_logits.dtype)

    # 展平
    student_logits = student_logits.reshape(-1, min_vocab)
    teacher_logits = teacher_logits.reshape(-1, min_vocab)

    if response_mask is not None:
        mask = response_mask.reshape(-1).bool()
        student_logits = student_logits[mask]
        teacher_logits = teacher_logits[mask]

    total_loss = 0.0
    num_chunks = (min_vocab + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        sl = student_logits[:, i*chunk_size:(i+1)*chunk_size]
        tl = teacher_logits[:, i*chunk_size:(i+1)*chunk_size]

        s_logprob = F.log_softmax(sl, -1)
        t_prob = F.softmax(tl, -1)
        t_logprob = F.log_softmax(tl, -1)

        if loss_type == "reverse_kl":
            s_prob = F.softmax(sl, -1)
            loss = (s_prob * (s_logprob - t_logprob)).sum(-1).mean()
        else:
            loss = (t_prob * (t_logprob - s_logprob)).sum(-1).mean()
        
        total_loss += loss

    return total_loss * beta

# ====================== 验证函数 ======================
@torch.no_grad()
def validate(model, teacher_model, dataloader, args, ctx):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="验证中", leave=False):
        input_ids = batch["input_ids"].to(args.device)
        mask = batch["response_mask"].to(args.device)

        with torch.no_grad():
            t_logits = teacher_model(input_ids).logits
        with ctx:
            s_logits = model(input_ids)[0]

        loss = chunked_divergence_loss(s_logits, t_logits, args.opd_loss_type, args.opd_chunk_size, mask, args.opd_beta)
        total_loss += loss.item()
    
    model.train()
    return total_loss / len(dataloader)


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




# ====================== 主函数 ======================
def main():
    args = parse_args()
    update_config_from_args(args)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.opd_checkpoint_dir, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    # 获取数据集配置
    dataset_config = DATASET_CONFIGS[args.dataset]
    print(f"📚 选中数据集: {args.dataset}")

    # 初始化分布式训练环境
    if not dist.is_initialized():
        try:
            dist.init_process_group(
                backend='nccl' if 'cuda' in args.device else 'gloo',
                init_method='tcp://localhost:12355',
                world_size=1,
                rank=0
            )
            print("✅ 分布式环境初始化成功（单卡模式）")
        except Exception as e:
            print(f"⚠️  分布式初始化跳过: {e}")

    try:
        # 1. 加载 Tokenizer
        print("="*60)
        print("📝 加载 Tokenizer")
        print("="*60)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        config.vocab_size = len(tokenizer)
        print(f"✅ 词汇量: {len(tokenizer)}")

        # 2. 加载数据集（通用版）
        print("\n" + "="*60)
        print(f"📊 加载数据集: {args.dataset}")
        print("="*60)
        
        # 加载 HuggingFace 数据集
        load_kwargs = {
            "path": dataset_config["hf_path"],
            "split": dataset_config["split"]
        }
        if dataset_config["hf_subset"]:
            load_kwargs["name"] = dataset_config["hf_subset"]
        if args.hf_cache_dir:
            load_kwargs["cache_dir"] = args.hf_cache_dir
            
        ds = load_dataset(**load_kwargs)
        ds = ds.shuffle(seed=42)
        
        # 自动切分训练/验证集
        val_size = int(len(ds) * args.val_size)
        val_ds_raw = ds.select(range(val_size))
        train_ds_raw = ds.select(range(val_size, len(ds)))

        # 使用统一数据集类
        train_dataset = UnifiedDataset(train_ds_raw, tokenizer, args.ctx_len, dataset_config)
        val_dataset = UnifiedDataset(val_ds_raw, tokenizer, args.ctx_len, dataset_config)

        from functools import partial
        collate = partial(collate_fn, tokenizer=tokenizer)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False, collate_fn=collate)

        # 3. 初始化模型
        print("\n" + "="*60)
        print("🤖 初始化学生模型")
        print("="*60)
        device = args.device
        model = Transformer().to(device)

        # 优化器
        optimizers = model.configure_optimizers(args.weight_decay, args.lr, device)
        scheduler = SequentialLR(optimizers[-1], [
            LinearLR(optimizers[-1], 0.1, total_iters=args.warmup_iters),
            CosineAnnealingLR(optimizers[-1], args.max_iters-args.warmup_iters, args.min_lr)
        ], milestones=[args.warmup_iters])

        # 4. 自动加载检查点
        start_iter = 0
        best_val_loss = float("inf")
        #checkpoint_path = None
        
        # 优先使用用户指定的检查点
        if args.student_ckpt and os.path.exists(args.student_ckpt):
            checkpoint_path = args.student_ckpt
            start_iter, best_val_loss = load_checkpoint(
                model, optimizers, scheduler, checkpoint_path, device
            )
        else:
            # 自动查找最新检查点
            checkpoint_path, _ = find_latest_checkpoint(args.checkpoint_dir)
        
        #if not os.path.exists(checkpoint_path):
           # print("🚀 未找到检查点，从头开始训练")
        total_params = print_detailed_summary(model, optimizers, start_iter, args.max_iters, best_val_loss)

        # 5. 加载教师模型
        print("\n" + "="*60)
        print("\n👨‍🏫 加载教师模型 Qwen3.5-9B")
        print("="*60)
        teacher_dtype = torch.float16
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.hf_teacher_model, 
            trust_remote_code=True, 
            torch_dtype=teacher_dtype,
            device_map=device
        ).eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

        # 6. 训练配置
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if "cuda" in device else nullcontext()
        scaler = amp.GradScaler(enabled="cuda" in device)
        global_step = start_iter

        # 7. 开始训练
        print("\n" + "="*60)
        print(f"🚀 开始 OPD 蒸馏训练 | 数据集: {args.dataset} | 起始迭代: {global_step}")
        print("="*60)
        
        while global_step < args.max_iters:
            for batch in train_loader:
                if global_step >= args.max_iters:
                    break
                global_step += 1

                input_ids = batch["input_ids"].to(device)
                mask = batch["response_mask"].to(device)

                # 梯度累积
                loss_accum = 0.0
                for _ in range(args.grad_accum_steps):
                    with torch.no_grad():
                        t_logits = teacher_model(input_ids).logits
                    with ctx:
                        s_logits = model(input_ids)[0]
                    
                    loss = chunked_divergence_loss(s_logits, t_logits, args.opd_loss_type, args.opd_chunk_size, mask, args.opd_beta)
                    loss = loss / args.grad_accum_steps
                    scaler.scale(loss).backward()
                    loss_accum += loss.item()

                # 参数更新
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                for opt in optimizers:
                    scaler.step(opt)
                    opt.zero_grad(set_to_none=True)
                scaler.update()
                scheduler.step()

                # 验证 & 保存
                if global_step % args.val_freq == 0:
                    val_loss = validate(model, teacher_model, val_loader, args, ctx)
                    print(f"Step {global_step:04d} | 训练损失: {loss_accum:.4f} | 验证损失: {val_loss:.4f} | 最优: {best_val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        # 保存检查点（包含完整状态）
                        save_path = os.path.join(args.opd_checkpoint_dir, f"best_model_step_{global_step}.pt")
                        model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
                        
                        torch.save({
                            "model": model_state,
                            "optimizer_states": [opt.state_dict() for opt in optimizers],
                            "scheduler": scheduler.state_dict(),
                            "iter": global_step,
                            "best_val_loss": best_val_loss,
                            "dataset": args.dataset  # 记录训练数据集
                        }, save_path)
                        print(f"✅ 保存最优模型: {save_path}")
                        
                        # 清理旧检查点
                        for old_file in glob.glob(os.path.join(args.opd_checkpoint_dir, "best_model_step_*.pt")):
                            if old_file != save_path:
                                try:
                                    os.remove(old_file)
                                except:
                                    pass

        print("\n🎉 训练完成！")

    finally:
        # 🔥 核心修复：优雅关闭分布式环境
        if dist.is_initialized():
            print("\n🔌 关闭分布式环境...")
            dist.destroy_process_group()
            print("✅ 分布式环境已关闭")

if __name__ == "__main__":
    main()
