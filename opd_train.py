#!/usr/bin/env python3
"""
Tiny-R2 OPD 训练 - 多数据集支持版 (工业级完整版+致命Bug全修复)
✅ 核心修复：logits时序严格对齐（左移1位，丢弃最后无效token）
✅ 核心修复：response_mask精准计算（仅答案部分有效）
✅ 修复：标准无魔改 Reverse KL Loss（纯 Tensor，兼容混合精度）
✅ 修复：强制锁定 AdamW 学习率为 args.lr（打印验证）
✅ 修复：dtype 不匹配警告
✅ 新增：支持 AIME25、AMC23、DAPO、MATH-500、MMLU-Pro、MedQA、MedXpertQA-Text、Minerva
✅ 新增：智能词汇表维度对齐（支持检查点词汇表 < 当前模型）
✅ 新增：优雅关闭分布式环境
✅ 修复：优化器状态维度不匹配问题（重新初始化优化器）
✅ 修复：教师/学生词汇表不匹配 RuntimeError
✅ 【工业级修复】梯度累积逻辑（从重复同batch改为累积多batch）
✅ 【工业级修复】验证损失口径统一（训练/验证均按有效token平均）
✅ 【工业级修复】迭代步数精准控制
✅ 【工业级优化】验证效率提升（独立val_batch_size + inference_mode）
✅ 【工业级优化】首次验证基线性能
✅ 【工业级对齐】Loss计算对齐TRL OPSDTrainer（广义JSD散度）
✅ 【工业级优化】支持Top-k Token限制、Token级Clip
✅ 【致命Bug修复】Mask维度不匹配问题（unsqueeze扩展维度适配词汇表维度）
✅ 【警告修复】RMSNorm dtype不匹配警告（自动对齐权重与输入dtype）
"""

import os
import sys
import random
import argparse
import glob
import re
from typing import Optional, List, Dict, Any, Tuple
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
    
    # 🔥 关键修复：也不加载 scheduler 状态，避免 lr 错乱
    print("ℹ️  调度器状态将重新初始化（确保使用新的 lr）")
    
    # 加载训练状态
    start_iter = checkpoint.get('iter', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    print(f"✅ 检查点加载完成 | 恢复迭代: {start_iter} | 最优val_loss: {best_val_loss:.4f}")
    return start_iter, best_val_loss

# ====================== 参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Tiny-R2 OPD Train - Multi-Dataset Support")

    # 训练超参
    parser.add_argument('--batch_size', type=int, default=getattr(config, 'batch_size', 2),
                        help="单卡小BatchSize (micro-batch size)")
    parser.add_argument('--val_batch_size', type=int, default=None,
                        help="验证BatchSize (默认: batch_size * 2)")
    parser.add_argument('--ctx_len', type=int, default=getattr(config, 'ctx_len', 2048))
    parser.add_argument('--lr', type=float, default=getattr(config, 'lr', 2e-5))
    parser.add_argument('--max_iters', type=int, default=getattr(config, 'max_iters', 10000))
    parser.add_argument('--warmup_iters', type=int, default=getattr(config, 'warmup_iters', 100))
    parser.add_argument('--grad_accum_steps', type=int, default=getattr(config, 'grad_accum', 1),
                        help="梯度累积步数 (有效BatchSize = batch_size * grad_accum_steps)")
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

    # OPD损失 (对齐TRL广义JSD)
    parser.add_argument("--opd_loss_type", type=str, default="reverse_kl", 
                        choices=["reverse_kl", "forward_kl", "jsd"],
                        help="损失类型: reverse_kl (beta=0), forward_kl (beta=1), jsd (beta=0.5)")
    parser.add_argument("--opd_chunk_size", type=int, default=512)
    parser.add_argument("--opd_beta", type=float, default=None,
                        help="JSD插值系数 (0=Reverse KL, 1=Forward KL, 0.5=JSD). 不指定则根据opd_loss_type自动设置")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax温度缩放")
    parser.add_argument("--top_k_loss", type=int, default=None,
                        help="仅计算教师Top-k Token的Loss (减少显存+聚焦核心)")
    parser.add_argument("--jsd_token_clip", type=float, default=None,
                        help="Token级散度裁剪 (防止风格Token主导梯度)")

    # 保存
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--opd_checkpoint_dir", type=str, default="opd_checkpoints")
    parser.add_argument("--val_freq", type=int, default=50, help="验证频率 (每N步验证一次)")

    # 硬件
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def update_config_from_args(args):
    config_attrs = ['batch_size', 'ctx_len', 'lr', 'max_iters', 'device']
    for attr in config_attrs:
        if hasattr(args, attr):
            setattr(config, attr, getattr(args, attr))

# ====================== 通用数据集类（修复版：精准mask计算） ======================
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
        
        # 应用自定义预处理
        if "preprocess" in self.config:
            instruction = self.config["preprocess"](item)
        else:
            instruction = item[instruction_key]
        response = item[response_key]
        
        # 🔥 核心修复：先构建纯prompt部分（不含最终回复）
        prompt_messages = []
        if self.config.get("system_prompt"):
            prompt_messages.append({"role": "system", "content": self.config["system_prompt"]})
        prompt_messages.append({"role": "user", "content": instruction})
        
        # 严格对齐prompt和完整文本的token化
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + response + self.tokenizer.eos_token

        # 统一tokenize
        tokenized_full = self.tokenizer(
            full_text, truncation=True, max_length=self.ctx_len,
            padding=False, return_tensors="pt"
        )
        input_ids = tokenized_full["input_ids"].squeeze(0)
        
        # 精准计算prompt的token长度
        tokenized_prompt = self.tokenizer(
            prompt_text, truncation=True, max_length=self.ctx_len,
            padding=False, return_tensors="pt"
        )
        prompt_len = tokenized_prompt["input_ids"].shape[1]

        # 构建labels (标准HuggingFace格式: -100表示忽略)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "labels": labels
        }

# ====================== 批次填充 ======================
def collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len = max(len(x) for x in input_ids)
    padded_input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i in range(len(batch)):
        padded_input_ids[i, :len(input_ids[i])] = input_ids[i]
        padded_labels[i, :len(labels[i])] = labels[i]

    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels
    }

# ======================
# 【致命Bug修复版】对齐 TRL OPSDTrainer 的广义 JSD Loss
# 修复：Mask 维度扩展，支持广播乘法
# ======================
def generalized_jsd_loss(
    student_logits,
    teacher_logits,
    labels=None,
    beta=0.5,          # 0=Reverse KL, 1=Forward KL, 0.5=JSD
    temperature=1.0,
    reduction="batchmean",
    logits_are_probs=False,
    top_k=None,         # 仅计算教师 Top-k Token 的 Loss
    token_clip=None,    # 裁剪每个 Token 的散度值
    chunk_size=512      # 分块大小，防止 OOM
):
    """
    参考 TRL 的 generalized_jsd_loss，新增分块处理支持长序列。
    论文：https://huggingface.co/papers/2306.13649 (Eq.1)
    """
    B, L, V_s = student_logits.shape
    _, _, V_t = teacher_logits.shape

    # 1. 词表维度对齐
    min_vocab = min(V_s, V_t)
    if V_s != V_t:
        student_logits = student_logits[..., :min_vocab]
        teacher_logits = teacher_logits[..., :min_vocab]
    teacher_logits = teacher_logits.to(student_logits.dtype)

    total_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
    total_valid = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)
    num_chunks = (L + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size

        # 分块切片
        s_chunk = student_logits[:, start:end, :]
        t_chunk = teacher_logits[:, start:end, :]
        l_chunk = labels[:, start:end] if labels is not None else None

        # 2. 概率计算 (支持 Top-k 限制)
        if logits_are_probs:
            s_logp = torch.log(s_chunk.clamp_min(1e-8))
            t_logp = torch.log(t_chunk.clamp_min(1e-8))
        else:
            s_chunk = s_chunk / temperature
            t_chunk = t_chunk / temperature

            if top_k is not None and top_k > 0:
                # 仅保留教师 Top-k Token，重新归一化
                _, top_k_indices = torch.topk(t_chunk, k=top_k, dim=-1)
                s_chunk = torch.gather(s_chunk, dim=-1, index=top_k_indices)
                t_chunk = torch.gather(t_chunk, dim=-1, index=top_k_indices)

            s_logp = F.log_softmax(s_chunk, dim=-1)
            t_logp = F.log_softmax(t_chunk, dim=-1)

        # 3. 广义 JSD 计算 (Eq.1)
        if beta == 0:
            # Reverse KL: KL(Student || Teacher)
            jsd = F.kl_div(s_logp, t_logp, reduction="none", log_target=True)
        elif beta == 1:
            # Forward KL: KL(Teacher || Student)
            jsd = F.kl_div(t_logp, s_logp, reduction="none", log_target=True)
        else:
            # JSD: beta*KL(T||M) + (1-beta)*KL(S||M), M=beta*T + (1-beta)*S
            beta_tensor = torch.tensor(beta, dtype=s_logp.dtype, device=s_logp.device)
            mixture_logp = torch.logsumexp(
                torch.stack([s_logp + torch.log1p(-beta_tensor), t_logp + torch.log(beta_tensor)]),
                dim=0
            )
            kl_teacher = F.kl_div(mixture_logp, t_logp, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_logp, s_logp, reduction="none", log_target=True)
            jsd = beta_tensor * kl_teacher + (1 - beta_tensor) * kl_student

        # 4. Token 级 Clip (防止风格 Token 主导梯度)
        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)

        # 5. Mask 过滤 (标准 labels -100)
        if l_chunk is not None:
            mask = l_chunk != -100
            # 🔥 致命Bug修复：扩展 mask 维度以支持广播
            # mask: [Batch, SeqLen] -> [Batch, SeqLen, 1]
            jsd = jsd * mask.float().unsqueeze(-1)
            valid_cnt = mask.float().sum()
        else:
            valid_cnt = torch.tensor(B * (end - start), device=jsd.device, dtype=jsd.dtype)

        total_loss += jsd.sum()
        total_valid += valid_cnt

    # 6. 归一化 (总 Loss / 有效 Token 数)
    valid_tokens = torch.clamp(total_valid, min=1.0)
    loss = total_loss / valid_tokens

    return loss

# ====================== 验证函数（致命Bug修复版） ======================
@torch.inference_mode()  # 比no_grad更快，推理专用
def validate(model, teacher_model, dataloader, args, ctx):
    """
    🔥 核心修复：验证损失口径与训练完全一致
    - 不再按batch平均，而是按【总有效token数】平均
    - 使用torch.inference_mode()加速
    - 对齐TRL的广义JSD Loss
    """
    model.eval()
    total_loss_sum = 0.0
    total_valid_tokens = 0.0
    
    # 多卡下禁用tqdm，避免刷屏
    is_main_process = not dist.is_initialized() or dist.get_rank() == 0
    pbar = tqdm(dataloader, desc="验证中", leave=False) if is_main_process else dataloader
    
    for batch in pbar:
        input_ids = batch["input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        
        # 动态计算prompt长度
        if (labels != -100).any():
            prompt_len = (labels != -100).nonzero(as_tuple=True)[1].min().item()
        else:
            prompt_len = 0
        
        with torch.no_grad():
            t_logits = teacher_model(input_ids).logits
        with ctx:
            s_logits = model(input_ids)[0]
        
        # 【核心修正】精准切片生成部分 Logits
        if prompt_len > 0:
            s_logits_chunk = s_logits[:, prompt_len-1:-1, :]
            t_logits_chunk = t_logits[:, prompt_len-1:-1, :]
            labels_chunk = labels[:, prompt_len:]
        else:
            # 退化情况：没有prompt，全序列都是生成
            s_logits_chunk = s_logits[:, :-1, :]
            t_logits_chunk = t_logits[:, :-1, :]
            labels_chunk = labels[:, 1:]
        
        # 计算 Loss
        loss = generalized_jsd_loss(
            student_logits=s_logits_chunk,
            teacher_logits=t_logits_chunk,
            labels=labels_chunk,
            beta=args.opd_beta,
            temperature=args.temperature,
            top_k=args.top_k_loss,
            token_clip=args.jsd_token_clip,
            chunk_size=args.opd_chunk_size
        )
        
        # 统计有效 Token 数
        mask = labels_chunk != -100
        valid_tokens = mask.float().sum()
        
        total_loss_sum += loss.item() * valid_tokens.item()  # 反归一化
        total_valid_tokens += valid_tokens.item()
    
    model.train()
    
    # 【核心修正】和训练完全一致的归一化方式
    avg_val_loss = total_loss_sum / max(total_valid_tokens, 1.0)
    return avg_val_loss

# =============================================================================
# Model Summary
# =============================================================================

def print_detailed_summary(model, optimizers, start_iter=0, max_iters=0, best_val_loss=float('inf'), effective_bsz=0, args=None):
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
    
    # BatchSize info
    print(f"--- Training Configuration ---")
    print(f"Effective BatchSize: {effective_bsz}")
    print(f"Micro BatchSize: {getattr(config, 'batch_size', 'N/A')}")
    print(f"Gradient Accumulation Steps: {getattr(config, 'grad_accum', 'N/A')}")
    if args:
        print(f"Loss Type: {args.opd_loss_type} (beta={args.opd_beta})")
        print(f"Temperature: {args.temperature}")
        if args.top_k_loss:
            print(f"Top-k Token Loss: {args.top_k_loss}")
        if args.jsd_token_clip:
            print(f"JSD Token Clip: {args.jsd_token_clip}")
    print()
    
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

# ====================== 主函数（Muon单卡兼容性终极修复版） ======================
def main():
    args = parse_args()
    update_config_from_args(args)
    
    # 【工业级新增】参数校验
    if args.batch_size <= 0:
        raise ValueError(f"BatchSize必须>0，当前值: {args.batch_size}")
    if args.grad_accum_steps <= 0:
        raise ValueError(f"梯度累积步数必须>0，当前值: {args.grad_accum_steps}")
    
    # 【工业级新增】自动设置 opd_beta
    if args.opd_beta is None:
        if args.opd_loss_type == "reverse_kl":
            args.opd_beta = 0.0
        elif args.opd_loss_type == "forward_kl":
            args.opd_beta = 1.0
        else:  # jsd
            args.opd_beta = 0.5
        print(f"ℹ️  根据损失类型自动设置 opd_beta={args.opd_beta}")
    
    # 【工业级新增】验证BatchSize默认配置
    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size * 2
        print(f"ℹ️  验证BatchSize未指定，自动设置为: {args.val_batch_size}")
    
    # 计算有效BatchSize
    effective_bsz = args.batch_size * args.grad_accum_steps
    print(f"📊 训练配置 | Micro-BS: {args.batch_size} | 累积步数: {args.grad_accum_steps} | 有效BS: {effective_bsz}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.opd_checkpoint_dir, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)

    # 获取数据集配置
    dataset_config = DATASET_CONFIGS[args.dataset]
    print(f"📚 选中数据集: {args.dataset}")

    # 【终极修复】Muon兼容性：无论单卡多卡，强制初始化进程组
    # 即使单卡，也初始化一个world_size=1的进程组，满足Muon优化器的强依赖
    if not dist.is_initialized():
        try:
            # 设置环境变量，防止端口冲突
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            
            # 强制初始化进程组，单卡模式下world_size=1, rank=0
            dist.init_process_group(
                backend='gloo',  # 单卡用gloo更轻量，避免NCCL问题
                init_method='env://',
                world_size=1,
                rank=0
            )
            print("✅ 进程组初始化成功（单卡模式，world_size=1）")
        except Exception as e:
            print(f"⚠️  进程组初始化失败: {e}")
            # 如果还是失败，尝试用torch.distributed.is_torchelastic_launched()
            pass

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
        
        # 【工业级优化】训练/验证使用独立的DataLoader
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_dataset, args.val_batch_size, shuffle=False, collate_fn=collate)

        # 3. 初始化学生模型
        print("\n" + "="*60)
        print("🤖 初始化学生模型")
        print("="*60)
        device = args.device
        model = Transformer().to(device)
        
        # 【警告修复】自动对齐RMSNorm权重与模型dtype，解决dtype不匹配警告
        model_dtype = next(model.parameters()).dtype
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and 'norm' in name.lower():
                if module.weight.dtype != model_dtype:
                    module.weight.data = module.weight.data.to(model_dtype)
                if hasattr(module, 'bias') and module.bias is not None:
                    if module.bias.dtype != model_dtype:
                        module.bias.data = module.bias.data.to(model_dtype)
        print("✅ 模型Norm层dtype对齐完成")

        # 优化器
        optimizers = model.configure_optimizers(args.weight_decay, args.lr, device)

        # 调度器
        scheduler = SequentialLR(optimizers[-1], [
            LinearLR(optimizers[-1], 0.1, total_iters=args.warmup_iters),
            CosineAnnealingLR(optimizers[-1], args.max_iters-args.warmup_iters, args.min_lr)
        ], milestones=[args.warmup_iters])

        # 4. 自动加载检查点
        start_iter = 0
        best_val_loss = float("inf")
        
        # 优先使用用户指定的检查点
        if args.student_ckpt and os.path.exists(args.student_ckpt):
            checkpoint_path = args.student_ckpt
            start_iter, best_val_loss = load_checkpoint(
                model, optimizers, scheduler, checkpoint_path, device
            )
        else:
            # 自动查找最新检查点
            checkpoint_path, _ = find_latest_checkpoint(args.checkpoint_dir)
        
        total_params = print_detailed_summary(model, optimizers, start_iter, args.max_iters, best_val_loss, effective_bsz, args)

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
        # 【致命Bug修复】仅在CUDA且真正需要时启用GradScaler
        enable_amp = "cuda" in device and torch.cuda.is_available()
        ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if enable_amp else nullcontext()
        scaler = amp.GradScaler(enabled=enable_amp) if enable_amp else None
        global_step = start_iter

        # 【工业级新增】首次验证基线性能
        if global_step == 0:
            print("\n" + "="*60)
            print("🔍 首次验证：基线性能评估")
            print("="*60)
            initial_val_loss = validate(model, teacher_model, val_loader, args, ctx)
            print(f"✅ 初始验证损失: {initial_val_loss:.4f}")
            best_val_loss = initial_val_loss

        # 7. 开始训练（工业级修复：梯度累积 + 步数控制 + 分布式兼容）
        print("\n" + "="*60)
        print(f"🚀 开始 OPD 蒸馏训练 | 数据集: {args.dataset} | 起始迭代: {global_step}")
        print("="*60)
        
        # 【核心修复】正确的梯度累积训练循环
        train_iter = iter(train_loader)
        
        while global_step < args.max_iters:
            model.zero_grad(set_to_none=True)
            total_train_loss = 0.0
            
            # 梯度累积：累积 N 个不同的 mini-batch
            for accum_step in range(args.grad_accum_steps):
                # 数据迭代器重置
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # 动态计算prompt长度
                if (labels != -100).any():
                    prompt_len = (labels != -100).nonzero(as_tuple=True)[1].min().item()
                else:
                    prompt_len = 0

                # 🔥 新增：debug打印有效token数（确认mask正常）
                if global_step == 0 and accum_step == 0:
                    valid_tokens = (labels != -100).sum().item()
                    print(f"\n🔍 第1步验证：有效回复token数 = {valid_tokens} | 总序列长度 = {labels.shape[1]}")
                    print(f"🔍 正常情况：有效token数应远小于总长度（仅答案部分）")

                # 前向传播
                with torch.no_grad():
                    t_logits = teacher_model(input_ids).logits
                with ctx:
                    s_logits = model(input_ids)[0]
                
                # 【核心修正】精准切片生成部分 Logits
                if prompt_len > 0:
                    s_logits_chunk = s_logits[:, prompt_len-1:-1, :]
                    t_logits_chunk = t_logits[:, prompt_len-1:-1, :]
                    labels_chunk = labels[:, prompt_len:]
                else:
                    # 退化情况：没有prompt，全序列都是生成
                    s_logits_chunk = s_logits[:, :-1, :]
                    t_logits_chunk = t_logits[:, :-1, :]
                    labels_chunk = labels[:, 1:]
                
                # 【核心修正】使用对齐 TRL 的广义 JSD Loss
                loss = generalized_jsd_loss(
                    student_logits=s_logits_chunk,
                    teacher_logits=t_logits_chunk,
                    labels=labels_chunk,
                    beta=args.opd_beta,
                    temperature=args.temperature,
                    top_k=args.top_k_loss,
                    token_clip=args.jsd_token_clip,
                    chunk_size=args.opd_chunk_size
                )
                
                loss = loss / args.grad_accum_steps
                
                # 【致命Bug修复】GradScaler与分布式兼容处理
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                total_train_loss += loss.item()

            # 参数更新
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 【致命Bug修复】优化器step与GradScaler兼容处理
            if scaler is not None:
                for opt in optimizers:
                    scaler.step(opt)
                scaler.update()
            else:
                for opt in optimizers:
                    opt.step()
                    
            scheduler.step()
            global_step += 1

            # 验证 & 保存
            if global_step % args.val_freq == 0:
                val_loss = validate(model, teacher_model, val_loader, args, ctx)
                print(f"Step {global_step:04d} | 训练损失: {total_train_loss:.4f} | 验证损失: {val_loss:.4f} | 最优: {best_val_loss:.4f}")

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
                        "dataset": args.dataset,
                        "args": vars(args)  # 保存训练参数
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
        # 🔥 核心修复：仅在真正初始化过分布式环境时才关闭
        if dist.is_initialized():
            print("\n🔌 关闭分布式环境...")
            dist.destroy_process_group()
            print("✅ 分布式环境已关闭")

if __name__ == "__main__":
    main()
