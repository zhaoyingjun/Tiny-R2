#!/usr/bin/env python3
"""
Tiny-R2 OPD 训练 - 多数据集 & RAG 增强版
(Teacher 9B + RAG -> Student Tiny-R2 / 0.8B)
"""

import os
import sys
import random
import argparse
import glob
import re
import json
from typing import Optional, List, Dict, Any, Tuple
from contextlib import nullcontext
import time

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

from datasets import load_dataset, Dataset as HFDataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# RAG 依赖
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    print("⚠️ 未安装 faiss 或 sentence_transformers，使用模拟 RAG")
    print("安装命令: pip install faiss-cpu sentence_transformers")
    RAG_AVAILABLE = False

# Tiny-R2 核心模块
try:
    import model as tiny_model
    from model import Transformer
    import config
    print("✅ 成功导入 Tiny-R2 核心模块")
    TINY_R2_AVAILABLE = True
except ImportError as e:
    print(f"ℹ️ 未检测到 Tiny-R2 模块，使用 HuggingFace 模型")
    TINY_R2_AVAILABLE = False

# 工具函数
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    return False

# 数据集配置
DATASET_CONFIGS = {
    "medquad": {
        "hf_path": "keivalya/MedQuad-MedicalQnADataset",
        "hf_subset": None,
        "split": "train",
        "language": "en",
        "instruction_key": "Question",
        "response_key": "Answer",
        "student_system_prompt": "You are a professional and helpful AI assistant. Please provide accurate and safe answers to the user's questions.",
        "teacher_system_prompt": "You are an authoritative expert. Please answer the user's question strictly based on the following [Authoritative Reference].\n\n[Authoritative Reference]\n{rag_context}"
    },
    "cmeqa": {
        "hf_path": "blcu-nlp/CMeQA",
        "hf_subset": None,
        "split": "train",
        "language": "zh",
        "instruction_key": "question",
        "response_key": "answer",
        "student_system_prompt": "你是一个专业、有用的人工智能助手。请准确、安全地解答用户的问题。",
        "teacher_system_prompt": "你是一个权威的专家。请根据以下【权威参考资料】，专业、严谨地解答用户的问题。\n\n[权威参考资料]\n{rag_context}"
    },
    "gpqa_diamond": {
        "hf_path": "Idavidrein/gpqa",
        "hf_subset": "gpqa_diamond",
        "split": "train",
        "language": "en",
        "instruction_key": "Question",
        "response_key": "Correct Answer",
        "student_system_prompt": "You are a helpful AI assistant. Please solve the following question.",
        "teacher_system_prompt": "You are an authoritative expert. Please solve the question based on the reference.\n\n[Reference]\n{rag_context}"
    },
    "math500": {
        "hf_path": "math-ai/math500",
        "hf_subset": None,
        "split": "test",
        "language": "en",
        "instruction_key": "problem",
        "response_key": "solution",
        "student_system_prompt": "Please solve the following math problem and provide a detailed step-by-step solution.",
        "teacher_system_prompt": "Please solve the math problem using the given [Reference Text].\n\n[Reference Text]\n{rag_context}"
    }
}

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None, 0
    
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
    
    best_pattern = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(best_pattern):
        return best_pattern, 0
    
    return None, 0

def print_detailed_summary(model, optimizers, schedulers, start_iter=0, max_iters=0, best_val_loss=float('inf')):
    print("\n" + "=" * 60)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.3f} M\n")
    
    print("--- Transformer Architecture ---")
    if TINY_R2_AVAILABLE and hasattr(config, 'n_layer'):
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
    
    print("--- MoE Configuration ---")
    if TINY_R2_AVAILABLE and hasattr(config, 'n_experts'):
        print(f"Number of experts: {config.n_experts}")
        print(f"Active experts per token: {getattr(config, 'num_exp', 'N/A')}")
        print(f"Shared experts: {getattr(config, 'shared_experts', 'N/A')}")
    print()
    
    print("--- Optimizers & Schedulers ---")
    for i, (opt, sch) in enumerate(zip(optimizers, schedulers)):
        opt_type = type(opt).__name__
        for j, pg in enumerate(opt.param_groups):
            lr = pg.get('lr', 'N/A')
            wd = pg.get('weight_decay', 'N/A')
            print(f"Optimizer {i} ({opt_type}) group {j}: lr={lr}, wd={wd}")
    
    print("\n" + "=" * 60 + "\n")
    return total_params

def load_checkpoint(model, optimizers, schedulers, checkpoint_path, device):
    print(f"📂 加载检查点: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_state = checkpoint.get('model', checkpoint)
    is_compiled_model = hasattr(model, '_orig_mod')
    is_compiled_ckpt = any(k.startswith("_orig_mod.") for k in model_state.keys())
    
    if is_compiled_model and not is_compiled_ckpt:
        model_state = {f"_orig_mod.{k}": v for k, v in model_state.items()}
    elif not is_compiled_model and is_compiled_ckpt:
        model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
    
    current_sd = model.state_dict()
    filtered_state = {}
    vocab_expanded = False 
    
    for key, ckpt_tensor in model_state.items():
        if key not in current_sd:
            continue
        current_tensor = current_sd[key]
        
        if ckpt_tensor.shape == current_tensor.shape:
            filtered_state[key] = ckpt_tensor
        elif key in ["token_embedding_table.weight", "lm_head.weight"] or "embed_tokens" in key or "lm_head" in key:
            ckpt_vocab, _ = ckpt_tensor.shape
            curr_vocab, _ = current_tensor.shape
            if ckpt_vocab < curr_vocab:
                new_tensor = current_tensor.clone()
                new_tensor[:ckpt_vocab, :] = ckpt_tensor
                filtered_state[key] = new_tensor
                vocab_expanded = True
            else:
                print(f"⚠️ 跳过 {key}: 检查点词表更大")
    
    model.load_state_dict(filtered_state, strict=False)
    print(f"✅ 模型权重加载成功！")
    
    if not vocab_expanded:
        if 'optimizer_states' in checkpoint and optimizers:
            try:
                for opt, state in zip(optimizers, checkpoint['optimizer_states']):
                    opt.load_state_dict(state)
                print("✅ 优化器状态加载成功！")
            except Exception as e:
                print(f"⚠️ 优化器状态恢复失败: {e}")
                
        if schedulers and 'scheduler_states' in checkpoint:
            try:
                for sch, state in zip(schedulers, checkpoint['scheduler_states']):
                    sch.load_state_dict(state)
                print("✅ 调度器状态加载成功！")
            except Exception as e:
                print(f"⚠️ 调度器状态恢复失败: {e}")

    start_iter = checkpoint.get('iter', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"✅ 检查点加载完成 | 迭代: {start_iter} | 最优loss: {best_val_loss:.4f}")
    return start_iter, best_val_loss

def save_atomic_checkpoint(model, optimizers, schedulers, iteration, filepath, best_val_loss, args):
    model_state = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        
    checkpoint = {
        'model': model_state,
        'optimizer_states': [opt.state_dict() for opt in optimizers],
        'scheduler_states': [sch.state_dict() for sch in schedulers],
        'iter': iteration,
        'best_val_loss': best_val_loss,
        'args': vars(args)
    }
    
    temp_path = filepath + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, filepath)

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Dataset & RAG-Augmented OPD Train")

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--ctx_len', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_iters', type=int, default=10000)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6)

    parser.add_argument("--dataset", type=str, default="medquad")
    parser.add_argument("--language", type=str, default=None, choices=["en", "zh"])
    parser.add_argument("--hf_teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    
    parser.add_argument("--student_model_name", type=str, default="tiny-r2")
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3.5-9B")

    parser.add_argument("--enable_rag_teacher", action="store_true", default=False)
    parser.add_argument("--rag_top_k", type=int, default=2)
    parser.add_argument("--rag_corpus_path", type=str, default=None)
    parser.add_argument("--custom_qa_path", type=str, default=None)

    parser.add_argument("--opd_loss_type", type=str, default="jsd", choices=["reverse_kl", "forward_kl", "jsd"])
    parser.add_argument("--opd_chunk_size", type=int, default=512)
    parser.add_argument("--opd_beta", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k_loss", type=int, default=None)
    parser.add_argument("--jsd_token_clip", type=float, default=None)
    
    parser.add_argument("--opd_checkpoint_dir", type=str, default="opd_checkpoints")
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

class RAGManager:
    def __init__(self, use_rag=True, corpus_texts=None, language="zh"):
        self.use_rag = use_rag and RAG_AVAILABLE
        self.corpus = corpus_texts if corpus_texts else []
        self.language = language
        self.embedder = None
        self.index = None
        
        if self.use_rag and len(self.corpus) > 0:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2' if language == "en" else 'shibing624/text2vec-base-chinese'
            self.embedder = SentenceTransformer(model_name)
            embeddings = self.embedder.encode(self.corpus, show_progress_bar=True)
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            print(f"✅ RAG 向量库构建完成，共 {len(self.corpus)} 条知识。")

    def search(self, query: str, top_k: int = 2) -> str:
        if not self.use_rag or self.index is None:
            return "据知识库显示，请严谨作答。" if self.language == "zh" else "According to the knowledge base, please answer cautiously."
            
        q_emb = self.embedder.encode([query])
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        
        retrieved = [self.corpus[idx] for idx in I[0] if idx < len(self.corpus)]
        return "\n\n".join(retrieved)

def smart_truncate_prompt(prompt_ids, max_len, tokenizer, system_end_token="<|im_end|>"):
    if len(prompt_ids) <= max_len:
        return prompt_ids
    
    system_end_ids = tokenizer.encode(system_end_token, add_special_tokens=False)
    prompt_list = prompt_ids.tolist()
    
    for i in range(len(prompt_list) - len(system_end_ids) + 1):
        if prompt_list[i:i+len(system_end_ids)] == system_end_ids:
            system_end_pos = i + len(system_end_ids)
            break
    else:
        return prompt_ids[-max_len:]
    
    system_part = prompt_ids[:system_end_pos]
    user_part = prompt_ids[system_end_pos:]
    
    if len(system_part) > max_len:
        return prompt_ids[-max_len:]
    else:
        remaining_len = max_len - len(system_part)
        return torch.cat([system_part, user_part[-remaining_len:]])

class DualPromptDataset(TorchDataset):
    def __init__(self, hf_dataset, tokenizer, ctx_len: int, dataset_config: Dict[str, Any], rag_manager: RAGManager, args):
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.dataset = hf_dataset
        self.config = dataset_config
        self.rag_manager = rag_manager
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.dataset[idx]
        instruction = item[self.config["instruction_key"]]
        response = item[self.config["response_key"]]
        
        response_text = response + self.tokenizer.eos_token
        response_ids = self.tokenizer(response_text, return_tensors="pt")["input_ids"].squeeze(0)
        
        max_response_len = self.ctx_len // 2
        if len(response_ids) > max_response_len:
            response_ids = response_ids[:max_response_len]
        max_prompt_len = self.ctx_len - len(response_ids)
        
        # 学生Prompt
        s_system = self.config["student_system_prompt"]
        s_messages = [{"role": "system", "content": s_system}, {"role": "user", "content": instruction}]
        s_prompt_text = self.tokenizer.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True)
        s_prompt_ids = self.tokenizer(s_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)
        s_prompt_ids = smart_truncate_prompt(s_prompt_ids, max_prompt_len, self.tokenizer)
        
        # 教师Prompt
        if self.args.enable_rag_teacher:
            retrieved_context = self.rag_manager.search(instruction, top_k=self.args.rag_top_k)
            t_system = self.config["teacher_system_prompt"].format(rag_context=retrieved_context)
            t_messages = [{"role": "system", "content": t_system}, {"role": "user", "content": instruction}]
            t_prompt_text = self.tokenizer.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True)
            t_prompt_ids = self.tokenizer(t_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)
            
            # 修正：不强制和学生等长，只要不超过最大剩余长度，保留 RAG 知识完整性
            t_prompt_ids = smart_truncate_prompt(t_prompt_ids, max_prompt_len, self.tokenizer)
        else:
            t_prompt_ids = s_prompt_ids.clone()
            
        s_input_ids = torch.cat([s_prompt_ids, response_ids])
        t_input_ids = torch.cat([t_prompt_ids, response_ids])
        
        s_labels = torch.full_like(s_input_ids, -100)
        s_labels[len(s_prompt_ids):] = response_ids
        t_labels = torch.full_like(t_input_ids, -100)
        t_labels[len(t_prompt_ids):] = response_ids

        return {
            "s_input_ids": s_input_ids, "s_labels": s_labels,
            "t_input_ids": t_input_ids, "t_labels": t_labels
        }

def dual_collate_fn(batch: List[Dict[str, torch.Tensor]], tokenizer):
    def pad_tensors(key_ids, key_labels):
        items_ids = [item[key_ids] for item in batch]
        items_labels = [item[key_labels] for item in batch]
        max_len = max(len(x) for x in items_ids)
        
        padded_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i in range(len(batch)):
            padded_ids[i, :len(items_ids[i])] = items_ids[i]
            padded_labels[i, :len(items_labels[i])] = items_labels[i]
        return padded_ids, padded_labels

    s_ids, s_labels = pad_tensors("s_input_ids", "s_labels")
    t_ids, t_labels = pad_tensors("t_input_ids", "t_labels")

    return {
        "s_input_ids": s_ids, "s_labels": s_labels,
        "t_input_ids": t_ids, "t_labels": t_labels
    }

def extract_aligned_response_logits(s_logits, t_logits, s_labels, t_labels):
    """
    分别提取师生的有效token，只要目标 response 文本一致，即使 prompt 长度不同也能严格对齐
    """
    # 学生侧
    s_shift_logits = s_logits[..., :-1, :].contiguous()
    s_shift_labels = s_labels[..., 1:].contiguous()
    s_valid_mask = s_shift_labels != -100
    
    # 教师侧
    t_shift_logits = t_logits[..., :-1, :].contiguous()
    t_shift_labels = t_labels[..., 1:].contiguous()
    t_valid_mask = t_shift_labels != -100
    
    # 提取有效token
    s_valid_logits = s_shift_logits[s_valid_mask]
    t_valid_logits = t_shift_logits[t_valid_mask]
    
    # 严格检查token数量是否一致
    if s_valid_logits.size(0) != t_valid_logits.size(0):
        raise RuntimeError(
            f"师生有效token数量不匹配: 学生={s_valid_logits.size(0)}, 教师={t_valid_logits.size(0)}\n"
            f"学生序列长度: {s_logits.size(1)}, 教师序列长度: {t_logits.size(1)}\n"
            "这通常是由于截断策略导致 response 被切断程度不同造成的"
        )
    
    return s_valid_logits, t_valid_logits

def generalized_jsd_loss_flat_correct(
    s_valid_logits, t_valid_logits, 
    beta=0.5, temperature=1.0, 
    top_k=None, token_clip=None, chunk_size=512
):
    N, V_s = s_valid_logits.shape
    M, V_t = t_valid_logits.shape
    
    if N != M:
        raise RuntimeError(f"Token数量不匹配: {N} vs {M}")

    # 防御词表大小不一致
    min_vocab = min(V_s, V_t)
    if V_s != V_t:
        s_valid_logits = s_valid_logits[:, :min_vocab]
        t_valid_logits = t_valid_logits[:, :min_vocab]
        
    s_valid_logits = s_valid_logits.float()
    t_valid_logits = t_valid_logits.float()
    
    total_loss = torch.tensor(0.0, device=s_valid_logits.device, dtype=torch.float32)
    num_chunks = (N + chunk_size - 1) // chunk_size
    eps = torch.finfo(torch.float32).eps

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, N)

        s_logits = s_valid_logits[start:end, :] / temperature
        t_logits = t_valid_logits[start:end, :] / temperature

        if top_k is not None and top_k > 0:
            # 教师获取 Top-K
            t_probs = F.softmax(t_logits, dim=-1)
            t_probs_topk, top_k_indices = torch.topk(t_probs, k=top_k, dim=-1)
            t_probs_topk = t_probs_topk / (t_probs_topk.sum(dim=-1, keepdim=True) + eps)
            
            # 修正：学生必须先做全局 Softmax，再 gather 获取局部概率分布
            s_probs_full = F.softmax(s_logits, dim=-1)
            s_probs_topk = torch.gather(s_probs_full, dim=-1, index=top_k_indices)
            s_probs_topk = s_probs_topk / (s_probs_topk.sum(dim=-1, keepdim=True) + eps)
            
            P = s_probs_topk
            Q = t_probs_topk
        else:
            P = F.softmax(s_logits, dim=-1)
            Q = F.softmax(t_logits, dim=-1)

        M = beta * Q + (1 - beta) * P
        kl_pm = F.kl_div((P + eps).log(), M, reduction="none", log_target=False).sum(dim=-1)
        kl_qm = F.kl_div((Q + eps).log(), M, reduction="none", log_target=False).sum(dim=-1)
        
        jsd = beta * kl_qm + (1 - beta) * kl_pm

        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)

        total_loss += jsd.sum()

    return total_loss / max(N, 1)

@torch.inference_mode()
def validate(student_model, teacher_model, dataloader, args, ctx):
    student_model.eval()
    total_loss_sum = 0.0
    total_valid_tokens = 0.0
    
    for batch in dataloader:
        s_input_ids = batch["s_input_ids"].to(args.device)
        s_labels = batch["s_labels"].to(args.device)
        t_input_ids = batch["t_input_ids"].to(args.device)
        t_labels = batch["t_labels"].to(args.device)
        
        t_logits = teacher_model(t_input_ids).logits
        with ctx:
            s_logits = student_model(s_input_ids).logits if hasattr(student_model, 'logits') else student_model(s_input_ids)[0]
            
        s_valid_logits, t_valid_logits = extract_aligned_response_logits(s_logits, t_logits, s_labels, t_labels)
        
        if s_valid_logits.size(0) > 0:
            loss = generalized_jsd_loss_flat_correct(
                s_valid_logits, t_valid_logits, 
                beta=args.opd_beta, 
                temperature=args.temperature,
                top_k=args.top_k_loss,
                token_clip=args.jsd_token_clip,
                chunk_size=args.opd_chunk_size
            )
            total_loss_sum += loss.item() * s_valid_logits.size(0)
            total_valid_tokens += s_valid_logits.size(0)
            
    student_model.train()
    return total_loss_sum / max(total_valid_tokens, 1.0)

def main():
    args = parse_args()
    
    if args.opd_beta is None:
        args.opd_beta = 0.0 if args.opd_loss_type == "reverse_kl" else (1.0 if args.opd_loss_type == "forward_kl" else 0.5)

    os.makedirs(args.opd_checkpoint_dir, exist_ok=True)
    device = args.device

    if not dist.is_initialized():
        try:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group(backend='gloo', init_method='env://', world_size=1, rank=0)
        except Exception:
            pass

    print("\n" + "="*60)
    print("🚀 启动 OPD 知识蒸馏训练")
    print(f"教师模型: {args.hf_teacher_model} | RAG: {args.enable_rag_teacher}")
    print(f"学生模型: {args.student_model_name}")
    print("="*60 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if TINY_R2_AVAILABLE:
        config.vocab_size = len(tokenizer)

    dataset_config = DATASET_CONFIGS.get(args.dataset, DATASET_CONFIGS["medquad"])
    if args.language:
        dataset_config["language"] = args.language

    if args.custom_qa_path:
        print(f"📂 加载自定义数据集: {args.custom_qa_path}")
        custom_data = []
        try:
            if args.custom_qa_path.endswith('.json'):
                with open(args.custom_qa_path, 'r', encoding='utf-8') as f:
                    custom_data = json.load(f)
            elif args.custom_qa_path.endswith('.jsonl'):
                with open(args.custom_qa_path, 'r', encoding='utf-8') as f:
                    custom_data = [json.loads(line) for line in f if line.strip()]
            
            sample = custom_data[0]
            q_key = next((k for k in ["question", "instruction", "q", "query"] if k in sample), None)
            a_key = next((k for k in ["answer", "response", "a", "output"] if k in sample), None)
            dataset_config["instruction_key"] = q_key
            dataset_config["response_key"] = a_key
            ds = HFDataset.from_list(custom_data)
        except Exception as e:
            print(f"❌ 数据集加载失败: {e}")
            sys.exit(1)
    else:
        ds = load_dataset(dataset_config["hf_path"], dataset_config["hf_subset"], split=dataset_config["split"])
        
    ds = ds.shuffle(seed=42)
    
    rag_corpus = []
    if args.enable_rag_teacher:
        if args.rag_corpus_path and os.path.exists(args.rag_corpus_path):
            try:
                if args.rag_corpus_path.endswith('.json'):
                    with open(args.rag_corpus_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for item in data:
                            if isinstance(item, str):
                                rag_corpus.append(item)
                            elif isinstance(item, dict):
                                text = item.get("text") or item.get("content")
                                if text:
                                    rag_corpus.append(text)
                elif args.rag_corpus_path.endswith('.jsonl'):
                    with open(args.rag_corpus_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            item = json.loads(line)
                            text = item.get("text") or item.get("content")
                            if text:
                                rag_corpus.append(text)
                else: 
                    with open(args.rag_corpus_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        paragraphs = [p.strip() for p in re.split(r'\n+', content) if p.strip()]
                        rag_corpus.extend(paragraphs)
            except Exception as e:
                print(f"❌ RAG 语料加载失败: {e}")

        dataset_corpus = [item[dataset_config["response_key"]] for item in ds.select(range(min(1000, len(ds))))]
        rag_corpus.extend(dataset_corpus)
        rag_corpus = list(set(rag_corpus))
        print(f"🔍 RAG 知识库大小: {len(rag_corpus)}")
        
    rag_manager = RAGManager(
        use_rag=args.enable_rag_teacher, 
        corpus_texts=rag_corpus,
        language=dataset_config.get("language", "zh") 
    )

    val_size = max(1, int(len(ds) * 0.1))
    train_dataset = DualPromptDataset(ds.select(range(val_size, len(ds))), tokenizer, args.ctx_len, dataset_config, rag_manager, args)
    val_dataset = DualPromptDataset(ds.select(range(val_size)), tokenizer, args.ctx_len, dataset_config, rag_manager, args)

    from functools import partial
    collate = partial(dual_collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate)

    compute_dtype = torch.bfloat16 if "cuda" in device and torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (compute_dtype == torch.float16)

    if args.student_model_name.lower() == "tiny-r2" and TINY_R2_AVAILABLE:
        print("\n🤖 加载 Tiny-R2 学生模型")
        student_model = Transformer().to(device)
        optimizers = student_model.configure_optimizers(args.weight_decay, args.lr, device)
    else:
        print(f"\n🤖 加载 HF 学生模型: {args.student_model_name}")
        student_model = AutoModelForCausalLM.from_pretrained(args.student_model_name, torch_dtype=compute_dtype, device_map=device)
        optimizers = [torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)]
        
    schedulers = []
    for opt in optimizers:
        warmup_scheduler = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_iters)
        cosine_scheduler = CosineAnnealingLR(opt, T_max=args.max_iters - args.warmup_iters, eta_min=args.min_lr)
        scheduler = SequentialLR(opt, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.warmup_iters])
        schedulers.append(scheduler)

    start_iter = 0
    best_val_loss = float("inf")
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        start_iter, best_val_loss = load_checkpoint(student_model, optimizers, schedulers, args.student_ckpt, device)

    if 'cuda' in device:
        try:
            # student_model = torch.compile(student_model)
            print("⚡ 模型编译完成")
        except Exception as e:
            print(f"⚠️ 模型编译跳过: {e}")

    print_detailed_summary(student_model, optimizers, schedulers, start_iter, args.max_iters, best_val_loss)

    print(f"\n👨‍🏫 加载教师模型: {args.hf_teacher_model}")
    # 修正：将 dtype 改为了 torch_dtype
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.hf_teacher_model, 
        torch_dtype=compute_dtype, 
        device_map=device,
        trust_remote_code=True
    ).eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    ctx = torch.amp.autocast(device_type="cuda", dtype=compute_dtype) if "cuda" in device else nullcontext()
    scaler = amp.GradScaler(enabled=use_scaler)
    
    global_step = start_iter
    train_iter = iter(train_loader)
    student_model.train()
    
    while global_step < args.max_iters:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
            
        total_train_loss = 0.0
        total_valid_tokens = 0
        
        for accum_step in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            s_input_ids, s_labels = batch["s_input_ids"].to(device), batch["s_labels"].to(device)
            t_input_ids, t_labels = batch["t_input_ids"].to(device), batch["t_labels"].to(device)

            with torch.no_grad():
                t_logits = teacher_model(t_input_ids).logits
            with ctx:
                s_out = student_model(s_input_ids)
                s_logits = s_out.logits if hasattr(s_out, 'logits') else s_out[0]
            
            s_valid_logits, t_valid_logits = extract_aligned_response_logits(s_logits, t_logits, s_labels, t_labels)
            
            if s_valid_logits.size(0) == 0:
                continue

            loss = generalized_jsd_loss_flat_correct(
                s_valid_logits, t_valid_logits, 
                beta=args.opd_beta,
                temperature=args.temperature,
                top_k=args.top_k_loss,
                token_clip=args.jsd_token_clip,
                chunk_size=args.opd_chunk_size
            )
            
            # 修正：进行梯度累加前的 loss 缩放
            scaled_loss = loss / args.grad_accum_steps
            
            if use_scaler: 
                scaler.scale(scaled_loss).backward()
            else: 
                scaled_loss.backward()
                
            total_train_loss += loss.item() * s_valid_logits.size(0)
            total_valid_tokens += s_valid_logits.size(0)

        avg_train_loss = total_train_loss / max(total_valid_tokens, 1)
        
        if use_scaler:
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            for opt in optimizers: 
                scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            for opt in optimizers: 
                opt.step()
                
        for sch in schedulers:
            sch.step()
            
        global_step += 1

        if global_step % args.val_freq == 0:
            val_loss = validate(student_model, teacher_model, val_loader, args, ctx)
            current_lr = schedulers[-1].get_last_lr()[0]
            print(f"Step {global_step:04d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Best: {best_val_loss:.4f} | LR: {current_lr:.6e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.opd_checkpoint_dir, f"best_model_step_{global_step}.pt")
                save_atomic_checkpoint(student_model, optimizers, schedulers, global_step, save_path, best_val_loss, args)
                print(f"🏆 保存最优模型: {save_path}")

    if dist.is_initialized():
        dist.destroy_process_group()
    
    print("\n🎉 训练完成！")
    print(f"最优验证损失: {best_val_loss:.4f}")
    print(f"模型保存路径: {args.opd_checkpoint_dir}")

if __name__ == "__main__":
    main()
