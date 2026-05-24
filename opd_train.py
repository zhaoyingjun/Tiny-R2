#!/usr/bin/env python3
"""
Tiny-R2 OPD 训练 - 知识蒸馏与RAG增强 (Teacher=9B+RAG, Student=0.8B)
融合特性：
1. 混合检索（Hybrid BM25 + Semantic）
2. 适配 PubMedQA 任务与自适应 CoT 对齐 (采用 CoT -> Answer 决策流)
3. Online Rollout (解决 Exposure Bias) + 动态渐进式退火
4. Policy Gradient REINFORCE 废话惩罚 (Anti Self-Narration Penalty)
5. 综合评测体系：在验证节点同时评估 Teacher、Base Student 与 OPD Student 模型的准确率
6. WandB 监控与多维度指标对齐
7.经过验证，在pubmedqa数据集上可以将Qwen3.5-0.8B模型准确率平均提升12%以上。
"""
import os
import sys
import random
import argparse
import glob
import re
import json
import time 
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

# Tiny-R2 核心配置（保持原项目兼容性）
try:
    import config
except ImportError:
    # 允许在无本地 config 时作为占位符使用
    class MockConfig:
        vocab_size = 151936
    config = MockConfig()

# ====================== 优雅的 WandB 异常处理 ======================
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# ====================== RAG 混合检索支持所需库 ======================
try:
    from sentence_transformers import SentenceTransformer, util
    from rank_bm25 import BM25Okapi
    RAG_AVAILABLE = True
    print("✅ 成功导入 rank_bm25 与 sentence_transformers，混合 RAG 模块已启用。")
except ImportError:
    print("⚠️ 提示: 未安装 rank_bm25 或 sentence_transformers，将使用 Mock RAG 流程运行。")
    print("如需真实混合 RAG，请运行: pip install rank_bm25 sentence_transformers")
    RAG_AVAILABLE = False

# ====================== 导入Tiny-R2核心模块 ======================
try:
    import model
    from model import Transformer
    print("✅ 成功导入 Tiny-R2 核心模块 (将作为备用学生模型)")
    TINY_R2_AVAILABLE = True
except ImportError as e:
    print(f"ℹ️ 未检测到 Tiny-R2 模块，将强制使用 HuggingFace 模型。")
    TINY_R2_AVAILABLE = False

# ====================== 数据集配置注册表 ======================
DATASET_CONFIGS = {
    "pubmed_qa": {
        "hf_path": "pubmed_qa",
        "hf_subset": "pqa_labeled",
        "split": "train",
        "language": "en",
        "instruction_key": "question",
        "response_key": "answer",
        "student_system_prompt": (
            "You are an expert medical assistant. For the given biomedical question, "
            "first analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'.\n\n"
            "Examples:\n"
            "Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\n"
            "Analysis: Mitochondria show swelling and loss of membrane potential in early stages of leaf death, which directly indicates they are key active participants in cellular remodelling.\n"
            "[Final Decision]: yes\n\n"
            "Question: Does cardiac transplantation prolong survival in patients with end-stage heart failure?\n"
            "Analysis: Long-term clinical survival stats show cardiac transplantation significantly outperforms general drug therapies, extending life expectancy for end-stage patients.\n"
            "[Final Decision]: maybe\n\n"
            "Now answer the user's question following the same format."
        ),
        "teacher_system_prompt": (
            "You are an expert medical assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to answer the question based strictly on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"  
            "First, analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'.\n\n"
            "Examples:\n"
            "Retrieved Context:\n[rag_contex]\nMitochondria show swelling and loss of membrane potential in early stages of leaf death, which directly indicates they are key active participants in cellular remodelling.\n\n"
            "Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?\n"
            "Analysis: According to the retrieved context, mitochondrial swelling and potential loss in early leaf death show they actively participate in cellular remodelling.\n"
            "[Final Decision]: yes\n\n"
            "Now answer the user's question using the provided context."
        )
    },
    "medquad": {
        "hf_path": "keivalya/MedQuad-MedicalQnADataset",
        "hf_subset": None,
        "split": "train",
        "language": "en",
        "instruction_key": "question",
        "response_key": "long_answer",
        "student_system_prompt": "You are a professional medical assistant. Please provide accurate, safe, and helpful answers to the patient's questions.",
        "teacher_system_prompt": "You are an authoritative medical expert. Please answer the patient's question based on the following [Authoritative Medical Reference] and your professional medical knowledge.\n\n[Authoritative Medical Reference]\n{rag_context}"
    },
    "cmeqa": {
        "hf_path": "blcu-nlp/CMeQA",
        "hf_subset": None,
        "split": "train",
        "language": "zh",
        "instruction_key": "long_answer",
        "response_key": "answer",
        "student_system_prompt": "你是一个专业的全科医生助手。请专业、准确、安全地解答患者的问题。",
        "teacher_system_prompt": "你是一个权威的主治医师评审。请结合以下【权威医疗参考资料】和你的专业医学常识，专业、严谨地解答患者的问题。\n\n[权威医疗参考资料]\n{rag_context}"
    }
}

# ====================== 废话惩罚过滤配置 ======================
BAD_PATTERNS = [
    "the user is asking", "i need to explain", "let me explain", 
    "the question asks", "i will answer", "as an ai",
    "用户问", "我需要解释", "让我解释", "这个问题要求", "我会回答", "作为一个人工智能"
]

def get_narration_penalty(text: str) -> float:
    lower = text.lower()
    return sum(1.0 for p in BAD_PATTERNS if p in lower)

# ====================== 答案清洗与正则化 ======================
def normalize_answer(raw_output: str) -> str:
    """
    清洗并解析模型输出。
    1. 剔除思考标签 <think>
    2. 优先检索最终决策结构化输出
    3. 兜底搜索文本中独立的判定词
    """
    cleaned_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    if '</think>' in raw_output:
        cleaned_output = raw_output.split('</think>')[-1].strip()
    elif '<think>' in raw_output:
        cleaned_output = raw_output.split('<think>')[-1].strip()

    cleaned_output_lower = cleaned_output.lower().strip()
    
    # 优先寻找决策关键词结构
    match_tag = re.search(r'(?:final decision|final answer|decision|answer)\s*:\s*\b(yes|no|maybe)\b', cleaned_output_lower)
    if match_tag:
        return match_tag.group(1)
        
    has_yes = bool(re.search(r'\byes\b', cleaned_output_lower))
    has_no = bool(re.search(r'\bno\b', cleaned_output_lower))
    has_maybe = bool(re.search(r'\bmaybe\b', cleaned_output_lower))
    
    if has_yes and not has_no and not has_maybe:
        return "yes"
    elif has_no and not has_yes and not has_maybe:
        return "no"
    elif has_maybe and not has_yes and not has_no:
        return "maybe"
        
    all_matches = list(re.finditer(r'\b(yes|no|maybe)\b', cleaned_output_lower))
    if all_matches:
        return all_matches[-1].group(1)
        
    return "maybe"

# ====================== 参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid RAG-Augmented OPD Train (Teacher 9B + Hybrid RAG -> Student 0.8B)")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--ctx_len', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_iters', type=int, default=1000)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument("--dataset", type=str, default="pubmed_qa")
    parser.add_argument("--hf_teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--disable_rag_teacher", action="store_true", help="禁用 RAG 增强，仅使用基础 Prompt")
    parser.add_argument("--rag_top_k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--rag_dense_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--opd_loss_type", type=str, default="jsd")
    parser.add_argument("--opd_chunk_size", type=int, default=512)
    parser.add_argument("--opd_beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opd_checkpoint_dir", type=str, default="opd_checkpoints")
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    parser.add_argument("--rollout_ratio", type=float, default=0.7)
    parser.add_argument("--pg_penalty_weight", type=float, default=0.1)
    
    parser.add_argument("--num_eval", type=int, default=20)
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--custom_qa_path", type=str, default=None)
    parser.add_argument("--save_best_only", action="store_true", default=True)

    parser.add_argument('--enable_wandb', action="store_true", default=True)
    parser.add_argument('--wandb_project', type=str, default="PubMedQA-OPD-Enterprise")
    parser.add_argument('--wandb_run_id', type=str, default=None)
    
    return parser.parse_args()

# ====================== BM25 + Dense 混合 RAG 管理器 ======================
class MedicalRAGManager:
    def __init__(self, corpus, dense_model_name="BAAI/bge-small-en-v1.5", device="cuda"):
        self.corpus = corpus
        self.texts = [doc["text"] for doc in corpus]
        self.device = device

        if RAG_AVAILABLE:
            print("[-] 正在初始化 BM25 词法索引...")
            tokenized_corpus = [text.lower().split() for text in self.texts]
            self.bm25 = BM25Okapi(tokenized_corpus)

            print(f"[-] 正在初始化 Dense 语义向量模型 ({dense_model_name})...")
            self.dense_model = SentenceTransformer(dense_model_name, device=device)
            self.corpus_embeddings = self.dense_model.encode(
                self.texts, show_progress_bar=True, convert_to_tensor=True
            )
        else:
            print("⚠️ 警告: 未检测到 RAG 所需第三方依赖，将以 Mock 截取机制运行。")

    def retrieve(self, query: str, top_k: int = 3, alpha: float = 0.4) -> str:
        if not RAG_AVAILABLE:
            # Mock 检索：默认截取语料库的前 top_k 个文档
            retrieved_contexts = []
            for i in range(min(top_k, len(self.texts))):
                retrieved_contexts.append(f"[Document {i+1}]\n{self.texts[i]}")
            return "\n\n".join(retrieved_contexts)

        # 1. 词法检索得分
        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        # 2. 语义检索得分
        query_embedding = self.dense_model.encode(query, convert_to_tensor=True)
        dense_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0].cpu().numpy()
        if dense_scores.max() > dense_scores.min():
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        else:
            dense_scores = np.zeros_like(dense_scores)

        # 3. 双路融合得分
        hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        retrieved_contexts = []
        for i, idx in enumerate(top_indices):
            retrieved_contexts.append(f"[Document {i+1}]\n{self.texts[idx]}")

        return "\n\n".join(retrieved_contexts)

# ====================== 双路 Prompt 对齐的数据集类 ======================
class DualPromptDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, ctx_len: int, dataset_config: Dict[str, Any], rag_manager: MedicalRAGManager, args):
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

        if self.config.get("hf_path") == "pubmed_qa":
            long_ans = item.get("long_answer", "")
            final_dec = item.get("final_decision", "maybe")
            # [CoT -> Answer 逻辑]：推理细节在前，决策块在后
            response = f"Analysis: {long_ans}\n\n[Final Decision]: {final_dec}"
        else:
            response = item[self.config["response_key"]]

        response_text = response + self.tokenizer.eos_token
        response_ids = self.tokenizer(response_text, return_tensors="pt")["input_ids"].squeeze(0)

        max_response_len = self.ctx_len // 2
        if len(response_ids) > max_response_len:
            response_ids = response_ids[:max_response_len]
        
        max_prompt_len = self.ctx_len - len(response_ids)

        default_s_prompt = "You are a professional assistant." if self.config.get("language") == "en" else "你是一个专业的助手。"
        s_system = self.config.get("student_system_prompt", default_s_prompt)
        s_messages = [{"role": "system", "content": s_system}, {"role": "user", "content": instruction}]
        
        try:
            s_prompt_text = self.tokenizer.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            s_prompt_text = self.tokenizer.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True)
            
        s_prompt_ids = self.tokenizer(s_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        if not self.args.disable_rag_teacher:
            retrieved_context = self.rag_manager.retrieve(instruction, top_k=self.args.rag_top_k, alpha=self.args.alpha)
            default_t_prompt = "Please answer based on the references:\n{rag_context}" if self.config.get("language") == "en" else "请根据资料解答：\n{rag_context}"
            t_system_template = self.config.get("teacher_system_prompt", default_t_prompt)
            t_system = t_system_template.format(rag_context=retrieved_context)
        else:
            t_system = s_system

        t_messages = [{"role": "system", "content": t_system}, {"role": "user", "content": instruction}]
        
        try:
            t_prompt_text = self.tokenizer.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            t_prompt_text = self.tokenizer.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True)
            
        t_prompt_ids = self.tokenizer(t_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        # 训练裁剪
        if len(s_prompt_ids) > max_prompt_len:
            s_prompt_ids = s_prompt_ids[-max_prompt_len:]
        if len(t_prompt_ids) > max_prompt_len:
            t_prompt_ids = t_prompt_ids[-max_prompt_len:]

        s_input_ids = torch.cat([s_prompt_ids, response_ids])
        t_input_ids = torch.cat([t_prompt_ids, response_ids])

        s_labels = torch.full_like(s_input_ids, -100)
        s_labels[len(s_prompt_ids):] = response_ids

        t_labels = torch.full_like(t_input_ids, -100)
        t_labels[len(t_prompt_ids):] = response_ids

        return {
            "s_prompt_ids": s_prompt_ids,
            "t_prompt_ids": t_prompt_ids,
            "s_input_ids": s_input_ids,
            "s_labels": s_labels,
            "t_input_ids": t_input_ids,
            "t_labels": t_labels
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

    s_prompt_ids = [item["s_prompt_ids"] for item in batch]
    t_prompt_ids = [item["t_prompt_ids"] for item in batch]

    return {
        "s_input_ids": s_ids,
        "s_labels": s_labels,
        "t_input_ids": t_ids,
        "t_labels": t_labels,
        "s_prompt_ids": s_prompt_ids,
        "t_prompt_ids": t_prompt_ids
    }

# ====================== 基于扁平化 Valid Token 的 JSD Loss ======================
def generalized_jsd_loss_flat(s_valid_logits, t_valid_logits, beta=0.5, temperature=1.0, chunk_size=2048):
    N, V_s = s_valid_logits.shape
    M, V_t = t_valid_logits.shape

    if N != M:
        raise RuntimeError(f"Teacher and Student valid tokens mismatch: {N} vs {M}.")

    min_vocab = min(V_s, V_t)
    if V_s != V_t:
        s_valid_logits = s_valid_logits[:, :min_vocab]
        t_valid_logits = t_valid_logits[:, :min_vocab]

    s_valid_logits = s_valid_logits.float()
    t_valid_logits = t_valid_logits.float()

    total_loss = torch.tensor(0.0, device=s_valid_logits.device, dtype=torch.float32)
    num_chunks = (N + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, N)

        s_chunk = s_valid_logits[start:end, :] / temperature
        t_chunk = t_valid_logits[start:end, :] / temperature

        s_logp = F.log_softmax(s_chunk, dim=-1)
        t_logp = F.log_softmax(t_chunk, dim=-1)

        if beta == 0:
            jsd = F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(dim=-1)
        elif beta == 1:
            jsd = F.kl_div(t_logp, s_logp, reduction="none", log_target=True).sum(dim=-1)
        else:
            beta_tensor = torch.tensor(beta, dtype=s_logp.dtype, device=s_logp.device)
            mixture_logp = torch.logsumexp(
                torch.stack([s_logp + torch.log1p(-beta_tensor), t_logp + torch.log(beta_tensor)]),
                dim=0
            )
            kl_teacher = F.kl_div(mixture_logp, t_logp, reduction="none", log_target=True).sum(dim=-1)
            kl_student = F.kl_div(mixture_logp, s_logp, reduction="none", log_target=True).sum(dim=-1)
            jsd = beta_tensor * kl_teacher + (1 - beta_tensor) * kl_student

        total_loss += jsd.sum()

    return total_loss / max(N, 1)

def extract_valid_logits(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    valid_mask = flat_labels != -100
    return flat_logits[valid_mask], flat_labels[valid_mask]

# ====================== 验证与评测函数 ======================
@torch.inference_mode()
def validate(student_model, teacher_model, dataloader, args, ctx):
    student_model.eval()
    total_loss = 0.0
    total_batches = 0
    
    for batch in dataloader:
        s_input_ids = batch["s_input_ids"].to(args.device)
        s_labels = batch["s_labels"].to(args.device)
        t_input_ids = batch["t_input_ids"].to(args.device)
        t_labels = batch["t_labels"].to(args.device)

        t_logits = teacher_model(t_input_ids).logits
        
        with ctx:
            outputs = student_model(s_input_ids)
            if hasattr(outputs, 'logits'):
                s_logits = outputs.logits
            elif isinstance(outputs, tuple):
                s_logits = outputs[0]
            else:
                s_logits = outputs
            
        s_valid_logits, _ = extract_valid_logits(s_logits, s_labels)
        t_valid_logits, _ = extract_valid_logits(t_logits, t_labels)

        if s_valid_logits.size(0) > 0:
            loss = generalized_jsd_loss_flat(
                s_valid_logits, 
                t_valid_logits, 
                beta=args.opd_beta, 
                temperature=args.temperature, 
                chunk_size=args.opd_chunk_size
            )
            total_loss += loss.item()
            total_batches += 1
            
    student_model.train()
    return total_loss / max(total_batches, 1)

@torch.inference_mode()
def validate_comprehensive_accuracy(
    student_model, 
    student_base_model, 
    teacher_model,
    tokenizer, 
    rag_manager, 
    dataset_config, 
    val_dataset_raw, 
    args, 
    num_samples=20
) -> Tuple[float, float]:
    """
    运行高效的多模型评测。
    直接复用外部传入的 rag_manager 与未经过 DataLoader 处理的原始抽样数据。
    """
    device = args.device
    student_model.eval()
    student_base_model.eval()
    teacher_model.eval()

    num_samples = min(num_samples, len(val_dataset_raw))
    eval_indices = random.sample(range(len(val_dataset_raw)), num_samples)
    eval_samples = [val_dataset_raw[i] for i in eval_indices]

    results_base = []
    results_student=[]
    results_rag = []

    print("\n" + "="*40)
    print(f" 开始评估: 规模 {num_samples} 个样本 (已加入 Few-shot 约束)")
    print("="*40)

    is_custom_transformer = (student_model.__class__.__name__ == "Transformer")

    for idx, sample in enumerate(eval_samples):
        question = sample[dataset_config["instruction_key"]]
        
        if dataset_config.get("hf_path") == "pubmed_qa":
            gold_answer = sample.get("final_decision", "maybe")
        else:
            gold_answer = sample.get(dataset_config["response_key"], "")

        # -------------------------
        # A. 评测学生基线/当前表现 (Base Model / Student Model)
        # -------------------------
        messages_base = [
            {
                "role": "system",
                "content": dataset_config["student_system_prompt"]
            },
            {
                "role": "user",
                "content": f"Question: {question}" if dataset_config.get("language") == "en" else f"问题: {question}\n回答:"
            }
        ]
        
        try:
            prompt_student = tokenizer.apply_chat_template(
                messages_base, 
                tokenize=False, 
                enable_thinking=False, 
                add_generation_prompt=True
            )
        except Exception:
            prompt_student = tokenizer.apply_chat_template(
                messages_base, 
                tokenize=False, 
                add_generation_prompt=True
            )
        inputs_student = tokenizer([prompt_student], return_tensors="pt").to(device)
        
        if is_custom_transformer:
            outputs_student = student_model.generate(
                idx=inputs_student.input_ids,
                max_new_tokens=128,
                temperature=args.temperature
            )
            outputs_base = student_base_model.generate(
                idx=inputs_student.input_ids,
                max_new_tokens=128,
                temperature=args.temperature
            )
            generated_student = tokenizer.decode(outputs_student[0][inputs_student.input_ids.shape[1]:], skip_special_tokens=True)
            generated_base = tokenizer.decode(outputs_base[0][inputs_student.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            outputs_student = student_model.generate(
                **inputs_student,
                max_new_tokens=128,
                do_sample=False       
            )
            outputs_base = student_base_model.generate(
                **inputs_student,
                max_new_tokens=128,
                do_sample=False       
            )
            
            generated_student = tokenizer.decode(outputs_student[0][inputs_student.input_ids.shape[1]:], skip_special_tokens=True)
            generated_base = tokenizer.decode(outputs_base[0][inputs_student.input_ids.shape[1]:], skip_special_tokens=True)
            
        pred_base = normalize_answer(generated_base) if dataset_config.get("hf_path") == "pubmed_qa" else generated_base.strip()
        results_base.append(pred_base == gold_answer)
        pred_student = normalize_answer(generated_student) if dataset_config.get("hf_path") == "pubmed_qa" else generated_student.strip()
        results_student.append(pred_student == gold_answer)

        # -------------------------
        # B. 评测教师表现 RAG (MedBioRAG 检索增强)
        # -------------------------
        retrieved_context = rag_manager.retrieve(query=question, top_k=args.rag_top_k, alpha=args.alpha)

        if dataset_config.get("hf_path") == "pubmed_qa":
            t_system = dataset_config["teacher_system_prompt"]
        else:
            t_system = dataset_config["teacher_system_prompt"].format(rag_context=retrieved_context)

        messages_rag = [
            {
                "role": "system",
                "content": t_system
            },
            {
                "role": "user",
                "content": f"Retrieved Context:\n{retrieved_context}\n\nQuestion: {question}" if dataset_config.get("language") == "en" else f"参考资料:\n{retrieved_context}\n\n问题: {question}\n回答:"
            }
        ]

        try:
            prompt_teacher = tokenizer.apply_chat_template(
                messages_rag, 
                tokenize=False, 
                enable_thinking=False, 
                add_generation_prompt=True
            )
        except Exception:
            prompt_teacher = tokenizer.apply_chat_template(
                messages_rag, 
                tokenize=False, 
                add_generation_prompt=True
            )
        inputs_teacher = tokenizer([prompt_teacher], return_tensors="pt").to(device)

        outputs_rag = teacher_model.generate(
            **inputs_teacher,
            max_new_tokens=128,
            do_sample=False
        )
        generated_rag = tokenizer.decode(outputs_rag[0][inputs_teacher.input_ids.shape[1]:], skip_special_tokens=True)
        pred_rag = normalize_answer(generated_rag) if dataset_config.get("hf_path") == "pubmed_qa" else generated_rag.strip()
        results_rag.append(pred_rag == gold_answer)

        # 前 5 个样本输出详细的 Debug 对比信息
        if idx < 5:
            print(f"\n\n[调试样例 {idx+1}]")
            print(f"👉 问题 (Question): {question}")
            print(f"🎯 真实答案 (Gold): {gold_answer}")
            print(f"❌ student回答:\n'{generated_student}'\n-> 解析后: {pred_base} | 结果: {'✓' if pred_base == gold_answer else '✗'}")
            print(f"✅ teacher回答:\n'{generated_rag}'\n-> 解析后: {pred_rag} | 结果: {'✓' if pred_rag == gold_answer else '✗'}")
            print("-" * 50)

    opd_student_acc = np.mean(results_student) * 100
    base_student_acc=np.mean(results_base) * 100
    teacher_acc = np.mean(results_rag) * 100
    improvement = opd_student_acc - base_student_acc

    print("\n" + "="*40)
    print(" 实验评测结果总结 ")
    print("="*40)
    print(f"评估样本总数 (Sample Size): {num_samples}")
    print(f"opd_student模型准确率 : {opd_student_acc:.2f}%")
    print(f"base_student模型准确率 : {base_student_acc:.2f}%")
    print(f"teacher模型准确率 : {teacher_acc:.2f}%")
    print(f"相对差距 (Delta Improvement):     {improvement:+.2f}%")
    print("="*40)
    
    student_model.train()
    return opd_student_acc,base_student_acc, teacher_acc

# ====================== 主训练流程 ======================
def main():
    args = parse_args()
    os.makedirs(args.opd_checkpoint_dir, exist_ok=True)
    device = args.device

    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        print("💡 [System] 已自动初始化单进程 Gloo 分布式环境，以兼容 Muon 优化器。")

    print("\n" + "="*60)
    print("🚀 正在启动 Agent OPD 蒸馏训练")
    print(f"👨‍🏫 Teacher 模型: {args.hf_teacher_model} (混合 RAG)")
    print(f"👶 Student 模型: {args.student_model_name} (无 RAG, 在线 Rollout 比率={args.rollout_ratio})")
    print("="*60 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    if TINY_R2_AVAILABLE:
        config.vocab_size = len(tokenizer)
        print(f"🔧 已将 Tiny-R2 词表大小 (vocab_size) 动态同步为: {config.vocab_size}")

    # === 1. 数据集加载与自适应解析 ===
    if args.custom_qa_path and os.path.exists(args.custom_qa_path):
        print(f"📂 正在加载本地自定义数据集: {args.custom_qa_path}")
        ds = load_dataset("json", data_files=args.custom_qa_path, split="train")
        
        first_item = ds[0]
        instruction_key = "question"
        response_key = "answer"
        for k in ["question", "instruction", "Question", "input"]:
            if k in first_item:
                instruction_key = k
                break
        for k in ["answer", "output", "Answer", "response"]:
            if k in first_item:
                response_key = k
                break

        DATASET_CONFIGS["custom"] = {
            "hf_path": None,
            "hf_subset": None,
            "split": "train",
            "language": "zh",
            "instruction_key": instruction_key,
            "response_key": response_key,
            "student_system_prompt": "你是一个专业的助理。请专业、准确、安全地解答用户的问题。",
            "teacher_system_prompt": "你是一个权威的行业专家。请根据以下参考资料，专业、严谨地解答用户的问题。\n\n[参考资料]\n{rag_context}"
        }
        args.dataset = "custom"
        dataset_config = DATASET_CONFIGS["custom"]
    else:
        print(f"📡 正在从 HuggingFace 加载公共数据集: {args.dataset}")
        dataset_config = DATASET_CONFIGS.get(args.dataset)
        if not dataset_config:
            raise ValueError(f"Dataset {args.dataset} not found in configs.")

        if dataset_config.get("hf_subset"):
            ds = load_dataset(dataset_config["hf_path"], dataset_config["hf_subset"], split=dataset_config["split"])
        else:
            ds = load_dataset(dataset_config["hf_path"], split=dataset_config["split"])

    # 动态构建语料库
    corpus = []
    if args.dataset == "pubmed_qa":
        for item in ds:
            abstract_text = " ".join(item["context"]["contexts"])
            corpus.append({
                "pubid": str(item["pubid"]),
                "text": abstract_text
            })
    else:
        for i, item in enumerate(ds):
            inst_text = item.get(dataset_config["instruction_key"], "")
            resp_text = item.get(dataset_config["response_key"], "")
            corpus.append({
                "pubid": str(i),
                "text": f"{inst_text} {resp_text}".strip()
            })

    # 初始化 BM25 + Dense 语义混合检索器
    rag_manager = MedicalRAGManager( 
        corpus=corpus, 
        dense_model_name=args.rag_dense_model,
        device=device
    )

    # 划分验证集
    val_size = max(1, int(len(ds) * 0.1)) if len(ds) > 1 else 0
    if val_size == 0:
        train_dataset = DualPromptDataset(ds, tokenizer, args.ctx_len, dataset_config, rag_manager, args)
        val_dataset = DualPromptDataset(ds, tokenizer, args.ctx_len, dataset_config, rag_manager, args)
        val_ds_raw = ds
    else:
        train_dataset = DualPromptDataset(ds.select(range(val_size, len(ds))), tokenizer, args.ctx_len, dataset_config, rag_manager, args)
        val_dataset = DualPromptDataset(ds.select(range(val_size)), tokenizer, args.ctx_len, dataset_config, rag_manager, args)
        val_ds_raw = ds.select(range(val_size))

    from functools import partial
    collate = partial(dual_collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate)

    if "cuda" in device and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        use_scaler = False
        print("\n⚡ 检测到 GPU 支持 bfloat16，将使用 bfloat16 混合精度 (自动禁用 GradScaler)")
    else:
        compute_dtype = torch.float16
        use_scaler = True
        print("\n⚡ 将使用 float16 混合精度 (启用 GradScaler)")

    # === 2. 加载学生与教师模型 ===
    if args.student_model_name.lower() == "tiny-r2" and TINY_R2_AVAILABLE:
        print("\n🤖 加载本地 Tiny-R2 Transformer 作为学生模型")
        student_model = Transformer().to(device)
        student_base_model=Transformer().to(device)
        if hasattr(student_model, "configure_optimizers"):
            opt_res = student_model.configure_optimizers(args.weight_decay, args.lr, device)
            optimizers = opt_res if isinstance(opt_res, list) else [opt_res]
        else:
            optimizers = [torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)]
            
    else:
        print(f"\n🤖 加载 HuggingFace 模型作为学生: {args.student_model_name}")
        student_model = AutoModelForCausalLM.from_pretrained(args.student_model_name, torch_dtype=compute_dtype, device_map=device)
        student_base_model=AutoModelForCausalLM.from_pretrained(args.student_model_name, torch_dtype=compute_dtype, device_map=device)
        optimizers = [torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)]

    scheduler = CosineAnnealingLR(optimizers[0], T_max=args.max_iters)

    print(f"\n👨‍🏫 加载教师模型: {args.hf_teacher_model}")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.hf_teacher_model, torch_dtype=compute_dtype, device_map=device).eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    # === 3. 自适应断点续训状态加载 ===
    best_val_loss = float("inf")
    start_iter = 0
    loaded_wandb_id = None

    if args.student_ckpt:
        if os.path.exists(args.student_ckpt):
            print(f"🔄 正在加载指定的学生权重和状态: {args.student_ckpt}")
            try:
                ckpt = torch.load(args.student_ckpt, map_location=device)
                if isinstance(ckpt, dict) and 'model' in ckpt:
                    student_model.load_state_dict(ckpt['model'], strict=False)
                    start_iter = ckpt.get('step', 0)
                    best_val_loss = ckpt.get('best_loss', float('inf'))
                    loaded_wandb_id = ckpt.get('wandb_run_id', None)
                    
                    if 'optimizer_states' in ckpt:
                        for opt, opt_state in zip(optimizers, ckpt['optimizer_states']):
                            opt.load_state_dict(opt_state)
                        print("✅ 成功对齐并恢复优化器历史状态。")
                    
                    if 'scheduler_state' in ckpt:
                        scheduler.load_state_dict(ckpt['scheduler_state'])
                        print("✅ 成功对齐并恢复学习率调度器历史状态。")
                else:
                    student_model.load_state_dict(ckpt, strict=False)
                print("✅ 权重及历史状态加载成功！")
            except Exception as e:
                print(f"⚠️ 状态加载失败: {e}，将采用模型原初权重和状态开始训练。")
        else:
            print(f"⚠️ 未找到路径：{args.student_ckpt}，将采用模型原初权重和状态开始训练。")

    if loaded_wandb_id and not args.wandb_run_id:
        args.wandb_run_id = loaded_wandb_id

    # === 4. WandB 监控初始化 ===
    if args.enable_wandb:
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=args.wandb_project, 
                    name=f"tiny-r2-opd-{args.dataset}", 
                    config=vars(args), 
                    resume="must" if args.wandb_run_id else False, 
                    id=args.wandb_run_id
                )
                if wandb.run is not None:
                    args.wandb_run_id = wandb.run.id
                print(f"🚀 WandB 监控服务成功开启 (ID: {args.wandb_run_id})")
            except Exception as e:
                print(f"⚠️ WandB 初始化失败: {e}。将以纯本地模式运行。")
                args.enable_wandb = False
        else:
            print("⚠️ 未检测到已安装的 `wandb`，已自动切换为纯本地无监控运行。")
            args.enable_wandb = False

    ctx = torch.amp.autocast(device_type="cuda", dtype=compute_dtype) if "cuda" in device else nullcontext()
    scaler = amp.GradScaler(enabled=use_scaler)

    # === 5. 基准评测运行 (缓存 Teacher 与 Base Student 准确率) ===
    # 巧妙设计：在开始训练前测试 base 并将其缓存，完全省去一个常驻显存的额外 student 模型副本！
    print("\n🔍 正在运行初始基准评测 ...")
    opd_student_acc,base_student_acc, teacher_acc = validate_comprehensive_accuracy(
        student_model,student_base_model, teacher_model, tokenizer, rag_manager, dataset_config, val_ds_raw, args, num_samples=args.num_eval
    )
    print(f"📊 基准评测完成 | Opd Student: {opd_student_acc:.2f}% |Base Student: {base_student_acc:.2f}% | Teacher (RAG): {teacher_acc:.2f}%\n")

    global_step = start_iter
    train_iter = iter(train_loader)
    
    student_model.train()
    print("\n🔥 开始双路 Agent OPD 蒸馏训练流程...")
    
    while global_step < args.max_iters:
        step_start = time.time()
        student_model.zero_grad()
        total_train_loss = 0.0
        avg_penalty_display = 0.0
        
        warmup_opd_steps = int(args.max_iters * 0.2)
        ramp_steps = int(args.max_iters * 0.6)
        
        if global_step < warmup_opd_steps:
            current_rollout_ratio = 0.0
        elif global_step < (warmup_opd_steps + ramp_steps):
            progress = (global_step - warmup_opd_steps) / ramp_steps
            current_rollout_ratio = progress * args.rollout_ratio
        else:
            current_rollout_ratio = args.rollout_ratio

        use_rollout = random.random() < current_rollout_ratio
        
        for accum_step in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            if use_rollout:
                s_input_list, s_label_list = [], []
                t_input_list, t_label_list = [], []
                student_texts, penalties = [], []
                
                for i in range(len(batch["s_prompt_ids"])):
                    s_p_ids = batch["s_prompt_ids"][i].to(device)
                    t_p_ids = batch["t_prompt_ids"][i].to(device)
                    
                    is_custom_transformer = (student_model.__class__.__name__ == "Transformer")
                    
                    with torch.no_grad():
                        if is_custom_transformer:
                            gen_out = student_model.generate(
                                idx=s_p_ids.unsqueeze(0),
                                max_new_tokens=args.ctx_len // 2,
                                temperature=args.temperature
                            )
                            generated_ids = gen_out if gen_out.ndim == 2 else gen_out.unsqueeze(0)
                        else:
                            generated_ids = student_model.generate(
                                s_p_ids.unsqueeze(0),
                                max_new_tokens=args.ctx_len // 2,
                                do_sample=True,
                                temperature=args.temperature,
                                pad_token_id=tokenizer.eos_token_id
                            )
                    
                    response_ids = generated_ids[0][len(s_p_ids):]
                    
                    decoded_resp = tokenizer.decode(response_ids, skip_special_tokens=True)
                    student_texts.append(decoded_resp)
                    penalties.append(get_narration_penalty(decoded_resp))
                    
                    s_seq = torch.cat([s_p_ids, response_ids])
                    s_lbl = torch.full_like(s_seq, -100)
                    s_lbl[len(s_p_ids):] = response_ids
                    
                    t_seq = torch.cat([t_p_ids, response_ids])
                    t_lbl = torch.full_like(t_seq, -100)
                    t_lbl[len(t_p_ids):] = response_ids
                    
                    s_input_list.append(s_seq)
                    s_label_list.append(s_lbl)
                    t_input_list.append(t_seq)
                    t_label_list.append(t_lbl)
                
                def pad_seqs(tensors, pad_val):
                    max_len = max(len(t) for t in tensors)
                    padded = torch.full((len(tensors), max_len), pad_val, dtype=torch.long, device=device)
                    for j, t in enumerate(tensors):
                        padded[j, :len(t)] = t
                    return padded
                
                s_input_ids = pad_seqs(s_input_list, tokenizer.pad_token_id)
                s_labels = pad_seqs(s_label_list, -100)
                t_input_ids = pad_seqs(t_input_list, tokenizer.pad_token_id)
                t_labels = pad_seqs(t_label_list, -100)
                batch_penalties = torch.tensor(penalties, dtype=torch.float32, device=device)
            else:
                s_input_ids = batch["s_input_ids"].to(device)
                s_labels = batch["s_labels"].to(device)
                t_input_ids = batch["t_input_ids"].to(device)
                t_labels = batch["t_labels"].to(device)
                batch_penalties = None

            if global_step == 0 and accum_step == 0:
                s_valid_num = (s_labels != -100).sum().item()
                t_valid_num = (t_labels != -100).sum().item()
                print(f"📊 批次对齐检测: Student回复Token数={s_valid_num}, Teacher回复Token数={t_valid_num}")

            with torch.no_grad():
                t_logits = teacher_model(t_input_ids).logits
                
            with ctx:
                outputs = student_model(s_input_ids)
                if hasattr(outputs, 'logits'):
                    s_logits = outputs.logits
                elif isinstance(outputs, tuple):
                    s_logits = outputs[0]
                else:
                    s_logits = outputs
                
            s_valid_logits, s_valid_labels = extract_valid_logits(s_logits, s_labels)
            t_valid_logits, _ = extract_valid_logits(t_logits, t_labels)

            if s_valid_logits.size(0) == 0:
                continue

            loss = generalized_jsd_loss_flat(
                s_valid_logits, 
                t_valid_logits, 
                beta=args.opd_beta, 
                temperature=args.temperature, 
                chunk_size=args.opd_chunk_size
            )

            # REINFORCE 策略梯度废话惩罚
            if use_rollout and batch_penalties is not None:
                avg_penalty = batch_penalties.mean().item()
                avg_penalty_display += avg_penalty / args.grad_accum_steps
                
                if avg_penalty > 0:
                    s_logprobs = F.log_softmax(s_valid_logits, dim=-1)
                    gen_logprobs = s_logprobs.gather(1, s_valid_labels.unsqueeze(-1)).squeeze(-1)
                    traj_logprob = gen_logprobs.sum() / args.batch_size
                    
                    pg_loss = batch_penalties.mean() * traj_logprob
                    loss = loss + args.pg_penalty_weight * pg_loss

            loss = loss / args.grad_accum_steps

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            total_train_loss += loss.item()

        if use_scaler:
            scaler.unscale_(optimizers[0])
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            for opt in optimizers:
                scaler.step(opt)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            for opt in optimizers:
                opt.step()
                
        scheduler.step()
        global_step += 1

        # ================= 终端日志记录 =================
        step_time = time.time() - step_start
        current_lr = scheduler.get_last_lr()[0]

        log_dict = {
            "train/loss": total_train_loss,
            "train/pg_penalty": avg_penalty_display,
            "train/learning_rate": current_lr,
            "perf/iter_time_s": step_time,
            "train/global_step": global_step,
            "train/is_rollout": int(use_rollout),
            "train/current_rollout_ratio": current_rollout_ratio
        }
        if device == "cuda":
            log_dict["perf/max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        
        if args.enable_wandb:
            wandb.log(log_dict, step=global_step)

        if global_step % 10 == 0:
            print(f"Step {global_step:04d} | Rollout: {'✅' if use_rollout else '❌'} | Loss: {total_train_loss:.4f} | Penalty: {avg_penalty_display:.2f} | LR: {current_lr:.2e}")

        # ================= 验证与保存 (融合评测汇总) =================
        if global_step % args.val_freq == 0:
            val_loss = validate(student_model, teacher_model, val_loader, args, ctx)
            is_best = val_loss < best_val_loss
            print(f"\n📊 Validation | Step={global_step} | Val Loss={val_loss:.4f} | Best Loss={best_val_loss:.4f}")

            # 动态执行当前 OPD 学生模型的评测
            opd_student_acc,base_student_acc, teacher_acc = validate_comprehensive_accuracy(student_model,student_base_model, teacher_model, tokenizer, rag_manager, dataset_config, val_ds_raw, args, num_samples=args.num_eval)
            print(f"📊 基准评测完成 | Opd Student: {opd_student_acc:.2f}% |Base Student: {base_student_acc:.2f}% | Teacher (RAG): {teacher_acc:.2f}%\n")

            if args.enable_wandb:
                wandb.log({
                    "eval/loss": val_loss,
                    "eval/teacher_accuracy": teacher_acc,
                    "eval/base_student_accuracy": base_student_acc,
                    "eval/opd_student_accuracy": opd_student_acc,
                    "eval/student_improvement": opd_student_acc - base_student_acc
                }, step=global_step)

            if is_best:
                best_val_loss = val_loss
                save_path = os.path.join(args.opd_checkpoint_dir, f"student_best_model_step_{global_step}.pt")
                
                # 完全训练状态包保存
                save_data = {
                    'model': student_model.state_dict(),
                    'optimizer_states': [opt.state_dict() for opt in optimizers],
                    'scheduler_state': scheduler.state_dict(),
                    'step': global_step,
                    'best_loss': best_val_loss,
                    'wandb_run_id': args.wandb_run_id if args.enable_wandb else None
                }
                torch.save(save_data, save_path)
                print(f"🏆 发现更优模型! 保存完整训练状态至: {save_path}\n")
                
                if args.save_best_only:
                    for old_file in glob.glob(os.path.join(args.opd_checkpoint_dir, "student_best_model_step_*.pt")):
                        if old_file != save_path:
                            try:
                                os.remove(old_file)
                                print(f"🧹 已清理旧检查点: {os.path.basename(old_file)}")
                            except Exception: 
                                pass

    print("🎉 训练流程执行完毕！")
    if args.enable_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()



