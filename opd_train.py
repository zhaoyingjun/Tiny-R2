#!/usr/bin/env python3
"""
Tiny-R2 OPD 训练 优势加权表征在策蒸馏学术版, GRP-Advantage 优势调节、表征对比对齐与神经网络语义验证器，多数据集适配、防泄露 RAG 管理和全面性 OOD 评测机制。
"""
import os
import sys
import re
import json
import time
import random
import argparse
import glob
from typing import Optional, List, Dict, Any, Tuple
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.amp as amp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification
)

# RAG 混合检索与向量模型支持
try:
    from sentence_transformers import SentenceTransformer, util
    from rank_bm25 import BM25Okapi
    RAG_AVAILABLE = True
    print("✅ 成功导入 rank_bm25 与 sentence_transformers，RAG 检索已启用。")
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ 警告: 未安装 rank_bm25 或 sentence_transformers，将启用 Mock 向量检索。")

# WandB 支持
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Tiny-R2 核心自定义模型支持
try:
    import model
    from model import Transformer
    TINY_R2_AVAILABLE = True
    print("✅ 成功检测并导入 Tiny-R2 核心模型")
except ImportError:
    TINY_R2_AVAILABLE = False
    class MockConfig:
        vocab_size = 151936
    config = MockConfig()


# ====================== 1. 数据集配置注册表与工具函数 ======================
DATASET_CONFIGS = {
    "pubmed_qa": {
        "hf_path": "qiaojin/PubMedQA",
        "hf_subset": "pqa_labeled",
        "split": "train",
        "language": "en",
        "instruction_key": "question",
        "response_key": "final_decision",
        "is_mcq": False,
        "student_system_prompt": (
            "You are an expert clinical assistant. For the given question, "
            "first analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'."
        ),
        "student_rag_system_prompt": (
            "You are an expert clinical assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to answer the question based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"  
            "First, analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'."
        ),
        "teacher_system_prompt": (
            "You are an expert clinical assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to answer the question based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"  
            "First, analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'."
        )
    },
    "medqa": {
        "hf_path": "GBaker/MedQA-USMLE-4-options",
        "hf_subset": None,
        "split": "train",
        "language": "en",
        "instruction_key": "question",
        "response_key": "answer_idx",
        "is_mcq": True,
        "option_keys": ["A", "B", "C", "D"],
        "student_system_prompt": (
            "You are an expert clinical assistant. For the given question, "
            "first analyze and reason step-by-step in plain text. "
            "Then, provide your final conclusion at the very end using the exact format: "
            "'[Final Decision]: A', '[Final Decision]: B', '[Final Decision]: C', or '[Final Decision]: D'."
        ),
        "student_rag_system_prompt": (
            "You are an expert clinical assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to select the correct option based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"
            "First, analyze and reason step-by-step in plain text. "
            "Then, provide your final conclusion at the very end using the exact format: "
            "'[Final Decision]: A', '[Final Decision]: B', '[Final Decision]: C', or '[Final Decision]: D'."
        ),
        "teacher_system_prompt": (
            "You are an expert clinical assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to select the correct option based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"
            "First, analyze and reason step-by-step in plain text. "
            "Then, provide your final conclusion at the very end using the exact format: "
            "'[Final Decision]: A', '[Final Decision]: B', '[Final Decision]: C', or '[Final Decision]: D'."
        )
    },
    "custom": {
        "hf_path": "custom",
        "hf_subset": None,
        "split": "train",
        "language": "en",
        "instruction_key": "question",
        "response_key": "answer",
        "is_mcq": False,
        "student_system_prompt": (
            "You are an expert assistant. For the given question, "
            "first analyze and reason step-by-step in plain text. "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'."
        ),
        "student_rag_system_prompt": (
            "You are an expert assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to answer the question based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"  
            "First, analyze and reason step-by-step in plain text. "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'."
        ),
        "teacher_system_prompt": (
            "You are an expert assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to answer the question based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"  
            "First, analyze and reason step-by-step in plain text. "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'."
        )
    }
}

BAD_PATTERNS = [
    "the user is asking", "i need to explain", "let me explain", 
    "the question asks", "i will answer", "as an ai",
    "用户问", "我需要解释", "让我解释", "这个问题要求", "我会回答", "作为一个人工智能"
]

def get_narration_penalty(text: str) -> float:
    lower = text.lower()
    return sum(1.0 for p in BAD_PATTERNS if p in lower)

def build_mcq_instruction(item: Dict[str, Any], instruction: str, config: Dict[str, Any]) -> str:
    option_keys = config.get("option_keys", [])
    options = []
    for idx_opt, key in enumerate(option_keys):
        label = chr(ord('A') + idx_opt)
        if key in item:
            opt_text = item[key]
        else:
            opt_text = item.get("options", {}).get(label, "")
        options.append(f"{label}: {opt_text}")
    opt_str = "\n".join(options)
    return f"Question: {instruction}\nOptions:\n{opt_str}"

def convert_mcq_answer(raw_answer: Any) -> str:
    if isinstance(raw_answer, int):
        return chr(ord('A') + raw_answer - 1)
    return str(raw_answer).strip().upper()

def normalize_answer(raw_output: str, is_mcq: bool = False) -> str:
    cleaned_output = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL).strip()
    if '</think>' in raw_output:
        cleaned_output = raw_output.split('</think>')[-1].strip()
    elif '<think>' in raw_output:
        cleaned_output = raw_output.split('<think>')[-1].strip()

    cleaned_output_lower = cleaned_output.lower().strip()
    
    if is_mcq:
        match_tag = re.search(r'(?:final decision|final answer|decision|answer)\]?\s*:\s*\*?\(?\b([a-d])\b\)?\*?', cleaned_output_lower)
        if match_tag:
            return match_tag.group(1).upper()
        match_alt = re.search(r'(?:correct choice|correct option|answer is|choice is)\s*\*?\(?\b([a-d])\b\)?\*?', cleaned_output_lower)
        if match_alt:
            return match_alt.group(1).upper()
        all_matches = list(re.finditer(r'\b([a-d])\b', cleaned_output_lower))
        if all_matches:
            return all_matches[-1].group(1).upper()
        return "A"
        
    match_tag = re.search(r'(?:final decision|final answer|decision|answer)\]?\s*:\s*\*?\b(yes|no|maybe)\b\*?', cleaned_output_lower)
    if match_tag:
        return match_tag.group(1)
    match_alt = re.search(r'(?:correct answer is|answer is)\s*\*?\b(yes|no|maybe)\b\*?', cleaned_output_lower)
    if match_alt:
        return match_alt.group(1)
        
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


# ====================== 2. 增强型双通路防泄露 RAG 管理器 ======================
class CleanMedicalRAGManager:
    """
    兼顾语义检索与词法匹配的双通路防泄露检索器 [1]
    """
    def __init__(self, corpus: List[Dict[str, str]], dense_model_name: str = "BAAI/bge-small-en-v1.5", device: str = "cuda"):
        self.corpus = corpus
        self.texts = [doc["text"] for doc in corpus]
        self.doc_ids = [doc.get("pubid", str(i)) for i, doc in enumerate(corpus)]
        self.device = device

        if RAG_AVAILABLE and len(self.texts) > 0:
            print("[-] 正在构建 BM25 词法匹配索引...")
            tokenized_corpus = [text.lower().split() for text in self.texts]
            self.bm25 = BM25Okapi(tokenized_corpus)

            print(f"[-] 正在初始化密向量检索模型 ({dense_model_name})...")
            self.dense_model = SentenceTransformer(dense_model_name, device=device)
            self.corpus_embeddings = self.dense_model.encode(
                self.texts, show_progress_bar=False, convert_to_tensor=True
            )
        else:
            print("⚠️ 警告: RAG 依赖未完备，采用降级 Mock 检索机制。")

    def retrieve(self, query: str, top_k: int = 3, alpha: float = 0.4, retrieval_mode: str = "hybrid") -> Tuple[str, List[str]]:
        if not RAG_AVAILABLE or len(self.texts) == 0:
            fallback_k = min(top_k, len(self.texts))
            ret_texts = [f"[Document {i+1}]\n{self.texts[i]}" for i in range(fallback_k)]
            return "\n\n".join(ret_texts), self.doc_ids[:fallback_k]

        tokenized_query = query.lower().split()
        bm25_scores = np.array(self.bm25.get_scores(tokenized_query))
        if bm25_scores.max() > bm25_scores.min():
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
        else:
            bm25_scores = np.zeros_like(bm25_scores)

        query_embedding = self.dense_model.encode(query, convert_to_tensor=True)
        dense_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0].cpu().numpy()
        if dense_scores.max() > dense_scores.min():
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        else:
            dense_scores = np.zeros_like(dense_scores)

        if retrieval_mode == "bm25":
            hybrid_scores = bm25_scores
        elif retrieval_mode == "dense":
            hybrid_scores = dense_scores
        else:
            hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores

        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        
        retrieved_contexts = []
        retrieved_ids = []
        for i, idx in enumerate(top_indices):
            retrieved_contexts.append(f"[Document {i+1}]\n{self.texts[idx]}")
            retrieved_ids.append(self.doc_ids[idx])

        return "\n\n".join(retrieved_contexts), retrieved_ids

    def evaluate_retrieval(self, query: str, gold_doc_id: str, top_k: int = 3, alpha: float = 0.4, retrieval_mode: str = "hybrid") -> Tuple[float, float]:
        _, retrieved_ids = self.retrieve(query, top_k=top_k, alpha=alpha, retrieval_mode=retrieval_mode)
        hit = 0.0
        mrr = 0.0
        if gold_doc_id in retrieved_ids:
            hit = 1.0
            rank = retrieved_ids.index(gold_doc_id) + 1
            mrr = 1.0 / rank
        return hit, mrr


# ====================== 3. NLI 神经验证器与奖励模型 ======================
class NeuralVerifierRewardModel:
    """
    语义蕴含验证器 (Neural Entailment Verifier)
    通过预测 Entailment 与 Contradiction 的概率差计算置信奖励
    公式: Reward = P(Entailment) - P(Contradiction)
    """
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: str = "cuda", cache_dir: Optional[str] = None):
        self.device = device
        try:
            print(f"[-] 正在初始化语义蕴含验证器 ({model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
            self.has_model = True
        except Exception as e:
            print(f"⚠️ 语义蕴含验证器加载失败 ({e})，降级为 Token 词汇重叠匹配。")
            self.has_model = False

    def compute_reward(self, context: str, response: str) -> float:
        if not self.has_model or not context or not response:
            ctx_words = set(re.findall(r'\b\w{3,}\b', context.lower()))
            resp_words = set(re.findall(r'\b\w{3,}\b', response.lower()))
            if not resp_words:
                return 0.0
            overlap = len(ctx_words.intersection(resp_words)) / len(resp_words)
            return float(overlap)

        inputs = self.tokenizer(
            context, 
            response, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)  # [0: contradiction, 1: entailment, 2: neutral]
            
        reward = probs[0, 1].item() - probs[0, 0].item()
        return reward


# ====================== 4. 表征流形与对比对齐引擎 ======================
class RepresentationAlignmentEngine(nn.Module):
    """
    表征层对齐流形映射模块 (Hidden Projection & Contrastive Alignment)
    在不改变学生模型基础维度的前提下，将其映射映射至教师模型特征空间进行细粒度对齐 [1]
    """
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.proj_hidden = nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else nn.Identity()

    def forward(self, s_hidden: torch.Tensor, t_hidden: torch.Tensor) -> torch.Tensor:
        """
        隐藏状态流形对齐
        """
        proj_s_hidden = self.proj_hidden(s_hidden.float())
        return F.mse_loss(proj_s_hidden, t_hidden.float(), reduction="mean")

    def compute_contrastive_loss(self, pos_hidden: torch.Tensor, neg_hidden: torch.Tensor, target_hidden: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
        """
        检索条件对比损失 (Retrieval-Conditioned Contrastive Loss)
        约束学生在正负外部知识干扰下的表征聚类分布
        """
        pos_proj = self.proj_hidden(pos_hidden.float())
        neg_proj = self.proj_hidden(neg_hidden.float())

        # 池化时间步
        pos_pooled = pos_proj.mean(dim=1)
        neg_pooled = neg_proj.mean(dim=1)
        target_pooled = target_hidden.mean(dim=1)

        sim_pos = F.cosine_similarity(pos_pooled, target_pooled, dim=-1) / temp
        sim_neg = F.cosine_similarity(neg_pooled, target_pooled, dim=-1) / temp

        logits = torch.stack([sim_pos, sim_neg], dim=-1)  # [B, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


# ====================== 5. 扩展型双通路正负对齐 Dataset ======================
class DualPromptDataset(Dataset):
    """
    支持对比检索增强（构建正向/负向对照条件）的双通路数据装载器
    """
    def __init__(self, hf_dataset, tokenizer, ctx_len: int, dataset_config: Dict[str, Any], rag_manager: CleanMedicalRAGManager, args):
        self.tokenizer = tokenizer
        self.ctx_len = ctx_len
        self.dataset = hf_dataset
        self.config = dataset_config
        self.rag_manager = rag_manager
        self.args = args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        instruction = item[self.config["instruction_key"]]

        # 1. 提取回答与标准化响应
        if self.args.dataset == "pubmed_qa" or "pubmed_qa" in str(self.config.get("hf_path", "")).lower():
            long_ans = item.get("long_answer", "")
            final_dec = item.get("final_decision", "maybe")
            response = f"Analysis: {long_ans}\n\n[Final Decision]: {final_dec}"
        elif self.config.get("is_mcq", False):
            instruction = build_mcq_instruction(item, instruction, self.config)
            gold_raw = item[self.config["response_key"]]
            gold_choice = convert_mcq_answer(gold_raw)
            response = f"Analysis: Guided by biomedical reasoning...\n\n[Final Decision]: {gold_choice}"
        else:
            response = item.get(self.config["response_key"], "")
            if "[Final Decision]" not in response:
                response = f"Analysis: Processed customized input.\n\n[Final Decision]: {response}"

        response_text = response + self.tokenizer.eos_token
        response_ids = self.tokenizer(response_text, return_tensors="pt")["input_ids"].squeeze(0)

        max_response_len = self.ctx_len // 2
        if len(response_ids) > max_response_len:
            response_ids = response_ids[:max_response_len]
        
        max_prompt_len = self.ctx_len - len(response_ids)

        # 2. 检索双通路背景 (正向匹配 / 负向匹配)
        need_rag = (not self.args.disable_rag_teacher) or self.args.student_use_rag
        if need_rag and self.args.ablation_mode != "vanilla_sft":
            pos_context, _ = self.rag_manager.retrieve(
                instruction, top_k=self.args.rag_top_k, alpha=self.args.alpha, retrieval_mode=self.args.retrieval_mode
            )
            # 随机选取不相干问题的 RAG 结果作为对照负样本
            rand_idx = (idx + random.randint(1, len(self.dataset) - 1)) % len(self.dataset)
            rand_item = self.dataset[rand_idx]
            rand_instruction = rand_item[self.config["instruction_key"]]
            neg_context, _ = self.rag_manager.retrieve(
                rand_instruction, top_k=self.args.rag_top_k, alpha=self.args.alpha, retrieval_mode=self.args.retrieval_mode
            )
        else:
            pos_context, neg_context = "", ""

        # 3. 组装正向学生端
        if self.args.student_use_rag and pos_context:
            s_system_template = self.config.get("student_rag_system_prompt", "Please answer based on reference:\n{rag_context}")
            s_system = s_system_template.format(rag_context=pos_context)
            s_prompt_input = f"Retrieved Context:\n{pos_context}\n\nQuestion: {instruction}"
        else:
            s_system = self.config.get("student_system_prompt", "You are a professional assistant.")
            s_prompt_input = f"Question: {instruction}"

        s_messages = [{"role": "system", "content": s_system}, {"role": "user", "content": s_prompt_input}]
        try:
            s_prompt_text = self.tokenizer.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            s_prompt_text = self.tokenizer.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True)
        s_prompt_ids = self.tokenizer(s_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        # 4. 组装负向学生端（用于表征对比学习）
        if self.args.student_use_rag and neg_context:
            s_neg_system = s_system_template.format(rag_context=neg_context)
            s_neg_prompt_input = f"Retrieved Context:\n{neg_context}\n\nQuestion: {instruction}"
        else:
            s_neg_system = s_system
            s_neg_prompt_input = f"Question: {instruction}"

        s_neg_messages = [{"role": "system", "content": s_neg_system}, {"role": "user", "content": s_neg_prompt_input}]
        try:
            s_neg_prompt_text = self.tokenizer.apply_chat_template(s_neg_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            s_neg_prompt_text = self.tokenizer.apply_chat_template(s_neg_messages, tokenize=False, add_generation_prompt=True)
        s_neg_prompt_ids = self.tokenizer(s_neg_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        # 5. 组装教师端
        if not self.args.disable_rag_teacher and self.args.ablation_mode != "vanilla_sft" and pos_context:
            t_system_template = self.config.get("teacher_system_prompt", "Please answer based on reference:\n{rag_context}")
            t_system = t_system_template.format(rag_context=pos_context)
            t_prompt_input = f"Retrieved Context:\n{pos_context}\n\nQuestion: {instruction}"
        else:
            t_system = self.config.get("student_system_prompt", "You are a professional assistant.")
            t_prompt_input = f"Question: {instruction}"

        t_messages = [{"role": "system", "content": t_system}, {"role": "user", "content": t_prompt_input}]
        try:
            t_prompt_text = self.tokenizer.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            t_prompt_text = self.tokenizer.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True)
        t_prompt_ids = self.tokenizer(t_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        # 序列截断
        if len(s_prompt_ids) > max_prompt_len:
            s_prompt_ids = s_prompt_ids[-max_prompt_len:]
        if len(s_neg_prompt_ids) > max_prompt_len:
            s_neg_prompt_ids = s_neg_prompt_ids[-max_prompt_len:]
        if len(t_prompt_ids) > max_prompt_len:
            t_prompt_ids = t_prompt_ids[-max_prompt_len:]

        # 拼接 Input 和 Label 映射
        s_input_ids = torch.cat([s_prompt_ids, response_ids])
        s_neg_input_ids = torch.cat([s_neg_prompt_ids, response_ids])
        t_input_ids = torch.cat([t_prompt_ids, response_ids])

        s_labels = torch.full_like(s_input_ids, -100)
        s_labels[len(s_prompt_ids):] = response_ids

        s_neg_labels = torch.full_like(s_neg_input_ids, -100)
        s_neg_labels[len(s_neg_prompt_ids):] = response_ids

        t_labels = torch.full_like(t_input_ids, -100)
        t_labels[len(t_prompt_ids):] = response_ids

        return {
            "instruction": instruction,
            "context": pos_context,
            "s_prompt_ids": s_prompt_ids,
            "s_neg_prompt_ids": s_neg_prompt_ids,
            "t_prompt_ids": t_prompt_ids,
            "s_input_ids": s_input_ids,
            "s_labels": s_labels,
            "s_neg_input_ids": s_neg_input_ids,
            "s_neg_labels": s_neg_labels,
            "t_input_ids": t_input_ids,
            "t_labels": t_labels
        }


def dual_collate_fn(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
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
    s_neg_ids, s_neg_labels = pad_tensors("s_neg_input_ids", "s_neg_labels")
    t_ids, t_labels = pad_tensors("t_input_ids", "t_labels")

    return {
        "s_input_ids": s_ids,
        "s_labels": s_labels,
        "s_neg_input_ids": s_neg_ids,
        "s_neg_labels": s_neg_labels,
        "t_input_ids": t_ids,
        "t_labels": t_labels,
        "s_prompt_ids": [item["s_prompt_ids"] for item in batch],
        "s_neg_prompt_ids": [item["s_neg_prompt_ids"] for item in batch],
        "t_prompt_ids": [item["t_prompt_ids"] for item in batch],
        "instructions": [item["instruction"] for item in batch],
        "contexts": [item["context"] for item in batch]
    }


# ====================== 6. JSD 分布对齐损失算子 ======================
def generalized_jsd_loss_flat(s_valid_logits, t_valid_logits, beta=0.5, temperature=1.0, chunk_size=2048):
    N, V_s = s_valid_logits.shape
    _, V_t = t_valid_logits.shape

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


# ====================== 7. 多维验证与评测 Benchmark 系统 ======================
@torch.inference_mode()
def validate(student_model, teacher_model, dataloader, args, ctx) -> float:
    student_model.eval()
    total_loss = 0.0
    total_batches = 0
    is_hf = hasattr(student_model, "config")
    
    for batch in dataloader:
        s_input_ids = batch["s_input_ids"].to(args.device)
        s_labels = batch["s_labels"].to(args.device)
        t_input_ids = batch["t_input_ids"].to(args.device)
        t_labels = batch["t_labels"].to(args.device)

        s_attn_mask = (s_input_ids != dataloader.dataset.tokenizer.pad_token_id).to(args.device) if is_hf else None

        if args.ablation_mode == "vanilla_sft":
            with ctx:
                outputs = student_model(s_input_ids, attention_mask=s_attn_mask) if is_hf else student_model(s_input_ids)
                s_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1), ignore_index=-100)
            total_loss += loss.item()
            total_batches += 1
        else:
            t_attn_mask = (t_input_ids != dataloader.dataset.tokenizer.pad_token_id).to(args.device) if is_hf else None
            t_logits = teacher_model(t_input_ids, attention_mask=t_attn_mask).logits if is_hf else teacher_model(t_input_ids).logits
            with ctx:
                outputs = student_model(s_input_ids, attention_mask=s_attn_mask) if is_hf else student_model(s_input_ids)
                s_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
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
    num_samples=20,
    ood_dataset_raw=None
) -> Tuple[float, float, float, float, float, float]:
    device = args.device
    student_model.eval()
    student_base_model.eval()
    if teacher_model:
        teacher_model.eval()

    num_samples = min(num_samples, len(val_dataset_raw))
    eval_indices = random.sample(range(len(val_dataset_raw)), num_samples)
    eval_samples = [val_dataset_raw[i] for i in eval_indices]

    results_base = []
    results_student = []
    results_rag = []
    
    retrieval_hits = []
    retrieval_mrrs = []

    print("\n" + "="*50)
    print(f" 开始在线全面泛化评测 (规模: {num_samples} 样本, 学生端 RAG: {'启用' if args.student_use_rag else '禁用'})")
    print("="*50)

    is_custom_transformer = (student_model.__class__.__name__ == "Transformer")
    is_pubmed_qa = (args.dataset == "pubmed_qa" or "pubmed_qa" in str(dataset_config.get("hf_path", "")).lower())
    is_mcq = dataset_config.get("is_mcq", False)

    for idx, sample in enumerate(eval_samples):
        question = sample[dataset_config["instruction_key"]]
        
        if is_mcq:
            question = build_mcq_instruction(sample, question, dataset_config)
        
        if is_pubmed_qa:
            gold_answer = sample.get("final_decision", "maybe")
            gold_doc_id = str(sample.get("pubid", ""))
        else:
            gold_raw = sample.get(dataset_config["response_key"], "")
            gold_answer = convert_mcq_answer(gold_raw) if is_mcq else gold_raw
            gold_doc_id = str(eval_indices[idx])

        # 1. 评估 RAG 指标
        if (not args.disable_rag_teacher or args.student_use_rag):
            hit, mrr = rag_manager.evaluate_retrieval(question, gold_doc_id, top_k=args.rag_top_k, alpha=args.alpha, retrieval_mode=args.retrieval_mode)
            retrieval_hits.append(hit)
            retrieval_mrrs.append(mrr)

        # 2. 检索并构建学生端评估 Prompt
        if args.student_use_rag:
            retrieved_context, _ = rag_manager.retrieve(query=question, top_k=args.rag_top_k, alpha=args.alpha, retrieval_mode=args.retrieval_mode)
            s_system_template = dataset_config.get("student_rag_system_prompt", "Please answer based on reference:\n{rag_context}")
            s_system = s_system_template.format(rag_context=retrieved_context)
            s_prompt_input = f"Retrieved Context:\n{retrieved_context}\n\nQuestion: {question}"
        else:
            s_system = dataset_config["student_system_prompt"]
            s_prompt_input = f"Question: {question}"

        messages_base = [{"role": "system", "content": s_system}, {"role": "user", "content": s_prompt_input}]
        try:
            prompt_student = tokenizer.apply_chat_template(messages_base, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        except Exception:
            prompt_student = tokenizer.apply_chat_template(messages_base, tokenize=False, add_generation_prompt=True)
            
        inputs_student = tokenizer([prompt_student], return_tensors="pt").to(device)
        
        # 生成
        if is_custom_transformer:
            outputs_student = student_model.generate(idx=inputs_student.input_ids, max_new_tokens=128, temperature=args.temperature)
            outputs_base = student_base_model.generate(idx=inputs_student.input_ids, max_new_tokens=128, temperature=args.temperature)
            generated_student = tokenizer.decode(outputs_student[0][inputs_student.input_ids.shape[1]:], skip_special_tokens=True)
            generated_base = tokenizer.decode(outputs_base[0][inputs_student.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            outputs_student = student_model.generate(**inputs_student, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            outputs_base = student_base_model.generate(**inputs_student, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated_student = tokenizer.decode(outputs_student[0][inputs_student.input_ids.shape[1]:], skip_special_tokens=True)
            generated_base = tokenizer.decode(outputs_base[0][inputs_student.input_ids.shape[1]:], skip_special_tokens=True)
            
        pred_base = normalize_answer(generated_base, is_mcq=is_mcq)
        results_base.append(pred_base == gold_answer)
        pred_student = normalize_answer(generated_student, is_mcq=is_mcq)
        results_student.append(pred_student == gold_answer)

        # 3. 评测 RAG 教师模型的准确率
        if teacher_model is not None:
            retrieved_context_t, _ = rag_manager.retrieve(query=question, top_k=args.rag_top_k, alpha=args.alpha, retrieval_mode=args.retrieval_mode)
            t_system = dataset_config["teacher_system_prompt"].format(rag_context=retrieved_context_t)
            messages_rag = [{"role": "system", "content": t_system}, {"role": "user", "content": f"Retrieved Context:\n{retrieved_context_t}\n\nQuestion: {question}"}]
            try:
                prompt_teacher = tokenizer.apply_chat_template(messages_rag, tokenize=False, enable_thinking=False, add_generation_prompt=True)
            except Exception:
                prompt_teacher = tokenizer.apply_chat_template(messages_rag, tokenize=False, add_generation_prompt=True)
            inputs_teacher = tokenizer([prompt_teacher], return_tensors="pt").to(device)
            outputs_rag = teacher_model.generate(**inputs_teacher, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            generated_rag = tokenizer.decode(outputs_rag[0][inputs_teacher.input_ids.shape[1]:], skip_special_tokens=True)
            pred_rag = normalize_answer(generated_rag, is_mcq=is_mcq)
            results_rag.append(pred_rag == gold_answer)
        else:
            results_rag.append(False)

        if idx < 2:
            print(f"\n[评估样例 {idx+1}]")
            print(f"👉 Q: {question[:120]}...")
            print(f"🎯 真实标注 (Gold): {gold_answer}")
            print(f"❌ Base 学生预测 : '{pred_base}' | {'✓' if pred_base == gold_answer else '✗'}")
            print(f"🚀 OPD 学生预测  : '{pred_student}' | {'✓' if pred_student == gold_answer else '✗'}")

    # 4. Out-of-Distribution (OOD) 泛化测试 (MedQA)
    ood_acc = 0.0
    if ood_dataset_raw is not None and len(ood_dataset_raw) > 0:
        print("\n" + "-"*15 + " 执行 OOD (MedQA) 泛化评测 " + "-"*15)
        ood_samples_num = min(num_samples, len(ood_dataset_raw))
        ood_indices = random.sample(range(len(ood_dataset_raw)), ood_samples_num)
        ood_results = []
        medqa_cfg = DATASET_CONFIGS["medqa"]

        for oidx in ood_indices:
            sample = ood_dataset_raw[oidx]
            q = sample["question"]
            
            option_keys = medqa_cfg.get("option_keys", [])
            options = [f"{chr(ord('A') + i)}: {sample[k] if k in sample else sample.get('options', {}).get(chr(ord('A') + i), '')}" for i, k in enumerate(option_keys)]
            opt_str = "\n".join(options)
            full_q = f"Question: {q}\nOptions:\n{opt_str}"
            gold_ans = sample["answer_idx"]

            if args.student_use_rag:
                ood_context, _ = rag_manager.retrieve(query=q, top_k=args.rag_top_k, alpha=args.alpha, retrieval_mode=args.retrieval_mode)
                s_system_template = medqa_cfg.get("student_rag_system_prompt", "Please answer based on reference:\n{rag_context}")
                s_system = s_system_template.format(rag_context=ood_context)
                user_content_ood = f"Retrieved Context:\n{ood_context}\n\nQuestion: {full_q}"
            else:
                s_system = medqa_cfg["student_system_prompt"]
                user_content_ood = full_q

            messages_ood = [{"role": "system", "content": s_system}, {"role": "user", "content": user_content_ood}]
            try:
                prompt_ood = tokenizer.apply_chat_template(messages_ood, tokenize=False, enable_thinking=False, add_generation_prompt=True)
            except Exception:
                prompt_ood = tokenizer.apply_chat_template(messages_ood, tokenize=False, add_generation_prompt=True)

            inputs_ood = tokenizer([prompt_ood], return_tensors="pt").to(device)
            with torch.no_grad():
                if is_custom_transformer:
                    outputs_ood = student_model.generate(idx=inputs_ood.input_ids, max_new_tokens=128, temperature=args.temperature)
                else:
                    outputs_ood = student_model.generate(**inputs_ood, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            gen_ood = tokenizer.decode(outputs_ood[0][inputs_ood.input_ids.shape[1]:], skip_special_tokens=True)
            pred_ood = normalize_answer(gen_ood, is_mcq=True)
            ood_results.append(pred_ood == gold_ans)

        ood_acc = np.mean(ood_results) * 100
        print(f"🎯 OOD (MedQA) 学生模型准确率: {ood_acc:.2f}%")

    opd_student_acc = np.mean(results_student) * 100
    base_student_acc = np.mean(results_base) * 100
    teacher_acc = np.mean(results_rag) * 100 if teacher_model else 0.0
    m_hit = np.mean(retrieval_hits) * 100 if retrieval_hits else 0.0
    m_mrr = np.mean(retrieval_mrrs) * 100 if retrieval_mrrs else 0.0

    print("\n" + "="*50)
    print(" 评测指标汇总统计 ")
    print("="*50)
    print(f"Base 初始学生模型精度 : {base_student_acc:.2f}%")
    print(f"OPD 蒸馏学生模型精度  : {opd_student_acc:.2f}%")
    if teacher_model:
        print(f"Teacher 检索增强模型精度: {teacher_acc:.2f}%")
    print(f"RAG Hit Rate @ {args.rag_top_k}  : {m_hit:.2f}%")
    print(f"RAG MRR                : {m_mrr:.2f}%")
    print("="*50 + "\n")
    
    student_model.train()
    return opd_student_acc, base_student_acc, teacher_acc, m_hit, m_mrr, ood_acc


# ====================== 8. 命令行参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Academic Representation-enhanced Advantage-weighted OPD Trainer")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--val_batch_size', type=int, default=2)
    parser.add_argument('--ctx_len', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--max_iters', type=int, default=100)
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--cache_dir', type=str, default="./hf_cache")
    
    # 损失控制与组训练超参数
    parser.add_argument('--contr_weight', type=float, default=0.1, help="表征对比损失比重")
    parser.add_argument('--opd_group_size', type=int, default=3, help="优势加权在策探索的组 Rollout 样本数 G")
    
    # 数据集与模型加载
    parser.add_argument("--dataset", type=str, default="pubmed_qa", choices=["pubmed_qa", "medqa", "custom"])
    parser.add_argument("--hf_teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3.5-9B")
    
    # 本地外部数据接口
    parser.add_argument("--custom_qa_path", type=str, default=None, help="本地 JSONL 格式 QA 数据路径")
    parser.add_argument("--rag_corpus_path", type=str, default=None, help="本地 RAG 背景段落路径")

    # 检索配置与开关
    parser.add_argument("--student_use_rag", action="store_true", default=False, help="学生端是否注入 RAG 背景输入")
    parser.add_argument("--disable_rag_teacher", action="store_true", default=False, help="是否禁用教师端 RAG 增强")
    parser.add_argument("--rag_top_k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4, help="检索双通路混合系数")
    parser.add_argument("--rag_dense_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--retrieval_mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"])
    
    # OPD 匹配配置
    parser.add_argument("--opd_chunk_size", type=int, default=512)
    parser.add_argument("--opd_beta", type=float, default=0.5, help="在策 JSD 对齐算子混合超参")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opd_checkpoint_dir", type=str, default="opd_v5_checkpoints")
    parser.add_argument("--val_freq", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 策略与采样
    parser.add_argument("--rollout_ratio", type=float, default=0.7, help="进入在策（On-policy）优势加权的训练步占比")
    parser.add_argument("--pg_penalty_weight", type=float, default=0.1, help="语言叙述性惩罚权重")
    parser.add_argument("--num_eval", type=int, default=10, help="在线 Benchmark 评测样本量")
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--save_best_only", action="store_true", default=True)
    parser.add_argument("--ablation_mode", type=str, default="none", choices=["none", "vanilla_sft", "offline_kd"])

    # 训练监控
    parser.add_argument('--enable_wandb', action="store_true", default=False)
    parser.add_argument('--wandb_project', type=str, default="PubMedQA-OPD-v5-Full")
    parser.add_argument('--wandb_run_id', type=str, default=None)
    
    return parser.parse_args()


# ====================== 9. 核心主训练逻辑 ======================
def main():
    args = parse_args()
    os.makedirs(args.opd_checkpoint_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    device = args.device

    # 初始化分布式环境以确保兼容
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29515'
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    print("\n" + "="*70)
    print("🚀 优势加权表征 OPD 训练流水线启动 (V5 学术融合版)")
    print(f"📦 缓存路径: {args.cache_dir}")
    print(f"👶 学生模型: {args.student_model_name} | 👨‍🏫 教师模型: {args.hf_teacher_model}")
    print(f"🛠️  消融实验模式: {args.ablation_mode}")
    print("="*70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    if TINY_R2_AVAILABLE:
        config.vocab_size = len(tokenizer)

    # 1. 载入核心数据集
    if args.custom_qa_path:
        print(f"📡 载入本地自定义 QA 数据集: {args.custom_qa_path}")
        custom_data = []
        with open(args.custom_qa_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    custom_data.append(json.loads(line))
        
        from datasets import Dataset
        ds = Dataset.from_list(custom_data)
        dataset_config = DATASET_CONFIGS["custom"]
        
        # 自适应中英文 Prompt 及多选题检测
        is_zh = False
        if len(custom_data) > 0:
            first_sample_str = str(custom_data[0])
            if re.search(r'[\u4e00-\u9fff]', first_sample_str):
                is_zh = True
        
        if is_zh:
            print("🇨🇳 检测到中文内容，适配中文引导 Prompt。")
            dataset_config["student_system_prompt"] = (
                "你是一个专业的临床医学助手。针对给定的问题，首先进行简明扼要的逐步分析与推理（控制在3句以内）。"
                "随后在最后使用以下格式给出你的最终结论: '[Final Decision]: yes'（是）、'[Final Decision]: no'（否）或 '[Final Decision]: maybe'（可能）。"
            )
            dataset_config["student_rag_system_prompt"] = (
                "你是一个专业的临床医学助手。你将获得‘检索到的背景信息’和‘问题’。请根据提供的背景信息回答问题。\n\n"
                "[Retrieved Context]\n{rag_context}\n\n"
                "首先进行简明扼要的逐步分析与推理（控制在3句以内），最后使用以下格式给出最终结论: '[Final Decision]: yes'、'[Final Decision]: no' 或 '[Final Decision]: maybe'。"
            )
            dataset_config["teacher_system_prompt"] = dataset_config["student_rag_system_prompt"]
        else:
            if len(custom_data) > 0:
                first_ans = str(custom_data[0].get("answer", "")).strip().upper()
                if len(first_ans) == 1 and first_ans in "ABCD":
                    dataset_config["is_mcq"] = True
                    print("📝 自动识别为单项选择题模式 (A/B/C/D)。")
    else:
        dataset_config = DATASET_CONFIGS.get(args.dataset)
        if not dataset_config:
            raise ValueError(f"无法匹配到预设的数据集: {args.dataset}")

        print(f"📡 载入 HuggingFace 预设数据集: {args.dataset}")
        from datasets import load_dataset
        if dataset_config.get("hf_subset"):
            ds = load_dataset(dataset_config["hf_path"], dataset_config["hf_subset"], split=dataset_config["split"], cache_dir=args.cache_dir)
        else:
            ds = load_dataset(dataset_config["hf_path"], split=dataset_config["split"], cache_dir=args.cache_dir)

    val_size = max(1, int(len(ds) * 0.1)) if len(ds) > 1 else 0
    if val_size == 0:
        train_ds = ds
        val_ds_raw = ds
    else:
        train_ds = ds.select(range(val_size, len(ds)))
        val_ds_raw = ds.select(range(val_size))

    # 2. 构造防泄露 RAG 语料库
    print("🔒 正在处理防泄露 RAG 检索库构建...")
    corpus = []
    if args.rag_corpus_path:
        print(f"🔒 正在从自定义文件载入: {args.rag_corpus_path}")
        if args.rag_corpus_path.endswith('.jsonl'):
            with open(args.rag_corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        data = json.loads(line)
                        text_content = data.get("text", data.get("content", str(data)))
                        corpus.append({"pubid": data.get("pubid", str(i)), "text": text_content})
        else:
            with open(args.rag_corpus_path, 'r', encoding='utf-8') as f:
                content = f.read()
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                for i, para in enumerate(paragraphs):
                    corpus.append({"pubid": f"doc_{i}", "text": para})
    elif args.dataset == "pubmed_qa":
        for item in ds:
            abstract_text = " ".join(item["context"]["contexts"])
            corpus.append({"pubid": str(item["pubid"]), "text": abstract_text})
    else:
        for i, item in enumerate(ds):
            inst_text = item.get(dataset_config["instruction_key"], "")
            corpus.append({"pubid": str(i), "text": inst_text.strip()})
    print(f"[-] 检索库处理完成，共载入 {len(corpus)} 条参考文档。")

    rag_manager = CleanMedicalRAGManager(
        corpus=corpus, 
        dense_model_name=args.rag_dense_model,
        device=device
    )

    train_dataset = DualPromptDataset(train_ds, tokenizer, args.ctx_len, dataset_config, rag_manager, args)
    val_dataset = DualPromptDataset(val_ds_raw, tokenizer, args.ctx_len, dataset_config, rag_manager, args)

    try:
        print("📡 加载 OOD 评测基准数据集 (MedQA-USMLE)...")
        ood_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", cache_dir=args.cache_dir)
    except Exception:
        print("⚠️ 提示: 未能自动装载 MedQA OOD 数据，将略过外部泛化性能评测。")
        ood_ds = None

    from functools import partial
    collate = partial(dual_collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate)

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # 3. 初始化 NLI 验证模型
    verifier = NeuralVerifierRewardModel(cache_dir=args.cache_dir, device=args.device)

    # 4. 加载模型
    print("👶 正在初始化学生模型实例...")
    if args.student_model_name.lower() == "tiny-r2" and TINY_R2_AVAILABLE:
        student_model = Transformer().to(device)
        student_base_model = Transformer().to(device)
    else:
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_name, torch_dtype=compute_dtype, device_map=device, cache_dir=args.cache_dir,
            output_hidden_states=True
        )
        student_base_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_name, torch_dtype=compute_dtype, device_map=device, cache_dir=args.cache_dir,
            output_hidden_states=True
        )

    is_hf = hasattr(student_model, "config")

    if args.ablation_mode != "vanilla_sft":
        print(f"👨‍🏫 正在载入蒸馏教师模型: {args.hf_teacher_model}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.hf_teacher_model, torch_dtype=compute_dtype, device_map=device, cache_dir=args.cache_dir,
            output_hidden_states=True
        ).eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
    else:
        teacher_model = None

    # 5. 表征映射对齐模块实例化
    student_dim = getattr(student_model.config, "hidden_size", 1024)
    teacher_dim = getattr(teacher_model.config, "hidden_size", 4096) if teacher_model else student_dim
    alignment_engine = RepresentationAlignmentEngine(student_dim, teacher_dim).to(device)

    # 6. 配置优化器与调度器
    params_list = list(student_model.parameters()) + list(alignment_engine.parameters())
    optimizer = torch.optim.AdamW(params_list, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_iters)

    best_val_loss = float("inf")
    start_iter = 0

    # 加载 Checkpoint 断点
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        try:
            ckpt = torch.load(args.student_ckpt, map_location=device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                student_model.load_state_dict(ckpt['model'], strict=False)
                alignment_engine.load_state_dict(ckpt['align_engine'], strict=False)
                start_iter = ckpt.get('step', 0)
                best_val_loss = ckpt.get('best_loss', float('inf'))
                optimizer.load_state_dict(ckpt['optimizer_states'][0])
                scheduler.load_state_dict(ckpt['scheduler_states'][0])
            else:
                student_model.load_state_dict(ckpt, strict=False)
            print("✅ 成功加载既有断点权重。")
        except Exception as e:
            print(f"⚠️ 权重载入失败: {e}")

    if args.enable_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project, 
            name=f"tiny-r2-opd-v5-{args.dataset}-ablation-{args.ablation_mode}", 
            config=vars(args)
        )

    ctx = torch.amp.autocast(device_type="cuda", dtype=compute_dtype) if "cuda" in device else nullcontext()

    print("\n🔍 启动初始性能评测...")
    _ = validate_comprehensive_accuracy(
        student_model, student_base_model, teacher_model, 
        tokenizer, rag_manager, dataset_config, val_ds_raw, args, num_samples=args.num_eval, ood_dataset_raw=ood_ds
    )

    global_step = start_iter
    train_iter = iter(train_loader)
    student_model.train()
    
    print("\n🔥 开始 OPD 优势加权表征蒸馏在策训练迭代流程...")
    
    while global_step < args.max_iters:
        optimizer.zero_grad()
        step_losses = {"total_loss": 0.0, "distill_loss": 0.0, "align_loss": 0.0, "contr_loss": 0.0, "mean_reward": 0.0}
        
        # 依据训练轮次规划在策 Rollout 触发比例
        if args.ablation_mode == "vanilla_sft" or args.ablation_mode == "offline_kd":
            current_rollout_ratio = 0.0
        else:
            warmup_opd_steps = int(args.max_iters * 0.1)
            if global_step < warmup_opd_steps:
                current_rollout_ratio = 0.0
            else:
                current_rollout_ratio = args.rollout_ratio

        use_rollout = (random.random() < current_rollout_ratio) if current_rollout_ratio > 0 else False
        
        for accum_step in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            if use_rollout and teacher_model is not None:
                # --- 一、 On-Policy Rollout 优势加权蒸馏路径 ---
                loss_opd = torch.tensor(0.0, device=device)
                total_samples = 0
                
                for p_idx in range(len(batch["s_prompt_ids"])):
                    s_p_ids = batch["s_prompt_ids"][p_idx].to(device)
                    pos_context = batch["contexts"][p_idx]
                    p_len = len(s_p_ids)

                    # 1. 对当前 Prompt 自主采样 G 个 Rollout 生成轨迹
                    rollout_ids = []
                    prompt_tensor = s_p_ids.unsqueeze(0)
                    with torch.no_grad():
                        for _ in range(args.opd_group_size):
                            gen_out = student_model.generate(
                                prompt_tensor,
                                max_new_tokens=128,
                                do_sample=True,
                                temperature=args.temperature,
                                pad_token_id=tokenizer.pad_token_id,
                                eos_token_id=tokenizer.eos_token_id
                            )
                            rollout_ids.append(gen_out[0])

                    # 2. 蕴含奖励与语言规则负向惩罚
                    rewards = []
                    for g in range(args.opd_group_size):
                        resp_str = tokenizer.decode(rollout_ids[g][p_len:], skip_special_tokens=True)
                        nli_reward = verifier.compute_reward(pos_context, resp_str)
                        penalty = get_narration_penalty(resp_str) * args.pg_penalty_weight
                        rewards.append(nli_reward - penalty)

                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                    step_losses["mean_reward"] += rewards_tensor.mean().item() / (args.grad_accum_steps * len(batch["s_prompt_ids"]))

                    # 3. 组相对优势标准化 (GRP-Advantage)
                    mean_r = rewards_tensor.mean()
                    std_r = rewards_tensor.std() + 1e-8
                    advantages = (rewards_tensor - mean_r) / std_r
                    opd_weights = torch.clamp(1.0 + advantages, min=0.2, max=2.0)

                    # 4. 对齐填充 Rollout 序列进行批推理
                    max_len = max(len(x) for x in rollout_ids)
                    padded_rollout = torch.full((args.opd_group_size, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
                    attn_mask = torch.zeros((args.opd_group_size, max_len), dtype=torch.long, device=device)
                    for g in range(args.opd_group_size):
                        curr_len = len(rollout_ids[g])
                        padded_rollout[g, :curr_len] = rollout_ids[g]
                        attn_mask[g, :curr_len] = 1

                    with ctx:
                        with torch.no_grad():
                            t_outputs = teacher_model(padded_rollout, attention_mask=attn_mask, output_hidden_states=True)
                            t_logits = t_outputs.logits
                            t_hidden = t_outputs.hidden_states[-1]

                        s_outputs = student_model(padded_rollout, attention_mask=attn_mask, output_hidden_states=True)
                        s_logits = s_outputs.logits
                        s_hidden = s_outputs.hidden_states[-1]

                    # 5. 计算 Token 对齐 JSD 与隐藏状态 MSE 损失
                    for g in range(args.opd_group_size):
                        valid_start = p_len - 1
                        valid_end = attn_mask[g].sum().item() - 1
                        if valid_end <= valid_start:
                            continue

                        s_tok_logits = s_logits[g, valid_start:valid_end]
                        t_tok_logits = t_logits[g, valid_start:valid_end]

                        s_valid_logits, s_valid_labels = extract_valid_logits(
                            s_logits[g:g+1, :valid_end+1], 
                            padded_rollout[g:g+1, :valid_end+1].clone().fill_(-100)
                        )
                        # 为 response 区域覆盖赋 label
                        padded_lbl = torch.full((1, valid_end+1), -100, dtype=torch.long, device=device)
                        padded_lbl[0, valid_start+1:valid_end+1] = padded_rollout[0, valid_start+1:valid_end+1]
                        
                        s_v_logits, _ = extract_valid_logits(s_logits[g:g+1, :valid_end+1], padded_lbl)
                        t_v_logits, _ = extract_valid_logits(t_logits[g:g+1, :valid_end+1], padded_lbl)

                        if s_v_logits.size(0) == 0:
                            continue

                        jsd_token_loss = generalized_jsd_loss_flat(
                            s_v_logits, t_v_logits, beta=args.opd_beta, temperature=args.temperature, chunk_size=args.opd_chunk_size
                        )

                        # 表征对齐损失
                        s_tok_hidden = s_hidden[g, valid_start:valid_end]
                        t_tok_hidden = t_hidden[g, valid_start:valid_end]
                        hidden_align_loss = alignment_engine(s_tok_hidden, t_tok_hidden)

                        # 应用组 relative 优势权重调节
                        w = opd_weights[g]
                        weighted_loss = w * (jsd_token_loss + 0.2 * hidden_align_loss)
                        
                        loss_opd += weighted_loss
                        step_losses["distill_loss"] += jsd_token_loss.item() / (args.grad_accum_steps * args.opd_group_size * len(batch["s_prompt_ids"]))
                        step_losses["align_loss"] += hidden_align_loss.item() / (args.grad_accum_steps * args.opd_group_size * len(batch["s_prompt_ids"]))
                        total_samples += 1

                if total_samples > 0:
                    loss_opd = (loss_opd / total_samples) / args.grad_accum_steps
                    loss_opd.backward()
                    step_losses["total_loss"] += loss_opd.item()

            else:
                # --- 二、 Off-Policy 黄金参考流形对齐路径 ---
                s_input_ids = batch["s_input_ids"].to(device)
                s_labels = batch["s_labels"].to(device)
                s_neg_input_ids = batch["s_neg_input_ids"].to(device)
                s_neg_labels = batch["s_neg_labels"].to(device)
                t_input_ids = batch["t_input_ids"].to(device)
                t_labels = batch["t_labels"].to(device)

                s_attn_mask = (s_input_ids != tokenizer.pad_token_id).to(device) if is_hf else None

                if args.ablation_mode == "vanilla_sft":
                    # 基线监督微调模式
                    with ctx:
                        outputs = student_model(s_input_ids, attention_mask=s_attn_mask) if is_hf else student_model(s_input_ids)
                        s_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1), ignore_index=-100)
                    loss = loss / args.grad_accum_steps
                    loss.backward()
                    step_losses["total_loss"] += loss.item()
                else:
                    t_attn_mask = (t_input_ids != tokenizer.pad_token_id).to(device) if is_hf else None
                    with ctx:
                        with torch.no_grad():
                            t_outputs = teacher_model(t_input_ids, attention_mask=t_attn_mask)
                            t_logits = t_outputs.logits
                            t_hidden = t_outputs.hidden_states[-1]

                        s_outputs = student_model(s_input_ids, attention_mask=s_attn_mask)
                        s_logits = s_outputs.logits
                        s_hidden = s_outputs.hidden_states[-1]

                        s_neg_outputs = student_model(s_neg_input_ids)
                        s_neg_hidden = s_neg_outputs.hidden_states[-1]

                    s_valid_logits, _ = extract_valid_logits(s_logits, s_labels)
                    t_valid_logits, _ = extract_valid_logits(t_logits, t_labels)

                    if s_valid_logits.size(0) == 0:
                        continue

                    # JSD 分布对齐
                    jsd_loss = generalized_jsd_loss_flat(
                        s_valid_logits, t_valid_logits, beta=args.opd_beta, temperature=args.temperature, chunk_size=args.opd_chunk_size
                    )

                    # 黄金表征空间投影对齐
                    s_valid_mask = (s_labels != -100)
                    t_valid_mask = (t_labels != -100)
                    min_valid_len = min(s_valid_mask.sum(dim=-1).min().item(), t_valid_mask.sum(dim=-1).min().item())

                    # 切割出 valid 的序列以计算表征 loss
                    s_rep_list = [s_hidden[i, s_valid_mask[i]][:min_valid_len] for i in range(s_hidden.size(0))]
                    t_rep_list = [t_hidden[i, t_valid_mask[i]][:min_valid_len] for i in range(t_hidden.size(0))]
                    s_rep = torch.stack(s_rep_list)
                    t_rep = torch.stack(t_rep_list)

                    align_loss = alignment_engine(s_rep, t_rep)

                    # 检索对比对齐 (正向 context 与 负向对照 context 表征对比)
                    s_neg_rep_list = [s_neg_hidden[i, s_neg_labels[i] != -100][:min_valid_len] for i in range(s_neg_hidden.size(0))]
                    s_neg_rep = torch.stack(s_neg_rep_list)
                    loss_contr = alignment_engine.compute_contrastive_loss(s_rep, s_neg_rep, t_rep)

                    # 综合损失
                    total_step_loss = jsd_loss + 0.2 * align_loss + args.contr_weight * loss_contr
                    total_step_loss = (total_step_loss / args.grad_accum_steps)
                    total_step_loss.backward()

                    step_losses["total_loss"] += total_step_loss.item()
                    step_losses["distill_loss"] += jsd_loss.item() / args.grad_accum_steps
                    step_losses["align_loss"] += align_loss.item() / args.grad_accum_steps
                    step_losses["contr_loss"] += (loss_contr.item() * args.contr_weight) / args.grad_accum_steps

        # 梯度剪裁与参数更新
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        global_step += 1

        if global_step % 5 == 0:
            print(f"Step {global_step:03d} | Rollout: {'✅' if use_rollout else '❌'} | Loss: {step_losses['total_loss']:.4f} "
                  f"[Distill: {step_losses['distill_loss']:.4f}, Align: {step_losses['align_loss']:.4f}, "
                  f"Contr: {step_losses['contr_loss']:.4f}] | Reward: {step_losses['mean_reward']:.3f}")

        # 5. 定期评测与最佳模型保存
        if global_step % args.val_freq == 0:
            val_loss = validate(student_model, teacher_model, val_loader, args, ctx)
            is_best = val_loss < best_val_loss
            print(f"\n📊 [Validation] Step={global_step} | Val Loss={val_loss:.4f} | 历史最佳={best_val_loss:.4f}")

            opd_student_acc, base_student_acc, teacher_acc, hit, mrr, ood_acc = validate_comprehensive_accuracy(
                student_model, student_base_model, teacher_model, 
                tokenizer, rag_manager, dataset_config, val_ds_raw, args, num_samples=args.num_eval, ood_dataset_raw=ood_ds
            )

            if args.enable_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "val/loss": val_loss,
                    "val/base_acc": base_student_acc,
                    "val/opd_acc": opd_student_acc,
                    "val/teacher_acc": teacher_acc,
                    "val/rag_hit": hit,
                    "val/ood_acc": ood_acc
                }, step=global_step)

            if is_best:
                best_val_loss = val_loss
                save_path = os.path.join(args.opd_checkpoint_dir, f"student_opd_v5_best.pt")
                torch.save({
                    'model': student_model.state_dict(),
                    'align_engine': alignment_engine.state_dict(),
                    'optimizer_states': [optimizer.state_dict()],
                    'scheduler_states': [scheduler.state_dict()],
                    'step': global_step,
                    'best_loss': best_val_loss
                }, save_path)
                print(f"🏆 发现更优对齐流形! 模型状态已安全存盘: {save_path}\n")

    print("🎉 OPD v5 融合版本训练流程顺利执行完毕。")
    if args.enable_wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()
