#!/usr/bin/env python3
"""
Tiny-R2 OPD 训练 v5 - 优势加权表征在策蒸馏 (Teacher=9B+RAG, Student=0.8B)
学术级融合版：结合 GRP-Advantage 强化加权、神经验证器、特征谱对齐与企业级多任务/防泄露 RAG 框架
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

# Tiny-R2 核心配置
try:
    import config
except ImportError:
    class MockConfig:
        vocab_size = 151936
    config = MockConfig()

# WandB 支持
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# RAG 支持
try:
    from sentence_transformers import SentenceTransformer, util
    from rank_bm25 import BM25Okapi
    RAG_AVAILABLE = True
    print("✅ 成功导入 rank_bm25 与 sentence_transformers，混合 RAG 模块已启用。")
except ImportError:
    print("⚠️ 提示: 未安装 rank_bm25 或 sentence_transformers，将启用 Mock 向量检索。")
    RAG_AVAILABLE = False

try:
    import model
    from model import Transformer
    print("✅ 成功导入 Tiny-R2 核心模块")
    TINY_R2_AVAILABLE = True
except ImportError as e:
    print(f"ℹ️ 未检测到 Tiny-R2 模块，将强制使用 HuggingFace 模型。")
    TINY_R2_AVAILABLE = False


# ====================== 1. 任务数据集注册表 ======================
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
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'.\n\n"
            "Now answer the user's question following the same format."
        ),
        "student_rag_system_prompt": (
            "You are an expert clinical assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to answer the question based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"  
            "First, analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'.\n\n"
            "Now answer the user's question using the provided context."
        ),
        "teacher_system_prompt": (
            "You are an expert clinical assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to answer the question based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"  
            "First, analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'.\n\n"
            "Now answer the user's question using the provided context."
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
            "'[Final Decision]: A', '[Final Decision]: B', '[Final Decision]: C', or '[Final Decision]: D'.\n\n"
            "Now answer the user's question following the same format."
        ),
        "student_rag_system_prompt": (
            "You are an expert clinical assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to select the correct option based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"
            "First, analyze and reason step-by-step in plain text. "
            "Then, provide your final conclusion at the very end using the exact format: "
            "'[Final Decision]: A', '[Final Decision]: B', '[Final Decision]: C', or '[Final Decision]: D'.\n\n"
            "Now answer the user's question using the provided context."
        ),
        "teacher_system_prompt": (
            "You are an expert clinical assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to select the correct option based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"
            "First, analyze and reason step-by-step in plain text. "
            "Then, provide your final conclusion at the very end using the exact format: "
            "'[Final Decision]: A', '[Final Decision]: B', '[Final Decision]: C', or '[Final Decision]: D'.\n\n"
            "Now answer the user's question using the provided context."
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
            "first analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'.\n\n"
            "Now answer the user's question following the same format."
        ),
        "student_rag_system_prompt": (
            "You are an expert assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to answer the question based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"  
            "First, analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'.\n\n"
            "Now answer the user's question using the provided context."
        ),
        "teacher_system_prompt": (
            "You are an expert assistant. You will be provided with a 'Retrieved Context' and a 'Question'. "
            "Your task is to answer the question based on the provided Retrieved Context.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"  
            "First, analyze and reason step-by-step in plain text (keep it concise, under 3 sentences). "
            "Then, provide your final conclusion at the very end using the exact format: '[Final Decision]: yes', '[Final Decision]: no', or '[Final Decision]: maybe'.\n\n"
            "Now answer the user's question using the provided context."
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


# ====================== 2. MCQ 统一工具函数 ======================
def build_mcq_instruction(item: Dict[str, Any], instruction: str, dataset_cfg: Dict[str, Any]) -> str:
    option_keys = dataset_cfg.get("option_keys", [])
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
        match_tag = re.search(
            r'(?:final decision|final answer|decision|answer)\]?\s*:\s*\*?\(?\b([a-d])\b\)?\*?', 
            cleaned_output_lower
        )
        if match_tag:
            return match_tag.group(1).upper()
            
        match_alt = re.search(
            r'(?:correct choice|correct option|answer is|choice is)\s*\*?\(?\b([a-d])\b\)?\*?', 
            cleaned_output_lower
        )
        if match_alt:
            return match_alt.group(1).upper()
            
        all_matches = list(re.finditer(r'\b([a-d])\b', cleaned_output_lower))
        if all_matches:
            return all_matches[-1].group(1).upper()
        return "A"
        
    match_tag = re.search(
        r'(?:final decision|final answer|decision|answer)\]?\s*:\s*\*?\b(yes|no|maybe)\b\*?', 
        cleaned_output_lower
    )
    if match_tag:
        return match_tag.group(1)
        
    match_alt = re.search(
        r'(?:correct answer is|answer is)\s*\*?\b(yes|no|maybe)\b\*?', 
        cleaned_output_lower
    )
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


# ====================== 3. 双通路 RAG 检索器模块 ======================
class CleanMedicalRAGManager:
    """
    自适应双通路检索管理，具备防泄露评估功能
    """
    def __init__(self, corpus: List[Dict[str, str]], dense_model_name="BAAI/bge-small-zh", device="cuda"):
        self.corpus = corpus
        self.texts = [doc["text"] for doc in corpus]
        self.doc_ids = [doc.get("pubid", str(i)) for i, doc in enumerate(corpus)]
        self.device = device

        if len(self.texts) > 0 and RAG_AVAILABLE:
            print("[-] 正在构建自适应 BM25 词法索引...")
            tokenized_corpus = [text.lower().split() for text in self.texts]
            self.bm25 = BM25Okapi(tokenized_corpus)

            print(f"[-] 正在初始化向量检索模型 ({dense_model_name})...")
            self.dense_model = SentenceTransformer(dense_model_name, device=device)
            self.corpus_embeddings = self.dense_model.encode(
                self.texts, show_progress_bar=False, convert_to_tensor=True
            )
        else:
            print("⚠️ 警告: RAG 依赖未完备，切换至 Mock 检索流程。")

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
            hybrid_scores = alpha * bm25_scores + (1.0 - alpha) * dense_scores

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


# ====================== 4. NLI 神经验证器 ======================
class NeuralVerifierRewardModel:
    """
    语义蕴含验证器模型 (Neural Entailment Verifier Model)
    计算公式: Reward = P(Entailment) - P(Contradiction)
    """
    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small", device: str = "cuda", cache_dir: Optional[str] = None):
        self.device = device
        try:
            print(f"[-] 正在载入 NLI 验证模型 ({model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir).to(device).eval()
            self.has_model = True
        except Exception as e:
            print(f"⚠️ NLI验证模型载入失败 ({e})，降级为 Token 词汇重叠度计算。")
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
            probs = F.softmax(logits, dim=-1)  # 一般 [0: contradiction, 1: entailment, 2: neutral]
            
        reward = probs[0, 1].item() - probs[0, 0].item()  # Entailment - Contradiction
        return reward


# ====================== 5. 表征流形投影与对比对齐引擎 ======================
class RepresentationAlignmentEngine(nn.Module):
    """
    隐藏流形投影与多通道注意力特征谱对齐核心引擎
    """
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        # 处理隐藏维度映射，完成多尺寸模型流形对齐
        self.proj_hidden = nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else nn.Identity()

    def forward(self, s_hidden: torch.Tensor, t_hidden: torch.Tensor) -> torch.Tensor:
        """
        隐藏状态流形 MSE 损失
        """
        proj_s_hidden = self.proj_hidden(s_hidden.float())
        return F.mse_loss(proj_s_hidden, t_hidden.float(), reduction="mean")

    def compute_contrastive_loss(self, pos_hidden: torch.Tensor, neg_hidden: torch.Tensor, target_hidden: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
        """
        检索环境下的条件对比表征损失 (Retrieval-Conditioned Contrastive Loss)
        """
        pos_proj = self.proj_hidden(pos_hidden.float())
        neg_proj = self.proj_hidden(neg_hidden.float())

        pos_pooled = pos_proj.mean(dim=1)
        neg_pooled = neg_proj.mean(dim=1)
        target_pooled = target_hidden.mean(dim=1)

        sim_pos = F.cosine_similarity(pos_pooled, target_pooled, dim=-1) / temp
        sim_neg = F.cosine_similarity(neg_pooled, target_pooled, dim=-1) / temp

        logits = torch.stack([sim_pos, sim_neg], dim=-1)  # [B, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


# ====================== 6. 双路 Prompt 数据装载与 Collate ======================
class DualPromptDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, ctx_len: int, dataset_config: Dict[str, Any], rag_manager: CleanMedicalRAGManager, args):
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

        # 答案提取与标准化
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

        # 检索及构建负向环境
        need_rag = (not self.args.disable_rag_teacher) or self.args.student_use_rag
        if need_rag and self.args.ablation_mode != "vanilla_sft":
            retrieved_context, _ = self.rag_manager.retrieve(
                instruction, top_k=self.args.rag_top_k, alpha=self.args.alpha, retrieval_mode=self.args.retrieval_mode
            )
            # 随机检索不匹配的外部文本用于 Contrastive Loss 对其增强
            random_idx = (idx + random.randint(1, len(self.dataset)-1)) % len(self.dataset)
            neg_item = self.dataset[random_idx]
            neg_inst = neg_item[self.config["instruction_key"]]
            if self.config.get("is_mcq", False):
                neg_inst = build_mcq_instruction(neg_item, neg_inst, self.config)
            neg_context, _ = self.rag_manager.retrieve(
                neg_inst, top_k=self.args.rag_top_k, alpha=self.args.alpha, retrieval_mode=self.args.retrieval_mode
            )
        else:
            retrieved_context = ""
            neg_context = ""

        # 学生正向 Prompt
        if self.args.student_use_rag and retrieved_context:
            s_system_template = self.config.get("student_rag_system_prompt", "Please answer based on reference:\n{rag_context}")
            s_system = s_system_template.format(rag_context=retrieved_context)
            s_prompt_input = f"Retrieved Context:\n{retrieved_context}\n\nQuestion: {instruction}"
        else:
            s_system = self.config.get("student_system_prompt", "You are a professional assistant.")
            s_prompt_input = f"Question: {instruction}"

        s_messages = [{"role": "system", "content": s_system}, {"role": "user", "content": s_prompt_input}]
        try:
            s_prompt_text = self.tokenizer.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            s_prompt_text = self.tokenizer.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True)
        s_prompt_ids = self.tokenizer(s_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        # 学生负向 Prompt
        if self.args.student_use_rag and neg_context:
            s_neg_system = s_system_template.format(rag_context=neg_context)
            s_neg_prompt_input = f"Retrieved Context:\n{neg_context}\n\nQuestion: {instruction}"
        else:
            s_neg_system = self.config.get("student_system_prompt", "You are a professional assistant.")
            s_neg_prompt_input = f"Question: {instruction}"

        s_neg_messages = [{"role": "system", "content": s_neg_system}, {"role": "user", "content": s_neg_prompt_input}]
        try:
            s_neg_prompt_text = self.tokenizer.apply_chat_template(s_neg_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            s_neg_prompt_text = self.tokenizer.apply_chat_template(s_neg_messages, tokenize=False, add_generation_prompt=True)
        s_neg_prompt_ids = self.tokenizer(s_neg_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        # 教师端 Prompt
        if not self.args.disable_rag_teacher and self.args.ablation_mode != "vanilla_sft" and retrieved_context:
            t_system_template = self.config.get("teacher_system_prompt", "Please answer based on reference:\n{rag_context}")
            t_system = t_system_template.format(rag_context=retrieved_context)
            t_prompt_input = f"Retrieved Context:\n{retrieved_context}\n\nQuestion: {instruction}"
        else:
            t_system = self.config.get("student_system_prompt", "You are a professional assistant.")
            t_prompt_input = f"Question: {instruction}"

        t_messages = [{"role": "system", "content": t_system}, {"role": "user", "content": t_prompt_input}]
        try:
            t_prompt_text = self.tokenizer.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            t_prompt_text = self.tokenizer.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True)
        t_prompt_ids = self.tokenizer(t_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        # 截断处理
        if len(s_prompt_ids) > max_prompt_len:
            s_prompt_ids = s_prompt_ids[-max_prompt_len:]
        if len(s_neg_prompt_ids) > max_prompt_len:
            s_neg_prompt_ids = s_neg_prompt_ids[-max_prompt_len:]
        if len(t_prompt_ids) > max_prompt_len:
            t_prompt_ids = t_prompt_ids[-max_prompt_len:]

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
            "s_prompt_ids": s_prompt_ids,
            "s_neg_prompt_ids": s_neg_prompt_ids,
            "t_prompt_ids": t_prompt_ids,
            "s_input_ids": s_input_ids,
            "s_labels": s_labels,
            "s_neg_input_ids": s_neg_input_ids,
            "s_neg_labels": s_neg_labels,
            "t_input_ids": t_input_ids,
            "t_labels": t_labels,
            "instruction": instruction,
            "context": retrieved_context
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


# ====================== 7. 优势加权在策蒸馏 (GRP-Advantage Engine) ======================
class AdvantageWeightedOPDEngine:
    """
    在策对齐蒸馏引擎 (GRP-Advantage On-Policy Distillation Engine)
    """
    def __init__(
        self, 
        student_model, 
        teacher_model, 
        alignment_engine, 
        verifier, 
        tokenizer, 
        opd_beta: float = 0.5,
        temp: float = 1.0,
        pg_penalty_weight: float = 0.1
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.align_engine = alignment_engine
        self.verifier = verifier
        self.tokenizer = tokenizer
        self.opd_beta = opd_beta
        self.temp = temp
        self.pg_penalty_weight = pg_penalty_weight

    def compute_on_policy_loss(
        self, 
        prompts: List[torch.Tensor], 
        contexts: List[str], 
        group_size: int = 3
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = next(self.student.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        
        metrics = {"distill_loss": 0.0, "align_loss": 0.0, "mean_reward": 0.0, "mean_penalty": 0.0}
        total_samples = 0
        is_hf = hasattr(self.student, "config")

        for p_idx, prompt in enumerate(prompts):
            context = contexts[p_idx]
            p_len = len(prompt)

            # On-policy 采样
            rollout_ids = []
            prompt_tensor = prompt.unsqueeze(0).to(device)
            with torch.no_grad():
                for _ in range(group_size):
                    if not is_hf:
                        gen_out = self.student.generate(
                            idx=prompt_tensor,
                            max_new_tokens=128,
                            temperature=self.temp
                        )
                    else:
                        gen_out = self.student.generate(
                            prompt_tensor,
                            max_new_tokens=128,
                            do_sample=True,
                            temperature=0.9,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    rollout_ids.append(gen_out[0])

            # 计算奖励值与话术惩罚
            rewards = []
            penalties = []
            for g in range(group_size):
                response_str = self.tokenizer.decode(rollout_ids[g][p_len:], skip_special_tokens=True)
                nli_score = self.verifier.compute_reward(context, response_str)
                penalty = get_narration_penalty(response_str) * self.pg_penalty_weight
                
                rewards.append(nli_score)
                penalties.append(penalty)

            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
            penalties_tensor = torch.tensor(penalties, dtype=torch.float32, device=device)
            
            # GRP 净奖励融合
            net_rewards = rewards_tensor - penalties_tensor
            metrics["mean_reward"] += rewards_tensor.mean().item()
            metrics["mean_penalty"] += penalties_tensor.mean().item()

            # 组内相对优势标准化
            mean_r = net_rewards.mean()
            std_r = net_rewards.std() + 1e-8
            advantages = (net_rewards - mean_r) / std_r

            # 计算尺度乘数，防数值失控
            opd_weights = torch.clamp(1.0 + advantages, min=0.2, max=2.0)

            max_len = max(len(x) for x in rollout_ids)
            padded_rollout = torch.full((group_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long, device=device)
            attn_mask = torch.zeros((group_size, max_len), dtype=torch.long, device=device)
            for g in range(group_size):
                curr_len = len(rollout_ids[g])
                padded_rollout[g, :curr_len] = rollout_ids[g]
                attn_mask[g, :curr_len] = 1

            # 提取教师和学生的状态表达
            if self.teacher is not None:
                with torch.no_grad():
                    if is_hf:
                        t_outputs = self.teacher(padded_rollout, attention_mask=attn_mask, output_hidden_states=True)
                        t_logits = t_outputs.logits
                        t_hidden = t_outputs.hidden_states[-1] if hasattr(t_outputs, "hidden_states") else None
                    else:
                        t_outputs = self.teacher(padded_rollout)
                        t_logits = t_outputs.logits if hasattr(t_outputs, "logits") else t_outputs[0]
                        t_hidden = None
            else:
                t_logits, t_hidden = None, None

            if is_hf:
                s_outputs = self.student(padded_rollout, attention_mask=attn_mask, output_hidden_states=True)
                s_logits = s_outputs.logits
                s_hidden = s_outputs.hidden_states[-1] if hasattr(s_outputs, "hidden_states") else None
            else:
                s_outputs = self.student(padded_rollout)
                s_logits = s_outputs.logits if hasattr(s_outputs, "logits") else s_outputs[0]
                s_hidden = None

            # 密集计算优势加权损失
            for g in range(group_size):
                valid_start = p_len - 1
                valid_end = attn_mask[g].sum().item() - 1
                if valid_end <= valid_start:
                    continue

                s_tok_logits = s_logits[g, valid_start:valid_end].float() / self.temp
                
                if t_logits is not None:
                    t_tok_logits = t_logits[g, valid_start:valid_end].float() / self.temp
                    s_logp = F.log_softmax(s_tok_logits, dim=-1)
                    t_logp = F.log_softmax(t_tok_logits, dim=-1)

                    beta_tensor = torch.tensor(self.opd_beta, dtype=s_logp.dtype, device=device)
                    mixture_logp = torch.logsumexp(
                        torch.stack([s_logp + torch.log1p(-beta_tensor), t_logp + torch.log(beta_tensor)]), dim=0
                    )
                    kl_teacher = F.kl_div(mixture_logp, t_logp, reduction="batchmean", log_target=True)
                    kl_student = F.kl_div(mixture_logp, s_logp, reduction="batchmean", log_target=True)
                    jsd_token_loss = beta_tensor * kl_teacher + (1.0 - beta_tensor) * kl_student
                else:
                    target_tokens = padded_rollout[g, valid_start+1:valid_end+1]
                    jsd_token_loss = F.cross_entropy(s_tok_logits, target_tokens, ignore_index=-100)

                # 隐藏层流形表征级对齐
                hidden_align_loss = torch.tensor(0.0, device=device)
                if s_hidden is not None and t_hidden is not None:
                    s_tok_hidden = s_hidden[g, valid_start:valid_end]
                    t_tok_hidden = t_hidden[g, valid_start:valid_end]
                    hidden_align_loss = self.align_engine(s_tok_hidden, t_tok_hidden)

                # 应用优势权重实施在策调节
                w = opd_weights[g]
                weighted_loss = w * (jsd_token_loss + 0.2 * hidden_align_loss)
                
                total_loss += weighted_loss
                metrics["distill_loss"] += jsd_token_loss.item()
                metrics["align_loss"] += hidden_align_loss.item()
                total_samples += 1

        if total_samples > 0:
            total_loss = total_loss / total_samples
            metrics["distill_loss"] /= total_samples
            metrics["align_loss"] /= total_samples
            metrics["mean_reward"] /= len(prompts)
            metrics["mean_penalty"] /= len(prompts)
        return total_loss, metrics


# ====================== 8. 扁平化 VRAM 密集 JSD ======================
def generalized_jsd_loss_flat(s_valid_logits, t_valid_logits, beta=0.5, temperature=1.0, chunk_size=2048):
    N, V_s = s_valid_logits.shape
    M, V_t = t_valid_logits.shape

    if N != M:
        raise RuntimeError(f"未成功完成对齐，Token 对数尺寸不一致: {N} vs {M}.")

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
            jsd = beta_tensor * kl_teacher + (1.0 - beta_tensor) * kl_student

        total_loss += jsd.sum()

    return total_loss / max(N, 1)

def extract_valid_logits(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    valid_mask = flat_labels != -100
    return flat_logits[valid_mask], flat_labels[valid_mask]


# ====================== 9. 闭合 Benchmark 泛化性评测 ======================
@torch.inference_mode()
def validate(student_model, teacher_model, dataloader, args, ctx):
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
    verifier,
    dataset_config, 
    val_dataset_raw, 
    args, 
    num_samples=20,
    ood_dataset_raw=None
) -> Tuple[float, float, float, float, float, float, float]:
    device = args.device
    student_model.eval()
    student_base_model.eval()
    teacher_model.eval()

    num_samples = min(num_samples, len(val_dataset_raw))
    eval_indices = random.sample(range(len(val_dataset_raw)), num_samples)
    eval_samples = [val_dataset_raw[i] for i in eval_indices]

    results_base = []
    results_student = []
    results_rag = []
    student_rewards = []
    
    retrieval_hits = []
    retrieval_mrrs = []

    print("\n" + "="*40)
    print(f" 开始评估: 规模 {num_samples} 个样本 (防泄露保障评估，学生使用 RAG: {'✅' if args.student_use_rag else '❌'})")
    print("="*40)

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

        # RAG 验证评估
        if (not args.disable_rag_teacher or args.student_use_rag):
            hit, mrr = rag_manager.evaluate_retrieval(question, gold_doc_id, top_k=args.rag_top_k, alpha=args.alpha, retrieval_mode=args.retrieval_mode)
            retrieval_hits.append(hit)
            retrieval_mrrs.append(mrr)

        # 检索并构建评测 prompt
        if args.student_use_rag:
            retrieved_context, _ = rag_manager.retrieve(query=question, top_k=args.rag_top_k, alpha=args.alpha, retrieval_mode=args.retrieval_mode)
            s_system_template = dataset_config.get("student_rag_system_prompt", "Please answer based on reference:\n{rag_context}")
            s_system = s_system_template.format(rag_context=retrieved_context)
            s_prompt_input = f"Retrieved Context:\n{retrieved_context}\n\nQuestion: {question}"
        else:
            retrieved_context = ""
            s_system = dataset_config["student_system_prompt"]
            s_prompt_input = f"Question: {question}"

        messages_base = [{"role": "system", "content": s_system}, {"role": "user", "content": s_prompt_input}]
        try:
            prompt_student = tokenizer.apply_chat_template(messages_base, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        except Exception:
            prompt_student = tokenizer.apply_chat_template(messages_base, tokenize=False, add_generation_prompt=True)
            
        inputs_student = tokenizer([prompt_student], return_tensors="pt").to(device)
        
        # 模型输出
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

        # 记录 NLI 忠实度打分
        reward_score = verifier.compute_reward(retrieved_context if retrieved_context else question, generated_student)
        student_rewards.append(reward_score)

        # 评测教师模型 (带 RAG)
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

        if idx < 3:
            print(f"\n[验证调试样例 {idx+1}]")
            print(f"👉 Q: {question[:150]}...")
            print(f"🎯 真实结果 (Gold): {gold_answer}")
            print(f"❌ Base Student  : '{pred_base}' | 对比结果: {'✓' if pred_base == gold_answer else '✗'}")
            print(f"🚀 OPD Student   : '{pred_student}' | 对比结果: {'✓' if pred_student == gold_answer else '✗'}")
            print(f"👨‍🏫 Teacher (RAG) : '{pred_rag}' | 对比结果: {'✓' if pred_rag == gold_answer else '✗'}")

    # ================= OOD 泛化测试 (MedQA) =================
    ood_acc = 0.0
    if ood_dataset_raw is not None and len(ood_dataset_raw) > 0:
        print("\n" + "-"*30 + " 执行 OOD (MedQA-USMLE) 泛化评测 " + "-"*30)
        ood_samples_num = min(num_samples, len(ood_dataset_raw))
        ood_indices = random.sample(range(len(ood_dataset_raw)), ood_samples_num)
        ood_results = []
        medqa_cfg = DATASET_CONFIGS["medqa"]

        for oidx in ood_indices:
            sample = ood_dataset_raw[oidx]
            q = sample["question"]
            
            option_keys = medqa_cfg.get("option_keys", [])
            options = []
            for idx_opt, key in enumerate(option_keys):
                label = chr(ord('A') + idx_opt)
                if key in sample:
                    opt_text = sample[key]
                else:
                    opt_text = sample.get("options", {}).get(label, "")
                options.append(f"{label}: {opt_text}")
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
        print(f"🎯 OOD (MedQA) 学生泛化准确率: {ood_acc:.2f}%")

    opd_student_acc = np.mean(results_student) * 100
    base_student_acc = np.mean(results_base) * 100
    teacher_acc = np.mean(results_rag) * 100
    avg_faithfulness = np.mean(student_rewards)
    m_hit = np.mean(retrieval_hits) * 100 if retrieval_hits else 0.0
    m_mrr = np.mean(retrieval_mrrs) * 100 if retrieval_mrrs else 0.0

    print("\n" + "="*40)
    print(" 评测指标汇总统计 ")
    print("="*40)
    print(f"评估样本总数 (Eval Count)  : {num_samples}")
    print(f"Base Student 准确率      : {base_student_acc:.2f}%")
    print(f"OPD Student  准确率      : {opd_student_acc:.2f}%")
    print(f"OPD Student  NLI忠实度   : {avg_faithfulness:.4f}")
    print(f"Teacher (RAG) 准确率     : {teacher_acc:.2f}%")
    print(f"RAG Hit Rate @ {args.rag_top_k}    : {m_hit:.2f}%")
    print(f"RAG MRR                  : {m_mrr:.2f}%")
    print(f"OOD MedQA 泛化准确率     : {ood_acc:.2f}%")
    print("="*40)
    
    student_model.train()
    return opd_student_acc, base_student_acc, teacher_acc, m_hit, m_mrr, ood_acc, avg_faithfulness


# ====================== 10. 训练控制控制中心 ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Academic RAG-Augmented OPD (Unified Adv-Weighted Pipeline)")
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
    parser.add_argument("--dataset", type=str, default="pubmed_qa", choices=["pubmed_qa", "medqa", "custom"])
    parser.add_argument("--hf_teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3.5-9B")
    
    # 路径参数
    parser.add_argument("--custom_qa_path", type=str, default=None, help="本地自定义 QA 数据路径")
    parser.add_argument("--rag_corpus_path", type=str, default=None, help="本地自定义 RAG 知识库路径")
    parser.add_argument('--cache_dir', type=str, default="./hf_cache")

    # 对齐控制
    parser.add_argument("--student_use_rag", action="store_true", default=False, help="学生端是否注入 RAG 支持")
    parser.add_argument("--disable_rag_teacher", action="store_true", help="禁用 RAG 教师增强")
    parser.add_argument('--contr_weight', type=float, default=0.1, help="检索表示对比对齐损失权重")
    parser.add_argument('--opd_group_size', type=int, default=3, help="强化探索 Rollout 的组尺寸 G")
    
    # RAG 参数
    parser.add_argument("--rag_top_k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--rag_dense_model", type=str, default="BAAI/bge-small-zh")
    parser.add_argument("--retrieval_mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"])
    
    # 对齐算法参数
    parser.add_argument("--opd_loss_type", type=str, default="jsd")
    parser.add_argument("--opd_chunk_size", type=int, default=512)
    parser.add_argument("--opd_beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opd_checkpoint_dir", type=str, default="opd_checkpoints")
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 强化学习与 Rollout
    parser.add_argument("--rollout_ratio", type=float, default=0.7)
    parser.add_argument("--pg_penalty_weight", type=float, default=0.1)
    
    # 评测配置
    parser.add_argument("--num_eval", type=int, default=20)
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--save_best_only", action="store_true", default=True)

    # 消融模式
    parser.add_argument("--ablation_mode", type=str, default="none", choices=["none", "vanilla_sft", "offline_kd"])

    # 监控
    parser.add_argument('--enable_wandb', action="store_true", default=False)
    parser.add_argument('--wandb_project', type=str, default="Tiny-R2-OPD-v5")
    parser.add_argument('--wandb_run_id', type=str, default=None)
    
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.opd_checkpoint_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    device = args.device

    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29533'
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    print("\n" + "="*70)
    print("🚀 Tiny-R2 优势加权表征 OPD (在策蒸馏) V5 合并版启动")
    print(f"📦 缓存地址: {args.cache_dir} | 消融状态: {args.ablation_mode}")
    print("="*70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    if TINY_R2_AVAILABLE:
        config.vocab_size = len(tokenizer)

    # 1. 载入数据集
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
        
        is_zh = False
        if len(custom_data) > 0:
            first_sample_str = str(custom_data[0])
            if re.search(r'[\u4e00-\u9fff]', first_sample_str):
                is_zh = True
        
        if is_zh:
            print("🇨🇳 检测到中文内容，自适应调整 System Prompt 。")
            dataset_config["student_system_prompt"] = (
                "你是一个专业的智能助手。针对给定的问题，首先在正文中进行清晰、简明扼要的逐步分析与推理（保持在3句以内）。"
                "然后在最后使用以下格式给出你的最终结论: '[Final Decision]: yes'（是）、'[Final Decision]: no'（否）或 '[Final Decision]: maybe'（可能）。\n\n"
                "现在请按照上述格式回答用户的问题。"
            )
            dataset_config["student_rag_system_prompt"] = (
                "你是一个专业的智能助手。你将获得‘检索到的背景信息’和‘问题’。请根据提供的背景信息回答问题。\n\n"
                "[Retrieved Context]\n{rag_context}\n\n"
                "首先在正文中进行清晰、简明扼要的逐步分析与推理（保持在3句以内），然后在最后使用以下格式给出最终结论: '[Final Decision]: yes'、'[Final Decision]: no' 或 '[Final Decision]: maybe'。\n\n"
                "现在请利用提供的背景信息回答问题。"
            )
            dataset_config["teacher_system_prompt"] = dataset_config["student_rag_system_prompt"]
            dataset_config["is_mcq"] = False
        else:
            if len(custom_data) > 0:
                first_ans = str(custom_data[0].get("answer", "")).strip().upper()
                if len(first_ans) == 1 and first_ans in "ABCD":
                    dataset_config["is_mcq"] = True
                    print("📝 自适应配置为 MCQ 题型。")
                else:
                    dataset_config["is_mcq"] = False
    else:
        dataset_config = DATASET_CONFIGS.get(args.dataset)
        if not dataset_config:
            raise ValueError(f"无法识别的数据集: {args.dataset}")

        print(f"📡 载入线上标准数据集: {args.dataset}")
        if dataset_config.get("hf_subset"):
            ds = data_loader(dataset_config["hf_path"], dataset_config["hf_subset"], split=dataset_config["split"], cache_dir=args.cache_dir)
        else:
            ds = data_loader(dataset_config["hf_path"], split=dataset_config["split"], cache_dir=args.cache_dir)

    val_size = max(1, int(len(ds) * 0.1)) if len(ds) > 1 else 0
    if val_size == 0:
        train_ds = ds
        val_ds_raw = ds
    else:
        train_ds = ds.select(range(val_size, len(ds)))
        val_ds_raw = ds.select(range(val_size))

    # 2. RAG 知识检索库初始化
    print("🔒 正在提取与清洗知识语料以防止 Label 泄露...")
    corpus = []
    if args.rag_corpus_path:
        print(f"🔒 读取本地知识语料库: {args.rag_corpus_path}")
        if args.rag_corpus_path.endswith('.jsonl'):
            with open(args.rag_corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        data = json.loads(line)
                        corpus.append({"pubid": data.get("pubid", str(i)), "text": data.get("text", data.get("content", str(data)))})
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
            corpus.append({"pubid": str(i), "text": item.get(dataset_config["instruction_key"], "").strip()})
    print(f"[-] 检索库提取完毕。总条目数: {len(corpus)}")

    rag_manager = CleanMedicalRAGManager(corpus=corpus, dense_model_name=args.rag_dense_model, device=device)

    train_dataset = DualPromptDataset(train_ds, tokenizer, args.ctx_len, dataset_config, rag_manager, args)
    val_dataset = DualPromptDataset(val_ds_raw, tokenizer, args.ctx_len, dataset_config, rag_manager, args)

    try:
        print("📡 尝试载入 OOD 验证集 (MedQA-USMLE-4-options)...")
        ood_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test", cache_dir=args.cache_dir)
    except Exception:
        print("⚠️ 提示: OOD 数据集载入未完备，训练中将自动跳过 OOD 验证。")
        ood_ds = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: dual_collate_fn(x, tokenizer))
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=lambda x: dual_collate_fn(x, tokenizer))
    data_loader=DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: dual_collate_fn(x, tokenizer))

    # 3. 初始化 NLI 神经验证器
    verifier = NeuralVerifierRewardModel(cache_dir=args.cache_dir, device=device)

    # 4. 载入模型结构
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = (compute_dtype == torch.float16)

    if args.student_model_name.lower() == "tiny-r2" and TINY_R2_AVAILABLE:
        student_model = Transformer().to(device)
        student_base_model = Transformer().to(device)
    else:
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_name, torch_dtype=compute_dtype, device_map=device, cache_dir=args.cache_dir,
            output_hidden_states=True
        )
        student_base_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_name, torch_dtype=compute_dtype, device_map=device, cache_dir=args.cache_dir
        )

    is_hf = hasattr(student_model, "config")

    if args.ablation_mode != "vanilla_sft":
        print(f"👨‍🏫 载入蒸馏教师模型: {args.hf_teacher_model}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.hf_teacher_model, torch_dtype=compute_dtype, device_map=device, cache_dir=args.cache_dir,
            output_hidden_states=True
        ).eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
    else:
        teacher_model = None

    # 5. 表征引擎与优化参数分配
    student_dim = student_model.config.hidden_size if is_hf else getattr(config, "hidden_size", 1024)
    teacher_dim = teacher_model.config.hidden_size if (teacher_model is not None and is_hf) else student_dim
    alignment_engine = RepresentationAlignmentEngine(student_dim, teacher_dim).to(device)

    # 强化加权在策蒸馏核心控制实例化
    opd_engine = AdvantageWeightedOPDEngine(
        student_model=student_model, 
        teacher_model=teacher_model, 
        alignment_engine=alignment_engine, 
        verifier=verifier, 
        tokenizer=tokenizer,
        opd_beta=args.opd_beta,
        temp=args.temperature,
        pg_penalty_weight=args.pg_penalty_weight
    )

    optimizers = [torch.optim.AdamW(
        list(student_model.parameters()) + list(alignment_engine.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )]
    schedulers = [CosineAnnealingLR(opt, T_max=args.max_iters) for opt in optimizers]

    # 断点续训支持
    best_val_loss = float("inf")
    start_iter = 0
    loaded_wandb_id = None

    if args.student_ckpt and os.path.exists(args.student_ckpt):
        try:
            ckpt = torch.load(args.student_ckpt, map_location=device)
            if isinstance(ckpt, dict) and 'model' in ckpt:
                student_model.load_state_dict(ckpt['model'], strict=False)
                if 'align_engine' in ckpt:
                    alignment_engine.load_state_dict(ckpt['align_engine'])
                start_iter = ckpt.get('step', 0)
                best_val_loss = ckpt.get('best_loss', float('inf'))
                loaded_wandb_id = ckpt.get('wandb_run_id', None)
            else:
                student_model.load_state_dict(ckpt, strict=False)
            print("✅ 成功断点续接恢复，对齐对齐参数一致。")
        except Exception as e:
            print(f"⚠️ 权重载入发生跳过或异常 ({e})。")

    if loaded_wandb_id and not args.wandb_run_id:
        args.wandb_run_id = loaded_wandb_id

    if args.enable_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=args.wandb_project, 
                name=f"TinyR2-OPD-v5-{args.dataset}-{args.ablation_mode}",
                config=vars(args),
                resume="must" if args.wandb_run_id else False,
                id=args.wandb_run_id
            )
            if wandb.run is not None:
                args.wandb_run_id = wandb.run.id
        except Exception:
            args.enable_wandb = False

    ctx = torch.amp.autocast(device_type="cuda", dtype=compute_dtype) if "cuda" in device else nullcontext()
    scaler = amp.GradScaler(enabled=use_scaler)

    print("\n🔍 正在执行初始评测以标定性能曲线...")
    _ = validate_comprehensive_accuracy(
        student_model, student_base_model, teacher_model if teacher_model else student_model, 
        tokenizer, rag_manager, verifier, dataset_config, val_ds_raw, args, num_samples=args.num_eval, ood_dataset_raw=ood_ds
    )

    global_step = start_iter
    train_iter = iter(train_loader)
    student_model.train()

    print("\n🔥 开始融合在策/离策多通道训练...")
    while global_step < args.max_iters:
        step_start = time.time()
        student_model.zero_grad()
        alignment_engine.zero_grad()
        
        step_losses = {"total_loss": 0.0, "distill_loss": 0.0, "align_loss": 0.0, "contr_loss": 0.0, "mean_reward": 0.0}

        # 计算自适应的在策 Rollout 比率
        if args.ablation_mode == "vanilla_sft" or args.ablation_mode == "offline_kd":
            current_rollout_ratio = 0.0
        else:
            warmup_opd_steps = int(args.max_iters * 0.1)
            ramp_steps = int(args.max_iters * 0.5)
            if global_step < warmup_opd_steps:
                current_rollout_ratio = 0.0
            elif global_step < (warmup_opd_steps + ramp_steps):
                current_rollout_ratio = ((global_step - warmup_opd_steps) / ramp_steps) * args.rollout_ratio
            else:
                current_rollout_ratio = args.rollout_ratio

        use_rollout = (random.random() < current_rollout_ratio) if current_rollout_ratio > 0 else False

        for accum_step in range(args.grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            if use_rollout:
                # --- [路径 A：在策强化探索 (GRP-Advantage Rollout)] ---
                loss_opd, opd_metrics = opd_engine.compute_on_policy_loss(
                    prompts=batch["s_prompt_ids"],
                    contexts=batch["contexts"],
                    group_size=args.opd_group_size
                )
                
                loss = loss_opd / args.grad_accum_steps
                step_losses["total_loss"] += loss.item()
                step_losses["distill_loss"] += opd_metrics["distill_loss"] / args.grad_accum_steps
                step_losses["align_loss"] += opd_metrics["align_loss"] / args.grad_accum_steps
                step_losses["mean_reward"] += opd_metrics["mean_reward"] / args.grad_accum_steps

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            else:
                # --- [路径 B：标准离策或 SFT 密集对齐路径 (Offline Engine)] ---
                s_input_ids = batch["s_input_ids"].to(device)
                s_labels = batch["s_labels"].to(device)
                s_neg_input_ids = batch["s_neg_input_ids"].to(device)
                t_input_ids = batch["t_input_ids"].to(device)
                t_labels = batch["t_labels"].to(device)

                s_attn_mask = (s_input_ids != tokenizer.pad_token_id).to(device) if is_hf else None

                if args.ablation_mode == "vanilla_sft":
                    with ctx:
                        outputs = student_model(s_input_ids, attention_mask=s_attn_mask) if is_hf else student_model(s_input_ids)
                        s_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                    loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1), ignore_index=-100)
                    
                    loss = loss / args.grad_accum_steps
                    step_losses["total_loss"] += loss.item()
                    
                    if use_scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                else:
                    t_attn_mask = (t_input_ids != tokenizer.pad_token_id).to(device) if is_hf else None
                    with torch.no_grad():
                        if is_hf:
                            t_outputs = teacher_model(t_input_ids, attention_mask=t_attn_mask, output_hidden_states=True)
                            t_logits = t_outputs.logits
                            t_hidden = t_outputs.hidden_states[-1] if hasattr(t_outputs, "hidden_states") else None
                        else:
                            t_outputs = teacher_model(t_input_ids)
                            t_logits = t_outputs.logits if hasattr(t_outputs, "logits") else t_outputs[0]
                            t_hidden = None

                    with ctx:
                        if is_hf:
                            s_outputs = student_model(s_input_ids, attention_mask=s_attn_mask, output_hidden_states=True)
                            s_logits = s_outputs.logits
                            s_hidden = s_outputs.hidden_states[-1] if hasattr(s_outputs, "hidden_states") else None
                        else:
                            s_outputs = student_model(s_input_ids)
                            s_logits = s_outputs.logits if hasattr(s_outputs, "logits") else s_outputs[0]
                            s_hidden = None

                    s_valid_logits, _ = extract_valid_logits(s_logits, s_labels)
                    t_valid_logits, _ = extract_valid_logits(t_logits, t_labels)

                    if s_valid_logits.size(0) > 0:
                        loss_jsd = generalized_jsd_loss_flat(
                            s_valid_logits, 
                            t_valid_logits, 
                            beta=args.opd_beta, 
                            temperature=args.temperature, 
                            chunk_size=args.opd_chunk_size
                        )

                        loss_align = torch.tensor(0.0, device=device)
                        if s_hidden is not None and t_hidden is not None:
                            loss_align = alignment_engine(s_hidden, t_hidden)

                        # 条件对比表征损失
                        loss_contr = torch.tensor(0.0, device=device)
                        if s_hidden is not None and t_hidden is not None:
                            with ctx:
                                if is_hf:
                                    s_neg_outputs = student_model(s_neg_input_ids, output_hidden_states=True)
                                    s_neg_hidden = s_neg_outputs.hidden_states[-1] if hasattr(s_neg_outputs, "hidden_states") else None
                                else:
                                    s_neg_outputs = student_model(s_neg_input_ids)
                                    s_neg_hidden = None

                            if s_neg_hidden is not None:
                                loss_contr = alignment_engine.compute_contrastive_loss(s_hidden, s_neg_hidden, t_hidden)

                        loss = (loss_jsd + 0.2 * loss_align + args.contr_weight * loss_contr) / args.grad_accum_steps
                        
                        step_losses["total_loss"] += loss.item()
                        step_losses["distill_loss"] += loss_jsd.item() / args.grad_accum_steps
                        step_losses["align_loss"] += loss_align.item() / args.grad_accum_steps
                        step_losses["contr_loss"] += loss_contr.item() / args.grad_accum_steps

                        if use_scaler:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()

        # 执行参数更新与步进
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
                
        for sched in schedulers:
            sched.step()
            
        global_step += 1
        current_lr = schedulers[0].get_last_lr()[0]

        if global_step % 10 == 0:
            print(f"Step {global_step:04d} | Rollout: {'✅' if use_rollout else '❌'} | Loss: {step_losses['total_loss']:.4f} "
                  f"[Logits-JSD: {step_losses['distill_loss']:.4f}, Hidden-Align: {step_losses['align_loss']:.4f}, Contrast: {step_losses['contr_loss']:.4f}] "
                  f"| Avg Reward: {step_losses['mean_reward']:.3f} | LR: {current_lr:.2e}")

        # ================= 验证与高保真状态转储 =================
        if global_step % args.val_freq == 0:
            val_loss = validate(student_model, teacher_model, val_loader, args, ctx)
            is_best = val_loss < best_val_loss
            print(f"\n📊 [验证指标检测] | Step={global_step} | Val Loss={val_loss:.4f} | Best Loss={best_val_loss:.4f}")

            opd_student_acc, base_student_acc, teacher_acc, hit, mrr, ood_acc, avg_faithfulness = validate_comprehensive_accuracy(
                student_model, student_base_model, teacher_model if teacher_model else student_model, 
                tokenizer, rag_manager, verifier, dataset_config, val_ds_raw, args, num_samples=args.num_eval, ood_dataset_raw=ood_ds
            )

            if args.enable_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "eval/loss": val_loss,
                    "eval/teacher_accuracy": teacher_acc,
                    "eval/base_student_accuracy": base_student_acc,
                    "eval/opd_student_accuracy": opd_student_acc,
                    "eval/student_faithfulness": avg_faithfulness,
                    "eval/student_improvement": opd_student_acc - base_student_acc,
                    "eval/rag_hit_rate": hit,
                    "eval/rag_mrr": mrr,
                    "eval/ood_accuracy": ood_acc
                }, step=global_step)

            if is_best:
                best_val_loss = val_loss
                save_path = os.path.join(args.opd_checkpoint_dir, f"student_best_model_step_{global_step}.pt")
                torch.save({
                    'model': student_model.state_dict(),
                    'align_engine': alignment_engine.state_dict(),
                    'optimizer_states': [opt.state_dict() for opt in optimizers],
                    'scheduler_state': schedulers[0].state_dict(),
                    'step': global_step,
                    'best_loss': best_val_loss,
                    'wandb_run_id': args.wandb_run_id if args.enable_wandb else None
                }, save_path)
                print(f"🏆 成功捕捉更好的模型状态，完整权重文件已保存至: {save_path}\n")
                
                if args.save_best_only:
                    for old_file in glob.glob(os.path.join(args.opd_checkpoint_dir, "student_best_model_step_*.pt")):
                        if old_file != save_path:
                            try:
                                os.remove(old_file)
                            except Exception: 
                                pass

    print("🎉 OPD 蒸馏训练流程顺利执行完毕。")
    if args.enable_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    main()
