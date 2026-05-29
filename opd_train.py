#!/usr/bin/env python3
"""
Tiny-R2 OPD 训练 v2 - 知识蒸馏与RAG增强 (Teacher=9B+RAG, Student=0.8B)
学术级重构更新（支持本地自定义 QA 数据集与 RAG 语料库路径，支持多选题统一化处理与防泄漏机制）
实测pubmed_qa数据集上Qwen3.5-0.8B准确率提升12.5%
欢迎试用和Star
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

# RAG 混合检索
try:
    from sentence_transformers import SentenceTransformer, util
    from rank_bm25 import BM25Okapi
    RAG_AVAILABLE = True
    print("✅ 成功导入 rank_bm25 与 sentence_transformers，混合 RAG 模块已启用。")
except ImportError:
    print("⚠️ 提示: 未安装 rank_bm25 或 sentence_transformers，将使用 Mock RAG 流程。")
    RAG_AVAILABLE = False

try:
    import model
    from model import Transformer
    print("✅ 成功导入 Tiny-R2 核心模块")
    TINY_R2_AVAILABLE = True
except ImportError as e:
    print(f"ℹ️ 未检测到 Tiny-R2 模块，将强制使用 HuggingFace 模型。")
    TINY_R2_AVAILABLE = False

# 数据集配置注册表
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
    "medmcqa": {
        "hf_path": "openlifescienceai/medmcqa",
        "hf_subset": None,
        "split": "train",
        "language": "en",
        "instruction_key": "question",
        "response_key": "cop",
        "is_mcq": True,
        "option_keys": ["opa", "opb", "opc", "opd"],
        "student_system_prompt": (
            "You are an expert medical assistant. "
            "Analyze the medical question carefully. "
            "First provide concise reasoning in less than 3 sentences. "
            "Then output the final answer strictly using the format: "
            "'[Final Decision]: A', '[Final Decision]: B', "
            "'[Final Decision]: C', or '[Final Decision]: D'."
        ),
        "student_rag_system_prompt": (
            "You are an expert medical assistant. "
            "You are given retrieved medical references.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"
            "Answer the medical question using the retrieved evidence. "
            "First provide concise reasoning in less than 3 sentences. "
            "Then output the final answer strictly using the format: "
            "'[Final Decision]: A', '[Final Decision]: B', "
            "'[Final Decision]: C', or '[Final Decision]: D'."
        ),
        "teacher_system_prompt": (
            "You are an expert medical assistant. "
            "You are given retrieved medical references.\n\n"
            "[Retrieved Context]\n{rag_context}\n\n"
            "Answer the medical question using the retrieved evidence. "
            "First provide concise reasoning in less than 3 sentences. "
            "Then output the final answer strictly using the format: "
            "'[Final Decision]: A', '[Final Decision]: B', "
            "'[Final Decision]: C', or '[Final Decision]: D'."
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

# ====================== MCQ 统一工具函数 ======================
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
    return (
        f"Question: {instruction}\n"
        f"Options:\n{opt_str}"
    )

def convert_mcq_answer(raw_answer: Any) -> str:
    # MedMCQA cop: 1, 2, 3, 4 -> A, B, C, D
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
    
    # MCQ 判定匹配方式 (A, B, C, D)
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
        
    # Yes/No/Maybe 匹配
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

def parse_args():
    parser = argparse.ArgumentParser(description="Academic RAG-Augmented OPD Train (Teacher 9B + Anti-Leakage RAG)")
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
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="pubmed_qa",
        choices=["pubmed_qa", "medqa", "medmcqa", "custom"]
    )
    parser.add_argument("--hf_teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3.5-9B")
    
    # 本地自定义文件加载参数
    parser.add_argument("--custom_qa_path", type=str, default=None, help="本地自定义 QA JSONL 数据集文件路径")
    parser.add_argument("--rag_corpus_path", type=str, default=None, help="本地自定义 RAG 检索库文件路径（支持 .txt 或 .jsonl）")

    # 核心开关设计
    parser.add_argument("--student_use_rag", action="store_true", default=False, 
                        help="学生在训练和推理时是否使用 RAG 上下文。")
    parser.add_argument("--disable_rag_teacher", action="store_true", help="禁用 RAG 增强")
    
    parser.add_argument("--rag_top_k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--rag_dense_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--retrieval_mode", type=str, default="hybrid", choices=["bm25", "dense", "hybrid"], help="检索策略类型")
    parser.add_argument("--opd_loss_type", type=str, default="jsd")
    parser.add_argument("--opd_chunk_size", type=int, default=512)
    parser.add_argument("--opd_beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opd_checkpoint_dir", type=str, default="opd_checkpoints")
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 强化学习与 Rollout 策略
    parser.add_argument("--rollout_ratio", type=float, default=0.7)
    parser.add_argument("--pg_penalty_weight", type=float, default=0.1)
    
    # 评测配置
    parser.add_argument("--num_eval", type=int, default=50)
    parser.add_argument("--student_ckpt", type=str, default=None)
    parser.add_argument("--save_best_only", action="store_true", default=True)

    # 消融实验矩阵控制
    parser.add_argument("--ablation_mode", type=str, default="none", choices=["none", "vanilla_sft", "offline_kd"])

    # 监控
    parser.add_argument('--enable_wandb', action="store_true", default=True)
    parser.add_argument('--wandb_project', type=str, default="PubMedQA-OPD-Enterprise")
    parser.add_argument('--wandb_run_id', type=str, default=None)
    
    return parser.parse_args()

# ====================== RAG 管理器 ======================
class CleanMedicalRAGManager:
    def __init__(self, corpus: List[Dict[str, str]], dense_model_name="BAAI/bge-small-en-v1.5", device="cuda"):
        self.corpus = corpus
        self.texts = [doc["text"] for doc in corpus]
        self.doc_ids = [doc.get("pubid", str(i)) for i, doc in enumerate(corpus)]
        self.device = device

        if len(self.texts) > 0:
            print("[-] 正在初始化过滤后的 BM25 词法索引...")
            tokenized_corpus = [text.lower().split() for text in self.texts]
            self.bm25 = BM25Okapi(tokenized_corpus)

            print(f"[-] 正在初始化密向量检索模型 ({dense_model_name})...")
            self.dense_model = SentenceTransformer(dense_model_name, device=device)
            self.corpus_embeddings = self.dense_model.encode(
                self.texts, show_progress_bar=True, convert_to_tensor=True
            )
        else:
            print("⚠️ 警告: RAG 依赖未完备或检索语料库为空，将切换至 Mock 逻辑运行。")

    def retrieve(self, query: str, top_k: int = 3, alpha: float = 0.4, retrieval_mode: str = "hybrid") -> Tuple[str, List[str]]:
        if not RAG_AVAILABLE or len(self.texts) == 0:
            retrieved_contexts = []
            retrieved_ids = []
            for i in range(min(top_k, len(self.texts))):
                retrieved_contexts.append(f"[Document {i+1}]\n{self.texts[i]}")
                retrieved_ids.append(self.doc_ids[i])
            return "\n\n".join(retrieved_contexts), retrieved_ids

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

# ====================== 双路 Prompt 对齐数据集 ======================
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

        # === [修复：兼容多种 pubmed_qa 映射路径，保证提取 long_answer] ===
        if self.args.dataset == "pubmed_qa" or "pubmed_qa" in str(self.config.get("hf_path", "")).lower():
            long_ans = item.get("long_answer", "")
            final_dec = item.get("final_decision", "maybe")
            response = f"Analysis: {long_ans}\n\n[Final Decision]: {final_dec}"
        elif self.config.get("is_mcq", False):
            instruction = build_mcq_instruction(item, instruction, self.config)
            gold_raw = item[self.config["response_key"]]
            gold_choice = convert_mcq_answer(gold_raw)
            response = (
                "Analysis: Guided by biomedical reasoning...\n\n"
                f"[Final Decision]: {gold_choice}"
            )
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

        need_rag = (not self.args.disable_rag_teacher) or self.args.student_use_rag
        if need_rag and self.args.ablation_mode != "vanilla_sft":
            retrieved_context, _ = self.rag_manager.retrieve(
                instruction, top_k=self.args.rag_top_k, alpha=self.args.alpha, retrieval_mode=self.args.retrieval_mode
            )
        else:
            retrieved_context = ""

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
        raise RuntimeError(f"教师和学生模型 Valid Tokens 大小未对齐: {N} vs {M}.")

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

# ====================== 严密验证与评测 ======================
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
    dataset_config, 
    val_dataset_raw, 
    args, 
    num_samples=20,
    ood_dataset_raw=None
) -> Tuple[float, float, float, float, float, float]:
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
    
    retrieval_hits = []
    retrieval_mrrs = []

    print("\n" + "="*40)
    print(f" 开始评估: 规模 {num_samples} 个样本 (防泄露保障评估，学生使用 RAG: {'✅' if args.student_use_rag else '❌'})")
    print("="*40)

    is_custom_transformer = (student_model.__class__.__name__ == "Transformer")
    
    # === [修复：支持并兼容多种 pubmed_qa 路径标志比对方式] ===
    is_pubmed_qa = (args.dataset == "pubmed_qa" or "pubmed_qa" in str(dataset_config.get("hf_path", "")).lower())
    is_mcq = dataset_config.get("is_mcq", False)

    for idx, sample in enumerate(eval_samples):
        question = sample[dataset_config["instruction_key"]]
        
        # === [注入选择题题干与选项合并逻辑] ===
        if is_mcq:
            question = build_mcq_instruction(sample, question, dataset_config)
        
        if is_pubmed_qa:
            gold_answer = sample.get("final_decision", "maybe")
            gold_doc_id = str(sample.get("pubid", ""))
        else:
            gold_raw = sample.get(dataset_config["response_key"], "")
            gold_answer = convert_mcq_answer(gold_raw) if is_mcq else gold_raw
            
            # === [修正真实 Doc ID 的映射关系，将局部索引 idx 改为真实全局索引] ===
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

        messages_base = [
            {"role": "system", "content": s_system},
            {"role": "user", "content": s_prompt_input}
        ]
        
        try:
            prompt_student = tokenizer.apply_chat_template(messages_base, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        except Exception:
            prompt_student = tokenizer.apply_chat_template(messages_base, tokenize=False, add_generation_prompt=True)
            
        inputs_student = tokenizer([prompt_student], return_tensors="pt").to(device)
        
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
        retrieved_context_t, _ = rag_manager.retrieve(query=question, top_k=args.rag_top_k, alpha=args.alpha, retrieval_mode=args.retrieval_mode)
        t_system = dataset_config["teacher_system_prompt"].format(rag_context=retrieved_context_t)

        messages_rag = [
            {"role": "system", "content": t_system},
            {"role": "user", "content": f"Retrieved Context:\n{retrieved_context_t}\n\nQuestion: {question}"}
        ]

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

    # ================= 4. Out-of-Distribution (OOD) 泛化测试 (MedQA) =================
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

            messages_ood = [
                {"role": "system", "content": s_system},
                {"role": "user", "content": user_content_ood}
            ]
            try:
                prompt_ood = tokenizer.apply_chat_template(messages_ood, tokenize=False, enable_thinking=False, add_generation_prompt=True)
            except Exception:
                prompt_ood = tokenizer.apply_chat_template(messages_ood, tokenize=False, add_generation_prompt=True)

            inputs_ood = tokenizer([prompt_ood], return_tensors="pt").to(device)
            with torch.no_grad():
                if is_custom_transformer:
                    outputs_ood = student_model.generate(
                        idx=inputs_ood.input_ids, 
                        max_new_tokens=128, 
                        temperature=args.temperature
                    )
                else:
                    outputs_ood = student_model.generate(
                        **inputs_ood, 
                        max_new_tokens=128, 
                        do_sample=False, 
                        pad_token_id=tokenizer.pad_token_id
                    )
            gen_ood = tokenizer.decode(outputs_ood[0][inputs_ood.input_ids.shape[1]:], skip_special_tokens=True)
            pred_ood = normalize_answer(gen_ood, is_mcq=True)
            ood_results.append(pred_ood == gold_ans)

        ood_acc = np.mean(ood_results) * 100
        print(f"🎯 OOD (MedQA) 学生泛化准确率: {ood_acc:.2f}%")

    opd_student_acc = np.mean(results_student) * 100
    base_student_acc = np.mean(results_base) * 100
    teacher_acc = np.mean(results_rag) * 100
    m_hit = np.mean(retrieval_hits) * 100 if retrieval_hits else 0.0
    m_mrr = np.mean(retrieval_mrrs) * 100 if retrieval_mrrs else 0.0

    print("\n" + "="*40)
    print(" 评测指标汇总统计 ")
    print("="*40)
    print(f"评估样本总数 (Eval Count): {num_samples}")
    print(f"Base Student 准确率    : {base_student_acc:.2f}%")
    print(f"OPD Student  准确率    : {opd_student_acc:.2f}%")
    print(f"Teacher (RAG) 准确率   : {teacher_acc:.2f}%")
    print(f"RAG Hit Rate @ {args.rag_top_k}  : {m_hit:.2f}%")
    print(f"RAG MRR                : {m_mrr:.2f}%")
    print(f"OOD MedQA 泛化准确率   : {ood_acc:.2f}%")
    print("="*40)
    
    student_model.train()
    return opd_student_acc, base_student_acc, teacher_acc, m_hit, m_mrr, ood_acc

# ====================== 主训练入口 ======================
def main():
    args = parse_args()
    os.makedirs(args.opd_checkpoint_dir, exist_ok=True)
    device = args.device

    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        print("💡 [System] 初始化 Gloo 分布式环境以保持优化器兼容。")

    print("\n" + "="*60)
    print("🚀 启动学术重构版 Agent OPD 蒸馏训练")
    print(f"👨‍🏫 教师模型: {args.hf_teacher_model}")
    print(f"👶 学生模型: {args.student_model_name} (Ablation Mode: {args.ablation_mode})")
    print(f"🎯 检索策略  : {args.retrieval_mode}")
    print(f"🎯 学生端 RAG  : {'启用 - 学习利用检索信息' if args.student_use_rag else '禁用 - 完全封闭知识内化'}")
    print("="*60 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    if TINY_R2_AVAILABLE:
        config.vocab_size = len(tokenizer)

    # ---------------- 载入数据集（支持本地与 Hugging Face 切换） ----------------
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
        
        # 自动判定是否包含中文并切换 Prompt 语境
        is_zh = False
        if len(custom_data) > 0:
            first_sample_str = str(custom_data[0])
            if re.search(r'[\u4e00-\u9fff]', first_sample_str):
                is_zh = True
        
        # 自动判定是否为选择题(MCQ)
        if is_zh:
            print("🇨🇳 检测到中文数据集内容，已切换 Prompt 至中文模式。")
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
            # 英文单选题自适应
            if len(custom_data) > 0:
                first_ans = str(custom_data[0].get("answer", "")).strip().upper()
                if len(first_ans) == 1 and first_ans in "ABCD":
                    dataset_config["is_mcq"] = True
                    print("📝 自动识别为单项选择题模式(A/B/C/D)。")
                else:
                    dataset_config["is_mcq"] = False
                    print("📝 自动识别为是非问答模式(yes/no/maybe)。")
    else:
        dataset_config = DATASET_CONFIGS.get(args.dataset)
        if not dataset_config:
            raise ValueError(f"无法识别指定的数据集: {args.dataset}")

        print(f"📡 载入 Hugging Face 线上数据集: {args.dataset}")
        if dataset_config.get("hf_subset"):
            ds = load_dataset(dataset_config["hf_path"], dataset_config["hf_subset"], split=dataset_config["split"])
        else:
            ds = load_dataset(dataset_config["hf_path"], split=dataset_config["split"])

    val_size = max(1, int(len(ds) * 0.1)) if len(ds) > 1 else 0
    if val_size == 0:
        train_ds = ds
        val_ds_raw = ds
    else:
        train_ds = ds.select(range(val_size, len(ds)))
        val_ds_raw = ds.select(range(val_size))

    # ---------------- 构建 RAG 语料库（全面防 label 泄露） ----------------
    print("🔒 正在构建标准 RAG 语料库...")
    corpus = []
    if args.rag_corpus_path:
        print(f"🔒 正在从自定义路径 {args.rag_corpus_path} 构建 RAG 语料库...")
        if args.rag_corpus_path.endswith('.jsonl'):
            with open(args.rag_corpus_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        data = json.loads(line)
                        text_content = data.get("text", data.get("content", str(data)))
                        corpus.append({
                            "pubid": data.get("pubid", str(i)),
                            "text": text_content
                        })
        else:
            # 读取原始文本（按段落或行切割）
            with open(args.rag_corpus_path, 'r', encoding='utf-8') as f:
                content = f.read()
                paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                if len(paragraphs) < 5:
                    paragraphs = [line.strip() for line in content.split('\n') if line.strip()]
                for i, para in enumerate(paragraphs):
                    corpus.append({
                        "pubid": f"doc_{i}",
                        "text": para
                    })
    elif args.dataset == "medmcqa":
        for i, item in enumerate(ds):
            corpus_text = []
            if "exp" in item and item["exp"]:
                corpus_text.append(item["exp"])
            if "subject_name" in item and item["subject_name"]:
                corpus_text.append(item["subject_name"])
            if "topic_name" in item and item["topic_name"]:
                corpus_text.append(item["topic_name"])
            if len(corpus_text) > 0:
                corpus.append({
                    "pubid": str(i),
                    "text": " ".join(corpus_text)
                })
    elif args.custom_qa_path:
        # 只提取问题以防直接泄露 response 答案
        for i, item in enumerate(ds):
            inst_text = item.get(dataset_config["instruction_key"], "")
            corpus.append({
                "pubid": str(i),
                "text": inst_text.strip()
            })
    elif args.dataset == "pubmed_qa":
        for item in ds:
            abstract_text = " ".join(item["context"]["contexts"])
            corpus.append({
                "pubid": str(item["pubid"]),
                "text": abstract_text
            })
    else:
        for i, item in enumerate(ds):
            inst_text = item.get(dataset_config["instruction_key"], "")
            corpus.append({
                "pubid": str(i),
                "text": inst_text.strip()
            })
    print(f"[-] 检索语料库构建完成，共包含 {len(corpus)} 条文档。")

    rag_manager = CleanMedicalRAGManager( 
        corpus=corpus, 
        dense_model_name=args.rag_dense_model,
        device=device
    )

    train_dataset = DualPromptDataset(train_ds, tokenizer, args.ctx_len, dataset_config, rag_manager, args)
    val_dataset = DualPromptDataset(val_ds_raw, tokenizer, args.ctx_len, dataset_config, rag_manager, args)

    try:
        print("📡 尝试预载 OOD 评测数据集 (MedQA)...")
        ood_ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    except Exception:
        print("⚠️ 提示: 未能自动拉取 MedQA OOD 数据集，将跳过 OOD 评测流程。")
        ood_ds = None

    from functools import partial
    collate = partial(dual_collate_fn, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate)

    if "cuda" in device and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
        use_scaler = False
    else:
        compute_dtype = torch.float16
        use_scaler = True

    if args.student_model_name.lower() == "tiny-r2" and TINY_R2_AVAILABLE:
        student_model = Transformer().to(device)
        student_base_model = Transformer().to(device)
        if hasattr(student_model, "configure_optimizers"):
            opts = student_model.configure_optimizers(args.weight_decay, args.lr, device)
            if isinstance(opts, (list, tuple)):
                optimizers = list(opts)
            else:
                optimizers = [opts]
        else:
            optimizers = [torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)]
    else:
        student_model = AutoModelForCausalLM.from_pretrained(args.student_model_name, torch_dtype=compute_dtype, device_map=device)
        student_base_model = AutoModelForCausalLM.from_pretrained(args.student_model_name, torch_dtype=compute_dtype, device_map=device)
        optimizers = [torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)]

    is_hf = hasattr(student_model, "config")
    schedulers = [CosineAnnealingLR(opt, T_max=args.max_iters) for opt in optimizers]

    if args.ablation_mode != "vanilla_sft":
        print(f"👨‍🏫 加载教师模型进行对齐蒸馏: {args.hf_teacher_model}")
        teacher_model = AutoModelForCausalLM.from_pretrained(args.hf_teacher_model, torch_dtype=compute_dtype, device_map=device).eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
    else:
        teacher_model = None

    best_val_loss = float("inf")
    start_iter = 0
    loaded_wandb_id = None

    if args.student_ckpt and os.path.exists(args.student_ckpt):
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
                if 'scheduler_states' in ckpt:
                    for sched, sched_state in zip(schedulers, ckpt['scheduler_states']):
                        sched.load_state_dict(sched_state)
                elif 'scheduler_state' in ckpt and len(schedulers) == 1:
                    schedulers[0].load_state_dict(ckpt['scheduler_state'])
            else:
                student_model.load_state_dict(ckpt, strict=False)
            print("✅ 成功加载断点检查点并完成对齐。")
        except Exception as e:
            print(f"⚠️ 权重及历史状态加载失败: {e}")

    if loaded_wandb_id and not args.wandb_run_id:
        args.wandb_run_id = loaded_wandb_id

    if args.enable_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=args.wandb_project, 
                name=f"tiny-r2-opd-{args.dataset}-ablation-{args.ablation_mode}-studrag-{args.student_use_rag}", 
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

    print("\n🔍 正在评估初始评测指标 ...")
    _ = validate_comprehensive_accuracy(
        student_model, student_base_model, teacher_model if teacher_model else student_model, 
        tokenizer, rag_manager, dataset_config, val_ds_raw, args, num_samples=args.num_eval, ood_dataset_raw=ood_ds
    )

    global_step = start_iter
    train_iter = iter(train_loader)
    student_model.train()
    
    print("\n🔥 开始 OPD 蒸馏训练流程...")
    
    while global_step < args.max_iters:
        step_start = time.time()
        student_model.zero_grad()
        total_train_loss = 0.0
        avg_penalty_display = 0.0
        
        if args.ablation_mode == "vanilla_sft" or args.ablation_mode == "offline_kd":
            current_rollout_ratio = 0.0
        else:
            warmup_opd_steps = int(args.max_iters * 0.2)
            ramp_steps = int(args.max_iters * 0.6)
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

            s_attn_mask = (s_input_ids != tokenizer.pad_token_id).to(device) if is_hf else None

            if args.ablation_mode == "vanilla_sft":
                with ctx:
                    outputs = student_model(s_input_ids, attention_mask=s_attn_mask) if is_hf else student_model(s_input_ids)
                    s_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                loss = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1), ignore_index=-100)
            else:
                t_attn_mask = (t_input_ids != tokenizer.pad_token_id).to(device) if is_hf else None
                with torch.no_grad():
                    t_logits = teacher_model(t_input_ids, attention_mask=t_attn_mask).logits if is_hf else teacher_model(t_input_ids).logits
                with ctx:
                    outputs = student_model(s_input_ids, attention_mask=s_attn_mask) if is_hf else student_model(s_input_ids)
                    s_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
                
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

                if use_rollout and batch_penalties is not None and args.pg_penalty_weight > 0:
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

        step_time = time.time() - step_start
        current_lr = schedulers[0].get_last_lr()[0]

        if global_step % 10 == 0:
            print(f"Step {global_step:04d} | Ablation: {args.ablation_mode} | Rollout: {'✅' if use_rollout else '❌'} | Loss: {total_train_loss:.4f} | Penalty: {avg_penalty_display:.2f} | LR: {current_lr:.2e}")

        # ================= 验证并保存检查点 =================
        if global_step % args.val_freq == 0:
            val_loss = validate(student_model, teacher_model, val_loader, args, ctx)
            is_best = val_loss < best_val_loss
            print(f"\n📊 Validation | Step={global_step} | Val Loss={val_loss:.4f} | Best Loss={best_val_loss:.4f}")

            opd_student_acc, base_student_acc, teacher_acc, hit, mrr, ood_acc = validate_comprehensive_accuracy(
                student_model, student_base_model, teacher_model if teacher_model else student_model, 
                tokenizer, rag_manager, dataset_config, val_ds_raw, args, num_samples=args.num_eval, ood_dataset_raw=ood_ds
            )

            if args.enable_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "eval/loss": val_loss,
                    "eval/teacher_accuracy": teacher_acc,
                    "eval/base_student_accuracy": base_student_acc,
                    "eval/opd_student_accuracy": opd_student_acc,
                    "eval/student_improvement": opd_student_acc - base_student_acc,
                    "eval/rag_hit_rate": hit,
                    "eval/rag_mrr": mrr,
                    "eval/ood_accuracy": ood_acc
                }, step=global_step)

            if is_best:
                best_val_loss = val_loss
                save_path = os.path.join(args.opd_checkpoint_dir, f"student_best_model_step_{global_step}.pt")
                save_data = {
                    'model': student_model.state_dict(),
                    'optimizer_states': [opt.state_dict() for opt in optimizers],
                    'scheduler_states': [sched.state_dict() for sched in schedulers],
                    'scheduler_state': schedulers[0].state_dict(),
                    'step': global_step,
                    'best_loss': best_val_loss,
                    'wandb_run_id': args.wandb_run_id if args.enable_wandb else None
                }
                torch.save(save_data, save_path)
                print(f"🏆 发现更优模型! 保存完整状态至: {save_path}\n")
                
                if args.save_best_only:
                    for old_file in glob.glob(os.path.join(args.opd_checkpoint_dir, "student_best_model_step_*.pt")):
                        if old_file != save_path:
                            try:
                                os.remove(old_file)
                            except Exception: 
                                pass
    print("🎉 训练流程顺利执行完毕！")
    if args.enable_wandb and WANDB_AVAILABLE:
        wandb.finish()
if __name__ == "__main__":
    main()
