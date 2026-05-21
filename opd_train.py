#!/usr/bin/env python3
"""
Tiny-R2 OPD 训练 - 知识蒸馏与RAG增强 (Teacher=9B+RAG, Student=0.8B)
融合特性：
1. Online Rollout (解决 Exposure Bias) + 动态渐进式退火
2. Policy Gradient REINFORCE 废话惩罚 (Anti Self-Narration Penalty)
适配数据集：keivalya/MedQuad-MedicalQnADataset 或本地自定义 JSONL 数据集

"""
import os
import sys
import random
import argparse
import glob
import re
import config
import json
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

# ====================== 新增：RAG 支持所需库 ======================
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    print("⚠️ 提示: 未安装 faiss 或 sentence_transformers，将使用 Mock RAG 进行流程演示。")
    print("如需真实 RAG，请运行: pip install faiss-cpu sentence_transformers")
    RAG_AVAILABLE = False

# ====================== 导入Tiny-R2核心模块 (可选) ======================
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
    "medquad": {
        "hf_path": "keivalya/MedQuad-MedicalQnADataset",
        "hf_subset": None,
        "split": "train",
        "language": "en",
        "instruction_key": "Question",
        "response_key": "Answer",
        "student_system_prompt": "You are a professional medical assistant. Please provide accurate, safe, and helpful answers to the patient's questions.",
        "teacher_system_prompt": "You are an authoritative medical expert. Please answer the patient's question strictly based on the following [Authoritative Medical Reference].\n\n[Authoritative Medical Reference]\n{rag_context}"
    },
    "cmeqa": {
        "hf_path": "blcu-nlp/CMeQA",
        "hf_subset": None,
        "split": "train",
        "language": "zh",
        "instruction_key": "question",
        "response_key": "answer",
        "student_system_prompt": "你是一个专业的全科医生助手。请专业、准确、安全地解答患者的问题。",
        "teacher_system_prompt": "你是一个权威的主治医师评审。请根据以下【权威医疗参考资料】，专业、严谨地解答患者的问题。\n\n[权威医疗参考资料]\n{rag_context}"
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

# ====================== 参数解析 ======================
def parse_args():
    parser = argparse.ArgumentParser(description="RAG-Augmented OPD Train (Teacher 9B + RAG -> Student 0.8B)")
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
    parser.add_argument("--dataset", type=str, default="medquad")
    parser.add_argument("--hf_teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--student_model_name", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--tokenizer_name", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--enable_rag_teacher", action="store_true", default=True)
    parser.add_argument("--rag_top_k", type=int, default=2)
    parser.add_argument("--rag_corpus_path", type=str, default=None)
    parser.add_argument("--opd_loss_type", type=str, default="jsd")
    parser.add_argument("--opd_chunk_size", type=int, default=512)
    parser.add_argument("--opd_beta", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--opd_checkpoint_dir", type=str, default="opd_checkpoints")
    parser.add_argument("--val_freq", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # OPD 策略比率和惩罚权重
    parser.add_argument("--rollout_ratio", type=float, default=0.7, help="学生模型自主生成轨迹的最大概率")
    parser.add_argument("--pg_penalty_weight", type=float, default=0.1, help="策略梯度废话惩罚权重")
    
    # ======= 新增：修复报错所需的命令行参数 =======
    parser.add_argument("--student_ckpt", type=str, default=None, help="加载学生模型权重路径 (.pt 文件)")
    parser.add_argument("--custom_qa_path", type=str, default=None, help="本地自定义问答 JSONL 数据集路径")
    parser.add_argument("--save_best_only", action="store_true", default=True, help="是否只保留表现最好的检查点")
    
    return parser.parse_args()

# ====================== 轻量级 RAG 管理器 ======================
class MedicalRAGManager:
    def __init__(self, use_rag=True, corpus_texts=None, language="en"):
        self.use_rag = use_rag and RAG_AVAILABLE
        self.corpus = corpus_texts if corpus_texts else []
        self.language = language
        self.embedder = None
        self.index = None
        if self.use_rag and len(self.corpus) > 0:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2' if language == "en" else 'shibing624/text2vec-base-chinese'
            print(f"🔍 正在初始化 RAG 检索器 ({model_name})...")
            self.embedder = SentenceTransformer(model_name)
            embeddings = self.embedder.encode(self.corpus, show_progress_bar=True)
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            print(f"✅ RAG 向量库构建完成，共包含 {len(self.corpus)} 条医学知识。")

    def search(self, query: str, top_k: int = 2) -> str:
        if not self.use_rag or self.index is None:
            if self.language == "en":
                return "According to medical guidelines, this condition requires a comprehensive assessment based on specific examination results. Please consult your doctor."
            return "据医学权威资料显示，该症状需要结合具体检查结果进行综合评估，建议遵循医嘱。"
        
        q_emb = self.embedder.encode([query])
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        retrieved = []
        for idx in I[0]:
            if idx < len(self.corpus):
                retrieved.append(self.corpus[idx])
        return "\n\n".join(retrieved)

# ====================== 双路 Prompt 对齐的数据集类 ======================
class DualPromptDataset(Dataset):
    """ 分离 Tokenize: 对 Response 和 Prompt 单独编码然后组合，确保标签绝对对齐 """
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
        response = item[self.config["response_key"]]

        # === 独立提取并 Tokenize Response ===
        response_text = response + self.tokenizer.eos_token
        response_ids = self.tokenizer(response_text, return_tensors="pt")["input_ids"].squeeze(0)

        # 预留 Response 的最大安全长度
        max_response_len = self.ctx_len // 2
        if len(response_ids) > max_response_len:
            response_ids = response_ids[:max_response_len]
        
        max_prompt_len = self.ctx_len - len(response_ids)

        # === 构建 Student Prompt ===
        default_s_prompt = "You are a professional assistant." if self.config.get("language") == "en" else "你是一个专业的助手。"
        s_system = self.config.get("student_system_prompt", default_s_prompt)
        s_messages = [{"role": "system", "content": s_system}, {"role": "user", "content": instruction}]
        s_prompt_text = self.tokenizer.apply_chat_template(s_messages, tokenize=False, add_generation_prompt=True)
        s_prompt_ids = self.tokenizer(s_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        # === 构建 Teacher Prompt (带 RAG) ===
        if self.args.enable_rag_teacher:
            retrieved_context = self.rag_manager.search(instruction, top_k=self.args.rag_top_k)
            default_t_prompt = "Please answer based on the references:\n{rag_context}" if self.config.get("language") == "en" else "请根据资料解答：\n{rag_context}"
            t_system_template = self.config.get("teacher_system_prompt", default_t_prompt)
            t_system = t_system_template.format(rag_context=retrieved_context)
        else:
            t_system = s_system

        t_messages = [{"role": "system", "content": t_system}, {"role": "user", "content": instruction}]
        t_prompt_text = self.tokenizer.apply_chat_template(t_messages, tokenize=False, add_generation_prompt=True)
        t_prompt_ids = self.tokenizer(t_prompt_text, return_tensors="pt")["input_ids"].squeeze(0)

        # === 对长出的 Prompt 进行安全的左侧截断 ===
        if len(s_prompt_ids) > max_prompt_len:
            s_prompt_ids = s_prompt_ids[-max_prompt_len:]
        if len(t_prompt_ids) > max_prompt_len:
            t_prompt_ids = t_prompt_ids[-max_prompt_len:]

        # === 拼接组装 ===
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

    # 保留原始的未拼接 Prompt tensors 以供在线训练时 Rollout 生成
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

    # 保险校验：确保对齐
    if N != M:
        raise RuntimeError(f"Teacher and Student valid tokens mismatch: {N} vs {M}. Check dataset tokenization logic.")

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

# ====================== 验证函数 ======================
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
            s_logits = student_model(s_input_ids).logits if hasattr(student_model, 'logits') else student_model(s_input_ids)[0]
            
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

# ====================== 主训练流程 ======================
def main():
    args = parse_args()
    os.makedirs(args.opd_checkpoint_dir, exist_ok=True)
    device = args.device

    # === 新增：初始化分布式进程组以兼容 Muon 优化器 ===
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        # 单卡/单进程训练下使用 'gloo' 后端完成初始化
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        print("💡 [System] 已自动初始化单进程 Gloo 分布式环境，以兼容 Muon 优化器。")

    print("\n" + "="*60)
    print("🚀 正在启动 Agent OPD 蒸馏训练")
    print(f"👨‍🏫 Teacher 模型: {args.hf_teacher_model} (外挂 RAG)")
    print(f"👶 Student 模型: {args.student_model_name} (无 RAG, 在线 Rollout 比率={args.rollout_ratio})")
    print("="*60 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
 
    # >>> 动态将 Tiny-R2 的词表大小与 Tokenizer 保持一致 <<<
    if TINY_R2_AVAILABLE:
        config.vocab_size = len(tokenizer)
        print(f"🔧 已将 Tiny-R2 词表大小 (vocab_size) 动态同步为: {config.vocab_size}")

    # === 1. 数据集加载与自适应解析 ===
    if args.custom_qa_path and os.path.exists(args.custom_qa_path):
        print(f"📂 正在加载本地自定义数据集: {args.custom_qa_path}")
        ds = load_dataset("json", data_files=args.custom_qa_path, split="train")
        
        # 尝试自适应提取常见 Key
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
    else:
        print(f"📡 正在从 HuggingFace 加载公共数据集: {args.dataset}")

    dataset_config = DATASET_CONFIGS.get(args.dataset)
    if not dataset_config:
        raise ValueError(f"Dataset {args.dataset} not found in configs.")

    if args.dataset != "custom":
        if dataset_config.get("hf_subset"):
            ds = load_dataset(dataset_config["hf_path"], dataset_config["hf_subset"], split=dataset_config["split"])
        else:
            ds = load_dataset(dataset_config["hf_path"], split=dataset_config["split"])

    ds = ds.shuffle(seed=42).select(range(min(5000, len(ds))))
    
    rag_corpus = []
    if args.enable_rag_teacher:
        print("📥 正在从数据集中提取知识，充当 RAG 背景语料...")
        rag_corpus = [item[dataset_config["response_key"]] for item in ds.select(range(min(1000, len(ds))))]

    rag_manager = MedicalRAGManager(
        use_rag=args.enable_rag_teacher, 
        corpus_texts=rag_corpus, 
        language=dataset_config.get("language", "zh")
    )

    val_size = int(len(ds) * 0.1)
    train_dataset = DualPromptDataset(ds.select(range(val_size, len(ds))), tokenizer, args.ctx_len, dataset_config, rag_manager, args)
    val_dataset = DualPromptDataset(ds.select(range(val_size)), tokenizer, args.ctx_len, dataset_config, rag_manager, args)

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

    # === 2. 加载学生模型与初始权重加载 ===
    if args.student_model_name.lower() == "tiny-r2" and TINY_R2_AVAILABLE:
        print("\n🤖 加载本地 Tiny-R2 Transformer 作为学生模型")
        student_model = Transformer().to(device)
        optimizers = student_model.configure_optimizers(args.weight_decay, args.lr, device)
    else:
        print(f"\n🤖 加载 HuggingFace 模型作为学生: {args.student_model_name}")
        student_model = AutoModelForCausalLM.from_pretrained(args.student_model_name, torch_dtype=compute_dtype, device_map=device)
        optimizers = [torch.optim.AdamW(student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)]

    if args.student_ckpt:
        if os.path.exists(args.student_ckpt):
            print(f"🔄 正在加载指定的学生权重初始状态: {args.student_ckpt}")
            try:
                ckpt = torch.load(args.student_ckpt, map_location=device)
                student_model.load_state_dict(ckpt, strict=False)
                print("✅ 权重加载成功！")
            except Exception as e:
                print(f"⚠️ 权重加载失败: {e}，将采用模型原初权重开始训练。")
        else:
            print(f"⚠️ 未找到路径：{args.student_ckpt}，将采用模型原初权重开始训练。")

    scheduler = CosineAnnealingLR(optimizers[0], T_max=args.max_iters)

    print(f"\n👨‍🏫 加载教师模型: {args.hf_teacher_model}")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.hf_teacher_model, torch_dtype=compute_dtype, device_map=device).eval()
    for p in teacher_model.parameters():
        p.requires_grad = False

    ctx = torch.amp.autocast(device_type="cuda", dtype=compute_dtype) if "cuda" in device else nullcontext()
    scaler = amp.GradScaler(enabled=use_scaler)

    global_step = 0
    best_val_loss = float("inf")
    train_iter = iter(train_loader)
    
    student_model.train()
    print("\n🔥 开始双路Agent OPD 蒸馏...")
    
    while global_step < args.max_iters:
        student_model.zero_grad()
        total_train_loss = 0.0
        
        # 动态渐进式退火计算
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
                
                # 逐样本动态 Rollout 生成
                for i in range(len(batch["s_prompt_ids"])):
                    s_p_ids = batch["s_prompt_ids"][i].to(device)
                    t_p_ids = batch["t_prompt_ids"][i].to(device)
                    
                    # === 适配器：智能判断是自定义 Tiny-R2 还是 HuggingFace 模型 ===
                    is_custom_transformer = (student_model.__class__.__name__ == "Transformer")
                    
                    with torch.no_grad():
                        if is_custom_transformer:
                            # 1. 本地 Tiny-R2 Transformer 模型生成逻辑
                            gen_out = student_model.generate(
                                idx=s_p_ids.unsqueeze(0),
                                max_new_tokens=args.ctx_len // 2,
                                temperature=args.temperature
                            )
                            # Tiny-R2 的 generate() 返回元组 (generated_ids, kv_cache_size_gb)
                            generated_ids = gen_out[0]
                        else:
                            # 2. HuggingFace 标准模型生成逻辑
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
                s_logits = student_model(s_input_ids).logits if hasattr(student_model, 'logits') else student_model(s_input_ids)[0]
                
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

            # REINFORCE 策略梯度废话惩罚逻辑
            if use_rollout and batch_penalties is not None:
                avg_penalty = batch_penalties.mean().item()
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

        if global_step % args.val_freq == 0:
            val_loss = validate(student_model, teacher_model, val_loader, args, ctx)
            print(f"Step {global_step:04d} | Rollout: {'✅' if use_rollout else '❌'} | Train Loss: {total_train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.opd_checkpoint_dir, f"student_best_model_step_{global_step}.pt")
                torch.save(student_model.state_dict(), save_path)
                print(f"✅ 保存新模型: {save_path}")
                
                # 修复原保存清理代码中对 args.save_dir 的未定义引用
                if args.save_best_only:
                    for old_file in glob.glob(os.path.join(args.opd_checkpoint_dir, "student_best_model_step_*.pt")):
                        if old_file != save_path:
                            try:
                                os.remove(old_file)
                                print(f"🧹 已清理旧检查点: {os.path.basename(old_file)}")
                            except Exception: pass

if __name__ == "__main__":
    main()
