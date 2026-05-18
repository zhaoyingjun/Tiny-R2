#!/usr/bin/env python3
"""
Tiny-R2 OPD 训练效果验证脚本
对比: 
1. 教师模型 (9B + RAG)
2. 初始学生模型 (0.8B Baseline, 无 RAG)
3. OPD蒸馏后的学生模型 (0.8B OPD, 无 RAG)

支持: 内置数据集评估 & 外部自定义数据集加载
"""

import os
import json
import argparse
import gc
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ====================== RAG 和 评估指标库 ======================
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("⚠️ 提示: 未安装 faiss/sentence_transformers，将使用 Mock RAG。")

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("⚠️ 提示: 未安装 nltk 或 rouge_score，定量指标将被跳过。")


# 内置数据集配置 (当不使用外部数据集时默认使用)
DATASET_CONFIGS = {
    "medquad": {
        "hf_path": "keivalya/MedQuad-MedicalQnADataset",
        "split": "train",
        "language": "en",
        "instruction_key": "Question",
        "response_key": "Answer",
        "student_system_prompt": "You are a professional medical assistant. Please provide accurate, safe, and helpful answers to the patient's questions.",
        "teacher_system_prompt": "You are an authoritative medical expert. Please answer the patient's question strictly based on the following [Authoritative Medical Reference].\n\n[Authoritative Medical Reference]\n{rag_context}"
    },
    "cmeqa": {
        "hf_path": "blcu-nlp/CMeQA",
        "split": "train",
        "language": "zh",
        "instruction_key": "question",
        "response_key": "answer",
        "student_system_prompt": "你是一个专业的全科医生助手。请专业、准确、安全地解答患者的问题。",
        "teacher_system_prompt": "你是一个权威的主治医师评审。请根据以下【权威医疗参考资料】，专业、严谨地解答患者的问题。\n\n[权威医疗参考资料]\n{rag_context}"
    }
}

class MedicalRAGManager:
    # 与训练脚本一致，用于给 Teacher 提供知识支持
    def __init__(self, corpus_texts=None, language="en"):
        self.corpus = corpus_texts if corpus_texts else []
        self.language = language
        if RAG_AVAILABLE and len(self.corpus) > 0:
            model_name = 'sentence-transformers/all-MiniLM-L6-v2' if language == "en" else 'shibing624/text2vec-base-chinese'
            self.embedder = SentenceTransformer(model_name)
            embeddings = self.embedder.encode(self.corpus, show_progress_bar=False)
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)

    def search(self, query: str, top_k: int = 2) -> str:
        if not RAG_AVAILABLE or not hasattr(self, 'index'):
            return "According to medical guidelines, consult a doctor." if self.language == "en" else "据医学权威资料显示，建议遵循医嘱。"
        q_emb = self.embedder.encode([query])
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        return "\n\n".join([self.corpus[idx] for idx in I[0] if idx < len(self.corpus)])

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OPD Distillation with built-in or external dataset")
    
    # 基础模型与检查点参数
    parser.add_argument("--hf_teacher_model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--student_base_model", type=str, default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--student_opd_model", type=str, required=True, help="Path to the trained student .pt file")
    
    # 内置数据集选择
    parser.add_argument("--dataset", type=str, default="medquad", choices=DATASET_CONFIGS.keys(), help="内置数据集名称 (仅当未指定外部数据集时生效)")
    
    # === 新增：外部数据集支持参数 ===
    parser.add_argument("--ext_dataset_path", type=str, default=None, help="外部数据集文件路径 (如: data.json, data.csv)。如果指定此项，则忽略 --dataset 参数")
    parser.add_argument("--ext_dataset_type", type=str, default="json", choices=["json", "csv", "parquet"], help="外部数据集格式类型 (默认: json)")
    parser.add_argument("--ext_instruction_key", type=str, default="question", help="外部数据集的问题/指令字段名 (默认: instruction)")
    parser.add_argument("--ext_response_key", type=str, default="answer", help="外部数据集的回答字段名 (默认: output)")
    parser.add_argument("--ext_language", type=str, default="zh", choices=["zh", "en"], help="外部数据集语言 (默认: zh)")
    parser.add_argument("--ext_student_prompt", type=str, default="你是一个智能助手。请解答用户的问题。", help="外部数据集的Student System Prompt")
    parser.add_argument("--ext_teacher_prompt", type=str, default="你是一个权威专家。请严格根据以下【参考资料】解答用户的问题。\n\n[参考资料]\n{rag_context}", help="外部数据集的Teacher System Prompt (需包含 {rag_context})")
    
    # 评估与生成参数
    parser.add_argument("--eval_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()

def generate_responses(model, tokenizer, prompts, args):
    """批量或单条生成回复"""
    model.eval()
    responses = []
    with torch.inference_mode():
        for prompt in tqdm(prompts, desc="Generating"):
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,  # 使用较低温度进行确定性评估
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            # 截取生成的回复部分
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            responses.append(response)
    return responses

def calculate_metrics(hypotheses, references, language="en"):
    """计算 ROUGE 和 BLEU"""
    if not METRICS_AVAILABLE:
        return {"rougeL": 0.0, "bleu": 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=(language=="en"))
    smoothie = SmoothingFunction().method1
    
    total_rougeL = 0.0
    total_bleu = 0.0
    
    for hyp, ref in zip(hypotheses, references):
        # ROUGE
        total_rougeL += scorer.score(ref, hyp)['rougeL'].fmeasure
        # BLEU
        ref_tokens = list(ref) if language == "zh" else ref.split()
        hyp_tokens = list(hyp) if language == "zh" else hyp.split()
        total_bleu += sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
        
    return {
        "rougeL": round(total_rougeL / len(hypotheses), 4),
        "bleu": round(total_bleu / len(hypotheses), 4)
    }

def main():
    args = parse_args()
    
    # 1. 判断并加载数据集 (外部优先)
    if args.ext_dataset_path:
        print(f"📂 正在加载外部数据集: {args.ext_dataset_path}")
        config = {
            "language": args.ext_language,
            "instruction_key": args.ext_instruction_key,
            "response_key": args.ext_response_key,
            "student_system_prompt": args.ext_student_prompt,
            "teacher_system_prompt": args.ext_teacher_prompt
        }
        # 使用 data_files 参数加载本地或外部文件
        ds = load_dataset(args.ext_dataset_type, data_files={"train": args.ext_dataset_path}, split="train")
    else:
        print(f"📦 正在加载内置数据集: {args.dataset}")
        config = DATASET_CONFIGS[args.dataset]
        ds = load_dataset(config["hf_path"], split=config["split"])
        
    ds = ds.shuffle(seed=42)
    
    # 数据格式校验
    if config["instruction_key"] not in ds.column_names or config["response_key"] not in ds.column_names:
        raise ValueError(f"❌ 数据集缺少指定的列名！当前数据集列名: {ds.column_names}，"
                         f"期望的 Instruction 列: '{config['instruction_key']}'，"
                         f"期望的 Response 列: '{config['response_key']}'。")
    
    # 2. 准备 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.student_base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 取部分数据充当 RAG 知识库
    rag_corpus = [item[config["response_key"]] for item in ds.select(range(min(1000, len(ds))))]
    rag_manager = MedicalRAGManager(corpus_texts=rag_corpus, language=config["language"])
    
    # 取未见过的测试数据 (或者随机抽取 N 条评估)
    eval_ds = ds.select(range(min(args.eval_samples, len(ds))))
    
    questions = [item[config["instruction_key"]] for item in eval_ds]
    ground_truths = [item[config["response_key"]] for item in eval_ds]
    
    # 3. 构建 Prompt
    student_prompts = []
    teacher_prompts = []
    for q in questions:
        # 学生 Prompt (无 RAG)
        s_msgs = [{"role": "system", "content": config["student_system_prompt"]}, {"role": "user", "content": str(q)}]
        student_prompts.append(tokenizer.apply_chat_template(s_msgs, tokenize=False, add_generation_prompt=True))
        
        # 教师 Prompt (带 RAG)
        context = rag_manager.search(str(q), top_k=2)
        t_sys = config["teacher_system_prompt"].format(rag_context=context)
        t_msgs = [{"role": "system", "content": t_sys}, {"role": "user", "content": str(q)}]
        teacher_prompts.append(tokenizer.apply_chat_template(t_msgs, tokenize=False, add_generation_prompt=True))

    results = {"questions": questions, "ground_truths": ground_truths}
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # ================= 顺序推理 (防止显存 OOM) =================
    
    # [模型 1] Base Student (未经过 OPD 蒸馏的 0.8B)
    print(f"\n[{'='*20} 正在评估 Base Student (0.8B) {'='*20}]")
    base_model = AutoModelForCausalLM.from_pretrained(args.student_base_model, torch_dtype=compute_dtype, device_map=args.device)
    results["base_student"] = generate_responses(base_model, tokenizer, student_prompts, args)
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # [模型 2] OPD Student (经过 OPD 蒸馏的 0.8B)
    print(f"\n[{'='*20} 正在评估 OPD Student (0.8B) {'='*20}]")
    opd_model = AutoModelForCausalLM.from_pretrained(args.student_base_model, torch_dtype=compute_dtype, device_map=args.device)
    print(f"📥 正在加载 OPD 权重: {args.student_opd_model}")
    opd_model.load_state_dict(torch.load(args.student_opd_model, map_location=args.device), strict=False)
    results["opd_student"] = generate_responses(opd_model, tokenizer, student_prompts, args)
    del opd_model
    torch.cuda.empty_cache()
    gc.collect()

    # [模型 3] Teacher (带 RAG 的 9B 大模型)
    print(f"\n[{'='*20} 正在评估 Teacher (9B + RAG) {'='*20}]")
    teacher_model = AutoModelForCausalLM.from_pretrained(args.hf_teacher_model, torch_dtype=compute_dtype, device_map=args.device)
    results["teacher_rag"] = generate_responses(teacher_model, tokenizer, teacher_prompts, args)
    del teacher_model
    torch.cuda.empty_cache()
    gc.collect()

    # ================= 计算指标与打印报告 =================
    print("\n" + "="*50)
    print("📈 评估指标对比 (以 Teacher+RAG 的回答作为金标准参考)")
    print("目标: OPD Student 分数应显著高于 Base Student，证明其吸收了Teacher特征")
    print("="*50)
    
    base_vs_teacher = calculate_metrics(results["base_student"], results["teacher_rag"], config["language"])
    opd_vs_teacher = calculate_metrics(results["opd_student"], results["teacher_rag"], config["language"])
    
    print(f"Base Student 🆚 Teacher -> ROUGE-L: {base_vs_teacher['rougeL']:.4f} | BLEU: {base_vs_teacher['bleu']:.4f}")
    print(f" OPD Student 🆚 Teacher -> ROUGE-L: {opd_vs_teacher['rougeL']:.4f} | BLEU: {opd_vs_teacher['bleu']:.4f}")
    
    # ================= 直观的定性打印对比 =================
    print("\n" + "="*50)
    print("🔍 随机抽样定性对比展示")
    print("="*50)
    import random
    sample_idx = random.randint(0, args.eval_samples - 1)
    
    print(f"【患者问题】:\n{results['questions'][sample_idx]}\n")
    print(f"【Teacher (9B+RAG) 权威回复】:\n{results['teacher_rag'][sample_idx]}\n")
    print(f"【Base Student (未经训练 0.8B) 回复】:\n{results['base_student'][sample_idx]}\n")
    print(f"【✅ OPD Student (蒸馏后 0.8B) 回复】:\n{results['opd_student'][sample_idx]}\n")

    # 保存全量结果
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 详细评估结果已保存至: {args.output_file}")

if __name__ == "__main__":
    main()
