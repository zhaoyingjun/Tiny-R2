import torch
import torch.nn as nn
import sys
import os
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import argparse
# 添加项目路径
sys.path.insert(0, "/content/Tiny-R2")

import config
from model import Transformer

def load_model(checkpoint_path, device='cuda'):
    """
    加载模型（适配你的检查点格式）
    """
    print(f"Loading model from {checkpoint_path}...")
    
    # 实例化模型
    model = Transformer()
    model.to(device)
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 提取 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"Found 'model' key in checkpoint, step: {checkpoint.get('step', 'unknown')}")
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 清理前缀（来自你的推理脚本）
    unwanted_prefixes = ['_orig_mod.', 'module.']
    for prefix in unwanted_prefixes:
        for k in list(state_dict.keys()):
            if k.startswith(prefix):
                state_dict[k[len(prefix):]] = state_dict.pop(k)
    
    # 检查词表大小
    output_head_key = 'lm_head.weight'
    if output_head_key in state_dict:
        ckpt_vocab_size = state_dict[output_head_key].shape[0]
        print(f"Checkpoint vocab size: {ckpt_vocab_size}")
        # 如果与 config 不匹配，可能需要调整
        if ckpt_vocab_size != config.vocab_size:
            print(f"Warning: vocab size mismatch (config: {config.vocab_size}, checkpoint: {ckpt_vocab_size})")
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        # 只显示前5个
        for k in missing_keys[:5]:
            print(f"  - {k}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        for k in unexpected_keys[:5]:
            print(f"  - {k}")
    
    model.eval()
    print("Model loaded successfully.\n")
    return model

def get_tokenizer():
    """
    加载分词器（优先使用本地，否则 GPT2）
    """
    # 尝试加载本地 tokenizer
    tokenizer_paths = [
        "/content/Tiny-R2/tokenizer.json",
        "/content/Tiny-R2/tokenizer",
        "tokenizer.json"
    ]
    
    for path in tokenizer_paths:
        if os.path.exists(path):
            try:
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(path)
                print(f"Loaded local tokenizer from {path}")
                # 包装成类似 HuggingFace 的接口
                class TokenizerWrapper:
                    def __init__(self, tokenizer):
                        self.tokenizer = tokenizer
                        self.eos_token = "<|endoftext|>"
                        self.pad_token = self.eos_token
                    
                    def encode(self, text):
                        return self.tokenizer.encode(text).ids
                    
                    def decode(self, ids):
                        return self.tokenizer.decode(ids)
                    
                    def __call__(self, text, return_tensors=None):
                        ids = self.encode(text)
                        if return_tensors == "pt":
                            return {"input_ids": torch.tensor([ids])}
                        return {"input_ids": ids}
                
                return TokenizerWrapper(tokenizer)
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    # 回退到 GPT2
    print("Using GPT-2 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@torch.no_grad()
def evaluate_wikitext103(model, tokenizer, device='cuda', max_samples=None, ctx_len=None):
    """
    在 WikiText-103 上计算困惑度
    """
    if ctx_len is None:
        ctx_len = getattr(config, 'ctx_len', 1536)
    
    print(f"Loading WikiText-103 dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Evaluating on {max_samples} samples...")
    else:
        print(f"Evaluating on full test set ({len(dataset)} samples)...")
    
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    # 滑动窗口参数
    max_length = ctx_len
    stride = ctx_len // 2  # 50% 重叠
    
    print(f"Context length: {max_length}, stride: {stride}\n")
    
    for i, text in enumerate(tqdm(dataset["text"], desc="Evaluating")):
        if len(text.strip()) == 0:
            continue
        
        # 编码文本
        if hasattr(tokenizer, 'encode'):
            input_ids = tokenizer.encode(text)
        else:
            input_ids = tokenizer(text, return_tensors="pt")["input_ids"][0].tolist()
        
        if len(input_ids) < 2:
            continue
        
        # 转换为 tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
        seq_len = input_ids.size(0)
        
        # 滑动窗口处理
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            # 获取当前块
            chunk_ids = input_ids[begin_loc:end_loc].unsqueeze(0)
            
            # 前向传播（自定义 Transformer 接口）
            try:
                # 你的模型 forward 返回 logits
                logits = model(chunk_ids)
                
                # 处理可能的元组输出（如 (logits, kv_cache)）
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # 计算 next-token prediction loss
                # logits: [1, seq_len, vocab_size]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = chunk_ids[:, 1:].contiguous()
                
                # 只计算新部分的 loss（避免重复计算）
                valid_len = min(trg_len - 1, shift_logits.size(1))
                if valid_len <= 0:
                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break
                    continue
                
                # 取最后 valid_len 个 token 的预测
                shift_logits = shift_logits[:, -valid_len:, :]
                shift_labels = shift_labels[:, -valid_len:]
                
                # CrossEntropy
                loss_fct = nn.CrossEntropyLoss(reduction='sum')
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                
                total_loss += loss.item()
                total_tokens += valid_len
                
            except Exception as e:
                print(f"\nError at sample {i}, position {begin_loc}-{end_loc}: {e}")
                print(f"Chunk shape: {chunk_ids.shape}")
                raise
            
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
    
    # 计算指标
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    bits_per_char = avg_loss / 0.6931471805599453  # log(2)
    
    return {
        "perplexity": perplexity.item(),
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "bits_per_char": bits_per_char
    }

def quick_test(model, tokenizer, device='cuda'):
    """快速测试模型"""
    print("Running quick test...")
    test_text = "The quick brown fox jumps over the lazy dog."
    
    if hasattr(tokenizer, 'encode'):
        input_ids = torch.tensor([tokenizer.encode(test_text)], dtype=torch.long, device=device)
    else:
        input_ids = tokenizer(test_text, return_tensors="pt")["input_ids"].to(device)
    
    with torch.no_grad():
        output = model(input_ids)
        if isinstance(output, tuple):
            output = output[0]
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output type: {type(output)}")
        print("  Test passed!\n")

def main():
    # 配置
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="模型权重路径")
    parser.add_argument('--device', type=str, default='cuda', help="推理设备")
    args = parser.parse_args()

    DEVICE = args.device
    CHECKPOINT_PATH=args.checkpoint

    
    print(f"Using device: {DEVICE}\n")
    
    # 加载模型
    model = load_model(CHECKPOINT_PATH, device=DEVICE)
    
    # 加载分词器
    tokenizer = get_tokenizer()
    
    # 快速测试
    quick_test(model, tokenizer, device=DEVICE)
    
    # 评估（可以先测试少量样本）
    # max_samples=100 用于快速验证，None 表示完整评估
    results = evaluate_wikitext103(
        model, 
        tokenizer, 
        device=DEVICE,
        max_samples=None,  # 设为 100 先测试
        ctx_len=getattr(config, 'ctx_len', 1536)
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("WikiText-103 Evaluation Results")
    print("="*60)
    print(f"Perplexity:           {results['perplexity']:.2f}")
    print(f"Average Loss:         {results['avg_loss']:.4f}")
    print(f"Bits per Character:   {results['bits_per_char']:.4f}")
    print(f"Total Tokens:         {results['total_tokens']:,}")
    print("="*60)
    
    # 对比参考值
    print("\nReference values:")
    print(f"  GPT-2 Small (124M): ~16.3 PPL")
    print(f"  GPT-2 Medium (345M): ~12.0 PPL")
    print(f"  GPT-2 Large (774M): ~10.6 PPL")
    
    # 保存结果
    import json
    output_file = "wikitext103_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
