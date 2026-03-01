import torch
import argparse
from model import Transformer  # 假设你的模型保存为 model.py
import config
from transformers import AutoTokenizer

def load_model(checkpoint_path, device='cuda'):
    """
    加载模型权重
    """
    print(f"Loading model from {checkpoint_path}...")
    model = Transformer()
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    #print(checkpoint['model'])
    
    # 如果是 state_dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
          if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        unwanted_prefix_ddp = 'module.'
        for k,v in list(state_dict.items()):
          if k.startswith(unwanted_prefix_ddp):
            state_dict[k[len(unwanted_prefix_ddp):]] = state_dict.pop(k)
        
        output_head_key = 'lm_head.weight' # Adjust if your output layer name is different
        if output_head_key in state_dict:
          ckpt_vocab_size = state_dict[output_head_key].shape[0]
        

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()  # 推理模式
    print("Model loaded successfully.")
    return model

def tokens_from_text(text, tokenizer):
    """
    假设你有 tokenizer，将文本转为 token idx
    """
    return torch.tensor([tokenizer.encode(text)], dtype=torch.long)

def text_from_tokens(tokens, tokenizer):
    """
    将生成的 token idx 转为文本
    """
    return tokenizer.decode(tokens[0].tolist())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help="模型权重路径")
    parser.add_argument('--prompt', type=str, required=True, help="输入文本 prompt")
    parser.add_argument('--max_new_tokens', type=int, default=50, help="生成长度")
    parser.add_argument('--temperature', type=float, default=1.0, help="采样温度")
    parser.add_argument('--top_k', type=int, default=None, help="Top-k 采样")
    parser.add_argument('--top_p', type=float, default=None, help="Top-p 采样概率")
    parser.add_argument('--device', type=str, default='cuda', help="推理设备")
    args = parser.parse_args()

    device = args.device

    # ---- 加载模型 ----
    model = load_model(args.checkpoint, device=device)

    # ---- 假设你有 tokenizer ----
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file("tokenizer.json")  # 修改为你自己的 tokenizer 文件
    except Exception as e:
        print("Tokenizer not found, using GPT-2 tokenizer")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        config.vocab_size = tokenizer.vocab_size

    # ---- 编码 prompt ----
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)

    # ---- 推理生成 ----
    with torch.no_grad():
        generated_ids, kv_cache_size_gb = model.generate(
            idx=input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            tiktoken_vocab_size=None  # 可选
        )

    generated_text = tokenizer.decode(generated_ids[0].tolist())

    print("===== Generated Text =====")
    print(generated_text)
    print(f"KV Cache Size (GB): {kv_cache_size_gb:.4f}")

if __name__ == "__main__":
    main()
