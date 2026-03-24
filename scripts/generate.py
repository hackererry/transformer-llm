#!/usr/bin/env python
"""
文本生成脚本
使用训练好的模型生成文本
"""
import os
import sys
import argparse
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModelConfig, CausalLMModel
from src.data import get_tokenizer
from src.training import load_model
from src.utils import set_seed, print_device_info


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="文本生成")

    # 模型参数
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Tokenizer路径")

    # 生成参数
    parser.add_argument("--prompt", type=str, default="", help="输入提示")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k采样")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p采样")
    parser.add_argument("--do_sample", action="store_true", default=True, help="是否采样")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="返回序列数")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--interactive", action="store_true", help="交互模式")

    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, tokenizer_path: str = None):
    """加载模型和tokenizer"""
    # 加载配置
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = ModelConfig.from_dict(config_dict)
    else:
        config = ModelConfig.tiny()

    # 创建模型
    model = CausalLMModel(config)

    # 加载权重 - 检查多个可能的路径
    # 1. model_path/final_model.pt
    # 2. model_path/final_model/pytorch_model.bin
    # 3. model_path/final_model/pytorch_model.bin
    loaded = False

    possible_files = [
        os.path.join(model_path, "final_model.pt"),
        os.path.join(model_path, "final_model", "pytorch_model.bin"),
        os.path.join(model_path, "pytorch_model.bin"),
    ]

    for model_file in possible_files:
        # 替换反斜杠为正斜杠
        model_file = model_file.replace("\\", "/")
        if os.path.exists(model_file) and os.path.getsize(model_file) > 0:
            print(f"Loading weights from: {model_file}")
            state = torch.load(model_file, map_location="cpu")
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                # 直接保存的state_dict
                model.load_state_dict(state, strict=False)
            loaded = True
            break

    if not loaded:
        print("Warning: No model weights found!")

    model.eval()

    # 加载tokenizer
    tokenizer_path = tokenizer_path or model_path
    # 优先查找 tokenizer.json（HuggingFace格式）
    tokenizer = None
    for tp in [tokenizer_path, os.path.join(tokenizer_path, "tokenizer"), os.path.join(tokenizer_path, "final_model")]:
        if tp:
            tp = tp.replace("\\", "/")
            tokenizer_file = os.path.join(tp, "tokenizer.json")
            if os.path.exists(tokenizer_file) and os.path.getsize(tokenizer_file) > 0:
                try:
                    tokenizer = get_tokenizer(tp, tokenizer_type="bpe", use_fast=True)
                    print(f"Loaded tokenizer from {tp}")
                    break
                except Exception as e:
                    print(f"Failed to load tokenizer from {tp}: {e}")

    if tokenizer is None:
        print("Warning: No tokenizer found, using default")
        tokenizer = get_tokenizer(None, tokenizer_type="bpe")

    return model, tokenizer, config


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    """生成文本"""
    # 编码输入
    if prompt:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids])
    else:
        input_ids = torch.tensor([[tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else 1]])

    # 生成
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else None,
            pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
        )

    # 解码
    generated_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

    return generated_text


def interactive_mode(model, tokenizer, args):
    """交互模式"""
    print("\n" + "=" * 50)
    print("Interactive Mode - Type 'quit' or 'exit' to stop")
    print("=" * 50 + "\n")

    while True:
        try:
            prompt = input("User: ").strip()

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if not prompt:
                continue

            print("\nAssistant: ", end="")
            generated = generate_text(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )

            # 只打印生成的部分（去掉prompt）
            if prompt in generated:
                response = generated[len(prompt):].strip()
            else:
                response = generated.strip()

            print(response)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 打印设备信息
    print_device_info()

    # 加载模型
    print(f"Loading model from {args.model_path}...")
    model, tokenizer, config = load_model_and_tokenizer(
        args.model_path,
        args.tokenizer_path,
    )

    print(f"Model loaded. Vocab size: {config.vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 交互模式
    if args.interactive:
        interactive_mode(model, tokenizer, args)
    else:
        # 单次生成
        print("\n" + "=" * 50)
        print("Prompt:", args.prompt)
        print("=" * 50)

        for i in range(args.num_return_sequences):
            generated = generate_text(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=args.do_sample,
            )

            print(f"\n--- Generated {i + 1} ---")
            print(generated)


if __name__ == "__main__":
    main()
