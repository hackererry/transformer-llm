#!/usr/bin/env python
"""
SFT 数据预处理脚本
将 JSON/JSONL 格式的微调数据 tokenize 并保存为 .pt 分片

Usage:
    # 使用已有 tokenizer
    python scripts/preprocess_sft_data.py \
        --train_file data/sft_train.jsonl \
        --output_dir output/sft_preprocessed \
        --template alpaca \
        --tokenizer_path output/preprocessed/tokenizer

    # 训练新 tokenizer
    python scripts/preprocess_sft_data.py \
        --train_file data/sft_train.jsonl \
        --output_dir output/sft_preprocessed \
        --template alpaca \
        --vocab_size 32000
"""
import os
import sys
import argparse
import json
import torch
from typing import List, Dict, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import HuggingFaceBPETokenizer, save_preprocessed_data


def format_prompt(item: Dict, template: str,
                  instruction_column: str = "instruction",
                  input_column: str = "input",
                  output_column: str = "output") -> Tuple[str, str]:
    """格式化 prompt（复用 FinetuneDataset 的模板逻辑）"""
    instruction = item.get(instruction_column, "")
    input_text = item.get(input_column, "")
    output = item.get(output_column, "")

    if template == "alpaca":
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        full_text = prompt + output
    elif template == "chat":
        messages = []
        if input_text:
            messages.append({"role": "user", "content": f"{instruction}\n{input_text}"})
        else:
            messages.append({"role": "user", "content": instruction})
        messages.append({"role": "assistant", "content": output})
        full_text = _format_chat(messages)
        prompt = full_text[:full_text.index(output)]
    else:
        # simple 模板
        prompt = f"Instruction: {instruction}\n"
        if input_text:
            prompt += f"Input: {input_text}\n"
        prompt += "Output: "
        full_text = prompt + output

    return prompt, full_text


def _format_chat(messages: List[Dict]) -> str:
    """格式化对话格式"""
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            formatted += f" Entities\n{content}\n"
        elif role == "assistant":
            formatted += f"<|assistant|)\n{content}\n"
    return formatted


def load_sft_data(file_path: str) -> List[Dict]:
    """加载 JSON/JSONL 数据"""
    if file_path.endswith(".jsonl"):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data
    elif file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use .json or .jsonl")


def process_sft_data(
    data: List[Dict],
    tokenizer,
    max_seq_length: int,
    template: str,
) -> List[Dict]:
    """处理 SFT 数据：格式化、tokenize、创建 labels"""
    examples = []

    for item in data:
        prompt, full_text = format_prompt(item, template)

        # 编码
        full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

        # 截断
        if len(full_tokens) > max_seq_length:
            full_tokens = full_tokens[:max_seq_length]

        # 创建 labels: prompt 部分用 -100 屏蔽
        labels = full_tokens.copy()
        prompt_len = min(len(prompt_tokens), len(full_tokens))
        for i in range(prompt_len):
            labels[i] = -100

        examples.append({
            "input_ids": full_tokens,
            "labels": labels,
            "attention_mask": [1] * len(full_tokens),
        })

    return examples


def save_shards(
    examples: List[Dict],
    output_dir: str,
    prefix: str,
    shard_size: int,
    metadata: Dict,
) -> List[str]:
    """分片保存数据"""
    shard_files = []
    num_shards = (len(examples) + shard_size - 1) // shard_size

    for i in range(num_shards):
        start = i * shard_size
        end = min(start + shard_size, len(examples))
        shard_examples = examples[start:end]

        shard_filename = f"{prefix}_{i:03d}.pt"
        shard_path = os.path.join(output_dir, shard_filename)

        shard_metadata = metadata.copy()
        shard_metadata["num_examples"] = len(shard_examples)

        save_preprocessed_data(shard_examples, shard_path, shard_metadata)
        shard_files.append(shard_filename)

    return shard_files


def parse_args():
    parser = argparse.ArgumentParser(description="SFT 数据预处理")

    # 数据参数
    parser.add_argument("--train_file", type=str, required=True,
                        help="训练数据文件路径 (JSON/JSONL)")
    parser.add_argument("--validation_file", type=str, default=None,
                        help="验证数据文件路径 (可选)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--template", type=str, default="alpaca",
                        choices=["alpaca", "chat", "simple"],
                        help="指令模板类型")

    # Tokenizer 参数
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="已有 tokenizer 路径 (可选，不提供则训练新的)")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="词表大小 (训练新 tokenizer 时使用)")
    parser.add_argument("--min_frequency", type=int, default=2,
                        help="最小词频 (训练新 tokenizer 时使用)")

    # 预处理参数
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--shard_size", type=int, default=10000,
                        help="分片大小")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 50)
    print("SFT Data Preprocessing")
    print("=" * 50)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载或训练 tokenizer
    if args.tokenizer_path:
        print(f"Loading tokenizer from {args.tokenizer_path}")
        tokenizer = HuggingFaceBPETokenizer.load(args.tokenizer_path)
    else:
        print(f"Training new tokenizer (vocab_size={args.vocab_size})")
        tokenizer = HuggingFaceBPETokenizer()

        # 从训练数据中采样文本训练 tokenizer
        print(f"Loading training data from {args.train_file}")
        train_data = load_sft_data(args.train_file)

        texts = []
        for item in train_data:
            _, full_text = format_prompt(item, args.template)
            texts.append(full_text)

        tokenizer.train(texts, vocab_size=args.vocab_size, min_frequency=args.min_frequency)

        # 保存 tokenizer
        tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
        tokenizer.save(tokenizer_dir)
        print(f"Tokenizer saved to {tokenizer_dir}")

    print(f"Tokenizer vocab_size: {tokenizer.vocab_size}")

    # 处理训练数据
    print(f"\nProcessing training data from {args.train_file}")
    train_data = load_sft_data(args.train_file)
    print(f"  Loaded {len(train_data)} examples")

    train_examples = process_sft_data(train_data, tokenizer, args.max_seq_length, args.template)
    print(f"  Processed {len(train_examples)} examples")

    metadata = {
        "max_seq_length": args.max_seq_length,
        "vocab_size": tokenizer.vocab_size,
        "template": args.template,
    }

    # 保存训练分片
    train_shard_files = save_shards(
        train_examples, args.output_dir, "sft_train", args.shard_size, metadata
    )
    print(f"  Saved {len(train_shard_files)} training shards")

    # 处理验证数据
    val_shard_files = []
    if args.validation_file:
        print(f"\nProcessing validation data from {args.validation_file}")
        val_data = load_sft_data(args.validation_file)
        print(f"  Loaded {len(val_data)} examples")

        val_examples = process_sft_data(val_data, tokenizer, args.max_seq_length, args.template)
        print(f"  Processed {len(val_examples)} examples")

        val_shard_files = save_shards(
            val_examples, args.output_dir, "sft_val", args.shard_size, metadata
        )
        print(f"  Saved {len(val_shard_files)} validation shards")

    # 保存 dataset_info.json
    dataset_info = {
        "version": "3.0",
        "config": metadata,
        "files": {},
        "shards": {},
        "summary": {
            "total_train_examples": len(train_examples),
            "total_val_examples": len(val_examples) if args.validation_file else 0,
            "total_shards": len(train_shard_files) + len(val_shard_files),
        },
    }

    for f in train_shard_files:
        shard_examples_count = 0
        shard_path = os.path.join(args.output_dir, f)
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_examples_count = len(data.get("examples", []))
        dataset_info["shards"][f] = {"num_examples": shard_examples_count}

    for f in val_shard_files:
        shard_path = os.path.join(args.output_dir, f)
        data = torch.load(shard_path, map_location="cpu", weights_only=False)
        shard_examples_count = len(data.get("examples", []))
        dataset_info["shards"][f] = {"num_examples": shard_examples_count}

    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"\nDataset info saved to {info_path}")

    # 如果 tokenizer 没有保存过（从已有 tokenizer 加载的情况），也保存一份
    if args.tokenizer_path:
        tokenizer_dir = os.path.join(args.output_dir, "tokenizer")
        if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer.json")):
            tokenizer.save(tokenizer_dir)
            print(f"Tokenizer copied to {tokenizer_dir}")

    print("\n" + "=" * 50)
    print("SFT data preprocessing completed!")
    print(f"  Training examples: {len(train_examples)}")
    if args.validation_file:
        print(f"  Validation examples: {len(val_examples)}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
