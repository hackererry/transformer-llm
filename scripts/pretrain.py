#!/usr/bin/env python
"""
预训练脚本
自动检测GPU/CPU，仅支持预处理数据
"""
import os
import sys
import argparse
import json

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import ModelConfig, CausalLMModel
from src.data import (
    get_tokenizer,
    get_collator,
    ShardedPreprocessedDataset,
)
from src.training import Trainer, TrainingConfig, save_pretrained
from src.utils import set_seed, print_device_info, get_memory_info, OptimizationProfiler


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="预训练语言模型")

    # 数据参数（仅支持预处理数据）
    parser.add_argument("--preprocessed_data", type=str, required=True,
                        help="预处理数据目录路径（必需）")

    # 模型参数
    parser.add_argument("--model_path", type=str, default=None,
                        help="预训练模型路径（用于继续训练）")
    parser.add_argument("--model_config", type=str, default="small",
                        choices=["tiny", "small", "medium", "moe_small", "moe_medium"],
                        help="模型配置: tiny(~1M)/small(~6M)/medium(~50M) 使用FFN, moe_small/moe_medium 使用MoE")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大步数(-1=不限制)")

    # 学习率调度
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine",
                        choices=["cosine", "linear", "constant"], help="调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")

    # GPU优化（仅在GPU时有效）
    parser.add_argument("--bf16", action="store_true", default=None,
                        help="使用BF16精度(自动检测时可省略)")
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="使用FP16精度")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载进程数")

    # 模型优化参数（覆盖 ModelConfig 默认值）
    parser.add_argument("--no_flash_attention", action="store_true",
                        help="禁用Flash Attention（默认启用）")
    parser.add_argument("--no_gqa", action="store_true",
                        help="禁用GQA分组查询注意力（默认启用）")
    parser.add_argument("--rope_scaling_factor", type=float, default=4.0,
                        help="YaRN长度外推因子（默认4.0，支持4倍外推）")

    # MoE 配置参数（默认启用）
    parser.add_argument("--no_moe", action="store_true",
                        help="禁用 MoE，使用标准 FFN（默认启用 MoE）")
    parser.add_argument("--num_experts", type=int, default=8,
                        help="专家数量（默认8）")
    parser.add_argument("--num_experts_per_tok", type=int, default=2,
                        help="每个 token 激活的专家数（默认2）")
    parser.add_argument("--aux_loss_alpha", type=float, default=0.01,
                        help="MoE 负载均衡损失系数（默认0.01）")

    # MLA 配置参数（默认启用）
    parser.add_argument("--no_mla", action="store_true",
                        help="禁用 MLA，使用标准 Attention（默认启用 MLA）")
    parser.add_argument("--kv_lora_rank", type=int, default=512,
                        help="MLA KV 压缩维度（默认512）")
    parser.add_argument("--q_lora_rank", type=int, default=1536,
                        help="MLA Q 压缩维度（默认1536）")

    # 日志和保存
    parser.add_argument("--logging_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="检查点数量限制")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="恢复训练")

    return parser.parse_args()


def detect_device_and_configure(args):
    """自动检测设备并配置精度参数"""
    if torch.cuda.is_available():
        device_type = "cuda"
        capability = torch.cuda.get_device_capability()
        gpu_name = torch.cuda.get_device_name(0)

        print(f"\n{'='*60}")
        print(f"  GPU Detected: {gpu_name}")
        print(f"  Compute Capability: {capability[0]}.{capability[1]}")
        print(f"{'='*60}")

        # 自动选择精度
        if args.bf16 is True:
            precision = "BF16"
        elif args.fp16 is True:
            precision = "FP16"
        elif capability[0] >= 8:  # Ampere+ (RTX 30/40 series)
            precision = "BF16"
            args.bf16 = True
            args.fp16 = False
        else:
            precision = "FP16"
            args.bf16 = False
            args.fp16 = True

        print(f"  Using {precision} precision")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'='*60}")
        print(f"  No GPU detected, using CPU")
        print(f"{'='*60}\n")
        args.bf16 = False
        args.fp16 = False


def load_or_create_model(args, config):
    """加载或创建模型"""
    model = None

    if args.model_path and os.path.exists(args.model_path):
        # 从已有模型加载
        config_path = os.path.join(args.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = ModelConfig.from_dict(config_dict)

        model = CausalLMModel(config)
        model_file = os.path.join(args.model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            state = torch.load(model_file, map_location="cpu", weights_only=False)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
    else:
        # 创建新模型
        model = CausalLMModel(config)

    return model, config


def load_datasets(args):
    """加载训练和验证数据集（仅支持预处理数据）"""
    # 使用预处理数据
    train_dataset = ShardedPreprocessedDataset(
        data_dir=args.preprocessed_data,
        prefix="train",
        shuffle=False,
    )

    eval_dataset = None
    val_shard_path = os.path.join(args.preprocessed_data, "val_000.pt")
    if os.path.exists(val_shard_path):
        eval_dataset = ShardedPreprocessedDataset(
            data_dir=args.preprocessed_data,
            prefix="val",
            shuffle=False,
        )

    return train_dataset, eval_dataset


def load_tokenizer(args):
    """加载tokenizer（从预处理数据目录）"""
    # 从预处理数据目录加载tokenizer
    tokenizer_path = os.path.join(args.preprocessed_data, "tokenizer")
    if os.path.exists(tokenizer_path):
        tokenizer = get_tokenizer(tokenizer_path, tokenizer_type="bpe", use_fast=True)
        print(f"Loaded tokenizer from {tokenizer_path}")
        return tokenizer, tokenizer.vocab_size
    else:
        raise FileNotFoundError(f"Tokenizer not found in {tokenizer_path}")


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 自动检测设备并配置
    detect_device_and_configure(args)

    # 打印设备信息
    print_device_info()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Starting Pretraining")
    print("=" * 60)

    # 验证输入（仅支持预处理数据）
    if not args.preprocessed_data:
        print("Error: 必须提供 --preprocessed_data")
        return

    # 加载tokenizer并获取词汇表大小
    tokenizer, vocab_size = load_tokenizer(args)

    # 从预处理数据获取max_seq_length
    dataset_info_path = os.path.join(args.preprocessed_data, "dataset_info.json")
    max_seq_length = 512
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            dataset_info = json.load(f)
            max_seq_length = dataset_info.get("max_seq_length", 512)

    # 模型配置
    if args.model_config == "tiny":
        config = ModelConfig.tiny()
    elif args.model_config == "small":
        config = ModelConfig.small()
    elif args.model_config == "medium":
        config = ModelConfig.medium()
    elif args.model_config == "moe_small":
        config = ModelConfig.moe_small()
    elif args.model_config == "moe_medium":
        config = ModelConfig.moe_medium()
    else:
        config = ModelConfig.small()

    config.vocab_size = vocab_size
    config.max_position_embeddings = max(config.max_position_embeddings, max_seq_length)
    config.gradient_checkpointing = args.gradient_checkpointing

    # 应用模型优化参数（覆盖默认值）
    if args.no_flash_attention:
        config.use_flash_attention = False
    if args.no_gqa:
        config.num_key_value_heads = config.num_attention_heads  # 禁用GQA
    if args.rope_scaling_factor != 4.0:
        config.rope_scaling = {"type": "yarn", "factor": args.rope_scaling_factor}

    # 应用 MoE 配置（默认启用，可禁用）
    if args.no_moe:
        config.use_moe = False
    else:
        config.use_moe = True
        config.num_experts = args.num_experts
        config.num_experts_per_tok = args.num_experts_per_tok
        config.aux_loss_alpha = args.aux_loss_alpha

    # 应用 MLA 配置（默认启用，可禁用）
    if args.no_mla:
        config.use_mla = False
    else:
        config.use_mla = True
        config.kv_lora_rank = args.kv_lora_rank
        config.q_lora_rank = args.q_lora_rank

    print(f"Model config: {config.to_dict()}")

    # 创建优化项性能分析器
    profiler = OptimizationProfiler()

    # 判断是否使用 GQA
    use_gqa = config.num_key_value_heads != config.num_attention_heads

    # 判断是否使用 YaRN
    use_yarn = config.rope_scaling is not None and config.rope_scaling.get("type") == "yarn"
    yarn_factor = config.rope_scaling.get("factor", 1.0) if use_yarn else 1.0

    # 判断混合精度
    if args.bf16:
        mixed_precision = "bf16"
    elif args.fp16:
        mixed_precision = "fp16"
    else:
        mixed_precision = "fp32"

    # 记录优化配置
    profiler.record_optimization_config(
        use_gqa=use_gqa,
        use_flash_attention=config.use_flash_attention,
        use_yarn=use_yarn,
        yarn_factor=yarn_factor,
        use_streaming_llm=config.use_streaming_llm,
        use_speculative_decoding=False,  # 训练时不使用
        gradient_checkpointing=args.gradient_checkpointing,
        mixed_precision=mixed_precision,
    )

    # 记录 GQA 指标
    profiler.record_gqa_metrics(
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        seq_len=max_seq_length,
        batch_size=args.per_device_train_batch_size,
        head_dim=config.head_dim,
        num_layers=config.num_hidden_layers,
    )

    # 记录 Flash Attention 理论估算（如果启用）
    if config.use_flash_attention:
        profiler.record_flash_attention_metrics(
            seq_len=max_seq_length,
            batch_size=args.per_device_train_batch_size,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_layers=config.num_hidden_layers,
            dtype_size=2 if args.bf16 or args.fp16 else 4,
        )

    # 加载或创建模型
    model, config = load_or_create_model(args, config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Memory info: {get_memory_info()}")

    # 加载数据集
    train_dataset, eval_dataset = load_datasets(args)

    print(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")

    # 数据整理器
    collate_fn = get_collator(
        collator_type="causal_lm",
        pad_token_id=tokenizer.pad_token_id,
        max_length=max_seq_length,
    )

    # 创建训练配置
    training_config = TrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16 or False,
        fp16=args.fp16 or False,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        config=training_config,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
    )

    # 开始训练
    results = trainer.train()

    print(f"\nTraining completed!")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Global step: {results['global_step']}")
    if 'training_time' in results:
        print(f"Training time: {results['training_time']/3600:.2f} hours")

    # 打印优化项性能报告
    profiler.print_report()

    # 保存性能报告到文件
    profiler.save_report(
        output_dir="logs/perf",
        prefix="pretrain",
        training_results=results,
        model_config=config.to_dict(),
    )

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model")
    save_pretrained(model, final_model_path, tokenizer)
    print(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
