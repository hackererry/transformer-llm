#!/usr/bin/env python
"""
GPU优化预训练脚本
针对RTX 4060Ti 8G优化
"""
import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.model import ModelConfig, CausalLMModel
from src.data import PretrainDataset, get_collator, BPETokenizer, get_tokenizer
from src.training.trainer_gpu import GPUTrainer, GPUTrainingConfig, get_gpu_memory_info, estimate_memory_requirements
from src.utils import set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="GPU优化预训练")

    # 数据参数
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件")
    parser.add_argument("--validation_file", type=str, default=None, help="验证数据文件")

    # 模型参数
    parser.add_argument("--model_config", type=str, default="tiny",
                       choices=["tiny", "small", "medium"], help="模型配置")
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型路径(继续训练)")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output/gpu_pretrain", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")

    # 学习率
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")

    # GPU优化
    parser.add_argument("--bf16", action="store_true", default=True, help="使用BF16精度")
    parser.add_argument("--fp16", action="store_false", dest="bf16", help="使用FP16精度")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--use_flash_attention", action="store_true", default=True, help="使用Flash Attention")

    # 数据加载
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="数据加载进程数")

    # 日志
    parser.add_argument("--logging_steps", type=int, default=10, help="日志间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存间隔")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大训练步数")
    parser.add_argument("--vocab_size", type=int, default=32000, help="词表大小")

    return parser.parse_args()


def main():
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    logger = setup_logger(args.output_dir, "gpu_pretrain")

    logger.info("=" * 60)
    logger.info("GPU Optimized Pretraining")
    logger.info("=" * 60)

    # 检查GPU
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! Please use pretrain.py for CPU training.")
        return

    # 打印GPU信息
    gpu_info = get_gpu_memory_info()
    logger.info(f"GPU: {gpu_info['device_name']}")
    logger.info(f"Total Memory: {gpu_info['total_gb']:.1f} GB")
    logger.info(f"Precision: {'BF16' if args.bf16 else 'FP16'}")

    # 创建模型配置
    if args.model_config == "tiny":
        config = ModelConfig.tiny()
    elif args.model_config == "small":
        config = ModelConfig.small()
    else:
        config = ModelConfig.medium()

    config.vocab_size = args.vocab_size
    config.max_position_embeddings = max(config.max_position_embeddings, args.max_seq_length)

    # 估算显存需求
    model_params = sum(p.numel() for p in CausalLMModel(config).parameters())
    memory_est = estimate_memory_requirements(
        model_params=model_params,
        batch_size=args.per_device_train_batch_size,
        seq_length=args.max_seq_length,
        hidden_size=config.hidden_size,
        precision="bf16" if args.bf16 else "fp16",
    )

    logger.info(f"\nMemory Estimation:")
    logger.info(f"  Model parameters: {model_params:,}")
    logger.info(f"  Model memory: {memory_est['model_gb']:.2f} GB")
    logger.info(f"  Gradient memory: {memory_est['gradient_gb']:.2f} GB")
    logger.info(f"  Optimizer memory: {memory_est['optimizer_gb']:.2f} GB")
    logger.info(f"  Activation memory: {memory_est['activation_gb']:.2f} GB")
    logger.info(f"  Total estimated: {memory_est['total_gb']:.2f} GB")
    logger.info(f"  Recommended GPU: {memory_est['recommended_gpu_gb']:.2f} GB")

    # 检查显存是否足够
    if memory_est['recommended_gpu_gb'] > gpu_info['total_gb']:
        logger.warning("\n⚠️  Warning: Estimated memory exceeds GPU capacity!")
        logger.warning("Consider:")
        logger.warning(f"  - Reducing batch_size (current: {args.per_device_train_batch_size})")
        logger.warning(f"  - Enabling gradient_checkpointing")
        logger.warning(f"  - Reducing max_seq_length (current: {args.max_seq_length})")
        logger.warning(f"  - Using smaller model config")

        # 自动调整建议
        if args.per_device_train_batch_size > 1:
            suggested_batch = max(1, int(gpu_info['total_gb'] / memory_est['recommended_gpu_gb'] * args.per_device_train_batch_size))
            logger.warning(f"\n  Suggested batch_size: {suggested_batch}")

    # 创建或加载模型
    if args.model_path and os.path.exists(args.model_path):
        logger.info(f"\nLoading model from {args.model_path}")
        import json
        config_path = os.path.join(args.model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = ModelConfig.from_dict(config_dict)

        model = CausalLMModel(config)
        model_file = os.path.join(args.model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            state = torch.load(model_file, map_location="cpu", weights_only=False)
            model.load_state_dict(state, strict=False)
            logger.info("Model weights loaded")
    else:
        logger.info("\nCreating new model")
        model = CausalLMModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # 创建tokenizer
    tokenizer_path = os.path.join(args.output_dir, "tokenizer")
    if args.model_path and os.path.exists(os.path.join(args.model_path, "vocab.json")):
        logger.info(f"\nLoading tokenizer from {args.model_path}")
        tokenizer = get_tokenizer(args.model_path, tokenizer_type="bpe")
    elif os.path.exists(tokenizer_path):
        logger.info(f"\nLoading tokenizer from {tokenizer_path}")
        tokenizer = get_tokenizer(tokenizer_path, tokenizer_type="bpe")
    else:
        logger.info("\nTraining new tokenizer...")
        tokenizer = BPETokenizer()
        texts = []
        with open(args.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        tokenizer.train(texts, vocab_size=args.vocab_size)
        logger.info(f"Tokenizer trained. Vocab size: {tokenizer.vocab_size}")

    # 创建数据集
    logger.info(f"\nLoading training data from {args.train_file}")
    train_dataset = PretrainDataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )
    logger.info(f"Training samples: {len(train_dataset)}")

    eval_dataset = None
    if args.validation_file:
        eval_dataset = PretrainDataset(
            data_path=args.validation_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )
        logger.info(f"Validation samples: {len(eval_dataset)}")

    # 数据整理器
    collate_fn = get_collator(
        collator_type="causal_lm",
        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
        max_length=args.max_seq_length,
    )

    # 创建训练配置
    training_config = GPUTrainingConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        use_flash_attention=args.use_flash_attention,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps,
    )

    # 创建GPU训练器
    trainer = GPUTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        collate_fn=collate_fn,
    )

    # 开始训练
    logger.info("\n" + "=" * 60)
    logger.info("Starting Training")
    logger.info("=" * 60)

    results = trainer.train()

    logger.info("\nTraining completed!")
    logger.info(f"Final loss: {results['train_loss']:.4f}")
    logger.info(f"Total steps: {results['global_step']}")
    logger.info(f"Training time: {results['training_time']/3600:.2f} hours")


if __name__ == "__main__":
    main()
