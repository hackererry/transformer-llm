#!/usr/bin/env python
"""
预训练脚本
用于从头开始训练语言模型
"""
import os
import sys
import argparse
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModelConfig, CausalLMModel
from src.data import PretrainDataset, get_collator, BPETokenizer, get_tokenizer
from src.training import Trainer, save_pretrained
from src.utils import setup_logger, set_seed, print_device_info, get_memory_info
from src.cpu_optim import optimize_for_cpu_training


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="预训练语言模型")

    # 数据参数
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件路径")
    parser.add_argument("--validation_file", type=str, default=None, help="验证数据文件路径")

    # 模型参数
    parser.add_argument("--model_path", type=str, default=None,
                        help="预训练模型路径（用于继续训练已有模型）")
    parser.add_argument("--model_config", type=str, default="tiny", choices=["tiny", "small", "medium"],
                        help="模型配置（仅当model_path为None时使用）")
    parser.add_argument("--vocab_size", type=int, default=32000, help="词表大小")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")

    # 学习率调度
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")

    # 日志和保存
    parser.add_argument("--logging_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志记录步数间隔")
    parser.add_argument("--save_steps", type=int, default=500, help="保存步数间隔")
    parser.add_argument("--save_total_limit", type=int, default=3, help="保存检查点数量限制")

    # 优化参数
    parser.add_argument("--bf16", action="store_true", default=True, help="使用BF16精度")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点")
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载器工作进程数")

    # 其他
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="恢复训练的检查点路径")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 设置日志
    logger = setup_logger(args.logging_dir, "pretrain")
    logger.info("=" * 50)
    logger.info("Starting pretraining")
    logger.info("=" * 50)

    # 打印设备信息
    print_device_info()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建模型配置
    if args.model_config == "tiny":
        config = ModelConfig.tiny()
    elif args.model_config == "small":
        config = ModelConfig.small()
    else:
        config = ModelConfig.medium()

    config.vocab_size = args.vocab_size
    config.max_position_embeddings = max(config.max_position_embeddings, args.max_seq_length)
    config.gradient_checkpointing = args.gradient_checkpointing

    logger.info(f"Model config: {config.to_dict()}")

    # 创建或加载模型
    if args.model_path and os.path.exists(args.model_path):
        # 从已有模型加载
        logger.info(f"Loading model from {args.model_path}")
        from src.training import load_model

        # 加载配置
        config_path = os.path.join(args.model_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config = ModelConfig.from_dict(config_dict)
            logger.info(f"Loaded config from {config_path}")

        # 创建模型并加载权重
        model = CausalLMModel(config)
        model_file = os.path.join(args.model_path, "pytorch_model.bin")
        if os.path.exists(model_file):
            state = torch.load(model_file, map_location="cpu", weights_only=False)
            if "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                model.load_state_dict(state, strict=False)
            logger.info(f"Loaded weights from {model_file}")
        else:
            logger.warning(f"Weights file not found: {model_file}")
    else:
        # 从头创建新模型
        model = CausalLMModel(config)
        logger.info("Created new model from scratch")

    # CPU优化
    if args.gradient_checkpointing:
        model = optimize_for_cpu_training(model, enable_gradient_checkpointing=True)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Memory info: {get_memory_info()}")

    # 创建或加载tokenizer
    tokenizer_path = os.path.join(args.output_dir, "tokenizer")

    # 优先级: 1. 从已有模型加载 2. 从output_dir加载 3. 从头训练
    if args.model_path and os.path.exists(args.model_path):
        # 从已有模型加载tokenizer
        model_tokenizer_path = args.model_path
        vocab_file = os.path.join(model_tokenizer_path, "vocab.json")
        if os.path.exists(vocab_file):
            logger.info(f"Loading tokenizer from model path: {model_tokenizer_path}")
            tokenizer = get_tokenizer(model_tokenizer_path, tokenizer_type="bpe")
        else:
            logger.warning(f"Tokenizer not found in model path, will train new one")
            tokenizer = None
    elif os.path.exists(tokenizer_path):
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = get_tokenizer(tokenizer_path, tokenizer_type="bpe")
    else:
        tokenizer = None

    if tokenizer is None:
        logger.info("Creating new tokenizer (will be trained on data)")
        tokenizer = BPETokenizer()

        # 训练tokenizer - 从训练数据中学习词汇表
        logger.info("Training tokenizer on data...")
        # 读取训练数据用于tokenizer训练
        texts = []
        with open(args.train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
        tokenizer.train(texts, vocab_size=args.vocab_size)
        logger.info(f"Tokenizer trained. Vocab size: {tokenizer.vocab_size}")

    # 更新模型词表大小以匹配tokenizer
    if tokenizer.vocab_size != config.vocab_size:
        logger.info(f"Updating vocab_size from {config.vocab_size} to {tokenizer.vocab_size}")
        config.vocab_size = tokenizer.vocab_size
        # 需要重新调整模型的embedding层
        model.resize_token_embeddings(tokenizer.vocab_size)

    # 创建数据集
    logger.info(f"Loading training data from {args.train_file}")
    train_dataset = PretrainDataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
    )

    eval_dataset = None
    if args.validation_file:
        logger.info(f"Loading validation data from {args.validation_file}")
        eval_dataset = PretrainDataset(
            data_path=args.validation_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
        )

    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation samples: {len(eval_dataset)}")

    # 创建数据整理器
    collate_fn = get_collator(
        collator_type="causal_lm",
        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
        max_length=args.max_seq_length,
    )

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        collate_fn=collate_fn,
    )

    # 记录超参数
    logger.log_hyperparams(vars(args))

    # 开始训练
    logger.info("Starting training...")
    results = trainer.train()

    logger.info(f"Training completed. Results: {results}")

    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model")
    save_pretrained(model, final_model_path, tokenizer)
    logger.info(f"Final model saved to {final_model_path}")

    # 保存tokenizer
    if hasattr(tokenizer, 'save'):
        tokenizer.save(tokenizer_path)
        logger.info(f"Tokenizer saved to {tokenizer_path}")

    logger.close()


if __name__ == "__main__":
    main()
