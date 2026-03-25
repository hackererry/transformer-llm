#!/usr/bin/env python
"""
微调脚本
用于指令微调(SFT)
"""
import os
import sys
import argparse
import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ModelConfig, CausalLMModel
from src.data import FinetuneDataset, get_collator, get_tokenizer
from src.training import Trainer, TrainingConfig, save_pretrained, load_model
from src.utils import setup_logger, set_seed, print_device_info


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="指令微调")

    # 数据参数
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件路径")
    parser.add_argument("--validation_file", type=str, default=None, help="验证数据文件路径")
    parser.add_argument("--template", type=str, default="alpaca", choices=["alpaca", "chat", "simple"],
                        help="指令模板类型")

    # 模型参数
    parser.add_argument("--model_path", type=str, default=None, help="预训练模型路径")
    parser.add_argument("--model_config", type=str, default="tiny", choices=["tiny", "small", "medium"],
                        help="模型配置(如果model_path为None)")

    # 训练参数
    parser.add_argument("--output_dir", type=str, default="./output_sft", help="输出目录")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")

    # 学习率调度
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="学习率调度器类型")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="预热比例")

    # 日志和保存
    parser.add_argument("--logging_dir", type=str, default="./logs_sft", help="日志目录")
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
    logger = setup_logger(args.logging_dir, "finetune")
    logger.info("=" * 50)
    logger.info("Starting instruction tuning (SFT)")
    logger.info("=" * 50)

    # 打印设备信息
    print_device_info()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载或创建模型
    if args.model_path:
        logger.info(f"Loading model from {args.model_path}")

        # 加载配置
        config_path = os.path.join(args.model_path, "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            config = ModelConfig.from_dict(config_dict)
        else:
            config = ModelConfig.tiny()

        model = CausalLMModel(config)
        # 优先尝试加载 internal format (.pt)
        pt_path = os.path.join(args.model_path, "final_model.pt")
        bin_path = os.path.join(args.model_path, "pytorch_model.bin")
        if os.path.exists(pt_path):
            load_model(model, pt_path)
        elif os.path.exists(bin_path):
            # HuggingFace 格式直接加载
            state = torch.load(bin_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(f"No model file found in {args.model_path}")
    else:
        logger.info(f"Creating new model with {args.model_config} config")
        if args.model_config == "tiny":
            config = ModelConfig.tiny()
        elif args.model_config == "small":
            config = ModelConfig.small()
        else:
            config = ModelConfig.medium()

        model = CausalLMModel(config)

    logger.info(f"Model config: {config.to_dict()}")

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # 加载tokenizer
    if args.model_path:
        tokenizer_path = args.model_path
    else:
        tokenizer_path = args.output_dir

    tokenizer = get_tokenizer(tokenizer_path, tokenizer_type="bpe", use_fast=True)

    # 创建数据集
    logger.info(f"Loading training data from {args.train_file}")
    train_dataset = FinetuneDataset(
        data_path=args.train_file,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        template=args.template,
    )

    eval_dataset = None
    if args.validation_file:
        logger.info(f"Loading validation data from {args.validation_file}")
        eval_dataset = FinetuneDataset(
            data_path=args.validation_file,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            template=args.template,
        )

    logger.info(f"Training samples: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Validation samples: {len(eval_dataset)}")

    # 创建数据整理器
    collate_fn = get_collator(
        collator_type="sft",
        pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0,
        max_length=args.max_seq_length,
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
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
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

    logger.close()


if __name__ == "__main__":
    main()
