"""
训练模块测试
测试优化器、调度器、检查点管理、训练器等
"""
import pytest
import torch
import torch.nn as nn
import sys
import os
import tempfile
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import (
    create_optimizer,
    create_scheduler,
    CheckpointManager,
    save_model,
    load_model,
    save_pretrained,
    Trainer,
    TrainingConfig,
    TrainerState,
    PerformanceMonitor,
    AdamWOptimizer,
    LAMB,
    get_optimizer,
    get_scheduler,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
)
from src.model import ModelConfig, CausalLMModel


class SimpleModel(nn.Module):
    """简单测试模型"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.embedding = nn.Embedding(100, 10)

    def forward(self, x):
        return self.linear(x)


class SimpleLMModel(nn.Module):
    """简单语言模型用于测试"""

    def __init__(self, vocab_size=100, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, labels=None):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        output = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        return output


class TestCreateOptimizer:
    """优化器创建测试"""

    def test_create_adamw(self):
        """测试创建AdamW"""
        model = SimpleModel()
        optimizer = create_optimizer(model, optimizer_type="adamw", learning_rate=1e-4)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_create_adam(self):
        """测试创建Adam"""
        model = SimpleModel()
        optimizer = create_optimizer(model, optimizer_type="adam", learning_rate=1e-4)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_create_sgd(self):
        """测试创建SGD"""
        model = SimpleModel()
        optimizer = create_optimizer(model, optimizer_type="sgd", learning_rate=1e-4)
        assert isinstance(optimizer, torch.optim.SGD)

    def test_weight_decay_separation(self):
        """测试权重衰减分离"""
        model = SimpleModel()
        optimizer = create_optimizer(model, optimizer_type="adamw", weight_decay=0.01)

        # 检查是否有两组参数（带衰减和不带衰减）
        assert len(optimizer.param_groups) == 2
        # 第一组有权重衰减
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        # 第二组(bias/norm)没有权重衰减
        assert optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_different_learning_rates(self):
        """测试不同学习率"""
        model = SimpleModel()
        optimizer = create_optimizer(
            model,
            learning_rate=1e-3,
            weight_decay=0.01,
        )
        assert optimizer.param_groups[0]["lr"] == 1e-3


class TestGetOptimizer:
    """优化器工厂函数测试"""

    def test_get_adamw(self):
        """测试获取AdamW"""
        model = SimpleModel()
        opt = get_optimizer(model, optimizer_type="adamw", learning_rate=1e-4)
        assert isinstance(opt, torch.optim.AdamW)


class TestScheduler:
    """学习率调度器测试"""

    def test_create_cosine_scheduler(self):
        """测试创建余弦调度器"""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_training_steps=100,
            num_warmup_steps=10,
        )
        assert scheduler is not None

    def test_create_linear_scheduler(self):
        """测试创建线性调度器"""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="linear",
            num_training_steps=100,
            num_warmup_steps=10,
        )
        assert scheduler is not None

    def test_create_constant_scheduler(self):
        """测试创建常数调度器"""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="constant",
            num_training_steps=100,
            num_warmup_steps=10,
        )
        assert scheduler is not None

    def test_scheduler_warmup(self):
        """测试预热阶段"""
        model = SimpleModel()
        optimizer = create_optimizer(model, learning_rate=1.0)
        scheduler = create_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_training_steps=100,
            num_warmup_steps=10,
        )

        # 在预热阶段，学习率应该逐渐增加
        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # 检查学习率是否在增加
        assert lrs[-1] > lrs[0]

    def test_cosine_schedule_function(self):
        """测试余弦调度函数"""
        model = SimpleModel()
        optimizer = create_optimizer(model, learning_rate=1.0)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
        )
        assert scheduler is not None

    def test_linear_schedule_function(self):
        """测试线性调度函数"""
        model = SimpleModel()
        optimizer = create_optimizer(model, learning_rate=1.0)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=10,
            num_training_steps=100,
        )
        assert scheduler is not None


class TestGetScheduler:
    """调度器工厂函数测试"""

    def test_get_cosine_scheduler(self):
        """测试获取余弦调度器"""
        model = SimpleModel()
        optimizer = create_optimizer(model)
        scheduler = get_scheduler(
            optimizer,
            scheduler_type="cosine",
            num_training_steps=100,
            num_warmup_steps=10,
        )
        assert scheduler is not None


class TestCheckpointManager:
    """检查点管理器测试"""

    def test_save_and_load(self, tmp_path):
        """测试保存和加载"""
        model = CausalLMModel(ModelConfig.tiny())
        optimizer = create_optimizer(model)

        manager = CheckpointManager(str(tmp_path))

        # 保存
        checkpoint_path = manager.save(
            model=model,
            optimizer=optimizer,
            step=100,
            epoch=1,
            metrics={"loss": 0.5},
        )
        assert os.path.exists(f"{checkpoint_path}.pt")

        # 加载
        state = manager.load(checkpoint_path, model, optimizer)
        assert state["step"] == 100
        assert state["epoch"] == 1

    def test_best_model_saving(self, tmp_path):
        """测试最佳模型保存"""
        model = CausalLMModel(ModelConfig.tiny())
        optimizer = create_optimizer(model)

        manager = CheckpointManager(
            str(tmp_path),
            save_best_only=True,
            metric_name="loss",
            metric_mode="min",
        )

        # 保存第一个检查点
        manager.save(model, optimizer, step=100, metrics={"loss": 1.0})

        # 保存更好的检查点
        manager.save(model, optimizer, step=200, metrics={"loss": 0.5})

        # 检查最佳模型是否存在
        assert os.path.exists(str(tmp_path / "best_model.pt"))

    def test_cleanup_old_checkpoints(self, tmp_path):
        """测试清理旧检查点"""
        model = CausalLMModel(ModelConfig.tiny())
        optimizer = create_optimizer(model)

        manager = CheckpointManager(str(tmp_path), max_checkpoints=2)

        # 保存3个检查点
        for i in range(3):
            manager.save(model, optimizer, step=i * 100, metrics={"loss": i * 0.1})

        # 应该只保留2个
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= 2

    def test_list_checkpoints(self, tmp_path):
        """测试列出检查点"""
        model = CausalLMModel(ModelConfig.tiny())
        optimizer = create_optimizer(model)

        manager = CheckpointManager(str(tmp_path))

        # 保存检查点
        manager.save(model, optimizer, step=100)
        manager.save(model, optimizer, step=200)

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2

    def test_get_checkpoint_info(self, tmp_path):
        """测试获取检查点信息"""
        model = CausalLMModel(ModelConfig.tiny())
        optimizer = create_optimizer(model)

        manager = CheckpointManager(str(tmp_path))
        manager.save(
            model, optimizer, step=100, epoch=1, metrics={"loss": 0.5}
        )

        checkpoints = manager.list_checkpoints()
        info = manager.get_checkpoint_info(checkpoints[0])
        assert info["step"] == 100


class TestSaveLoadModel:
    """模型保存加载测试"""

    def test_save_model(self, tmp_path):
        """测试保存模型"""
        model = CausalLMModel(ModelConfig.tiny())
        output_path = str(tmp_path / "model")

        save_model(model, output_path)
        assert os.path.exists(f"{output_path}.pt")
        assert os.path.exists(f"{output_path}-config.json")

    def test_load_model(self, tmp_path):
        """测试加载模型"""
        model = CausalLMModel(ModelConfig.tiny())
        output_path = str(tmp_path / "model")

        save_model(model, output_path)

        # 创建新模型并加载
        new_model = CausalLMModel(ModelConfig.tiny())
        state = load_model(new_model, f"{output_path}.pt")
        assert "model_state_dict" in state

    def test_save_pretrained(self, tmp_path):
        """测试HuggingFace格式保存"""
        model = CausalLMModel(ModelConfig.tiny())
        output_dir = str(tmp_path / "pretrained")

        save_pretrained(model, output_dir)
        assert os.path.exists(os.path.join(output_dir, "pytorch_model.bin"))
        assert os.path.exists(os.path.join(output_dir, "config.json"))


class TestPerformanceMonitor:
    """性能监控器测试"""

    def test_monitor_creation(self):
        """测试监控器创建"""
        monitor = PerformanceMonitor()
        assert monitor.enabled

    def test_measure_stage(self):
        """测试阶段计时"""
        monitor = PerformanceMonitor()

        with monitor.measure("forward"):
            x = torch.randn(100, 100)
            _ = x @ x

        summary = monitor.get_summary()
        assert "forward" in summary
        assert summary["forward"]["count"] == 1

    def test_record_batch(self):
        """测试记录批次时间"""
        monitor = PerformanceMonitor()
        monitor.record_batch(0.1)
        monitor.record_batch(0.2)

        summary = monitor.get_summary()
        assert summary["batch_total"]["count"] == 2

    def test_print_summary(self, capsys):
        """测试打印摘要"""
        monitor = PerformanceMonitor()
        monitor.measure("forward")
        monitor.record_batch(0.1)

        monitor.print_summary()
        captured = capsys.readouterr()
        assert "Performance Summary" in captured.out


class TestTrainerState:
    """训练状态测试"""

    def test_state_creation(self):
        """测试状态创建"""
        state = TrainerState()
        assert state.epoch == 0
        assert state.global_step == 0

    def test_to_dict(self):
        """测试转换为字典"""
        state = TrainerState()
        state.epoch = 1
        state.global_step = 100
        state.best_metric = 0.5

        d = state.to_dict()
        assert d["epoch"] == 1
        assert d["global_step"] == 100
        assert d["best_metric"] == 0.5

    def test_from_dict(self):
        """测试从字典创建"""
        d = {
            "epoch": 2,
            "global_step": 200,
            "best_metric": 0.3,
            "learning_rate": 1e-4,
            "loss_history": [0.5, 0.4, 0.3],
        }

        state = TrainerState.from_dict(d)
        assert state.epoch == 2
        assert state.global_step == 200
        assert state.best_metric == 0.3


class TestTrainingConfig:
    """训练配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = TrainingConfig()
        assert config.output_dir == "./output"
        assert config.num_train_epochs == 3
        assert config.learning_rate == 5e-5

    def test_custom_config(self):
        """测试自定义配置"""
        config = TrainingConfig(
            output_dir="./custom_output",
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=1e-4,
        )
        assert config.output_dir == "./custom_output"
        assert config.num_train_epochs == 5
        assert config.per_device_train_batch_size == 16
        assert config.learning_rate == 1e-4

    def test_effective_batch_size(self):
        """测试有效批次大小"""
        config = TrainingConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
        )
        assert config.effective_batch_size == 32


class TestTrainer:
    """训练器测试"""

    def test_trainer_creation(self, tmp_path):
        """测试训练器创建"""
        model = CausalLMModel(ModelConfig.tiny())

        # 创建简单数据集
        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, 100, (32,)),
                    "labels": torch.randint(0, 100, (32,)),
                }

        dataset = SimpleDataset()

        config = TrainingConfig(
            output_dir=str(tmp_path),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            bf16=False,  # 禁用BF16用于CPU测试
        )

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            config=config,
        )

        assert trainer is not None
        assert trainer.device.type in ["cpu", "cuda"]

    def test_trainer_device_detection(self, tmp_path):
        """测试设备检测"""
        model = CausalLMModel(ModelConfig.tiny())

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 5

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, 100, (16,)),
                    "labels": torch.randint(0, 100, (16,)),
                }

        dataset = SimpleDataset()
        config = TrainingConfig(
            output_dir=str(tmp_path),
            num_train_epochs=1,
            per_device_train_batch_size=1,
            bf16=False,
        )

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            config=config,
        )

        if torch.cuda.is_available():
            assert trainer.is_gpu
        else:
            assert not trainer.is_gpu

    def test_trainer_with_collator(self, tmp_path):
        """测试带collator的训练器"""
        model = CausalLMModel(ModelConfig.tiny())

        class SimpleDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10

            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, 100, (32,)),
                }

        dataset = SimpleDataset()

        collator = get_collator("causal_lm", pad_token_id=0)

        config = TrainingConfig(
            output_dir=str(tmp_path),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            bf16=False,
        )

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            config=config,
            collate_fn=collator,
        )

        assert trainer.collate_fn is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
