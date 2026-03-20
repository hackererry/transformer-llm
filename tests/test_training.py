"""
训练模块测试
"""
import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training import (
    create_optimizer,
    create_scheduler,
    CheckpointManager,
    Trainer,
)
from src.model import ModelConfig, CausalLMModel


class SimpleModel(nn.Module):
    """简单测试模型"""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestOptimizer:
    """优化器测试"""

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

        # 检查是否有两组参数
        assert len(optimizer.param_groups) == 2
        # 第一组有权重衰减
        assert optimizer.param_groups[0]["weight_decay"] == 0.01
        # 第二组( bias/norm )没有权重衰减
        assert optimizer.param_groups[1]["weight_decay"] == 0.0


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

        trainer = Trainer(
            model=model,
            train_dataset=dataset,
            output_dir=str(tmp_path),
            num_train_epochs=1,
            per_device_train_batch_size=2,
        )

        assert trainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
