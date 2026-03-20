"""
梯度检查点模块
通过重计算中间激活值来减少内存使用
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, List, Tuple, Any, Callable, Union
import warnings


class GradientCheckpointFunction(torch.autograd.Function):
    """
    自定义梯度检查点函数
    """

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.save_for_backward(*args)

        with torch.no_grad():
            outputs = run_function(*args)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        inputs = ctx.saved_tensors

        # 重新计算前向传播
        with torch.enable_grad():
            outputs = ctx.run_function(*inputs)

        # 计算梯度
        torch.autograd.backward(outputs, grad_outputs)

        return (None, None) + tuple(inp.grad if inp.grad is not None else torch.zeros_like(inp) for inp in inputs)


def gradient_checkpoint(
    function: Callable,
    *args,
    use_reentrant: bool = True,
    preserve_rng_state: bool = True,
    **kwargs,
) -> Any:
    """
    梯度检查点包装函数

    Args:
        function: 要包装的函数
        *args: 函数参数
        use_reentrant: 是否使用可重入模式
        preserve_rng_state: 是否保存随机状态
        **kwargs: 额外参数

    Returns:
        函数输出
    """
    if use_reentrant:
        return checkpoint(function, *args, use_reentrant=True, preserve_rng_state=preserve_rng_state, **kwargs)
    else:
        return checkpoint(function, *args, use_reentrant=False, preserve_rng_state=preserve_rng_state, **kwargs)


class CheckpointedModule(nn.Module):
    """
    带梯度检查点的模块包装器
    """

    def __init__(
        self,
        module: nn.Module,
        use_checkpoint: bool = True,
        checkpoint_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint
        self.checkpoint_fn = checkpoint_fn or checkpoint

    def forward(self, *args, **kwargs):
        if self.use_checkpoint and self.training:
            return self.checkpoint_fn(self.module, *args, use_reentrant=False, **kwargs)
        return self.module(*args, **kwargs)


class CheckpointedSequential(nn.Module):
    """
    带梯度检查点的序列模块
    对序列中的每个子模块应用检查点
    """

    def __init__(
        self,
        modules: List[nn.Module],
        use_checkpoint: bool = True,
        checkpoint_segments: int = 1,
    ):
        """
        Args:
            modules: 模块列表
            use_checkpoint: 是否使用梯度检查点
            checkpoint_segments: 将序列分成多少段进行检查点
        """
        super().__init__()
        self.modules_list = nn.ModuleList(modules)
        self.use_checkpoint = use_checkpoint
        self.checkpoint_segments = checkpoint_segments

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_checkpoint or not self.training:
            for module in self.modules_list:
                x = module(x)
            return x

        # 分段检查点
        segment_size = len(self.modules_list) // self.checkpoint_segments

        def run_segment(segment_modules, hidden_states):
            for module in segment_modules:
                hidden_states = module(hidden_states)
            return hidden_states

        for i in range(0, len(self.modules_list), segment_size):
            segment = list(self.modules_list[i:i + segment_size])
            x = checkpoint(run_segment, segment, x, use_reentrant=False)

        return x


class SelectiveCheckpoint:
    """
    选择性梯度检查点
    只对部分层应用检查点
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_layers: Optional[List[int]] = None,
        checkpoint_ratio: float = 0.5,
    ):
        """
        Args:
            model: 模型
            checkpoint_layers: 要检查点的层索引列表
            checkpoint_ratio: 如果checkpoint_layers为None，检查点的层比例
        """
        self.model = model
        self.checkpoint_layers = checkpoint_layers
        self.checkpoint_ratio = checkpoint_ratio

        self._setup_checkpoints()

    def _setup_checkpoints(self):
        """设置检查点"""
        # 获取所有可检查点的层
        layers = self._get_layers()

        if self.checkpoint_layers is None:
            # 根据比例选择层
            num_checkpoint = int(len(layers) * self.checkpoint_ratio)
            self.checkpoint_layers = list(range(0, len(layers), max(1, len(layers) // num_checkpoint)))[:num_checkpoint]

        # 应用检查点
        for idx in self.checkpoint_layers:
            if idx < len(layers):
                self._apply_checkpoint(layers[idx])

    def _get_layers(self) -> List[nn.Module]:
        """获取模型的所有层"""
        layers = []

        def collect_layers(module):
            for name, child in module.named_children():
                if hasattr(child, "layers"):
                    # 如果是Transformer风格的模型
                    layers.extend(child.layers)
                elif "layer" in name.lower() or "block" in name.lower():
                    layers.append(child)
                else:
                    collect_layers(child)

        collect_layers(self.model)
        return layers

    def _apply_checkpoint(self, layer: nn.Module):
        """对层应用检查点"""
        original_forward = layer.forward

        def checkpointed_forward(*args, **kwargs):
            if self.model.training:
                return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
            return original_forward(*args, **kwargs)

        layer.forward = checkpointed_forward


def enable_gradient_checkpointing(
    model: nn.Module,
    checkpoint_fn: Optional[Callable] = None,
) -> nn.Module:
    """
    启用模型的梯度检查点

    Args:
        model: 模型
        checkpoint_fn: 自定义检查点函数

    Returns:
        启用检查点后的模型
    """
    checkpoint_fn = checkpoint_fn or checkpoint

    def enable_module(module):
        for name, child in module.named_children():
            # 检查是否是Transformer块
            if "block" in name.lower() or "layer" in name.lower():
                original_forward = child.forward

                def make_checkpointed_forward(orig_fn):
                    def checkpointed_forward(*args, **kwargs):
                        if module.training:
                            return checkpoint_fn(orig_fn, *args, use_reentrant=False, **kwargs)
                        return orig_fn(*args, **kwargs)
                    return checkpointed_forward

                child.forward = make_checkpointed_forward(original_forward)
            else:
                enable_module(child)

    enable_module(model)
    return model


def set_gradient_checkpointing(
    model: nn.Module,
    value: bool = True,
) -> None:
    """
    设置模型的梯度检查点状态

    Args:
        model: 模型
        value: 是否启用
    """
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = value

    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value


class ActivationCheckpointWrapper(nn.Module):
    """
    激活检查点包装器
    可以动态开关检查点
    """

    def __init__(
        self,
        module: nn.Module,
        enabled: bool = True,
    ):
        super().__init__()
        self.module = module
        self.enabled = enabled

    def forward(self, *args, **kwargs):
        if self.enabled and self.training:
            return checkpoint(self.module, *args, use_reentrant=False, **kwargs)
        return self.module(*args, **kwargs)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False


def estimate_memory_savings(
    model: nn.Module,
    batch_size: int = 1,
    seq_length: int = 512,
    hidden_size: int = 768,
    num_layers: int = 12,
) -> dict:
    """
    估计梯度检查点可以节省的内存

    Args:
        model: 模型
        batch_size: 批次大小
        seq_length: 序列长度
        hidden_size: 隐藏层大小
        num_layers: 层数

    Returns:
        内存估计字典
    """
    # 激活值内存 (假设float32)
    activation_memory_per_layer = batch_size * seq_length * hidden_size * 4  # bytes

    # 不使用检查点的激活值内存
    total_activation_memory = activation_memory_per_layer * num_layers

    # 使用检查点后，只需要保存输入，其他激活值在反向传播时重计算
    # 假设每隔一层检查点
    checkpointed_activation_memory = activation_memory_per_layer * (num_layers // 2 + 1)

    saved_memory = total_activation_memory - checkpointed_activation_memory

    return {
        "total_activation_memory_mb": total_activation_memory / (1024 * 1024),
        "checkpointed_memory_mb": checkpointed_activation_memory / (1024 * 1024),
        "saved_memory_mb": saved_memory / (1024 * 1024),
        "savings_percent": (saved_memory / total_activation_memory) * 100 if total_activation_memory > 0 else 0,
    }
