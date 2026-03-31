# MoE 子模块 (src/model/moe/)

DeepSeek-V3 风格的 Mixture of Experts（混合专家）实现。

## 文件概览

```
src/model/moe/
├── __init__.py           # 统一导出
├── expert.py             # SwiGLUExpert / SharedExpert
├── router.py            # TopKRouter — Top-K 路由
├── moe_layer.py         # DeepSeekMoE — MoE 层
└── load_balancing.py    # 负载均衡损失
```

## 架构设计

```
Input
  │
  ▼
┌─────────────────────────┐
│   共享专家 (1-N个)       │  ← 始终激活，处理通用知识
│   SharedExpert × N      │
└────────────┬────────────┘
             │ +
             ▼
┌─────────────────────────┐
│   路由器 (Top-K Router)  │  ← 选择 top-K 个专家
│   TopKRouter            │
└─────┬─────────┬─────────┘
      │         │
      ▼         ▼
┌─────────┐ ┌─────────┐
│ Expert1 │ │ Expert2 │ ... ExpertN  ← 按需激活
└─────────┘ └─────────┘
      │         │
      └────┬────┘
           │ weighted sum
           ▼
        Output
```

## 核心组件

### expert.py — 专家模块

| 类 | 说明 |
|-----|------|
| `SwiGLUExpert` | 单个 SwiGLU 专家（gate/up/down 三投影） |
| `SharedExpert` | 共享专家（始终激活，无路由） |

**SwiGLUExpert 结构：**

```
x → gate_proj → Swish(x)  ⊗  up_proj(x)  → down_proj → output
```

### router.py — TopKRouter

**功能：**
- Top-K 专家选择（每个 token 激活 K 个专家）
- 训练时添加噪声（促进负载均衡）
- 归一化路由权重

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_experts` | 8 | 专家总数 |
| `top_k` | 2 | 每个 token 激活的专家数 |
| `noise_std` | 0.1 | 路由噪声标准差 |
| `routing_bias` | False | 路由偏置 |

### moe_layer.py — DeepSeekMoE

完整 MoE 层实现，整合专家 + 路由器 + 共享专家。

| 参数 | 说明 |
|------|------|
| `num_experts` | 路由专家数量 |
| `num_shared_experts` | 共享专家数量（默认 1） |
| `top_k` | Top-K 选择数 |
| `aux_loss_alpha` | 辅助损失系数 |
| `routed_scaling_factor` | 路由输出缩放因子 |

### load_balancing.py — 负载均衡损失

| 函数 | 说明 |
|------|------|
| `compute_load_balancing_loss(router_probs, num_experts)` | 路由器概率均衡损失 |
| `compute_z_loss(logits)` | Z-Loss（防止 logits 过大） |
| `compute_moe_aux_loss(moe_output)` | 综合辅助损失 |

**辅助损失公式：**

```
aux_loss = α × (load_balance_loss + z_loss)
```

其中 `α` 默认 0.01，在总损失中权重较小。

## 配置示例

```python
from src.model import ModelConfig, CausalLMModel

# 小型 MoE（推荐用于 ~100M 模型）
config = ModelConfig.moe_small()
model = CausalLMModel(config)
# 参数: 8 experts, top-2 路由, 1 共享专家

# 中型 MoE（推荐用于 ~500M+ 模型）
config = ModelConfig.moe_medium()
model = CausalLMModel(config)
# 参数: 16 experts, top-4 路由, 1 共享专家

# 完全自定义
config = ModelConfig(
    hidden_size=512,
    num_hidden_layers=12,
    use_moe=True,
    num_experts=8,
    num_shared_experts=1,
    num_experts_per_tok=2,
    aux_loss_alpha=0.01,
)
```

## MoE vs FFN 对比

| 维度 | FFN (dense) | MoE (sparse) |
|------|-------------|--------------|
| 总参数量 | 全部激活 | 全部存储 |
| 激活参数 | 100% | top-K / N |
| 计算量 | O(N) | O(K×N/E) |
| 显存 | 标准 | 专家权重额外占用 |
| 适用规模 | < 100M | > 100M |

## 显存估算

8 个专家 + top-2 路由：

- 模型权重：增加 ~7 倍（只有 2/8 激活）
- 激活值：减少 ~75%（每 token 只计算 2 个专家）
- 适合场景：长序列、长上下文、多专家路由

## 最佳实践

1. **小模型禁用 MoE** — 参数量 < 50M 推荐 FFN
2. **专家数量** — 8-16 个专家效果较好
3. **Top-K** — top-2 或 top-4 平衡效果和效率
4. **辅助损失** — `aux_loss_alpha=0.01` 促进专家均衡
5. **共享专家** — 提供通用知识处理能力
