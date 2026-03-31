# 模型层 (src/model/)

Transformer 架构的核心实现，支持标准 FFN 与 MoE（Mixture of Experts）双模式。

## 目录结构

```
src/model/
├── config.py              # ModelConfig — 模型配置（参数量、GQA、RoPE、MoE、MLA）
├── transformer.py          # TransformerModel / TransformerBlock / CausalLMModel
├── embedding.py            # TokenEmbedding / RotaryEmbedding / TransformerEmbedding
├── layers.py              # RMSNorm / SwiGLUFFN / FeedForward / MLP / LayerNorm
├── lm_head.py             # LMHead / TiedLMHead / AdaptiveLMHead
├── attention/             # 注意力机制子模块
│   ├── __init__.py        # 工厂函数 create_attention
│   ├── base.py            # AttentionBase 基类、RoPE 工具函数
│   ├── standard.py        # StandardAttention — 标准注意力
│   ├── flash.py           # FlashAttention — GPU Flash Attention
│   ├── gqa.py             # GroupedQueryAttention — 分组查询注意力
│   ├── streaming.py        # StreamingAttention — StreamingLLM 滑动窗口注意力
│   └── mla.py             # MultiHeadLatentAttention — MLA (DeepSeek-V2/V3)
└── moe/                   # MoE 子模块
    ├── __init__.py
    ├── expert.py           # SwiGLUExpert / SharedExpert
    ├── router.py           # TopKRouter — 路由选择
    ├── moe_layer.py        # DeepSeekMoE — MoE 层
    └── load_balancing.py    # 负载均衡损失（aux loss / z loss）
```

## 核心组件

### config.py — ModelConfig

模型配置类，定义所有模型参数。

**关键配置项：**

| 类别 | 参数 | 说明 |
|------|------|------|
| 基础 | `vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size` | 模型规模参数 |
| GQA | `num_key_value_heads` | KV 头数，默认 Q 头数的 1/4 |
| RoPE | `max_position_embeddings`, `rope_theta`, `rope_scaling` | 旋转位置编码，YaRN 4 倍外推 |
| StreamingLLM | `use_streaming_llm`, `sink_size`, `streaming_window_size` | 无限长度推理 |
| MoE | `use_moe`, `num_experts`, `num_shared_experts`, `num_experts_per_tok`, `aux_loss_alpha` | 混合专家 |
| MLA | `use_mla`, `kv_lora_rank`, `q_lora_rank`, `rope_head_dim`, `v_head_dim` | 多头潜在注意力 |

**预设配置：**

| 配置 | 参数量 | 特点 |
|------|--------|------|
| `ModelConfig.tiny()` | ~10M | FFN，禁用 MoE，启用 MLA |
| `ModelConfig.small()` | ~100M | FFN，禁用 MoE，启用 MLA |
| `ModelConfig.medium()` | ~500M | FFN + MoE (8 experts, top-2) |
| `ModelConfig.moe_small()` | ~20M active | MoE (8 experts, top-2) |
| `ModelConfig.moe_medium()` | ~100M active | MoE (16 experts, top-4) |

### transformer.py — 核心模型

- **`TransformerModel`** — GPT 风格 Decoder-only Transformer，使用 Pre-Norm 架构
- **`TransformerBlock`** — 单层 TransformerBlock，支持 Attention + FFN/MoE
- **`CausalLMModel`** — 完整的因果语言模型（Transformer + LMHead）

**架构流程：**
```
Token IDs → TokenEmbedding → TransformerEmbedding(RoPE)
    → [TransformerBlock × N] → RMSNorm → LMHead → logits
```

### attention/ — 注意力机制

`create_attention()` 工厂函数自动选择最优实现：

| 优先级 | 类型 | 说明 |
|--------|------|------|
| 1 | **MLA** | `use_mla=True` — Multi-Head Latent Attention，KV 压缩 |
| 2 | **StreamingLLM** | `use_streaming_llm=True` — 滑动窗口 + Attention Sink |
| 3 | **GQA** | `use_gqa=True` 或 `num_key_value_heads != num_attention_heads` |
| 4 | **Flash Attention** | GPU 可用且已安装 flash-attn |
| 5 | **Chunked** | `use_chunked=True` — 分块注意力 |
| 6 | **Standard** | 兼容性最好的标准注意力 |

**子模块说明：**

| 文件 | 类 | 说明 |
|------|-----|------|
| `base.py` | `AttentionBase`, `rotate_half`, `apply_rotary_emb`, `create_causal_mask` | 基类和工具函数 |
| `standard.py` | `StandardAttention`, `ChunkedAttention`, `CrossAttention` | 标准实现 |
| `flash.py` | `FlashAttention`, `ScaledDotProductAttention` | Flash Attention 2/3 + SDPA |
| `gqa.py` | `GroupedQueryAttention`, `MultiQueryAttention` | 分组查询注意力，减少 75% KV 缓存 |
| `streaming.py` | `StreamingAttention`, `StreamingKVCache` | StreamingLLM 固定显存无限推理 |
| `mla.py` | `MultiHeadLatentAttention`, `MLAKVCache` | MLA KV 压缩，DeepSeek-V2/V3 风格 |

### moe/ — 混合专家

**架构：** 共享专家（始终激活）+ 路由专家（Top-K 选择）

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `expert.py` | `SwiGLUExpert`, `SharedExpert` | 单个 SwiGLU 专家 / 共享专家 |
| `router.py` | `TopKRouter` | Top-K 路由选择，支持训练噪声 |
| `moe_layer.py` | `DeepSeekMoE`, `MoEMLP` | 完整 MoE 层实现 |
| `load_balancing.py` | `compute_load_balancing_loss`, `compute_z_loss`, `compute_moe_aux_loss` | 辅助损失函数 |

### embedding.py

| 类 | 说明 |
|-----|------|
| `TokenEmbedding` | Token ID → 嵌入向量 |
| `RotaryEmbedding` | 旋转位置编码（RoPE），支持 YaRN 长度外推 |
| `TransformerEmbedding` | 组合 Token 嵌入和 RoPE |

### layers.py

| 类 | 说明 |
|-----|------|
| `RMSNorm` | Root Mean Square Layer Normalization，比 LayerNorm 更高效 |
| `SwiGLUFFN` | SwiGLU 前馈网络：`Swish(xW₁) ⊗ (xW₂)` |
| `FeedForward` | 标准 FFN |
| `MLP` | 多层感知机 |
| `LayerNorm` | 标准 Layer Normalization |
| `TransformerMLP` | Transformer 风格的 MLP |

### lm_head.py

| 类 | 说明 |
|-----|------|
| `LMHead` | 标准语言模型输出头 |
| `TiedLMHead` | 共享权重的 LM 头（与输入嵌入绑定） |
| `AdaptiveLMHead` | 自适应 LM 头 |
| `MLPHead` | MLP 风格的输出头 |
| `Pooler` | 池化层 |
| `SequenceSummary` | 序列摘要层 |

## 使用示例

```python
from src.model import ModelConfig, CausalLMModel, create_model

# 方式1：使用预设配置
model = create_model(config_name="small")

# 方式2：自定义配置
config = ModelConfig(
    vocab_size=32000,
    hidden_size=512,
    num_hidden_layers=12,
    num_attention_heads=8,
    intermediate_size=1024,
    use_moe=True,         # 启用 MoE
    num_experts=8,        # 8 个专家
    num_experts_per_tok=2,  # Top-2 路由
    use_mla=True,         # 启用 MLA
)
model = CausalLMModel(config)

# 前向传播
import torch
input_ids = torch.randint(0, 32000, (2, 128))
outputs = model(input_ids=input_ids, labels=input_ids)
print(outputs["loss"])  # 包含 MoE aux_loss
```

## MoE 与 MLA 的协同

DeepSeek-V3 风格的设计：**MoE** 减少 FFN 激活参数量，**MLA** 减少 Attention KV 缓存。二者互补：

| 优化维度 | 技术 | 效果 |
|---------|------|------|
| FFN 参数 | MoE (8 experts, top-2) | 减少 ~75% 激活参数 |
| Attention KV | MLA + GQA | 减少 ~75% KV 缓存 |
| 位置外推 | YaRN | 4 倍序列长度外推 |

## 关键特性

- **Pre-Norm 架构** — 训练更稳定
- **DeepSeekMoE** — 共享专家 + Top-K 路由 + 辅助损失
- **Multi-Head Latent Attention** — KV 压缩 + 解耦 RoPE
- **StreamingLLM** — 固定显存无限长度推理
- **YaRN** — 无需微调的长度外推
