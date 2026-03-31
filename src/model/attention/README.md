# Attention 子模块 (src/model/attention/)

多种注意力机制实现，支持从标准注意力到高级优化（Flash Attention、GQA、StreamingLLM、MLA）。

## 文件概览

```
src/model/attention/
├── __init__.py    # 工厂函数 create_attention()
├── base.py        # AttentionBase 基类、RoPE 工具函数
├── standard.py    # StandardAttention — 标准注意力
├── flash.py       # FlashAttention — GPU Flash Attention 2/3
├── gqa.py        # GroupedQueryAttention — 分组查询注意力
├── streaming.py    # StreamingAttention — StreamingLLM 滑动窗口
└── mla.py        # MultiHeadLatentAttention — MLA KV 压缩
```

## 工厂函数 — create_attention()

自动选择最优 Attention 实现：

```
优先级 1: use_mla=True         → MultiHeadLatentAttention
    ↓ 2: use_streaming_llm=True → StreamingAttention
    ↓ 3: use_gqa=True          → GroupedQueryAttention
    ↓ 4: GPU + flash-attn       → FlashAttention
    ↓ 5: use_chunked=True       → ChunkedAttention
    ↓ 6: (其他)                 → StandardAttention
```

## Attention 实现对比

| 类型 | 实现文件 | KV 缓存 | 适用场景 |
|------|---------|---------|---------|
| **Standard** | `standard.py` | O(n) 完整 | CPU / 兼容性 |
| **Flash** | `flash.py` | O(n) 完整 | GPU 加速 |
| **GQA** | `gqa.py` | O(n×G/Q) 减少 75% | 长序列 |
| **Streaming** | `streaming.py` | O(sink+window) 固定 | 无限长度 |
| **MLA** | `mla.py` | O(n×d_c) 压缩 | DeepSeek-V2/V3 |

## 子模块详解

### base.py

`AttentionBase` — 所有 Attention 的基类，定义统一接口。

**工具函数：**

| 函数 | 说明 |
|------|------|
| `rotate_half(x)` | 旋转半个张量（RoPE 核心） |
| `apply_rotary_emb(x, cos, sin)` | 应用 RoPE |
| `apply_rotary_emb_qk(q, k, cos, sin)` | QK 分别应用 RoPE |
| `create_causal_mask(seq_len, device)` | 创建因果掩码 |
| `repeat_kv(x, n_rep)` | KV 重复（支持 GQA） |

### standard.py

`StandardAttention` — 标准多头注意力，最通用的实现。

- 分离 QKV 投影
- 支持 RoPE 位置编码
- 支持 KV 缓存（推理加速）
- CPU/GPU 通用

### flash.py

`FlashAttention` — GPU Flash Attention 实现。

**优先级：** Flash Attention 2 > PyTorch SDPA > 标准实现

**优势：**
- IO-aware 注意力计算
- 显存节省 ~50%（无需存储 S/P 矩阵）
- 计算速度提升 2-4x

**自动检测：**
- 检查 `flash_attn` 是否安装
- 检查 `torch.cuda.is_available()`
- 回退到 PyTorch SDPA 或标准实现

### gqa.py

`GroupedQueryAttention` — 分组查询注意力（Llama 2/3 使用）。

**原理：**
- Q 有 8 个头，K/V 只有 2 个头
- 每个 K/V 头被 4 个 Q 头共享
- KV 缓存减少 4 倍

**适用场景：**
- 长序列推理
- 显存受限场景
- 几乎无精度损失

### streaming.py

`StreamingAttention` — StreamingLLM 滑动窗口注意力。

**结构：** `[Sinks | Sliding Window]`

- **Sinks**：前 N 个 token（如 4 个），始终参与注意力
- **Sliding Window**：最近 W 个 token（如 4096 个）

**效果：**
- 显存占用固定 O(sink + window)
- 支持「无限」长度推理
- 无需重新训练

### mla.py

`MultiHeadLatentAttention` — DeepSeek-V2/V3 风格 MLA。

**核心优化：**

1. **KV 压缩** — 将 KV 压缩到低维潜在空间（`kv_lora_rank`）
2. **解耦 RoPE** — 只对部分维度应用 RoPE
3. **吸收投影** — 推理时可合并投影矩阵

**对比 GQA：**

| 维度 | GQA | MLA |
|------|-----|-----|
| KV 头数 | Q 头数的 1/N | 1 |
| KV 缓存 | 减少 N 倍 | 减少更多 |
| 压缩 | 无 | 低秩分解 |

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `kv_lora_rank` | 512 | KV 压缩维度 |
| `q_lora_rank` | 1536 | Q 压缩维度 |
| `rope_head_dim` | 64 | RoPE 应用维度 |
| `v_head_dim` | 128 | V 的 head 维度 |
