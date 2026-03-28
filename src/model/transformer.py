"""
Transformer主模型
完整的GPT风格Decoder-only Transformer实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import math

from .config import ModelConfig
from .layers import RMSNorm, SwiGLUFFN
from .attention import create_attention, MultiHeadLatentAttention, MLAKVCache
from .embedding import TransformerEmbedding

# MoE 支持
from .moe import DeepSeekMoE


class TransformerBlock(nn.Module):
    """
    单个Transformer块
    包含Self-Attention和FFN/MoE，使用Pre-Norm架构

    支持特性:
    - 标准 Attention 或 MLA (Multi-Head Latent Attention)
    - 标准 FFN 或 MoE (Mixture of Experts)
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config

        # MoE 辅助损失系数
        self.aux_loss_alpha = config.aux_loss_alpha if config.use_moe else 0.0

        # 注意力层选择
        if config.use_mla:
            # 使用 MLA (Multi-Head Latent Attention)
            self.self_attn = MultiHeadLatentAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                kv_lora_rank=config.kv_lora_rank,
                q_lora_rank=config.q_lora_rank,
                rope_head_dim=config.rope_head_dim,
                v_head_dim=config.v_head_dim,
                attention_dropout=config.attention_dropout,
                hidden_dropout=config.hidden_dropout,
                max_position_embeddings=config.max_position_embeddings,
                use_flash_attn=config.use_flash_attention,
            )
        else:
            # 使用标准 Attention（通过统一工厂函数）
            self.self_attn = create_attention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                head_dim=config.head_dim,
                num_key_value_heads=config.num_key_value_heads,
                attention_dropout=config.attention_dropout,
                hidden_dropout=config.hidden_dropout,
                max_position_embeddings=config.max_position_embeddings,
                use_flash=config.use_flash_attention,
                use_gqa=config.use_gqa,
                use_streaming_llm=config.use_streaming_llm,
                sink_size=config.sink_size,
                streaming_window_size=config.streaming_window_size,
            )

        # FFN/MoE 层选择
        if config.use_moe:
            # 使用 MoE (Mixture of Experts)
            expert_intermediate_size = config.expert_intermediate_size or config.intermediate_size
            self.mlp = DeepSeekMoE(
                hidden_size=config.hidden_size,
                intermediate_size=expert_intermediate_size,
                num_experts=config.num_experts,
                num_shared_experts=config.num_shared_experts,
                top_k=config.num_experts_per_tok,
                hidden_dropout=config.hidden_dropout,
                router_noise_std=config.router_noise_std,
                routed_scaling_factor=config.routed_scaling_factor,
                aux_loss_alpha=config.aux_loss_alpha,
            )
        else:
            # 使用标准 FFN
            self.mlp = SwiGLUFFN(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_dropout=config.hidden_dropout,
            )

        # Layer Norm (Pre-Norm架构)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """
        Args:
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            cos, sin: RoPE位置编码
            past_key_value: KV缓存
            use_cache: 是否使用缓存

        Returns:
            hidden_states: 输出张量
            present_key_value: KV缓存
            aux_loss: MoE 辅助损失（如果使用 MoE）
        """
        residual = hidden_states

        # Pre-Norm + Self-Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cos=cos,
            sin=sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Pre-Norm + FFN/MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MoE 返回 (output, aux_outputs) 元组
        aux_loss = None
        if self.config.use_moe:
            hidden_states, aux_outputs = self.mlp(hidden_states)
            aux_loss = aux_outputs.get("aux_loss") if aux_outputs else None
        else:
            hidden_states = self.mlp(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, aux_loss


class TransformerModel(nn.Module):
    """
    完整的Transformer模型
    GPT风格的Decoder-only架构
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # 嵌入层
        self.embedding = TransformerEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_position_embeddings=config.max_position_embeddings,
            dropout=config.hidden_dropout,
            position_embedding_type="rope",
            rope_theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )

        # Transformer层
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # 最终LayerNorm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]], torch.Tensor]:
        """
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            past_key_values: 过去的KV缓存列表
            use_cache: 是否使用KV缓存
        Returns:
            hidden_states: 最后一层隐藏状态 [batch_size, seq_len, hidden_size]
            present_key_values: 当前的KV缓存列表（如果 use_cache=True）
            aux_loss: MoE 辅助损失（如果不使用 MoE 则为 0）
        """
        batch_size, seq_len = input_ids.shape

        # 生成position_ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 嵌入
        hidden_states, (cos, sin) = self.embedding(input_ids, position_ids)

        # 准备注意力掩码
        if attention_mask is not None:
            # 扩展掩码形状 [batch, 1, 1, seq_len] 或 [batch, 1, seq_len, seq_len]
            attention_mask = self._prepare_attention_mask(attention_mask, hidden_states.dtype)

        # 准备KV缓存
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        present_key_values = []

        # MoE 辅助损失累加
        total_aux_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        # 逐层计算
        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values else None

            hidden_states, present_kv, aux_loss = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cos=cos,
                sin=sin,
                past_key_value=past_kv,
                use_cache=use_cache,
            )

            # 累加 MoE 辅助损失
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

            if use_cache:
                present_key_values.append(present_kv)

        # 最终LayerNorm
        hidden_states = self.norm(hidden_states)

        return hidden_states, present_key_values if use_cache else None, total_aux_loss

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        准备注意力掩码
        将2D掩码转换为4D，并添加因果掩码
        """
        batch_size, seq_len = attention_mask.shape

        # 因果掩码: 下三角矩阵
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=attention_mask.device, dtype=dtype),
            diagonal=1,
        )
        causal_mask = causal_mask * -1e9  # 负无穷

        # 结合padding掩码
        # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
        attention_mask = attention_mask[:, None, None, :].to(dtype)
        attention_mask = (1.0 - attention_mask) * -1e9

        # 组合因果掩码和padding掩码
        combined_mask = causal_mask[None, None, :, :] + attention_mask

        return combined_mask

    def get_input_embeddings(self) -> nn.Embedding:
        """获取输入嵌入层"""
        return self.embedding.token_embedding.embedding

    def set_input_embeddings(self, value: nn.Embedding):
        """设置输入嵌入层"""
        self.embedding.token_embedding.embedding = value


class CausalLMModel(nn.Module):
    """
    因果语言模型
    添加了语言模型头的完整模型
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Transformer主体
        self.model = TransformerModel(config)

        # 语言模型头
        if config.tie_word_embeddings:
            # 共享嵌入权重
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 标签平滑
        self.label_smoothing = 0.0

    def get_output_embeddings(self) -> Optional[nn.Linear]:
        """获取输出嵌入层"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        """设置输出嵌入层"""
        self.lm_head = new_embeddings

    def resize_token_embeddings(self, new_num_tokens: int):
        """
        调整token嵌入层的大小
        用于在加载预训练模型后扩展词表

        Args:
            new_num_tokens: 新的词表大小
        """
        old_embeddings = self.model.get_input_embeddings()
        old_num_tokens = old_embeddings.weight.size(0)

        if new_num_tokens == old_num_tokens:
            return

        # 创建新的嵌入层
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.weight.size(1))
        new_embeddings.to(old_embeddings.weight.device)

        # 复制旧权重
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy] = old_embeddings.weight.data[:num_tokens_to_copy]

        # 如果扩展，用旧权重的均值初始化新token
        if new_num_tokens > old_num_tokens:
            mean_weight = old_embeddings.weight.data.mean(dim=0)
            new_embeddings.weight.data[old_num_tokens:] = mean_weight

        # 替换嵌入层
        self.model.embedding.token_embedding.embedding = new_embeddings

        # 调整lm_head
        old_lm_head = self.lm_head
        self.lm_head = nn.Linear(old_lm_head.in_features, new_num_tokens, bias=False)
        self.lm_head.to(old_lm_head.weight.device)

        # 复制lm_head权重
        self.lm_head.weight.data[:num_tokens_to_copy] = old_lm_head.weight.data[:num_tokens_to_copy]
        if new_num_tokens > old_num_tokens:
            self.lm_head.weight.data[old_num_tokens:] = old_lm_head.weight.data.mean(dim=0)

        # 更新配置
        self.config.vocab_size = new_num_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> dict:
        """
        Args:
            input_ids: 输入token ID [batch_size, seq_len]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            labels: 标签(用于计算loss)
            past_key_values: KV缓存
            use_cache: 是否使用缓存
        Returns:
            dict包含:
                - logits: 输出logits [batch_size, seq_len, vocab_size]
                - loss: 交叉熵损失(如果提供了labels)
                - aux_loss: MoE 辅助损失(如果使用 MoE)
                - past_key_values: KV缓存(如果use_cache=True)
        """
        # Transformer前向传播
        hidden_states, past_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # 计算logits
        if self.config.tie_word_embeddings and self.lm_head is None:
            # 使用共享的嵌入权重
            logits = F.linear(hidden_states, self.model.get_input_embeddings().weight)
        else:
            logits = self.lm_head(hidden_states)

        # 准备输出
        output = {"logits": logits}

        # 计算损失
        if labels is not None:
            loss = self._compute_loss(logits, labels)
            # 如果使用 MoE，将辅助损失加到总损失中
            if self.config.use_moe and aux_loss is not None:
                loss = loss + self.config.aux_loss_alpha * aux_loss
            output["loss"] = loss

        # MoE 辅助损失（用于监控）
        if self.config.use_moe and aux_loss is not None:
            output["aux_loss"] = aux_loss

        if use_cache:
            output["past_key_values"] = past_key_values

        return output

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """计算交叉熵损失"""
        # 移位: 预测下一个token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 展平
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)

        # 交叉熵损失
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )

        return loss

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        为文本生成准备输入
        支持KV缓存的高效生成
        """
        # 如果有缓存，只需要处理最后一个token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        简单的文本生成方法
        支持多种采样策略
        """
        self.eval()

        for _ in range(max_new_tokens):
            # 前向传播
            outputs = self(input_ids, use_cache=False)
            logits = outputs["logits"]

            # 获取最后一个token的logits
            next_token_logits = logits[:, -1, :]

            # 温度缩放
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # 采样或贪婪解码
            if do_sample:
                # Top-k过滤
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float("Inf")

                # Top-p (nucleus) 过滤
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float("Inf")

                # 采样
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪解码
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 检查EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids

    @torch.no_grad()
    def generate_streaming(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        StreamingLLM 生成方法
        支持无限长度推理，固定显存占用

        使用滑动窗口 + Attention Sink 技术，可以生成任意长度的文本
        而不会随着序列增长而增加显存占用

        注意：需要设置 config.use_streaming_llm=True 才能启用 StreamingLLM
        """
        self.eval()

        # 如果未启用 StreamingLLM，使用标准 generate
        if not self.config.use_streaming_llm:
            return self.generate(
                input_ids, max_new_tokens, temperature, top_k, top_p, do_sample, None, eos_token_id
            )

        # 初始化 KV 缓存
        past_key_values = None

        for step in range(max_new_tokens):
            # 第一步处理整个 prompt，之后只处理最后一个 token
            if step == 0:
                step_input = input_ids
            else:
                step_input = input_ids[:, -1:]

            # 前向传播（使用 KV 缓存）
            outputs = self(
                input_ids=step_input,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # 更新 KV 缓存
            past_key_values = outputs.get("past_key_values")

            # 获取 logits
            logits = outputs["logits"][:, -1, :]

            # 温度缩放
            if temperature != 1.0:
                logits = logits / temperature

            # 采样
            if do_sample:
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # 检查 EOS
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

        return input_ids


def create_model(config: Optional[ModelConfig] = None, config_name: str = "tiny") -> CausalLMModel:
    """
    工厂函数: 创建模型
    """
    if config is None:
        config_factory = getattr(ModelConfig, config_name, ModelConfig.tiny)
        config = config_factory()

    return CausalLMModel(config)
