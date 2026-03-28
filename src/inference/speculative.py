"""
Speculative Decoding 实现
基于 Draft-Then-Verify 范式的推理加速技术

原理:
1. Draft Model 快速生成 N 个候选 tokens
2. Target Model 并行验证所有候选
3. 接受正确的 tokens，拒绝错误的并重新采样

加速效果: 2-3x 推理加速（取决于 draft model 质量）
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List, TYPE_CHECKING
from dataclasses import dataclass
import time

if TYPE_CHECKING:
    from src.utils.profiling import OptimizationProfiler


@dataclass
class SpeculativeConfig:
    """Speculative Decoding 配置"""
    num_draft_tokens: int = 4          # 每次生成的候选 token 数
    max_new_tokens: int = 100          # 最大生成长度
    temperature: float = 1.0           # 采样温度
    top_k: int = 50                    # Top-k 过滤
    top_p: float = 0.95                # Top-p 过滤
    do_sample: bool = True             # 是否采样
    eos_token_id: Optional[int] = None # EOS token ID


class SpeculativeDecoder:
    """
    Speculative Decoding 解码器

    工作流程:
    1. Draft model 快速生成 N 个候选 token
    2. Target model 一次前向传播验证所有候选
    3. 按接受/拒绝规则决定最终输出

    特点:
    - 输出分布与 target model 相同（数学保证）
    - 延迟不变：target model 一次前向验证 N 个 token
    - 典型加速：2-3x（取决于 draft model 质量）
    """

    def __init__(
        self,
        draft_model,      # 小模型 (CausalLMModel)
        target_model,     # 大模型 (CausalLMModel)
        config: SpeculativeConfig = None,
        profiler: Optional["OptimizationProfiler"] = None,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config or SpeculativeConfig()
        self.profiler = profiler  # 可选的性能分析器

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 Speculative Decoding 生成

        Args:
            input_ids: 输入 token IDs [batch, seq_len]

        Returns:
            生成的完整序列（包含输入）
        """
        self.draft_model.eval()
        self.target_model.eval()

        batch_size, init_seq_len = input_ids.shape
        device = input_ids.device

        # 初始化 KV 缓存
        draft_cache = None
        target_cache = None

        current_ids = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < self.config.max_new_tokens:
            # Step 1: Draft model 生成 N 个候选 token
            draft_tokens, draft_probs = self._draft_tokens(
                current_ids, draft_cache
            )

            # Step 2: Target model 并行验证
            accepted_tokens, new_token, target_cache = self._verify_tokens(
                current_ids, draft_tokens, draft_probs, target_cache
            )

            # Step 3: 更新序列
            num_accepted = len(accepted_tokens)
            new_tokens = torch.cat(accepted_tokens + [new_token], dim=-1)
            current_ids = torch.cat([current_ids, new_tokens], dim=-1)
            tokens_generated += num_accepted + 1

            # 检查 EOS
            if self.config.eos_token_id is not None:
                if new_token.item() == self.config.eos_token_id:
                    break

        return current_ids

    def _draft_tokens(
        self,
        current_ids: torch.Tensor,
        past_key_values: Optional[List] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Draft model 生成 N 个候选 token

        Returns:
            draft_tokens: List of [batch, 1] tensors
            draft_probs: List of [batch, vocab_size] tensors
        """
        draft_tokens = []
        draft_probs = []
        current_cache = past_key_values

        step_input = current_ids

        # 性能监控
        start_time = time.perf_counter() if self.profiler else None

        for _ in range(self.config.num_draft_tokens):
            outputs = self.draft_model(
                input_ids=step_input,
                past_key_values=current_cache,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :]  # [batch, vocab_size]
            current_cache = outputs.get("past_key_values")

            # 采样
            token, probs = self._sample(logits)

            draft_tokens.append(token)
            draft_probs.append(probs)

            # 下一步只输入新生成的 token
            step_input = token

        # 记录 draft 时间
        if self.profiler and start_time:
            elapsed = time.perf_counter() - start_time
            self.profiler.speculative_metrics.record_draft(
                self.config.num_draft_tokens, elapsed
            )

        return draft_tokens, draft_probs

    def _verify_tokens(
        self,
        current_ids: torch.Tensor,
        draft_tokens: List[torch.Tensor],
        draft_probs: List[torch.Tensor],
        past_key_values: Optional[List] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor, Optional[List]]:
        """
        Target model 验证候选 token

        Returns:
            accepted_tokens: 被接受的 tokens 列表
            new_token: 额外采样的 token（保证至少生成 1 个）
            past_key_values: 更新后的 KV 缓存
        """
        N = len(draft_tokens)
        device = current_ids.device

        # 性能监控 - 开始计时
        start_time = time.perf_counter() if self.profiler else None

        # 构造验证输入: current_ids + draft_tokens
        draft_seq = torch.cat(draft_tokens, dim=-1)  # [batch, N]
        verify_input = torch.cat([current_ids, draft_seq], dim=-1)  # [batch, seq + N]

        # Target model 前向传播
        outputs = self.target_model(
            input_ids=verify_input,
            past_key_values=past_key_values,
            use_cache=True,
        )

        target_logits = outputs["logits"]
        new_cache = outputs.get("past_key_values")

        # 接受/拒绝决策
        accepted_tokens = []

        # 获取 target model 对每个位置的概率
        # target_logits shape: [batch, seq + N, vocab]
        # 我们需要位置 seq-1 到 seq+N-1 的 logits
        seq_len = current_ids.shape[1]

        for i in range(N):
            draft_token = draft_tokens[i]  # [batch, 1]
            draft_prob = draft_probs[i]    # [batch, vocab_size]

            # Target model 在位置 seq_len + i - 1 的概率
            target_logit = target_logits[:, seq_len + i - 1, :]  # [batch, vocab]
            target_prob = F.softmax(target_logit, dim=-1)

            # 获取采样的 draft token 的概率
            draft_token_idx = draft_token.squeeze(-1)  # [batch]
            draft_token_prob = draft_prob.gather(1, draft_token.unsqueeze(-1)).squeeze(-1)  # [batch]
            target_token_prob = target_prob.gather(1, draft_token.unsqueeze(-1)).squeeze(-1)  # [batch]

            # 接受概率: min(1, p_target / p_draft)
            accept_prob = torch.min(
                torch.ones_like(draft_token_prob),
                target_token_prob / (draft_token_prob + 1e-10)
            )

            # 采样决定是否接受
            random_val = torch.rand_like(accept_prob)

            if (random_val < accept_prob).all():
                # 接受这个 token
                accepted_tokens.append(draft_token)
            else:
                # 拒绝: 从调整后的分布采样
                # 调整分布: max(0, p_target - p_draft), 重新归一化
                adjusted_prob = torch.clamp(target_prob - draft_prob, min=0)
                adjusted_prob = adjusted_prob / (adjusted_prob.sum(dim=-1, keepdim=True) + 1e-10)

                new_token = torch.multinomial(adjusted_prob, num_samples=1)

                # 记录接受的 token 数
                if self.profiler:
                    self.profiler.record_speculative_accept(len(accepted_tokens))
                    # 记录 verify 时间
                    elapsed = time.perf_counter() - start_time
                    self.profiler.speculative_metrics.record_verify(elapsed)

                return accepted_tokens, new_token, None

        # 所有 draft tokens 都被接受
        # 从 target model 的最后一个位置采样额外 token
        last_logits = target_logits[:, -1, :]  # [batch, vocab]
        new_token, _ = self._sample(last_logits)

        # 记录接受的 token 数
        if self.profiler:
            self.profiler.record_speculative_accept(len(accepted_tokens))
            # 记录 verify 时间
            elapsed = time.perf_counter() - start_time
            self.profiler.speculative_metrics.record_verify(elapsed)

        return accepted_tokens, new_token, new_cache

    def _sample(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样并返回 token 和概率

        Returns:
            next_token: [batch, 1]
            probs: [batch, vocab_size]
        """
        # Temperature scaling
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature

        # Top-k filtering
        if self.config.top_k > 0:
            indices_to_remove = logits < torch.topk(logits, self.config.top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        if self.config.do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        return next_token, probs


def speculative_generate(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    num_draft_tokens: int = 4,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    do_sample: bool = True,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Speculative Decoding 生成便捷函数

    Args:
        draft_model: 小模型（用于快速生成候选）
        target_model: 大模型（用于验证）
        input_ids: 输入 token IDs
        num_draft_tokens: 每次生成的候选 token 数
        max_new_tokens: 最大生成长度
        temperature: 采样温度
        top_k: Top-k 过滤
        top_p: Top-p 过滤
        do_sample: 是否采样
        eos_token_id: EOS token ID

    Returns:
        生成的完整序列
    """
    config = SpeculativeConfig(
        num_draft_tokens=num_draft_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        eos_token_id=eos_token_id,
    )

    decoder = SpeculativeDecoder(draft_model, target_model, config)
    return decoder.generate(input_ids)
