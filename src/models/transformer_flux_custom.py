"""Custom Flux transformer with padding-mask and per-sample RoPE support.

Multi-Resolution Training Support
==================================

This module extends the Flux transformer to support multi-resolution training with per-sample
rotary position embeddings (RoPE).

Key Components:
--------------
1. FluxPosEmbedBatched: 支持为每个样本生成独立的 RoPE
2. FluxAttnProcessorPerSample: 支持应用 per-sample RoPE 的 attention processor
3. FluxTransformer2DModel: 自动检测模式（shared vs per-sample）的主模型

使用示例：
---------
    # Shared mode (原有用法，向后兼容)
    >>> model = FluxTransformer2DModel.from_pretrained(...)
    >>> img_ids = torch.zeros(1024, 3)  # (seq, 3)
    >>> output = model(hidden_states, img_ids=img_ids)

    # Per-sample mode (多分辨率训练)
    >>> img_ids_batched = torch.zeros(4, 1024, 3)  # (B, seq, 3)
    >>> attention_mask = torch.ones(4, 1024 + txt_len, dtype=torch.bool)
    >>> output = model(
    ...     hidden_states,
    ...     img_ids=img_ids_batched,  # 3D tensor triggers per-sample mode
    ...     attention_mask=attention_mask,
    ... )

模式检测：
--------
- img_ids.ndim == 2: shared mode (所有样本共享同一个 RoPE)
- img_ids.ndim == 3: per-sample mode (每个样本独立的 RoPE)

性能优化：
--------
- RoPE 缓存：相同尺寸的样本会复用已计算的 RoPE
- 自动降级：单样本或同尺寸 batch 自动使用 shared mode
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn.functional as F

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (  # type: ignore
    FluxTransformer2DModel as _FluxTransformer2DModel,
    FluxPosEmbed as _FluxPosEmbed,
    FluxAttention,
    _get_qkv_projections,
)
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_dispatch import dispatch_attention_fn

logger = logging.getLogger(__name__)


class FluxPosEmbedBatched(_FluxPosEmbed):
    """扩展版的 FluxPosEmbed，支持 per-sample RoPE

    主要改进：
    1. 支持 batched ids 输入 (B, seq, 3)
    2. 为每个样本生成独立的 RoPE
    3. 支持 RoPE 缓存以提高性能
    4. 向后兼容原有的 shared mode
    """

    def __init__(self, theta: int, axes_dim: List[int], enable_cache: bool = True):
        super().__init__(theta, axes_dim)
        self.enable_cache = enable_cache
        self.rope_cache = {} if enable_cache else None

    def _infer_hw_from_ids(self, ids: torch.Tensor, valid_length: Optional[int] = None) -> Tuple[int, int]:
        """从 img_ids 推断图像的 height 和 width"""
        if valid_length is not None:
            ids = ids[:valid_length]

        h = ids[:, 1].max().item() + 1
        w = ids[:, 2].max().item() + 1
        return int(h), int(w)

    def _compute_rope_cached(
        self, ids: torch.Tensor, cache_key: Optional[Tuple[int, int]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算 RoPE，支持缓存"""
        if self.enable_cache and cache_key is not None:
            if cache_key in self.rope_cache:
                return self.rope_cache[cache_key]

        freqs_cos, freqs_sin = super().forward(ids)

        if self.enable_cache and cache_key is not None:
            self.rope_cache[cache_key] = (freqs_cos, freqs_sin)

        return freqs_cos, freqs_sin

    def forward_batched(
        self,
        ids: torch.Tensor,
        valid_lengths: Optional[torch.Tensor] = None,
        pad_to_max: bool = True,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """为每个样本生成独立的 RoPE

        Args:
            ids: (B, seq, 3) - batched image IDs
            valid_lengths: (B,) - 每个样本的实际序列长度（不含padding）
            pad_to_max: 是否将 RoPE padding 到 max_seq 长度

        Returns:
            List of (freqs_cos, freqs_sin) tuples, one per sample
        """
        batch_size = ids.shape[0]
        max_seq = ids.shape[1]
        rope_list = []

        for b in range(batch_size):
            ids_b = ids[b]

            if valid_lengths is not None:
                valid_len = int(valid_lengths[b].item())
            else:
                non_zero_rows = (ids_b != 0).any(dim=1)
                valid_len = non_zero_rows.sum().item() if non_zero_rows.any() else max_seq

            ids_valid = ids_b[:valid_len]

            h, w = self._infer_hw_from_ids(ids_valid)
            cache_key = (h, w) if self.enable_cache else None

            freqs_cos, freqs_sin = self._compute_rope_cached(ids_valid, cache_key)

            if pad_to_max and valid_len < max_seq:
                pad_len = max_seq - valid_len
                D = freqs_cos.shape[-1]

                pad_zeros_cos = torch.zeros(pad_len, D, device=freqs_cos.device, dtype=freqs_cos.dtype)
                pad_zeros_sin = torch.zeros(pad_len, D, device=freqs_sin.device, dtype=freqs_sin.dtype)

                freqs_cos = torch.cat([freqs_cos, pad_zeros_cos], dim=0)
                freqs_sin = torch.cat([freqs_sin, pad_zeros_sin], dim=0)

            rope_list.append((freqs_cos, freqs_sin))

        return rope_list

    def forward(self, ids: torch.Tensor, **kwargs):
        """自动检测模式的 forward 方法"""
        if ids.ndim == 3:
            return self.forward_batched(ids, **kwargs)
        elif ids.ndim == 2:
            return super().forward(ids)
        else:
            raise ValueError(f"Expected ids to have 2 or 3 dimensions, got {ids.ndim}")

    def clear_cache(self):
        """清空 RoPE 缓存"""
        if self.rope_cache is not None:
            self.rope_cache.clear()


class FluxAttnProcessorPerSample:
    """支持 per-sample RoPE 的 Flux Attention Processor

    主要特性：
    1. 自动检测 RoPE 模式（shared vs per-sample）
    2. 为每个样本应用独立的 RoPE
    3. 与 attention mask 协同工作
    4. 向后兼容原有的 shared mode
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                f"{self.__class__.__name__} requires PyTorch 2.0. " "Please upgrade your pytorch version."
            )

    def _apply_rope_per_sample(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb_list: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """为每个样本应用独立的 RoPE

        优化版本：直接在原始张量上操作，避免切片和拼接带来的数值误差
        """
        batch_size = query.shape[0]

        # 预分配输出张量，避免多次拼接操作
        query_out = torch.empty_like(query)
        key_out = torch.empty_like(key)

        for b in range(batch_size):
            freqs_cos, freqs_sin = image_rotary_emb_list[b]
            rope_tuple = (freqs_cos, freqs_sin)

            # 直接对切片应用 RoPE，然后写入预分配的输出张量
            # 这样可以避免创建临时张量和拼接操作
            query_out[b:b+1] = apply_rotary_emb(query[b:b+1], rope_tuple, sequence_dim=1)
            key_out[b:b+1] = apply_rotary_emb(key[b:b+1], rope_tuple, sequence_dim=1)

        return query_out, key_out

    def __call__(
        self,
        attn: FluxAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[
            Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
        ] = None,
    ) -> torch.Tensor:
        """Attention forward with per-sample RoPE support

        Args:
            image_rotary_emb:
                - Shared mode: (freqs_cos, freqs_sin) with shape (seq, D)
                - Per-sample mode: List of (freqs_cos, freqs_sin)
        """
        # 检查 attn 模块是否有 per-sample RoPE 引用
        if hasattr(attn, '_per_sample_rope_ref'):
            image_rotary_emb = attn._per_sample_rope_ref

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None and encoder_hidden_states is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        # Apply RoPE - 关键修改点
        if image_rotary_emb is not None:
            if isinstance(image_rotary_emb, list):
                # Per-sample mode
                query, key = self._apply_rope_per_sample(query, key, image_rotary_emb)
            else:
                # Shared mode
                query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            seq_txt = encoder_hidden_states.shape[1]
            encoder_hidden_states_out = hidden_states[:, :seq_txt]
            hidden_states = hidden_states[:, seq_txt:]

            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states, encoder_hidden_states_out
        else:
            return hidden_states


class FluxTransformer2DModel(_FluxTransformer2DModel):
    """Extension of diffusers' :class:`FluxTransformer2DModel` with padding masks and per-sample RoPE.

    主要功能：
    1. 支持 attention mask（padding token 处理）
    2. 支持 per-sample RoPE（多分辨率训练）
    3. 自动检测模式（shared vs per-sample）
    4. 向后兼容原有实现

    Multi-resolution training features:
    - Accepts img_ids with shape (B, seq, 3) for per-sample mode
    - Uses FluxPosEmbedBatched to generate per-sample RoPE
    - Applies FluxAttnProcessorPerSample when needed
    - Works with attention mask to handle padded tokens
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 替换为支持 batched 的 pos_embed
        if hasattr(self, "pos_embed"):
            self.pos_embed = FluxPosEmbedBatched(theta=10000, axes_dim=list(self.config.axes_dims_rope))

    def _enable_per_sample_processors(self, per_sample_rope=None):
        """临时启用 per-sample RoPE processors

        Args:
            per_sample_rope: List of per-sample RoPE tuples to inject into attention modules
        """
        original_processors = {}
        for name, module in self.named_modules():
            if isinstance(module, FluxAttention):
                original_processors[name] = module.processor
                module.processor = FluxAttnProcessorPerSample()
                # 将 per-sample RoPE 存储在 attention 模块上
                if per_sample_rope is not None:
                    module._per_sample_rope_ref = per_sample_rope
        return original_processors

    def _restore_processors(self, original_processors: Dict[str, Any]):
        """恢复原始 processors 并清理临时属性"""
        for name, module in self.named_modules():
            if name in original_processors:
                module.processor = original_processors[name]
                # 清理临时 RoPE 引用
                if hasattr(module, '_per_sample_rope_ref'):
                    delattr(module, '_per_sample_rope_ref')

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        pooled_projections: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        img_ids: Optional[torch.Tensor] = None,
        txt_ids: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples: Optional[torch.Tensor] = None,
        controlnet_single_block_samples: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
        """Extended forward with per-sample RoPE support

        Parameters
        ----------
        img_ids:
            - Shape (seq, 3): shared mode - all samples use same position encoding
            - Shape (B, seq, 3): per-sample mode - each sample has independent position encoding
        attention_mask:
            Optional padding mask over the concatenated `[text_tokens, image_tokens]` sequence.
            Shape `(batch, seq_txt + seq_img)`; `True/1` marks valid tokens.
        """

        joint_attention_kwargs = dict(joint_attention_kwargs or {})

        # 检测 per-sample RoPE 模式
        per_sample_mode = img_ids is not None and img_ids.ndim == 3
        original_processors = None

        if per_sample_mode:
            # 生成 per-sample RoPE
            # 从 attention_mask 推断 valid_lengths
            if attention_mask is not None:
                batch_size = img_ids.shape[0]
                seq_img = img_ids.shape[1]
                seq_txt = 0
                if txt_ids is not None:
                    seq_txt = txt_ids.shape[1] if txt_ids.ndim == 3 else txt_ids.shape[0]

                # 计算每个样本的有效图像 token 数量
                img_mask = attention_mask[:, seq_txt:seq_txt + seq_img]
                valid_img_lengths = img_mask.sum(dim=1)
            else:
                valid_img_lengths = None

            # 生成 per-sample img RoPE
            img_rotary_emb_list = self.pos_embed(img_ids, valid_lengths=valid_img_lengths)

            # 生成 txt RoPE（shared across all samples）
            if txt_ids is not None:
                if txt_ids.ndim == 3:
                    txt_ids_2d = txt_ids[0]
                else:
                    txt_ids_2d = txt_ids
                txt_rotary_emb = self.pos_embed(txt_ids_2d)
            else:
                txt_rotary_emb = None

            # 组合 txt + img RoPE for each sample
            if txt_rotary_emb is not None:
                combined_rope_list = []
                for img_rope in img_rotary_emb_list:
                    # Concatenate text and image RoPE
                    combined_cos = torch.cat([txt_rotary_emb[0], img_rope[0]], dim=0)
                    combined_sin = torch.cat([txt_rotary_emb[1], img_rope[1]], dim=0)
                    combined_rope_list.append((combined_cos, combined_sin))
                per_sample_rope = combined_rope_list
            else:
                per_sample_rope = img_rotary_emb_list

            # 启用 per-sample attention processors 并注入 RoPE
            original_processors = self._enable_per_sample_processors(per_sample_rope)

            # 将 img_ids 转为 2D 以避免父类警告（父类会忽略它，因为我们已经生成了 RoPE）
            img_ids = img_ids[0]
            if txt_ids is not None and txt_ids.ndim == 3:
                txt_ids = txt_ids[0]

        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must have shape (batch, total_sequence_length).")

            mask = attention_mask
            if mask.dtype != torch.bool:
                mask = mask > 0
            mask = mask.to(hidden_states.device)

            batch_size, total_len = mask.shape
            seq_img = hidden_states.shape[1]
            seq_txt = 0
            if txt_ids is not None:
                seq_txt = txt_ids.shape[1] if txt_ids.ndim == 3 else txt_ids.shape[0]

            expected_len = seq_txt + seq_img
            if total_len < expected_len:
                raise ValueError(
                    f"attention_mask length {total_len} is smaller than expected sequence length {expected_len}."
                )

            # Zero-out padded queries so they do not influence attention scores
            img_mask = mask[:, seq_txt: seq_txt + seq_img]
            hidden_states = hidden_states.masked_fill(~img_mask.unsqueeze(-1), 0)
            if encoder_hidden_states is not None and seq_txt > 0:
                text_mask = mask[:, :seq_txt]
                encoder_hidden_states = encoder_hidden_states.masked_fill(~text_mask.unsqueeze(-1), 0)

            additive_mask = torch.zeros(
                batch_size,
                1,
                1,
                total_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            additive_mask = additive_mask.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float("-inf"))
            joint_attention_kwargs["attention_mask"] = additive_mask

        result = super().forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=return_dict,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
        )

        # 恢复原始 processors（会自动清理临时 RoPE 引用）
        if per_sample_mode and original_processors is not None:
            self._restore_processors(original_processors)

        return result
