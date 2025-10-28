"""Custom Qwen transformer with padding-mask and per-sample RoPE support.

Multi-Resolution Training Support
==================================

This module extends the Qwen transformer to support multi-resolution training with per-sample
rotary position embeddings (RoPE).

Key Components:
--------------
1. QwenEmbedRopeBatched: 支持为每个样本生成独立的 RoPE
2. QwenDoubleStreamAttnProcessorPerSample: 支持应用 per-sample RoPE 的 attention processor
3. QwenImageTransformer2DModel: 自动检测模式（shared vs per-sample）的主模型

使用示例：
---------
    # Shared mode (原有用法，向后兼容)
    >>> model = QwenImageTransformer2DModel.from_pretrained(...)
    >>> img_shapes = [(1, 32, 32)]  # List of (F, H, W)
    >>> output = model(hidden_states, img_shapes=img_shapes, txt_seq_lens=[77])

    # Per-sample mode (多分辨率训练)
    >>> img_shapes_batched = [
    ...     [(1, 32, 32), (1, 64, 64)],  # Sample 0: 2 images
    ...     [(1, 48, 48), (1, 0, 0)],    # Sample 1: 1 image + padding
    ... ]
    >>> attention_mask = torch.ones(2, 77 + max_img_tokens, dtype=torch.bool)
    >>> output = model(
    ...     hidden_states,
    ...     img_shapes=img_shapes_batched,  # 2D list triggers per-sample mode
    ...     txt_seq_lens=[77, 77],
    ...     attention_mask=attention_mask,
    ... )

模式检测：
--------
- img_shapes: List[Tuple] → shared mode (所有样本共享同一个 RoPE)
- img_shapes: List[List[Tuple]] → per-sample mode (每个样本独立的 RoPE)

性能优化：
--------
- RoPE 缓存：相同尺寸的样本会复用已计算的 RoPE
- 自动降级：单样本或同尺寸 batch 自动使用 shared mode
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from qflux.models.transformer_qwenimage import (
    QwenEmbedRope as _QwenEmbedRope,
)
from qflux.models.transformer_qwenimage import (
    QwenImageTransformer2DModel as _QwenImageTransformer2DModel,
)
from qflux.models.transformer_qwenimage import (
    apply_rotary_emb_qwen,
)


logger = logging.getLogger(__name__)


class QwenEmbedRopeBatched(_QwenEmbedRope):
    """扩展版的 QwenEmbedRope，支持 per-sample RoPE

    主要改进：
    1. 支持 batched img_shapes 输入 (每个样本独立的形状列表)
    2. 为每个样本生成独立的 RoPE
    3. 支持 RoPE 缓存以提高性能
    4. 向后兼容原有的 shared mode
    """

    def __init__(self, theta: int, axes_dim: list[int], scale_rope: bool = False):
        super().__init__(theta, axes_dim, scale_rope)
        # rope_cache 已在父类初始化

    def forward_batched(
        self,
        img_shapes_batch: list[list[tuple[int, int, int]]],
        txt_seq_lens: list[int],
        device: torch.device,
        valid_img_lengths: torch.Tensor | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """为每个样本生成独立的 RoPE

        Args:
            img_shapes_batch: List of img_shapes for each sample
                Example: [[(1, 32, 32)], [(1, 64, 64)]] for 1 samples
            txt_seq_lens: Text sequence length for each sample
            device: Target device
            valid_img_lengths: Optional (B,) - 每个样本的实际图像 token 数量

        Returns:
            List of (img_freqs, txt_freqs) tuples, one per sample
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        batch_size = len(img_shapes_batch)
        rope_list = []

        for b in range(batch_size):
            img_shapes = [img_shapes_batch[b]]  # keep batch dimension
            txt_len = txt_seq_lens[b] if isinstance(txt_seq_lens, list) else txt_seq_lens

            # 计算该样本的 RoPE (复用父类逻辑)
            print("img_shapes in forward_batched", img_shapes)
            img_freqs, txt_freqs = super().forward(img_shapes, [txt_len], device)
            print("img_freqs", img_freqs.shape, txt_freqs.shape)

            rope_list.append((img_freqs, txt_freqs))

        return rope_list

    def forward(  # type: ignore
        self,
        img_shapes: list[tuple[int, int, int]] | list[list[tuple[int, int, int]]],
        txt_seq_lens: int | list[int],
        device: torch.device,
        **kwargs,
    ):
        """自动检测模式的 forward 方法

        Args:
            img_shapes:
                - List[Tuple]: shared mode, e.g. [(1, 32, 32)]
                - List[List[Tuple]]: per-sample mode, e.g. [[(1, 32, 32)], [(1, 64, 64)]]
        """
        # 检测是否为 per-sample mode
        is_per_sample = isinstance(img_shapes, list) and len(img_shapes) > 0 and isinstance(img_shapes[0], list)

        if is_per_sample:
            # Per-sample mode
            if not isinstance(txt_seq_lens, list):
                txt_seq_lens = [txt_seq_lens] * len(img_shapes)

            return self.forward_batched(img_shapes, txt_seq_lens, device, **kwargs)  # type: ignore[arg-type]
        else:
            # Shared mode - 调用父类方法
            return super().forward(img_shapes, txt_seq_lens, device)


class QwenDoubleStreamAttnProcessorPerSample:
    """支持 per-sample RoPE 的 Qwen Double Stream Attention Processor

    主要特性：
    1. 自动检测 RoPE 模式（shared vs per-sample）
    2. 为每个样本应用独立的 RoPE
    3. 与 attention mask 协同工作
    4. 向后兼容原有的 shared mode
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0")

    def _apply_rope_per_sample(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb_list: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """为每个样本应用独立的 RoPE

        优化版本：使用 contiguous 和 in-place 操作减少数值误差

        Args:
            query: [B, seq_txt + seq_img, H, D]
            key: [B, seq_txt + seq_img, H, D]
            image_rotary_emb_list: List of (img_freqs, txt_freqs) for each sample
        """
        batch_size = query.shape[0]
        total_seq_len = query.shape[1]

        # 使用 contiguous 确保内存布局连续，减少切片操作的开销
        query = query.contiguous()
        key = key.contiguous()

        # 预分配输出张量 - 使用 clone 确保 padding 部分也被初始化
        # 这样 padding 位置保持原始值（会被 attention mask 遮盖）
        query_out = query.clone()
        key_out = key.clone()

        for b in range(batch_size):
            img_freqs, txt_freqs = image_rotary_emb_list[b]

            # 计算序列长度
            seq_txt = txt_freqs.shape[0]
            seq_img = img_freqs.shape[0]

            # 确保不超出边界
            assert seq_txt + seq_img <= total_seq_len, (
                f"Sample {b}: seq_txt({seq_txt}) + seq_img({seq_img}) = {seq_txt + seq_img} "
                f"exceeds total_seq_len({total_seq_len})"
            )

            # 分别应用 text 和 image RoPE
            # Text part - 使用连续的切片减少内存碎片
            query_out[b : b + 1, :seq_txt] = apply_rotary_emb_qwen(
                query[b : b + 1, :seq_txt].contiguous(), txt_freqs, use_real=False
            )
            key_out[b : b + 1, :seq_txt] = apply_rotary_emb_qwen(
                key[b : b + 1, :seq_txt].contiguous(), txt_freqs, use_real=False
            )

            # Image part
            query_out[b : b + 1, seq_txt : seq_txt + seq_img] = apply_rotary_emb_qwen(
                query[b : b + 1, seq_txt : seq_txt + seq_img].contiguous(), img_freqs, use_real=False
            )
            key_out[b : b + 1, seq_txt : seq_txt + seq_img] = apply_rotary_emb_qwen(
                key[b : b + 1, seq_txt : seq_txt + seq_img].contiguous(), img_freqs, use_real=False
            )

            # Padding 部分保持不变（会被 attention mask 遮盖，不影响结果）

        return query_out, key_out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Attention forward with per-sample RoPE support

        Args:
            image_rotary_emb:
                - Shared mode: (img_freqs, txt_freqs) with shape (seq, D)
                - Per-sample mode: List of (img_freqs, txt_freqs)
        """
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessorPerSample requires encoder_hidden_states")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Concatenate for joint attention [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Apply RoPE - 关键修改点
        if image_rotary_emb is not None:
            if isinstance(image_rotary_emb, list):
                # Per-sample mode
                joint_query, joint_key = self._apply_rope_per_sample(joint_query, joint_key, image_rotary_emb)
            else:
                # Shared mode
                img_freqs, txt_freqs = image_rotary_emb

                # Apply to text part
                joint_query[:, :seq_txt] = apply_rotary_emb_qwen(joint_query[:, :seq_txt], txt_freqs, use_real=False)
                joint_key[:, :seq_txt] = apply_rotary_emb_qwen(joint_key[:, :seq_txt], txt_freqs, use_real=False)

                # Apply to image part
                joint_query[:, seq_txt:] = apply_rotary_emb_qwen(joint_query[:, seq_txt:], img_freqs, use_real=False)
                joint_key[:, seq_txt:] = apply_rotary_emb_qwen(joint_key[:, seq_txt:], img_freqs, use_real=False)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class QwenImageTransformer2DModel(_QwenImageTransformer2DModel):
    """Extension of QwenImageTransformer2DModel with padding masks and per-sample RoPE.

    主要功能：
    1. 支持 attention mask（padding token 处理）
    2. 支持 per-sample RoPE（多分辨率训练）
    3. 自动检测模式（shared vs per-sample）
    4. 向后兼容原有实现

    Multi-resolution training features:
    - Accepts img_shapes as List[List[Tuple]] for per-sample mode
    - Uses QwenEmbedRopeBatched to generate per-sample RoPE
    - Applies QwenDoubleStreamAttnProcessorPerSample when needed
    - Works with attention mask to handle padded tokens
    """

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: int | None = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

        # 替换为支持 batched 的 pos_embed
        if hasattr(self, "pos_embed"):
            self.pos_embed = QwenEmbedRopeBatched(
                theta=10000, axes_dim=list(self.config.axes_dims_rope), scale_rope=True
            )

        # 一次性替换所有 attention 的 processor
        for _name, module in self.named_modules():
            if isinstance(module, Attention) and hasattr(module, "add_q_proj"):
                # 这是双流 attention
                module.processor = QwenDoubleStreamAttnProcessorPerSample()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_hidden_states_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        img_shapes: list[tuple[int, int, int]] | list[list[tuple[int, int, int]]] | None = None,
        txt_seq_lens: int | list[int] | None = None,
        guidance: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        """Extended forward with per-sample RoPE and attention mask support

        Parameters
        ----------
        img_shapes:
            - List[Tuple]: shared mode - all samples use same shapes
            - List[List[Tuple]]: per-sample mode - each sample has independent shapes
        attention_mask:
            Optional padding mask over the concatenated `[text_tokens, image_tokens]` sequence.
            Shape `(batch, seq_txt + seq_img)`; `True/1` marks valid tokens.
        """
        # ============ 1. Handle attention_kwargs and LoRA ============
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            attention_kwargs = {}
            lora_scale = 1.0

        try:
            from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
        except ImportError:
            USE_PEFT_BACKEND = False

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        # ============ 2. Embeddings ============
        hidden_states = self.img_in(hidden_states)

        if timestep is not None:
            timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        # ============ 3. Generate RoPE (per-sample or shared) ============
        # 检测输入格式
        is_batched_format = (
            isinstance(img_shapes, list) and len(img_shapes) > 0 and isinstance(img_shapes[0], list)
            if img_shapes
            else False
        )

        if is_batched_format:
            # 输入格式为 List[List[Tuple]]
            batch_size = len(img_shapes)  # type: ignore[arg-type]

            # 确保 txt_seq_lens 是列表
            if not isinstance(txt_seq_lens, list):
                txt_seq_lens = [txt_seq_lens] * batch_size  # type: ignore[list-item]

            # 检查是否所有样本的 img_shapes 都完全相同
            # 如果相同，使用原始的 shared mode 以保证数值一致性
            all_same = all(shapes == img_shapes[0] for shapes in img_shapes) if img_shapes else False

            if all_same:
                # 所有样本形状相同 -> 使用原始的 shared mode
                # 传入完整的 img_shapes，原始模型会取 img_shapes[0] 来处理
                # 这样可以保证与原始实现完全一致
                image_rotary_emb = _QwenEmbedRope.forward(
                    self.pos_embed, img_shapes, txt_seq_lens, device=hidden_states.device
                )
            else:
                # 样本形状不同 -> 使用 per-sample mode
                image_rotary_emb = self.pos_embed.forward_batched(img_shapes, txt_seq_lens, device=hidden_states.device)
        else:
            # 输入格式为 List[Tuple] -> 原始的 shared mode
            image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        # ============ 4. Handle attention mask ============
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError("attention_mask must have shape (batch, total_sequence_length).")

            mask = attention_mask
            if mask.dtype != torch.bool:
                mask = mask > 0
            mask = mask.to(hidden_states.device)

            batch_size, total_len = mask.shape
            seq_img = hidden_states.shape[1]
            seq_txt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0

            expected_len = seq_txt + seq_img
            if total_len < expected_len:
                raise ValueError(
                    f"attention_mask length {total_len} is smaller than expected sequence length {expected_len}."
                )

            # Zero-out padded queries
            text_mask = mask[:, :seq_txt]
            img_mask = mask[:, seq_txt : seq_txt + seq_img]

            if encoder_hidden_states is not None and seq_txt > 0:
                encoder_hidden_states = encoder_hidden_states.masked_fill(~text_mask.unsqueeze(-1), 0)
            hidden_states = hidden_states.masked_fill(~img_mask.unsqueeze(-1), 0)

            # Create additive mask for attention
            additive_mask = torch.zeros(
                batch_size,
                1,
                1,
                total_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            additive_mask = additive_mask.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float("-inf"))
            attention_kwargs["attention_mask"] = additive_mask

        # ============ 5. Transformer blocks ============
        for _index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

            # Mask padding after each block to prevent residual/MLP leakage
            # This is critical for numerical accuracy with padding
            if attention_mask is not None:
                seq_img_block = hidden_states.shape[1]
                seq_txt_block = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
                img_mask_block = attention_mask[:, seq_txt_block : seq_txt_block + seq_img_block]

                # Convert to bool if needed
                if img_mask_block.dtype != torch.bool:
                    img_mask_block = img_mask_block > 0

                # Skip masking if all True (no actual padding) to avoid unnecessary computation
                if not img_mask_block.all():
                    img_mask_block = img_mask_block.to(hidden_states.device)
                    hidden_states = hidden_states * img_mask_block.unsqueeze(-1)

        # ============ 6. Output projection ============
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # Mask padding positions in output if attention_mask was provided
        if attention_mask is not None:
            # Recompute img_mask for output masking
            seq_img = output.shape[1]
            seq_txt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            img_mask = attention_mask[:, seq_txt : seq_txt + seq_img]
            output = output.masked_fill(~img_mask.unsqueeze(-1), 0)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
