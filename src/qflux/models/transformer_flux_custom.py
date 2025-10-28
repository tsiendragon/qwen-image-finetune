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

import logging
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_dispatch import dispatch_attention_fn

# from diffusers.models.modeling_outputs import Transformer2DModelOutput
# from diffusers.models.transformers.transformer_flux import (  # type: ignore
#     FluxTransformer2DModel as _FluxTransformer2DModel,
#     FluxPosEmbed as _FluxPosEmbed,
#     FluxAttention,
#     _get_qkv_projections,
# )
from diffusers.models.embeddings import apply_rotary_emb

from qflux.models.transformer_flux import FluxAttention, Transformer2DModelOutput, _get_qkv_projections
from qflux.models.transformer_flux import FluxPosEmbed as _FluxPosEmbed
from qflux.models.transformer_flux import FluxTransformer2DModel as _FluxTransformer2DModel


logger = logging.getLogger(__name__)


class FluxPosEmbedBatched(_FluxPosEmbed):
    """扩展版的 FluxPosEmbed，支持 per-sample RoPE

    主要改进：
    1. 支持 batched ids 输入 (B, seq, 3)
    2. 为每个样本生成独立的 RoPE
    3. 支持 RoPE 缓存以提高性能
    4. 向后兼容原有的 shared mode
    """

    def __init__(self, theta: int, axes_dim: list[int], enable_cache: bool = True):
        super().__init__(theta, axes_dim)
        self.enable_cache = enable_cache
        self.rope_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] | None = {} if enable_cache else None

    def _infer_hw_from_ids(self, ids: torch.Tensor, valid_length: int | None = None) -> tuple[int, int]:
        """从 img_ids 推断图像的 height 和 width"""
        if valid_length is not None:
            ids = ids[:valid_length]

        h = ids[:, 1].max().item() + 1
        w = ids[:, 2].max().item() + 1
        return int(h), int(w)

    def _compute_rope_cached(
        self, ids: torch.Tensor, cache_key: tuple[int, int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 RoPE，支持缓存"""
        if self.enable_cache and cache_key is not None and self.rope_cache is not None:
            if cache_key in self.rope_cache:
                return self.rope_cache[cache_key]

        freqs_cos, freqs_sin = super().forward(ids)

        if self.enable_cache and cache_key is not None and self.rope_cache is not None:
            self.rope_cache[cache_key] = (freqs_cos, freqs_sin)

        return freqs_cos, freqs_sin

    def forward_batched(
        self,
        ids: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
        pad_to_max: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
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

                # Padding should be identity rotation: cos=1, sin=0
                # This ensures apply_rotary_emb doesn't modify padding positions
                pad_identity_cos = torch.ones(pad_len, D, device=freqs_cos.device, dtype=freqs_cos.dtype)
                pad_identity_sin = torch.zeros(pad_len, D, device=freqs_sin.device, dtype=freqs_sin.dtype)

                freqs_cos = torch.cat([freqs_cos, pad_identity_cos], dim=0)
                freqs_sin = torch.cat([freqs_sin, pad_identity_sin], dim=0)

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
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

    def _apply_rope_per_sample(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        image_rotary_emb_list: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            query_out[b : b + 1] = apply_rotary_emb(query[b : b + 1], rope_tuple, sequence_dim=1)
            key_out[b : b + 1] = apply_rotary_emb(key[b : b + 1], rope_tuple, sequence_dim=1)

        return query_out, key_out

    def __call__(
        self,
        attn: FluxAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """Attention forward with per-sample RoPE support

        Args:
            image_rotary_emb:
                - Shared mode: (freqs_cos, freqs_sin) with shape (seq, D)
                - Per-sample mode: List of (freqs_cos, freqs_sin)
        """
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
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
                # Per-sample mode - but check if all samples have identical RoPE
                # If so, fall back to shared mode for better numerical stability
                all_same = True
                if len(image_rotary_emb) > 1:
                    first_cos, first_sin = image_rotary_emb[0]
                    for cos, sin in image_rotary_emb[1:]:
                        if not (torch.equal(cos, first_cos) and torch.equal(sin, first_sin)):
                            all_same = False
                            break

                if all_same:
                    # All samples have identical RoPE - use shared mode for numerical stability
                    query = apply_rotary_emb(query, image_rotary_emb[0], sequence_dim=1)
                    key = apply_rotary_emb(key, image_rotary_emb[0], sequence_dim=1)
                else:
                    # Samples have different RoPE - use per-sample mode
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
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
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

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: tuple[int, int, int] = (16, 56, 56),
    ):
        print("patch_size", patch_size)
        print("in_channels", in_channels)
        print("out_channels", out_channels)
        print("num_layers", num_layers)
        print("num_single_layers", num_single_layers)
        print("attention_head_dim", attention_head_dim)
        print("num_attention_heads", num_attention_heads)
        print("joint_attention_dim", joint_attention_dim)
        print("pooled_projection_dim", pooled_projection_dim)
        print("guidance_embeds", guidance_embeds)
        print("axes_dims_rope", axes_dims_rope)

        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )
        # 替换为支持 batched 的 pos_embed
        if hasattr(self, "pos_embed"):
            self.pos_embed = FluxPosEmbedBatched(theta=10000, axes_dim=list(self.config.axes_dims_rope))

        #  一次性替换所有 attention 的 processor；训练全程固定
        for _, module in self.named_modules():
            if isinstance(module, FluxAttention):
                module.processor = FluxAttnProcessorPerSample()

    def forward(  # noqa: F811
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        img_ids: torch.Tensor | None = None,
        txt_ids: torch.Tensor | None = None,
        guidance: torch.Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: torch.Tensor | None = None,
        controlnet_single_block_samples: torch.Tensor | None = None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
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
        # ============ 1. Handle joint_attention_kwargs and LoRA ============
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            joint_attention_kwargs = {}
            lora_scale = 1.0

        try:
            from diffusers.loaders.peft import USE_PEFT_BACKEND
            from peft.tuners.tuners_utils import scale_lora_layers, unscale_lora_layers
        except ImportError:
            USE_PEFT_BACKEND = False

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        # ============ 2. Embeddings ============
        hidden_states = self.x_embedder(hidden_states)

        # Immediately mask padding after x_embedder to prevent bias leakage
        # CRITICAL: Must mask here because Linear layer bias makes zero inputs non-zero
        # Only apply if there's actual padding (not all True)
        if attention_mask is not None:
            seq_img = hidden_states.shape[1]
            # Calculate text sequence length from txt_ids (NOT encoder_hidden_states which hasn't been embedded yet)
            if txt_ids is not None:
                seq_txt = txt_ids.shape[1] if txt_ids.ndim == 3 else txt_ids.shape[0]
            else:
                seq_txt = 0

            # Extract and apply image mask
            img_mask_early = attention_mask[:, seq_txt : seq_txt + seq_img]
            if img_mask_early.dtype != torch.bool:
                img_mask_early = img_mask_early > 0
            # Skip masking if all True (no actual padding)
            if not img_mask_early.all():
                img_mask_early = img_mask_early.to(hidden_states.device)
                hidden_states = hidden_states.masked_fill(~img_mask_early.unsqueeze(-1), 0)

        if timestep is not None:
            timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        # return Transformer2DModelOutput(sample=temb.unsqueeze(2).unsqueeze(2))

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        # ============ 3. Generate RoPE (per-sample or shared) ============
        # Use per-sample mode if:
        # 1. img_ids is 3D (explicit per-sample ids), OR
        # 2. attention_mask is present (indicating variable-length sequences)
        per_sample_mode = (img_ids is not None and img_ids.ndim == 3) or attention_mask is not None

        batch_size = hidden_states.shape[0]
        if per_sample_mode:
            # Ensure img_ids is 3D for per-sample processing
            if img_ids is not None:
                if img_ids.ndim == 2:
                    # img_ids is (seq, 3), expand to (batch, seq, 3)
                    img_ids = img_ids.unsqueeze(0).expand(batch_size, -1, -1)
                elif img_ids.ndim == 3:
                    # Check if all samples have identical ids
                    # If so and no attention_mask, downgrade to shared mode for numerical stability
                    if attention_mask is None and img_ids.shape[0] > 1:
                        all_ids_same = True
                        first_ids = img_ids[0]
                        for i in range(1, img_ids.shape[0]):
                            if not torch.equal(img_ids[i], first_ids):
                                all_ids_same = False
                                break

                        if all_ids_same:
                            # All samples have identical ids - use shared mode
                            img_ids = first_ids  # Convert to 2D
                            # per_sample_mode = False
                elif img_ids.ndim != 3 and img_ids.ndim != 2:
                    raise ValueError(f"img_ids must be 2D or 3D, got {img_ids.ndim}D")

        # Per-sample mode: generate independent RoPE for each sample
        if per_sample_mode:
            seq_img = img_ids.shape[1] if img_ids is not None else hidden_states.shape[1]

            # 计算每个样本的有效图像 token 数量
            if attention_mask is not None:
                if txt_ids is not None and txt_ids.ndim == 3:
                    seq_txt = txt_ids.shape[1]
                elif txt_ids is not None:
                    seq_txt = txt_ids.shape[0]
                else:
                    seq_txt = 0
                img_mask = attention_mask[:, seq_txt : seq_txt + seq_img]
                valid_img_lengths = img_mask.sum(dim=1)
            else:
                valid_img_lengths = None

            # Generate per-sample RoPE by concatenating txt and img ids first,
            # then computing RoPE (to match shared mode behavior exactly)
            if txt_ids is not None:
                txt_ids_2d = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
                seq_txt_len = txt_ids_2d.shape[0]
            else:
                txt_ids_2d = None
                seq_txt_len = 0

            combined_rope_list = []
            for b in range(batch_size):
                img_ids_b = img_ids[b]  # type: ignore[index]

                if valid_img_lengths is not None:
                    valid_len = int(valid_img_lengths[b].item())
                else:
                    non_zero_rows = (img_ids_b != 0).any(dim=1)
                    valid_len = non_zero_rows.sum().item() if non_zero_rows.any() else seq_img

                img_ids_valid = img_ids_b[:valid_len]

                # Concatenate txt and img ids first, then compute RoPE
                # This matches the shared mode behavior exactly
                if txt_ids_2d is not None:
                    combined_ids = torch.cat([txt_ids_2d, img_ids_valid], dim=0)
                else:
                    combined_ids = img_ids_valid

                # Compute RoPE for combined sequence
                freqs_cos, freqs_sin = self.pos_embed(combined_ids)

                # Pad img portion to max_seq if needed
                if valid_len < seq_img:
                    # Split into txt and img portions
                    txt_cos = freqs_cos[:seq_txt_len] if txt_ids_2d is not None else None
                    txt_sin = freqs_sin[:seq_txt_len] if txt_ids_2d is not None else None
                    img_cos = freqs_cos[seq_txt_len:]
                    img_sin = freqs_sin[seq_txt_len:]

                    # Pad img portion with identity rotation
                    pad_len = seq_img - valid_len
                    D = img_cos.shape[-1]
                    pad_identity_cos = torch.ones(pad_len, D, device=img_cos.device, dtype=img_cos.dtype)
                    pad_identity_sin = torch.zeros(pad_len, D, device=img_sin.device, dtype=img_sin.dtype)

                    img_cos_padded = torch.cat([img_cos, pad_identity_cos], dim=0)
                    img_sin_padded = torch.cat([img_sin, pad_identity_sin], dim=0)

                    # Recombine
                    if txt_cos is not None:
                        freqs_cos = torch.cat([txt_cos, img_cos_padded], dim=0)
                        freqs_sin = torch.cat([txt_sin, img_sin_padded], dim=0)
                    else:
                        freqs_cos = img_cos_padded
                        freqs_sin = img_sin_padded

                combined_rope_list.append((freqs_cos, freqs_sin))

            image_rotary_emb = combined_rope_list

        else:
            # Shared mode: standard processing
            if txt_ids is not None and txt_ids.ndim == 3:
                logger.warning(
                    "Passing `txt_ids` 3d torch.Tensor is deprecated. "
                    "Please remove the batch dimension and pass it as a 2d torch Tensor"
                )
                txt_ids = txt_ids[0]
            if img_ids is not None and img_ids.ndim == 3:
                logger.warning(
                    "Passing `img_ids` 3d torch.Tensor is deprecated. "
                    "Please remove the batch dimension and pass it as a 2d torch Tensor"
                )
                img_ids = img_ids[0]

            ids = torch.cat((txt_ids, img_ids), dim=0)
            image_rotary_emb = self.pos_embed(ids)

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

            # Create additive attention mask (don't zero-out hidden_states)
            # The mask is applied only in attention computation, not to the hidden states directly
            # This prevents "leakage" through residual connections and other operations
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

        # ============ 5. Handle IP-Adapter ============
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # ============ 6. Transformer blocks ============
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            # print('hidden_states shape', hidden_states.shape)
            # return Transformer2DModelOutput(sample=hidden_states)

            # Mask padding after each block to prevent residual/MLP leakage
            # Only apply if there's actual padding (not all True)
            if attention_mask is not None:
                seq_img_block = hidden_states.shape[1]
                if txt_ids is not None:
                    seq_txt_block = txt_ids.shape[1] if txt_ids.ndim == 3 else txt_ids.shape[0]
                else:
                    seq_txt_block = 0
                img_mask_block = attention_mask[:, seq_txt_block : seq_txt_block + seq_img_block]
                if img_mask_block.dtype != torch.bool:
                    img_mask_block = img_mask_block > 0
                # Skip masking if all True (no actual padding) to avoid numerical precision issues
                if not img_mask_block.all():
                    img_mask_block = img_mask_block.to(hidden_states.device)
                    hidden_states = hidden_states * img_mask_block.unsqueeze(-1)

            # controlnet residual
            if controlnet_block_samples is not None:
                import numpy as np

                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        # ============ 7. Single transformer blocks ============
        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # Mask padding after each block to prevent residual/MLP leakage
            # Only apply if there's actual padding (not all True)
            if attention_mask is not None:
                seq_img_block = hidden_states.shape[1]
                if txt_ids is not None:
                    seq_txt_block = txt_ids.shape[1] if txt_ids.ndim == 3 else txt_ids.shape[0]
                else:
                    seq_txt_block = 0
                img_mask_block = attention_mask[:, seq_txt_block : seq_txt_block + seq_img_block]
                if img_mask_block.dtype != torch.bool:
                    img_mask_block = img_mask_block > 0
                # Skip masking if all True (no actual padding) to avoid numerical precision issues
                if not img_mask_block.all():
                    img_mask_block = img_mask_block.to(hidden_states.device)
                    hidden_states = hidden_states * img_mask_block.unsqueeze(-1)

            # controlnet residual
            if controlnet_single_block_samples is not None:
                import numpy as np

                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        # ============ 8. Output projection ============
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # Mask padding regions in the output
        # Only apply if there's actual padding (not all True)
        if attention_mask is not None:
            # Recompute sequence lengths for output masking
            seq_img_out = output.shape[1]
            seq_txt_out = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            img_mask = attention_mask[:, seq_txt_out : seq_txt_out + seq_img_out]
            if img_mask.dtype != torch.bool:
                img_mask = img_mask > 0
            # Skip masking if all True (no actual padding)
            if not img_mask.all():
                output = output.masked_fill(~img_mask.unsqueeze(-1), 0)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
