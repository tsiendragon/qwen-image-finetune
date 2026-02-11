"""
Flux2 Klein LoRA Trainer

重新实现版：
- 训练接口与 BaseTrainer / FluxKontextLoraTrainer 对齐（fit/cache）
- 仅使用单路文本 encoder（Qwen3 + Qwen2TokenizerFast）
- 训练走自定义 flow-matching loss
- 推理 / 验证统一走 diffusers.Flux2KleinPipeline，保证与官方实现一致
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils.torch_utils import randn_tensor

from qflux.models.flux2_klein_loader import (
    load_flux2_klein_scheduler,
    load_flux2_klein_text_encoder,
    load_flux2_klein_tokenizer,
    load_flux2_klein_transformer,
    load_flux2_klein_vae,
)
from qflux.trainer.base_trainer import BaseTrainer


logger = logging.getLogger(__name__)


class Flux2KleinLoraTrainer(BaseTrainer):
    """
    简化版 Flux2 Klein Trainer：
    - 关注训练（fit/cache），不强求完全复刻 FluxKontext 的所有 multi-res 细节
    - predict 使用官方 Flux2KleinPipeline 封装，保证推理行为正确
    """

    def __init__(self, config):
        super().__init__(config)

        # 模型组件
        self.vae: torch.nn.Module | None = None
        self.text_encoder: torch.nn.Module | None = None  # Qwen3ForCausalLM
        self.dit: torch.nn.Module | None = None  # Flux2Transformer2DModel
        self.tokenizer = None  # Qwen2TokenizerFast
        self.scheduler = None

        # VAE 参数
        self.vae_scale_factor: int | None = None
        self.latent_channels: int | None = None  # patch 前的 latent 通道数

        # 文本长度
        self.tokenizer_max_length = 512
        self.max_sequence_length = 512
        # 缓存一个 pipeline 实例，避免每次 predict 都重新构建
        self._pipe: Flux2KleinPipeline | None = None

    # --------------------------------------------------------------------- #
    # BaseTrainer 抽象接口实现
    # --------------------------------------------------------------------- #

    def get_pipeline_class(self):
        return Flux2KleinPipeline

    # --------------------------- 模型加载 --------------------------- #

    def load_model(self, **kwargs):
        """
        从预训练 Flux2 Klein checkpoint 拆分组件。
        """
        logging.info("Loading Flux2Klein components...")

        pretrains = self.config.model.pretrained_embeddings
        model_path = self.config.model.pretrained_model_name_or_path

        # VAE
        if pretrains is not None and "vae" in pretrains:
            vae_path = pretrains["vae"]
        else:
            vae_path = model_path
        self.vae = load_flux2_klein_vae(vae_path, weight_dtype=self.weight_dtype).to("cpu")

        # 文本 encoder (Qwen3)
        if pretrains is not None and "text_encoder" in pretrains:
            txt_path = pretrains["text_encoder"]
        else:
            txt_path = model_path
        self.text_encoder = load_flux2_klein_text_encoder(
            txt_path,
            weight_dtype=self.weight_dtype,
        ).to("cpu")

        # Transformer
        self.dit = load_flux2_klein_transformer(
            model_path,
            weight_dtype=self.weight_dtype,
            device_map="cpu",
        ).to("cpu")

        # Tokenizer & scheduler
        self.tokenizer = load_flux2_klein_tokenizer(model_path)
        self.scheduler = load_flux2_klein_scheduler(model_path)
        import copy

        self.sampling_scheduler = copy.deepcopy(self.scheduler)

        # VAE 参数
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.latent_channels = self.vae.config.latent_channels if getattr(self, "vae", None) else 16

        # 冻结 VAE & 文本 encoder，仅训练 LoRA
        self.vae.requires_grad_(False).eval()
        self.text_encoder.requires_grad_(False).eval()

        # latent 通道数：Flux2 使用 2x2 patch，再 pack，in_channels = latent_channels * 4
        if self.dit is not None:
            self.num_channels_latents = self.dit.config.in_channels // 4
        else:
            self.num_channels_latents = 16

        logging.info(
            f"Flux2Klein components loaded: vae_scale_factor={self.vae_scale_factor}, "
            f"latent_channels={self.latent_channels}, num_channels_latents={self.num_channels_latents}"
        )

    # --------------------------- 文本编码 --------------------------- #

    @staticmethod
    def _get_qwen3_prompt_embeds(
        text_encoder,
        tokenizer,
        prompt,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        max_sequence_length: int = 512,
        hidden_states_layers: tuple[int, ...] = (9, 18, 27),
    ) -> torch.Tensor:
        """
        与 diffusers.Flux2KleinPipeline._get_qwen3_prompt_embeds 一致。
        返回 [B, L, D]。
        """
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

        assert isinstance(tokenizer, Qwen2TokenizerFast)
        assert isinstance(text_encoder, Qwen3ForCausalLM)

        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        return prompt_embeds

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor, t_coord: torch.Tensor | None = None) -> torch.Tensor:
        """
        文本 position ids，与 Flux2Pipeline._prepare_text_ids 一致。
        输入: x [B, L, D]，输出: [B, L, 4]。
        """
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        device_text_encoder: torch.device | None = None,
        device_text_encoder_2: torch.device | None = None,
        max_sequence_length: int = 512,
    ):
        """
        与 BaseTrainer 要求的接口对齐：
        返回 pooled_prompt_embeds, prompt_embeds, text_ids。
        Klein 只有一条文本路，prompt_2 被忽略。
        """
        _ = prompt_2, device_text_encoder_2  # 未使用
        prompt = [prompt] if isinstance(prompt, str) else prompt

        assert self.text_encoder is not None and self.tokenizer is not None
        device = device_text_encoder or next(self.text_encoder.parameters()).device

        with torch.inference_mode():
            prompt_embeds = self._get_qwen3_prompt_embeds(
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                prompt=prompt,
                dtype=self.weight_dtype,
                device=device,
                max_sequence_length=max_sequence_length,
            )
            pooled_prompt_embeds = prompt_embeds.mean(dim=1)
            text_ids = self._prepare_text_ids(prompt_embeds).to(device=device, dtype=self.weight_dtype)

        return pooled_prompt_embeds, prompt_embeds, text_ids

    # --------------------------- VAE / latents --------------------------- #

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) -> (B, C*4, H/2, W/2)，来自 Flux2Pipeline._patchify_latents。
        """
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        (B, C, H, W) -> (B, H*W, C)，来自 Flux2Pipeline._pack_latents。
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _prepare_latent_ids(latents_4d: torch.Tensor) -> torch.Tensor:
        """
        为 latent 生成位置 ids，与 Flux2Pipeline._prepare_latent_ids 一致。
        输入: latents_4d [B, C, H, W]，输出: [B, H*W, 4]。
        """
        batch_size, _, height, width = latents_4d.shape
        device = latents_4d.device

        t = torch.arange(1, device=device)
        h = torch.arange(height, device=device)
        w = torch.arange(width, device=device)
        l = torch.arange(1, device=device)

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)
        return latent_ids

    def encode_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Klein 风格 VAE 编码：
        - vae.encode -> latent_dist.mode()
        - patchify
        - 使用 VAE 的 BN running_mean / running_var 做标准化
        - pack 成 [B, T, C]
        """
        assert self.vae is not None
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        image = image.to(self.vae.device, dtype=self.weight_dtype)

        with torch.inference_mode():
            latents = self.vae.encode(image).latent_dist.mode()
            latents = self._patchify_latents(latents)

            bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
            latents = (latents - bn_mean) / bn_std

        latents = self._pack_latents(latents)
        return latents

    def prepare_latents(
        self,
        image: torch.Tensor | None,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ):
        """
        为训练/缓存准备 image_latents（和必要的占位 latent ids）。
        这里只返回 image_latents 和 latent_ids，采样时我们走官方 pipeline。
        """
        assert self.vae is not None and self.vae_scale_factor is not None

        # 调整到 VAE / packing 需要的尺寸
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        if image is not None:
            image_latents = self.encode_vae_image(image)
        else:
            image_latents = None

        # 构造 latent_ids（仅用于训练 forward）
        H_lat = height // 2
        W_lat = width // 2
        dummy = torch.zeros(batch_size, num_channels_latents * 4, H_lat, W_lat, device=self.vae.device, dtype=dtype)
        latent_ids = self._prepare_latent_ids(dummy)
        return image_latents, latent_ids

    # --------------------------- 设备分配 --------------------------- #

    def setup_model_device_train_mode(self, stage: str = "fit", cache: bool = False):
        """
        简化的设备分配逻辑：
        - fit: VAE/TextEncoder/Transformer 到训练设备，仅 LoRA 参与训练
        - cache: 只需要 encoder，不需要 transformer
        - predict: 所有组件到 predict.devices
        """
        if stage == "fit":
            assert hasattr(self, "accelerator")

        if stage == "fit":
            self.vae.to(self.accelerator.device).requires_grad_(False).eval()
            self.text_encoder.to(self.accelerator.device).requires_grad_(False).eval()
            self.dit.to(self.accelerator.device)

            self.dit.requires_grad_(False)
            self.dit.train()
            for name, p in self.dit.named_parameters():
                p.requires_grad = "lora" in name

        elif stage == "cache":
            self.vae = self.vae.to(self.config.cache.devices.vae, non_blocking=True)
            self.text_encoder = self.text_encoder.to(
                self.config.cache.devices.text_encoder,
                non_blocking=True,
            )
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.dit.cpu()
            torch.cuda.empty_cache()

        elif stage == "predict":
            devices = self.config.predict.devices
            self.vae.to(devices.vae).requires_grad_(False).eval()
            self.text_encoder.to(devices.text_encoder).requires_grad_(False).eval()
            self.dit.to(devices.dit).requires_grad_(False).eval()

    # --------------------------- Embeddings / Cache --------------------------- #

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict[str, torch.Tensor]:
        """
        - 归一化 image / control
        - 编码 prompt -> pooled_prompt_embeds / prompt_embeds / text_ids
        - 编码 image / control -> image_latents / control_latents / control_ids
        """
        if "image" in batch:
            batch["image"] = self.normalize_image(batch["image"])

        if "control" in batch:
            batch["control"] = self.normalize_image(batch["control"])

        n_controls = batch.get("n_controls", 0)
        if not isinstance(n_controls, int):
            n_controls = int(n_controls[0])

        for i in range(n_controls):
            key = f"control_{i + 1}"
            if key in batch:
                batch[key] = self.normalize_image(batch[key])

        # 文本编码
        pooled_prompt_embeds, prompt_embeds, text_ids = self.encode_prompt(
            prompt=batch["prompt"],
            max_sequence_length=self.max_sequence_length,
        )
        batch["pooled_prompt_embeds"] = pooled_prompt_embeds
        batch["prompt_embeds"] = prompt_embeds
        batch["text_ids"] = text_ids

        # image -> image_latents
        if "image" in batch:
            image = batch["image"]
            b, _, h, w = image.shape
            image_latents, _ = self.prepare_latents(
                image=image,
                batch_size=b,
                num_channels_latents=self.num_channels_latents,
                height=h,
                width=w,
                dtype=self.weight_dtype,
            )
            batch["image_latents"] = image_latents

        # control(s)
        control_latents_list: list[torch.Tensor] = []
        control_ids_list: list[torch.Tensor] = []

        if "control" in batch:
            control = batch["control"]
            b, _, h, w = control.shape
            control_latents, control_ids = self.prepare_latents(
                image=control,
                batch_size=b,
                num_channels_latents=self.num_channels_latents,
                height=h,
                width=w,
                dtype=self.weight_dtype,
            )
            # control_0 标记为 1
            control_ids[..., 0] = 1
            control_latents_list.append(control_latents)
            control_ids_list.append(control_ids)

        for i in range(1, n_controls + 1):
            key = f"control_{i}"
            if key not in batch:
                continue
            control = batch[key]
            b, _, h, w = control.shape
            ctl_latents, ctl_ids = self.prepare_latents(
                image=control,
                batch_size=b,
                num_channels_latents=self.num_channels_latents,
                height=h,
                width=w,
                dtype=self.weight_dtype,
            )
            ctl_ids[..., 0] = i + 1
            control_latents_list.append(ctl_latents)
            control_ids_list.append(ctl_ids)

        if control_latents_list:
            batch["control_latents"] = torch.cat(control_latents_list, dim=1)
            batch["control_ids"] = torch.cat(control_ids_list, dim=1)

        return batch

    def cache_step(self, data: dict):
        """
        缓存 image_latents / prompt_embeds 等，用于加速训练。
        结构简化，仅存一份。
        """
        image_latents = data["image_latents"].detach().cpu()
        pooled_prompt_embeds = data["pooled_prompt_embeds"].detach().cpu()
        prompt_embeds = data["prompt_embeds"].detach().cpu()
        text_ids = data["text_ids"].detach().cpu()

        cache_embeddings = {
            "image_latents": image_latents[0],
            "pooled_prompt_embeds": pooled_prompt_embeds[0],
            "prompt_embeds": prompt_embeds[0],
            "text_ids": text_ids[0],
        }
        map_keys = {
            "image_latents": "image_hash",
            "pooled_prompt_embeds": "prompt_hash",
            "prompt_embeds": "prompt_hash",
            "text_ids": "prompt_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    def prepare_cached_embeddings(self, batch: dict) -> dict[str, torch.Tensor]:
        batch["text_ids"] = batch["text_ids"][0]
        return batch

    # --------------------------- Loss 计算 --------------------------- #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """
        单一分辨率下的 flow-matching loss。
        多分辨率 / 多 control 的复杂处理暂未实现。
        """
        image_latents = embeddings["image_latents"]  # [B, T_img, C]
        text_ids = embeddings["text_ids"]  # [B, L, 4]
        control_latents = embeddings.get("control_latents", None)  # [B, T_ctl, C]
        control_ids = embeddings.get("control_ids", None)  # [B, T_ctl, 4]
        prompt_embeds = embeddings["prompt_embeds"]

        assert self.accelerator is not None
        device = self.accelerator.device

        image_latents = image_latents.to(device)
        text_ids = text_ids.to(device)
        prompt_embeds = prompt_embeds.to(device)
        if control_latents is not None:
            control_latents = control_latents.to(device)
        if control_ids is not None:
            control_ids = control_ids.to(device)

        # 计算 H_lat / W_lat
        b, T_img, C = image_latents.shape
        # 这里假设图像已经按 vae_scale_factor / packing_factor 处理完，可用 sqrt 恢复
        H_lat = W_lat = int((T_img) ** 0.5)
        assert H_lat * W_lat == T_img, f"latent seq len {T_img} 不是平方数，当前实现仅支持方形 latent。"

        with torch.no_grad():
            if "noise" in embeddings:
                noise = embeddings["noise"].to(device)
            else:
                noise = torch.randn_like(image_latents, device=device, dtype=self.weight_dtype)
            if "timestep" in embeddings:
                t = embeddings["timestep"].to(device)
            else:
                t = torch.rand((b,), device=device, dtype=self.weight_dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            noisy_model_input = (1.0 - t_) * image_latents + t_ * noise

            # 构造 latent_ids
            dummy = torch.zeros(b, C, H_lat, W_lat, device=device, dtype=self.weight_dtype)
            latent_ids = self._prepare_latent_ids(dummy)

            latent_model_input = noisy_model_input
            if control_latents is not None and control_ids is not None:
                latent_model_input = torch.cat([noisy_model_input, control_latents], dim=1)
                latent_ids = torch.cat([latent_ids, control_ids], dim=1)

        guidance = (
            torch.ones((b,), device=device, dtype=self.weight_dtype)
            if getattr(self.dit.config, "guidance_embeds", False)
            else None
        )

        latent_model_input = latent_model_input.to(self.weight_dtype)
        prompt_embeds = prompt_embeds.to(self.weight_dtype)
        image_latents = image_latents.to(self.weight_dtype)
        t = t.to(self.weight_dtype)
        text_ids = text_ids.to(device)

        model_pred = self.dit(
            hidden_states=latent_model_input,
            timestep=t,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        model_pred = model_pred[:, : image_latents.size(1)]
        target = noise - image_latents

        edit_mask = embeddings.get("edit_mask", None)
        if edit_mask is not None:
            edit_mask = edit_mask.to(self.weight_dtype).to(device)

        loss = self.forward_loss(model_pred, target, weighting=None, edit_mask=edit_mask)
        return loss

    # --------------------------- predict：官方 pipeline 封装 --------------------------- #

    def prepare_predict_batch_data(self, *args, **kwargs) -> dict:
        """
        为兼容 BaseTrainer 接口，predict 不使用此方法，直接走 Flux2KleinPipeline。
        """
        raise NotImplementedError(
            "Flux2KleinLoraTrainer.predict 已直接封装 Flux2KleinPipeline，不需要 prepare_predict_batch_data"
        )

    def decode_vae_latent(self, latents: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """
        不再使用 BaseTrainer 的 latent->image 解码路径，
        统一由 Flux2KleinPipeline 负责。
        """
        raise NotImplementedError("decode_vae_latent 未在 Flux2KleinLoraTrainer 中使用。请使用 predict() 完整推理。")

    def sampling_from_embeddings(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        不实现自定义采样，推理时使用 Flux2KleinPipeline。
        """
        raise NotImplementedError("sampling_from_embeddings 未在 Flux2KleinLoraTrainer 中使用。请使用 predict() 完整推理。")

    def predict(
        self,
        image: Any,
        prompt: str | list[str] | None = None,
        num_inference_steps: int = 20,
        **kwargs,
    ):
        """
        推理：直接使用 diffusers.Flux2KleinPipeline
        - 使用当前 trainer 中的组件（含 LoRA）
        - 保证推理行为与官方一致
        """
        # 确保权重 / LoRA / 设备设定完毕
        self.setup_predict()

        device = next(self.dit.parameters()).device

        # 只在第一次 predict 时构建一次官方 pipeline，后续复用
        if self._pipe is None:
            self._pipe = Flux2KleinPipeline(
                scheduler=self.sampling_scheduler or self.scheduler,
                vae=self.vae,
                text_encoder=self.text_encoder,
                tokenizer=self.tokenizer,
                transformer=self.dit,
                is_distilled=getattr(self.scheduler.config, "is_distilled", False),
            ).to(device)
        pipe = self._pipe

        guidance_scale = kwargs.pop("guidance_scale", kwargs.pop("guidance", 4.0))
        height = kwargs.pop("height", None)
        width = kwargs.pop("width", None)
        output_type = kwargs.pop("output_type", "pil")

        result = pipe(
            image=image,
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            output_type=output_type,
            **kwargs,
        )
        images = result.images if hasattr(result, "images") else result[0]
        return images

