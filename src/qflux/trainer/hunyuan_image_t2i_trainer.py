"""Trainer for tencent/HunyuanImage-3.0-Instruct-DistilA (text-to-image, DiT flow-matching).

Architecture overview:
  - Transformer: HunyuanImageTransformer2DModel (MMDiT, 20 dual-stream + 40 single-stream blocks)
  - VAE: AutoencoderKLHunyuanImage (spatial_compression_ratio=32)
  - Text encoder 1: Qwen2.5-VL-7B-Instruct → hidden_states[-3] (hidden_state_skip_layer=2)
  - Text encoder 2: ByT5 (T5EncoderModel) for glyph/OCR text
  - Scheduler: FlowMatchEulerDiscreteScheduler
  - Guidance: distilled_guidance_scale * 1000 (only when transformer.config.guidance_embeds=True)
  - Latents: 4D (B, 64, H//32, W//32), no packing, vae_scale_factor = spatial_compression_ratio
  - Training target: noise - image_latents (standard flow matching velocity)
  - Timestep: sigma in [0, 1] range (not /1000 like Qwen)
"""

import copy
import gc
import logging
import re
from typing import Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from qflux.trainer.base_trainer import BaseTrainer
from qflux.utils.images import make_image_shape_devisible


logger = logging.getLogger(__name__)

# Prompt template for Qwen2.5-VL text encoder (from HunyuanImagePipeline)
PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>"
)
PROMPT_TEMPLATE_DROP_IDX = 34  # tokens to skip (system prompt prefix)
TOKENIZER_MAX_LENGTH = 1000
TOKENIZER_2_MAX_LENGTH = 128
HIDDEN_STATE_SKIP_LAYER = 2  # use hidden_states[-(2+1)] = hidden_states[-3]


def _extract_glyph_text(prompt: str) -> str | None:
    """Extract quoted text from prompt for ByT5 glyph encoding."""
    texts = []
    for pattern in (r"\'(.*?)\'", r'"(.*?)"', r"\u2018(.*?)\u2019", r"\u201c(.*?)\u201d"):
        texts.extend(re.findall(pattern, prompt))
    if texts:
        return ". ".join([f'Text "{t}"' for t in texts]) + ". "
    return None


class HunyuanImageT2ITrainer(BaseTrainer):
    """Trainer for HunyuanImage-3.0-Instruct-DistilA text-to-image model.

    Uses dual text encoders (Qwen2.5-VL + ByT5) and a custom VAE with 32x spatial
    compression. Supports distilled guidance via an embedded guidance tensor.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        from diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage import HunyuanImagePipeline

        return HunyuanImagePipeline

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs):
        """Load all HunyuanImage components."""
        from diffusers.models import AutoencoderKLHunyuanImage, HunyuanImageTransformer2DModel
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import AutoTokenizer, ByT5Tokenizer, T5EncoderModel

        logger.info("Loading HunyuanImagePipeline components...")
        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings or {}

        # ----- VAE -----
        vae_path = pretrains.get("vae", model_path)
        self.vae = AutoencoderKLHunyuanImage.from_pretrained(
            vae_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        )

        # ----- Text encoder 1: Qwen2.5-VL -----
        te_path = pretrains.get("text_encoder", model_path)
        from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

        self.tokenizer: Qwen2Tokenizer = AutoTokenizer.from_pretrained(
            te_path, subfolder="tokenizer"
        )
        self.text_encoder: Qwen2_5_VLForConditionalGeneration = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                te_path,
                subfolder="text_encoder",
                torch_dtype=self.weight_dtype,
            )
        )

        # ----- Text encoder 2: ByT5 -----
        te2_path = pretrains.get("text_encoder_2", model_path)
        self.tokenizer_2: ByT5Tokenizer = ByT5Tokenizer.from_pretrained(
            te2_path, subfolder="tokenizer_2"
        )
        self.text_encoder_2: T5EncoderModel = T5EncoderModel.from_pretrained(
            te2_path,
            subfolder="text_encoder_2",
            torch_dtype=self.weight_dtype,
        )

        # ----- Transformer -----
        self.dit: HunyuanImageTransformer2DModel = (
            HunyuanImageTransformer2DModel.from_pretrained(
                model_path,
                subfolder="transformer",
                torch_dtype=self.weight_dtype,
            )
        )

        # ----- Scheduler -----
        self.scheduler: FlowMatchEulerDiscreteScheduler = (
            FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_path, subfolder="scheduler"
            )
        )
        self.sampling_scheduler: FlowMatchEulerDiscreteScheduler = copy.deepcopy(self.scheduler)

        # ----- Parameters -----
        self.vae_scale_factor: int = (
            self.vae.config.spatial_compression_ratio
            if hasattr(self.vae.config, "spatial_compression_ratio")
            else 32
        )
        self._vae_scaling_factor: float | None = getattr(self.vae.config, "scaling_factor", None)
        self.num_channels_latents: int = self.dit.config.in_channels
        self.tokenizer_max_length = TOKENIZER_MAX_LENGTH
        self.tokenizer_2_max_length = TOKENIZER_2_MAX_LENGTH

        # Freeze non-trainable components
        self.text_encoder.requires_grad_(False).eval()
        self.text_encoder_2.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logger.info(
            f"HunyuanImage components loaded. VAE scale: {self.vae_scale_factor}, "
            f"latent channels: {self.num_channels_latents}, "
            f"guidance_embeds: {self.dit.config.guidance_embeds}, "
            f"use_meanflow: {getattr(self.dit.config, 'use_meanflow', False)}"
        )

    # ------------------------------------------------------------------ #
    # Prompt encoding                                                      #
    # ------------------------------------------------------------------ #

    def _encode_qwen(
        self,
        prompt: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with Qwen2.5-VL text encoder.

        Returns:
            prompt_embeds: (B, TOKENIZER_MAX_LENGTH, hidden_dim)
            attention_mask: (B, TOKENIZER_MAX_LENGTH)
        """
        drop_idx = PROMPT_TEMPLATE_DROP_IDX
        txt = [PROMPT_TEMPLATE.format(p) for p in prompt]
        txt_tokens = self.tokenizer(
            txt,
            max_length=self.tokenizer_max_length + drop_idx,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            outputs = self.text_encoder(
                input_ids=txt_tokens.input_ids,
                attention_mask=txt_tokens.attention_mask,
                output_hidden_states=True,
            )
        # Skip the last (HIDDEN_STATE_SKIP_LAYER + 1) hidden states → use [-3]
        prompt_embeds = outputs.hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]
        prompt_embeds = prompt_embeds[:, drop_idx:].to(dtype=self.weight_dtype, device=device)
        attn_mask = txt_tokens.attention_mask[:, drop_idx:].to(device)
        return prompt_embeds, attn_mask

    def _encode_byt5_single(
        self,
        glyph_text: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a single glyph string with ByT5.

        Returns:
            embeds: (1, TOKENIZER_2_MAX_LENGTH, byt5_dim)
            mask: (1, TOKENIZER_2_MAX_LENGTH)
        """
        txt_tokens = self.tokenizer_2(
            glyph_text,
            padding="max_length",
            max_length=self.tokenizer_2_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            embeds = self.text_encoder_2(
                input_ids=txt_tokens.input_ids,
                attention_mask=txt_tokens.attention_mask.float(),
            )[0]

        return embeds.to(dtype=self.weight_dtype, device=device), txt_tokens.attention_mask.to(device)

    def _encode_byt5_batch(
        self,
        prompt: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract and encode glyph text for a batch of prompts.

        Returns:
            prompt_embeds_2: (B, TOKENIZER_2_MAX_LENGTH, byt5_dim)
            attention_mask_2: (B, TOKENIZER_2_MAX_LENGTH)
        """
        embeds_list, mask_list = [], []
        byt5_dim = self.text_encoder_2.config.d_model

        for p in prompt:
            glyph = _extract_glyph_text(p)
            if glyph is None:
                # Zero-fill for prompts without quoted text
                embeds_list.append(
                    torch.zeros(1, self.tokenizer_2_max_length, byt5_dim, dtype=self.weight_dtype, device=device)
                )
                mask_list.append(
                    torch.zeros(1, self.tokenizer_2_max_length, dtype=torch.int64, device=device)
                )
            else:
                e, m = self._encode_byt5_single(glyph, device)
                embeds_list.append(e)
                mask_list.append(m)

        return torch.cat(embeds_list, dim=0), torch.cat(mask_list, dim=0)

    def encode_prompt(
        self,
        prompt: str | list[str],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompts with both text encoders.

        Returns:
            prompt_embeds: (B, seq_len, qwen_dim)
            prompt_embeds_mask: (B, seq_len)
            prompt_embeds_2: (B, byt5_seq_len, byt5_dim)
            prompt_embeds_mask_2: (B, byt5_seq_len)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = next(self.text_encoder.parameters()).device

        prompt_embeds, prompt_embeds_mask = self._encode_qwen(prompt, device)
        prompt_embeds_2, prompt_embeds_mask_2 = self._encode_byt5_batch(prompt, device)

        return prompt_embeds, prompt_embeds_mask, prompt_embeds_2, prompt_embeds_mask_2

    # ------------------------------------------------------------------ #
    # VAE                                                                  #
    # ------------------------------------------------------------------ #

    def _vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        """VAE-encode a preprocessed image to model-space latents.

        Applies scaling_factor if configured (model operates on scaled latents).
        """
        with torch.inference_mode():
            raw = self.vae.encode(image).latent_dist.sample()
        if self._vae_scaling_factor is not None:
            return raw * self._vae_scaling_factor
        return raw

    def prepare_latents(
        self,
        image: torch.Tensor | None,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare noise latents and optionally VAE-encoded image latents."""
        h_lat = int(height) // self.vae_scale_factor
        w_lat = int(width) // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, h_lat, w_lat)
        device = next(self.vae.parameters()).device

        noise = randn_tensor(shape, device=device, dtype=dtype)

        image_latents = None
        if image is not None:
            image_latents = self._vae_encode(image.to(device=device, dtype=dtype))

        return noise, image_latents

    def decode_vae_latent(
        self, latents: torch.Tensor, target_height: int, target_width: int
    ) -> torch.Tensor:
        """Decode latents to RGB image in [0, 1]."""
        latents = latents.to(self.vae.device, dtype=self.vae.dtype)
        if self._vae_scaling_factor is not None:
            latents = latents / self._vae_scaling_factor
        with torch.inference_mode():
            image = self.vae.decode(latents, return_dict=False)[0]
        return (image / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------ #
    # Embedding preparation                                                #
    # ------------------------------------------------------------------ #

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
        """Prepare embeddings for training (fit/cache) or inference (predict)."""
        device = next(self.text_encoder.parameters()).device

        # ----- Prompt encoding -----
        pe, pm, pe2, pm2 = self.encode_prompt(batch["prompt"])
        batch["prompt_embeds"] = pe
        batch["prompt_embeds_mask"] = pm
        batch["prompt_embeds_2"] = pe2
        batch["prompt_embeds_mask_2"] = pm2

        if stage == "cache":
            empty_pe, empty_pm, empty_pe2, empty_pm2 = self.encode_prompt([""])
            batch["empty_prompt_embeds"] = empty_pe
            batch["empty_prompt_embeds_mask"] = empty_pm
            batch["empty_prompt_embeds_2"] = empty_pe2
            batch["empty_prompt_embeds_mask_2"] = empty_pm2

        # ----- VAE encode target image -----
        if "image" in batch:
            image = batch["image"]
            if isinstance(image, torch.Tensor) and image.ndim == 5:
                image = image.squeeze(2)  # (B, 3, 1, H, W) → (B, 3, H, W)
            # Normalize [0,1] → [-1,1] if needed
            if image.max() <= 1.0 + 1e-6:
                image = image * 2.0 - 1.0
            image = image.to(device=device, dtype=self.weight_dtype)

            batch["height"] = image.shape[2]
            batch["width"] = image.shape[3]

            with torch.inference_mode():
                image_latents = self._vae_encode(image)
            batch["image_latents"] = image_latents

        return batch

    def prepare_cached_embeddings(self, batch: dict) -> dict:
        """Load cached embeddings (no transformation needed)."""
        return batch

    # ------------------------------------------------------------------ #
    # Training loss                                                        #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """Flow-matching training loss for HunyuanImage."""
        assert self.accelerator is not None
        device = self.accelerator.device

        image_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        prompt_embeds_mask = embeddings["prompt_embeds_mask"].to(device)
        prompt_embeds_2 = embeddings["prompt_embeds_2"].to(self.weight_dtype).to(device)
        prompt_embeds_mask_2 = embeddings["prompt_embeds_mask_2"].to(device)

        batch_size = image_latents.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(image_latents)

            # Sample sigma uniformly in (0, 1) — this IS the timestep for HunyuanImage
            sigma = torch.rand(batch_size, device=device, dtype=self.weight_dtype)
            sigma_4d = sigma[:, None, None, None]

            # Flow-matching: x_t = (1 - sigma)*x_0 + sigma*noise
            noisy_latents = (1.0 - sigma_4d) * image_latents + sigma_4d * noise

        # Distilled guidance embedding (sigma * 1000 * distilled_guidance_scale)
        # For LoRA training we use the model's default distilled_guidance_scale
        if self.dit.config.guidance_embeds:
            guidance_scale = getattr(self.config.model, "distilled_guidance_scale", 3.25)
            guidance = torch.full(
                (batch_size,),
                guidance_scale * 1000.0,
                device=device,
                dtype=self.weight_dtype,
            )
        else:
            guidance = None

        # meanflow: timestep_r is the "next" sigma (use a slightly smaller value)
        use_meanflow = getattr(self.dit.config, "use_meanflow", False)
        if use_meanflow:
            # During training, approximate next sigma (not exact but reasonable)
            timestep_r = (sigma - 1.0 / self.scheduler.config.num_train_timesteps).clamp(min=0.0)
        else:
            timestep_r = None

        model_pred = self.dit(
            hidden_states=noisy_latents,
            timestep=sigma,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            encoder_hidden_states_2=prompt_embeds_2,
            encoder_attention_mask_2=prompt_embeds_mask_2,
            guidance=guidance,
            timestep_r=timestep_r,
            return_dict=False,
        )[0]

        # Standard flow-matching target: velocity = noise - image
        target = noise - image_latents

        loss = self.forward_loss(model_pred, target)
        return loss

    # ------------------------------------------------------------------ #
    # Cache step                                                           #
    # ------------------------------------------------------------------ #

    def cache_step(self, data: dict):
        """Save single-sample embeddings to disk cache."""
        cache_embeddings = {
            "image_latents": data["image_latents"].detach().cpu()[0],
            "prompt_embeds": data["prompt_embeds"].detach().cpu()[0],
            "prompt_embeds_mask": data["prompt_embeds_mask"].detach().cpu()[0],
            "prompt_embeds_2": data["prompt_embeds_2"].detach().cpu()[0],
            "prompt_embeds_mask_2": data["prompt_embeds_mask_2"].detach().cpu()[0],
            "empty_prompt_embeds": data["empty_prompt_embeds"].detach().cpu()[0],
            "empty_prompt_embeds_mask": data["empty_prompt_embeds_mask"].detach().cpu()[0],
            "empty_prompt_embeds_2": data["empty_prompt_embeds_2"].detach().cpu()[0],
            "empty_prompt_embeds_mask_2": data["empty_prompt_embeds_mask_2"].detach().cpu()[0],
        }
        map_keys = {
            "image_latents": "image_hash",
            "prompt_embeds": "prompt_hash",
            "prompt_embeds_mask": "prompt_hash",
            "prompt_embeds_2": "prompt_hash",
            "prompt_embeds_mask_2": "prompt_hash",
            "empty_prompt_embeds": "prompt_hash",
            "empty_prompt_embeds_mask": "prompt_hash",
            "empty_prompt_embeds_2": "prompt_hash",
            "empty_prompt_embeds_mask_2": "prompt_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    # ------------------------------------------------------------------ #
    # Device management                                                    #
    # ------------------------------------------------------------------ #

    def setup_model_device_train_mode(self, stage: str = "fit", cache: bool = False):
        """Allocate models to devices for each stage."""
        if stage == "fit":
            assert hasattr(self, "accelerator"), "accelerator must be set before calling this"

            if self.cache_exist and self.use_cache:
                # Cache exists: only need transformer
                for attr in ("text_encoder", "text_encoder_2", "vae"):
                    if hasattr(self, attr):
                        getattr(self, attr).cpu()
                if not self.config.validation.enabled:
                    for attr in ("text_encoder", "text_encoder_2", "vae"):
                        if hasattr(self, attr):
                            delattr(self, attr)
                gc.collect()
                torch.cuda.empty_cache()

                self.dit.to(self.accelerator.device)
                self.dit.train()
                for name, param in self.dit.named_parameters():
                    param.requires_grad = "lora" in name

            else:
                # No cache: move all models to device
                for attr in ("vae", "text_encoder", "text_encoder_2", "dit"):
                    if hasattr(self, attr):
                        getattr(self, attr).to(self.accelerator.device)
                for attr in ("vae", "text_encoder", "text_encoder_2"):
                    if hasattr(self, attr):
                        getattr(self, attr).requires_grad_(False).eval()
                self.dit.train()
                for name, param in self.dit.named_parameters():
                    param.requires_grad = "lora" in name

        elif stage == "cache":
            # Cache stage: need VAE + text encoders, free transformer
            self.vae.to(self.config.cache.devices.vae)
            self.text_encoder.to(self.config.cache.devices.text_encoder)
            te2_device = (
                self.config.cache.devices.text_encoder_2
                or self.config.cache.devices.text_encoder
            )
            self.text_encoder_2.to(te2_device)

            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.text_encoder_2.requires_grad_(False).eval()

            if hasattr(self, "dit"):
                self.dit.cpu()
                del self.dit
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("cache mode: VAE + text_encoder + text_encoder_2 on device; transformer freed")

        elif stage == "predict":
            devices = self.config.predict.devices
            self.vae.to(devices.vae)
            self.text_encoder.to(devices.text_encoder)
            te2_device = devices.text_encoder_2 or devices.text_encoder
            self.text_encoder_2.to(te2_device)
            self.dit.to(devices.dit)

            for attr in ("vae", "text_encoder", "text_encoder_2", "dit"):
                if hasattr(self, attr):
                    getattr(self, attr).requires_grad_(False).eval()
            logger.info(
                f"predict mode: vae={devices.vae}, te={devices.text_encoder}, "
                f"te2={te2_device}, dit={devices.dit}"
            )

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def prepare_predict_batch_data(
        self,
        prompt: str | list[str],
        height: int = 1024,
        width: int = 1024,
        negative_prompt: str | list[str] | None = None,
        distilled_guidance_scale: float = 3.25,
        num_inference_steps: int = 8,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> dict:
        """Prepare inference batch for HunyuanImage T2I."""
        if isinstance(prompt, str):
            prompt = [prompt]
        self.weight_dtype = weight_dtype

        height, width = make_image_shape_devisible(height, width, self.vae_scale_factor)

        data: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "distilled_guidance_scale": distilled_guidance_scale,
        }
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            data["negative_prompt"] = negative_prompt

        return data

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """Denoising loop for HunyuanImage (no guider system, direct CFG if needed).

        For the distilled model (guidance_embeds=True), passes guidance tensor directly.
        Returns final latents (B, C, H_lat, W_lat).
        """
        num_inference_steps = embeddings["num_inference_steps"]
        distilled_guidance_scale = embeddings.get("distilled_guidance_scale", 3.25)
        prompt_embeds = embeddings["prompt_embeds"]
        prompt_embeds_mask = embeddings["prompt_embeds_mask"]
        prompt_embeds_2 = embeddings["prompt_embeds_2"]
        prompt_embeds_mask_2 = embeddings["prompt_embeds_mask_2"]
        height = embeddings["height"]
        width = embeddings["width"]
        batch_size = prompt_embeds.shape[0]

        device = self.dit.device
        dtype = self.weight_dtype

        # Move embeddings to device
        prompt_embeds = prompt_embeds.to(device, dtype=dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(device)
        prompt_embeds_2 = prompt_embeds_2.to(device, dtype=dtype)
        prompt_embeds_mask_2 = prompt_embeds_mask_2.to(device)

        # Prepare initial latents
        h_lat = int(height) // self.vae_scale_factor
        w_lat = int(width) // self.vae_scale_factor

        if "latents" in embeddings:
            latents = embeddings["latents"].to(device, dtype=torch.float32)
        else:
            latents = randn_tensor(
                (batch_size, self.num_channels_latents, h_lat, w_lat),
                device=device,
                dtype=torch.float32,
            )

        # Prepare guidance embedding
        if self.dit.config.guidance_embeds:
            guidance = torch.tensor(
                [distilled_guidance_scale * 1000.0] * batch_size,
                dtype=dtype,
                device=device,
            )
        else:
            guidance = None

        # Prepare timesteps (sigmas from 1.0 → 0.0)
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        timesteps, num_inference_steps = _retrieve_timesteps(
            self.sampling_scheduler, num_inference_steps, device, sigmas=sigmas
        )
        self.sampling_scheduler.set_begin_index(0)

        use_meanflow = getattr(self.dit.config, "use_meanflow", False)

        with torch.inference_mode():
            for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="HunyuanImage generating"):
                timestep = t.expand(batch_size).to(dtype)

                if use_meanflow:
                    timestep_r = (
                        torch.tensor([0.0], device=device)
                        if i == len(timesteps) - 1
                        else timesteps[i + 1]
                    )
                    timestep_r = timestep_r.expand(batch_size).to(dtype)
                else:
                    timestep_r = None

                noise_pred = self.dit(
                    hidden_states=latents.to(dtype),
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_embeds_mask,
                    encoder_hidden_states_2=prompt_embeds_2,
                    encoder_attention_mask_2=prompt_embeds_mask_2,
                    guidance=guidance,
                    timestep_r=timestep_r,
                    return_dict=False,
                )[0]

                latents = self.sampling_scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]

        return latents


def _retrieve_timesteps(scheduler, num_inference_steps, device, sigmas):
    """Simplified retrieve_timesteps matching HunyuanImagePipeline convention."""
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps, num_inference_steps
