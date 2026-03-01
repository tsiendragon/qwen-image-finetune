"""Trainer for hunyuanvideo-community/HunyuanImage-2.1-Refiner-Diffusers (image-to-image).

Architecture overview:
  - Transformer: HunyuanImageTransformer2DModel (same architecture as T2I but with in_channels doubled)
  - VAE: AutoencoderKLHunyuanImageRefiner (3D causal, spatial_compression_ratio=16)
  - Text encoder: Qwen2.5-VL-7B-Instruct → hidden_states[-3] (Llama-style template, drop_idx=36)
  - Scheduler: FlowMatchEulerDiscreteScheduler
  - Guidance: distilled_guidance_scale * 1000

Latent format:
  - 5D tensors (B, latent_channels, 1, H//16, W//16) where latent_channels = in_channels // 2
  - _reorder_image_tokens: (B, C, 1, H, W) → (B, 2C, 1, H, W) interleaving
  - _restore_image_tokens_order: inverse operation

Conditioning:
  - target_latents: flow-matching target (encoded output image, reordered)
  - cond_latents:   strength * noise + (1-strength) * source_latents
  - Model input:    cat([noisy_target, cond_latents], dim=1)
  - Dataset:        'image' key = target, 'control' key = source (defaults to 'image' for self-refinement)

Training target: noise - target_latents  (standard flow matching velocity)
"""

import copy
import gc
import logging
from typing import Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from qflux.trainer.base_trainer import BaseTrainer
from qflux.utils.images import make_image_shape_devisible


logger = logging.getLogger(__name__)

# Prompt template for refiner pipeline (Llama-style, differs from T2I Qwen-style)
PROMPT_TEMPLATE = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
PROMPT_TEMPLATE_DROP_IDX = 36  # tokens to skip (system prompt prefix)
TOKENIZER_MAX_LENGTH = 256
HIDDEN_STATE_SKIP_LAYER = 2  # use hidden_states[-(2+1)] = hidden_states[-3]

# How much noise to mix with the source image to form the conditioning latent.
# Matches the pipeline's default 'strength' parameter.
CONDITIONING_STRENGTH = 0.25


class HunyuanImageIT2ITrainer(BaseTrainer):
    """Trainer for HunyuanImage-2.1-Refiner image-to-image model.

    Takes a conditioning (source) image and a prompt, and is trained to generate
    a refined/enhanced output image. Uses a single Qwen text encoder and a 3D
    causal VAE with 5D latents.

    Dataset format:
      - 'image': target / ground-truth output image (B, 3, H, W)
      - 'control': source / conditioning image (B, 3, H, W) — optional,
                   defaults to 'image' for self-refinement when absent
      - 'prompt': list of caption strings
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        from diffusers.pipelines.hunyuan_image.pipeline_hunyuanimage_refiner import (
            HunyuanImageRefinerPipeline,
        )

        return HunyuanImageRefinerPipeline

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs):
        """Load all HunyuanImage Refiner components."""
        from diffusers.models import AutoencoderKLHunyuanImageRefiner, HunyuanImageTransformer2DModel
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading HunyuanImageRefinerPipeline components...")
        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings or {}

        # ----- VAE (3D causal, spatial_compression=16) -----
        vae_path = pretrains.get("vae", model_path)
        self.vae: AutoencoderKLHunyuanImageRefiner = AutoencoderKLHunyuanImageRefiner.from_pretrained(
            vae_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        )

        # ----- Text encoder: Qwen2.5-VL (Llama-style template) -----
        te_path = pretrains.get("text_encoder", model_path)
        from transformers import Qwen2Tokenizer

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
            else 16
        )
        self._vae_scaling_factor: float = getattr(self.vae.config, "scaling_factor", 1.03682)
        # Transformer in_channels is 2x latent_channels because model takes cat([noise, cond], dim=1)
        self.num_channels_latents: int = self.dit.config.in_channels // 2
        self.tokenizer_max_length = TOKENIZER_MAX_LENGTH

        # Freeze non-trainable components
        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logger.info(
            f"HunyuanImage Refiner components loaded. VAE scale: {self.vae_scale_factor}, "
            f"latent channels: {self.num_channels_latents}, "
            f"transformer in_channels: {self.dit.config.in_channels}, "
            f"guidance_embeds: {self.dit.config.guidance_embeds}"
        )

    # ------------------------------------------------------------------ #
    # Token interleaving (matches pipeline's static methods)              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _reorder_image_tokens(image_latents: torch.Tensor) -> torch.Tensor:
        """(B, C, 1, H, W) → (B, 2C, 1, H, W) via frame duplication + channel interleaving."""
        # Prepend a copy of frame 0 so we have 2 frames to interleave
        image_latents = torch.cat((image_latents[:, :, :1], image_latents), dim=2)
        B, C, F, H, W = image_latents.shape
        image_latents = image_latents.permute(0, 2, 1, 3, 4)         # (B, F, C, H, W)
        image_latents = image_latents.reshape(B, F // 2, C * 2, H, W)  # (B, F//2, 2C, H, W)
        image_latents = image_latents.permute(0, 2, 1, 3, 4).contiguous()  # (B, 2C, F//2, H, W)
        return image_latents

    @staticmethod
    def _restore_image_tokens_order(latents: torch.Tensor) -> torch.Tensor:
        """(B, 2C, 1, H, W) → (B, C, 1, H, W) — inverse of _reorder_image_tokens."""
        B, C2, F, H, W = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4)            # (B, F, 2C, H, W)
        latents = latents.reshape(B, F * 2, C2 // 2, H, W)  # (B, 2F, C, H, W)
        latents = latents.permute(0, 2, 1, 3, 4)            # (B, C, 2F, H, W)
        latents = latents[:, :, 1:]                          # remove first frame → (B, C, 2F-1, H, W)
        return latents

    # ------------------------------------------------------------------ #
    # Prompt encoding                                                      #
    # ------------------------------------------------------------------ #

    def _encode_qwen(
        self,
        prompt: list[str],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with Qwen2.5-VL using the refiner's Llama-style template.

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
        prompt_embeds = outputs.hidden_states[-(HIDDEN_STATE_SKIP_LAYER + 1)]
        prompt_embeds = prompt_embeds[:, drop_idx:].to(dtype=self.weight_dtype, device=device)
        attn_mask = txt_tokens.attention_mask[:, drop_idx:].to(device)
        return prompt_embeds, attn_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with the Qwen text encoder.

        Returns:
            prompt_embeds: (B, seq_len, qwen_dim)
            prompt_embeds_mask: (B, seq_len)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = next(self.text_encoder.parameters()).device
        return self._encode_qwen(prompt, device)

    # ------------------------------------------------------------------ #
    # VAE                                                                  #
    # ------------------------------------------------------------------ #

    def _vae_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a (B, 3, H, W) image to 5D latents (B, latent_channels, 1, H', W').

        Adds the frame dimension, encodes with the 3D causal VAE, applies
        _reorder_image_tokens, and scales by scaling_factor.
        """
        image = image.unsqueeze(2)  # (B, 3, H, W) → (B, 3, 1, H, W)
        with torch.inference_mode():
            raw = self.vae.encode(image).latent_dist.sample()  # (B, vae_C, 1, H', W')
        reordered = self._reorder_image_tokens(raw)  # (B, 2*vae_C, 1, H', W')
        return reordered * self._vae_scaling_factor

    def prepare_latents(
        self,
        image: torch.Tensor | None,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare noise latents and optionally VAE-encoded image latents (5D)."""
        h_lat = int(height) // self.vae_scale_factor
        w_lat = int(width) // self.vae_scale_factor
        shape = (batch_size, num_channels_latents, 1, h_lat, w_lat)
        device = next(self.vae.parameters()).device

        noise = randn_tensor(shape, device=device, dtype=dtype)

        image_latents = None
        if image is not None:
            image_latents = self._vae_encode_image(image.to(device=device, dtype=dtype))

        return noise, image_latents

    def decode_vae_latent(
        self, latents: torch.Tensor, target_height: int, target_width: int
    ) -> torch.Tensor:
        """Decode 5D latents (B, latent_channels, 1, H', W') to RGB image (B, 3, H, W) in [0, 1]."""
        latents = latents.to(self.vae.device, dtype=self.vae.dtype)
        latents = latents / self._vae_scaling_factor
        latents = self._restore_image_tokens_order(latents)  # (B, vae_C, 1, H', W')
        with torch.inference_mode():
            image = self.vae.decode(latents, return_dict=False)[0]  # (B, 3, 1, H, W)
        image = image.squeeze(2)  # (B, 3, H, W)
        return (image / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------ #
    # Embedding preparation                                                #
    # ------------------------------------------------------------------ #

    def _normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize [0,1] image to [-1,1] for VAE."""
        if image.max() <= 1.0 + 1e-6:
            return image * 2.0 - 1.0
        return image

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
        """Prepare embeddings for training or inference."""
        device = next(self.text_encoder.parameters()).device

        # ----- Prompt encoding -----
        pe, pm = self.encode_prompt(batch["prompt"])
        batch["prompt_embeds"] = pe
        batch["prompt_embeds_mask"] = pm

        if stage == "cache":
            empty_pe, empty_pm = self.encode_prompt([""])
            batch["empty_prompt_embeds"] = empty_pe
            batch["empty_prompt_embeds_mask"] = empty_pm

        # ----- VAE encode target image ('image' key) -----
        if "image" in batch:
            image = batch["image"]
            if isinstance(image, torch.Tensor) and image.ndim == 5:
                image = image.squeeze(2)
            image = self._normalize_image(image).to(device=device, dtype=self.weight_dtype)

            batch["height"] = image.shape[2]
            batch["width"] = image.shape[3]

            with torch.inference_mode():
                target_latents = self._vae_encode_image(image)
            batch["image_latents"] = target_latents  # (B, latent_channels, 1, H', W')

        # ----- VAE encode source/conditioning image ('control' key, or reuse target) -----
        if "control" in batch:
            ctrl = batch["control"]
            if isinstance(ctrl, torch.Tensor) and ctrl.ndim == 5:
                ctrl = ctrl.squeeze(2)
            ctrl = self._normalize_image(ctrl).to(device=device, dtype=self.weight_dtype)
            with torch.inference_mode():
                source_latents = self._vae_encode_image(ctrl)
            batch["source_latents"] = source_latents
        elif "image_latents" in batch:
            # Self-refinement: condition on the same image
            batch["source_latents"] = batch["image_latents"]

        return batch

    def prepare_cached_embeddings(self, batch: dict) -> dict:
        """Load cached embeddings (no transformation needed)."""
        # If source_latents not cached separately, fall back to image_latents
        if "source_latents" not in batch and "image_latents" in batch:
            batch["source_latents"] = batch["image_latents"]
        return batch

    # ------------------------------------------------------------------ #
    # Training loss                                                        #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """Flow-matching training loss for HunyuanImage Refiner.

        Model input = cat([noisy_target, cond_latents], dim=1) where:
          - noisy_target = (1-sigma) * target_latents + sigma * noise
          - cond_latents = strength * cond_noise + (1-strength) * source_latents
        """
        assert self.accelerator is not None
        device = self.accelerator.device

        target_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        source_latents = embeddings.get("source_latents", target_latents)
        source_latents = source_latents.to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        prompt_embeds_mask = embeddings["prompt_embeds_mask"].to(device)

        batch_size = target_latents.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(target_latents)
            cond_noise = torch.randn_like(target_latents)

            # Sample sigma uniformly in (0, 1)
            sigma = torch.rand(batch_size, device=device, dtype=self.weight_dtype)
            # Broadcast to 5D: (B, 1, 1, 1, 1)
            sigma_5d = sigma[:, None, None, None, None]

            # Flow-matching: x_t = (1 - sigma)*x_0 + sigma*noise
            noisy_target = (1.0 - sigma_5d) * target_latents + sigma_5d * noise

            # Conditioning latents: blend source with noise at fixed strength
            cond_latents = CONDITIONING_STRENGTH * cond_noise + (1.0 - CONDITIONING_STRENGTH) * source_latents

        # Concatenate along channel dim: (B, 2*latent_channels, 1, H', W')
        latent_model_input = torch.cat([noisy_target, cond_latents], dim=1)

        # Distilled guidance embedding
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

        model_pred = self.dit(
            hidden_states=latent_model_input,
            timestep=sigma,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            guidance=guidance,
            return_dict=False,
        )[0]

        # Standard flow-matching target: velocity = noise - target
        target = noise - target_latents

        loss = self.forward_loss(model_pred, target)
        return loss

    # ------------------------------------------------------------------ #
    # Cache step                                                           #
    # ------------------------------------------------------------------ #

    def cache_step(self, data: dict):
        """Save single-sample embeddings to disk cache."""
        cache_embeddings = {
            "image_latents": data["image_latents"].detach().cpu()[0],
            "source_latents": data["source_latents"].detach().cpu()[0],
            "prompt_embeds": data["prompt_embeds"].detach().cpu()[0],
            "prompt_embeds_mask": data["prompt_embeds_mask"].detach().cpu()[0],
            "empty_prompt_embeds": data["empty_prompt_embeds"].detach().cpu()[0],
            "empty_prompt_embeds_mask": data["empty_prompt_embeds_mask"].detach().cpu()[0],
        }

        file_hashes = data["file_hashes"]
        # source_hash: use 'control_hash' if present (paired editing), else 'image_hash'
        source_hash_key = "control_hash" if "control_hash" in file_hashes else "image_hash"

        map_keys = {
            "image_latents": "image_hash",
            "source_latents": source_hash_key,
            "prompt_embeds": "prompt_hash",
            "prompt_embeds_mask": "prompt_hash",
            "empty_prompt_embeds": "prompt_hash",
            "empty_prompt_embeds_mask": "prompt_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, file_hashes)

    # ------------------------------------------------------------------ #
    # Device management                                                    #
    # ------------------------------------------------------------------ #

    def setup_model_device_train_mode(self, stage: str = "fit", cache: bool = False):
        """Allocate models to devices for each stage."""
        if stage == "fit":
            assert hasattr(self, "accelerator"), "accelerator must be set before calling this"

            if self.cache_exist and self.use_cache:
                # Cache exists: only need transformer
                for attr in ("text_encoder", "vae"):
                    if hasattr(self, attr):
                        getattr(self, attr).cpu()
                if not self.config.validation.enabled:
                    for attr in ("text_encoder", "vae"):
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
                for attr in ("vae", "text_encoder", "dit"):
                    if hasattr(self, attr):
                        getattr(self, attr).to(self.accelerator.device)
                for attr in ("vae", "text_encoder"):
                    if hasattr(self, attr):
                        getattr(self, attr).requires_grad_(False).eval()
                self.dit.train()
                for name, param in self.dit.named_parameters():
                    param.requires_grad = "lora" in name

        elif stage == "cache":
            # Cache stage: need VAE + text encoder; free transformer
            self.vae.to(self.config.cache.devices.vae)
            self.text_encoder.to(self.config.cache.devices.text_encoder)

            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()

            if hasattr(self, "dit"):
                self.dit.cpu()
                del self.dit
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("cache mode: VAE + text_encoder on device; transformer freed")

        elif stage == "predict":
            devices = self.config.predict.devices
            self.vae.to(devices.vae)
            self.text_encoder.to(devices.text_encoder)
            self.dit.to(devices.dit)

            for attr in ("vae", "text_encoder", "dit"):
                if hasattr(self, attr):
                    getattr(self, attr).requires_grad_(False).eval()
            logger.info(
                f"predict mode: vae={devices.vae}, te={devices.text_encoder}, dit={devices.dit}"
            )

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def prepare_predict_batch_data(
        self,
        prompt: str | list[str],
        image: torch.Tensor | None = None,
        height: int = 1024,
        width: int = 1024,
        negative_prompt: str | list[str] | None = None,
        distilled_guidance_scale: float = 3.25,
        num_inference_steps: int = 4,
        strength: float = CONDITIONING_STRENGTH,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> dict:
        """Prepare inference batch for HunyuanImage IT2I.

        Args:
            prompt: text prompt(s)
            image: conditioning source image (B, 3, H, W) in [0, 1]; required for IT2I
            height / width: output resolution (will be snapped to vae_scale_factor multiple)
            strength: noise level for conditioning (0 = fully preserve source, 1 = full noise)
        """
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
            "strength": strength,
        }
        if image is not None:
            data["control"] = image
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            data["negative_prompt"] = negative_prompt

        return data

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """Denoising loop for HunyuanImage IT2I.

        Uses pre-computed embeddings. Source image must be in embeddings['source_latents']
        (or will be re-encoded from embeddings['control']).

        Returns final latents (B, latent_channels, 1, H_lat, W_lat).
        """
        num_inference_steps = embeddings["num_inference_steps"]
        distilled_guidance_scale = embeddings.get("distilled_guidance_scale", 3.25)
        strength = embeddings.get("strength", CONDITIONING_STRENGTH)
        prompt_embeds = embeddings["prompt_embeds"]
        prompt_embeds_mask = embeddings["prompt_embeds_mask"]
        height = embeddings["height"]
        width = embeddings["width"]
        batch_size = prompt_embeds.shape[0]

        device = self.dit.device
        dtype = self.weight_dtype

        # Move embeddings to device
        prompt_embeds = prompt_embeds.to(device, dtype=dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(device)

        # Prepare source latents (conditioning)
        if "source_latents" in embeddings:
            source_latents = embeddings["source_latents"].to(device, dtype=dtype)
        elif "control" in embeddings:
            ctrl = self._normalize_image(embeddings["control"])
            ctrl = ctrl.to(device=device, dtype=dtype)
            source_latents = self._vae_encode_image(ctrl)
        else:
            raise ValueError("IT2I sampling requires 'source_latents' or 'control' image in embeddings")

        h_lat = int(height) // self.vae_scale_factor
        w_lat = int(width) // self.vae_scale_factor

        if "latents" in embeddings:
            latents = embeddings["latents"].to(device, dtype=torch.float32)
        else:
            latents = randn_tensor(
                (batch_size, self.num_channels_latents, 1, h_lat, w_lat),
                device=device,
                dtype=torch.float32,
            )

        # Build conditioning latents from source (fixed throughout denoising)
        cond_noise = randn_tensor(source_latents.shape, device=device, dtype=torch.float32)
        cond_latents = (
            strength * cond_noise.to(dtype) + (1.0 - strength) * source_latents
        ).to(torch.float32)

        # Guidance embedding
        if self.dit.config.guidance_embeds:
            guidance = torch.tensor(
                [distilled_guidance_scale * 1000.0] * batch_size,
                dtype=dtype,
                device=device,
            )
        else:
            guidance = None

        # Timesteps: sigmas from 1.0 → 0.0
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        self.sampling_scheduler.set_timesteps(sigmas=sigmas, device=device)
        timesteps = self.sampling_scheduler.timesteps
        self.sampling_scheduler.set_begin_index(0)

        with torch.inference_mode():
            for i, t in tqdm(
                enumerate(timesteps), total=len(timesteps), desc="HunyuanImage Refiner"
            ):
                timestep = t.expand(batch_size).to(dtype)

                # Concatenate denoising latents with (fixed) conditioning latents
                latent_model_input = torch.cat(
                    [latents.to(dtype), cond_latents.to(dtype)], dim=1
                )

                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_embeds_mask,
                    guidance=guidance,
                    return_dict=False,
                )[0]

                latents = self.sampling_scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]

        return latents
