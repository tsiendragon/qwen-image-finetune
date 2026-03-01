"""Trainer for Efficient-Large-Model/Sana_Sprint_1.6B (text-to-image, SCM flow matching).

Architecture overview:
  - Transformer: SanaTransformer2DModel (4D latents, not packed)
  - VAE: AutoencoderDC, vae_scale_factor = 2^(len(encoder_block_out_channels)-1) ≈ 32
  - Text encoder: Gemma2Model + GemmaTokenizerFast, max_length=300
    - select_index = [0] + list(range(-max_length+1, 0)) — keeps BOS + last N-1 tokens
  - Scheduler: DPMSolverMultistepScheduler

SCM (Stochastic Consistency Matching) details:
  - Raw timestep t ∈ [0, π/2]; SCM transform: scm_t = sin(t)/(cos(t)+sin(t)) ∈ [0, 1]
  - Latent model input (in sigma_data space):
      lmi = (noisy_latents / sigma_data) * sqrt(scm_t² + (1-scm_t)²)
  - Model raw output v_raw, post-processed for DPM solver:
      v_post = ((1-2s)*lmi + (1-2s+2s²)*v_raw) / sqrt(s²+(1-s)²)  × sigma_data
  - Training target (SCM-consistent, so that post_processed = velocity = noise - x_0):
      v_raw_target = scale * [(1-s+2s²)*noise - (2-3s+2s²)*x_0] / (1-2s+2s²)
      where scale = sqrt(s²+(1-s)²), s = scm_t

Latent format:
  - 4D (B, C, H//vae_scale, W//vae_scale) — not packed
  - num_channels_latents = transformer.config.in_channels
  - Sigma_data-scaled during denoising; decode via: latents/sigma_data/vae_scaling_factor → vae.decode()

VAE:
  - AutoencoderDC: encode returns .latent (not .latent_dist)
  - Decode: vae.decode(latents).sample → [-1, 1] image
"""

import copy
import gc
import logging
import math
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from qflux.trainer.base_trainer import BaseTrainer
from qflux.utils.images import make_image_shape_devisible


logger = logging.getLogger(__name__)

MAX_SEQUENCE_LENGTH = 300


class SanaSprintT2ITrainer(BaseTrainer):
    """Trainer for Sana Sprint text-to-image model.

    Uses a Gemma2 text encoder with a SanaTransformer2DModel using SCM
    (Stochastic Consistency Matching) flow matching. Latents are 4D (not packed).
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        from diffusers.pipelines.sana.pipeline_sana_sprint import SanaSprintPipeline
        return SanaSprintPipeline

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs):
        from diffusers.models import AutoencoderDC, SanaTransformer2DModel
        from diffusers.schedulers import DPMSolverMultistepScheduler
        from transformers import AutoTokenizer, Gemma2Model

        logger.info("Loading SanaSprint components...")
        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings or {}

        # ----- VAE (AutoencoderDC) -----
        vae_path = pretrains.get("vae", model_path)
        self.vae: AutoencoderDC = AutoencoderDC.from_pretrained(
            vae_path, subfolder="vae", torch_dtype=self.weight_dtype
        )

        # ----- Text encoder: Gemma2 -----
        te_path = pretrains.get("text_encoder", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(te_path, subfolder="tokenizer")
        self.text_encoder: Gemma2Model = Gemma2Model.from_pretrained(
            te_path, subfolder="text_encoder", torch_dtype=self.weight_dtype
        )

        # ----- Transformer -----
        self.dit: SanaTransformer2DModel = SanaTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        )

        # ----- Scheduler -----
        self.scheduler: DPMSolverMultistepScheduler = (
            DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")
        )
        self.sampling_scheduler = copy.deepcopy(self.scheduler)

        # ----- Derived parameters -----
        if hasattr(self.vae.config, "encoder_block_out_channels"):
            self.vae_scale_factor: int = (
                2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            )
        else:
            self.vae_scale_factor = 32
        self._vae_scaling_factor: float = self.vae.config.scaling_factor
        self.num_channels_latents: int = self.dit.config.in_channels
        # sigma_data: latent scale convention for SCM (typically 1.0)
        self.sigma_data: float = float(
            getattr(self.scheduler.config, "sigma_data", 1.0)
        )
        # Distilled guidance scale (embedding fed to transformer each step)
        self._guidance_scale: float = float(
            getattr(self.config.model, "distilled_guidance_scale", None) or 5.0
        )

        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logger.info(
            f"SanaSprint loaded. vae_scale={self.vae_scale_factor}, "
            f"latent_channels={self.num_channels_latents}, "
            f"sigma_data={self.sigma_data}"
        )

    # ------------------------------------------------------------------ #
    # Prompt encoding                                                      #
    # ------------------------------------------------------------------ #

    def encode_prompt(
        self,
        prompt: str | list[str],
        max_length: int = MAX_SEQUENCE_LENGTH,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt with Gemma2 text encoder.

        Token selection: [BOS_token] + [last (max_length-1) tokens].
        With padding="max_length" + truncation this covers the full sequence.

        Returns:
            prompt_embeds:  (B, max_length, hidden_dim)
            attention_mask: (B, max_length) — bool, 1 = real token
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = next(self.text_encoder.parameters()).device

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        with torch.inference_mode():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        prompt_embeds = outputs[0]  # (B, seq_len, hidden)

        # Select [first token] + [last max_length-1 tokens]
        select_index = [0] + list(range(-max_length + 1, 0))
        prompt_embeds = prompt_embeds[:, select_index].to(dtype=self.weight_dtype)
        attention_mask = attention_mask[:, select_index].bool()

        return prompt_embeds, attention_mask

    def prepare_latents(
        self,
        image: torch.Tensor | None,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare noise latents and (optionally) VAE-encoded image latents."""
        h_lat = int(height) // self.vae_scale_factor
        w_lat = int(width) // self.vae_scale_factor
        device = next(self.vae.parameters()).device
        noise = randn_tensor((batch_size, num_channels_latents, h_lat, w_lat), device=device, dtype=dtype)
        image_latents = None
        if image is not None:
            image_latents = self._vae_encode(image.to(device=device, dtype=dtype))
        return noise, image_latents

    # ------------------------------------------------------------------ #
    # VAE helpers                                                          #
    # ------------------------------------------------------------------ #

    def _vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode (B, 3, H, W) → (B, C, H_lat, W_lat) using AutoencoderDC.

        AutoencoderDC returns .latent (not .latent_dist.sample()).
        Returns VAE-scaling-factor-normalized latents (NOT sigma_data-scaled).
        """
        with torch.inference_mode():
            return self.vae.encode(image).latent * self._vae_scaling_factor

    def decode_vae_latent(self, latents: torch.Tensor, target_height: int = 0, target_width: int = 0) -> torch.Tensor:
        """Decode latents → RGB (B, 3, H, W) in [0, 1].

        Input latents must be in sigma_data-normalized + VAE-scaling-factor space.
        Decode: latents / sigma_data / vae_scaling_factor → vae.decode()
        """
        latents = latents.to(self.vae.device, dtype=self.vae.dtype)
        latents = latents / self.sigma_data / self._vae_scaling_factor
        with torch.inference_mode():
            image = self.vae.decode(latents).sample
        return (image / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------ #
    # Embedding preparation                                                #
    # ------------------------------------------------------------------ #

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
        device = next(self.text_encoder.parameters()).device

        pe, attn_mask = self.encode_prompt(batch["prompt"])
        batch["prompt_embeds"] = pe
        batch["prompt_embeds_mask"] = attn_mask

        if stage == "cache":
            empty_pe, empty_mask = self.encode_prompt([""] * len(batch["prompt"]))
            batch["empty_prompt_embeds"] = empty_pe
            batch["empty_prompt_embeds_mask"] = empty_mask

        if "image" in batch:
            image = batch["image"]
            if isinstance(image, torch.Tensor) and image.ndim == 5:
                image = image.squeeze(2)
            if image.max() <= 1.0 + 1e-6:
                image = image * 2.0 - 1.0
            image = image.to(device=device, dtype=self.weight_dtype)

            batch["height"] = image.shape[2]
            batch["width"] = image.shape[3]

            with torch.inference_mode():
                # _vae_encode returns VAE-scale-normalized latents (without sigma_data)
                batch["image_latents"] = self._vae_encode(image)

        return batch

    def prepare_cached_embeddings(self, batch: dict) -> dict:
        return batch

    # ------------------------------------------------------------------ #
    # Training loss                                                        #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        assert self.accelerator is not None
        device = self.accelerator.device

        # image_latents: VAE-scale-normalized, WITHOUT sigma_data
        x_0 = embeddings["image_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        prompt_mask = embeddings["prompt_embeds_mask"].to(device)
        B, C, H, W = x_0.shape

        # sigma_data-scale everything for consistency with inference
        x_0_sigma = x_0 * self.sigma_data  # (B, C, H, W)

        with torch.no_grad():
            noise = torch.randn_like(x_0_sigma) * self.sigma_data  # same scale

            # Sample scm_t ~ U(0, 1) — the SCM noise fraction
            scm_t = torch.rand(B, device=device, dtype=self.weight_dtype)
            s = scm_t[:, None, None, None]

            # Noisy latents (sigma_data space)
            noisy = (1.0 - s) * x_0_sigma + s * noise

            # Scale for transformer input
            scale = torch.sqrt(s ** 2 + (1.0 - s) ** 2)  # (B, 1, 1, 1)
            lmi = (noisy / self.sigma_data) * scale  # normalized + time-scaled

        # Guidance tensor per sample
        guidance = self._guidance_scale * torch.ones(B, device=device, dtype=self.weight_dtype)

        model_pred = self.dit(
            lmi,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            guidance=guidance,
            timestep=scm_t,
            return_dict=False,
        )[0]

        # SCM-consistent training target (derives from post-processing inversion):
        # v_raw_target = scale * [(1-s+2s²)*noise_norm - (2-3s+2s²)*x_0_norm] / (1-2s+2s²)
        # where noise_norm, x_0_norm are in the lmi space (normalized, no sigma_data)
        noise_norm = noise / self.sigma_data  # (B, C, H, W)
        x_0_norm = x_0_sigma / self.sigma_data  # = x_0 in VAE-scale-normalized

        denom = 1.0 - 2.0 * s + 2.0 * s ** 2  # always ≥ 0.5, never zero
        coef_noise = 1.0 - s + 2.0 * s ** 2
        coef_x0 = 2.0 - 3.0 * s + 2.0 * s ** 2
        target = scale * (coef_noise * noise_norm - coef_x0 * x_0_norm) / denom

        return self.forward_loss(model_pred, target)

    # ------------------------------------------------------------------ #
    # Cache step                                                           #
    # ------------------------------------------------------------------ #

    def cache_step(self, data: dict):
        cache_embeddings = {
            "image_latents": data["image_latents"].detach().cpu()[0],
            "prompt_embeds": data["prompt_embeds"].detach().cpu()[0],
            "prompt_embeds_mask": data["prompt_embeds_mask"].detach().cpu()[0],
            "empty_prompt_embeds": data["empty_prompt_embeds"].detach().cpu()[0],
            "empty_prompt_embeds_mask": data["empty_prompt_embeds_mask"].detach().cpu()[0],
        }
        map_keys = {
            "image_latents": "image_hash",
            "prompt_embeds": "prompt_hash",
            "prompt_embeds_mask": "prompt_hash",
            "empty_prompt_embeds": "prompt_hash",
            "empty_prompt_embeds_mask": "prompt_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    # ------------------------------------------------------------------ #
    # Device management                                                    #
    # ------------------------------------------------------------------ #

    def setup_model_device_train_mode(self, stage: str = "fit", cache: bool = False):
        if stage == "fit":
            assert hasattr(self, "accelerator")
            if self.cache_exist and self.use_cache:
                for attr in ("text_encoder", "vae"):
                    if hasattr(self, attr):
                        getattr(self, attr).cpu()
                if not self.config.validation.enabled:
                    for attr in ("text_encoder", "vae"):
                        if hasattr(self, attr):
                            delattr(self, attr)
                gc.collect(); torch.cuda.empty_cache()
                self.dit.to(self.accelerator.device).train()
                for n, p in self.dit.named_parameters():
                    p.requires_grad = "lora" in n
            else:
                for attr in ("vae", "text_encoder", "dit"):
                    if hasattr(self, attr):
                        getattr(self, attr).to(self.accelerator.device)
                for attr in ("vae", "text_encoder"):
                    if hasattr(self, attr):
                        getattr(self, attr).requires_grad_(False).eval()
                self.dit.train()
                for n, p in self.dit.named_parameters():
                    p.requires_grad = "lora" in n

        elif stage == "cache":
            self.vae.to(self.config.cache.devices.vae)
            self.text_encoder.to(self.config.cache.devices.text_encoder)
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            if hasattr(self, "dit"):
                self.dit.cpu(); del self.dit
            gc.collect(); torch.cuda.empty_cache()

        elif stage == "predict":
            d = self.config.predict.devices
            self.vae.to(d.vae)
            self.text_encoder.to(d.text_encoder)
            self.dit.to(d.dit)
            for attr in ("vae", "text_encoder", "dit"):
                if hasattr(self, attr):
                    getattr(self, attr).requires_grad_(False).eval()

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def prepare_predict_batch_data(
        self,
        prompt: str | list[str],
        height: int = 1024,
        width: int = 1024,
        negative_prompt: str | list[str] = "",
        guidance_scale: float = 5.0,
        num_inference_steps: int = 20,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> dict:
        if isinstance(prompt, str):
            prompt = [prompt]
        self.weight_dtype = weight_dtype
        height, width = make_image_shape_devisible(height, width, self.vae_scale_factor)
        data: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        }
        if negative_prompt:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            data["negative_prompt"] = negative_prompt
        return data

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """Denoising loop for Sana Sprint. Returns decoded image (B, 3, H, W) in [0, 1].

        Follows the pipeline's SCM inference exactly:
          scm_t = sin(t) / (cos(t) + sin(t))
          lmi = (latents / sigma_data) * sqrt(scm_t² + (1-scm_t)²)
          v_post = ((1-2s)*lmi + (1-2s+2s²)*v_raw) / sqrt(s²+(1-s)²) * sigma_data
        """
        num_steps = embeddings["num_inference_steps"]
        guidance_scale = embeddings.get("guidance_scale", self._guidance_scale)
        prompt_embeds = embeddings["prompt_embeds"]
        prompt_mask = embeddings.get("prompt_embeds_mask")
        height = embeddings["height"]
        width = embeddings["width"]
        batch_size = prompt_embeds.shape[0]
        device = self.dit.device
        dtype = self.weight_dtype

        prompt_embeds = prompt_embeds.to(device, dtype=dtype)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(device)

        h_lat = height // self.vae_scale_factor
        w_lat = width // self.vae_scale_factor

        if "latents" in embeddings:
            latents = embeddings["latents"].to(device, dtype=torch.float32)
            # Ensure sigma_data scaling
            if latents.abs().mean() < self.sigma_data * 0.5:
                latents = latents * self.sigma_data
        else:
            latents = randn_tensor(
                (batch_size, self.num_channels_latents, h_lat, w_lat),
                device=device, dtype=torch.float32,
            ) * self.sigma_data

        self.sampling_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.sampling_scheduler.timesteps

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg_prompts = embeddings.get("negative_prompt", [""] * batch_size)
            neg_embeds, neg_mask = self.encode_prompt(neg_prompts)
            neg_embeds = neg_embeds.to(device, dtype=dtype)
            if neg_mask is not None:
                neg_mask = neg_mask.to(device)

        with torch.inference_mode():
            for t in tqdm(timesteps, desc="Sana Sprint generating"):
                t_scalar = float(t)
                scm_t_val = math.sin(t_scalar) / (math.cos(t_scalar) + math.sin(t_scalar) + 1e-12)
                scm_t = torch.full((batch_size,), scm_t_val, device=device, dtype=dtype)
                s = scm_t_val

                scale = math.sqrt(s ** 2 + (1.0 - s) ** 2)
                lmi = (latents.to(dtype) / self.sigma_data) * scale

                guidance = guidance_scale * torch.ones(batch_size, device=device, dtype=dtype)

                noise_pred = self.dit(
                    lmi,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_mask,
                    guidance=guidance,
                    timestep=scm_t,
                    return_dict=False,
                )[0]

                if do_cfg:
                    neg_guidance = torch.ones(batch_size, device=device, dtype=dtype)
                    neg_pred = self.dit(
                        lmi,
                        encoder_hidden_states=neg_embeds,
                        encoder_attention_mask=neg_mask,
                        guidance=neg_guidance,
                        timestep=scm_t,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

                # SCM post-processing: convert raw output to velocity for DPM solver
                # v_post = ((1-2s)*lmi + (1-2s+2s²)*v_raw) / sqrt(s²+(1-s)²) * sigma_data
                coef_lmi = 1.0 - 2.0 * s
                coef_vraw = 1.0 - 2.0 * s + 2.0 * s ** 2
                noise_pred_post = (
                    (coef_lmi * lmi.to(torch.float32) + coef_vraw * noise_pred.to(torch.float32)) / scale
                ) * self.sigma_data

                latents = self.sampling_scheduler.step(
                    noise_pred_post, t, latents.to(torch.float32), return_dict=False
                )[0]

        # Decode: latents / sigma_data / vae_scaling_factor → vae.decode()
        return self.decode_vae_latent(latents)
