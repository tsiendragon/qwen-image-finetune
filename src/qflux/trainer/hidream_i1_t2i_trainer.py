"""Trainer for HiDream-ai/HiDream-I1-Full (text-to-image, 4-encoder flow matching).

Architecture overview:
  - Transformer: HiDreamImageTransformer2DModel (double-stream + single-stream blocks, MoE FFN)
  - VAE: AutoencoderKL, in_channels=64, vae_scale_factor = 2^(len(block_out_channels)-1)
  - Text encoders: CLIP×2 (pooled) + T5 (sequence) + Llama-3.1 (all hidden layers)
  - Scheduler: FlowMatchEulerDiscreteScheduler

Four text encoders:
  1. CLIP #1: CLIPTextModelWithProjection → text_embeds (B, clip_proj_dim)
  2. CLIP #2: CLIPTextModelWithProjection → text_embeds (B, clip_proj_dim)
  3. T5:      T5EncoderModel              → last_hidden_state (B, seq, hidden)
  4. Llama:   LlamaForCausalLM            → hidden_states[1:] stacked (num_layers, B, seq, hidden)
  pooled_prompt_embeds = cat([clip1_pooled, clip2_pooled], dim=-1)

Transformer call signature:
  transformer(
      hidden_states=latents,                      # (B, 64, H, W) — 4D, patching done internally
      timesteps=sigma,                            # (B,)
      encoder_hidden_states_t5=t5_embeds,         # (B, seq, hidden)
      encoder_hidden_states_llama3=llama_embeds,  # (num_layers, B, seq, hidden)
      pooled_embeds=pooled_embeds,                # (B, clip1_dim+clip2_dim)
      return_dict=False,
  )

IMPORTANT: Model output must be negated before passing to scheduler:
  noise_pred = -transformer_output
  → training target = x_0 - noise (NOT noise - x_0)

Cache strategy:
  - Cached: image_latents, t5_embeds, llama_embeds (stacked), pooled_embeds
  - At training with cache: CLIP + T5 + VAE offloaded; Llama may be offloaded too

Latent normalization for decode:
  latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
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

CLIP_MAX_LENGTH = 77      # standard CLIP context length
T5_MAX_LENGTH = 128       # T5 sequence length
LLAMA_MAX_LENGTH = 128    # Llama sequence length


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                     base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class HiDreamI1T2ITrainer(BaseTrainer):
    """Trainer for HiDream-I1 text-to-image model.

    Uses 4 text encoders (2×CLIP pooled, T5 sequence, Llama all-layer hidden states)
    with a flow matching DiT. The transformer output is negated before computing loss.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        from diffusers.pipelines.hidream_image.pipeline_hidream_image import HiDreamImagePipeline
        return HiDreamImagePipeline

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs):
        from diffusers.models import AutoencoderKL
        from diffusers.models.transformers.transformer_hidream_image import HiDreamImageTransformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
            T5EncoderModel,
            T5Tokenizer,
        )

        logger.info("Loading HiDream-I1 components...")
        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings or {}

        # ----- VAE -----
        vae_path = pretrains.get("vae", model_path)
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            vae_path, subfolder="vae", torch_dtype=self.weight_dtype
        )

        # ----- Text encoder 1: CLIP #1 -----
        te1_path = pretrains.get("text_encoder", model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(te1_path, subfolder="tokenizer")
        self.text_encoder: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
            te1_path, subfolder="text_encoder", torch_dtype=self.weight_dtype
        )

        # ----- Text encoder 2: CLIP #2 -----
        te2_path = pretrains.get("text_encoder_2", model_path)
        self.tokenizer_2 = CLIPTokenizer.from_pretrained(te2_path, subfolder="tokenizer_2")
        self.text_encoder_2: CLIPTextModelWithProjection = CLIPTextModelWithProjection.from_pretrained(
            te2_path, subfolder="text_encoder_2", torch_dtype=self.weight_dtype
        )

        # ----- Text encoder 3: T5 -----
        te3_path = pretrains.get("text_encoder_3", model_path)
        self.tokenizer_3 = T5Tokenizer.from_pretrained(te3_path, subfolder="tokenizer_3")
        self.text_encoder_3: T5EncoderModel = T5EncoderModel.from_pretrained(
            te3_path, subfolder="text_encoder_3", torch_dtype=self.weight_dtype
        )

        # ----- Text encoder 4: Llama-3.1 -----
        te4_path = pretrains.get("text_encoder_4", model_path)
        self.tokenizer_4 = AutoTokenizer.from_pretrained(te4_path, subfolder="tokenizer_4")
        if self.tokenizer_4.pad_token is None:
            self.tokenizer_4.pad_token = self.tokenizer_4.eos_token
        self.text_encoder_4 = AutoModelForCausalLM.from_pretrained(
            te4_path, subfolder="text_encoder_4",
            torch_dtype=self.weight_dtype,
        )

        # ----- Transformer -----
        self.dit: HiDreamImageTransformer2DModel = HiDreamImageTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        )

        # ----- Scheduler -----
        self.scheduler: FlowMatchEulerDiscreteScheduler = (
            FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        )
        self.sampling_scheduler = copy.deepcopy(self.scheduler)

        # ----- Derived parameters -----
        self.vae_scale_factor: int = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self.vae.config, "block_out_channels") else 8
        )
        self._vae_scaling_factor: float = self.vae.config.scaling_factor
        self._vae_shift_factor: float = getattr(self.vae.config, "shift_factor", 0.0)
        # in_channels includes patchification: num latent channels = in_channels / (patch_size^2)
        patch_size: int = getattr(self.dit.config, "patch_size", 2)
        self.num_channels_latents: int = self.dit.config.in_channels // (patch_size ** 2)
        self._patch_size = patch_size

        for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "vae", "dit"):
            if hasattr(self, attr):
                getattr(self, attr).requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logger.info(
            f"HiDream-I1 loaded. vae_scale={self.vae_scale_factor}, "
            f"latent_channels={self.num_channels_latents}, patch_size={self._patch_size}"
        )

    # ------------------------------------------------------------------ #
    # Prompt encoding                                                      #
    # ------------------------------------------------------------------ #

    def encode_prompt(
        self,
        prompt: str | list[str],
        max_sequence_length: int = T5_MAX_LENGTH,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompt with all 4 text encoders.

        Returns:
            t5_embeds:     (B, T5_MAX_LENGTH, t5_hidden)
            llama_embeds:  (num_layers, B, LLAMA_MAX_LENGTH, llama_hidden)
            pooled_embeds: (B, clip1_proj_dim + clip2_proj_dim)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        B = len(prompt)
        device_clip = next(self.text_encoder.parameters()).device
        device_t5 = next(self.text_encoder_3.parameters()).device
        device_llama = next(self.text_encoder_4.parameters()).device

        # ── CLIP #1 ──────────────────────────────────────────────────────
        tokens_1 = self.tokenizer(
            prompt, padding="max_length", max_length=CLIP_MAX_LENGTH,
            truncation=True, return_tensors="pt",
        )
        with torch.inference_mode():
            clip1_out = self.text_encoder(
                tokens_1.input_ids.to(device_clip),
                output_hidden_states=False,
            )
        pooled_1 = clip1_out.text_embeds.to(dtype=self.weight_dtype)  # (B, proj_dim)

        # ── CLIP #2 ──────────────────────────────────────────────────────
        tokens_2 = self.tokenizer_2(
            prompt, padding="max_length", max_length=CLIP_MAX_LENGTH,
            truncation=True, return_tensors="pt",
        )
        with torch.inference_mode():
            clip2_out = self.text_encoder_2(
                tokens_2.input_ids.to(device_clip),
                output_hidden_states=False,
            )
        pooled_2 = clip2_out.text_embeds.to(dtype=self.weight_dtype)  # (B, proj_dim)

        pooled_embeds = torch.cat([pooled_1, pooled_2], dim=-1)  # (B, 2*proj_dim)

        # ── T5 ───────────────────────────────────────────────────────────
        tokens_3 = self.tokenizer_3(
            prompt, padding="max_length", max_length=max_sequence_length,
            truncation=True, return_tensors="pt",
        )
        with torch.inference_mode():
            t5_out = self.text_encoder_3(
                input_ids=tokens_3.input_ids.to(device_t5),
                attention_mask=tokens_3.attention_mask.to(device_t5),
            )
        t5_embeds = t5_out[0].to(dtype=self.weight_dtype)  # (B, seq, hidden)

        # ── Llama-3.1 ────────────────────────────────────────────────────
        tokens_4 = self.tokenizer_4(
            prompt, padding="max_length", max_length=LLAMA_MAX_LENGTH,
            truncation=True, return_tensors="pt",
        )
        with torch.inference_mode():
            llama_out = self.text_encoder_4(
                input_ids=tokens_4.input_ids.to(device_llama),
                attention_mask=tokens_4.attention_mask.to(device_llama),
                output_hidden_states=True,
            )
        # Stack all transformer layers (skip embedding layer at index 0)
        llama_embeds = torch.stack(
            [h.to(dtype=self.weight_dtype) for h in llama_out.hidden_states[1:]], dim=0
        )  # (num_layers, B, seq, hidden)

        return t5_embeds, llama_embeds, pooled_embeds

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
    # VAE                                                                  #
    # ------------------------------------------------------------------ #

    def _vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode (B, 3, H, W) → (B, C, H_lat, W_lat) latents."""
        with torch.inference_mode():
            return self.vae.encode(image).latent_dist.sample()

    def decode_vae_latent(self, latents: torch.Tensor, target_height: int = 0, target_width: int = 0) -> torch.Tensor:
        """Decode latents → RGB (B, 3, H, W) in [0, 1]."""
        latents = latents.to(self.vae.device, dtype=self.vae.dtype)
        latents = (latents / self._vae_scaling_factor) + self._vae_shift_factor
        with torch.inference_mode():
            image = self.vae.decode(latents, return_dict=False)[0]
        return (image / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------ #
    # Embedding preparation                                                #
    # ------------------------------------------------------------------ #

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
        device = next(self.text_encoder.parameters()).device

        t5_pe, llama_pe, pooled_pe = self.encode_prompt(batch["prompt"])
        batch["prompt_embeds_t5"] = t5_pe
        batch["prompt_embeds_llama3"] = llama_pe
        batch["pooled_prompt_embeds"] = pooled_pe

        if stage == "cache":
            empty_t5, empty_llama, empty_pooled = self.encode_prompt([""] * len(batch["prompt"]))
            batch["empty_prompt_embeds_t5"] = empty_t5
            batch["empty_prompt_embeds_llama3"] = empty_llama
            batch["empty_pooled_prompt_embeds"] = empty_pooled

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

        image_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        t5_embeds = embeddings["prompt_embeds_t5"].to(self.weight_dtype).to(device)
        llama_embeds = embeddings["prompt_embeds_llama3"].to(self.weight_dtype).to(device)
        pooled_embeds = embeddings["pooled_prompt_embeds"].to(self.weight_dtype).to(device)
        B = image_latents.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(image_latents)
            sigma = torch.rand(B, device=device, dtype=self.weight_dtype)
            s = sigma[:, None, None, None]
            noisy = (1.0 - s) * image_latents + s * noise

        model_out = self.dit(
            hidden_states=noisy,
            timesteps=sigma,
            encoder_hidden_states_t5=t5_embeds,
            encoder_hidden_states_llama3=llama_embeds,
            pooled_embeds=pooled_embeds,
            return_dict=False,
        )[0]

        # HiDream negates: noise_pred = -model_out → target for -model_out is velocity = noise - x_0
        # So target for raw model_out is -(noise - x_0) = x_0 - noise
        target = image_latents - noise  # reversed sign: x_0 - noise
        return self.forward_loss(-model_out, -target)  # equivalent: forward_loss(model_out, x_0-noise)

    # ------------------------------------------------------------------ #
    # Cache step                                                           #
    # ------------------------------------------------------------------ #

    def cache_step(self, data: dict):
        cache_embeddings = {
            "image_latents": data["image_latents"].detach().cpu()[0],
            "prompt_embeds_t5": data["prompt_embeds_t5"].detach().cpu()[0],
            "prompt_embeds_llama3": data["prompt_embeds_llama3"].detach().cpu()[:, 0],  # (num_layers, seq, hidden)
            "pooled_prompt_embeds": data["pooled_prompt_embeds"].detach().cpu()[0],
            "empty_prompt_embeds_t5": data["empty_prompt_embeds_t5"].detach().cpu()[0],
            "empty_prompt_embeds_llama3": data["empty_prompt_embeds_llama3"].detach().cpu()[:, 0],
            "empty_pooled_prompt_embeds": data["empty_pooled_prompt_embeds"].detach().cpu()[0],
        }
        map_keys = {
            "image_latents": "image_hash",
            "prompt_embeds_t5": "prompt_hash",
            "prompt_embeds_llama3": "prompt_hash",
            "pooled_prompt_embeds": "prompt_hash",
            "empty_prompt_embeds_t5": "prompt_hash",
            "empty_prompt_embeds_llama3": "prompt_hash",
            "empty_pooled_prompt_embeds": "prompt_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    # ------------------------------------------------------------------ #
    # Device management                                                    #
    # ------------------------------------------------------------------ #

    def setup_model_device_train_mode(self, stage: str = "fit", cache: bool = False):
        if stage == "fit":
            assert hasattr(self, "accelerator")
            if self.cache_exist and self.use_cache:
                # Offload all encoders + VAE; keep only DIT
                for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "vae"):
                    if hasattr(self, attr):
                        getattr(self, attr).cpu()
                if not self.config.validation.enabled:
                    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "vae"):
                        if hasattr(self, attr):
                            delattr(self, attr)
                gc.collect(); torch.cuda.empty_cache()
                self.dit.to(self.accelerator.device).train()
                for n, p in self.dit.named_parameters():
                    p.requires_grad = "lora" in n
            else:
                for attr in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "dit"):
                    if hasattr(self, attr):
                        getattr(self, attr).to(self.accelerator.device)
                for attr in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"):
                    if hasattr(self, attr):
                        getattr(self, attr).requires_grad_(False).eval()
                self.dit.train()
                for n, p in self.dit.named_parameters():
                    p.requires_grad = "lora" in n

        elif stage == "cache":
            for attr in ("vae", "text_encoder", "text_encoder_2"):
                if hasattr(self, attr):
                    getattr(self, attr).to(self.config.cache.devices.vae)
            for attr in ("text_encoder_3",):
                if hasattr(self, attr):
                    getattr(self, attr).to(self.config.cache.devices.text_encoder)
            # Llama might need a separate device; use text_encoder device as fallback
            if hasattr(self, "text_encoder_4"):
                te4_device = getattr(self.config.cache.devices, "text_encoder_4",
                                     self.config.cache.devices.text_encoder)
                self.text_encoder_4.to(te4_device)
            for attr in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4"):
                if hasattr(self, attr):
                    getattr(self, attr).requires_grad_(False).eval()
            if hasattr(self, "dit"):
                self.dit.cpu(); del self.dit
            gc.collect(); torch.cuda.empty_cache()

        elif stage == "predict":
            d = self.config.predict.devices
            for attr in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
                if hasattr(self, attr):
                    getattr(self, attr).to(d.text_encoder)
            if hasattr(self, "text_encoder_4"):
                te4_device = getattr(d, "text_encoder_4", d.text_encoder)
                self.text_encoder_4.to(te4_device)
            self.dit.to(d.dit)
            self.vae.to(d.vae)
            for attr in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4", "dit"):
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
        num_inference_steps: int = 50,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> dict:
        if isinstance(prompt, str):
            prompt = [prompt]
        self.weight_dtype = weight_dtype
        # Ensure divisible by vae_scale_factor * patch_size
        height, width = make_image_shape_devisible(height, width, self.vae_scale_factor * self._patch_size)
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
        """Denoising loop for HiDream-I1. Returns decoded image (B, 3, H, W) in [0, 1]."""
        num_steps = embeddings["num_inference_steps"]
        guidance_scale = embeddings.get("guidance_scale", 5.0)
        t5_embeds = embeddings["prompt_embeds_t5"]
        llama_embeds = embeddings["prompt_embeds_llama3"]
        pooled_embeds = embeddings["pooled_prompt_embeds"]
        height = embeddings["height"]
        width = embeddings["width"]
        batch_size = t5_embeds.shape[0]
        device = self.dit.device
        dtype = self.weight_dtype

        t5_embeds = t5_embeds.to(device, dtype=dtype)
        llama_embeds = llama_embeds.to(device, dtype=dtype)
        pooled_embeds = pooled_embeds.to(device, dtype=dtype)

        h_lat = height // self.vae_scale_factor
        w_lat = width // self.vae_scale_factor

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg_prompts = embeddings.get("negative_prompt", [""] * batch_size)
            neg_t5, neg_llama, neg_pooled = self.encode_prompt(neg_prompts)
            neg_t5 = neg_t5.to(device, dtype=dtype)
            neg_llama = neg_llama.to(device, dtype=dtype)
            neg_pooled = neg_pooled.to(device, dtype=dtype)

        if "latents" in embeddings:
            latents = embeddings["latents"].to(device, dtype=torch.float32)
        else:
            latents = randn_tensor(
                (batch_size, self.num_channels_latents, h_lat, w_lat),
                device=device, dtype=torch.float32,
            )

        # Dynamic shift for FlowMatch scheduler
        image_seq_len = (h_lat // self._patch_size) * (w_lat // self._patch_size)
        has_shift = (
            "base_image_seq_len" in self.sampling_scheduler.config
            and "max_image_seq_len" in self.sampling_scheduler.config
        )
        if has_shift:
            mu = _calculate_shift(
                image_seq_len,
                self.sampling_scheduler.config.get("base_image_seq_len", 256),
                self.sampling_scheduler.config.get("max_image_seq_len", 4096),
            )
            sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
            self.sampling_scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        else:
            self.sampling_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.sampling_scheduler.timesteps

        with torch.inference_mode():
            for t in tqdm(timesteps, desc="HiDream-I1 generating"):
                timestep = t.expand(batch_size).to(dtype)

                model_out = self.dit(
                    hidden_states=latents.to(dtype),
                    timesteps=timestep,
                    encoder_hidden_states_t5=t5_embeds,
                    encoder_hidden_states_llama3=llama_embeds,
                    pooled_embeds=pooled_embeds,
                    return_dict=False,
                )[0]
                noise_pred = -model_out.to(torch.float32)  # negate

                if do_cfg:
                    neg_out = self.dit(
                        hidden_states=latents.to(dtype),
                        timesteps=timestep,
                        encoder_hidden_states_t5=neg_t5,
                        encoder_hidden_states_llama3=neg_llama,
                        pooled_embeds=neg_pooled,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond = -neg_out.to(torch.float32)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                latents = self.sampling_scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

        return self.decode_vae_latent(latents)
