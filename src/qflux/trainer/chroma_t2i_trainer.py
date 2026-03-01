"""Trainer for lodestones/Chroma1-HD (text-to-image, FLUX-like MMDiT).

Architecture overview:
  - Transformer: ChromaTransformer2DModel (FLUX-like MMDiT with packed 2×2 patch latents)
  - VAE: AutoencoderKL, vae_scale_factor = 2^(len(block_out_channels)-1) ≈ 8
  - Text encoder: T5EncoderModel + T5TokenizerFast (google/t5-v1_1-xxl, max_length=512)
  - Scheduler: FlowMatchEulerDiscreteScheduler

Key differences from OvisImage/LongCat:
  - T5 text encoder (not Qwen)
  - attention_mask passed to transformer (T5 padding mask)
  - text_ids: all zeros (B=0, no sequential offset) — FLUX convention
  - img_ids: [0, row, col]

Latent format:
  - Unpacked: (B, C, H//8, W//8)  ← stored in cache, used for VAE ops
  - Packed:   (B, (H//16)*(W//16), C*4) ← passed to transformer
  - num_channels_latents = transformer.config.in_channels // 4

Timestep:
  - Training: sigma ~ Uniform(0, 1) passed directly to transformer
  - Inference: t from scheduler → t / 1000 for transformer

VAE decode:
  - Unpack, then: (latents / scaling_factor) + shift_factor → vae.decode()
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

MAX_SEQUENCE_LENGTH = 512  # T5 max tokens


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                     base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class ChromaT2ITrainer(BaseTrainer):
    """Trainer for Chroma text-to-image model.

    Uses a T5 text encoder (google/t5-v1_1-xxl) with a packed FLUX-like transformer.
    Latents are packed into 2×2 patches before being passed to the transformer.
    An attention_mask from the tokenizer is passed to the transformer.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        from diffusers.pipelines.chroma.pipeline_chroma import ChromaPipeline
        return ChromaPipeline

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs):
        from diffusers.models import AutoencoderKL, ChromaTransformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import T5EncoderModel, T5TokenizerFast

        logger.info("Loading ChromaPipeline components...")
        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings or {}

        # ----- VAE -----
        vae_path = pretrains.get("vae", model_path)
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            vae_path, subfolder="vae", torch_dtype=self.weight_dtype
        )

        # ----- Text encoder: T5 -----
        te_path = pretrains.get("text_encoder", model_path)
        self.tokenizer = T5TokenizerFast.from_pretrained(te_path, subfolder="tokenizer")
        self.text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(
            te_path, subfolder="text_encoder", torch_dtype=self.weight_dtype
        )

        # ----- Transformer -----
        self.dit: ChromaTransformer2DModel = ChromaTransformer2DModel.from_pretrained(
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
        # Packed latents: transformer in_channels = num_channels_latents * 4
        self.num_channels_latents: int = self.dit.config.in_channels // 4

        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logger.info(
            f"Chroma loaded. vae_scale={self.vae_scale_factor}, "
            f"latent_channels={self.num_channels_latents}, "
            f"transformer in_channels={self.dit.config.in_channels}"
        )

    # ------------------------------------------------------------------ #
    # Packed latent helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, (H//2)*(W//2), C*4)."""
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
        """(B, N, C*4) → (B, C, H, W)."""
        H = 2 * (int(height) // (vae_scale_factor * 2))
        W = 2 * (int(width) // (vae_scale_factor * 2))
        B, N, C4 = latents.shape
        latents = latents.view(B, H // 2, W // 2, C4 // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(B, C4 // 4, H, W)
        return latents

    @staticmethod
    def _prepare_latent_image_ids(h_patch: int, w_patch: int, device, dtype) -> torch.Tensor:
        """Build img_ids (h_patch*w_patch, 3): [0, row, col] — FLUX convention."""
        ids = torch.zeros(h_patch, w_patch, 3)
        ids[..., 1] += torch.arange(h_patch)[:, None].float()
        ids[..., 2] += torch.arange(w_patch)[None, :].float()
        return ids.reshape(h_patch * w_patch, 3).to(device=device, dtype=dtype)

    @staticmethod
    def _prepare_text_ids(seq_len: int, device, dtype) -> torch.Tensor:
        """Build text_ids (seq_len, 3): all zeros — FLUX/Chroma convention."""
        return torch.zeros(seq_len, 3, device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Prompt encoding                                                      #
    # ------------------------------------------------------------------ #

    def encode_prompt(
        self,
        prompt: str | list[str],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompt with T5 text encoder.

        Returns:
            prompt_embeds:      (B, MAX_SEQUENCE_LENGTH, hidden_dim)
            prompt_attn_mask:   (B, MAX_SEQUENCE_LENGTH) — bool, 1 = real token
            text_ids:           (MAX_SEQUENCE_LENGTH, 3) — all zeros
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = next(self.text_encoder.parameters()).device

        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)

        with torch.inference_mode():
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        prompt_embeds = outputs.last_hidden_state.to(dtype=self.weight_dtype, device=device)

        text_ids = self._prepare_text_ids(prompt_embeds.shape[1], device=device, dtype=self.weight_dtype)
        return prompt_embeds, attention_mask.bool(), text_ids

    def prepare_latents(
        self,
        image: torch.Tensor | None,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare noise latents and (optionally) VAE-encoded image latents.

        Dimensions are kept divisible by 2 to support 2×2 packing.
        """
        h_lat = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_lat = 2 * (int(width) // (self.vae_scale_factor * 2))
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
        """Encode (B, 3, H, W) → (B, C, H_lat, W_lat) unpacked latents."""
        with torch.inference_mode():
            return self.vae.encode(image).latent_dist.sample()

    def decode_vae_latent(self, latents: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Decode packed latents → RGB (B, 3, H, W) in [0, 1]."""
        latents = self._unpack_latents(latents, target_height, target_width, self.vae_scale_factor)
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

        pe, attn_mask, text_ids = self.encode_prompt(batch["prompt"])
        batch["prompt_embeds"] = pe
        batch["prompt_embeds_mask"] = attn_mask
        batch["text_ids"] = text_ids

        if stage == "cache":
            empty_pe, empty_mask, empty_text_ids = self.encode_prompt([""] * len(batch["prompt"]))
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
                batch["image_latents"] = self._vae_encode(image)  # unpacked

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
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        prompt_mask = embeddings["prompt_embeds_mask"].to(device)
        B, C, H, W = image_latents.shape

        with torch.no_grad():
            noise = torch.randn_like(image_latents)
            sigma = torch.rand(B, device=device, dtype=self.weight_dtype)
            s = sigma[:, None, None, None]
            noisy_latents = (1.0 - s) * image_latents + s * noise

        # Pack for transformer
        noisy_packed = self._pack_latents(noisy_latents)
        image_packed = self._pack_latents(image_latents)
        noise_packed = self._pack_latents(noise)

        h_patch, w_patch = H // 2, W // 2
        text_ids = self._prepare_text_ids(prompt_embeds.shape[1], device=device, dtype=self.weight_dtype)
        img_ids = self._prepare_latent_image_ids(h_patch, w_patch, device=device, dtype=self.weight_dtype)

        model_pred = self.dit(
            hidden_states=noisy_packed,
            timestep=sigma,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            attention_mask=prompt_mask,
            return_dict=False,
        )[0]

        target = noise_packed - image_packed
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
        guidance_scale: float = 0.0,
        num_inference_steps: int = 50,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> dict:
        if isinstance(prompt, str):
            prompt = [prompt]
        self.weight_dtype = weight_dtype
        height, width = make_image_shape_devisible(height, width, self.vae_scale_factor * 2)
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
        """Denoising loop for Chroma. Returns final packed latents (B, N, C*4)."""
        num_steps = embeddings["num_inference_steps"]
        guidance_scale = embeddings.get("guidance_scale", 0.0)
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

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg = embeddings.get("negative_prompt", [""] * batch_size)
            neg_embeds, neg_mask, neg_text_ids = self.encode_prompt(neg)
            neg_embeds = neg_embeds.to(device, dtype=dtype)
            if neg_mask is not None:
                neg_mask = neg_mask.to(device)

        h_lat = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_lat = 2 * (int(width) // (self.vae_scale_factor * 2))

        if "latents" in embeddings:
            raw = embeddings["latents"].to(device, dtype=torch.float32)
            latents = self._pack_latents(raw)
        else:
            raw = randn_tensor(
                (batch_size, self.num_channels_latents, h_lat, w_lat),
                device=device, dtype=torch.float32,
            )
            latents = self._pack_latents(raw)

        h_patch, w_patch = h_lat // 2, w_lat // 2
        text_ids = self._prepare_text_ids(prompt_embeds.shape[1], device=device, dtype=dtype)
        img_ids = self._prepare_latent_image_ids(h_patch, w_patch, device=device, dtype=dtype)

        # Use calculate_shift if scheduler has the relevant config keys
        image_seq_len = latents.shape[1]
        has_shift = (
            "base_image_seq_len" in self.sampling_scheduler.config
            and "max_image_seq_len" in self.sampling_scheduler.config
        )
        if has_shift:
            mu = _calculate_shift(
                image_seq_len,
                self.sampling_scheduler.config.get("base_image_seq_len", 256),
                self.sampling_scheduler.config.get("max_image_seq_len", 4096),
                self.sampling_scheduler.config.get("base_shift", 0.5),
                self.sampling_scheduler.config.get("max_shift", 1.15),
            )
            sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
            self.sampling_scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        else:
            self.sampling_scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.sampling_scheduler.timesteps
        self.sampling_scheduler.set_begin_index(0)

        with torch.inference_mode():
            for t in tqdm(timesteps, desc="Chroma generating"):
                timestep = t.expand(batch_size).to(dtype)

                noise_pred = self.dit(
                    hidden_states=latents.to(dtype),
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    attention_mask=prompt_mask,
                    return_dict=False,
                )[0]

                if do_cfg:
                    neg_text_ids = self._prepare_text_ids(neg_embeds.shape[1], device=device, dtype=dtype)
                    neg_pred = self.dit(
                        hidden_states=latents.to(dtype),
                        timestep=timestep / 1000,
                        encoder_hidden_states=neg_embeds,
                        txt_ids=neg_text_ids,
                        img_ids=img_ids,
                        attention_mask=neg_mask,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

                latents = self.sampling_scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]

        return latents  # packed (B, N, C*4)
