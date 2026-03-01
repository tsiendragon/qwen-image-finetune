"""Trainer for briaai/FIBO (text-to-image, flow matching).

Architecture overview:
  - Transformer: BriaFiboTransformer2DModel
  - VAE: AutoencoderKLWan (5D causal VAE, vae_scale_factor=16)
  - Text encoder: SmolLM3ForCausalLM + AutoTokenizer, max_length=256 (configurable)
    - encoder_hidden_states: cat(hidden_states[-1], hidden_states[-2], dim=-1) → (B, seq, 4096)
    - text_encoder_layers: ALL hidden states (tuple) passed per-layer to transformer
    - attention_mask: (B, seq) binary → converted to (B, 1, seq, seq) matrix at compute time
  - Scheduler: FlowMatchEulerDiscreteScheduler

Latent format:
  - VAE: 5D (B, C, 1, H, W) for encode/decode (T=1 temporal frame)
  - After encode: squeeze to 4D (B, C, H, W), apply mean/std normalization
  - Packed to sequence: (B, C, H, W) → (B, H*W, C) via permute+reshape (no 2×2 patch packing)
  - num_channels_latents = transformer.config.in_channels
  - vae_scale_factor = 16

Latent normalization:
  - Encode: normalized = (raw_latent - latents_mean) * latents_std
  - Decode: raw_latent = normalized / latents_std + latents_mean
  - latents_mean, latents_std: per-channel from vae.config (shape (C,))

Transformer call:
  - hidden_states: (B, H*W, C) packed latents
  - encoder_hidden_states: (B, seq, 4096) — last 2 layers concat
  - text_encoder_layers: list of (B, seq, 2048) per-layer hidden states
  - txt_ids: (B, seq, 3) zeros
  - img_ids: (H*W, 3) with [0, row, col]
  - timestep: (B,) sigma values
  - joint_attention_kwargs: {"attention_mask": (B, 1, seq, seq) matrix}

Position IDs:
  - txt_ids: (B, max_tokens, 3) zeros
  - img_ids: (H*W, 3), [0, row, col]

Note: text_encoder_layers is NOT cached (too large to store per-sample);
the text encoder is re-run at training time even when cache is enabled.
Only image_latents, prompt_embeds, and prompt_embeds_mask are cached.
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

TOKENIZER_MAX_LENGTH = 256  # practical default; SmolLM3 supports up to 3000


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                     base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class BriaFiboT2ITrainer(BaseTrainer):
    """Trainer for Bria FIBO text-to-image model.

    Uses a SmolLM3 text encoder with a BriaFiboTransformer2DModel. Latents are
    packed as sequences (no 2×2 patch packing). The transformer receives per-layer
    text encoder outputs for fine-grained text conditioning.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        from diffusers.pipelines.bria_fibo.pipeline_bria_fibo import BriaFiboPipeline
        return BriaFiboPipeline

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs):
        from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan
        from diffusers.models.transformers.transformer_bria_fibo import BriaFiboTransformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading BriaFibo T2I components...")
        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings or {}

        # ----- VAE (AutoencoderKLWan) -----
        vae_path = pretrains.get("vae", model_path)
        self.vae: AutoencoderKLWan = AutoencoderKLWan.from_pretrained(
            vae_path, subfolder="vae", torch_dtype=self.weight_dtype
        )

        # ----- Text encoder: SmolLM3 -----
        te_path = pretrains.get("text_encoder", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(te_path, subfolder="tokenizer")
        # SmolLM3ForCausalLM loaded via AutoModelForCausalLM
        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            te_path, subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
            output_hidden_states=True,
        )

        # ----- Transformer -----
        self.dit: BriaFiboTransformer2DModel = BriaFiboTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        )

        # ----- Scheduler -----
        self.scheduler: FlowMatchEulerDiscreteScheduler = (
            FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        )
        self.sampling_scheduler = copy.deepcopy(self.scheduler)

        # ----- Derived parameters -----
        self.vae_scale_factor: int = 16  # AutoencoderKLWan always 16
        self.num_channels_latents: int = self.dit.config.in_channels

        # Per-channel mean/std from VAE config for latent normalization
        self._latents_mean = torch.tensor(self.vae.config.latents_mean, dtype=torch.float32)
        self._latents_std = torch.tensor(self.vae.config.latents_std, dtype=torch.float32)

        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logger.info(
            f"BriaFibo T2I loaded. vae_scale={self.vae_scale_factor}, "
            f"latent_channels={self.num_channels_latents}"
        )

    # ------------------------------------------------------------------ #
    # Latent normalization helpers                                         #
    # ------------------------------------------------------------------ #

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply per-channel normalization: (latent - mean) / std.

        Matches pipeline convention: latents_std_var = 1/config.latents_std,
        so (latent - mean) * (1/std) == (latent - mean) / std.

        latents: (B, C, H, W) — 4D VAE-scale latents.
        """
        mean = self._latents_mean.to(latents.device, dtype=latents.dtype)
        std = self._latents_std.to(latents.device, dtype=latents.dtype)
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)
        return (latents - mean) / std

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Undo per-channel normalization: latent * std + mean."""
        mean = self._latents_mean.to(latents.device, dtype=latents.dtype)
        std = self._latents_std.to(latents.device, dtype=latents.dtype)
        mean = mean.reshape(1, -1, 1, 1)
        std = std.reshape(1, -1, 1, 1)
        return latents * std + mean

    # ------------------------------------------------------------------ #
    # Packing / unpacking helpers                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H*W, C) — no 2×2 patch packing."""
        B, C, H, W = latents.shape
        return latents.permute(0, 2, 3, 1).reshape(B, H * W, C)

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """(B, H*W, C) → (B, C, H, W)."""
        B, N, C = latents.shape
        return latents.reshape(B, H, W, C).permute(0, 3, 1, 2)

    @staticmethod
    def _prepare_image_ids(H: int, W: int, device, dtype) -> torch.Tensor:
        """Build img_ids (H*W, 3) with values [0, row, col]."""
        ids = torch.zeros(H, W, 3)
        ids[..., 1] = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        ids[..., 2] = torch.arange(W, dtype=torch.float32).unsqueeze(0)
        return ids.reshape(H * W, 3).to(device=device, dtype=dtype)

    @staticmethod
    def _prepare_text_ids(B: int, seq_len: int, device, dtype) -> torch.Tensor:
        """Build txt_ids (B, seq_len, 3) all zeros."""
        return torch.zeros(B, seq_len, 3, device=device, dtype=dtype)

    @staticmethod
    def _build_attention_matrix(attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert padding mask (B, seq_len) → (B, 1, seq_len, seq_len).

        Output: 0 where both tokens are real, -inf otherwise (for attention logits).
        """
        mask = attention_mask.float()
        matrix = torch.einsum("bi,bj->bij", mask, mask)
        matrix = torch.where(matrix == 1.0, 0.0, float("-inf"))
        return matrix.unsqueeze(1)  # (B, 1, seq, seq)

    # ------------------------------------------------------------------ #
    # Prompt encoding                                                      #
    # ------------------------------------------------------------------ #

    def encode_prompt(
        self,
        prompt: str | list[str],
        max_length: int = TOKENIZER_MAX_LENGTH,
        **kwargs,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """Encode prompt with SmolLM3 text encoder.

        Returns:
            encoder_hidden_states: (B, seq, 4096) — last 2 layers concatenated
            text_encoder_layers:   list of (B, seq, 2048) per-layer hidden states
            attention_mask:        (B, seq) binary mask (1=real, 0=pad)
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
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        all_hidden = outputs.hidden_states  # tuple of (B, seq, 2048) per layer
        # encoder_hidden_states = concat last 2 hidden layers → (B, seq, 4096)
        encoder_hs = torch.cat(
            [all_hidden[-1], all_hidden[-2]], dim=-1
        ).to(dtype=self.weight_dtype)
        # Per-layer hidden states for transformer
        layers = [h.to(dtype=self.weight_dtype) for h in all_hidden]

        return encoder_hs, layers, attention_mask.bool()

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
        """Encode (B, 3, H, W) → (B, C, H, W) normalized latents.

        AutoencoderKLWan expects 5D input (B, C, T, H, W); T=1 here.
        """
        with torch.inference_mode():
            latents = self.vae.encode(image.unsqueeze(2)).latent_dist.mean
        latents = latents.squeeze(2)  # (B, C, H, W)
        return self._normalize_latents(latents.float()).to(self.weight_dtype)

    def decode_vae_latent(self, latents: torch.Tensor, target_height: int = 0, target_width: int = 0) -> torch.Tensor:
        """Decode (B, H*W, C) packed latents → RGB (B, 3, H, W) in [0, 1].

        Caller must supply H, W as keyword args or embedded in embeddings dict.
        """
        h_lat = target_height // self.vae_scale_factor if target_height else None
        w_lat = target_width // self.vae_scale_factor if target_width else None
        if h_lat is None or w_lat is None:
            # Try to infer from latents shape
            N = latents.shape[1]
            side = int(N ** 0.5)
            h_lat = w_lat = side

        latents_4d = self._unpack_latents(latents, h_lat, w_lat)
        latents_4d = self._denormalize_latents(latents_4d.float()).to(self.vae.dtype)
        latents_5d = latents_4d.unsqueeze(2)  # (B, C, 1, H, W)
        with torch.inference_mode():
            image = self.vae.decode(latents_5d).sample
        image = image.squeeze(2)  # (B, 3, H, W)
        return (image / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------ #
    # Embedding preparation                                                #
    # ------------------------------------------------------------------ #

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
        device = next(self.text_encoder.parameters()).device

        # text_encoder_layers is NOT stored in batch (too large to cache)
        # It will be recomputed at training time or stored inline for predict
        pe, layers, attn_mask = self.encode_prompt(batch["prompt"])
        batch["prompt_embeds"] = pe
        batch["prompt_embeds_mask"] = attn_mask
        batch["text_encoder_layers"] = layers  # available inline (not cached)

        if stage == "cache":
            empty_pe, _, empty_mask = self.encode_prompt([""] * len(batch["prompt"]))
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
                batch["image_latents"] = self._vae_encode(image)

        return batch

    def prepare_cached_embeddings(self, batch: dict) -> dict:
        """Re-run text encoder to get per-layer hidden states (not cached)."""
        device = next(self.text_encoder.parameters()).device if hasattr(self, "text_encoder") else None
        if device is None:
            return batch

        _, layers, _ = self.encode_prompt(batch["prompt"])
        batch["text_encoder_layers"] = layers
        return batch

    # ------------------------------------------------------------------ #
    # Training loss                                                        #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        assert self.accelerator is not None
        device = self.accelerator.device

        image_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        attn_mask = embeddings["prompt_embeds_mask"].to(device)
        B, C, H, W = image_latents.shape

        # text_encoder_layers must be available (re-run if missing from cache)
        layers = embeddings.get("text_encoder_layers")
        if layers is None:
            _, layers, _ = self.encode_prompt(embeddings["prompt"])
        layers = [l.to(self.weight_dtype).to(device) for l in layers]

        with torch.no_grad():
            noise = torch.randn_like(image_latents)
            sigma = torch.rand(B, device=device, dtype=self.weight_dtype)
            s = sigma[:, None, None, None]
            noisy = (1.0 - s) * image_latents + s * noise
            target = noise - image_latents  # standard velocity

        # Pack latents to sequence
        noisy_packed = self._pack_latents(noisy)

        img_ids = self._prepare_image_ids(H, W, device=device, dtype=self.weight_dtype)
        txt_ids = self._prepare_text_ids(B, prompt_embeds.shape[1], device=device, dtype=self.weight_dtype)
        attn_matrix = self._build_attention_matrix(attn_mask)

        model_pred = self.dit(
            hidden_states=noisy_packed,
            timestep=sigma,
            encoder_hidden_states=prompt_embeds,
            text_encoder_layers=layers,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs={"attention_mask": attn_matrix},
            return_dict=False,
        )[0]

        # Unpack prediction and compare to target
        model_pred_4d = self._unpack_latents(model_pred, H, W)
        return self.forward_loss(model_pred_4d, target)

    # ------------------------------------------------------------------ #
    # Cache step                                                           #
    # ------------------------------------------------------------------ #

    def cache_step(self, data: dict):
        """Cache image_latents + prompt_embeds + mask. Skips text_encoder_layers."""
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
                # VAE not needed; text encoder must stay for text_encoder_layers
                if hasattr(self, "vae"):
                    self.vae.cpu()
                if not self.config.validation.enabled and hasattr(self, "vae"):
                    del self.vae
                gc.collect(); torch.cuda.empty_cache()
                # Keep text_encoder on device (needed for text_encoder_layers)
                if hasattr(self, "text_encoder"):
                    self.text_encoder.to(self.accelerator.device)
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
        guidance_scale: float = 3.5,
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
        """Denoising loop for Bria FIBO. Returns decoded image (B, 3, H, W) in [0, 1]."""
        num_steps = embeddings["num_inference_steps"]
        guidance_scale = embeddings.get("guidance_scale", 3.5)
        prompt_embeds = embeddings["prompt_embeds"]
        attn_mask = embeddings.get("prompt_embeds_mask")
        height = embeddings["height"]
        width = embeddings["width"]
        batch_size = prompt_embeds.shape[0]
        device = self.dit.device
        dtype = self.weight_dtype

        prompt_embeds = prompt_embeds.to(device, dtype=dtype)
        text_layers = embeddings.get("text_encoder_layers")
        if text_layers is None:
            _, text_layers, _ = self.encode_prompt(embeddings["prompt"])
        text_layers = [l.to(device, dtype=dtype) for l in text_layers]

        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
            attn_matrix = self._build_attention_matrix(attn_mask).to(dtype=dtype)
        else:
            attn_matrix = None

        h_lat = height // self.vae_scale_factor
        w_lat = width // self.vae_scale_factor

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg_prompts = embeddings.get("negative_prompt", [""] * batch_size)
            neg_embeds, neg_layers, neg_attn = self.encode_prompt(neg_prompts)
            neg_embeds = neg_embeds.to(device, dtype=dtype)
            neg_layers = [l.to(device, dtype=dtype) for l in neg_layers]
            neg_attn_matrix = (
                self._build_attention_matrix(neg_attn.to(device)).to(dtype=dtype)
                if neg_attn is not None else None
            )

        if "latents" in embeddings:
            latents = embeddings["latents"].to(device, dtype=torch.float32)
        else:
            latents = randn_tensor(
                (batch_size, self.num_channels_latents, h_lat, w_lat),
                device=device, dtype=torch.float32,
            )

        # Dynamic shift based on image sequence length
        image_seq_len = h_lat * w_lat
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

        img_ids = self._prepare_image_ids(h_lat, w_lat, device=device, dtype=dtype)

        with torch.inference_mode():
            for t in tqdm(timesteps, desc="Bria FIBO generating"):
                latents_packed = self._pack_latents(latents.to(dtype))
                txt_ids = self._prepare_text_ids(batch_size, prompt_embeds.shape[1], device=device, dtype=dtype)
                timestep = t.expand(batch_size).to(dtype)

                noise_pred = self.dit(
                    hidden_states=latents_packed,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    text_encoder_layers=text_layers,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs={"attention_mask": attn_matrix},
                    return_dict=False,
                )[0]
                noise_pred_4d = self._unpack_latents(noise_pred, h_lat, w_lat)

                if do_cfg:
                    neg_txt_ids = self._prepare_text_ids(batch_size, neg_embeds.shape[1], device=device, dtype=dtype)
                    neg_pred = self.dit(
                        hidden_states=latents_packed,
                        timestep=timestep,
                        encoder_hidden_states=neg_embeds,
                        text_encoder_layers=neg_layers,
                        txt_ids=neg_txt_ids,
                        img_ids=img_ids,
                        joint_attention_kwargs={"attention_mask": neg_attn_matrix},
                        return_dict=False,
                    )[0]
                    neg_pred_4d = self._unpack_latents(neg_pred, h_lat, w_lat)
                    noise_pred_4d = neg_pred_4d + guidance_scale * (noise_pred_4d - neg_pred_4d)

                latents = self.sampling_scheduler.step(
                    noise_pred_4d.to(torch.float32), t, latents, return_dict=False
                )[0]

        return self.decode_vae_latent(
            self._pack_latents(latents.to(dtype)), target_height=height, target_width=width
        )
