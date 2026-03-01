"""Trainer for Tongyi-MAI/Z-Image-Turbo (text-to-image, S3-DiT architecture).

Key architecture differences from Qwen/FLUX trainers:
  - Transformer API: takes list[Tensor(C, 1, H, W)] per sample (not batched tensor)
  - Prompt embeddings: variable-length per sample (chat template + hidden_states[-2] masking)
  - Timestep convention: (1000 - t) / 1000 (reversed — 0 = noisy, 1 = clean)
  - Model output negation: noise_pred = -model_output (inference); target = image_latents - noise (training)
  - VAE: standard AutoencoderKL with scaling_factor + shift_factor
  - Latents: 4D (B, C, H, W), no packing, no temporal dim beyond the extra 1 added for the transformer
  - vae_scale_factor: 2 ** (len(vae.config.block_out_channels) - 1)
"""

import copy
import gc
import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from qflux.trainer.base_trainer import BaseTrainer
from qflux.utils.images import make_image_shape_devisible


logger = logging.getLogger(__name__)


class ZImageT2ITrainer(BaseTrainer):
    """Trainer for Tongyi-MAI/Z-Image-Turbo text-to-image model.

    Implements flow-matching fine-tuning (LoRA) of the Z-Image S3-DiT.
    The denoising transformer takes per-sample lists of latent/prompt tensors.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline

        return ZImagePipeline

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs):
        """Load components from ZImagePipeline."""
        from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )

        logger.info("Loading ZImagePipeline and separating components...")
        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings or {}

        # Load full pipeline to grab scheduler/tokenizer, then extract components
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=self.weight_dtype,
            transformer=None,
            vae=None,
        )
        pipe.to("cpu")

        # ----- VAE -----
        vae_path = pretrains.get("vae", model_path)
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            subfolder="vae",
            torch_dtype=self.weight_dtype,
        )

        # ----- Text encoder + tokenizer -----
        te_path = pretrains.get("text_encoder", model_path)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(te_path, subfolder="tokenizer")
        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            te_path,
            subfolder="text_encoder",
            torch_dtype=self.weight_dtype,
        )

        # ----- Transformer -----
        from diffusers.models.transformers import ZImageTransformer2DModel

        self.dit = ZImageTransformer2DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            torch_dtype=self.weight_dtype,
        )

        # ----- Scheduler -----
        self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
        self.sampling_scheduler: FlowMatchEulerDiscreteScheduler = copy.deepcopy(self.scheduler)

        # ----- VAE parameters -----
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.latent_channels = self.vae.config.latent_channels
        self.num_channels_latents = self.latent_channels
        self.max_sequence_length = 512

        # Freeze non-trainable components
        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logger.info(
            f"Z-Image components loaded. VAE scale factor: {self.vae_scale_factor}, "
            f"latent channels: {self.latent_channels}"
        )

    # ------------------------------------------------------------------ #
    # Prompt encoding                                                      #
    # ------------------------------------------------------------------ #

    def _raw_encode_prompt(
        self,
        prompt: list[str],
        device: torch.device,
        max_sequence_length: int = 512,
    ) -> list[torch.FloatTensor]:
        """Encode a batch of prompts → list of variable-length tensors.

        Mirrors ZImagePipeline._encode_prompt() exactly.
        """
        # Apply chat template
        formatted = []
        for p in prompt:
            messages = [{"role": "user", "content": p}]
            formatted.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            )

        text_inputs = self.tokenizer(
            formatted,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            hidden = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                output_hidden_states=True,
            ).hidden_states[-2]  # (B, max_seq_len, hidden_dim)

        masks = text_inputs.attention_mask.bool()  # (B, max_seq_len)
        return [hidden[i][masks[i]] for i in range(hidden.size(0))]

    def encode_prompt(
        self,
        prompt: str | list[str],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts and return padded (embeds, mask) suitable for batching/caching.

        Returns:
            prompt_embeds: (B, max_seq_len, hidden_dim)
            prompt_embeds_mask: (B, max_seq_len) bool
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = next(self.text_encoder.parameters()).device
        embeds_list = self._raw_encode_prompt(prompt, device=device)

        max_len = max(e.size(0) for e in embeds_list)
        hidden_dim = embeds_list[0].size(1)
        batch_size = len(embeds_list)

        padded = torch.zeros(batch_size, max_len, hidden_dim, dtype=self.weight_dtype, device=device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        for i, e in enumerate(embeds_list):
            padded[i, : e.size(0)] = e
            mask[i, : e.size(0)] = True

        return padded, mask

    def _pad_prompt_embeds(
        self, embeds_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of variable-length tensors to a single batched tensor."""
        max_len = max(e.size(0) for e in embeds_list)
        hidden_dim = embeds_list[0].size(-1)
        batch_size = len(embeds_list)
        device = embeds_list[0].device

        padded = torch.zeros(batch_size, max_len, hidden_dim, dtype=embeds_list[0].dtype, device=device)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        for i, e in enumerate(embeds_list):
            padded[i, : e.size(0)] = e
            mask[i, : e.size(0)] = True
        return padded, mask

    def _unpad_prompt_embeds(
        self, padded: torch.Tensor, mask: torch.Tensor
    ) -> list[torch.Tensor]:
        """Convert padded batch tensor back to list of variable-length tensors."""
        return [padded[i][mask[i]] for i in range(padded.size(0))]

    # ------------------------------------------------------------------ #
    # VAE                                                                  #
    # ------------------------------------------------------------------ #

    def _vae_encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """VAE-encode a preprocessed image tensor.

        Args:
            image: (B, 3, H, W) in [-1, 1]
        Returns:
            latents: (B, C, H_lat, W_lat)
        """
        with torch.inference_mode():
            latents = self.vae.encode(image).latent_dist.sample()
        # Apply Z-Image VAE scaling (inverse of decode: (lat / scale) + shift)
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents

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

        Returns:
            noise_latents: (B, C, H_lat, W_lat)
            image_latents: (B, C, H_lat, W_lat) or None
        """
        h_lat = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_lat = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, h_lat, w_lat)
        device = next(self.vae.parameters()).device

        noise = randn_tensor(shape, device=device, dtype=dtype)

        image_latents = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            image_latents = self._vae_encode_image(image)

        return noise, image_latents

    def decode_vae_latent(
        self, latents: torch.Tensor, target_height: int, target_width: int
    ) -> torch.Tensor:
        """Decode VAE latents to RGB images in [0, 1].

        Mirrors ZImagePipeline's VAE decode:
            image = vae.decode((latents / scaling_factor) + shift_factor)
        """
        latents = latents.to(self.vae.device, dtype=self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.inference_mode():
            image = self.vae.decode(latents, return_dict=False)[0]  # (B, 3, H, W) in [-1, 1]
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    # ------------------------------------------------------------------ #
    # Batch / embedding preparation                                        #
    # ------------------------------------------------------------------ #

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image tensor from [0, 1] to [-1, 1] and ensure (B, 3, H, W)."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        # images from dataset are typically [0, 1]; convert to [-1, 1]
        if image.max() <= 1.0 + 1e-6:
            image = image * 2.0 - 1.0
        return image

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
        """Prepare embeddings for training (fit) or caching (cache).

        Encodes prompts and (during fit/cache) VAE-encodes target images.
        """
        device = next(self.text_encoder.parameters()).device

        # ----- Prompt encoding -----
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(batch["prompt"])
        batch["prompt_embeds"] = prompt_embeds
        batch["prompt_embeds_mask"] = prompt_embeds_mask

        if stage == "cache":
            empty_prompt_embeds, empty_prompt_embeds_mask = self.encode_prompt([""])
            batch["empty_prompt_embeds"] = empty_prompt_embeds
            batch["empty_prompt_embeds_mask"] = empty_prompt_embeds_mask

        if "negative_prompt" in batch:
            neg_embeds, neg_mask = self.encode_prompt(batch["negative_prompt"])
            batch["negative_prompt_embeds"] = neg_embeds
            batch["negative_prompt_embeds_mask"] = neg_mask

        # ----- VAE encode target image -----
        if "image" in batch:
            image = batch["image"]
            if isinstance(image, torch.Tensor) and image.ndim == 5:
                # Some dataloaders add a temporal dim: (B, 3, 1, H, W) → (B, 3, H, W)
                image = image.squeeze(2)
            image = self._preprocess_image(image)
            image = image.to(device=device, dtype=self.weight_dtype)

            batch["height"] = image.shape[2]
            batch["width"] = image.shape[3]

            with torch.inference_mode():
                image_latents = self._vae_encode_image(image)
            batch["image_latents"] = image_latents

        return batch

    def prepare_cached_embeddings(self, batch: dict) -> dict:
        """Load cached embeddings from batch dict (no transformation needed)."""
        return batch

    # ------------------------------------------------------------------ #
    # Training loss                                                        #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """Flow-matching training loss for Z-Image.

        Z-Image transformer takes list[Tensor(C, 1, H, W)] per sample and
        produces list[Tensor(C, 1, H, W)]; output is NEGATED before denoising.
        Therefore the training target is (image_latents - noise) not (noise - image_latents).
        """
        assert self.accelerator is not None
        device = self.accelerator.device

        image_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        prompt_embeds_mask = embeddings["prompt_embeds_mask"].to(device)  # bool

        batch_size = image_latents.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(image_latents)

            # Sample sigma uniformly in [0, 1]
            sigma = torch.rand(batch_size, device=device, dtype=self.weight_dtype)
            sigma_4d = sigma[:, None, None, None]  # (B, 1, 1, 1) for broadcasting

            # Flow-matching noisy input: x_t = (1 - sigma) * x_0 + sigma * noise
            noisy_latents = (1.0 - sigma_4d) * image_latents + sigma_4d * noise

            # Z-Image reversed timestep convention: 0 = pure noise, 1 = clean
            model_timestep = 1.0 - sigma  # (B,)

            # Build per-sample prompt embed lists (strip padding via mask)
            prompt_list = self._unpad_prompt_embeds(prompt_embeds, prompt_embeds_mask)

        # Build per-sample latent list: (B, C, H, W) → list of (C, 1, H, W)
        # (matches ZImagePipeline: latents.unsqueeze(2).unbind(0))
        latent_list = list(noisy_latents.unsqueeze(2).unbind(0))

        # Transformer forward
        model_out_list = self.dit(
            latent_list,
            model_timestep,
            prompt_list,
            return_dict=False,
        )[0]

        # Stack outputs: list of (C, 1, H, W) → (B, C, 1, H, W) → (B, C, H, W)
        model_pred = torch.stack([o.float() for o in model_out_list], dim=0).squeeze(2)

        # Target: image_latents - noise  (because inference does noise_pred = -model_output)
        target = (image_latents - noise).float()

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
    # Device / model mode management                                       #
    # ------------------------------------------------------------------ #

    def setup_model_device_train_mode(self, stage: str = "fit", cache: bool = False):
        """Allocate models to appropriate devices for each stage."""
        if stage == "fit":
            assert hasattr(self, "accelerator"), "accelerator must be set before calling this"

            if self.cache_exist and self.use_cache:
                # Cache exists: only need transformer for training
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
                # No cache: need all components
                for attr in ("vae", "text_encoder", "dit"):
                    if hasattr(self, attr):
                        getattr(self, attr).to(self.accelerator.device)
                if hasattr(self, "vae"):
                    self.vae.requires_grad_(False).eval()
                if hasattr(self, "text_encoder"):
                    self.text_encoder.requires_grad_(False).eval()
                self.dit.train()
                for name, param in self.dit.named_parameters():
                    param.requires_grad = "lora" in name

        elif stage == "cache":
            # Cache stage: need VAE + text_encoder, no transformer
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
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.dit.requires_grad_(False).eval()
            logger.info(
                f"predict mode: vae={devices.vae}, text_encoder={devices.text_encoder}, dit={devices.dit}"
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
        guidance_scale: float = 0.0,
        num_inference_steps: int = 8,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> dict:
        """Prepare inference batch for Z-Image text-to-image generation."""
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
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            data["negative_prompt"] = negative_prompt

        return data

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """Z-Image denoising loop using the list-based transformer API.

        Supports optional classifier-free guidance (guidance_scale > 0).
        Returns final latents (B, C, H_lat, W_lat).
        """
        num_inference_steps = embeddings["num_inference_steps"]
        guidance_scale = embeddings.get("guidance_scale", 0.0)
        prompt_embeds = embeddings["prompt_embeds"]       # (B, max_len, hidden)
        prompt_embeds_mask = embeddings["prompt_embeds_mask"]  # (B, max_len) bool
        height = embeddings["height"]
        width = embeddings["width"]
        batch_size = prompt_embeds.shape[0]

        do_cfg = guidance_scale > 0
        if do_cfg:
            neg_embeds = embeddings.get("negative_prompt_embeds", prompt_embeds.new_zeros(*prompt_embeds.shape))
            neg_mask = embeddings.get("negative_prompt_embeds_mask", prompt_embeds_mask)

        device = self.dit.device

        # Prepare prompt list
        prompt_list = self._unpad_prompt_embeds(
            prompt_embeds.to(device, dtype=self.weight_dtype),
            prompt_embeds_mask.to(device),
        )
        if do_cfg:
            neg_prompt_list = self._unpad_prompt_embeds(
                neg_embeds.to(device, dtype=self.weight_dtype),
                neg_mask.to(device),
            )

        # Prepare initial latents
        h_lat = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_lat = 2 * (int(width) // (self.vae_scale_factor * 2))

        if "latents" in embeddings:
            latents = embeddings["latents"].to(device, dtype=torch.float32)
        else:
            latents = randn_tensor(
                (batch_size, self.num_channels_latents, h_lat, w_lat),
                device=device,
                dtype=torch.float32,
            )

        # Prepare timesteps with dynamic shift (matches ZImagePipeline)
        image_seq_len = (h_lat // 2) * (w_lat // 2)
        timesteps, num_inference_steps = self._prepare_zimage_timesteps(
            num_inference_steps, image_seq_len, device
        )

        with torch.inference_mode():
            for t in tqdm(timesteps, total=len(timesteps), desc="Z-Image generating"):
                timestep = t.expand(batch_size)
                model_timestep = ((1000 - timestep) / 1000).to(dtype=self.weight_dtype)

                latents_input = latents.to(self.dit.dtype)
                latent_list = list(latents_input.unsqueeze(2).unbind(0))

                if do_cfg:
                    # Positive + negative in one forward pass (doubled batch)
                    combined_latent_list = latent_list + latent_list
                    combined_prompt_list = prompt_list + neg_prompt_list
                    combined_timestep = model_timestep.repeat(2)

                    out_list = self.dit(
                        combined_latent_list,
                        combined_timestep,
                        combined_prompt_list,
                        return_dict=False,
                    )[0]

                    pos_out = out_list[:batch_size]
                    neg_out = out_list[batch_size:]

                    noise_pred = []
                    for j in range(batch_size):
                        pred = pos_out[j].float() + guidance_scale * (pos_out[j].float() - neg_out[j].float())
                        noise_pred.append(pred)
                    noise_pred = torch.stack(noise_pred, dim=0).squeeze(2)
                else:
                    out_list = self.dit(
                        latent_list,
                        model_timestep,
                        prompt_list,
                        return_dict=False,
                    )[0]
                    noise_pred = torch.stack([o.float() for o in out_list], dim=0).squeeze(2)

                # Z-Image convention: negate model output before scheduler step
                noise_pred = -noise_pred

                latents = self.sampling_scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]

        return latents

    def _prepare_zimage_timesteps(
        self, num_inference_steps: int, image_seq_len: int, device: torch.device
    ) -> tuple[torch.Tensor, int]:
        """Set up timesteps with dynamic shift (mirrors ZImagePipeline)."""
        from qflux.utils.sampling import calculate_shift, retrieve_timesteps

        mu = calculate_shift(
            image_seq_len,
            self.sampling_scheduler.config.get("base_image_seq_len", 256),
            self.sampling_scheduler.config.get("max_image_seq_len", 4096),
            self.sampling_scheduler.config.get("base_shift", 0.5),
            self.sampling_scheduler.config.get("max_shift", 1.15),
        )
        self.sampling_scheduler.sigma_min = 0.0
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        timesteps, num_steps = retrieve_timesteps(
            self.sampling_scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_steps
