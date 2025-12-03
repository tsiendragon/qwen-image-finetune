"""
FLUX.2 LoRA Trainer Implementation
Supporting both Text-to-Image (T2I) and Image-to-Image (I2I) modes.

Key differences from FluxKontext:
- Single text encoder: Mistral3 (VLM) instead of CLIP + T5
- No pooled_prompt_embeds
- text_ids shape: [seq, 4] instead of [seq, 3]
- VAE with batch_norm architecture
- Uses compute_empirical_mu() for timestep shift
"""

import copy
import gc
import logging

import numpy as np
import PIL
import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from qflux.models.flux2_loader import (
    compute_empirical_mu,
    load_flux2_scheduler,
    load_flux2_text_encoder,
    load_flux2_tokenizer,
    load_flux2_transformer,
    load_flux2_vae,
)
from qflux.trainer.base_trainer import BaseTrainer
from qflux.utils.images import make_image_devisible, make_image_shape_devisible
from qflux.utils.sampling import retrieve_timesteps


logger = logging.getLogger(__name__)


class Flux2LoraTrainer(BaseTrainer):
    """
    FLUX.2 LoRA Trainer supporting both T2I and I2I modes.

    Inherits from BaseTrainer to ensure consistent interface.

    Modes:
    - T2I (Text-to-Image): Pure text-conditioned generation without reference images
    - I2I (Image-to-Image): Generation with optional condition/reference image

    The mode is automatically detected based on batch data:
    - If 'control' or 'condition_image' is present and not None -> I2I mode
    - Otherwise -> T2I mode
    """

    def __init__(self, config):
        """Initialize FLUX.2 trainer with configuration."""
        super().__init__(config)

        # FLUX.2 specific components (single text encoder unlike FluxKontext)
        self.vae = None  # AutoencoderKLFlux2
        self.text_encoder = None  # Mistral3ForConditionalGeneration (single VLM)
        self.tokenizer = None  # PixtralProcessor
        self.dit = None  # Flux2Transformer2DModel
        self.scheduler = None  # FlowMatchEulerDiscreteScheduler

        # Note: No text_encoder_2, tokenizer_2 (unlike FluxKontext)

        # VAE parameters
        self.vae_scale_factor = None
        self.latent_channels = None

        # FLUX.2 specific attributes
        self.num_channels_latents = None
        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False

        # System message for Mistral3 (matches original Flux2Pipeline exactly)
        # Note: The original has a newline between "object" and "attribution"
        self.system_message = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object
attribution and actions without speculation."""

    def get_pipeline_class(self):
        """Return Flux2Pipeline for LoRA saving."""
        from diffusers import Flux2Pipeline

        return Flux2Pipeline

    def load_model(self):
        """
        Load and separate components from Flux2Pipeline.
        Follows QwenImageEditTrainer/FluxKontextLoraTrainer patterns.
        """
        logging.info("Loading FLUX.2 components...")

        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings

        # Load VAE
        if pretrains is not None and "vae" in pretrains:
            self.vae = load_flux2_vae(pretrains["vae"], weight_dtype=self.weight_dtype).to("cpu")
            logging.info(f"Loaded VAE from {pretrains['vae']}")
        else:
            self.vae = load_flux2_vae(model_path, weight_dtype=self.weight_dtype).to("cpu")

        # Load text encoder (single Mistral3)
        if pretrains is not None and "text_encoder" in pretrains:
            self.text_encoder = load_flux2_text_encoder(pretrains["text_encoder"], weight_dtype=self.weight_dtype).to(
                "cpu"
            )
            logging.info(f"Loaded text encoder from {pretrains['text_encoder']}")
        else:
            self.text_encoder = load_flux2_text_encoder(model_path, weight_dtype=self.weight_dtype).to("cpu")

        # Load tokenizer
        self.tokenizer = load_flux2_tokenizer(model_path)

        # Load transformer
        self.dit = load_flux2_transformer(model_path, weight_dtype=self.weight_dtype).to("cpu")

        # Load scheduler (with copy for sampling)
        self.scheduler = load_flux2_scheduler(model_path)
        self.sampling_scheduler = copy.deepcopy(self.scheduler)

        # Set VAE parameters
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if self.vae else 8
        self.latent_channels = self.vae.config.latent_channels if self.vae else 16
        self.num_channels_latents = self.dit.config.in_channels // 4 if self.dit else 16

        # Set models to eval mode
        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()

        # Image processor
        from diffusers.image_processor import VaeImageProcessor

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        torch.cuda.empty_cache()
        logging.info(f"FLUX.2 components loaded. VAE scale factor: {self.vae_scale_factor}")

    def setup_model_device_train_mode(self, stage="fit", cache=False):
        """Set model device allocation and train mode."""
        if stage == "fit":
            assert hasattr(self, "accelerator"), "accelerator must be set before setting model devices"

        if self.cache_exist and self.use_cache and stage == "fit":
            # Cache mode with training: only need transformer
            self.text_encoder.cpu()
            self.text_encoder.requires_grad_(False).eval()
            torch.cuda.empty_cache()
            self.vae.cpu()
            torch.cuda.empty_cache()

            if not self.config.validation.enabled:
                del self.vae
                del self.text_encoder
            else:
                self.vae.requires_grad_(False).eval()

            gc.collect()
            self.dit.to(self.accelerator.device)
            self.dit.requires_grad_(False)
            self.dit.train()
            for name, param in self.dit.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif stage == "fit":
            # Non-cache mode: need all encoders
            self.vae.to(self.accelerator.device)
            self.text_encoder.to(self.accelerator.device)
            self.dit.to(self.accelerator.device)
            self.vae.decoder.to("cpu")

            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.dit.requires_grad_(False)
            self.dit.train()
            for name, param in self.dit.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif stage == "cache":
            # Cache mode: need encoders, don't need transformer
            self.vae = self.vae.to(self.config.cache.devices.vae, non_blocking=True)
            self.vae.decoder.to("cpu")
            self.text_encoder = self.text_encoder.to(self.config.cache.devices.text_encoder, non_blocking=True)

            torch.cuda.synchronize()
            self.dit.cpu()
            torch.cuda.empty_cache()
            del self.dit
            gc.collect()
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            logging.info("Cache mode device setting complete")

        elif stage == "predict":
            # Predict mode: allocate to different GPUs according to configuration
            devices = self.config.predict.devices
            self.vae.to(devices.vae)
            self.text_encoder.to(devices.text_encoder)
            self.dit.to(devices.dit)
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.dit.requires_grad_(False).eval()

    def _detect_mode(self, batch: dict) -> str:
        """
        Detect T2I or I2I mode from batch data.

        Returns:
            "t2i" if no condition image, "i2i" if condition image present
        """
        # Check for condition image (can be 'control' or 'condition_image')
        has_condition = False
        for key in ["control", "condition_image"]:
            if key in batch and batch[key] is not None:
                has_condition = True
                break
        return "i2i" if has_condition else "t2i"

    def _format_text_input(self, prompts: list[str], system_message: str | None = None):
        """
        Format prompts into conversation format for Mistral3.

        Args:
            prompts: List of text prompts
            system_message: System message for the conversation

        Returns:
            List of formatted conversation dictionaries
        """
        if system_message is None:
            system_message = self.system_message

        # Remove [IMG] tokens to avoid validation issues
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

        return [
            [
                {"role": "system", "content": [{"type": "text", "text": system_message}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]

    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        max_sequence_length: int = 512,
        hidden_states_layers: tuple[int, ...] = (10, 20, 30),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts using Mistral3.

        Unlike FluxKontext which uses CLIP+T5, FLUX.2 uses single Mistral3 VLM.

        Args:
            prompt: Text prompt or list of prompts
            device: Device to place embeddings on
            max_sequence_length: Maximum sequence length
            hidden_states_layers: Tuple of layer indices to extract hidden states from (default: (10, 20, 30))

        Returns:
            prompt_embeds: [B, seq, num_layers * hidden_dim]
            text_ids: [seq, 4]  # Note: 4 dims, not 3 like FluxKontext
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = device or self.text_encoder.device

        # Format prompts for Mistral3 (returns list of message lists)
        messages_batch = self._format_text_input(prompt)

        with torch.inference_mode():
            # Process all messages at once using apply_chat_template with tokenize=True
            # This matches the original Flux2Pipeline implementation
            inputs = self.tokenizer.apply_chat_template(
                messages_batch,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            # Move to device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            # Forward pass through the model
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            # Only use outputs from intermediate layers and stack them
            # This matches the original Flux2Pipeline._get_mistral_3_small_prompt_embeds implementation
            out = torch.stack([outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
            out = out.to(dtype=self.weight_dtype, device=device)

            batch_size, num_channels, seq_len, hidden_dim = out.shape
            prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

            # Create text IDs using _prepare_text_ids (matches original Flux2Pipeline)
            text_ids = self._prepare_text_ids(prompt_embeds)
            text_ids = text_ids.to(device)

        return prompt_embeds, text_ids

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor, t_coord: torch.Tensor | None = None) -> torch.Tensor:
        """
        Prepare text position IDs.
        Matches original Flux2Pipeline._prepare_text_ids implementation.

        Args:
            x: Prompt embeddings [B, L, D]
            t_coord: Optional time coordinates

        Returns:
            text_ids: [B, L, 4]
        """
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l_coords = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l_coords)
            out_ids.append(coords)

        return torch.stack(out_ids)

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
        Prepare latents for training/inference.
        Matches original Flux2Pipeline.prepare_latents implementation.

        Args:
            image: Input image tensor or None
            batch_size: Batch size
            num_channels_latents: Number of latent channels
            height: Image height
            width: Image width
            dtype: Data type

        Returns:
            latents: Random noise latents [B, H*W, C]
            image_latents: Encoded image latents (if image provided)
            latent_ids: Position IDs for target latents [B, H*W, 4]
            image_ids: Position IDs for image latents
        """
        # VAE applies 8x compression, account for packing
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        # Shape with patchify (*4) and half dimensions
        shape = (batch_size, num_channels_latents * 4, height // 2, width // 2)
        device = next(self.vae.parameters()).device

        image_latents = None
        image_ids = None

        if image is not None:
            image = image.to(device=device, dtype=dtype)
            with torch.inference_mode():
                image_latents = self._encode_vae_image(image)

            image_latent_height, image_latent_width = image_latents.shape[2:]
            # Use _prepare_latent_ids for image latents
            image_ids = self._prepare_latent_ids(image_latents)
            image_ids = image_ids.to(device)
            # Pack image latents
            image_latents = self._pack_latents_simple(image_latents)

        # Create random noise latents
        latents = randn_tensor(shape, device=device, dtype=dtype)

        # Prepare latent IDs before packing
        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(device)

        # Pack latents: [B, C, H, W] -> [B, H*W, C]
        latents = self._pack_latents_simple(latents)

        return latents, image_latents, latent_ids, image_ids

    @staticmethod
    def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
        """
        Generates 4D position coordinates (T, H, W, L) for latent tensors.
        Matches original Flux2Pipeline._prepare_latent_ids implementation.

        Args:
            latents: Latent tensor of shape (B, C, H, W)

        Returns:
            Position IDs tensor of shape (B, H*W, 4)
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l_dim = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l_dim)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _pack_latents_simple(latents: torch.Tensor) -> torch.Tensor:
        """
        Pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
        Matches original Flux2Pipeline._pack_latents implementation.
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    def _encode_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image through VAE with FLUX.2 specific processing.
        Matches original Flux2Pipeline._encode_vae_image implementation.
        """
        if image.ndim != 4:
            raise ValueError(f"Expected image dims 4, got {image.ndim}.")

        # Encode image
        image_latents = self.vae.encode(image).latent_dist.mode()

        # Patchify latents (FLUX.2 specific)
        image_latents = self._patchify_latents(image_latents)

        # Apply batch_norm normalization (FLUX.2 specific)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(image_latents.device, image_latents.dtype)
        latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        image_latents = (image_latents - latents_bn_mean) / latents_bn_std

        return image_latents

    @staticmethod
    def _prepare_image_ids(
        image_latents: list[torch.Tensor],  # [(1, C, H, W), (1, C, H, W), ...]
        scale: int = 10,
    ) -> torch.Tensor:
        """
        Generates 4D time-space coordinates (T, H, W, L) for a sequence of image latents.
        Matches original Flux2Pipeline._prepare_image_ids implementation.

        Args:
            image_latents: A list of image latent feature tensors, typically of shape (1, C, H, W).
            scale: A factor used to define the time separation (T-coordinate) between latents.

        Returns:
            The combined coordinate tensor. Shape: (1, N_total, 4)
        """
        if not isinstance(image_latents, list):
            raise ValueError(f"Expected `image_latents` to be a list, got {type(image_latents)}.")

        # create time offset for each reference image
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    def prepare_image_latents(
        self,
        images: list[torch.Tensor],
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare image latents for I2I mode.
        Matches original Flux2Pipeline.prepare_image_latents implementation.

        Args:
            images: List of preprocessed image tensors
            batch_size: Batch size
            generator: Random generator
            device: Device to use
            dtype: Data type

        Returns:
            Tuple of (image_latents, image_latent_ids)
        """
        image_latents = []
        for image in images:
            image = image.to(device=device, dtype=dtype)
            image_latent = self._encode_vae_image(image=image)
            image_latents.append(image_latent)  # (1, 128, H//16, W//16)

        image_latent_ids = self._prepare_image_ids(image_latents)

        # Pack each latent and concatenate
        packed_latents = []
        for latent in image_latents:
            # latent: (1, 128, H, W)
            packed = self._pack_latents_simple(latent)  # (1, H*W, 128)
            packed = packed.squeeze(0)  # (H*W, 128) - remove batch dim
            packed_latents.append(packed)

        # Concatenate all reference tokens along sequence dimension
        image_latents = torch.cat(packed_latents, dim=0)  # (N*H*W, 128)
        image_latents = image_latents.unsqueeze(0)  # (1, N*H*W, 128)

        image_latents = image_latents.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.repeat(batch_size, 1, 1)
        image_latent_ids = image_latent_ids.to(device)

        return image_latents, image_latent_ids

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        Patchify latents into 2x2 patches.
        Matches original Flux2Pipeline._patchify_latents implementation.

        Args:
            latents: [B, C, H, W]

        Returns:
            Patchified latents [B, C*4, H//2, W//2]
        """
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, num_channels_latents * 4, height // 2, width // 2)
        return latents

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
        """
        Prepare embeddings supporting both T2I and I2I modes.

        Args:
            batch: Input batch with image, prompt, and optional condition_image/control
            stage: "fit", "cache", or "predict"

        Returns:
            Updated batch with all embeddings
        """
        # Detect mode
        mode = self._detect_mode(batch)
        batch["mode"] = mode

        # === Common: Text encoding ===
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=batch["prompt"],
            max_sequence_length=512,
        )
        batch["prompt_embeds"] = prompt_embeds
        batch["text_ids"] = text_ids

        # Empty prompt for cache mode
        if stage == "cache":
            empty_embeds, _ = self.encode_prompt(prompt=[""])
            batch["empty_prompt_embeds"] = empty_embeds

        # === Target image encoding (always needed for training) ===
        if "image" in batch:
            image = self._preprocess_image(batch["image"])
            batch["image"] = image
            batch["height"] = image.shape[2]
            batch["width"] = image.shape[3]

            _, image_latents, latent_ids, _ = self.prepare_latents(
                image=image,
                batch_size=image.shape[0],
                num_channels_latents=self.num_channels_latents,
                height=batch["height"],
                width=batch["width"],
                dtype=self.weight_dtype,
            )
            batch["image_latents"] = image_latents
            batch["latent_ids"] = latent_ids

        # === Condition image encoding (I2I mode only) ===
        condition_key = "control" if "control" in batch else "condition_image"
        if mode == "i2i" and condition_key in batch and batch[condition_key] is not None:
            condition = self._preprocess_image(batch[condition_key])
            batch[condition_key] = condition
            batch["condition_height"] = condition.shape[2]
            batch["condition_width"] = condition.shape[3]

            _, condition_latents, _, condition_ids = self.prepare_latents(
                image=condition,
                batch_size=condition.shape[0],
                num_channels_latents=self.num_channels_latents,
                height=batch["condition_height"],
                width=batch["condition_width"],
                dtype=self.weight_dtype,
            )
            # Mark condition latents with domain ID = 10 (matches Flux2Pipeline._prepare_image_ids with scale=10)
            if condition_ids is not None:
                condition_ids[..., 0] = 10
            batch["condition_latents"] = condition_latents
            batch["condition_ids"] = condition_ids
        else:
            batch["condition_latents"] = None
            batch["condition_ids"] = None

        # === Negative prompt (predict mode) ===
        if stage == "predict" and "negative_prompt" in batch and batch.get("true_cfg_scale", 1.0) > 1.0:
            neg_embeds, neg_ids = self.encode_prompt(batch["negative_prompt"])
            batch["negative_prompt_embeds"] = neg_embeds
            batch["negative_text_ids"] = neg_ids

        return batch

    def _preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image to [-1, 1] range.

        Handles images in various ranges:
        - [0, 255]: normalize to [0, 1] then scale to [-1, 1]
        - [0, 1]: scale to [-1, 1]
        - [-1, 1]: return as-is (already preprocessed by VaeImageProcessor)
        """
        # Already in [-1, 1] range (from VaeImageProcessor)
        if image.min() < 0:
            return image
        # [0, 255] range
        if image.max() > 1.0:
            image = image / 255.0
        # [0, 1] range -> [-1, 1]
        return image * 2.0 - 1.0

    def prepare_cached_embeddings(self, batch: dict) -> dict:
        """
        Load cached embeddings for both T2I and I2I modes.
        """
        mode = batch.get("mode", "t2i")
        batch["mode"] = mode

        # Common processing
        if "text_ids" in batch and batch["text_ids"].dim() > 2:
            batch["text_ids"] = batch["text_ids"][0]

        # I2I mode: load condition cache
        if mode == "i2i" and "condition_ids" in batch and batch["condition_ids"] is not None:
            if batch["condition_ids"].dim() > 2:
                batch["condition_ids"] = batch["condition_ids"][0]
        else:
            batch["condition_latents"] = None
            batch["condition_ids"] = None

        return batch

    def cache_step(self, data: dict):
        """
        Cache embeddings for both T2I and I2I modes.
        """
        mode = data.get("mode", "t2i")

        # Common cache items
        cache_embeddings = {
            "image_latents": data["image_latents"].detach().cpu()[0],
            "prompt_embeds": data["prompt_embeds"].detach().cpu()[0],
            "text_ids": data["text_ids"].detach().cpu(),
            "empty_prompt_embeds": data["empty_prompt_embeds"].detach().cpu()[0],
            "mode": mode,
        }

        map_keys = {
            "image_latents": "image_hash",
            "prompt_embeds": "prompt_hash",
            "text_ids": "prompt_hash",
            "empty_prompt_embeds": "prompt_hash",
        }

        # I2I mode: add condition cache
        if mode == "i2i" and data.get("condition_latents") is not None:
            cache_embeddings["condition_latents"] = data["condition_latents"].detach().cpu()[0]
            cache_embeddings["condition_ids"] = data["condition_ids"].detach().cpu()
            map_keys["condition_latents"] = "control_hash"
            map_keys["condition_ids"] = "control_hash"

        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """
        Compute flow matching loss for both T2I and I2I modes.

        The key difference is whether condition_latents are concatenated.
        """
        device = self.accelerator.device
        mode = embeddings.get("mode", "t2i")

        # Extract common embeddings
        image_latents = embeddings["image_latents"].to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(device)
        text_ids = embeddings["text_ids"].to(device)

        # Handle latent_ids - may need to generate if not cached
        if "latent_ids" in embeddings and embeddings["latent_ids"] is not None:
            latent_ids = embeddings["latent_ids"].to(device)
        else:
            # Generate latent IDs from image shape
            batch_size = image_latents.shape[0]
            seq_len = image_latents.shape[1]
            # Estimate height/width from sequence length
            hw = int(np.sqrt(seq_len))
            latent_ids = self._prepare_latent_image_ids(batch_size, hw, hw, device, self.weight_dtype)

        batch_size = image_latents.shape[0]

        with torch.no_grad():
            # Sample noise
            noise = torch.randn_like(image_latents, device=device, dtype=self.weight_dtype)

            # Sample timestep (uniform)
            t = torch.rand((batch_size,), device=device, dtype=self.weight_dtype)
            t_ = t.view(-1, 1, 1)

            # Create noisy input
            noisy_model_input = (1.0 - t_) * image_latents + t_ * noise

            # === Mode-specific input preparation ===
            if mode == "i2i" and embeddings.get("condition_latents") is not None:
                # I2I: Concatenate with condition latents
                condition_latents = embeddings["condition_latents"].to(device)
                condition_ids = embeddings["condition_ids"].to(device)

                hidden_states = torch.cat([noisy_model_input, condition_latents], dim=1)
                img_ids = torch.cat([latent_ids, condition_ids], dim=0)
            else:
                # T2I: Only noisy latents
                hidden_states = noisy_model_input
                img_ids = latent_ids

        # Prepare guidance
        guidance = None
        if self.dit.config.guidance_embeds:
            guidance = torch.ones((batch_size,), device=device, dtype=self.weight_dtype)

        # Forward pass
        model_pred = self.dit(
            hidden_states=hidden_states.to(self.weight_dtype),
            timestep=t,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds.to(self.weight_dtype),
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        # Extract prediction for target latents only
        model_pred = model_pred[:, : image_latents.size(1)]

        # Compute loss
        target = noise - image_latents

        # Use edit_mask if available
        edit_mask = embeddings.get("edit_mask")
        if edit_mask is not None:
            edit_mask = edit_mask.to(self.weight_dtype).to(device)

        loss = self.forward_loss(model_pred, target, weighting=None, edit_mask=edit_mask)

        return loss

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """
        Run denoising loop for both T2I and I2I modes.
        """
        mode = embeddings.get("mode", "t2i")
        device = self.dit.device

        num_inference_steps = embeddings.get("num_inference_steps", 50)
        batch_size = embeddings["prompt_embeds"].shape[0]
        height = embeddings.get("height", 1024)
        width = embeddings.get("width", 1024)

        # Create initial noise latents
        latents, _, latent_ids, _ = self.prepare_latents(
            image=None,
            batch_size=batch_size,
            num_channels_latents=self.num_channels_latents,
            height=height,
            width=width,
            dtype=self.weight_dtype,
        )
        latents = latents.to(device)
        latent_ids = latent_ids.to(device)

        # Prepare condition (I2I mode)
        if mode == "i2i" and embeddings.get("condition_latents") is not None:
            condition_latents = embeddings["condition_latents"].to(device)
            condition_ids = embeddings["condition_ids"].to(device)
        else:
            condition_latents = None
            condition_ids = None

        # Prepare text embeddings
        prompt_embeds = embeddings["prompt_embeds"].to(device)
        text_ids = embeddings["text_ids"].to(device)

        # Calculate timesteps using FLUX.2 specific mu (matches original Flux2Pipeline)
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)

        # Prepare sigmas (matches original Flux2Pipeline)
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        if hasattr(self.sampling_scheduler.config, "use_flow_sigmas") and self.sampling_scheduler.config.use_flow_sigmas:
            sigmas = None

        timesteps, num_inference_steps = retrieve_timesteps(
            self.sampling_scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        # Guidance (matches original Flux2Pipeline implementation)
        guidance_scale = embeddings.get("guidance", 2.5)
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])

        # Denoising loop
        self.sampling_scheduler.set_begin_index(0)

        # Check for true CFG
        true_cfg_scale = embeddings.get("true_cfg_scale", 1.0)
        do_cfg = true_cfg_scale > 1.0 and "negative_prompt_embeds" in embeddings

        with torch.inference_mode():
            for t in tqdm(timesteps, desc="FLUX.2 Sampling"):
                # Prepare model input (matches original Flux2Pipeline)
                latent_model_input = latents.to(self.dit.dtype)
                latent_image_ids = latent_ids

                if condition_latents is not None:
                    latent_model_input = torch.cat([latents, condition_latents], dim=1).to(self.dit.dtype)
                    latent_image_ids = torch.cat([latent_ids, condition_ids], dim=1)

                hidden_states = latent_model_input
                img_ids = latent_image_ids

                timestep = t.expand(batch_size).to(latents.dtype)

                # Forward
                noise_pred = self.dit(
                    hidden_states=hidden_states,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]

                # Extract target prediction
                noise_pred = noise_pred[:, : latents.size(1)]

                # Apply CFG if needed
                if do_cfg:
                    neg_embeds = embeddings["negative_prompt_embeds"].to(device)
                    neg_ids = embeddings.get("negative_text_ids", text_ids).to(device)

                    neg_noise_pred = self.dit(
                        hidden_states=hidden_states,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states=neg_embeds,
                        txt_ids=neg_ids,
                        img_ids=img_ids,
                        joint_attention_kwargs={},
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # Scheduler step
                latents = self.sampling_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents, latent_ids

    def decode_vae_latent(self, latents: torch.Tensor, height: int, width: int, latent_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Decode latents using FLUX.2 VAE with batch_norm denormalization.

        Args:
            latents: Packed latents from transformer [B, seq, C]
            height: Target image height
            width: Target image width
            latent_ids: Position IDs for latents [seq, 4] (required for FLUX.2)
        """
        latents = latents.to(self.vae.device, dtype=self.weight_dtype)

        if latent_ids is None:
            # Fallback to simple unpacking if latent_ids not provided
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        else:
            # Use position IDs to unpack latents (matches original Flux2Pipeline)
            latent_ids = latent_ids.to(self.vae.device)
            latents = self._unpack_latents_with_ids(latents, latent_ids)

        # FLUX.2 specific: batch_norm inverse transform
        if hasattr(self.vae, "bn"):
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(
                self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
            ).to(latents.device, latents.dtype)
            latents = latents * latents_bn_std + latents_bn_mean

        # Unpatchify latents (unpack 2x2 patches)
        latents = self._unpatchify_latents(latents)

        # VAE decode
        with torch.inference_mode():
            image = self.vae.decode(latents, return_dict=False)[0]

        image = self.image_processor.postprocess(image, output_type="pt")
        return image

    def prepare_predict_batch_data(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image] | None = None,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        num_inference_steps: int = 50,
        height: int | None = None,
        width: int | None = None,
        guidance_scale: float = 2.5,
        true_cfg_scale: float = 1.0,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> dict:
        """
        Prepare batch data for prediction.

        Uses VaeImageProcessor for image preprocessing to match Flux2Pipeline behavior.

        Args:
            image: Condition image (None for T2I mode, PIL Image for I2I mode)
            prompt: Text prompt
            negative_prompt: Negative prompt for CFG
            num_inference_steps: Number of denoising steps
            height: Output height (if None, uses condition image height or 1024)
            width: Output width (if None, uses condition image width or 1024)
            guidance_scale: Guidance scale
            true_cfg_scale: True CFG scale (>1.0 to enable CFG)
            weight_dtype: Data type

        Returns:
            Batch dictionary for prepare_embeddings
        """
        self.weight_dtype = weight_dtype

        if isinstance(prompt, str):
            prompt = [prompt]

        data = {"prompt": prompt}

        # Determine mode based on image presence
        if image is not None:
            # I2I mode
            if isinstance(image, PIL.Image.Image):
                image = [image]

            # Get original image dimensions for size calculation
            orig_width, orig_height = image[0].size

            # Use original image size if not specified
            if height is None:
                height = orig_height
            if width is None:
                width = orig_width

            # Ensure divisibility (match Pipeline behavior: vae_scale_factor * 2)
            multiple_of = self.vae_scale_factor * 2
            width = (width // multiple_of) * multiple_of
            height = (height // multiple_of) * multiple_of

            # Process condition images using VaeImageProcessor (matches Flux2Pipeline)
            # VaeImageProcessor.preprocess returns [-1, 1] range tensor
            condition_images = []
            for img in image:
                processed = self.image_processor.preprocess(
                    img, height=height, width=width, resize_mode="crop"
                )
                condition_images.append(processed)

            # Stack and squeeze to get [B, C, H, W]
            condition = torch.cat(condition_images, dim=0)
            data["control"] = condition
        else:
            # T2I mode - use default or specified size
            height = height or 1024
            width = width or 1024

            # Ensure divisibility
            width, height = make_image_shape_devisible(width, height, self.vae_scale_factor)

        data["height"] = height
        data["width"] = width
        data["num_inference_steps"] = num_inference_steps
        data["guidance"] = guidance_scale
        data["true_cfg_scale"] = true_cfg_scale
        data["n_controls"] = 0  # FLUX.2 doesn't use multiple controls

        # Negative prompt
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            data["negative_prompt"] = negative_prompt

        logging.info(f"Prepared predict batch: mode={'i2i' if image else 't2i'}, size={width}x{height}")

        return data

    # === Static helper methods ===

    @staticmethod
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Pack latents for transformer input."""
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    @staticmethod
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        """
        Unpack latents using position IDs to scatter tokens into place.
        Matches original Flux2Pipeline._unpack_latents_with_ids implementation.

        Args:
            x: Packed latents [B, seq, C]
            x_ids: Position IDs [seq, 4] or [B, seq, 4]

        Returns:
            Unpacked latents [B, C, H, W]
        """
        x_list = []
        # Handle both [seq, 4] and [B, seq, 4] cases
        if x_ids.ndim == 2:
            x_ids = x_ids.unsqueeze(0).expand(x.shape[0], -1, -1)

        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            # Reshape from (H * W, C) to (C, H, W)
            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """
        Unpatchify latents: unpack 2x2 patches.
        Matches original Flux2Pipeline._unpatchify_latents implementation.

        Args:
            latents: [B, C, H, W] where C is divisible by 4

        Returns:
            Unpacked latents [B, C//4, H*2, W*2]
        """
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
        """Unpack latents from transformer output (fallback method)."""
        batch_size, num_patches, channels = latents.shape
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    @staticmethod
    def _prepare_latent_image_ids(
        batch_size: int, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Prepare latent image IDs for position encoding.

        Returns:
            latent_image_ids: [height * width, 4] for FLUX.2
        """
        # FLUX.2 uses 4-dim IDs: [domain, h_pos, w_pos, extra]
        latent_image_ids = torch.zeros(height, width, 4, device=device, dtype=dtype)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width, device=device)[None, :]

        latent_image_ids = latent_image_ids.reshape(height * width, 4)
        return latent_image_ids
