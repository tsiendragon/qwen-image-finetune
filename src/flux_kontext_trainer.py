"""
Flux Kontext LoRA Trainer Implementation
Following QwenImageEditTrainer patterns with dual text encoder support.
"""

import copy
import os
import shutil
import torch
import PIL
import gc
import numpy as np
from typing import Optional, Union, List, Tuple
from PIL import Image
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft.utils import get_peft_model_state_dict
from peft import LoraConfig

from diffusers import FluxKontextPipeline
from src.base_trainer import BaseTrainer
from src.models.flux_kontext_loader import (
    load_flux_kontext_vae, load_flux_kontext_clip,
    load_flux_kontext_t5, load_flux_kontext_transformer,
    load_flux_kontext_tokenizers, load_flux_kontext_scheduler
)
from src.utils.logger import get_logger
from src.data.cache_manager import check_cache_exists
import logging

logger = get_logger(__name__, log_level="INFO")


def get_lora_layers(model):
    """Traverse the model to find all LoRA-related modules"""
    lora_layers = {}

    def fn_recursive_find_lora_layer(name: str, module: torch.nn.Module, processors):
        if "lora" in name:
            lora_layers[name] = module
        for sub_name, child in module.named_children():
            fn_recursive_find_lora_layer(f"{name}.{sub_name}", child, lora_layers)
        return lora_layers

    for name, module in model.named_children():
        fn_recursive_find_lora_layer(name, module, lora_layers)

    return lora_layers


class FluxKontextLoraTrainer(BaseTrainer):
    """
    Flux Kontext LoRA Trainer implementation following QwenImageEditTrainer patterns.
    Inherits from BaseTrainer to ensure consistent interface.
    """

    def __init__(self, config):
        """Initialize Flux Kontext trainer with configuration."""
        super().__init__(config)

        # Flux Kontext specific components (similar to QwenImageEditTrainer structure)
        self.vae = None                    # FluxVAE
        self.text_encoder = None           # CLIP text encoder
        self.text_encoder_2 = None         # T5 text encoder
        self.transformer = None            # Flux transformer
        self.tokenizer = None              # CLIP tokenizer
        self.tokenizer_2 = None            # T5 tokenizer
        self.scheduler = None              # FlowMatchEulerDiscreteScheduler

        # Cache-related attributes (following QwenImageEditTrainer pattern)
        self.cache_exist = check_cache_exists(config.cache.cache_dir)

        # Flux-specific configurations
        self.quantize = config.model.quantize
        self.prompt_image_dropout_rate = config.data.init_args.get('prompt_image_dropout_rate', 0.1)

        # VAE parameters (similar to QwenImageEditTrainer)
        self.vae_scale_factor = None
        self.vae_latent_mean = None
        self.vae_latent_std = None
        self.vae_z_dim = None

        # Flux-specific attributes
        self.latent_channels = None
        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False

        self.log_model_info()

    def load_model(self, text_encoder_device=None, text_encoder_2_device=None):
        """
        Load and separate components from FluxKontextPipeline.
        Follows QwenImageEditTrainer.load_model() pattern exactly.
        """
        logging.info("Loading FluxKontextPipeline and separating components...")

        # Separate individual components using flux_kontext_loader
        self.vae = load_flux_kontext_vae(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )
        self.text_encoder = load_flux_kontext_clip(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )
        self.text_encoder_2 = load_flux_kontext_t5(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )
        self.transformer = load_flux_kontext_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )

        # Load tokenizers and scheduler
        self.tokenizer, self.tokenizer_2 = load_flux_kontext_tokenizers(
            self.config.model.pretrained_model_name_or_path
        )
        self.scheduler = load_flux_kontext_scheduler(
            self.config.model.pretrained_model_name_or_path
        )

        # Set VAE-related parameters (following QwenImageEditTrainer pattern)
        if hasattr(self.vae.config, 'block_out_channels'):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8  # Default value

        # Set Flux-specific VAE configuration
        if hasattr(self.vae.config, 'latent_channels'):
            self.vae_z_dim = self.vae.config.latent_channels
        else:
            self.vae_z_dim = 16  # Typical Flux latent channels

        # Additional Flux-specific attributes
        self.latent_channels = self.vae_z_dim

        # Set models to training/evaluation mode (same as QwenImageEditTrainer)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        torch.cuda.empty_cache()

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def cache(self, train_dataloader):
        """
        Pre-compute and cache embeddings (exactly same signature as QwenImageEditTrainer).
        Implements dual text encoder caching for CLIP + T5.
        """
        from tqdm import tqdm

        self.cache_manager = train_dataloader.cache_manager
        vae_encoder_device = self.config.cache.vae_encoder_device
        text_encoder_device = self.config.cache.text_encoder_device
        text_encoder_2_device = self.config.cache.get('text_encoder_2_device', text_encoder_device)

        logging.info("Starting embedding caching process...")

        # Load models (following QwenImageEditTrainer pattern)
        self.load_model(
            text_encoder_device=text_encoder_device,
            text_encoder_2_device=text_encoder_2_device
        )
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.vae.eval()
        self.set_model_devices(mode="cache")

        # Cache for each item (same loop structure as QwenImageEditTrainer)
        dataset = train_dataloader.dataset
        for data in tqdm(dataset, total=len(dataset), desc="cache_embeddings"):
            self.cache_step(data, vae_encoder_device, text_encoder_device, text_encoder_2_device)

        logging.info("Cache completed")

        # Clean up models (same as QwenImageEditTrainer)
        self.text_encoder.cpu()
        self.text_encoder_2.cpu()
        self.vae.cpu()
        del self.text_encoder
        del self.text_encoder_2
        del self.vae

    def cache_step(self, data: dict, vae_encoder_device: str, text_encoder_device: str, text_encoder_2_device: str):
        """
        Cache VAE latents and dual prompt embeddings.
        Follows QwenImageEditTrainer.cache_step() structure exactly.
        """
        image, control, prompt = data["image"], data["control"], data["prompt"]

        # Image processing (same as QwenImageEditTrainer)
        image = image.transpose(1, 2, 0)
        control = control.transpose(1, 2, 0)
        image = Image.fromarray(image)
        control = Image.fromarray(control)

        # Calculate embeddings for both encoders
        clip_embeds, clip_mask = self.encode_clip_prompt([prompt], device=text_encoder_device)
        t5_embeds, t5_mask = self.encode_t5_prompt([prompt], device=text_encoder_2_device)

        # Empty prompt embeddings for CFG
        empty_clip_embeds, empty_clip_mask = self.encode_clip_prompt([""], device=text_encoder_device)
        empty_t5_embeds, empty_t5_mask = self.encode_t5_prompt([""], device=text_encoder_2_device)

        # VAE encoding (similar logic as QwenImageEditTrainer)
        # Process images to tensors for VAE encoding
        image_tensor = self._preprocess_image_for_vae(image)
        control_tensor = self._preprocess_image_for_vae(control)

        # Encode with VAE
        image_latents = self._encode_vae_image(image_tensor.unsqueeze(0).to(vae_encoder_device))
        control_latents = self._encode_vae_image(control_tensor.unsqueeze(0).to(vae_encoder_device))

        # Save to cache (following QwenImageEditTrainer pattern)
        file_hashes = data["file_hashes"]
        self.cache_manager.save_cache("pixel_latent", file_hashes["image_hash"], image_latents[0].cpu())
        self.cache_manager.save_cache("control_latent", file_hashes["control_hash"], control_latents[0].cpu())
        self.cache_manager.save_cache("clip_prompt_embed", file_hashes["prompt_hash"], clip_embeds[0].cpu())
        self.cache_manager.save_cache("clip_prompt_mask", file_hashes["prompt_hash"], clip_mask[0].cpu())
        self.cache_manager.save_cache("t5_prompt_embed", file_hashes["prompt_hash"], t5_embeds[0].cpu())
        self.cache_manager.save_cache("t5_prompt_mask", file_hashes["prompt_hash"], t5_mask[0].cpu())
        self.cache_manager.save_cache("empty_clip_prompt_embed", file_hashes["empty_prompt_hash"], empty_clip_embeds[0].cpu())
        self.cache_manager.save_cache("empty_t5_prompt_embed", file_hashes["empty_prompt_hash"], empty_t5_embeds[0].cpu())

    def _preprocess_image_for_vae(self, image: PIL.Image) -> torch.Tensor:
        """Preprocess PIL image for VAE encoding."""
        # Convert PIL to tensor
        image_array = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)  # HWC -> CHW
        # Normalize to [-1, 1]
        image_tensor = image_tensor * 2.0 - 1.0
        return image_tensor

    def _encode_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image using VAE."""
        with torch.no_grad():
            # Ensure correct input format for Flux VAE
            if image.dim() == 3:
                image = image.unsqueeze(0)  # Add batch dimension

            latents = self.vae.encode(image).latent_dist.sample()

            # Apply scaling factor if available
            if hasattr(self.vae.config, 'scaling_factor'):
                latents = latents * self.vae.config.scaling_factor

            return latents

    def fit(self, train_dataloader):
        """
        Main training loop (exactly same signature as QwenImageEditTrainer).
        Implements Flux Kontext specific training with LoRA.
        """
        logging.info("Starting training process...")

        # Setup components (same order as QwenImageEditTrainer)
        self.setup_accelerator()
        self.load_model()

        self.set_lora()  # Flux-specific LoRA setup
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.vae.eval()
        self.configure_optimizers()
        self.set_model_devices(mode="train")
        train_dataloader = self.accelerator_prepare(train_dataloader)

        logging.info("***** Running training *****")
        logging.info(f"  Instantaneous batch size per device = {self.batch_size}")
        logging.info(f"  Gradient Accumulation steps = {self.config.train.gradient_accumulation_steps}")
        logging.info(f"  Use cache: {self.use_cache}, Cache exists: {self.cache_exist}")

        # Training loop implementation (following QwenImageEditTrainer structure)
        # Progress bar
        progress_bar = tqdm(
            range(0, self.config.train.max_train_steps),
            desc="train",
            disable=not self.accelerator.is_local_main_process,
        )

        # Training loop
        train_loss = 0.0
        running_loss = 0.0

        for epoch in range(self.config.train.num_epochs):
            for _, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.transformer):
                    loss = self.training_step(batch)

                    # Backward pass
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.transformer.parameters(),
                            self.config.train.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Update when syncing gradients
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                    # Calculate average loss
                    avg_loss = self.accelerator.gather(
                        loss.repeat(self.batch_size)
                    ).mean()
                    train_loss += avg_loss.item() / self.config.train.gradient_accumulation_steps
                    running_loss = train_loss

                    # Log metrics
                    self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                    train_loss = 0.0

                    # Save checkpoint
                    if self.global_step % self.config.train.checkpointing_steps == 0:
                        self.save_checkpoint(epoch, self.global_step)

                # Update progress bar
                logs = {
                    "loss": f"{running_loss:.3f}",
                    "lr": f"{self.lr_scheduler.get_last_lr()[0]:.1e}",
                }
                progress_bar.set_postfix(**logs)

                # Check if maximum steps reached
                if self.global_step >= self.config.train.max_train_steps:
                    break

        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def training_step(self, batch):
        """Execute a single training step (same signature as QwenImageEditTrainer)."""
        # Check if cached data is available (same logic as QwenImageEditTrainer)
        if 'clip_prompt_embed' in batch and 't5_prompt_embed' in batch and 'pixel_latent' in batch:
            return self._training_step_cached(batch)
        else:
            return self._training_step_compute(batch)

    def _training_step_cached(self, batch):
        """Training step using cached embeddings (follows QwenImageEditTrainer pattern)."""
        pixel_latents = batch["pixel_latent"].to(self.accelerator.device, dtype=self.weight_dtype)
        control_latents = batch["control_latent"].to(self.accelerator.device, dtype=self.weight_dtype)

        # Dual encoder embeddings (Flux-specific)
        clip_embeds = batch["clip_prompt_embed"].to(self.accelerator.device)
        t5_embeds = batch["t5_prompt_embed"].to(self.accelerator.device)
        clip_mask = batch["clip_prompt_mask"].to(self.accelerator.device)
        t5_mask = batch["t5_prompt_mask"].to(self.accelerator.device)

        # Combine embeddings for Flux Kontext (model-specific logic)
        combined_embeds, combined_mask = self.combine_text_embeddings(
            clip_embeds, t5_embeds, clip_mask, t5_mask
        )

        return self._compute_loss(pixel_latents, control_latents, combined_embeds, combined_mask)

    def _training_step_compute(self, batch):
        """Training step with embedding computation (no cache)"""
        # Similar to QwenImageEditTrainer but with dual encoders
        image, control, prompt = batch["image"], batch["control"], batch["prompt"]

        # Convert to PIL images
        image = [Image.fromarray(x.cpu().numpy().transpose(1, 2, 0)) for x in image]
        control = [Image.fromarray(x.cpu().numpy().transpose(1, 2, 0)) for x in control]

        # Encode prompts with both encoders
        clip_embeds, clip_mask = self.encode_clip_prompt(prompt, device=self.accelerator.device)
        t5_embeds, t5_mask = self.encode_t5_prompt(prompt, device=self.accelerator.device)

        # Combine embeddings
        combined_embeds, combined_mask = self.combine_text_embeddings(
            clip_embeds, t5_embeds, clip_mask, t5_mask
        )

        # Encode images with VAE
        image_tensors = torch.stack([self._preprocess_image_for_vae(img) for img in image])
        control_tensors = torch.stack([self._preprocess_image_for_vae(img) for img in control])

        image_latents = self._encode_vae_image(image_tensors.to(self.accelerator.device))
        control_latents = self._encode_vae_image(control_tensors.to(self.accelerator.device))

        return self._compute_loss(image_latents, control_latents, combined_embeds, combined_mask)

    def combine_text_embeddings(self, clip_embeds, t5_embeds, clip_mask, t5_mask):
        """
        Combine CLIP and T5 embeddings for Flux Kontext.
        This is a simplified implementation - actual Flux may use different combination methods.
        """
        # Simple concatenation approach
        combined_embeds = torch.cat([clip_embeds, t5_embeds], dim=-1)

        # For masks, we use the longer sequence
        if clip_mask.size(1) >= t5_mask.size(1):
            combined_mask = clip_mask
        else:
            combined_mask = t5_mask

        return combined_embeds, combined_mask

    def _compute_loss(self, pixel_latents, control_latents, prompt_embeds, prompt_embeds_mask):
        """Calculate the flow matching loss (same structure as QwenImageEditTrainer)."""

        with torch.no_grad():
            batch_size = pixel_latents.shape[0]
            noise = torch.randn_like(
                pixel_latents, device=self.accelerator.device, dtype=self.weight_dtype
            )

            # Sample timesteps
            u = compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            timesteps = self.scheduler.timesteps[indices].to(device=pixel_latents.device)

            sigmas = self._get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

            # Prepare input for transformer
            packed_input = torch.cat([noisy_model_input, control_latents], dim=1)

        # Forward pass through transformer
        model_pred = self.transformer(
            hidden_states=packed_input,
            timestep=timesteps / 1000,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            return_dict=False,
        )[0]

        model_pred = model_pred[:, :pixel_latents.size(1)]

        # Calculate loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        target = noise - pixel_latents

        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(
                target.shape[0], -1
            ),
            1,
        )
        loss = loss.mean()
        return loss

    def _get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """Calculate sigma values for noise scheduler"""
        noise_scheduler_copy = copy.deepcopy(self.scheduler)
        sigmas = noise_scheduler_copy.sigmas.to(device=self.accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def predict(
        self,
        prompt_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Union[str, List[str]],
        negative_prompt: Union[None, str, List[str]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Prediction method for Flux Kontext model inference.
        Implements the complete Flux Kontext inference pipeline.

        Args:
            prompt_image: Input image(s) for conditioning
            prompt: Text prompt(s) for generation
            negative_prompt: Negative text prompt(s) for CFG
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height: Output image height (if None, calculated from input)
            width: Output image width (if None, calculated from input)
            generator: Random generator for reproducible results
            weight_dtype: Weight data type for computations

        Returns:
            Generated image(s) as numpy arrays
        """
        logging.info(f"Starting Flux Kontext prediction with {num_inference_steps} steps, " +
                     f"guidance scale: {guidance_scale}")

        assert prompt_image is not None, "prompt_image is required"
        assert prompt is not None, "prompt is required"

        # 1. Process input format
        if isinstance(prompt_image, PIL.Image.Image):
            image = [prompt_image]
            batch_size = 1
        else:
            image = prompt_image
            batch_size = len(image)

        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        elif len(prompt) != batch_size:
            raise ValueError(f"Number of prompts ({len(prompt)}) must match number of " +
                             f"images ({batch_size})")

        self.weight_dtype = weight_dtype

        # 2. Calculate image dimensions
        if height is None or width is None:
            original_size = image[0].size
            # Flux typically uses 1024x1024 or maintains aspect ratio
            if height is None and width is None:
                height = width = 1024
            elif height is None:
                height = int(width * original_size[1] / original_size[0])
            elif width is None:
                width = int(height * original_size[0] / original_size[1])

        # Ensure dimensions are multiples of 16 (VAE downsampling factor)
        height = (height // 16) * 16
        width = (width // 16) * 16

        # 3. Get device configurations
        device_vae = self.config.predict.devices.get('vae', 'cuda:0')
        device_text_encoder = self.config.predict.devices.get('text_encoder', 'cuda:0')
        device_text_encoder_2 = self.config.predict.devices.get('text_encoder_2', 'cuda:0')
        device_transformer = self.config.predict.devices.get('transformer', 'cuda:0')

        logging.info(f"Using devices - VAE: {device_vae}, Text Encoders: {device_text_encoder}/" +
                     f"{device_text_encoder_2}, Transformer: {device_transformer}")

        # 4. Preprocess images
        processed_images = []
        for img in image:
            img_resized = img.resize((width, height), PIL.Image.LANCZOS)
            processed_images.append(img_resized)

        # 5. Encode prompts (dual encoder: CLIP + T5)
        do_classifier_free_guidance = guidance_scale > 1.0 and negative_prompt is not None

        # Encode positive prompts
        clip_prompt_embeds, t5_prompt_embeds = self.encode_flux_prompt(
            prompt, processed_images, device_text_encoder, device_text_encoder_2
        )

        # Encode negative prompts for CFG
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [""] * batch_size
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            clip_negative_embeds, t5_negative_embeds = self.encode_flux_prompt(
                negative_prompt, processed_images, device_text_encoder, device_text_encoder_2
            )

            # Concatenate for CFG
            clip_prompt_embeds = torch.cat([clip_negative_embeds, clip_prompt_embeds])
            t5_prompt_embeds = torch.cat([t5_negative_embeds, t5_prompt_embeds])

        # 6. Prepare latents
        latents = self.prepare_flux_latents(
            batch_size, height, width, device_transformer, generator, weight_dtype
        )

        # Encode input images to latents for conditioning
        image_latents = self.encode_flux_images(processed_images, device_vae, weight_dtype)

        # 7. Prepare scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=device_transformer)
        timesteps = self.scheduler.timesteps

        # 8. Denoising loop
        with torch.inference_mode():
            for i, t in enumerate(tqdm(timesteps, desc="Flux Kontext Generation")):
                # Prepare model input
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                    image_latent_input = torch.cat([image_latents] * 2)
                else:
                    latent_model_input = latents
                    image_latent_input = image_latents

                # Scale model input
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # Prepare conditioning
                conditioning = torch.cat([image_latent_input, latent_model_input], dim=1)

                # Forward pass through transformer
                noise_pred = self.transformer(
                    hidden_states=conditioning,
                    timestep=t,
                    encoder_hidden_states=clip_prompt_embeds,
                    pooled_projections=t5_prompt_embeds,
                    return_dict=False
                )[0]

                # Perform CFG
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute previous sample
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # 9. Decode latents to images
        self.vae.to(device_vae)
        latents = latents.to(device_vae, dtype=self.vae.dtype)

        # Scale latents back to original range
        latents = latents / self.vae.config.scaling_factor

        # Decode with VAE
        with torch.inference_mode():
            images = self.vae.decode(latents, return_dict=False)[0]

        # 10. Post-process images
        images = (images / 2 + 0.5).clamp(0, 1)  # Denormalize
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()  # BCHW -> BHWC

        # Convert to PIL images
        pil_images = []
        for img_array in images:
            img_array = (img_array * 255).astype(np.uint8)
            pil_img = PIL.Image.fromarray(img_array)
            pil_images.append(pil_img)

        logging.info(f"Successfully generated {len(pil_images)} images")

        if batch_size == 1:
            return pil_images[0]
        else:
            return pil_images

    def encode_flux_prompt(
        self,
        prompt: List[str],
        images: List[PIL.Image.Image],
        device_text_encoder: str,
        device_text_encoder_2: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prompts using both CLIP and T5 encoders for Flux Kontext.

        Args:
            prompt: List of text prompts
            images: List of conditioning images (for potential image-aware encoding)
            device_text_encoder: Device for CLIP encoder
            device_text_encoder_2: Device for T5 encoder

        Returns:
            Tuple of (clip_embeddings, t5_embeddings)
        """

        # Encode with CLIP text encoder
        self.text_encoder.to(device_text_encoder)
        clip_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        with torch.inference_mode():
            clip_embeddings = self.text_encoder(
                clip_inputs.input_ids.to(device_text_encoder)
            )[0]  # [batch_size, seq_len, hidden_size]

        # Encode with T5 text encoder
        self.text_encoder_2.to(device_text_encoder_2)
        t5_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=512,  # T5 typical max length
            truncation=True,
            return_tensors="pt"
        )

        with torch.inference_mode():
            t5_embeddings = self.text_encoder_2(
                t5_inputs.input_ids.to(device_text_encoder_2)
            )[0]  # [batch_size, seq_len, hidden_size]

        return clip_embeddings, t5_embeddings

    def prepare_flux_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: str,
        generator: Optional[torch.Generator],
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Prepare random latents for Flux generation.

        Args:
            batch_size: Number of images to generate
            height: Height of output images
            width: Width of output images
            device: Device to create latents on
            generator: Random generator for reproducible results
            dtype: Data type for latents

        Returns:
            Random latents tensor
        """
        # Flux uses 16x downsampling factor
        latent_height = height // 8  # VAE downsampling
        latent_width = width // 8

        # Flux latent channels (typically 4 for VAE)
        latent_channels = getattr(self.transformer.config, 'in_channels', 16) // 2  # Divide by 2

        shape = (batch_size, latent_channels, latent_height, latent_width)

        if generator is not None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = torch.randn(shape, device=device, dtype=dtype)

        # Scale latents by scheduler's init noise sigma
        if hasattr(self.scheduler, 'init_noise_sigma'):
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    def encode_flux_images(
        self,
        images: List[PIL.Image.Image],
        device: str,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Encode input images to latent space using VAE.

        Args:
            images: List of PIL images
            device: Device for VAE
            dtype: Data type for computation

        Returns:
            Encoded image latents
        """
        self.vae.to(device)

        # Convert PIL images to tensor
        image_tensors = []
        for img in images:
            # Convert to tensor and normalize to [-1, 1]
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
            img_tensor = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
            image_tensors.append(img_tensor)

        # Stack into batch
        image_batch = torch.stack(image_tensors).to(device, dtype=dtype)

        # Encode with VAE
        with torch.inference_mode():
            image_latents = self.vae.encode(image_batch).latent_dist.sample()
            image_latents = image_latents * self.vae.config.scaling_factor

        return image_latents

    # Additional methods following QwenImageEditTrainer patterns
    def set_model_devices(self, mode="train"):
        """Set model device allocation (same signature as QwenImageEditTrainer)."""
        if mode == "train":
            assert hasattr(self, "accelerator"), "accelerator must be set before setting model devices"

        if self.cache_exist and self.use_cache and mode == "train":
            # Cache mode: only need transformer
            self.text_encoder.cpu()
            self.text_encoder_2.cpu()
            torch.cuda.empty_cache()
            self.vae.cpu()
            torch.cuda.empty_cache()
            del self.text_encoder
            del self.text_encoder_2
            del self.vae
            gc.collect()
            self.transformer.to(self.accelerator.device)

        elif not self.use_cache and mode == "train":
            # Non-cache mode: need all encoders
            self.vae.to(self.accelerator.device)
            self.text_encoder.to(self.accelerator.device)
            self.text_encoder_2.to(self.accelerator.device)
            self.transformer.to(self.accelerator.device)

        elif mode == "cache":
            # Cache mode: need encoders, don't need transformer
            self.vae = self.vae.to(self.config.cache.vae_encoder_device, non_blocking=True)
            self.text_encoder = self.text_encoder.to(self.config.cache.text_encoder_device, non_blocking=True)
            self.text_encoder_2 = self.text_encoder_2.to(
                self.config.cache.get('text_encoder_2_device', self.config.cache.text_encoder_device),
                non_blocking=True
            )

            torch.cuda.synchronize()
            self.transformer.cpu()
            torch.cuda.empty_cache()
            del self.transformer
            gc.collect()

        elif mode == "predict":
            # Predict mode: allocate to different GPUs according to configuration
            devices = self.config.predict.devices
            self.vae.to(devices['vae'])
            self.text_encoder.to(devices['text_encoder'])
            self.text_encoder_2.to(devices.get('text_encoder_2', devices['text_encoder']))
            self.transformer.to(devices['transformer'])

    def encode_prompt(self, prompt, image=None, device=None):
        """
        Main prompt encoding interface that combines CLIP and T5.
        Follows QwenImageEditTrainer.encode_prompt() signature.
        """
        # Encode with both encoders
        clip_embeds, clip_mask = self.encode_clip_prompt(prompt, device=device)
        t5_embeds, t5_mask = self.encode_t5_prompt(prompt, device=device)

        # Combine embeddings
        combined_embeds, combined_mask = self.combine_text_embeddings(
            clip_embeds, t5_embeds, clip_mask, t5_mask
        )

        return combined_embeds, combined_mask

    def encode_clip_prompt(self, prompt: Union[str, List[str]], device=None):
        """Get CLIP text embeddings."""
        if isinstance(prompt, str):
            prompt = [prompt]

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device or self.text_encoder.device)

        # Encode
        with torch.no_grad():
            outputs = self.text_encoder(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs.attention_mask

        return embeddings, attention_mask

    def encode_t5_prompt(self, prompt: Union[str, List[str]], device=None):
        """Get T5 text embeddings."""
        if isinstance(prompt, str):
            prompt = [prompt]

        # Tokenize
        inputs = self.tokenizer_2(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device or self.text_encoder_2.device)

        # Encode
        with torch.no_grad():
            outputs = self.text_encoder_2(**inputs)
            embeddings = outputs.last_hidden_state
            attention_mask = inputs.attention_mask

        return embeddings, attention_mask

    def set_lora(self):
        """Set LoRA configuration (same structure as QwenImageEditTrainer)."""
        if self.quantize:
            self.transformer = self.quantize_model(self.transformer, self.accelerator.device)
        else:
            self.transformer.to(self.accelerator.device)

        # Same LoRA config structure as QwenImageEditTrainer
        lora_config = LoraConfig(
            r=self.config.model.lora.r,
            lora_alpha=self.config.model.lora.lora_alpha,
            init_lora_weights=self.config.model.lora.init_lora_weights,
            target_modules=self.config.model.lora.target_modules,  # Flux-specific modules
        )

        self.transformer.add_adapter(lora_config)

        # Load pretrained LoRA weights if specified
        if hasattr(self.config.model.lora, 'pretrained_weight') and self.config.model.lora.pretrained_weight:
            try:
                self.load_lora(self.config.model.lora.pretrained_weight)
                logging.info(f"Successfully loaded pretrained LoRA weights: {self.config.model.lora.pretrained_weight}")
            except Exception as e:
                logging.error(f"Failed to load pretrained LoRA weights: {e}")
                logging.info("Continuing with initialized LoRA weights")

        # Same gradient setup as QwenImageEditTrainer
        self.transformer.requires_grad_(False)
        self.transformer.train()

        # Enable gradient checkpointing if configured
        if self.config.train.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            logging.info("Gradient checkpointing enabled for memory efficiency")

        # Train only LoRA parameters
        trainable_params = 0
        for name, param in self.transformer.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        logging.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    def quantize_model(self, model, device):
        """Quantize model for memory efficiency."""
        # Placeholder for quantization logic
        # This would implement FP8/FP4 quantization similar to QwenImageEditTrainer
        logging.info("Quantization not implemented yet - using original model")
        return model.to(device)

    def load_lora(self, pretrained_weight):
        """Load pretrained LoRA weights"""
        if pretrained_weight is not None:
            self.transformer.load_lora_adapter(pretrained_weight)
            logging.info(f"Loaded LoRA weights from {pretrained_weight}")

    def save_lora(self, save_path):
        """Save LoRA weights"""
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod

        lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_transformer)
        )

        # Use FluxKontextPipeline's save method if available, otherwise use generic method
        try:
            FluxKontextPipeline.save_lora_weights(
                save_path, lora_state_dict, safe_serialization=True
            )
        except AttributeError:
            # Fallback to torch.save
            torch.save(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.safetensors"))

        logging.info(f"Saved LoRA weights to {save_path}")

    # Inherit common methods from BaseTrainer and QwenImageEditTrainer pattern
    def setup_accelerator(self):
        """Initialize accelerator and logging configuration."""
        # Setup versioned logging directory
        self.setup_versioned_logging_dir()

        # Set logging_dir to the versioned output directory directly
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.logging.output_dir,
            logging_dir=self.config.logging.output_dir
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            mixed_precision=self.config.train.mixed_precision,
            log_with=self.config.logging.report_to,
            project_config=accelerator_project_config,
        )

        # Initialize tracker
        if self.config.logging.report_to == "tensorboard":
            try:
                simple_config = {
                    "learning_rate": float(self.config.optimizer.init_args.get("lr", 0.0001)),
                    "batch_size": int(self.config.data.batch_size),
                    "max_train_steps": int(self.config.train.max_train_steps),
                    "model_type": "flux_kontext",
                    "lora_r": int(self.config.model.lora.r),
                    "lora_alpha": int(self.config.model.lora.lora_alpha),
                }
                self.accelerator.init_trackers("", config=simple_config)
            except Exception as e:
                logging.warning(f"Failed to initialize trackers with config: {e}")
                self.accelerator.init_trackers("")

        logging.info(f"Number of devices used in DDP training: {self.accelerator.num_processes}")

        # Set weight data type
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Create output directory
        if (
            self.accelerator.is_main_process
            and self.config.logging.output_dir is not None
        ):
            os.makedirs(self.config.logging.output_dir, exist_ok=True)

        logging.info(f"Mixed precision: {self.accelerator.mixed_precision}")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        lora_layers = filter(lambda p: p.requires_grad, self.transformer.parameters())

        # Use optimizer parameters from configuration
        optimizer_config = self.config.optimizer.init_args
        self.optimizer = torch.optim.AdamW(
            lora_layers,
            lr=optimizer_config["lr"],
            betas=optimizer_config["betas"],
            weight_decay=optimizer_config.get("weight_decay", 0.01),
            eps=optimizer_config.get("eps", 1e-8),
        )

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_scheduler.warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.train.max_train_steps * self.accelerator.num_processes,
        )

    def accelerator_prepare(self, train_dataloader):
        """Prepare accelerator"""
        lora_layers_model = AttnProcsLayers(get_lora_layers(self.transformer))

        # Enable gradient checkpointing if configured
        if self.config.train.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        lora_layers_model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            lora_layers_model, self.optimizer, train_dataloader, self.lr_scheduler
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        return train_dataloader

    def save_checkpoint(self, epoch, global_step):
        """Save checkpoint"""
        if not self.accelerator.is_main_process:
            return

        # Manage checkpoint count
        if self.config.train.checkpoints_total_limit is not None:
            checkpoints = os.listdir(self.config.logging.output_dir)
            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

            if len(checkpoints) >= self.config.train.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - self.config.train.checkpoints_total_limit + 1
                removing_checkpoints = checkpoints[0:num_to_remove]

                for removing_checkpoint in removing_checkpoints:
                    removing_checkpoint = os.path.join(
                        self.config.logging.output_dir, removing_checkpoint
                    )
                    shutil.rmtree(removing_checkpoint)

        save_path = os.path.join(
            self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}"
        )
        os.makedirs(save_path, exist_ok=True)

        # Save LoRA weights
        self.save_lora(save_path)

    def setup_versioned_logging_dir(self):
        """Set up versioned logging directory (same as QwenImageEditTrainer)."""
        base_output_dir = self.config.logging.output_dir
        project_name = self.config.logging.tracker_project_name

        # Create project directory structure: output_dir/project_name/v0
        project_dir = os.path.join(base_output_dir, project_name)

        # If project directory doesn't exist, use v0
        if not os.path.exists(project_dir):
            versioned_dir = os.path.join(project_dir, "v0")
            self.config.logging.output_dir = versioned_dir
            logging.info(f"Created new training version directory: {versioned_dir}")
            return

        # Find existing versions and create next version
        existing_versions = []
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            if os.path.isdir(item_path) and item.startswith('v') and item[1:].isdigit():
                version_num = int(item[1:])
                existing_versions.append(version_num)

        # Determine new version number
        if existing_versions:
            next_version = max(existing_versions) + 1
        else:
            next_version = 0

        # Create new version directory
        versioned_dir = os.path.join(project_dir, f"v{next_version}")
        self.config.logging.output_dir = versioned_dir
        logging.info(f"Using training version directory: {versioned_dir}")
