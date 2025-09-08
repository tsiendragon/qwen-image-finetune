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
import torch.nn.functional as F
import inspect

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
from diffusers.utils.torch_utils import randn_tensor
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
from src.utils.lora_utils import classify
from src.data.cache_manager import check_cache_exists
import logging


logger = get_logger(__name__, log_level="INFO")


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


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
        self.adapter_name = self.config.model.lora.adapter_name
        self.log_model_info()

    def __repr__(self) -> str:
        msg = f"FluxKontextLoraTrainer(config={self.config})"
        return msg

    def load_model(self):
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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8

        # Set Flux-specific VAE configuration
        self.vae_z_dim = self.vae.config.latent_channels if getattr(self, "vae", None) else 16

        # Additional Flux-specific attributes
        self.latent_channels = self.vae_z_dim

        # Set models to training/evaluation mode (same as QwenImageEditTrainer)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.num_channels_latents = self.transformer.config.in_channels // 4 if getattr(self, "transformer", None) else 16
        self.tokenizer_max_length = 77  # clip encoder maximal token length
        self.max_sequence_length = 512  # T5 encoder maximal token length

        torch.cuda.empty_cache()

        from diffusers.image_processor import VaeImageProcessor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def cache(self, train_dataloader):
        """
        Pre-compute and cache embeddings (exactly same signature as QwenImageEditTrainer).
        Implements dual text encoder caching for CLIP + T5.
        """
        self.cache_manager = train_dataloader.cache_manager
        vae_encoder_device = self.config.cache.vae_encoder_device
        text_encoder_device = self.config.cache.text_encoder_device
        text_encoder_2_device = self.config.cache.get('text_encoder_2_device', text_encoder_device)

        logging.info("Starting embedding caching process...")

        # Load models (following QwenImageEditTrainer pattern)
        self.load_model()
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
        # image: RGB，C,H,W

        image = torch.from_numpy(image).unsqueeze(0)  # [1,C,H,W]
        image = self.preprocess_image(image)

        control = torch.from_numpy(control).unsqueeze(0)  # [1,C,H,W]
        control = self.preprocess_image(control)

        # Calculate embeddings for both encoders
        pooled_prompt_embeds, prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device_text_encoder=text_encoder_device,
            device_text_encoder_2=text_encoder_2_device,
            max_sequence_length=self.max_sequence_length,
        )

        empty_pooled_prompt_embeds, empty_prompt_embeds, _ = self.encode_prompt(
            prompt="",
            prompt_2="",
            device_text_encoder=text_encoder_device,
            device_text_encoder_2=text_encoder_2_device,
            max_sequence_length=self.max_sequence_length,
        )
        height_image, width_image = image.shape[2:]
        heigth_control, width_control = control.shape[2:]
        _, image_latents, _, _ = self.prepare_latents(
            image,
            1,
            16,
            height_image,
            width_image,
            self.weight_dtype,
            vae_encoder_device,
        )

        _, control_latents, _, _ = self.prepare_latents(
            control,
            1,
            16,
            heigth_control,
            width_control,
            self.weight_dtype,
            vae_encoder_device,
        )

        image_latents = image_latents[0].detach().cpu()
        control_latents = control_latents[0].detach().cpu()
        pooled_prompt_embeds = pooled_prompt_embeds[0].detach().cpu()
        prompt_embeds = prompt_embeds[0].detach().unsqueezecpu()
        empty_pooled_prompt_embeds = empty_pooled_prompt_embeds[0].detach().cpu()
        empty_prompt_embeds = empty_prompt_embeds[0].detach().cpu()
        # Save to cache (following QwenImageEditTrainer pattern)
        file_hashes = data["file_hashes"]

        # use same cache for all latents/embeddings
        self.cache_manager.save_cache("pixel_latent", file_hashes["prompt_hash"], image_latents)
        self.cache_manager.save_cache("control_latent", file_hashes["prompt_hash"], control_latents)
        self.cache_manager.save_cache("pooled_prompt_embed", file_hashes["prompt_hash"], pooled_prompt_embeds)
        self.cache_manager.save_cache("prompt_embed", file_hashes["prompt_hash"], prompt_embeds)
        self.cache_manager.save_cache(
            "empty_pooled_prompt_embed",
            file_hashes["prompt_hash"],
            empty_pooled_prompt_embeds
        )
        self.cache_manager.save_cache("empty_prompt_embed", file_hashes["prompt_hash"], empty_prompt_embeds)

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

    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image for Flux Kontext VAE.
        image: RGB tensor input [B,C,H,W]
        Logic is
        1. Resize it make it devisible by 16
        2. Input should be RGB format
        3. Convert to [0,1] range by devide 255
        4. Input shape should be [B,3,H,W]
        """
        # get proper image shape
        h, w = image.shape[2:]
        h = h // 16 * 16
        w = w // 16 * 16
        image = image.to(self.weight_dtype)
        image = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
        image = image/255.0
        # image = image.astype(self.weight_dtype)
        image = image.to(self.weight_dtype)
        return image

    def get_clip_prompt_embeds(self, prompt: str, device: str, )-> torch.Tensor:
        """
        Get CLIP prompt embeddings.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)
        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.view(batch_size, -1)
        return prompt_embeds

    def get_t5_prompt_embeds(self, prompt: str, device: str, max_sequence_length: int = 512) -> torch.Tensor:
        dtype = self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]
        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            prompt_2: Optional[Union[str, List[str]]] = None,
            device_text_encoder: Optional[torch.device] = None,
            device_text_encoder_2: Optional[torch.device] = None,
            max_sequence_length: int = 512) -> List[torch.Tensor]:
        """
        Encode prompts using both CLIP and T5 encoders.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
        # We only use the pooled prompt output from the CLIPTextModel
        with torch.inference_mode():
            pooled_prompt_embeds = self.get_clip_prompt_embeds(
                prompt=prompt,
                device=device_text_encoder,
            )
            prompt_embeds = self.get_t5_prompt_embeds(
                prompt=prompt_2,
                max_sequence_length=max_sequence_length,
                device=device_text_encoder_2,
            )
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device_text_encoder_2, dtype=self.weight_dtype)
        return pooled_prompt_embeds, prompt_embeds, text_ids

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)

    def prepare_latents(
        self,
        image: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
    ):

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height, width = image.shape[2:]
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)

        image_latents = image_ids = None

        image = image.to(device=device, dtype=dtype)
        with torch.inference_mode():
            image_latents = self.encode_vae_image(image=image)

        image_latent_height, image_latent_width = image_latents.shape[2:]
        image_latents = self._pack_latents(
            image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width
        )
        print('image latent shape after pack', image_latents.shape)
        image_ids = self._prepare_latent_image_ids(
            batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
        )
        print('image_ids',image_ids.shape, image_ids)
        # image ids are the same as latent ids with the first dimension set to 1 instead of 0
        image_ids[..., 0] = 1  # for reference image ids
        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        latents = randn_tensor(shape, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, image_latents, latent_ids, image_ids

    def encode_vae_image(self, image: torch.Tensor):
        image_latents = self.vae.encode(image)
        image_latents = image_latents.latent_dist.mode()
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return image_latents

    def predict(
        self,
        prompt_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[None, str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        weight_dtype: torch.dtype = torch.bfloat16,
        true_cfg_scale: float = 1.0,
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
            guidance_scale: Classifier-free guidance scale for guided distilled model
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
        self.weight_dtype = weight_dtype
        if not hasattr(self, "vae") or self.vae is None:
            logging.info("Loading model...")
            self.load_model()
        # 1. Process input format
        # convert imgage to tensor
        if not isinstance(prompt_image, list):
            prompt_image = [prompt_image]
        image = []
        for img in prompt_image:
            if isinstance(img, PIL.Image.Image):
                img = np.array(img).astype(np.float32)
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img.astype(np.float32))
            img = img.permute(2, 0, 1)  # [H,W,C] ->  [C,H,W]
            image.append(img)
        image = torch.stack(image, dim=0)
        image = self.preprocess_image(image)

        batch_size = image.shape[0]
        if isinstance(prompt, str):
            prompt = [prompt] * batch_size
        elif len(prompt) != batch_size:
            raise ValueError(f"Number of prompts ({len(prompt)}) must match number of " +
                             f"images ({batch_size})")

        # 2. Calculate image dimensions
        height, width = image.shape[2:]  # use original image size

        # 3. Get device configurations
        device_vae = self.config.predict.devices.get('vae', 'cuda:0')
        device_text_encoder = self.config.predict.devices.get('text_encoder', 'cuda:0')
        device_text_encoder_2 = self.config.predict.devices.get('text_encoder_2', 'cuda:0')
        device_transformer = self.config.predict.devices.get('transformer', 'cuda:0')

        logging.info(f"Using devices - VAE: {device_vae}, Text Encoders: {device_text_encoder}/" +
                     f"{device_text_encoder_2}, Transformer: {device_transformer}")

        self.vae = self.vae.to(device_vae)
        self.text_encoder = self.text_encoder.to(device_text_encoder)
        self.text_encoder_2 = self.text_encoder_2.to(device_text_encoder_2)
        self.transformer = self.transformer.to(device_transformer)

        # 4. Encode prompts (dual encoder: CLIP + T5)
        pooled_prompt_embeds, prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=negative_prompt,
            device_text_encoder=device_text_encoder,
            device_text_encoder_2=device_text_encoder_2,
            max_sequence_length=self.max_sequence_length,
        )
        if true_cfg_scale > 1.0 and negative_prompt is not None:
            (
                negative_pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                device_text_encoder=device_text_encoder,
                device_text_encoder_2=device_text_encoder_2,
                max_sequence_length=self.max_sequence_length,
            )

        # 5. Prepare latents
        print('image', image.shape)
        print('batch_size', batch_size)
        print('height', height)
        print('width', width)
        print('self.weight_dtype', self.weight_dtype)
        print('device_vae', device_vae)
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
            image,
            batch_size,
            16,
            height,
            width,
            self.weight_dtype,
            device_vae,
        )
        print('shape of latents', latents.shape)
        print('shape of image_latents', image_latents.shape)
        print('shape of latent_ids', latent_ids.shape)
        print('shape of image_ids', image_ids.shape)

        latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

        # 6. timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        # sigmas [1.   0.95 0.9  0.85 0.8  0.75 0.7  0.65 0.6  0.55 0.5  0.45 0.4  0.35
        #  0.3  0.25 0.2  0.15 0.1  0.05]
        # image_seq_len 4081
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        # mu: 1.1474609375
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device_transformer,
            sigmas=sigmas,
            mu=mu,
        )

        # 7. Guidance
        if guidance_scale > 1.0 and self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device_transformer, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 8. Denoising loop
        # # move all tensors to device_transformer
        latents = latents.to(device_transformer)
        guidance = guidance.to(device_transformer)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device_transformer)
        prompt_embeds = prompt_embeds.to(device_transformer)
        text_ids = text_ids.to(device_transformer)
        if true_cfg_scale > 1.0 and negative_prompt is not None:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device_transformer)
            negative_prompt_embeds = negative_prompt_embeds.to(device_transformer)
            negative_text_ids = negative_text_ids.to(device_transformer)
        with torch.inference_mode():
            for _, t in enumerate(tqdm(timesteps, total=num_inference_steps, desc="Flux Kontext Generation")):
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                # latent_model_input torch.Size([1, 8137, 64])
                # latent_model_input shape torch.Size([1, 183040, 64])
                # timestep tensor([1000.], device='cuda:0', dtype=torch.bfloat16)
                # guidance None
                # pooled_prompt_embeds torch.Size([1, 768])
                # prompt_embeds torch.Size([1, 512, 4096])
                # text_ids torch.Size([512, 3])
                # latent_ids torch.Size([183040, 3])
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]
                # noise_pred torch.Size([1, 8137, 64])
                noise_pred = noise_pred[:, : latents.size(1)]
                # noise_pred after choose first 4081 torch.Size([1, 4081, 64])
                if true_cfg_scale > 1.0 and negative_prompt is not None:
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        joint_attention_kwargs={},
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        # latents after unpack torch.Size([1, 16, 106, 154])
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type='pil')
        return image

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
            self.vae.decoder.to('cpu')
            self.text_encoder.to(self.accelerator.device)
            self.text_encoder_2.to(self.accelerator.device)
            self.transformer.to(self.accelerator.device)

        elif mode == "cache":
            # Cache mode: need encoders, don't need transformer
            self.vae = self.vae.to(self.config.cache.vae_encoder_device, non_blocking=True)
            self.vae.decoder.to('cpu')
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

    def save_lora(self, save_path):
        """Save LoRA weights"""
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod

        lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_transformer)
        )
        # Use FluxKontextPipeline's save method if available, otherwise use generic method
        FluxKontextPipeline.save_lora_weights(
            save_path, lora_state_dict, safe_serialization=True
            )
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
