"""
Flux Kontext LoRA Trainer Implementation
Following QwenImageEditTrainer patterns with dual text encoder support.
"""

import copy
import os
import json
import torch
import PIL
import random
import gc
import torch.nn.functional as F
import inspect

import numpy as np
from typing import Optional, Union, List
from tqdm.auto import tqdm
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils.torch_utils import randn_tensor
from peft.utils import get_peft_model_state_dict

from diffusers import FluxKontextPipeline
from src.models.flux_kontext_loader import (
    load_flux_kontext_vae,
    load_flux_kontext_clip,
    load_flux_kontext_t5,
    load_flux_kontext_transformer,
    load_flux_kontext_tokenizers,
    load_flux_kontext_scheduler,
)
from src.loss.edit_mask_loss import map_mask_to_latent
from src.utils.images import resize_bhw
from src.trainer.base_trainer import BaseTrainer
import logging


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
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
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
        self.vae = None  # FluxVAE
        self.text_encoder = None  # CLIP text encoder
        self.text_encoder_2 = None  # T5 text encoder
        self.dit = None  # Flux transformer
        self.tokenizer = None  # CLIP tokenizer
        self.tokenizer_2 = None  # T5 tokenizer
        self.scheduler = None  # FlowMatchEulerDiscreteScheduler

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

    def load_model(self):
        """
        Load and separate components from FluxKontextPipeline.
        Follows QwenImageEditTrainer.load_model() pattern exactly.
        """
        logging.info("Loading FluxKontextPipeline and separating components...")

        # Separate individual components using flux_kontext_loader
        self.vae = load_flux_kontext_vae(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        )
        self.text_encoder = load_flux_kontext_clip(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        )
        self.text_encoder_2 = load_flux_kontext_t5(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        )
        self.dit = load_flux_kontext_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        )

        # Load tokenizers and scheduler
        self.tokenizer, self.tokenizer_2 = load_flux_kontext_tokenizers(
            self.config.model.pretrained_model_name_or_path
        )
        self.scheduler = load_flux_kontext_scheduler(
            self.config.model.pretrained_model_name_or_path
        )

        # Set VAE-related parameters (following QwenImageEditTrainer pattern)
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if getattr(self, "vae", None)
            else 8
        )

        # Set Flux-specific VAE configuration
        self.vae_z_dim = (
            self.vae.config.latent_channels if getattr(self, "vae", None) else 16
        )

        # Additional Flux-specific attributes
        self.latent_channels = self.vae_z_dim

        # Set models to training/evaluation mode (same as QwenImageEditTrainer)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)

        if getattr(self, "transformer", None):
            self.num_channels_latents = self.dit.config.in_channels // 4
        else:
            self.num_channels_latents = 16
        self.tokenizer_max_length = 77  # clip encoder maximal token length
        self.max_sequence_length = 512  # T5 encoder maximal token length

        torch.cuda.empty_cache()

        from diffusers.image_processor import VaeImageProcessor

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )

        logging.info(
            f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}"
        )

    def save_lora(self, save_path):
        """Save LoRA weights"""
        unwrapped_transformer = self.accelerator.unwrap_model(self.dit)
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

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        device_text_encoder: Optional[torch.device] = None,
        device_text_encoder_2: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.Tensor]:
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
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
                device=device_text_encoder_2, dtype=self.weight_dtype
            )
        return pooled_prompt_embeds, prompt_embeds, text_ids

    def prepare_latents(
        self,
        image: Optional[torch.Tensor],
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ):

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height, width = image.shape[2:]
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        device = next(self.vae.parameters()).device
        image = image.to(device=device, dtype=dtype)
        with torch.inference_mode():
            image_latents = self.encode_vae_image(image=image)
        image_latent_height, image_latent_width = image_latents.shape[2:]
        image_latents = self._pack_latents(
            image_latents,
            batch_size,
            num_channels_latents,
            image_latent_height,
            image_latent_width,
        )
        image_ids = self._prepare_latent_image_ids(
            batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype
        )
        # image ids are the same as latent ids with the first dimension set to 1 instead of 0
        image_ids[..., 0] = 1  # for reference image ids
        latent_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )
        latents = randn_tensor(shape, device=device, dtype=dtype)
        latents = self._pack_latents(
            latents, batch_size, num_channels_latents, height, width
        )
        return latents, image_latents, latent_ids, image_ids

    def setup_model_device_train_mode(self, stage='fit', cache=False):
        """Set model device allocation and train mode."""
        if stage == "train":
            assert hasattr(
                self, "accelerator"
            ), "accelerator must be set before setting model devices"

        if self.cache_exist and self.use_cache and stage == "fit":
            # Cache mode: only need transformer
            self.text_encoder.cpu()
            self.text_encoder_2.cpu()
            torch.cuda.empty_cache()
            self.vae.cpu()
            torch.cuda.empty_cache()
            del self.text_encoder
            del self.text_encoder_2

            if not self.config.logging.sampling.enable:
                del self.vae
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

        elif stage == "train":
            # Non-cache mode: need all encoders
            self.vae.to(self.accelerator.device)
            self.text_encoder.to(self.accelerator.device)
            self.text_encoder_2.to(self.accelerator.device)
            self.dit.to(self.accelerator.device)
            self.vae.decoder.to("cpu")

            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.text_encoder_2.requires_grad_(False).eval()
            self.dit.requires_grad_(False)
            self.dit.train()
            for name, param in self.dit.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif stage == "cache":
            # Cache mode: need encoders, don't need transformer
            self.vae = self.vae.to(
                self.config.cache.devices.vae, non_blocking=True
            )
            self.vae.decoder.to("cpu")
            self.text_encoder = self.text_encoder.to(
                self.config.cache.devices.text_encoder, non_blocking=True
            )
            self.text_encoder_2 = self.text_encoder_2.to(
                self.config.cache.devices.text_encoder_2,
                non_blocking=True,
            )

            torch.cuda.synchronize()
            self.dit.cpu()
            torch.cuda.empty_cache()
            del self.dit
            gc.collect()
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.text_encoder_2.requires_grad_(False).eval()

        elif stage == "predict":
            # Predict mode: allocate to different GPUs according to configuration
            devices = self.config.predict.devices
            self.vae.to(devices.vae)
            self.text_encoder.to(devices.text_encoder)
            self.text_encoder_2.to(devices.text_encoder_2)
            self.dit.to(devices.dit)
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.text_encoder_2.requires_grad_(False).eval()
            self.dit.requires_grad_(False).eval()

    def prepare_embeddings(self, batch, stage='fit'):
        # for predict, not 'image' key, but 'pixel_latent' key
        # torch.tensor of image, control [B,C,H,W], in range [0,1]
        if 'image' in batch:
            batch['image'] = self.normalize_image(batch['image'])

        if 'control' in batch:
            batch['control'] = self.normalize_image(batch['control'])

        for i in range(100):
            additional_control_key = f'control_{i}'
            if additional_control_key in batch:
                batch[additional_control_key] = self.normalize_image(batch[additional_control_key])
            else:
                num_additional_controls = i
                break

        if 'prompt_2' in batch:
            prompt_2 = batch['prompt_2']
        else:
            prompt_2 = batch['prompt']

        pooled_prompt_embeds, prompt_embeds, text_ids = self.encode_prompt(
                prompt=batch['prompt'],
                prompt_2=prompt_2,
                max_sequence_length=self.max_sequence_length,
            )
        batch['pooled_prompt_embeds'] = pooled_prompt_embeds
        batch['prompt_embeds'] = prompt_embeds
        batch['text_ids'] = text_ids

        if 'negative_prompt' in batch:
            if 'negative_prompt_2' in batch:
                prompt_2 = batch['negative_prompt_2']
            else:
                prompt_2 = batch['negative_prompt']
            negative_pooled_prompt_embeds, negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=batch['negative_prompt'],
                prompt_2=prompt_2,
                max_sequence_length=self.max_sequence_length,
            )
            batch['negative_pooled_prompt_embeds'] = negative_pooled_prompt_embeds
            batch['negative_prompt_embeds'] = negative_prompt_embeds
            batch['negative_text_ids'] = negative_text_ids

        if 'image' in batch:
            image = batch['image']
            batch_size = image.shape[0]
            image_height, image_width = image.shape[2:]
            latents, image_latents, latent_ids, image_ids = self.prepare_latents(
                image,
                batch_size,
                16,
                image_height,
                image_width,
                self.weight_dtype,
                self.accelerator.device,
            )

            batch['image_latents'] = image_latents
            batch['image_ids'] = image_ids
            if stage == 'predict':
                batch['latents'] = latents
                batch['latent_ids'] = latent_ids

        if 'control' in batch:
            control = batch['control']
            batch_size = control.shape[0]
            control_height, control_width = control.shape[2:]
            _, control_latents, _, control_ids = self.prepare_latents(
                control,
                batch_size,
                16,
                control_height,
                control_width,
                self.weight_dtype,
                self.accelerator.device,
            )
            control_ids[..., 0] = 1
            batch['control_latents'] = control_latents
            batch['control_ids'] = control_ids

        for i in range(1, num_additional_controls+1):
            control_key = f'control_{i}'
            control = batch[control_key]
            batch_size = control.shape[0]
            control_height, control_width = control.shape[2:]
            _, control_latents, _, control_ids = self.prepare_latents(
                control,
                batch_size,
                16,
                control_height,
                control_width,
                self.weight_dtype,
                self.accelerator.device,
            )
            control_ids[..., 0] = i + 1
            batch[f'control_{i}_latents'] = control_latents
            batch[f'control_{i}_ids'] = control_ids
        if self.config.loss.mask_loss and 'mask' in batch:
            mask = batch['mask']
            batch['mask'] = resize_bhw(mask, image_height, image_width)
            batch['mask'] = map_mask_to_latent(batch['mask'])
        return batch

    def prepare_cached_embeddings(self, batch):
        if self.config.loss.mask_loss and 'mask' in batch:
            image_height, image_width = batch['image'].shape[2:]
            mask = batch['mask']
            batch['mask'] = resize_bhw(mask, image_height, image_width)
            batch['mask'] = map_mask_to_latent(batch['mask'])
        return batch

    def cache(self, train_dataloader):
        """
        Pre-compute and cache embeddings (exactly same signature as QwenImageEditTrainer).
        Implements dual text encoder caching for CLIP + T5.
        """
        vae_encoder_device = self.config.cache.vae_encoder_device
        text_encoder_device = self.config.cache.text_encoder_device
        text_encoder_2_device = self.config.cache.text_encoder_2_device

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
            self.cache_step(
                data, vae_encoder_device, text_encoder_device, text_encoder_2_device
            )

        logging.info("Cache completed")

        # Clean up models (same as QwenImageEditTrainer)
        self.text_encoder.cpu()
        self.text_encoder_2.cpu()
        self.vae.cpu()
        del self.text_encoder
        del self.text_encoder_2
        del self.vae

    def cache_step(
        self,
        data: dict,
        vae_encoder_device: str,
        text_encoder_device: str,
        text_encoder_2_device: str,
    ):
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
        prompt_embeds = prompt_embeds[0].detach().cpu()
        empty_pooled_prompt_embeds = empty_pooled_prompt_embeds[0].detach().cpu()
        empty_prompt_embeds = empty_prompt_embeds[0].detach().cpu()
        shape_info = torch.tensor([height_image, width_image, heigth_control, width_control], dtype=torch.int32)

        cache_embeddings = {
            'pixel_latent': image_latents,
            'control_latent': control_latents,
            'pooled_prompt_embed': pooled_prompt_embeds,
            'prompt_embed': prompt_embeds,
            'empty_pooled_prompt_embed': empty_pooled_prompt_embeds,
            'empty_prompt_embed': empty_prompt_embeds,
            'shape_info': shape_info,
        }

        map_keys = {
            'pixel_latent', 'image_hash',
            'control_latent', 'control_hash',
            'pooled_prompt_embed', 'prompt_hash',
            'prompt_embed', 'prompt_hash',
            'empty_pooled_prompt_embed', 'prompt_hash',
            'empty_prompt_embed', 'prompt_hash',
            'shape_info', 'image_hash',
        }

        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    def fit(self, train_dataloader):
        logging.info("Starting training process...")

        # Setup components
        self.setup_accelerator()
        self.load_model()
        if self.config.train.resume_from_checkpoint is not None:
            # add the checkpoint in lora.pretrained_weight config
            self.config.model.lora.pretrained_weight = os.path.join(
                self.config.train.resume_from_checkpoint, "model.safetensors"
            )
            logging.info(
                f"Loaded checkpoint from {self.config.model.lora.pretrained_weight}"
            )

        self.load_pretrain_lora_model(self.dit, self.config, self.adapter_name)
        self.text_encoder.requires_grad_(False).eval()
        self.text_encoder_2.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.set_model_devices(mode="train")

        self.dit.requires_grad_(False)
        self.dit.train()

        # Train only LoRA parameters
        trainable_params = 0
        total_params = 0
        for name, param in self.dit.named_parameters():
            total_params += param.numel()
            if "lora" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        logging.info(
            f"Trainable/Total parameters: {trainable_params / 1e6:.2f}M / {total_params / 1e9:.2f}B"
        )

        self.configure_optimizers()

        self.set_criterion()

        train_dataloader = self.accelerator_prepare(train_dataloader)

        # 添加validation sampling setup (如果启用)
        validation_sampler = None
        if hasattr(self.config.logging, 'sampling') and self.config.logging.sampling.enable:
            from src.validation.validation_sampler import ValidationSampler

            validation_sampler = ValidationSampler(
                config=self.config.logging.sampling,
                accelerator=self.accelerator,
                weight_dtype=self.weight_dtype
            )

            # Setup validation dataset
            validation_sampler.setup_validation_dataset(train_dataloader.dataset)

            # Cache embeddings for validation using trainer methods
            embeddings_config = {
                'cache_vae_embeddings': True,      # Cache VAE latents
                'cache_text_embeddings': True,     # Cache text embeddings
            }
            validation_sampler.cache_embeddings(self, embeddings_config)

            logging.info("Validation sampling setup completed")

        logging.info("***** Running training *****")
        logging.info(f"  Instantaneous batch size per device = {self.batch_size}")
        logging.info(
            f"  Gradient Accumulation steps = {self.config.train.gradient_accumulation_steps}"
        )
        logging.info(f"  Use cache: {self.use_cache}, Cache exists: {self.cache_exist}")
        if validation_sampler:
            logging.info(f"  Validation sampling enabled: every {self.config.logging.sampling.validation_steps} steps")

        # Training loop implementation (following QwenImageEditTrainer structure)
        # Progress bar

        # Training loop
        train_loss = 0.0
        running_loss = 0.0
        if self.config.train.resume_from_checkpoint is not None:
            with open(
                os.path.join(self.config.train.resume_from_checkpoint, "state.json")
            ) as f:
                st = json.load(f)
            self.global_step = st["global_step"]
            start_epoch = st["epoch"]
        else:
            self.global_step = 0
            start_epoch = 0
        # Progress bar
        progress_bar = tqdm(
            range(self.global_step, self.config.train.max_train_steps),
            desc="train",
            disable=not self.accelerator.is_local_main_process,
        )
        for epoch in range(start_epoch, self.config.train.num_epochs):
            for _, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.dit):
                    loss = self.training_step(batch)

                    # Backward pass
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.dit.parameters(),
                            self.config.train.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Update when syncing gradients
                lr = self.lr_scheduler.get_last_lr()[0]
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    self.global_step += 1

                    # Calculate average loss
                    avg_loss = self.accelerator.gather(
                        loss.repeat(self.batch_size)
                    ).mean()
                    train_loss += (
                        avg_loss.item() / self.config.train.gradient_accumulation_steps
                    )
                    running_loss = train_loss

                    # Log metrics
                    self.accelerator.log(
                        {"train_loss": train_loss}, step=self.global_step
                    )
                    self.accelerator.log({"lr": lr}, step=self.global_step)
                    train_loss = 0.0

                    # Save checkpoint
                    if self.global_step % self.config.train.checkpointing_steps == 0:
                        self.save_checkpoint(epoch, self.global_step)

                    # 添加validation sampling
                    if validation_sampler and validation_sampler.should_run_validation(self.global_step):
                        try:
                            validation_sampler.run_validation_loop(
                                global_step=self.global_step,
                                trainer=self  # 传入trainer实例
                            )
                        except Exception as e:
                            self.accelerator.print(f"Validation sampling failed: {e}")

                # Update progress bar
                logs = {
                    "loss": f"{running_loss:.3f}",
                    "lr": f"{lr:.1e}",
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
        if self.use_cache and self.cache_exist:
            return self._training_step_cached(batch)
        else:
            return self._training_step_compute(batch)

    def _training_step_cached(self, batch):
        """Training step using cached embeddings (follows QwenImageEditTrainer pattern)."""
        pixel_latents = batch["pixel_latent"].to(
            self.accelerator.device, dtype=self.weight_dtype
        )
        control_latents = batch["control_latent"].to(
            self.accelerator.device, dtype=self.weight_dtype
        )

        # Dual encoder embeddings (Flux-specific)
        pooled_prompt_embed = batch["pooled_prompt_embed"].to(
            self.accelerator.device, dtype=self.weight_dtype
        )
        prompt_embed = batch["prompt_embed"].to(
            self.accelerator.device, dtype=self.weight_dtype
        )

        image = batch["image"]  # torch.tensor: B,C,H,W
        control = batch["control"]  # torch.tensor: B,C,H,W
        image = self.preprocess_image(image)
        control = self.preprocess_image(control)
        image_height, image_width = image.shape[2:]
        control_height, control_width = control.shape[2:]

        if self.config.loss.mask_loss:
            edit_mask = batch["mask"]  # torch.tensor: B,H,W
            edit_mask = resize_bhw(edit_mask, image_height, image_width)
            edit_mask = map_mask_to_latent(edit_mask)
        else:
            edit_mask = None

        return self._compute_loss(
            pixel_latents,
            control_latents,
            prompt_embed,
            pooled_prompt_embed,
            image_width=image_width,
            image_height=image_height,
            control_width=control_width,
            control_height=control_height,
            edit_mask=edit_mask,
        )

    def _training_step_compute(self, batch):
        """Training step with embedding computation (no cache)"""
        # Similar to QwenImageEditTrainer but with dual encoders
        image, control, prompt = batch["image"], batch["control"], batch["prompt"]
        # torch.tensor of image, control [B,C,H,W], could be different shape
        image = self.preprocess_image(image)
        control = self.preprocess_image(control)
        image_height, image_width = image.shape[2:]
        control_height, control_width = control.shape[2:]

        batch_size = image.shape[0]

        # random drop prompt to ""
        if self.config.data.init_args.get("caption_dropout_rate", 0.0) > 0:
            prompt = [
                (
                    ""
                    if random.random()
                    < self.config.data.init_args.get("caption_dropout_rate", 0.0)
                    else p
                )
                for p in prompt
            ]

        pooled_prompt_embeds, prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            device_text_encoder=self.accelerator.device,
            device_text_encoder_2=self.accelerator.device,
            max_sequence_length=self.max_sequence_length,
        )

        _, image_latents, _, _ = self.prepare_latents(
            image,
            batch_size,
            16,
            image_height,
            image_width,
            self.weight_dtype,
            self.accelerator.device,
        )

        _, control_latents, _, _ = self.prepare_latents(
            control,
            batch_size,
            16,
            control_height,
            control_width,
            self.weight_dtype,
            self.accelerator.device,
        )

        return self._compute_loss(
            image_latents,
            control_latents,
            prompt_embeds,
            pooled_prompt_embeds,
            image_width=image_width,
            image_height=image_height,
            control_width=control_width,
            control_height=control_height,
            edit_mask=None,
        )

    def _compute_loss(
        self,
        pixel_latents,
        control_latents,
        prompt_embeds,
        pooled_prompt_embeds,
        image_width: int = 0,
        image_height: int = 0,
        control_width: int = 0,
        control_height: int = 0,
        edit_mask=None,
    ) -> torch.Tensor:
        """Calculate the flow matching loss (same structure as QwenImageEditTrainer).
        args:
            image_width: width of the ground truth image, image to be generated. After preprocess
            image_height: height of the ground truth image
            control_width: width of the control image
            control_height: height of the control image
        """

        with torch.no_grad():
            batch_size = pixel_latents.shape[0]
            noise = torch.randn_like(
                pixel_latents, device=self.accelerator.device, dtype=self.weight_dtype
            )

            # # Sample timesteps
            # u = compute_density_for_timestep_sampling(
            #     weighting_scheme="none",
            #     batch_size=batch_size,
            #     logit_mean=0.0,
            #     logit_std=1.0,
            #     mode_scale=1.29,
            # )
            # indices = (u * self.scheduler.config.num_train_timesteps).long()
            # timesteps = self.scheduler.timesteps[indices].to(
            #     device=pixel_latents.device
            # )

            # sigmas = self._get_sigmas(
            #     timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype
            # )
            t = torch.rand((noise.shape[0],), device=self.accelerator.device, dtype=self.weight_dtype)  # random time t
            t_ = t.unsqueeze(1).unsqueeze(1)

            noisy_model_input = (1.0 - t_) * pixel_latents + t_ * noise
            # noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

            # prepare text ids
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
                device=self.accelerator.device, dtype=self.weight_dtype
            )

            # prepare laten_id and image_id

            image_width = int(image_width) // (self.vae_scale_factor * 2)
            image_height = int(image_height) // (self.vae_scale_factor * 2)

            latent_ids = self._prepare_latent_image_ids(
                batch_size,
                image_height,
                image_width,
                self.accelerator.device,
                self.weight_dtype,
            )

            control_width = int(control_width) // (self.vae_scale_factor * 2)
            control_height = int(control_height) // (self.vae_scale_factor * 2)
            image_ids = self._prepare_latent_image_ids(
                batch_size,
                control_height,
                control_width,
                self.accelerator.device,
                self.weight_dtype,
            )
            image_ids[..., 0] = 1  # for reference image ids

            # Prepare input for transformer
            latent_model_input = torch.cat([noisy_model_input, control_latents], dim=1)
            latent_ids = torch.cat(
                [latent_ids, image_ids], dim=0
            )  # dim 0 is sequence dimension

        # Prepare guidance
        guidance = (
            torch.ones((noise.shape[0],)).to(self.accelerator.device)
            if self.dit.config.guidance_embeds
            else None
        )
        # convert dtype to self.weight_dtype
        latent_model_input = latent_model_input.to(self.weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.weight_dtype)
        prompt_embeds = prompt_embeds.to(self.weight_dtype)
        guidance = guidance.to(self.weight_dtype)
        t = t.to(self.weight_dtype)

        model_pred = self.dit(
            hidden_states=latent_model_input,
            # timestep=timesteps / 1000,
            timestep=t,
            guidance=guidance,  # must pass to guidance for FluxKontextDev
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        # noise_pred torch.Size([1, 8137, 64])
        model_pred = model_pred[:, : pixel_latents.size(1)]

        # Calculate loss
        # weighting = compute_loss_weighting_for_sd3(
        #     weighting_scheme="none", sigmas=sigmas
        # )

        target = noise - pixel_latents

        loss = self.forward_loss(model_pred, target, weighting=None, edit_mask=edit_mask)
        return loss

    def forward_loss(self, model_pred, target, weighting=None, edit_mask=None):
        if edit_mask is None:
            if weighting is None:
                loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")
            else:
                loss = torch.mean(
                    (
                        weighting.float() * (model_pred.float() - target.float()) ** 2
                    ).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
        else:
            # shape torch.Size([4, 864, 1216]) torch.Size([4, 4104, 64]) torch.Size([4, 4104, 64]) torch.Size([4, 1, 1])
            loss = self.criterion(edit_mask, model_pred, target, weighting)
        return loss

    def _get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """Calculate sigma values for noise scheduler"""
        noise_scheduler_copy = copy.deepcopy(self.scheduler)
        sigmas = noise_scheduler_copy.sigmas.to(
            device=self.accelerator.device, dtype=dtype
        )
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
        3. Convert to [-1,1] range by devide 255
        4. Input shape should be [B,3,H,W]
        """
        # get proper image shape
        h, w = image.shape[2:]
        h = h // 16 * 16
        w = w // 16 * 16
        image = image.to(self.weight_dtype)
        if image.shape[2] != h or image.shape[3] != w:
            image = F.interpolate(
                image, size=(h, w), mode="bilinear", align_corners=False
            )
        image = image / 255.0
        # image = image.astype(self.weight_dtype)
        image = image.to(self.weight_dtype)
        image = 2.0 * image - 1.0
        return image

    def get_clip_prompt_embeds(self, prompt: str) -> torch.Tensor:
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
        device = next(self.text_encoder.parameters()).device
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )
        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.view(batch_size, -1)
        return prompt_embeds

    def get_t5_prompt_embeds(
        self, prompt: str, max_sequence_length: int = 512
    ) -> torch.Tensor:
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
        device = next(self.text_encoder_2.parameters()).device
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_2(
            text_input_ids.to(device), output_hidden_states=False
        )[0]
        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(
        latents: torch.Tensor,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )
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
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width)[None, :]
        )
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = (
            latent_image_ids.shape
        )
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)

    def encode_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        image_latents = self.vae.encode(image)
        image_latents = image_latents.latent_dist.mode()
        image_latents = (
            image_latents - self.vae.config.shift_factor
        ) * self.vae.config.scaling_factor
        return image_latents

    def setup_predict(self):
        if not hasattr(self, "vae") or self.vae is None:
            logging.info("Loading model...")
            self.load_model()

        self.guidance = 3.5
        #  Get device configurations
        device_vae = self.config.predict.devices.get("vae", "cuda:0")
        device_text_encoder = self.config.predict.devices.get("text_encoder", "cuda:0")
        device_text_encoder_2 = self.config.predict.devices.get(
            "text_encoder_2", "cuda:0"
        )
        device_transformer = self.config.predict.devices.get("transformer", "cuda:0")

        logging.info(
            f"Using devices - VAE: {device_vae}, Text Encoders: {device_text_encoder}/"
            + f"{device_text_encoder_2}, Transformer: {device_transformer}"
        )

        self.vae = self.vae.to(device_vae)
        self.text_encoder = self.text_encoder.to(device_text_encoder)
        self.text_encoder_2 = self.text_encoder_2.to(device_text_encoder_2)
        self.dit = self.dit.to(device_transformer)
        self.vae.eval()
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.dit.eval()
        if self.config.model.lora.pretrained_weight:
            self.load_pretrain_lora_model(self.dit, self.config, self.adapter_name)
        self.predict_setted = True

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        num_additional_controls = embeddings['num_additional_controls']
        num_inference_steps = embeddings['num_inference_steps']
        true_cfg_scale = embeddings['true_cfg_scale']
        latent_ids = []
        if 'latent_ids' in embeddings:
            latent_ids.append(embeddings['latent_ids'])
        if 'control_ids' in embeddings:
            latent_ids.append(embeddings['control_ids'])
        for i in range(1, num_additional_controls+1):
            control_ids = f'control_{i}_ids'
            if control_ids in embeddings:
                latent_ids.append(embeddings[control_ids])
        latent_ids = torch.cat(latent_ids, dim=0)
        batch_size = embeddings['latents'].shape[0]
        image_seq_len = embeddings['latents'].shape[1]
        timesteps, num_inference_steps = self.prepare_predict_timesteps(num_inference_steps, image_seq_len)
        dit_device = next(self.dit.parameters()).device
        guidance = torch.full(
            [1], self.guidance, device=dit_device, dtype=torch.float32
        )
        guidance = guidance.expand(batch_size)
        self.scheduler.set_begin_index(0)
        # move all tensors to dit_device
        latent_ids = latent_ids.to(dit_device)
        guidance = guidance.to(dit_device)
        pooled_prompt_embeds = embeddings['pooled_prompt_embeds'].to(dit_device).to(self.weight_dtype)
        prompt_embeds = embeddings['prompt_embeds'].to(dit_device).to(self.weight_dtype)
        text_ids = embeddings['text_ids'].to(dit_device).to(self.weight_dtype)
        latents = embeddings['latents'].to(dit_device).to(self.weight_dtype)
        if true_cfg_scale > 1.0 and 'negative_pooled_prompt_embeds' in embeddings:
            negative_pooled_prompt_embeds = embeddings['negative_pooled_prompt_embeds'].to(dit_device).to(self.weight_dtype)
            negative_prompt_embeds = embeddings['negative_prompt_embeds'].to(dit_device).to(self.weight_dtype)
            negative_text_ids = embeddings['negative_text_ids'].to(dit_device).to(self.weight_dtype)

        with torch.inference_mode():
            for _, t in enumerate(
                tqdm(
                    timesteps, total=num_inference_steps, desc="Flux Kontext Generation"
                )
            ):
            latent_model_input =[latents]
            if 'control_latents' in embeddings:
                latent_model_input.append(embeddings['control_latents'].to(dit_device).to(self.weight_dtype))
            for i in range(1, num_additional_controls+1):
                control_latents = f'control_{i}_latents'
                if control_latents in embeddings:
                    latent_model_input.append(embeddings[control_latents].to(dit_device).to(self.weight_dtype))
            latent_model_input = torch.cat(latent_model_input, dim=1)
            timestep = t.expand(batch_size).to(dit_device).to(self.weight_dtype)
            noise_pred = self.dit(
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
            noise_pred = noise_pred[:, : image_seq_len]
            if true_cfg_scale > 1.0 and 'negative_pooled_prompt_embeds' in embeddings:
                neg_noise_pred = self.dit(
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
                neg_noise_pred = neg_noise_pred[:, : image_seq_len]
                noise_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                )
            latents = self.scheduler.step(
                noise_pred, t, latents, return_dict=False
            )[0]
        return latents


    def decode_vae_latent(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        # latents after unpack torch.Size([1, 16, 106, 154])
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pt")
        return image

    def prepare_predict_batch_data(
        self,
        prompt_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Union[str, List[str]],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[None, str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 20,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        weight_dtype: torch.dtype = torch.bfloat16,
        true_cfg_scale: float = 1.0,
        **kwargs,
    ) -> dict:
        assert prompt_image is not None, "prompt_image is required"
        assert prompt is not None, "prompt is required"
        self.weight_dtype = weight_dtype
        if not isinstance(prompt_image, list):
            prompt_image = [prompt_image]
