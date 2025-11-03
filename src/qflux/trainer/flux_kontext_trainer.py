"""
Flux Kontext LoRA Trainer Implementation
Following QwenImageEditTrainer patterns with dual text encoder support.
"""

import gc
import logging
from typing import Any

import PIL
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from qflux.models.flux_kontext_loader import (
    load_flux_kontext_clip,
    load_flux_kontext_scheduler,
    load_flux_kontext_t5,
    load_flux_kontext_tokenizers,
    load_flux_kontext_transformer,
    load_flux_kontext_vae,
)
from qflux.trainer.base_trainer import BaseTrainer
from qflux.utils.images import make_image_devisible, make_image_shape_devisible
from qflux.utils.tools import pad_latents_for_multi_res


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

    def get_pipeline_class(self):
        return FluxKontextPipeline

    def load_model(self):
        """
        Load and separate components from FluxKontextPipeline.
        Follows QwenImageEditTrainer.load_model() pattern exactly.
        """
        logging.info("Loading FluxKontextPipeline and separating components...")

        # Separate individual components using flux_kontext_loader
        pretrains = self.config.model.pretrained_embeddings
        if pretrains is not None and "vae" in pretrains:
            self.vae = load_flux_kontext_vae(
                pretrains["vae"],
                weight_dtype=self.weight_dtype,
            ).to("cpu")
            logging.info(f"loaded vae from {pretrains['vae']}")
        else:
            self.vae = load_flux_kontext_vae(
                self.config.model.pretrained_model_name_or_path,
                weight_dtype=self.weight_dtype,
            ).to("cpu")
        if pretrains is not None and "text_encoder" in pretrains:
            self.text_encoder = load_flux_kontext_clip(
                pretrains["text_encoder"],
                weight_dtype=self.weight_dtype,
            ).to("cpu")
            logging.info(f"loaded text_encoder from {pretrains['text_encoder']}")
        else:
            self.text_encoder = load_flux_kontext_clip(
                self.config.model.pretrained_model_name_or_path,
                weight_dtype=self.weight_dtype,
            ).to("cpu")

        if pretrains is not None and "text_encoder_2" in pretrains:
            self.text_encoder_2 = load_flux_kontext_t5(
                pretrains["text_encoder_2"],
                weight_dtype=self.weight_dtype,
            ).to("cpu")
            logging.info(f"loaded text_encoder_2 from {pretrains['text_encoder_2']}")
        else:
            self.text_encoder_2 = load_flux_kontext_t5(
                self.config.model.pretrained_model_name_or_path,
                weight_dtype=self.weight_dtype,
            ).to("cpu")

        use_multi_resolution = (
            hasattr(self.config.data.init_args.processor.init_args, "multi_resolutions")
            and self.config.data.init_args.processor.init_args.multi_resolutions is not None
        )

        self.dit = load_flux_kontext_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
            use_multi_resolution=use_multi_resolution,
        ).to("cpu")

        # Load tokenizers and scheduler
        self.tokenizer, self.tokenizer_2 = load_flux_kontext_tokenizers(self.config.model.pretrained_model_name_or_path)
        # Create two schedulers: one for training, one for sampling/validation
        self.scheduler = load_flux_kontext_scheduler(self.config.model.pretrained_model_name_or_path)
        import copy

        self.sampling_scheduler = copy.deepcopy(self.scheduler)  # Independent scheduler for validation/sampling

        # Set VAE-related parameters (following QwenImageEditTrainer pattern)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8

        # Set Flux-specific VAE configuration
        self.vae_z_dim = self.vae.config.latent_channels if getattr(self, "vae", None) else 16

        # Additional Flux-specific attributes
        self.latent_channels = self.vae_z_dim

        # Set models to training/evaluation mode (same as QwenImageEditTrainer)
        self.text_encoder.requires_grad_(False).eval()
        self.text_encoder_2.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()

        if self.dit is not None:
            self.num_channels_latents = self.dit.config.in_channels // 4
        else:
            self.num_channels_latents = 16
        self.tokenizer_max_length = 77  # clip encoder maximal token length
        self.max_sequence_length = 512  # T5 encoder maximal token length

        torch.cuda.empty_cache()

        from diffusers.image_processor import VaeImageProcessor

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        device_text_encoder: torch.device | None = None,
        device_text_encoder_2: torch.device | None = None,
        max_sequence_length: int = 512,
    ) -> list[torch.Tensor]:
        """
        Encode prompts using both CLIP and T5 encoders.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt_2 = prompt_2 or prompt
        prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
        # We only use the pooled prompt output from the CLIPTextModel
        with torch.inference_mode():
            pooled_prompt_embeds = self.get_clip_prompt_embeds(
                prompt=prompt,  # type: ignore[arg-type]
            )
            prompt_embeds = self.get_t5_prompt_embeds(
                prompt=prompt_2,  # type: ignore[arg-type]
            )
            text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device_text_encoder_2, dtype=self.weight_dtype)
        return pooled_prompt_embeds, prompt_embeds, text_ids  # type: ignore[return-value]

    def prepare_latents(
        self,
        image: torch.Tensor | None,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        # height, width = image.shape[2:]
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        device = next(self.vae.parameters()).device
        if image is not None:
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
        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        latents = randn_tensor(shape, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, image_latents, latent_ids, image_ids

    def setup_model_device_train_mode(self, stage="fit", cache=False):
        """Set model device allocation and train mode."""
        if stage == "fit":
            assert hasattr(self, "accelerator"), "accelerator must be set before setting model devices"
        if self.cache_exist and self.use_cache and stage == "fit":
            # Cache mode: only need transformer
            self.text_encoder.cpu()
            self.text_encoder_2.cpu()
            self.text_encoder.requires_grad_(False).eval()
            self.text_encoder_2.requires_grad_(False).eval()
            torch.cuda.empty_cache()
            self.vae.cpu()
            torch.cuda.empty_cache()
            # del self.text_encoder
            # del self.text_encoder_2

            if not self.config.validation.enabled:
                del self.vae
                del self.text_encoder
                del self.text_encoder_2
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
            print("dit device", self.dit.device)
            import time

            time.sleep(10)

        elif stage == "fit":
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
            self.vae = self.vae.to(self.config.cache.devices.vae, non_blocking=True)
            print("self.config.cache.devices.vae", self.config.cache.devices.vae)
            print("vae device ", next(self.vae.parameters()).device)

            self.vae.decoder.to("cpu")
            print("vae device ", next(self.vae.parameters()).device)

            self.text_encoder = self.text_encoder.to(self.config.cache.devices.text_encoder, non_blocking=True)
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
            logging.info("cache mode device setting")
            print("vae device ", next(self.vae.parameters()).device)

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

    def prepare_embeddings(self, batch, stage="fit"):
        # for predict, not 'image' key, but 'pixel_latent' key
        # torch.tensor of image, control [B,C,H,W], in range [0,1]
        # for predict: add extra latent_ids, latents, guidance
        # for cache: add empty_pooled_prompt_embeds, empty_prompt_embeds
        if "image" in batch:
            batch["image"] = self.normalize_image(batch["image"])

        if "control" in batch:
            batch["control"] = self.normalize_image(batch["control"])

        num_additional_controls = (
            batch["n_controls"] if isinstance(batch["n_controls"], int) else batch["n_controls"][0]
        )

        for i in range(num_additional_controls):
            additional_control_key = f"control_{i + 1}"
            if additional_control_key in batch:
                batch[additional_control_key] = self.normalize_image(batch[additional_control_key])

        if "prompt_2" in batch:
            prompt_2 = batch["prompt_2"]
        else:
            prompt_2 = batch["prompt"]

        pooled_prompt_embeds, prompt_embeds, text_ids = self.encode_prompt(
            prompt=batch["prompt"],
            prompt_2=prompt_2,
            max_sequence_length=self.max_sequence_length,
        )
        batch["pooled_prompt_embeds"] = pooled_prompt_embeds
        batch["prompt_embeds"] = prompt_embeds
        batch["text_ids"] = text_ids

        if stage == "cache":
            pooled_prompt_embeds, prompt_embeds, _ = self.encode_prompt(
                prompt=[""],
                prompt_2=None,
                max_sequence_length=self.max_sequence_length,
            )
            batch["empty_pooled_prompt_embeds"] = pooled_prompt_embeds
            batch["empty_prompt_embeds"] = prompt_embeds

        if "negative_prompt" in batch:
            if "negative_prompt_2" in batch:
                prompt_2 = batch["negative_prompt_2"]
            else:
                prompt_2 = batch["negative_prompt"]
            negative_pooled_prompt_embeds, negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=batch["negative_prompt"],
                prompt_2=prompt_2,
                max_sequence_length=self.max_sequence_length,
            )
            batch["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            batch["negative_prompt_embeds"] = negative_prompt_embeds
            batch["negative_text_ids"] = negative_text_ids
        if "image" in batch:
            # get latent for target image
            image = batch["image"]  # single images
            batch_size = image.shape[0]
            image_height, image_width = image.shape[2:]
            # device = next(self.vae.parameters()).device
            latents, image_latents, latent_ids, image_ids = self.prepare_latents(
                image,
                batch_size,
                16,
                image_height,
                image_width,
                self.weight_dtype,
            )

            batch["image_latents"] = image_latents

        if "control" in batch:
            # get latent for first control
            control = batch["control"]
            batch_size = control.shape[0]

            control_height, control_width = control.shape[2:]
            _, control_latents, _, control_ids = self.prepare_latents(
                control,
                batch_size,
                16,
                control_height,
                control_width,
                self.weight_dtype,
            )
            control_ids[..., 0] = 1
            batch["control_latents"] = [control_latents]
            batch["control_ids"] = [control_ids]

        for i in range(1, num_additional_controls + 1):
            # get latent for additional controls
            control_key = f"control_{i}"
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
            )
            control_ids[..., 0] = i + 1
            batch["control_latents"].append(control_latents)
            batch["control_ids"].append(control_ids)

        # Only concat control_latents and control_ids in fit/cache mode
        # In predict mode with multi-resolution, keep them as lists for later processing
        if "control_latents" in batch:
            batch["control_latents"] = torch.cat(batch["control_latents"], dim=1)
            batch["control_ids"] = torch.cat(batch["control_ids"], dim=0)

        return batch

    def cache_step(
        self,
        data: dict,
    ):
        """
        cache image embedding and vae embedding.
        which is calculated in prepare_embeddings()
        """
        image_latents = data["image_latents"].detach().cpu()
        control_latents = data["control_latents"].detach().cpu()
        pooled_prompt_embeds = data["pooled_prompt_embeds"].detach().cpu()
        prompt_embeds = data["prompt_embeds"].detach().cpu()
        empty_pooled_prompt_embeds = data["empty_pooled_prompt_embeds"].detach().cpu()
        empty_prompt_embeds = data["empty_prompt_embeds"].detach().cpu()
        text_ids = data["text_ids"].detach().cpu()
        # image_ids = data["image_ids"].detach().cpu()
        control_ids = data["control_ids"].detach().cpu()
        cache_embeddings = {
            "image_latents": image_latents[0],
            "control_latents": control_latents[0],
            "pooled_prompt_embeds": pooled_prompt_embeds[0],
            "prompt_embeds": prompt_embeds[0],
            "empty_pooled_prompt_embeds": empty_pooled_prompt_embeds[0],
            "empty_prompt_embeds": empty_prompt_embeds[0],
            "control_ids": control_ids,
            "text_ids": text_ids,
        }
        map_keys = {
            "image_latents": "image_hash",
            "control_latents": "control_hash",
            "pooled_prompt_embeds": "prompt_hash",
            "prompt_embeds": "prompt_hash",
            "empty_pooled_prompt_embeds": "prompt_hash",
            "empty_prompt_embeds": "prompt_hash",
            "control_ids": "control_hash",
            "text_ids": "prompt_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    def prepare_cached_embeddings(self, batch):
        batch["control_ids"] = batch["control_ids"][0]  # remove batch dim
        batch["text_ids"] = batch["text_ids"][0]  # remove batch dim
        return batch

    def _compute_loss(self, embeddings) -> torch.Tensor:
        """
        Compute training loss with support for both shared and multi-resolution modes.

        The method automatically detects whether to use multi-resolution mode based on:
        1. Configuration: multi_resolutions must be configured
        2. Batch structure: batch_size > 1 and samples have different resolutions

        For multi-resolution mode, it applies padding and attention masking to handle
        variable-length sequences efficiently.
        """
        # Check if multi-resolution mode should be used
        use_multi_res = self.should_use_multi_resolution_mode(embeddings)

        if use_multi_res:
            return self._compute_loss_multi_resolution_mode(embeddings)
        else:
            return self._compute_loss_shared_mode(embeddings)

    def _compute_loss_shared_mode(self, embeddings, return_pred=False) -> torch.Tensor:
        """Original loss computation for shared resolution batches"""
        image_latents = embeddings["image_latents"]
        text_ids = embeddings["text_ids"]
        control_latents = embeddings["control_latents"]
        control_ids = embeddings["control_ids"]
        pooled_prompt_embeds = embeddings["pooled_prompt_embeds"]
        prompt_embeds = embeddings["prompt_embeds"]
        assert self.accelerator is not None, "accelerator is not set"
        device = self.accelerator.device
        image_height, image_width = embeddings["image"].shape[2:]
        # move to self.dit device
        image_latents = image_latents.to(device)
        text_ids = text_ids.to(device)
        control_latents = control_latents.to(device)
        control_ids = control_ids.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        prompt_embeds = prompt_embeds.to(device)

        with torch.no_grad():
            batch_size = image_latents.shape[0]
            if "noise" in embeddings:
                noise = embeddings["noise"].to(device)
            else:
                noise = torch.randn_like(image_latents, device=self.accelerator.device, dtype=self.weight_dtype)
            if "timestep" in embeddings:
                t = embeddings["timestep"]
            else:
                t = torch.rand((noise.shape[0],), device=device, dtype=self.weight_dtype)  # random time t

            t_ = t.unsqueeze(1).unsqueeze(1)
            noisy_model_input = (1.0 - t_) * image_latents + t_ * noise

            image_width = int(image_width) // (self.vae_scale_factor * 2)
            image_height = int(image_height) // (self.vae_scale_factor * 2)

            latent_ids = self._prepare_latent_image_ids(
                batch_size,
                image_height,
                image_width,
                self.accelerator.device,
                self.weight_dtype,
            )
            # Prepare input for transformer
            latent_model_input = torch.cat([noisy_model_input, control_latents], dim=1)
            latent_ids = torch.cat([latent_ids, control_ids], dim=0)
            # dim 0 is sequence dimension

        # Prepare guidance
        guidance = (
            torch.ones((noise.shape[0],)).to(self.accelerator.device) if self.dit.config.guidance_embeds else None
        )
        # convert dtype to self.weight_dtype
        latent_model_input = latent_model_input.to(self.weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.weight_dtype)
        prompt_embeds = prompt_embeds.to(self.weight_dtype)
        if guidance is not None:
            guidance = guidance.to(self.weight_dtype)
        image_latents = image_latents.to(self.weight_dtype).to(self.accelerator.device)
        t = t.to(self.weight_dtype)

        model_pred = self.dit(
            hidden_states=latent_model_input,
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
        model_pred = model_pred[:, : image_latents.size(1)]
        target = noise - image_latents
        # Use edit_mask from collate_fn (already in latent space)
        if "edit_mask" in embeddings:
            edit_mask = embeddings["edit_mask"].to(self.weight_dtype).to(self.accelerator.device)
        else:
            edit_mask = None
        loss = self.forward_loss(model_pred, target, weighting=None, edit_mask=edit_mask)
        if return_pred:
            return model_pred, loss
        return loss

    def _compute_loss_multi_resolution_mode(self, embeddings, return_pred=False) -> torch.Tensor:
        """Loss computation for multi-resolution batches with padding and masking

        This method handles batches with images of different resolutions by:
        1. Converting img_shapes to latent space dimensions
        2. Generating per-sample image IDs for RoPE
        3. Padding latents to maximum sequence length
        4. Creating attention mask for valid tokens
        5. Computing loss with masking to ignore padded tokens

        Note: This is a simplified implementation that uses padding and masking.
        For optimal performance with very different resolutions, custom attention
        processors (Phase 2.2-2.3) should be implemented.
        """
        assert self.accelerator is not None, "accelerator is not set"
        device = self.accelerator.device
        dtype = self.weight_dtype

        # Extract embeddings
        text_ids = embeddings["text_ids"]  # Shared text IDs
        control_latents_padded = embeddings["control_latents"]  # padded control latents in batch format
        image_latents_padded = embeddings["image_latents"]  # padded image latents in batch format
        pooled_prompt_embeds = embeddings["pooled_prompt_embeds"]
        prompt_embeds = embeddings["prompt_embeds"]
        img_shapes = embeddings["img_shapes"]

        text_ids = text_ids.to(device)
        control_latents_padded = control_latents_padded.to(device)
        image_latents_padded = image_latents_padded.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        prompt_embeds = prompt_embeds.to(device)

        # [[(C, H, W), ...,(C,H,W)], [(C, H, W), ...,(C,H,W)], ...] in pixel space, in batch format
        # for each sample, the first image shape is the target image shape, the rest are control images shapes

        batch_size = prompt_embeds.shape[0]

        # Step 1: Convert img_shapes to latent space
        img_shapes_latent = [
            self.convert_img_shapes_to_latent(img_shape, self.vae_scale_factor, 2) for img_shape in img_shapes
        ]

        # Step 2: Generate per-sample image IDs for RoPE
        # Extract (H, W) from latent shapes
        image_ids_list = []  # batch format
        control_ids_list = []  # batch format
        seq_len_latents = torch.zeros(batch_size, device=device, dtype=torch.int32)
        seq_len_control_latents = torch.zeros(batch_size, device=device, dtype=torch.int32)
        for i, img_shapes_batch_i in enumerate(img_shapes_latent):
            latent_image_ids = FluxKontextLoraTrainer._prepare_latent_image_ids(
                batch_size=1,  # Not used in the implementation
                height=img_shapes_batch_i[0][1],
                width=img_shapes_batch_i[0][2],
                device=device,
                dtype=dtype,
            )
            image_ids_list.append(latent_image_ids)
            seq_len_latents[i] = latent_image_ids.shape[0]

            concat_control_latent_ids = []
            for j, img_shape_batch_i_j in enumerate(img_shapes_batch_i[1:]):
                latent_control_ids = FluxKontextLoraTrainer._prepare_latent_image_ids(
                    batch_size=1,  # Not used in the implementation
                    height=img_shape_batch_i_j[1],
                    width=img_shape_batch_i_j[2],
                    device=device,
                    dtype=dtype,
                )
                latent_control_ids[..., 0] = j + 1
                # [seq,3]
                concat_control_latent_ids.append(latent_control_ids)
            concat_control_latent_ids = torch.cat(concat_control_latent_ids, dim=0)  # [sum_i seq_i, 3]
            control_ids_list.append(concat_control_latent_ids)
            seq_len_control_latents[i] = concat_control_latent_ids.shape[0]

            # seq len tensor(1040., device='cuda:0', dtype=torch.bfloat16)
            # seq len control tensor(1040., device='cuda:0', dtype=torch.bfloat16)
            # latent control ids torch.Size([1040, 3])
            # latent image ids torch.Size([1040, 3])
            # seq len tensor(1600., device='cuda:0', dtype=torch.bfloat16)
            # seq len control tensor(1600., device='cuda:0', dtype=torch.bfloat16)
            # latent control ids torch.Size([1600, 3])
            # latent image ids torch.Size([1600, 3])
        # Step 4: Add noise and prepare noisy input
        with torch.no_grad():
            seq_len_max = int(seq_len_latents.max())
            channels = image_latents_padded.shape[-1]  # Get channel dimension from latents
            # Used for loss calculation
            image_latent_mask = torch.zeros(batch_size, seq_len_max, device=device, dtype=dtype)
            timestep_input = torch.zeros(batch_size, device=device, dtype=dtype)
            noise_input = torch.zeros(batch_size, seq_len_max, channels, device=device, dtype=dtype)

            latent_model_input_list = []
            latent_ids_list = []
            for ii in range(batch_size):
                seq_len_image_latent = int(seq_len_latents[ii])
                seq_len_control_latent = int(seq_len_control_latents[ii])
                image_latent = image_latents_padded[ii, :seq_len_image_latent, :]
                control_latent = control_latents_padded[ii, :seq_len_control_latent, :]
                # Shape: [seq_len_image_latent, channels]
                if "noise" in embeddings:
                    noise = embeddings["noise"][ii].to(device).to(self.weight_dtype)
                    assert noise.shape == image_latent.shape, (
                        f"noise length {noise.shape} must equal {image_latent.shape}"
                    )
                else:
                    noise = torch.randn_like(image_latent, device=device, dtype=self.weight_dtype)

                if "timestep" in embeddings:
                    t = embeddings["timestep"][ii].to(device).to(self.weight_dtype)
                else:
                    t = torch.rand((1,), device=device, dtype=self.weight_dtype)
                timestep_input[ii] = t
                noise_input[ii, :seq_len_image_latent] = noise
                t_ = t.unsqueeze(1)
                noisy_model_input = (1.0 - t_) * image_latent + t_ * noise

                latent_model_input = torch.cat([noisy_model_input, control_latent], dim=0)
                latent_model_input_list.append(latent_model_input)

                latent_ids = torch.cat([image_ids_list[ii], control_ids_list[ii]], dim=0)
                latent_ids_list.append(latent_ids)

                assert latent_ids.shape[0] == latent_model_input.shape[0], (
                    f"latent_ids seq len {latent_ids.shape[0]} != "
                    f"noisy_model_input seq len {latent_model_input.shape[0]} at {ii}"
                )

                image_latent_mask[ii, :seq_len_image_latent] = 1

            # 0 seq len image latent 1040
            # 0 seq len control latent 1040
            # 0 image latent shape torch.Size([1040, 64])
            # 0 control latent shape torch.Size([1040, 64])
            # 0 latent_model_input shape torch.Size([2080, 64])
            # 0 latent_ids shape torch.Size([2080, 3])
            # 1 seq len image latent 1600
            # 1 seq len control latent 1600
            # 1 image latent shape torch.Size([1600, 64])
            # 1 control latent shape torch.Size([1600, 64])
            # 1 latent_model_input shape torch.Size([3200, 64])
            # 1 latent_ids shape torch.Size([3200, 3])
            latent_model_input, img_attention_mask = pad_latents_for_multi_res(
                latent_model_input_list, max_seq_len=None
            )
            latent_ids, _ = pad_latents_for_multi_res(latent_ids_list, max_seq_len=None)

            # latent_model_input shape after pad torch.Size([2, 3200, 64])
            # latent_ids shape after pad torch.Size([2, 3200, 3])
            # img_attention_mask torch.Size([2, 3200]) tensor([2080, 3200], device='cuda:0')
            # timesteps tensor([0.6211, 0.7109], device='cuda:0', dtype=torch.bfloat16)
            # guidance tensor([1., 1.], device='cuda:0', dtype=torch.bfloat16)

        # Step 5: Prepare guidance

        guidance = torch.ones((batch_size,)).to(device) if self.dit.config.guidance_embeds else None

        # Convert dtypes
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.weight_dtype)
        prompt_embeds = prompt_embeds.to(self.weight_dtype)
        if guidance is not None:
            guidance = guidance.to(self.weight_dtype)
        text_ids = text_ids.to(device)

        # Build complete attention mask: [batch, seq_txt + seq_img]
        # Text tokens are always valid (all 1s), image tokens use img_attention_mask
        seq_txt = text_ids.shape[0]  # text_ids is shared, shape [seq_txt, 3]
        seq_img = img_attention_mask.shape[1]  # img_attention_mask is [batch, seq_img]
        # Create full attention mask [batch, seq_txt + seq_img]
        full_attention_mask = torch.ones(batch_size, seq_txt + seq_img, device=device, dtype=torch.bool)
        # The image part uses the padded mask
        full_attention_mask[:, seq_txt:] = img_attention_mask
        latent_model_input = latent_model_input.to(self.weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.weight_dtype)
        prompt_embeds = prompt_embeds.to(self.weight_dtype)

        #  Step 6: Forward pass through transformer
        model_pred = self.dit(
            hidden_states=latent_model_input,
            timestep=timestep_input,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,  # Use batched IDs
            attention_mask=full_attention_mask,  # Full mask covering text + image tokens
            joint_attention_kwargs={},
            return_dict=False,
        )[0]

        # Extract prediction for image latents only
        model_pred = model_pred[:, :seq_len_max, :]
        # Ensure image_latents_padded is also sliced to seq_len_max for consistency
        target = noise_input - image_latents_padded[:, :seq_len_max, :]
        if "edit_mask" in embeddings:
            edit_mask = embeddings["edit_mask"].to(self.weight_dtype).to(self.accelerator.device)
        else:
            edit_mask = None
        # Use the new forward_loss method which automatically detects parameters
        # image_latent_mask torch.Size([2, 1600]) tensor([1040., 1600.], device='cuda:0', dtype=torch.bfloat16)
        # model_pred torch.Size([2, 1600, 64])
        # target torch.Size([2, 1600, 64])
        loss = self.forward_loss(
            model_pred=model_pred, target=target, attention_mask=image_latent_mask, edit_mask=edit_mask, weighting=None
        )
        if return_pred:
            return loss, {
                "latent_model_input": latent_model_input,
                "timestep_input": timestep_input,
                "guidance": guidance,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "prompt_embeds": prompt_embeds,
                "text_ids": text_ids,
                "latent_ids": latent_ids,
                "full_attention_mask": full_attention_mask,
                "model_pred": model_pred,
            }
        return loss

    def get_clip_prompt_embeds(self, prompt: str) -> torch.Tensor:
        """
        Get CLIP prompt embeddings.
        """
        prompt_list: list[str] = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)
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
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)
        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.view(batch_size, -1)
        return prompt_embeds

    def get_t5_prompt_embeds(self, prompt: str, max_sequence_length: int = 512) -> torch.Tensor:
        dtype = self.text_encoder.dtype
        prompt_list: list[str] = [prompt] if isinstance(prompt, str) else prompt
        text_inputs = self.tokenizer_2(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        device = next(self.text_encoder_2.parameters()).device
        text_input_ids = text_inputs.input_ids
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]
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
        """Prepare latent image ids for single sample
        Return:
            latent_image_ids: [height * width, 3]
        """
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )
        return latent_image_ids.to(device=device, dtype=dtype)

    def encode_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        image_latents = self.vae.encode(image)
        image_latents = image_latents.latent_dist.mode()
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return image_latents

    def create_sampling_latents(
        self, height: int, width: int, batch_size: int, num_channels_latents: int, device=None, dtype=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        latents = randn_tensor(shape, device=device, dtype=self.weight_dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, latent_ids

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """
        embeddgings dict from prepare embeddings
        """

        # num_additional_controls = embeddings["num_additional_controls"]
        dit_device = self.dit.device
        num_inference_steps = embeddings["num_inference_steps"]
        true_cfg_scale = embeddings["true_cfg_scale"]
        control_ids = embeddings["control_ids"].to(dit_device)
        control_latents = embeddings["control_latents"].to(dit_device)
        batch_size = embeddings["control_latents"].shape[0]
        if "latents" in embeddings:
            latents = embeddings["latents"].to(dit_device).to(self.weight_dtype)
            latent_ids = embeddings["latent_ids"].to(dit_device)
        else:
            latents, latent_ids = self.create_sampling_latents(
                embeddings["height"], embeddings["width"], batch_size, 16, dit_device, self.weight_dtype
            )
        latent_ids = torch.cat([latent_ids, control_ids], dim=0)
        image_seq_len = latents.shape[1]
        timesteps, num_inference_steps = self.prepare_predict_timesteps(
            num_inference_steps, image_seq_len, scheduler=self.sampling_scheduler
        )
        guidance = torch.full([1], embeddings["guidance"], device=dit_device, dtype=torch.float32)
        guidance = guidance.expand(batch_size)
        assert self.sampling_scheduler is not None, "sampling_scheduler is not set"
        self.sampling_scheduler.set_begin_index(0)
        # move all tensors to dit_device
        latent_ids = latent_ids.to(dit_device)
        guidance = guidance.to(dit_device)
        pooled_prompt_embeds = embeddings["pooled_prompt_embeds"].to(dit_device).to(self.weight_dtype)
        prompt_embeds = embeddings["prompt_embeds"].to(dit_device).to(self.weight_dtype)
        text_ids = embeddings["text_ids"].to(dit_device).to(self.weight_dtype)
        control_latents = control_latents.to(dit_device).to(self.weight_dtype)
        if true_cfg_scale > 1.0 and "negative_pooled_prompt_embeds" in embeddings:
            negative_pooled_prompt_embeds = (
                embeddings["negative_pooled_prompt_embeds"].to(dit_device).to(self.weight_dtype)
            )
            negative_prompt_embeds = embeddings["negative_prompt_embeds"].to(dit_device).to(self.weight_dtype)
            negative_text_ids = embeddings["negative_text_ids"].to(dit_device).to(self.weight_dtype)

        with torch.inference_mode():
            for _, t in enumerate(tqdm(timesteps, total=num_inference_steps, desc="Flux Kontext Generation")):
                latent_model_input = torch.cat([latents, control_latents], dim=1)
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
                noise_pred = noise_pred[:, :image_seq_len]
                if true_cfg_scale > 1.0 and "negative_pooled_prompt_embeds" in embeddings:
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
                    neg_noise_pred = neg_noise_pred[:, :image_seq_len]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                latents = self.sampling_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        return latents

    def decode_vae_latent(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        # latents after unpack torch.Size([1, 16, 106, 154])
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        latents = latents.to(self.vae.device)
        with torch.inference_mode():
            image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pt")
        return image

    def preprocess_image_predict(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess image for Flux Kontext VAE.
        image: RGB tensor input [B,C,H,W]
        Logic is
        1. Resize it make it devisible by 16
        2. Input should be RGB format
        3. Convert to [-1,1] range by devide 255
        4. Input shape should be [B,3,H,W]
        """
        import numpy as np
        import torch.nn.functional as F

        # get proper image shape
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.shape[2] <= 4:
            image = image.permute(2, 0, 1)  # [C,H,W]
        h, w = image.shape[1:]
        h = h // 16 * 16
        w = w // 16 * 16
        image = image.to(self.weight_dtype)
        if image.shape[-2] != h or image.shape[-1] != w:
            image = F.interpolate(image, size=(h, w), mode="bilinear", align_corners=False)
        image = image / 255.0
        # image = image.astype(self.weight_dtype)
        image = image.to(self.weight_dtype)
        # image = 2.0 * image - 1.0
        return image

    def prepare_predict_batch_data(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        prompt: str | list[str] | None = None,
        prompt_2: str | list[str] | None = None,
        negative_prompt: None | str | list[str] = None,
        negative_prompt_2: str | list[str] | None = None,
        num_inference_steps: int = 20,
        height: int | None = None,
        width: int | None = None,
        guidance_scale: float = 3.5,
        controls_size: list[list[int]] | None = None,
        generator: torch.Generator | None = None,
        weight_dtype: torch.dtype = torch.bfloat16,
        true_cfg_scale: float = 1.0,
        use_native_size: bool = False,
        use_multi_resolution: bool = False,
        **kwargs,
    ) -> dict:
        """
        Prepare batch data for prediction.
        args:
            additional_controls: [[control_1, control2], [control_1, control_2]]
            controls_size: [(h_0,w_0),(h1,w1), (h2,w2)]
            height: height for the generated image
            width: width for the generated image
            use_native_size: if True, the prompt image and additional
                controls will use their own size. That is they will not be resized.
            use_multi_resolution: if True, use multi-resolution processing mode
                with padding and attention masking for different image sizes.
        """
        if not isinstance(image, list):
            image = [image]
        # image = [make_image_devisible(image, self.vae_scale_factor) for image in image]
        prompt_image = [image[0]]
        additional_controls = [image[1:]] if len(image) > 1 else None

        assert prompt_image is not None, "prompt_image is required"
        assert prompt is not None, "prompt is required"
        self.weight_dtype = weight_dtype
        logging.info("Start predict")
        logging.info("image size format [H,W]")

        # Multi-resolution processing mode
        if use_multi_resolution:
            # Get multi-resolution configuration from data processor if available
            multi_resolutions = None
            if hasattr(self.preprocessor, "processor_config") and hasattr(
                self.preprocessor.processor_config, "multi_resolutions"
            ):
                multi_resolutions = self.preprocessor.processor_config.multi_resolutions
            return self._prepare_predict_batch_data_multi_resolution(
                prompt_image=prompt_image,
                prompt=prompt,
                additional_controls=additional_controls,
                prompt_2=prompt_2,
                negative_prompt=negative_prompt,
                negative_prompt_2=negative_prompt_2,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                generator=generator,
                weight_dtype=weight_dtype,
                true_cfg_scale=true_cfg_scale,
                multi_resolutions=multi_resolutions,
                **kwargs,
            )

        # Original processing logic
        if isinstance(prompt_image, PIL.Image.Image):
            prompt_image = [prompt_image]
        prompt_image = [make_image_devisible(image, self.vae_scale_factor) for image in prompt_image]

        if height is None or width is None:
            width, height = prompt_image[0].size
        else:
            width, height = make_image_shape_devisible(width, height, self.vae_scale_factor)

        if additional_controls and controls_size:
            assert len(additional_controls) + 1 == len(controls_size), (
                "the number of additional_controls_size should be same of additional_controls"
            )  # NOQA
        if controls_size:
            for item in controls_size:
                assert len(item) == 2, "the size of controls_size should be (h,w)"  # NOQA

        if not isinstance(prompt_image, list):
            prompt_image = [prompt_image]

        if use_native_size or controls_size is None:  # controls size not exist, use target size
            controls_size_list: list[list[int]] = [[prompt_image[0].size[1], prompt_image[0].size[0]]]
            if additional_controls:
                controls_size_list.extend([[ctrl.size[1], ctrl.size[0]] for ctrl in additional_controls[0]])
            controls_size = controls_size_list
            print("use native size", controls_size)
        print("controls_size", controls_size)
        logging.info("#" * 50)
        logging.info(f"image shapes for controls: {controls_size}")
        logging.info(f"image shape for target: [{height}, {width}]")
        data = {}
        control = []
        for img in prompt_image:
            img = self.preprocessor.preprocess({"control": img}, controls_size=controls_size)["control"]
            # img = self.preprocess_image_predict(img)
            control.append(img)
        control = torch.stack(control, dim=0)
        data["control"] = control

        if isinstance(prompt, str):
            prompt = [prompt]
        data["prompt"] = prompt

        if additional_controls:
            n_controls = len(additional_controls[0])
            new_controls: dict[str, list] = {f"control_{i + 1}": [] for i in range(n_controls)}
            # [control_1_batch1, control1_batch2, ..], [control2_batch1, control2_batch2, ..]
            for contorls in additional_controls:
                controls = self.preprocessor.preprocess({"controls": contorls}, controls_size=controls_size)["controls"]
                for i, control in enumerate(controls):
                    new_controls[f"control_{i + 1}"].append(control)
            for k, v in new_controls.items():
                print(k, type(v), type(v[0]), type(v[0][0]))
            for i in range(n_controls):
                control_stack = torch.stack(new_controls[f"control_{i + 1}"], dim=0)
                print("new controls", control_stack.shape, f"control_{i + 1}")
                data[f"control_{i + 1}"] = control_stack
            data["n_controls"] = n_controls
        else:
            data["n_controls"] = 0

        if prompt_2 is not None:
            if isinstance(prompt_2, str):
                prompt_2 = [prompt_2]
            assert len(prompt_2) == len(data["prompt"]), "the number of prompt_2 should be same of control"  # NOQA
            data["prompt_2"] = prompt_2

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            assert len(negative_prompt) == len(data["prompt"]), (
                "the number of negative_prompt should be same of control"
            )  # NOQA
            data["negative_prompt"] = negative_prompt

        if negative_prompt_2 is not None:
            if isinstance(negative_prompt_2, str):
                negative_prompt_2 = [negative_prompt_2]
            assert len(negative_prompt_2) == len(data["prompt"]), (
                "the number of negative_prompt_2 should be same of control"
            )  # NOQA
            data["negative_prompt_2"] = negative_prompt_2

        data["num_inference_steps"] = num_inference_steps
        data["height"] = height if height is not None else 1024
        data["width"] = width if width is not None else 1024
        data["true_cfg_scale"] = true_cfg_scale if true_cfg_scale is not None else 1.0
        data["guidance"] = guidance_scale
        print("data keys", data.keys())
        for k, v in data.items():
            print(k, type(v))
        return data

    def _prepare_predict_batch_data_multi_resolution(
        self,
        prompt_image: PIL.Image.Image | list[PIL.Image.Image] = None,  # Batch
        prompt: str | list[str] | None = None,  # Batch
        additional_controls: list[list[PIL.Image.Image]] = None,  # Batch
        num_inference_steps: int = 20,
        height: int | None = None,
        width: int | None = None,
        guidance_scale: float = 3.5,
        weight_dtype: torch.dtype = torch.bfloat16,
        true_cfg_scale: float = 1.0,
        multi_resolutions: list[int | str] | dict[str, Any] | None = None,
        **kwargs,
    ) -> dict:
        """
        Prepare batch data for multi-resolution prediction.

        This method handles multiple images with different resolutions using
        padding and attention masking for efficient batch processing.

        Args:
            prompt_image: PIL Image or list of PIL Images (control_0 for each sample)
            prompt: Text prompt or list of text prompts
            additional_controls: List of lists of additional control images
                Format: [[control_1_sample0, control_2_sample0], [control_1_sample1, control_2_sample1]]
                Note: control_0 is the prompt_image, additional_controls contains control_1, control_2, etc.
            num_inference_steps: Number of denoising steps
            height: Target height (optional, will use image size if not provided)
            width: Target width (optional, will use image size if not provided)
            guidance_scale: Guidance scale for generation
            weight_dtype: Data type for computation
            true_cfg_scale: True CFG scale for classifier-free guidance
            multi_resolutions: Multi-resolution configuration (currently not used)

        Returns:
            combined_embeddings: Dictionary containing all prepared data for sampling
                Structure:
                    - pooled_prompt_embeds: torch.Tensor [B, 768] - CLIP pooled embeddings
                    - prompt_embeds: torch.Tensor [B, 512, 4096] - T5 text embeddings
                    - text_ids: torch.Tensor [512, 3] - Text position IDs (shared across batch)
                    - control_latents_per_sample: List[List[torch.Tensor]] - Control latents per sample
                        Format: [[ctrl0_latent, ctrl1_latent], [ctrl0_latent, ctrl1_latent]]
                        Each control latent shape: [seq_ctrl_i, C] where C=64 (unpacked latent channels)
                    - control_ids_per_sample: List[List[torch.Tensor]] - Control position IDs per sample
                        Format: [[ctrl0_ids, ctrl1_ids], [ctrl0_ids, ctrl1_ids]]
                        Each control ID shape: [seq_ctrl_i, 3] with values [ctrl_idx, h_pos, w_pos]
                    - img_shapes: List[List[Tuple[int, int, int]]] - Image shapes in pixel space
                        Format: [[(C, H_target, W_target), (C, H_ctrl0, W_ctrl0), (C, H_ctrl1, W_ctrl1)], ...]
                    - img_shapes_latent: List[List[Tuple[int, int, int]]] - Image shapes in latent space
                        Format: [[(1, H_latent, W_latent), ...], ...]
                        Note: First dim is always 1 (channel in latent space after packing)
                    - num_inference_steps: int - Number of denoising steps
                    - true_cfg_scale: float - CFG scale
                    - guidance: float - Guidance scale

            Example (batch_size=2, with control_0 and control_1, target and control_0 same shape):
                Assume:
                    - Sample 0: target 512x512, control_0 512x512, control_1 256x256
                    - Sample 1: target 768x768, control_0 768x768, control_1 384x384
                    - VAE scale factor = 8, packing factor = 2
                    - Latent channels after packing = 64

                {
                    "pooled_prompt_embeds": torch.Tensor,  # shape: [2, 768]
                    "prompt_embeds": torch.Tensor,         # shape: [2, 512, 4096]
                    "text_ids": torch.Tensor,              # shape: [512, 3]

                    "control_latents_per_sample": [
                        # Sample 0
                        [
                            torch.Tensor,  # control_0: [4096, 64] (512/8*2 * 512/8*2 = 64*64 = 4096)
                            torch.Tensor,  # control_1: [1024, 64] (256/8*2 * 256/8*2 = 32*32 = 1024)
                        ],
                        # Sample 1
                        [
                            torch.Tensor,  # control_0: [9216, 64] (768/8*2 * 768/8*2 = 96*96 = 9216)
                            torch.Tensor,  # control_1: [2304, 64] (384/8*2 * 384/8*2 = 48*48 = 2304)
                        ]
                    ],

                    "control_ids_per_sample": [
                        # Sample 0
                        [
                            torch.Tensor,  # control_0 IDs: [4096, 3], values: [1, h, w]
                            torch.Tensor,  # control_1 IDs: [1024, 3], values: [2, h, w]
                        ],
                        # Sample 1
                        [
                            torch.Tensor,  # control_0 IDs: [9216, 3], values: [1, h, w]
                            torch.Tensor,  # control_1 IDs: [2304, 3], values: [2, h, w]
                        ]
                    ],

                    "img_shapes": [
                        # Sample 0: [target, control_0, control_1]
                        [(3, 512, 512), (3, 512, 512), (3, 256, 256)],
                        # Sample 1: [target, control_0, control_1]
                        [(3, 768, 768), (3, 768, 768), (3, 384, 384)]
                    ],

                    "img_shapes_latent": [
                        # Sample 0: [target, control_0, control_1] in latent space
                        # Note: First dim is 1 (channel in latent space), then H_latent, W_latent
                        [(1, 64, 64), (1, 64, 64), (1, 32, 32)],
                        # Sample 1: [target, control_0, control_1] in latent space
                        [(1, 96, 96), (1, 96, 96), (1, 48, 48)]
                    ],

                    "num_inference_steps": 20,
                    "true_cfg_scale": 1.0,
                    "guidance": 3.5
                }

        Note:
            - Image latents are NOT created in this stage (predict mode)
            - Image latents will be created as noise in sampling_from_embeddings_multi_resolution
            - Control latents are stored separately for each sample (not concatenated)
            - Control IDs have first dimension set to control index (1 for control_0, 2 for control_1, etc.)
        """
        # Convert to lists
        if isinstance(prompt_image, PIL.Image.Image):
            prompt_image = [prompt_image]
        if isinstance(prompt, str):
            prompt = [prompt]

        if additional_controls is None:
            additional_controls = [None for _ in range(len(prompt_image))]  # type: ignore[misc]

        # Make images divisible by scale factor
        prompt_image = [make_image_devisible(image, self.vae_scale_factor) for image in prompt_image]

        batch_size = len(prompt_image)
        logging.info(f"Multi-resolution mode: processing {batch_size} images")

        # Parse multi-resolution configuration (for future use)
        # multi_res_config = self._parse_multi_resolution_config(multi_resolutions)

        # Prepare img_shapes for multi-resolution processing

        img_shapes = []

        # Process each sample individually and combine embeddings
        embeddings_list = []
        for _, (img, prompt_text, sample_controls) in enumerate(
            zip(prompt_image, prompt, additional_controls, strict=False)  # type: ignore[arg-type]
        ):
            # Prepare single sample data
            single_data = {}
            # sample_controls: [control_1, control_2] for single batch

            multi_res_target = [img.size[1] * img.size[0]]
            multi_res_controls = [[img.size[1] * img.size[0]]]

            # Process control image
            if sample_controls:
                for c in sample_controls:
                    multi_res_controls.append([c.size[1] * c.size[0]])

            print("multi_res_controls", multi_res_controls)
            print("multi_res_target", multi_res_target)

            # Preprocess control image
            control = self.preprocessor.preprocess(
                {"control": img}, multi_res_controls=multi_res_controls, multi_res_target=multi_res_target
            )["control"]

            control = control.unsqueeze(0)  # add batch dimension  # [1,C,H,W]
            print("control", control.shape)
            single_data["control"] = control

            shape_this_sample = [(3, control.shape[2], control.shape[3]), (3, control.shape[2], control.shape[3])]

            # Process additional controls
            if sample_controls:
                n_controls = len(sample_controls)
                # sample_controls: [control_1, control2]

                controls = self.preprocessor.preprocess(
                    {"controls": sample_controls},
                    multi_res_controls=multi_res_controls,
                    multi_res_target=multi_res_target,
                )["controls"]

                for j, control in enumerate(controls):
                    shape_this_sample.append((3, control.shape[1], control.shape[2]))
                    single_data[f"control_{j + 1}"] = control.unsqueeze(0)
                single_data["n_controls"] = n_controls
            else:
                single_data["n_controls"] = 0

            # Add prompts
            single_data["prompt"] = [prompt_text]

            # Add other parameters
            single_data["num_inference_steps"] = num_inference_steps
            single_data["height"] = single_data["control"].shape[2]
            single_data["width"] = single_data["control"].shape[3]
            single_data["true_cfg_scale"] = true_cfg_scale
            single_data["guidance"] = guidance_scale

            embeddings_list.append(single_data)
            img_shapes.append(shape_this_sample)

        print("img_shapes", img_shapes)
        # Prepare embeddings for each sample
        for i, embeddings in enumerate(embeddings_list):
            # Prepare embeddings using the existing method
            prepared_embeddings = self.prepare_embeddings(embeddings, stage="predict")
            embeddings_list[i] = prepared_embeddings  # embedding shape has batch dimension

        # Combine embeddings for multi-resolution processing
        combined_embeddings = {}

        # Handle shared embeddings (text-related) - shapes are the same, so just concat
        shared_keys = ["pooled_prompt_embeds", "prompt_embeds"]
        for key in shared_keys:
            if key in embeddings_list[0]:
                if isinstance(embeddings_list[0][key], torch.Tensor):
                    # Concat tensors along batch dimension
                    combined_embeddings[key] = torch.cat([emb[key] for emb in embeddings_list], dim=0)
                    print(f"shape of {key}", combined_embeddings[key].shape)
                    # shape of pooled_prompt_embeds torch.Size([2, 768])
                    # shape of prompt_embeds torch.Size([2, 512, 4096])
                else:
                    # Use first sample's value (should be same for all)
                    combined_embeddings[key] = embeddings_list[0][key]

        # Handle text_ids - should be the same for all samples, use first one
        if "text_ids" in embeddings_list[0]:
            combined_embeddings["text_ids"] = embeddings_list[0]["text_ids"]
            print("shape of text_ids", combined_embeddings["text_ids"].shape)
        # shape of text_ids torch.Size([2, 512, 3])

        # Handle control_latents and control_ids - keep as lists for sampling stage
        # In predict mode, control_latents and control_ids are lists of tensors
        if "control_latents" in embeddings_list[0]:
            # Each embeddings_list[i]['control_latents'] is a list [control_0, control_1, ...]
            # Collect all control_latents lists for each sample
            control_latents_per_sample = []
            control_ids_per_sample = []

            for emb in embeddings_list:
                # Remove batch dimension from each control latent
                control_latents_per_sample.append(emb["control_latents"][0])  # already concated
                control_ids_per_sample.append(emb["control_ids"])

            # Store as list of lists: [[sample0_ctrl0, sample0_ctrl1], [sample1_ctrl0, sample1_ctrl1]]
            combined_embeddings["control_latents_per_sample"] = control_latents_per_sample  # remove batch dimension
            combined_embeddings["control_ids_per_sample"] = control_ids_per_sample

        # Add other parameters
        combined_embeddings["num_inference_steps"] = num_inference_steps
        combined_embeddings["true_cfg_scale"] = true_cfg_scale
        combined_embeddings["guidance"] = guidance_scale

        # Add img_shapes for multi-resolution processing (in pixel space)
        combined_embeddings["img_shapes"] = img_shapes

        # Convert img_shapes to latent space for sampling stage
        img_shapes_latent = [
            self.convert_img_shapes_to_latent(
                img_shape_perbatch, vae_scale_factor=self.vae_scale_factor, packing_factor=2
            )
            for img_shape_perbatch in img_shapes
        ]
        combined_embeddings["img_shapes_latent"] = img_shapes_latent

        logging.info(f"Multi-resolution embeddings prepared for {batch_size} samples")
        logging.info(f"Image shapes: {img_shapes}")
        logging.info(f"Image shapes (latent): {img_shapes_latent}")

        return combined_embeddings

    def _parse_multi_resolution_config(
        self, multi_resolutions: list[int | str] | dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Parse multi-resolution configuration similar to training format.

        Args:
            multi_resolutions: Multi-resolution configuration
                - List format: ["1024*1024", "512*512"] - applies to all images
                - Dict format: {target: [...], controls: [[...], [...], [...]]}

        Returns:
            Parsed configuration with target and controls candidates
        """
        if multi_resolutions is None:
            # Default configuration if none provided
            return {
                "mode": "simple",
                "target": [300 * 450, 630 * 945],  # Default test resolutions
                "controls": [[300 * 450, 630 * 945]],
            }

        # Format 1: Simple list - applies to all images
        if isinstance(multi_resolutions, list):
            parsed_target = []
            for item in multi_resolutions:
                if isinstance(item, (int,)):
                    parsed_target.append(int(item))
                elif isinstance(item, str):
                    # Parse expressions like "1024*1024"
                    if "*" in item:
                        parts = item.split("*")
                        if len(parts) == 2:
                            parsed_target.append(int(parts[0]) * int(parts[1]))
                        else:
                            raise ValueError(f"Invalid pixel expression: {item}")
                    else:
                        parsed_target.append(int(item))
                else:
                    raise ValueError(f"Invalid multi_resolutions item: {item}")
            return {
                "mode": "simple",
                "target": parsed_target,
                "controls": [parsed_target],  # Same for all controls
            }

        # Format 2: Advanced dict - separate configs per image type
        elif isinstance(multi_resolutions, dict):
            parsed_dict = {}

            # Parse target candidates
            if "target" in multi_resolutions:
                target_candidates = multi_resolutions["target"]
                parsed_target = []
                for item in target_candidates:
                    if isinstance(item, (int,)):
                        parsed_target.append(int(item))
                    elif isinstance(item, str):
                        if "*" in item:
                            parts = item.split("*")
                            if len(parts) == 2:
                                parsed_target.append(int(parts[0]) * int(parts[1]))
                            else:
                                raise ValueError(f"Invalid pixel expression: {item}")
                        else:
                            parsed_target.append(int(item))
                    else:
                        raise ValueError(f"Invalid target item: {item}")
                parsed_dict["target"] = parsed_target
            else:
                # Fallback to first control if no target specified
                if "controls" in multi_resolutions and multi_resolutions["controls"]:
                    parsed_dict["target"] = multi_resolutions["controls"][0]
                else:
                    raise ValueError("No target or controls specified in multi_resolutions dict")

            # Parse controls candidates
            if "controls" in multi_resolutions:
                parsed_controls = []
                for control_group in multi_resolutions["controls"]:
                    parsed_group = []
                    for item in control_group:
                        if isinstance(item, (int,)):
                            parsed_group.append(int(item))
                        elif isinstance(item, str):
                            if "*" in item:
                                parts = item.split("*")
                                if len(parts) == 2:
                                    parsed_group.append(int(parts[0]) * int(parts[1]))
                                else:
                                    raise ValueError(f"Invalid pixel expression: {item}")
                            else:
                                parsed_group.append(int(item))
                        else:
                            raise ValueError(f"Invalid control item: {item}")
                    parsed_controls.append(parsed_group)
                parsed_dict["controls"] = parsed_controls  # type: ignore[assignment]
            else:
                # Fallback: use target candidates for all controls
                parsed_dict["controls"] = [list(parsed_dict["target"])]  # type: ignore[list-item]

            return {"mode": "advanced", "target": parsed_dict["target"], "controls": parsed_dict["controls"]}

        else:
            raise ValueError(f"multi_resolutions must be list or dict, got {type(multi_resolutions)}")

    def test_multi_resolution_sampling_comparison(
        self,
        test_images: list[PIL.Image.Image],
        test_prompts: list[str],
        test_controls: list[list[PIL.Image.Image]],
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int | None = None,
        output_type: str = "pil",
    ) -> list[PIL.Image.Image]:
        """
        Multi-resolution sampling that returns PIL images like base_trainer.predict().

        This function generates images using multi-resolution method and returns
        PIL images directly without saving to disk.

        Args:
            test_images: List of PIL images with different resolutions
            test_prompts: List of prompts corresponding to each image
            test_controls: List of control images for each sample
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale for sampling
            seed: Random seed for reproducibility (optional)
            output_type: Output type, "pil" for PIL images or "pt" for tensors

        Returns:
            List of PIL.Image.Image objects (one per sample)
        """
        import numpy as np

        # Set random seed for reproducibility if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Prepare multi-resolution batch data
        multi_res_embeddings = self.prepare_predict_batch_data(
            image=test_images + test_controls,
            prompt=test_prompts,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            use_multi_resolution=True,  # Use multi-resolution mode
        )

        # Generate images using multi-resolution method
        multi_res_latents = self.sampling_from_embeddings_multi_resolution(multi_res_embeddings)
        img_shapes = multi_res_embeddings["img_shapes"]

        # Decode latents to images (returns list of tensors)
        images_tensor = []
        for i, img_shape in enumerate(img_shapes):
            # Decode multi-resolution result
            # multi_res_latents is a list of tensors, add batch dimension for decode
            multi_res_latent = multi_res_latents[i].unsqueeze(0)  # Add batch dimension
            decoded_image = self.decode_vae_latent(multi_res_latent, img_shape[0][1], img_shape[0][2])[
                0
            ]  # Remove batch dimension, shape: [C, H, W]
            images_tensor.append(decoded_image)

        # Convert to PIL images (following base_trainer.predict pattern)
        if output_type == "pil":
            pil_images = []
            for img_tensor in images_tensor:
                # Convert tensor [C, H, W] to numpy [H, W, C] in range [0, 255]
                img_np = img_tensor.detach().permute(1, 2, 0).float().cpu().numpy()
                img_np = (img_np * 255).round().astype("uint8")

                # Convert to PIL Image
                if img_np.shape[-1] == 1:
                    # Grayscale
                    pil_img = PIL.Image.fromarray(img_np.squeeze(), mode="L")
                else:
                    # RGB
                    pil_img = PIL.Image.fromarray(img_np)

                pil_images.append(pil_img)

            return pil_images

        # Return tensors if output_type is "pt"
        return images_tensor

    def sampling_from_embeddings_multi_resolution(self, embeddings: dict) -> torch.Tensor:
        """
        Multi-resolution sampling with correct concat + padding logic per step.

        This method handles batches with different image resolutions by:
        1. Creating noise for each sample based on its shape
        2. For each denoising step:
           a. Concat image_latent and control_latents for each sample
           b. Pad all samples to max length and create attention mask
           c. Generate per-sample latent_ids and pad them
           d. Forward pass through model
           e. Update each sample's image_latent using the mask

        Args:
            embeddings: Dictionary containing batch data with img_shapes_latent key

        Returns:
            List of generated latents (one per sample, unpadded)
        """
        dit_device = next(self.dit.parameters()).device
        num_inference_steps = embeddings["num_inference_steps"]
        true_cfg_scale = embeddings["true_cfg_scale"]
        batch_size = embeddings["pooled_prompt_embeds"].shape[0]

        # Extract multi-resolution data
        img_shapes_latent = embeddings["img_shapes_latent"]
        pooled_prompt_embeds = embeddings["pooled_prompt_embeds"]
        prompt_embeds = embeddings["prompt_embeds"]
        text_ids = embeddings["text_ids"]

        # Extract control latents and ids per sample
        control_latents_per_sample = embeddings.get("control_latents_per_sample", [[] for _ in range(batch_size)])
        control_ids_per_sample = embeddings.get("control_ids_per_sample", [[] for _ in range(batch_size)])
        for control_ids in control_ids_per_sample:
            print(
                "control_ids",
                len(control_ids),
                len(control_ids[0]),
                type(control_ids),
                type(control_ids[0]),
                type(control_ids[0][0]),
            )
            print("control_ids[0]", control_ids[0][0:10][0:10])

        # Step 1: Create noise for each sample based on its shape
        image_latents_list = []
        image_seq_lens = []
        print("img_shapes_latent", img_shapes_latent)
        for _, img_shapes_batch_i in enumerate(img_shapes_latent):
            height = img_shapes_batch_i[0][1] * self.vae_scale_factor * 2  # Convert back to pixel space
            width = img_shapes_batch_i[0][2] * self.vae_scale_factor * 2
            print("shape create image latents", height, width)
            latents, _ = self.create_sampling_latents(height, width, 1, 16, dit_device, self.weight_dtype)
            print("latent shape", latents.shape)
            image_latents_list.append(latents[0])  # Remove batch dimension, shape: [seq, C]
            image_seq_lens.append(latents[0].shape[0])

        # Prepare timesteps (use max image seq len for scheduler)
        max_image_seq_len = max(image_seq_lens)
        timesteps, num_inference_steps = self.prepare_predict_timesteps(
            num_inference_steps, max_image_seq_len, scheduler=self.sampling_scheduler
        )

        # Prepare guidance
        guidance = torch.full([1], embeddings["guidance"], device=dit_device, dtype=torch.float32)
        guidance = guidance.expand(batch_size)
        assert self.sampling_scheduler is not None, "sampling_scheduler is not set"
        self.sampling_scheduler.set_begin_index(0)

        # Move tensors to device
        pooled_prompt_embeds = pooled_prompt_embeds.to(dit_device).to(self.weight_dtype)
        prompt_embeds = prompt_embeds.to(dit_device).to(self.weight_dtype)
        text_ids = text_ids.to(dit_device).to(self.weight_dtype)
        guidance = guidance.to(dit_device)

        # Move control latents to device
        for i in range(batch_size):
            control_latents_per_sample[i] = control_latents_per_sample[i].to(dit_device).to(self.weight_dtype)
            control_ids_per_sample[i] = control_ids_per_sample[i].to(dit_device).to(self.weight_dtype)

        # Prepare negative prompts if needed
        if true_cfg_scale > 1.0 and "negative_pooled_prompt_embeds" in embeddings:
            negative_pooled_prompt_embeds = (
                embeddings["negative_pooled_prompt_embeds"].to(dit_device).to(self.weight_dtype)
            )
            negative_prompt_embeds = embeddings["negative_prompt_embeds"].to(dit_device).to(self.weight_dtype)
            negative_text_ids = embeddings["negative_text_ids"].to(dit_device).to(self.weight_dtype)

        # Sampling loop
        with torch.inference_mode():
            for _, t in enumerate(
                tqdm(timesteps, total=num_inference_steps, desc="Flux Kontext Multi-Resolution Generation")
            ):
                timestep = t.expand(batch_size).to(dit_device).to(self.weight_dtype)

                # Step 2a: Concat image_latent and control_latents for each sample
                concat_latents_list = []
                for i in range(batch_size):
                    # Concat image latent with all control latents for this sample
                    sample_latents = [image_latents_list[i], control_latents_per_sample[i]]
                    print("image_latents_list[i]", image_latents_list[i].shape)
                    print("control_latents_per_sample[i]", control_latents_per_sample[i].shape)
                    concat_sample_latent = torch.cat(sample_latents, dim=0)  # [seq_i, C]
                    concat_latents_list.append(concat_sample_latent)

                # Step 2b: Pad all samples to max length and create attention mask
                latent_model_input, latent_attention_mask = pad_latents_for_multi_res(
                    concat_latents_list, max_seq_len=None
                )
                # Step 2c: Generate per-sample latent_ids and pad them
                latent_ids_list = []
                for i, img_shapes_batch_i in enumerate(img_shapes_latent):
                    # Generate image IDs for target image
                    latent_image_ids = self._prepare_latent_image_ids(
                        batch_size=1,
                        height=img_shapes_batch_i[0][1],
                        width=img_shapes_batch_i[0][2],
                        device=dit_device,
                        dtype=self.weight_dtype,
                    )
                    print("latent_image_ids", latent_image_ids.shape)
                    print("height", img_shapes_batch_i[0][1], "width", img_shapes_batch_i[0][2])
                    # Concat with control IDs for this sample
                    sample_ids = [latent_image_ids, control_ids_per_sample[i]]
                    concat_sample_ids = torch.cat(sample_ids, dim=0)  # [seq_i, 3]
                    latent_ids_list.append(concat_sample_ids)

                # Pad latent_ids
                latent_ids_padded, _ = pad_latents_for_multi_res(latent_ids_list, max_seq_len=None)
                # Create full attention mask (text + image)
                seq_txt = text_ids.shape[0]
                seq_img = latent_attention_mask.shape[1]
                full_attention_mask = torch.ones(batch_size, seq_txt + seq_img, device=dit_device, dtype=torch.bool)
                full_attention_mask[:, seq_txt:] = latent_attention_mask

                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids_padded,
                    attention_mask=full_attention_mask,
                    joint_attention_kwargs={},
                    return_dict=False,
                )[0]

                # Apply CFG if needed
                if true_cfg_scale > 1.0 and "negative_pooled_prompt_embeds" in embeddings:
                    neg_noise_pred = self.dit(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids_padded,
                        attention_mask=full_attention_mask,
                        joint_attention_kwargs={},
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                # Step 2e: Apply scheduler step on padded latents, then extract valid parts
                # First, pad image latents to match the padded shape
                image_latents_padded, _ = pad_latents_for_multi_res(image_latents_list, max_seq_len=None)

                # Extract only the image latent part of noise prediction
                max_image_seq_len = max(image_seq_lens)
                image_noise_pred = noise_pred[:, :max_image_seq_len]

                # Apply scheduler step on padded tensors (one call for the whole batch)
                assert self.sampling_scheduler is not None, "sampling_scheduler is not set"
                updated_latents_padded = self.sampling_scheduler.step(
                    image_noise_pred, t, image_latents_padded, return_dict=False
                )[0]

                # Extract valid parts for each sample
                for i in range(batch_size):
                    seq_len = image_seq_lens[i]
                    image_latents_list[i] = updated_latents_padded[i, :seq_len]

        # Return unpadded latents list (each sample has its original shape)
        return image_latents_list
