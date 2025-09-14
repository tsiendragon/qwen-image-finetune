"""
Flux Kontext LoRA Trainer Implementation
Following QwenImageEditTrainer patterns with dual text encoder support.
"""
import torch
import PIL
import gc
import inspect
from typing import Optional, Union, List, Tuple
from tqdm.auto import tqdm
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
        pretrains = self.config.model.pretrained_embeddings
        if pretrains is not None and 'vae' in pretrains:
            self.vae = load_flux_kontext_vae(
                pretrains['vae'],
                weight_dtype=self.weight_dtype,
            ).to('cpu')
            logging.info(f'loaded vae from {pretrains["vae"]}')
        else:
            self.vae = load_flux_kontext_vae(
                self.config.model.pretrained_model_name_or_path,
                weight_dtype=self.weight_dtype,
            ).to('cpu')
        if pretrains is not None and 'text_encoder' in pretrains:
            self.text_encoder = load_flux_kontext_clip(
                pretrains['text_encoder'],
                weight_dtype=self.weight_dtype,
            ).to('cpu')
            logging.info(f'loaded text_encoder from {pretrains["text_encoder"]}')
        else:
            self.text_encoder = load_flux_kontext_clip(
                self.config.model.pretrained_model_name_or_path,
                weight_dtype=self.weight_dtype,
            ).to('cpu')

        if pretrains is not None and 'text_encoder_2' in pretrains:
            self.text_encoder_2 = load_flux_kontext_t5(
                pretrains['text_encoder_2'],
                weight_dtype=self.weight_dtype,
            ).to('cpu')
            logging.info(f'loaded text_encoder_2 from {pretrains["text_encoder_2"]}')
        else:
            self.text_encoder_2 = load_flux_kontext_t5(
                self.config.model.pretrained_model_name_or_path,
                weight_dtype=self.weight_dtype,
            ).to('cpu')

        self.dit = load_flux_kontext_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        ).to('cpu')

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

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )

        logging.info(
            f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}"
        )

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
            )
            prompt_embeds = self.get_t5_prompt_embeds(
                prompt=prompt_2,
                max_sequence_length=max_sequence_length,
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
        # height, width = image.shape[2:]
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

    def setup_model_device_train_mode(self, stage="fit", cache=False):
        """Set model device allocation and train mode."""
        if stage == "fit":
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
            print('self.config.cache.devices.vae', self.config.cache.devices.vae)
            print('vae device ', next(self.vae.parameters()).device)

            self.vae.decoder.to("cpu")
            print('vae device ', next(self.vae.parameters()).device)

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
            logging.info('cache mode device setting')
            print('vae device ', next(self.vae.parameters()).device)

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
        logging.info('prepare_embeddings')
        if "image" in batch:
            batch["image"] = self.normalize_image(batch["image"])

        if "control" in batch:
            batch["control"] = self.normalize_image(batch["control"])

        num_additional_controls = batch["n_controls"] if isinstance(batch["n_controls"], int) else batch["n_controls"][0]

        for i in range(num_additional_controls):
            additional_control_key = f"control_{i+1}"
            if additional_control_key in batch:
                batch[additional_control_key] = self.normalize_image(
                    batch[additional_control_key]
                )
        logging.info('process controls')

        if "prompt_2" in batch:
            prompt_2 = batch["prompt_2"]
        else:
            prompt_2 = batch["prompt"]

        logging.info('encode prompt')
        pooled_prompt_embeds, prompt_embeds, text_ids = self.encode_prompt(
            prompt=batch["prompt"],
            prompt_2=prompt_2,
            max_sequence_length=self.max_sequence_length,
        )
        batch["pooled_prompt_embeds"] = pooled_prompt_embeds
        batch["prompt_embeds"] = prompt_embeds
        batch["text_ids"] = text_ids
        logging.info('encode prompt')

        if stage == 'cache':
            pooled_prompt_embeds, prompt_embeds, _ = self.encode_prompt(
                prompt=[""],
                prompt_2=None,
                max_sequence_length=self.max_sequence_length,
            )
            batch["empty_pooled_prompt_embeds"] = pooled_prompt_embeds
            batch["empty_prompt_embeds"] = prompt_embeds
            logging.info('process empty prompt')

        if "negative_prompt" in batch:
            if "negative_prompt_2" in batch:
                prompt_2 = batch["negative_prompt_2"]
            else:
                prompt_2 = batch["negative_prompt"]
            negative_pooled_prompt_embeds, negative_prompt_embeds, negative_text_ids = (
                self.encode_prompt(
                    prompt=batch["negative_prompt"],
                    prompt_2=prompt_2,
                    max_sequence_length=self.max_sequence_length,
                )
            )
            batch["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
            batch["negative_prompt_embeds"] = negative_prompt_embeds
            batch["negative_text_ids"] = negative_text_ids
            logging.info('process negative prompt')
        if "image" in batch:
            logging.info('process image')
            image = batch["image"]  # single images
            print('image shaope',image.shape)
            batch_size = image.shape[0]
            image_height, image_width = image.shape[2:]
            print('batch_size', batch_size, image_height, image_width)
            device = next(self.vae.parameters()).device
            print('device', device)

            latents, image_latents, latent_ids, image_ids = self.prepare_latents(
                image,
                batch_size,
                16,
                image_height,
                image_width,
                self.weight_dtype,
            )

            batch["image_latents"] = image_latents
            logging.info(f"stage: {stage}")

        logging.info(f"batch: {batch.keys()}")

        if "control" in batch:
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
            logging.info(f"control_ids: {control_ids}")

        for i in range(1, num_additional_controls + 1):
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

        if "control_latents" in batch:
            batch["control_latents"] = torch.cat(batch["control_latents"], dim=1)
            batch["control_ids"] = torch.cat(batch["control_ids"], dim=0)

        if self.config.loss.mask_loss and "mask" in batch:
            mask = batch["mask"]
            batch["mask"] = resize_bhw(mask, image_height, image_width)
            batch["mask"] = map_mask_to_latent(batch["mask"])
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
        self.cache_manager.save_cache_embedding(
            cache_embeddings, map_keys, data["file_hashes"]
        )

    def prepare_cached_embeddings(self, batch):
        if self.config.loss.mask_loss and "mask" in batch:
            image_height, image_width = batch["image"].shape[2:]
            mask = batch["mask"]
            batch["mask"] = resize_bhw(mask, image_height, image_width)
            batch["mask"] = map_mask_to_latent(batch["mask"])
        batch["control_ids"] = batch["control_ids"][0]  # remove batch dim
        batch["text_ids"] = batch["text_ids"][0]  # remove batch dim
        return batch

    def _compute_loss(self, embeddings) -> torch.Tensor:
        image_latents = embeddings["image_latents"]
        text_ids = embeddings["text_ids"]
        control_latents = embeddings["control_latents"]
        control_ids = embeddings["control_ids"]
        pooled_prompt_embeds = embeddings["pooled_prompt_embeds"]
        prompt_embeds = embeddings["prompt_embeds"]
        device = self.accelerator.device
        image_height, image_width = embeddings['image'].shape[2:]

        with torch.no_grad():
            batch_size = image_latents.shape[0]
            noise = torch.randn_like(
                image_latents, device=self.accelerator.device, dtype=self.weight_dtype
            )
            t = torch.rand(
                (noise.shape[0],), device=device, dtype=self.weight_dtype
            )  # random time t
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
            torch.ones((noise.shape[0],)).to(self.accelerator.device)
            if self.dit.config.guidance_embeds
            else None
        )
        # convert dtype to self.weight_dtype
        latent_model_input = latent_model_input.to(self.weight_dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.weight_dtype)
        prompt_embeds = prompt_embeds.to(self.weight_dtype)
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
        if "mask" in embeddings:
            edit_mask = embeddings["mask"].to(self.weight_dtype).to(self.accelerator.device)
        else:
            edit_mask = None
        loss = self.forward_loss(
            model_pred, target, weighting=None, edit_mask=edit_mask
        )
        return loss

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

    def create_sampling_latents(self,
                                height: int,
                                width: int,
                                batch_size: int,
                                num_channels_latents: int,
                                device=None,
                                dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        latent_ids = self._prepare_latent_image_ids(
            batch_size, height // 2, width // 2, device, dtype
        )
        latents = randn_tensor(shape, device=device, dtype=self.weight_dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        return latents, latent_ids

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """
        embeddgings dict from prepare embeddings
        """

        # num_additional_controls = embeddings["num_additional_controls"]
        num_inference_steps = embeddings["num_inference_steps"]
        true_cfg_scale = embeddings["true_cfg_scale"]
        control_ids = embeddings["control_ids"]
        control_latents = embeddings["control_latents"]
        dit_device = next(self.dit.parameters()).device
        batch_size = embeddings["control_latents"].shape[0]

        latents, latent_ids = self.create_sampling_latents(
            embeddings["height"],
            embeddings["width"],
            batch_size,
            16,
            dit_device,
            self.weight_dtype)

        latent_ids = torch.cat([latent_ids, control_ids], dim=0)
        image_seq_len = latents.shape[1]
        timesteps, num_inference_steps = self.prepare_predict_timesteps(
            num_inference_steps, image_seq_len
        )
        guidance = torch.full(
            [1], embeddings['guidance'], device=dit_device, dtype=torch.float32
        )
        guidance = guidance.expand(batch_size)
        self.scheduler.set_begin_index(0)
        # move all tensors to dit_device
        latent_ids = latent_ids.to(dit_device)
        guidance = guidance.to(dit_device)
        pooled_prompt_embeds = (
            embeddings["pooled_prompt_embeds"].to(dit_device).to(self.weight_dtype)
        )
        prompt_embeds = embeddings["prompt_embeds"].to(dit_device).to(self.weight_dtype)
        text_ids = embeddings["text_ids"].to(dit_device).to(self.weight_dtype)
        control_latents = control_latents.to(dit_device).to(self.weight_dtype)
        if true_cfg_scale > 1.0 and "negative_pooled_prompt_embeds" in embeddings:
            negative_pooled_prompt_embeds = (
                embeddings["negative_pooled_prompt_embeds"]
                .to(dit_device)
                .to(self.weight_dtype)
            )
            negative_prompt_embeds = (
                embeddings["negative_prompt_embeds"]
                .to(dit_device)
                .to(self.weight_dtype)
            )
            negative_text_ids = (
                embeddings["negative_text_ids"].to(dit_device).to(self.weight_dtype)
            )

        with torch.inference_mode():
            for i, t in enumerate(
                tqdm(
                    timesteps, total=num_inference_steps, desc="Flux Kontext Generation"
                )
            ):
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
                if (
                    true_cfg_scale > 1.0
                    and "negative_pooled_prompt_embeds" in embeddings
                ):
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
                    noise_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
        return latents

    def decode_vae_latent(
        self, latents: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        # latents after unpack torch.Size([1, 16, 106, 154])
        latents = (
            latents / self.vae.config.scaling_factor
        ) + self.vae.config.shift_factor
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
        import torch.nn.functional as F
        import numpy as np
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
            image = F.interpolate(
                image, size=(h, w), mode="bilinear", align_corners=False
            )
        image = image / 255.0
        # image = image.astype(self.weight_dtype)
        image = image.to(self.weight_dtype)
        # image = 2.0 * image - 1.0
        return image

    def prepare_predict_batch_data(
        self,
        prompt_image: Union[PIL.Image.Image, List[PIL.Image.Image]] = None,
        prompt: Union[str, List[str]] = None,
        additional_controls: List[PIL.Image.Image] = [],
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[None, str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 20,
        height: Optional[int] = None,
        width: Optional[int] = None,
        guidance_scale: float = 3.5,
        additional_controls_size: Optional[List[int]] = None,
        generator: Optional[torch.Generator] = None,
        weight_dtype: torch.dtype = torch.bfloat16,
        true_cfg_scale: float = 1.0,
        auto_resize: bool = False,
        **kwargs,
    ) -> dict:
        """
        Prepare batch data for prediction.
        args:
            additional_controls: [[control_1, control2], [control_1, control_2]]
            additional_controls_size: [(h1,w1), (h2,w2)]
            height: height for the generated image
            width: width for the generated image
        """
        assert prompt_image is not None, "prompt_image is required"
        assert prompt is not None, "prompt is required"
        self.weight_dtype = weight_dtype

        if additional_controls_size:
            assert len(additional_controls_size) == len(
                additional_controls[0]
            ), "the number of additional_controls_size should be same of additional_controls"  # NOQA
            assert (
                len(additional_controls_size[0]) == 2
            ), "the size of additional_controls_size should be (h,w)"  # NOQA
            controls_size = [[height, width]] + additional_controls_size
        else:
            controls_size = [[height, width]]

        if not isinstance(prompt_image, list):
            prompt_image = [prompt_image]

        data = {}
        control = []
        for img in prompt_image:
            # img = self.preprocessor.preprocess(
            #     {"control": img}, controls_size=controls_size
            # )["control"]
            img = self.preprocess_image_predict(img)
            control.append(img)
        control = torch.stack(control, dim=0)
        data["control"] = control

        if isinstance(prompt, str):
            prompt = [prompt]
        data["prompt"] = prompt

        if additional_controls:
            new_controls = {f"control_{i}": [] for i in range(len(additional_controls))}
            # [control_1_batch1, control1_batch2, ..], [control2_batch1, control2_batch2, ..]
            for contorls in additional_controls:
                controls = self.preprocessor.preprocess(
                    {"control": contorls}, controls_size=controls_size
                )["controls"]
                for i, control in enumerate(controls):
                    new_controls[f"control_{i}"].append(control)
            for i in range(len(additional_controls)):
                new_controls[f"control_{i}"] = torch.stack(
                    new_controls[f"control_{i}"], dim=0
                )
                data[f"control_{i}"] = new_controls[f"control_{i}"]

        if prompt_2 is not None:
            if isinstance(prompt_2, str):
                prompt_2 = [prompt_2]
            assert len(prompt_2) == len(
                data["prompt"]
            ), "the number of prompt_2 should be same of control"  # NOQA
            data["prompt_2"] = prompt_2

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            assert len(negative_prompt) == len(
                data["prompt"]
            ), "the number of negative_prompt should be same of control"  # NOQA
            data["negative_prompt"] = negative_prompt

        if negative_prompt_2 is not None:
            if isinstance(negative_prompt_2, str):
                negative_prompt_2 = [negative_prompt_2]
            assert len(negative_prompt_2) == len(
                data["prompt"]
            ), "the number of negative_prompt_2 should be same of control"  # NOQA
            data["negative_prompt_2"] = negative_prompt_2

        data["num_inference_steps"] = num_inference_steps
        data["height"] = height
        data["width"] = width
        data["true_cfg_scale"] = true_cfg_scale
        data['guidance'] = guidance_scale
        return data
