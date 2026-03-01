"""Trainer for Qwen/Qwen-Image (text-to-image, no editing / no control image).

Key differences from QwenImageEditTrainer:
  - Text-only prompt encoding: uses self.tokenizer (not Qwen2VLProcessor)
  - No control image / no control_latents
  - img_shapes has a single shape per sample (the output image only)
  - prepare_latents only generates noise latents; no VAE-encoding of a control image
"""

import copy
import logging
from typing import Any

import numpy as np
import PIL
import torch
import torch.nn.functional as F  # NOQA
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import randn_tensor
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from tqdm.auto import tqdm

from qflux.models.load_model import load_transformer, load_vae
from qflux.trainer.qwen_image_edit_trainer import QwenImageEditTrainer
from qflux.utils.images import make_image_shape_devisible


class QwenImageT2ITrainer(QwenImageEditTrainer):
    """Trainer for Qwen/Qwen-Image text-to-image model.

    Inherits core infrastructure from QwenImageEditTrainer but removes all
    control-image logic.  The denoising step operates on pure noise latents;
    no control image is concatenated.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        return QwenImagePipeline

    _pack_latents = staticmethod(QwenImagePipeline._pack_latents)
    _unpack_latents = staticmethod(QwenImagePipeline._unpack_latents)

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, text_encoder_device=None):
        """Load components from QwenImagePipeline (text-to-image)."""
        logging.info("Loading QwenImagePipeline (T2I) and separating components...")

        pipe = QwenImagePipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype,
            transformer=None,
            vae=None,
        )
        pipe.to("cpu")

        self.vae = load_vae(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        )

        # The T2I pipeline uses a plain Qwen2Tokenizer (not Qwen2VLProcessor).
        # Use pretrained_embeddings.text_encoder if set for offline / local use.
        _text_encoder_path = (
            (self.config.model.pretrained_embeddings or {}).get("text_encoder")
            or self.config.model.pretrained_model_name_or_path
        )
        # Load only the LLM backbone (not the VL processor version)
        from transformers import AutoModelForCausalLM

        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            _text_encoder_path,
            torch_dtype=self.weight_dtype,
        )
        logging.info(f"text_encoder device: {self.text_encoder.device}")

        self.dit = load_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype,
        )

        from diffusers.image_processor import VaeImageProcessor
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

        self.tokenizer: Qwen2Tokenizer = pipe.tokenizer
        self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
        self.sampling_scheduler: FlowMatchEulerDiscreteScheduler = copy.deepcopy(self.scheduler)

        # VAE parameters
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.vae_latent_mean = self.vae.config.latents_mean
        self.vae_latent_std = self.vae.config.latents_std
        self.vae_z_dim = self.vae.config.z_dim
        self.latent_channels = self.vae.config.z_dim

        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False

        # Copy prompt template from pipeline
        self.prompt_template_encode = pipe.prompt_template_encode
        self.prompt_template_encode_start_idx = pipe.prompt_template_encode_start_idx
        self.tokenizer_max_length = pipe.tokenizer_max_length

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.num_channels_latents = self.dit.config.in_channels // 4

        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logging.info(f"T2I components loaded. VAE scale factor: {self.vae_scale_factor}")

    # ------------------------------------------------------------------ #
    # Prompt encoding (text-only)                                          #
    # ------------------------------------------------------------------ #

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] | None = None,
        image=None,  # ignored – T2I has no conditioning image
        device: torch.device = "cuda",
        dtype: torch.dtype = None,
    ):
        """Text-only prompt embedding using the Qwen2 tokenizer."""
        assert prompt is not None, "prompt is required"
        dtype = dtype or self.weight_dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]

        txt_tokens = self.tokenizer(
            txt,
            max_length=self.tokenizer_max_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.cuda.device(txt_tokens.input_ids.device):
            outputs = self.text_encoder(
                input_ids=txt_tokens.input_ids,
                attention_mask=txt_tokens.attention_mask,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden_states
        ]
        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))])
                for u in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_seq_len - u.size(0))])
                for u in attn_mask_list
            ]
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: str | list[str],
        image=None,  # ignored
    ):
        device = self.text_encoder.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        with torch.inference_mode():
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt, device=device
            )
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size, seq_len)
        return prompt_embeds, prompt_embeds_mask

    # ------------------------------------------------------------------ #
    # Batch preparation                                                    #
    # ------------------------------------------------------------------ #

    def prepare_embeddings(self, batch, stage="fit", debug=False):
        """Prepare embeddings for T2I.

        Unlike the edit trainer there is no control image – we only encode
        the text prompt and (during training) VAE-encode the target image.
        """
        # VAE-encode target image (present during training / cache stages)
        if "image" in batch:
            batch["image"] = self.preprocess_image_for_vae_encoder(batch["image"])  # B,3,1,H,W
            batch["width"] = batch["image"].shape[4]
            batch["height"] = batch["image"].shape[3]

        # Text-only prompt encoding
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt=batch["prompt"])
        batch["prompt_embeds"] = prompt_embeds
        batch["prompt_embeds_mask"] = prompt_embeds_mask

        if stage == "cache":
            empty_prompt_embeds, empty_prompt_embeds_mask = self.encode_prompt(prompt=[""])
            batch["empty_prompt_embeds"] = empty_prompt_embeds
            batch["empty_prompt_embeds_mask"] = empty_prompt_embeds_mask

        if "negative_prompt" in batch and batch.get("true_cfg_scale", 1.0) > 1:
            neg_embeds, neg_mask = self.encode_prompt(prompt=batch["negative_prompt"])
            batch["negative_prompt_embeds"] = neg_embeds
            batch["negative_prompt_embeds_mask"] = neg_mask

        # VAE-encode target image → image_latents (for training loss)
        if "image" in batch:
            image = batch["image"]
            bs = image.shape[0]
            h_img, w_img = batch["height"], batch["width"]
            _, image_latents = self.prepare_latents(
                image,
                bs,
                self.num_channels_latents,
                h_img,
                w_img,
                self.weight_dtype,
            )
            batch["image_latents"] = image_latents

        # img_shapes: single shape per sample (target only, no control)
        img_shapes = batch.get("img_shapes")
        if img_shapes is None:
            h_img = batch.get("height", batch["image"].shape[3] if "image" in batch else None)
            w_img = batch.get("width", batch["image"].shape[4] if "image" in batch else None)
            img_shapes = [[(3, h_img, w_img)]] * (batch["image"].shape[0] if "image" in batch else 1)
        img_shapes = self.convert_img_shapes_to_latent_space(img_shapes)
        batch["img_shapes"] = img_shapes

        return batch

    def prepare_predict_batch_data(
        self,
        prompt: str | list[str],
        height: int = 1024,
        width: int = 1024,
        negative_prompt: None | str | list[str] = None,
        guidance_scale: float = None,
        num_inference_steps: int = 50,
        true_cfg_scale: float = 4.0,
        weight_dtype=torch.bfloat16,
        image=None,  # not used for T2I – accepted for API compatibility
        controls_size=None,  # not used
        **kwargs,
    ) -> dict:
        """Prepare inference batch for text-to-image generation."""
        assert prompt is not None, "prompt is required"
        if isinstance(prompt, str):
            prompt = [prompt]

        self.weight_dtype = weight_dtype

        height, width = make_image_shape_devisible(height, width, self.vae_scale_factor)
        img_shapes = [[(3, height, width)]] * len(prompt)

        data: dict[str, Any] = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "true_cfg_scale": true_cfg_scale,
            "guidance": guidance_scale,
            "img_shapes": img_shapes,
            "n_controls": 0,
        }

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            data["negative_prompt"] = negative_prompt

        return data

    # ------------------------------------------------------------------ #
    # Training loss                                                        #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """Training loss for T2I (no control concatenation)."""
        assert self.accelerator is not None
        device = self.accelerator.device

        image_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        prompt_embeds_mask = embeddings["prompt_embeds_mask"].to(dtype=torch.int64).to(device)
        img_shapes = embeddings["img_shapes"]
        batch_size = image_latents.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(image_latents, device=device, dtype=self.weight_dtype)

            u = compute_density_for_timestep_sampling(
                weighting_scheme="none",
                batch_size=batch_size,
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            timesteps = self.scheduler.timesteps[indices].to(device=device)

            sigmas = self._get_sigmas(timesteps, n_dim=image_latents.ndim, dtype=image_latents.dtype)
            noisy_model_input = (1.0 - sigmas) * image_latents + sigmas * noise
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        # T2I: hidden_states is just the noisy target latent (no control concat)
        model_pred = self.dit(
            hidden_states=noisy_model_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        model_pred = model_pred[:, : image_latents.size(1)]
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        target = noise - image_latents
        loss = self.forward_loss(model_pred, target, weighting, edit_mask=None)
        return loss

    # ------------------------------------------------------------------ #
    # Inference / sampling                                                 #
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Cache step (no control latents)                                      #
    # ------------------------------------------------------------------ #

    def cache_step(self, data: dict):
        """Cache embeddings for T2I (no control latents)."""
        image_latents = data["image_latents"].detach().cpu()[0]
        prompt_embeds = data["prompt_embeds"].detach().cpu()[0]
        prompt_embeds_mask = data["prompt_embeds_mask"].detach().cpu()[0]
        empty_prompt_embeds = data["empty_prompt_embeds"].detach().cpu()[0]
        empty_prompt_embeds_mask = data["empty_prompt_embeds_mask"].detach().cpu()[0]

        cache_embeddings = {
            "image_latents": image_latents,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "empty_prompt_embeds": empty_prompt_embeds,
            "empty_prompt_embeds_mask": empty_prompt_embeds_mask,
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
    # Inference / sampling                                                 #
    # ------------------------------------------------------------------ #

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """Denoising loop for T2I (pure noise latents, no control concat)."""
        num_inference_steps = embeddings["num_inference_steps"]
        true_cfg_scale = embeddings["true_cfg_scale"]
        prompt_embeds = embeddings["prompt_embeds"]
        prompt_embeds_mask = embeddings["prompt_embeds_mask"]
        img_shapes = embeddings["img_shapes"]
        height_image = embeddings["height"]
        width_image = embeddings["width"]
        batch_size = prompt_embeds.shape[0]

        negative_prompt = embeddings.get("negative_prompt")
        has_neg_prompt = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        device = self.dit.device

        num_channels_latents = self.dit.config.in_channels // 4
        height_latent = 2 * (int(height_image) // (self.vae_scale_factor * 2))
        width_latent = 2 * (int(width_image) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height_latent, width_latent)
        if "latents" in embeddings:
            latents = embeddings["latents"].to(device, dtype=self.weight_dtype)
        else:
            latents = randn_tensor(shape, generator=None, device=device, dtype=self.weight_dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height_latent, width_latent)
        image_seq_len = latents.shape[1]

        logging.info(f"T2I latents shape: {latents.shape}")
        logging.info(f"img_shapes: {img_shapes}")

        timesteps, num_inference_steps = self.prepare_predict_timesteps(
            num_inference_steps, image_seq_len, scheduler=self.sampling_scheduler
        )
        self._num_timesteps = len(timesteps)

        guidance_scale = embeddings.get("guidance")
        if self.dit.config.guidance_embeds:
            if guidance_scale is None:
                raise ValueError("guidance_scale is required for guidance-distilled model")
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        if do_true_cfg:
            negative_prompt_embeds = embeddings["negative_prompt_embeds"]
            negative_prompt_embeds_mask = embeddings["negative_prompt_embeds_mask"]
            negative_txt_seq_lens = negative_prompt_embeds_mask.sum(dim=1).tolist()
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None
            negative_txt_seq_lens = None

        self.sampling_scheduler.set_begin_index(0)
        self.attention_kwargs: dict[str, Any] = {}

        prompt_embeds = prompt_embeds.to(device, dtype=self.weight_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(device, dtype=torch.int64)

        if do_true_cfg:
            negative_prompt_embeds = negative_prompt_embeds.to(device, dtype=self.weight_dtype)
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device, dtype=torch.int64)

        with torch.inference_mode():
            for _, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="T2I Generating"):
                self._current_timestep = t
                latents = latents.to(device, dtype=self.weight_dtype)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                with self.dit.cache_context("cond"):
                    noise_pred = self.dit(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    noise_pred_cpu = noise_pred.cpu()
                    torch.cuda.empty_cache()
                    with self.dit.cache_context("uncond"):
                        neg_noise_pred = self.dit(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]

                    noise_pred = noise_pred_cpu.to(device)
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)
                    del neg_noise_pred, comb_pred, cond_norm, noise_norm
                    torch.cuda.empty_cache()

                latents_dtype = latents.dtype
                latents = self.sampling_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

        self._current_timestep = None
        return latents
