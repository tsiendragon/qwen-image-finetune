"""Trainer for briaai/fibo-edit (image-to-image editing, flow matching).

Extends BriaFiboT2ITrainer with source image conditioning:
  - Source image encoded by VAE → normalized latents → packed to sequence
  - Source img_ids have first dimension set to 1 (marks source vs target)
  - Model input: cat([target_latents, source_latents], dim=1) along sequence dim
  - Model output sliced to [:, :target_seq_len, :] (only target predictions used)

For all other details (text encoder, VAE, packing, etc.) see bria_fibo_t2i_trainer.py.
"""

import logging

import torch

from qflux.trainer.bria_fibo_t2i_trainer import BriaFiboT2ITrainer


logger = logging.getLogger(__name__)


class BriaFiboEditTrainer(BriaFiboT2ITrainer):
    """Trainer for Bria FIBO image editing (IT2I) model.

    Source image (control) is encoded, packed, and concatenated with the noisy
    target latents along the sequence dimension before passing to the transformer.
    The model predicts only over the target sequence (first half of the input).
    """

    def get_pipeline_class(self):
        from diffusers.pipelines.bria_fibo.pipeline_bria_fibo_edit import BriaFiboEditPipeline
        return BriaFiboEditPipeline

    # ------------------------------------------------------------------ #
    # Source image helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _prepare_source_image_ids(H: int, W: int, device, dtype) -> torch.Tensor:
        """Build source img_ids (H*W, 3) with values [1, row, col].

        First dimension = 1 marks these as source/reference positions,
        distinct from target positions which use [0, row, col].
        """
        ids = torch.zeros(H, W, 3)
        ids[..., 0] = 1.0  # source modality marker
        ids[..., 1] = torch.arange(H, dtype=torch.float32).unsqueeze(1)
        ids[..., 2] = torch.arange(W, dtype=torch.float32).unsqueeze(0)
        return ids.reshape(H * W, 3).to(device=device, dtype=dtype)

    def _encode_source_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode source/control image (B, 3, H, W) → normalized 4D latents."""
        return self._vae_encode(image)

    # ------------------------------------------------------------------ #
    # Embedding preparation                                               #
    # ------------------------------------------------------------------ #

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
        batch = super().prepare_embeddings(batch, stage=stage)

        # Encode control/source image
        if "control" in batch:
            control = batch["control"]
            if isinstance(control, torch.Tensor) and control.ndim == 5:
                control = control.squeeze(2)
            if control.max() <= 1.0 + 1e-6:
                control = control * 2.0 - 1.0
            device = next(self.text_encoder.parameters()).device
            control = control.to(device=device, dtype=self.weight_dtype)
            with torch.inference_mode():
                batch["control_latents"] = self._encode_source_image(control)

        return batch

    # ------------------------------------------------------------------ #
    # Cache step                                                           #
    # ------------------------------------------------------------------ #

    def cache_step(self, data: dict):
        """Cache image_latents, control_latents, prompt_embeds, and mask."""
        cache_embeddings = {
            "image_latents": data["image_latents"].detach().cpu()[0],
            "control_latents": data["control_latents"].detach().cpu()[0],
            "prompt_embeds": data["prompt_embeds"].detach().cpu()[0],
            "prompt_embeds_mask": data["prompt_embeds_mask"].detach().cpu()[0],
            "empty_prompt_embeds": data["empty_prompt_embeds"].detach().cpu()[0],
            "empty_prompt_embeds_mask": data["empty_prompt_embeds_mask"].detach().cpu()[0],
        }
        map_keys = {
            "image_latents": "image_hash",
            "control_latents": "control_hash",
            "prompt_embeds": "prompt_hash",
            "prompt_embeds_mask": "prompt_hash",
            "empty_prompt_embeds": "prompt_hash",
            "empty_prompt_embeds_mask": "prompt_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    # ------------------------------------------------------------------ #
    # Training loss                                                        #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        assert self.accelerator is not None
        device = self.accelerator.device

        image_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        control_latents = embeddings["control_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        attn_mask = embeddings["prompt_embeds_mask"].to(device)
        B, C, H, W = image_latents.shape

        layers = embeddings.get("text_encoder_layers")
        if layers is None:
            _, layers, _ = self.encode_prompt(embeddings["prompt"])
        layers = [l.to(self.weight_dtype).to(device) for l in layers]

        with torch.no_grad():
            noise = torch.randn_like(image_latents)
            sigma = torch.rand(B, device=device, dtype=self.weight_dtype)
            s = sigma[:, None, None, None]
            noisy = (1.0 - s) * image_latents + s * noise
            target = noise - image_latents  # velocity target

        # Pack target (noisy) and source latents to sequences
        noisy_packed = self._pack_latents(noisy)         # (B, H*W, C)
        source_packed = self._pack_latents(control_latents)  # (B, H*W, C)
        # Concatenate along sequence dimension
        latent_model_input = torch.cat([noisy_packed, source_packed], dim=1)  # (B, 2*H*W, C)

        target_seq_len = noisy_packed.shape[1]
        img_ids = self._prepare_image_ids(H, W, device=device, dtype=self.weight_dtype)
        src_ids = self._prepare_source_image_ids(H, W, device=device, dtype=self.weight_dtype)
        combined_img_ids = torch.cat([img_ids, src_ids], dim=0)  # (2*H*W, 3)

        txt_ids = self._prepare_text_ids(B, prompt_embeds.shape[1], device=device, dtype=self.weight_dtype)
        attn_matrix = self._build_attention_matrix(attn_mask)

        model_pred = self.dit(
            hidden_states=latent_model_input,
            timestep=sigma,
            encoder_hidden_states=prompt_embeds,
            text_encoder_layers=layers,
            txt_ids=txt_ids,
            img_ids=combined_img_ids,
            joint_attention_kwargs={"attention_mask": attn_matrix},
            return_dict=False,
        )[0]

        # Slice to target sequence only
        model_pred_packed = model_pred[:, :target_seq_len, :]
        model_pred_4d = self._unpack_latents(model_pred_packed, H, W)
        return self.forward_loss(model_pred_4d, target)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """Denoising loop for Bria FIBO Edit. Returns decoded image in [0, 1]."""
        import numpy as np
        from diffusers.utils.torch_utils import randn_tensor
        from tqdm.auto import tqdm

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

        # Source (control) latents
        control_latents = embeddings.get("control_latents")
        assert control_latents is not None, "control_latents required for FIBO Edit"
        control_latents = control_latents.to(device, dtype=dtype)
        source_packed = self._pack_latents(control_latents)  # (B, H*W, C)

        if "latents" in embeddings:
            latents = embeddings["latents"].to(device, dtype=torch.float32)
        else:
            latents = randn_tensor(
                (batch_size, self.num_channels_latents, h_lat, w_lat),
                device=device, dtype=torch.float32,
            )

        from qflux.trainer.bria_fibo_t2i_trainer import _calculate_shift
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
        src_ids = self._prepare_source_image_ids(h_lat, w_lat, device=device, dtype=dtype)
        combined_img_ids = torch.cat([img_ids, src_ids], dim=0)  # (2*H*W, 3)

        with torch.inference_mode():
            for t in tqdm(timesteps, desc="Bria FIBO Edit generating"):
                noisy_packed = self._pack_latents(latents.to(dtype))
                lmi = torch.cat([noisy_packed, source_packed], dim=1)  # (B, 2*H*W, C)
                target_seq_len = noisy_packed.shape[1]

                txt_ids = self._prepare_text_ids(batch_size, prompt_embeds.shape[1], device=device, dtype=dtype)
                timestep = t.expand(batch_size).to(dtype)

                noise_pred = self.dit(
                    hidden_states=lmi,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    text_encoder_layers=text_layers,
                    txt_ids=txt_ids,
                    img_ids=combined_img_ids,
                    joint_attention_kwargs={"attention_mask": attn_matrix},
                    return_dict=False,
                )[0]
                # Slice target predictions
                noise_pred_4d = self._unpack_latents(noise_pred[:, :target_seq_len, :], h_lat, w_lat)

                if do_cfg:
                    neg_txt_ids = self._prepare_text_ids(batch_size, neg_embeds.shape[1], device=device, dtype=dtype)
                    neg_pred = self.dit(
                        hidden_states=lmi,
                        timestep=timestep,
                        encoder_hidden_states=neg_embeds,
                        text_encoder_layers=neg_layers,
                        txt_ids=neg_txt_ids,
                        img_ids=combined_img_ids,
                        joint_attention_kwargs={"attention_mask": neg_attn_matrix},
                        return_dict=False,
                    )[0]
                    neg_pred_4d = self._unpack_latents(neg_pred[:, :target_seq_len, :], h_lat, w_lat)
                    noise_pred_4d = neg_pred_4d + guidance_scale * (noise_pred_4d - neg_pred_4d)

                latents = self.sampling_scheduler.step(
                    noise_pred_4d.to(torch.float32), t, latents, return_dict=False
                )[0]

        return self.decode_vae_latent(
            self._pack_latents(latents.to(dtype)), target_height=height, target_width=width
        )
