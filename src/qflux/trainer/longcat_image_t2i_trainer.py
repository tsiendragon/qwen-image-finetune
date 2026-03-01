"""Trainer for meituan-longcat/LongCat-Image (text-to-image, FLUX-like MMDiT).

Architecture overview:
  - Transformer: LongCatImageTransformer2DModel (packed 2×2 patch latents)
  - VAE: AutoencoderKL, vae_scale_factor = 2^(len(block_out_channels)-1) ≈ 8
  - Text encoder: Qwen2_5_VLForConditionalGeneration + Qwen2Tokenizer
  - Scheduler: FlowMatchEulerDiscreteScheduler (with calculate_shift mu)

Encoding:
  - Prefix: "<|im_start|>system\\nAs an image captioning expert...\\n<|im_start|>user\\n"
  - Suffix: "<|im_end|>\\n<|im_start|>assistant\\n"
  - Content tokens padded to tokenizer_max_length=512
  - Passes prefix+content+suffix to text_encoder, extracts content hidden_states[-1]
  - split_quotation: quoted text is tokenized character-by-character (for text rendering)

Position IDs (LongCat uses 3D RoPE with modality embedding):
  - text_ids:  (seq_len, 3)  = [[0, i, i] for i in range(seq_len)]
  - img_ids:   (N_patches, 3) = [[1, tokenizer_max_length+row, tokenizer_max_length+col]]
    where spatial positions are offset by tokenizer_max_length (=512)

Latent format (same as OvisImage):
  - Unpacked: (B, C, H//8, W//8) ← in cache; num_channels_latents = 16 (hardcoded)
  - Packed:   (B, (H//16)*(W//16), C*4) ← passed to transformer

Timestep:
  - Training: sigma ~ Uniform(0, 1) passed directly
  - Inference: t / 1000 (same as OvisImage)

VAE decode:
  - Unpack, then: (latents / scaling_factor) + shift_factor → vae.decode()
"""

import copy
import gc
import logging
import re
from typing import Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from tqdm.auto import tqdm

from qflux.trainer.base_trainer import BaseTrainer
from qflux.utils.images import make_image_shape_devisible


logger = logging.getLogger(__name__)

PROMPT_TEMPLATE_PREFIX = (
    "<|im_start|>system\n"
    "As an image captioning expert, generate a descriptive text prompt based on an image "
    "content, suitable for input to a text-to-image model."
    "<|im_end|>\n<|im_start|>user\n"
)
PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TOKENIZER_MAX_LENGTH = 512   # content tokens
NUM_CHANNELS_LATENTS = 16    # unpacked channels (hardcoded in pipeline)


def _get_prompt_language(prompt: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", prompt) else "en"


def _split_quotation(prompt: str) -> list[tuple[str, bool]]:
    """Split prompt on quoted substrings.

    Returns list of (text, is_quoted) pairs.
    Quoted text should be tokenized character-by-character for OCR rendering.
    """
    word_internal_re = re.compile(r"[a-zA-Z]+'[a-zA-Z]+")
    matches = word_internal_re.findall(prompt)
    mapping: list[tuple[str, str]] = []
    for i, w in enumerate(set(matches)):
        placeholder = "longcat_$##$_longcat" * (i + 1)
        prompt = prompt.replace(w, placeholder)
        mapping.append((w, placeholder))

    quote_pairs = [("'", "'"), ('"', '"'), ("\u2018", "\u2019"), ("\u201c", "\u201d")]
    pattern = "|".join(
        [re.escape(q1) + r"[^" + re.escape(q1 + q2) + r"]*?" + re.escape(q2) for q1, q2 in quote_pairs]
    )
    parts = re.split(f"({pattern})", prompt)

    result = []
    for part in parts:
        for w_src, w_tgt in mapping:
            part = part.replace(w_tgt, w_src)
        if not part:
            continue
        if re.match(pattern, part):
            result.append((part, True))
        else:
            result.append((part, False))
    return result


def _calculate_shift(image_seq_len, base_seq_len=256, max_seq_len=4096,
                     base_shift=0.5, max_shift=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class LongCatImageT2ITrainer(BaseTrainer):
    """Trainer for LongCat-Image text-to-image model.

    Uses Qwen2.5-VL as a text-only encoder with prefix/suffix templates.
    Latents are packed into 2×2 patches (same as OvisImage/FLUX pattern).
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        from diffusers.pipelines.longcat_image.pipeline_longcat_image import LongCatImagePipeline
        return LongCatImagePipeline

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs):
        from diffusers.models.autoencoders import AutoencoderKL
        from diffusers.models.transformers import LongCatImageTransformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading LongCatImagePipeline components...")
        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings or {}

        # ----- VAE -----
        vae_path = pretrains.get("vae", model_path)
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            vae_path, subfolder="vae", torch_dtype=self.weight_dtype
        )

        # ----- Text encoder: Qwen2.5-VL (text-only mode) -----
        te_path = pretrains.get("text_encoder", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(te_path, subfolder="tokenizer")
        self.text_encoder: Qwen2_5_VLForConditionalGeneration = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                te_path, subfolder="text_encoder", torch_dtype=self.weight_dtype
            )
        )

        # ----- Transformer -----
        self.dit: LongCatImageTransformer2DModel = LongCatImageTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=self.weight_dtype
        )

        # ----- Scheduler -----
        self.scheduler: FlowMatchEulerDiscreteScheduler = (
            FlowMatchEulerDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        )
        self.sampling_scheduler = copy.deepcopy(self.scheduler)

        # ----- Derived parameters -----
        self.vae_scale_factor: int = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self.vae.config, "block_out_channels") else 8
        )
        self._vae_scaling_factor: float = self.vae.config.scaling_factor
        self._vae_shift_factor: float = getattr(self.vae.config, "shift_factor", 0.0)
        self.num_channels_latents: int = NUM_CHANNELS_LATENTS
        self.tokenizer_max_length: int = TOKENIZER_MAX_LENGTH

        # Pre-compute prefix/suffix token ids
        self._prefix_ids = self.tokenizer(PROMPT_TEMPLATE_PREFIX, add_special_tokens=False)["input_ids"]
        self._suffix_ids = self.tokenizer(PROMPT_TEMPLATE_SUFFIX, add_special_tokens=False)["input_ids"]

        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logger.info(
            f"LongCatImage loaded. vae_scale={self.vae_scale_factor}, "
            f"latent_channels={self.num_channels_latents}, "
            f"prefix_len={len(self._prefix_ids)}, suffix_len={len(self._suffix_ids)}"
        )

    # ------------------------------------------------------------------ #
    # Packed latent helpers (identical to OvisImage)                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
        H = 2 * (int(height) // (vae_scale_factor * 2))
        W = 2 * (int(width) // (vae_scale_factor * 2))
        B, N, C4 = latents.shape
        latents = latents.view(B, H // 2, W // 2, C4 // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(B, C4 // 4, H, W)
        return latents

    @staticmethod
    def _prepare_latent_image_ids(
        h_patch: int, w_patch: int, offset: int, device, dtype
    ) -> torch.Tensor:
        """Build img_ids (h_patch*w_patch, 3): [1, offset+row, offset+col]."""
        ids = torch.zeros(h_patch, w_patch, 3)
        ids[..., 0] = 1  # modality_id = 1 for image
        ids[..., 1] += (torch.arange(h_patch)[:, None].float() + offset)
        ids[..., 2] += (torch.arange(w_patch)[None, :].float() + offset)
        return ids.reshape(h_patch * w_patch, 3).to(device=device, dtype=dtype)

    @staticmethod
    def _prepare_text_ids(seq_len: int, device, dtype) -> torch.Tensor:
        """Build text_ids (seq_len, 3): [0, i, i]."""
        ids = torch.zeros(seq_len, 3)
        ids[:, 1] = torch.arange(seq_len).float()
        ids[:, 2] = torch.arange(seq_len).float()
        return ids.to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Prompt encoding                                                      #
    # ------------------------------------------------------------------ #

    def _tokenize_prompt_content(self, prompt: str) -> list[int]:
        """Tokenize prompt content with split_quotation (char-level for quoted text)."""
        all_tokens: list[int] = []
        for text, is_quoted in _split_quotation(prompt):
            if is_quoted:
                for ch in text:
                    all_tokens.extend(
                        self.tokenizer(ch, add_special_tokens=False)["input_ids"]
                    )
            else:
                all_tokens.extend(
                    self.tokenizer(text, add_special_tokens=False)["input_ids"]
                )
        return all_tokens[:TOKENIZER_MAX_LENGTH]

    def encode_prompt(
        self,
        prompt: str | list[str],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompts with Qwen2.5-VL (text-only, prefix+suffix template).

        Returns:
            prompt_embeds: (B, TOKENIZER_MAX_LENGTH=512, hidden_dim)
            text_ids:      (512, 3)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        device = next(self.text_encoder.parameters()).device

        batch_content_ids: list[list[int]] = []
        for p in prompt:
            batch_content_ids.append(self._tokenize_prompt_content(p))

        # Pad content to max_length
        tokens_and_mask = self.tokenizer.pad(
            {"input_ids": batch_content_ids},
            max_length=TOKENIZER_MAX_LENGTH,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        B = tokens_and_mask.input_ids.shape[0]
        dtype_ids = tokens_and_mask.input_ids.dtype
        dtype_mask = tokens_and_mask.attention_mask.dtype

        prefix_t = torch.tensor(self._prefix_ids, dtype=dtype_ids).unsqueeze(0).expand(B, -1)
        suffix_t = torch.tensor(self._suffix_ids, dtype=dtype_ids).unsqueeze(0).expand(B, -1)
        prefix_m = torch.ones(B, len(self._prefix_ids), dtype=dtype_mask)
        suffix_m = torch.ones(B, len(self._suffix_ids), dtype=dtype_mask)

        input_ids = torch.cat([prefix_t, tokens_and_mask.input_ids, suffix_t], dim=-1).to(device)
        attention_mask = torch.cat([prefix_m, tokens_and_mask.attention_mask, suffix_m], dim=-1).to(device)

        with torch.inference_mode():
            out = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        # Extract content slice (drop prefix and suffix)
        pl = len(self._prefix_ids)
        sl = len(self._suffix_ids)
        prompt_embeds = out.hidden_states[-1][:, pl:-sl, :].detach()
        prompt_embeds = prompt_embeds.to(dtype=self.weight_dtype, device=device)

        text_ids = self._prepare_text_ids(prompt_embeds.shape[1], device=device, dtype=self.weight_dtype)
        return prompt_embeds, text_ids

    # ------------------------------------------------------------------ #
    # VAE                                                                  #
    # ------------------------------------------------------------------ #

    def _vae_encode(self, image: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return self.vae.encode(image).latent_dist.sample()

    def prepare_latents(self, image, batch_size, num_channels_latents, height, width, dtype):
        h_lat = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_lat = 2 * (int(width) // (self.vae_scale_factor * 2))
        device = next(self.vae.parameters()).device
        noise = randn_tensor((batch_size, num_channels_latents, h_lat, w_lat), device=device, dtype=dtype)
        image_latents = None
        if image is not None:
            image_latents = self._vae_encode(image.to(device=device, dtype=dtype))
        return noise, image_latents

    def decode_vae_latent(self, latents: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        latents = self._unpack_latents(latents, target_height, target_width, self.vae_scale_factor)
        latents = latents.to(self.vae.device, dtype=self.vae.dtype)
        latents = (latents / self._vae_scaling_factor) + self._vae_shift_factor
        with torch.inference_mode():
            image = self.vae.decode(latents, return_dict=False)[0]
        return (image / 2 + 0.5).clamp(0, 1)

    # ------------------------------------------------------------------ #
    # Embedding preparation                                                #
    # ------------------------------------------------------------------ #

    def prepare_embeddings(self, batch: dict, stage: str = "fit") -> dict:
        device = next(self.text_encoder.parameters()).device

        pe, text_ids = self.encode_prompt(batch["prompt"])
        batch["prompt_embeds"] = pe
        batch["text_ids"] = text_ids

        if stage == "cache":
            empty_pe, empty_text_ids = self.encode_prompt([""])
            batch["empty_prompt_embeds"] = empty_pe
            batch["empty_text_ids"] = empty_text_ids

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
        return batch

    # ------------------------------------------------------------------ #
    # Training loss                                                        #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        assert self.accelerator is not None
        device = self.accelerator.device

        image_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        B, C, H, W = image_latents.shape

        with torch.no_grad():
            noise = torch.randn_like(image_latents)
            sigma = torch.rand(B, device=device, dtype=self.weight_dtype)
            s = sigma[:, None, None, None]
            noisy = (1.0 - s) * image_latents + s * noise

        noisy_packed = self._pack_latents(noisy)
        image_packed = self._pack_latents(image_latents)
        noise_packed = self._pack_latents(noise)

        h_patch, w_patch = H // 2, W // 2
        text_ids = self._prepare_text_ids(prompt_embeds.shape[1], device=device, dtype=self.weight_dtype)
        img_ids = self._prepare_latent_image_ids(
            h_patch, w_patch, offset=TOKENIZER_MAX_LENGTH, device=device, dtype=self.weight_dtype
        )

        model_pred = self.dit(
            hidden_states=noisy_packed,
            timestep=sigma,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

        target = noise_packed - image_packed
        return self.forward_loss(model_pred, target)

    # ------------------------------------------------------------------ #
    # Cache step                                                           #
    # ------------------------------------------------------------------ #

    def cache_step(self, data: dict):
        cache_embeddings = {
            "image_latents": data["image_latents"].detach().cpu()[0],
            "prompt_embeds": data["prompt_embeds"].detach().cpu()[0],
            "empty_prompt_embeds": data["empty_prompt_embeds"].detach().cpu()[0],
        }
        map_keys = {
            "image_latents": "image_hash",
            "prompt_embeds": "prompt_hash",
            "empty_prompt_embeds": "prompt_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    # ------------------------------------------------------------------ #
    # Device management                                                    #
    # ------------------------------------------------------------------ #

    def setup_model_device_train_mode(self, stage: str = "fit", cache: bool = False):
        if stage == "fit":
            assert hasattr(self, "accelerator")
            if self.cache_exist and self.use_cache:
                for attr in ("text_encoder", "vae"):
                    if hasattr(self, attr):
                        getattr(self, attr).cpu()
                if not self.config.validation.enabled:
                    for attr in ("text_encoder", "vae"):
                        if hasattr(self, attr):
                            delattr(self, attr)
                gc.collect(); torch.cuda.empty_cache()
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
            self.vae.to(d.vae); self.text_encoder.to(d.text_encoder); self.dit.to(d.dit)
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
        guidance_scale: float = 4.5,
        num_inference_steps: int = 50,
        weight_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> dict:
        if isinstance(prompt, str):
            prompt = [prompt]
        self.weight_dtype = weight_dtype
        height, width = make_image_shape_devisible(height, width, self.vae_scale_factor * 2)
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
        """Denoising loop for LongCat-Image. Returns final packed latents (B, N, C*4)."""
        num_steps = embeddings["num_inference_steps"]
        guidance_scale = embeddings.get("guidance_scale", 4.5)
        prompt_embeds = embeddings["prompt_embeds"]
        height = embeddings["height"]
        width = embeddings["width"]
        batch_size = prompt_embeds.shape[0]
        device = self.dit.device
        dtype = self.weight_dtype

        prompt_embeds = prompt_embeds.to(device, dtype=dtype)

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg = embeddings.get("negative_prompt", [""] * batch_size)
            neg_embeds, neg_text_ids = self.encode_prompt(neg)
            neg_embeds = neg_embeds.to(device, dtype=dtype)

        h_lat = 2 * (int(height) // (self.vae_scale_factor * 2))
        w_lat = 2 * (int(width) // (self.vae_scale_factor * 2))

        if "latents" in embeddings:
            raw = embeddings["latents"].to(device, dtype=torch.float32)
            latents = self._pack_latents(raw)
        else:
            raw = randn_tensor(
                (batch_size, self.num_channels_latents, h_lat, w_lat),
                device=device, dtype=torch.float32,
            )
            latents = self._pack_latents(raw)

        h_patch, w_patch = h_lat // 2, w_lat // 2
        text_ids = self._prepare_text_ids(prompt_embeds.shape[1], device=device, dtype=dtype)
        img_ids = self._prepare_latent_image_ids(
            h_patch, w_patch, offset=TOKENIZER_MAX_LENGTH, device=device, dtype=dtype
        )
        if do_cfg:
            neg_text_ids = self._prepare_text_ids(neg_embeds.shape[1], device=device, dtype=dtype)

        image_seq_len = latents.shape[1]
        mu = _calculate_shift(
            image_seq_len,
            self.sampling_scheduler.config.get("base_image_seq_len", 256),
            self.sampling_scheduler.config.get("max_image_seq_len", 4096),
            self.sampling_scheduler.config.get("base_shift", 0.5),
            self.sampling_scheduler.config.get("max_shift", 1.15),
        )
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        self.sampling_scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)
        timesteps = self.sampling_scheduler.timesteps
        self.sampling_scheduler.set_begin_index(0)

        with torch.inference_mode():
            for t in tqdm(timesteps, desc="LongCatImage generating"):
                timestep = t.expand(batch_size).to(dtype)

                noise_pred = self.dit(
                    hidden_states=latents.to(dtype),
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                if do_cfg:
                    neg_pred = self.dit(
                        hidden_states=latents.to(dtype),
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=neg_embeds,
                        txt_ids=neg_text_ids,
                        img_ids=img_ids,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

                latents = self.sampling_scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]

        return latents
