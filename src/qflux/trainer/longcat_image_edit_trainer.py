"""Trainer for meituan-longcat/LongCat-Image-Edit (image-to-image editing, FLUX-like MMDiT).

Architecture overview:
  - Transformer: LongCatImageTransformer2DModel (packed 2×2 patch latents)
  - VAE: AutoencoderKL (same as T2I)
  - Text encoder: Qwen2_5_VLForConditionalGeneration + Qwen2Tokenizer + Qwen2VLProcessor
  - Scheduler: FlowMatchEulerDiscreteScheduler (with calculate_shift mu)

Encoding (multimodal):
  - Prefix: "<|im_start|>system\\n...<|im_end|>\\n<|im_start|>user\\n<|vision_start|><|image_pad|>×N<|vision_end|>"
  - Suffix: "<|im_end|>\\n<|im_start|>assistant\\n"
  - Source image (half-res) → Qwen2VLProcessor → pixel_values, image_grid_thw
  - Image tokens expand: <|image_pad|> → N copies where N = image_grid_thw.prod() // merge_size**2
  - prefix_len = index of <|vision_start|> in prefix_tokens (drops system header)
  - hidden_states[-1][:, prefix_len:-suffix_len, :] — keeps vision_start + image tokens + vision_end + content
  - split_quotation: quoted text is tokenized character-by-character

Position IDs:
  - text_ids: (seq_len, 3) = [[0, i, i] for i in range(seq_len)]
  - target_img_ids: (N_patches, 3) = [[1, seq_len+row, seq_len+col]]
  - source_img_ids: (N_patches, 3) = [[2, seq_len+row, seq_len+col]]
  - combined_img_ids = cat([target_img_ids, source_img_ids], dim=0)

Transformer call:
  - latent_model_input = cat([noisy_packed, source_packed], dim=1)
  - img_ids = combined_img_ids (includes both target and source tokens)
  - model_pred = dit(...)[0][:, :image_seq_len]  ← slice to target only

VAE:
  - Source encode: (latents - shift_factor) * scaling_factor  (argmax / mode)
  - Target encode: latent_dist.sample() * scaling_factor  (for training)
  - Decode: (latents / scaling_factor) + shift_factor

Training data format:
  - batch["image"]   = target image (after editing), (B, 3, H, W) in [0, 1]
  - batch["control"] = source image (before editing), (B, 3, H, W) in [0, 1]
  - batch["prompt"]  = edit instruction text
"""

import copy
import gc
import hashlib
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

EDIT_SYSTEM_PROMPT = (
    "As an image editing expert, first analyze the content and attributes of the input image(s). "
    "Then, based on the user's editing instructions, clearly and precisely determine how to modify "
    "the given image(s), ensuring that only the specified parts are altered and all other aspects "
    "remain consistent with the original(s)."
)
PROMPT_TEMPLATE_PREFIX = (
    f"<|im_start|>system\n{EDIT_SYSTEM_PROMPT}<|im_end|>\n"
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
)
PROMPT_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
IMAGE_TOKEN = "<|image_pad|>"
TOKENIZER_MAX_LENGTH = 512   # content tokens (not counting vision tokens)
NUM_CHANNELS_LATENTS = 16    # hardcoded in pipeline


def _get_prompt_language(prompt: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", prompt) else "en"


def _split_quotation(prompt: str) -> list[tuple[str, bool]]:
    """Split prompt on quoted substrings for char-level tokenization of OCR text."""
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


class LongCatImageEditTrainer(BaseTrainer):
    """Trainer for LongCat-Image-Edit image-to-image editing model.

    Source image is provided both as visual context for the text encoder
    (via Qwen2VLProcessor at half resolution) and as latent conditioning
    for the transformer (via VAE at full resolution).

    Training data: batch["image"] = target, batch["control"] = source,
    batch["prompt"] = edit instruction.
    """

    def __init__(self, config):
        super().__init__(config)

    def get_pipeline_class(self):
        from diffusers.pipelines.longcat_image.pipeline_longcat_image_edit import (
            LongCatImageEditPipeline,
        )
        return LongCatImageEditPipeline

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def load_model(self, **kwargs):
        from diffusers.models.autoencoders import AutoencoderKL
        from diffusers.models.transformers import LongCatImageTransformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info("Loading LongCatImageEditPipeline components...")
        model_path = self.config.model.pretrained_model_name_or_path
        pretrains = self.config.model.pretrained_embeddings or {}

        # ----- VAE -----
        vae_path = pretrains.get("vae", model_path)
        self.vae: AutoencoderKL = AutoencoderKL.from_pretrained(
            vae_path, subfolder="vae", torch_dtype=self.weight_dtype
        )

        # ----- Text encoder: Qwen2.5-VL (multimodal mode) -----
        te_path = pretrains.get("text_encoder", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(te_path, subfolder="tokenizer")
        self.vl_processor = Qwen2VLProcessor.from_pretrained(te_path, subfolder="text_processor")
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

        # Pre-compute prefix and suffix token ids (before image expansion)
        self._suffix_ids = self.tokenizer(PROMPT_TEMPLATE_SUFFIX, add_special_tokens=False)["input_ids"]
        self._vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self._merge_size = getattr(self.vl_processor.image_processor, "merge_size", 2)

        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logger.info(
            f"LongCatImageEdit loaded. vae_scale={self.vae_scale_factor}, "
            f"latent_channels={self.num_channels_latents}, "
            f"suffix_len={len(self._suffix_ids)}, "
            f"merge_size={self._merge_size}"
        )

    # ------------------------------------------------------------------ #
    # Packed latent helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, (H//2)*(W//2), C*4)."""
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // 2, 2, W // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(B, (H // 2) * (W // 2), C * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents: torch.Tensor, height: int, width: int, vae_scale_factor: int) -> torch.Tensor:
        """(B, N, C*4) → (B, C, H, W)."""
        H = 2 * (int(height) // (vae_scale_factor * 2))
        W = 2 * (int(width) // (vae_scale_factor * 2))
        B, N, C4 = latents.shape
        latents = latents.view(B, H // 2, W // 2, C4 // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(B, C4 // 4, H, W)
        return latents

    @staticmethod
    def _prepare_pos_ids_text(seq_len: int, device, dtype) -> torch.Tensor:
        """Build text_ids (seq_len, 3): [0, i, i]."""
        ids = torch.zeros(seq_len, 3)
        ids[:, 1] = torch.arange(seq_len).float()
        ids[:, 2] = torch.arange(seq_len).float()
        return ids.to(device=device, dtype=dtype)

    @staticmethod
    def _prepare_pos_ids_image(
        h_patch: int, w_patch: int, modality_id: int, offset: int, device, dtype
    ) -> torch.Tensor:
        """Build img_ids (h_patch*w_patch, 3): [modality_id, offset+row, offset+col]."""
        ids = torch.zeros(h_patch, w_patch, 3)
        ids[..., 0] = modality_id
        ids[..., 1] += (torch.arange(h_patch)[:, None].float() + offset)
        ids[..., 2] += (torch.arange(w_patch)[None, :].float() + offset)
        return ids.reshape(h_patch * w_patch, 3).to(device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Prompt encoding (multimodal)                                         #
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

    def _build_prefix_tokens(self, image_grid_thw: torch.Tensor) -> tuple[list[int], int]:
        """Expand image tokens in prefix template and compute prefix_len.

        Returns:
            prefix_tokens: list of token ids (with expanded image tokens)
            prefix_len: index of <|vision_start|> (tokens to drop from hidden_states)
        """
        merge_length = self._merge_size ** 2
        num_image_tokens = int(image_grid_thw.prod().item()) // merge_length

        text = PROMPT_TEMPLATE_PREFIX
        text = text.replace(IMAGE_TOKEN, IMAGE_TOKEN * num_image_tokens, 1)

        prefix_tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        prefix_len = prefix_tokens.index(self._vision_start_token_id)
        return prefix_tokens, prefix_len

    def encode_prompt(
        self,
        prompt: str | list[str],
        source_image: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt with source image using Qwen2.5-VL multimodal encoder.

        Args:
            prompt: text prompt(s)
            source_image: (B, 3, H, W) tensor in [0, 1], or None for empty encoding

        Returns:
            prompt_embeds: (B, seq_len, hidden_dim) where seq_len includes vision tokens + content
            text_ids:      (seq_len, 3) position IDs
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        B = len(prompt)
        device = next(self.text_encoder.parameters()).device

        # Prepare source image for VL processor (half resolution, as PIL)
        if source_image is not None:
            # source_image: (B, 3, H, W) in [0, 1]
            src = source_image
            if src.max() > 1.0 + 1e-6:
                src = (src + 1.0) / 2.0  # if [-1, 1] → [0, 1]
            # Resize to half resolution for VL
            _, _, H, W = src.shape
            src_half = torch.nn.functional.interpolate(
                src, size=(H // 2, W // 2), mode="bilinear", align_corners=False
            )
            # Convert to PIL list
            import PIL.Image
            pil_images = []
            for b in range(B):
                arr = (src_half[b].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype("uint8")
                pil_images.append(PIL.Image.fromarray(arr))
        else:
            # No source image: create a blank image for each sample
            import PIL.Image
            pil_images = [PIL.Image.new("RGB", (512, 512), color=(128, 128, 128))] * B

        # Tokenize content
        batch_content_ids: list[list[int]] = []
        for p in prompt:
            batch_content_ids.append(self._tokenize_prompt_content(p))

        # Pad content to TOKENIZER_MAX_LENGTH
        content_tokens_and_mask = self.tokenizer.pad(
            {"input_ids": batch_content_ids},
            max_length=TOKENIZER_MAX_LENGTH,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Process each sample (may have different image grid sizes)
        all_embeds = []
        for b in range(B):
            # VL processor for this sample
            vl_out = self.vl_processor.image_processor(images=[pil_images[b]], return_tensors="pt")
            pixel_values = vl_out["pixel_values"].to(device=device, dtype=self.weight_dtype)
            image_grid_thw = vl_out["image_grid_thw"].to(device=device)

            # Build expanded prefix
            prefix_tokens, prefix_len = self._build_prefix_tokens(image_grid_thw[0])
            suffix_tokens = self._suffix_ids

            # Assemble input_ids for this sample
            dtype_ids = content_tokens_and_mask.input_ids.dtype
            dtype_mask = content_tokens_and_mask.attention_mask.dtype
            prefix_t = torch.tensor(prefix_tokens, dtype=dtype_ids).unsqueeze(0).to(device)
            suffix_t = torch.tensor(suffix_tokens, dtype=dtype_ids).unsqueeze(0).to(device)
            content_t = content_tokens_and_mask.input_ids[b:b+1].to(device)
            content_m = content_tokens_and_mask.attention_mask[b:b+1].to(device)

            prefix_m = torch.ones(1, len(prefix_tokens), dtype=dtype_mask, device=device)
            suffix_m = torch.ones(1, len(suffix_tokens), dtype=dtype_mask, device=device)

            input_ids = torch.cat([prefix_t, content_t, suffix_t], dim=-1)
            attention_mask = torch.cat([prefix_m, content_m, suffix_m], dim=-1)

            with torch.inference_mode():
                out = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    output_hidden_states=True,
                )

            # Extract content slice: from vision_start to before suffix
            sl = len(suffix_tokens)
            embeds = out.hidden_states[-1][:, prefix_len:-sl, :].detach()
            all_embeds.append(embeds.to(dtype=self.weight_dtype, device=device))

        # For batching, pad to max seq_len if different (usually same for fixed resolution)
        max_seq = max(e.shape[1] for e in all_embeds)
        if max_seq > all_embeds[0].shape[1] or B > 1:
            padded = []
            for e in all_embeds:
                pad_len = max_seq - e.shape[1]
                if pad_len > 0:
                    e = torch.nn.functional.pad(e, (0, 0, 0, pad_len))
                padded.append(e)
            prompt_embeds = torch.cat(padded, dim=0)
        else:
            prompt_embeds = torch.cat(all_embeds, dim=0)

        text_ids = self._prepare_pos_ids_text(
            prompt_embeds.shape[1], device=device, dtype=self.weight_dtype
        )
        return prompt_embeds, text_ids

    # ------------------------------------------------------------------ #
    # VAE                                                                  #
    # ------------------------------------------------------------------ #

    def _vae_encode_target(self, image: torch.Tensor) -> torch.Tensor:
        """Encode target image (B, 3, H, W) in [-1, 1] → (B, C, H_lat, W_lat) unpacked.

        Uses sample() for training stochasticity.
        """
        with torch.inference_mode():
            return self.vae.encode(image).latent_dist.sample()

    def _vae_encode_source(self, image: torch.Tensor) -> torch.Tensor:
        """Encode source image (B, 3, H, W) in [-1, 1] → (B, C, H_lat, W_lat) unpacked.

        Uses mode() (argmax) to match pipeline inference behavior.
        Applies (latents - shift_factor) * scaling_factor.
        """
        with torch.inference_mode():
            latents = self.vae.encode(image).latent_dist.mode()
        latents = (latents - self._vae_shift_factor) * self._vae_scaling_factor
        return latents

    def decode_vae_latent(self, latents: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Decode packed latents → RGB (B, 3, H, W) in [0, 1]."""
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

        # Prepare source image
        source_image = batch.get("control", batch.get("image"))
        if isinstance(source_image, torch.Tensor) and source_image.ndim == 5:
            source_image = source_image.squeeze(2)
        source_tensor_01 = source_image
        if source_tensor_01.max() > 1.0 + 1e-6:
            source_tensor_01 = (source_tensor_01 + 1.0) / 2.0
        source_neg1_1 = source_tensor_01 * 2.0 - 1.0
        source_tensor_01 = source_tensor_01.to(device=device)
        source_neg1_1 = source_neg1_1.to(device=device, dtype=self.weight_dtype)

        # Encode prompt with source image (multimodal)
        pe, text_ids = self.encode_prompt(batch["prompt"], source_image=source_tensor_01)
        batch["prompt_embeds"] = pe
        batch["text_ids"] = text_ids

        if stage == "cache":
            empty_pe, empty_text_ids = self.encode_prompt(
                [""] * len(batch["prompt"]), source_image=source_tensor_01
            )
            batch["empty_prompt_embeds"] = empty_pe
            batch["empty_text_ids"] = empty_text_ids

        # VAE encode source for latent conditioning
        batch["source_latents"] = self._vae_encode_source(source_neg1_1)

        # VAE encode target
        if "image" in batch:
            target_image = batch["image"]
            if isinstance(target_image, torch.Tensor) and target_image.ndim == 5:
                target_image = target_image.squeeze(2)
            if target_image.max() <= 1.0 + 1e-6:
                target_image = target_image * 2.0 - 1.0
            target_image = target_image.to(device=device, dtype=self.weight_dtype)
            batch["height"] = target_image.shape[2]
            batch["width"] = target_image.shape[3]
            batch["target_latents"] = self._vae_encode_target(target_image)

        return batch

    def prepare_cached_embeddings(self, batch: dict) -> dict:
        return batch

    # ------------------------------------------------------------------ #
    # Training loss                                                        #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        assert self.accelerator is not None
        device = self.accelerator.device

        target_latents = embeddings["target_latents"].to(self.weight_dtype).to(device)
        source_latents = embeddings["source_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        B, C, H, W = target_latents.shape

        with torch.no_grad():
            noise = torch.randn_like(target_latents)
            sigma = torch.rand(B, device=device, dtype=self.weight_dtype)
            s = sigma[:, None, None, None]
            noisy_target = (1.0 - s) * target_latents + s * noise

        # Pack latents
        noisy_packed = self._pack_latents(noisy_target)    # (B, N, C*4)
        target_packed = self._pack_latents(target_latents) # (B, N, C*4)
        noise_packed = self._pack_latents(noise)           # (B, N, C*4)
        source_packed = self._pack_latents(source_latents) # (B, N, C*4)

        image_seq_len = noisy_packed.shape[1]
        h_patch, w_patch = H // 2, W // 2
        seq_len = prompt_embeds.shape[1]

        # Position IDs
        text_ids = self._prepare_pos_ids_text(seq_len, device=device, dtype=self.weight_dtype)
        target_img_ids = self._prepare_pos_ids_image(
            h_patch, w_patch, modality_id=1, offset=seq_len, device=device, dtype=self.weight_dtype
        )
        source_img_ids = self._prepare_pos_ids_image(
            h_patch, w_patch, modality_id=2, offset=seq_len, device=device, dtype=self.weight_dtype
        )
        combined_img_ids = torch.cat([target_img_ids, source_img_ids], dim=0)

        # Model input: cat target and source along token dimension
        latent_model_input = torch.cat([noisy_packed, source_packed], dim=1)

        model_pred = self.dit(
            hidden_states=latent_model_input,
            timestep=sigma,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=combined_img_ids,
            return_dict=False,
        )[0]
        # Slice: only the target tokens
        model_pred = model_pred[:, :image_seq_len]

        target = noise_packed - target_packed
        return self.forward_loss(model_pred, target)

    # ------------------------------------------------------------------ #
    # Cache step                                                           #
    # ------------------------------------------------------------------ #

    def cache_step(self, data: dict):
        # Use a combined hash for multimodal prompt_embeds (text + source image)
        control_h = data["file_hashes"].get("control_hash", data["file_hashes"].get("image_hash", ""))
        prompt_h = data["file_hashes"].get("prompt_hash", "")
        combined_hash = hashlib.md5(f"{control_h}|{prompt_h}".encode()).hexdigest()

        file_hashes = dict(data["file_hashes"])
        file_hashes["combined_hash"] = combined_hash

        cache_embeddings = {
            "target_latents": data["target_latents"].detach().cpu()[0],
            "source_latents": data["source_latents"].detach().cpu()[0],
            "prompt_embeds": data["prompt_embeds"].detach().cpu()[0],
            "empty_prompt_embeds": data["empty_prompt_embeds"].detach().cpu()[0],
        }
        map_keys = {
            "target_latents": "image_hash",
            "source_latents": "control_hash",
            "prompt_embeds": "combined_hash",
            "empty_prompt_embeds": "combined_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, file_hashes)

    # ------------------------------------------------------------------ #
    # Device management                                                    #
    # ------------------------------------------------------------------ #

    def setup_model_device_train_mode(self, stage: str = "fit", cache: bool = False):
        if stage == "fit":
            assert hasattr(self, "accelerator")
            if self.cache_exist and self.use_cache:
                for attr in ("text_encoder", "vae", "vl_processor"):
                    if hasattr(self, attr):
                        if hasattr(getattr(self, attr), "cpu"):
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
            self.vae.to(d.vae)
            self.text_encoder.to(d.text_encoder)
            self.dit.to(d.dit)
            for attr in ("vae", "text_encoder", "dit"):
                if hasattr(self, attr):
                    getattr(self, attr).requires_grad_(False).eval()

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def prepare_predict_batch_data(
        self,
        prompt: str | list[str],
        source_image: torch.Tensor | None = None,
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
        if source_image is not None:
            data["control"] = source_image if isinstance(source_image, torch.Tensor) else source_image
            data["image"] = source_image  # also set image for prepare_embeddings
        if negative_prompt:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            data["negative_prompt"] = negative_prompt
        return data

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """Denoising loop for LongCat-Image-Edit. Returns final packed latents (B, N, C*4)."""
        num_steps = embeddings["num_inference_steps"]
        guidance_scale = embeddings.get("guidance_scale", 4.5)
        prompt_embeds = embeddings["prompt_embeds"]
        source_latents = embeddings["source_latents"]
        height = embeddings["height"]
        width = embeddings["width"]
        batch_size = prompt_embeds.shape[0]
        device = self.dit.device
        dtype = self.weight_dtype

        prompt_embeds = prompt_embeds.to(device, dtype=dtype)
        source_latents = source_latents.to(device, dtype=dtype)

        do_cfg = guidance_scale > 1.0
        if do_cfg:
            neg = embeddings.get("negative_prompt", [""] * batch_size)
            source_tensor_01 = embeddings.get("_source_tensor_01")
            neg_embeds, neg_text_ids = self.encode_prompt(neg, source_image=source_tensor_01)
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

        source_packed = self._pack_latents(source_latents)
        image_seq_len = latents.shape[1]
        h_patch, w_patch = h_lat // 2, w_lat // 2
        seq_len = prompt_embeds.shape[1]

        text_ids = self._prepare_pos_ids_text(seq_len, device=device, dtype=dtype)
        target_img_ids = self._prepare_pos_ids_image(
            h_patch, w_patch, modality_id=1, offset=seq_len, device=device, dtype=dtype
        )
        source_img_ids = self._prepare_pos_ids_image(
            h_patch, w_patch, modality_id=2, offset=seq_len, device=device, dtype=dtype
        )
        combined_img_ids = torch.cat([target_img_ids, source_img_ids], dim=0)

        if do_cfg:
            neg_text_ids = self._prepare_pos_ids_text(neg_embeds.shape[1], device=device, dtype=dtype)
            neg_combined_ids = torch.cat([
                self._prepare_pos_ids_image(h_patch, w_patch, 1, neg_embeds.shape[1], device, dtype),
                self._prepare_pos_ids_image(h_patch, w_patch, 2, neg_embeds.shape[1], device, dtype),
            ], dim=0)

        image_seq_len_for_shift = image_seq_len
        mu = _calculate_shift(
            image_seq_len_for_shift,
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
            for t in tqdm(timesteps, desc="LongCatImageEdit generating"):
                latent_model_input = torch.cat([latents.to(dtype), source_packed], dim=1)
                timestep = t.expand(batch_size).to(dtype)

                noise_pred = self.dit(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=combined_img_ids,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :image_seq_len]

                if do_cfg:
                    latent_model_input_neg = torch.cat([latents.to(dtype), source_packed], dim=1)
                    neg_pred = self.dit(
                        hidden_states=latent_model_input_neg,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=neg_embeds,
                        txt_ids=neg_text_ids,
                        img_ids=neg_combined_ids,
                        return_dict=False,
                    )[0]
                    neg_pred = neg_pred[:, :image_seq_len]
                    noise_pred = neg_pred + guidance_scale * (noise_pred - neg_pred)

                latents = self.sampling_scheduler.step(
                    noise_pred.to(torch.float32), t, latents, return_dict=False
                )[0]

        return latents
