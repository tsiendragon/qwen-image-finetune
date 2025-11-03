"""
DreamOmni2 LoRA Trainer Implementation
Extends FluxKontextLoraTrainer with cumulative offsets for multi-image support.
"""

import logging
import re
from typing import Union

import PIL.Image
import torch
from diffusers.utils import load_image

from qflux.models.pipeline_dreamomni2 import DreamOmni2Pipeline
from qflux.trainer.flux_kontext_trainer import FluxKontextLoraTrainer


class DreamOmni2Trainer(FluxKontextLoraTrainer):
    """
    DreamOmni2 LoRA Trainer implementation.

    Key difference from FluxKontextLoraTrainer:
    - Uses cumulative offsets (w_offset, h_offset) for multi-image position encoding
    - This ensures RoPE correctly understands spatial relationships between multiple images
    - Supports VLM-based prompt optimization for better instruction understanding

    Inherits from FluxKontextLoraTrainer since most functionality is identical.
    Only requires modifications to:
    1. Pipeline class reference
    2. prepare_embeddings() to add cumulative offset calculation
    3. VLM prompt optimization support (optional)
    """

    def __init__(self, config):
        """Initialize DreamOmni2 trainer with configuration."""
        super().__init__(config)
        # VLM components for prompt optimization (optional)
        self.vlm_model = None
        self.vlm_processor = None
        self.use_vlm_prompt_enhancer = config.model.use_vlm_prompt_enhancer

    def get_pipeline_class(self):
        """Return DreamOmni2Pipeline class for LoRA weight saving."""
        return DreamOmni2Pipeline

    def load_model(self):
        """
        Load model components including VLM for prompt optimization.

        Extends parent's load_model to also load VLM if enabled in config.
        """
        # Load base models (transformer, VAE, text encoders, etc.)
        super().load_model()

        # Load VLM model if prompt enhancement is enabled
        if self.use_vlm_prompt_enhancer:
            logging.info("VLM prompt enhancement enabled, loading VLM model...")
            self.load_vlm_model()  # Always load to CPU initially
        else:
            logging.info("VLM prompt enhancement disabled, skipping VLM model loading.")

    def setup_model_device_train_mode(self, stage="fit", cache=False):
        """
        Set model device allocation and train mode, including VLM model.

        Extends parent's device setup to also manage VLM model placement.
        """
        # Call parent's setup first
        super().setup_model_device_train_mode(stage, cache)

        # Manage VLM model device based on stage and configuration
        if not self.use_vlm_prompt_enhancer or self.vlm_model is None:
            return

        # Determine target device based on stage
        if stage == "cache":
            device = self.config.cache.devices.prompt_enhancer or "cuda:0"
            self.vlm_model.to(device)
            logging.info(f"VLM model moved to {device} (cache mode)")
        elif stage == "predict":
            device = self.config.predict.devices.prompt_enhancer or "cuda:0"
            self.vlm_model.to(device)
            logging.info(f"VLM model moved to {device} (predict mode)")
        elif stage == "fit" and not cache:
            # Fit without cache: move VLM to GPU
            device = self.config.predict.devices.prompt_enhancer or "cuda:0"
            self.vlm_model.to(device)
            logging.info(f"VLM model moved to {device} (fit without cache mode)")
        else:
            # Fit with cache: VLM not needed, keep on CPU
            self.vlm_model.to("cpu")
            logging.info("VLM model kept on CPU (fit with cache mode)")

    def load_vlm_model(self, vlm_path: str | None = None, device: str | torch.device = "cpu"):
        """
        Load VLM model and processor for prompt optimization.

        Args:
            vlm_path: Path to VLM model. If None, uses default "xiabs/DreamOmni2".
            device: Device to load model on. Defaults to "cpu".
        """
        if vlm_path is None:
            vlm_path = "xiabs/DreamOmni2"

        if self.vlm_model is not None:
            logging.info("VLM model already loaded, skipping reload.")
            return

        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logging.info(f"Loading VLM model from {vlm_path} to device {device}...")
        self.vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_path,
            subfolder="vlm-model",
            torch_dtype=self.weight_dtype,
            device_map=str(device),
        )
        self.vlm_processor = AutoProcessor.from_pretrained(vlm_path, subfolder="vlm-model", trust_remote_code=True)
        self.vlm_model.requires_grad_(False).eval()
        logging.info(f"VLM model loaded successfully on {device}.")

    def _should_use_vlm_for_stage(self, stage: str) -> bool:
        """
        Determine if VLM prompt optimization should be used for the given stage.

        Args:
            stage: Training stage ("fit", "cache", "predict")

        Returns:
            True if VLM should be used, False otherwise
        """
        if not self.use_vlm_prompt_enhancer or self.vlm_model is None:
            return False

        # Check if using cache in fit mode
        is_fit_with_cache = (stage == "fit") and self.cache_exist and self.use_cache

        # Use VLM in cache mode and fit without cache mode, but not in predict or fit with cache
        if stage == "cache":
            return True
        elif stage == "predict":
            return True
        elif stage == "fit":
            if is_fit_with_cache:
                return False
            else:
                return True
        else:
            return False

    def _tensor2pil(self, tensor: torch.Tensor) -> PIL.Image.Image:
        """
        Convert a normalized tensor image to PIL Image.

        Args:
            tensor: Tensor of shape [C, H, W] in range [0, 1]

        Returns:
            PIL Image
        """
        # Denormalize from [0, 1] to [0, 255]
        tensor = tensor.clamp(0, 1)
        tensor = (tensor * 255).to(torch.uint8)
        # Convert to numpy and PIL
        numpy_image = tensor.cpu().numpy()
        # Transpose from CHW to HWC
        numpy_image = numpy_image.transpose(1, 2, 0)
        # Convert to PIL
        if numpy_image.shape[2] == 3:
            return PIL.Image.fromarray(numpy_image, mode="RGB")
        elif numpy_image.shape[2] == 1:
            return PIL.Image.fromarray(numpy_image[:, :, 0], mode="L")
        else:
            raise ValueError(f"Unexpected number of channels: {numpy_image.shape[2]}")

    def _extract_gen_content(self, text: str) -> str:
        """
        Extract generated content from VLM response.

        The VLM response may contain special tokens or formatting.
        This method extracts the actual prompt content.
        """
        # Remove common special tokens and formatting
        # Pattern: <|im_start|>...<|im_end|> or similar
        text = re.sub(r"<\|im_start\|>", "", text)
        text = re.sub(r"<\|im_end\|>", "", text)
        text = re.sub(r"<\|.*?\|>", "", text)
        text = text.strip()
        return text

    def _prepare_images_for_vlm(
        self, images: Union[PIL.Image.Image, list[PIL.Image.Image], list[str]]
    ) -> list[PIL.Image.Image]:
        """
        Prepare images for VLM input.

        Args:
            images: PIL Images, list of PIL Images, or list of image paths

        Returns:
            List of PIL Images ready for VLM processing
        """
        if isinstance(images, str):
            images = [images]
        if isinstance(images, PIL.Image.Image):
            images = [images]

        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = load_image(img)
            # Resize input if needed (similar to resizeinput utility)
            # For now, we'll pass images as-is
            processed_images.append(img)

        return processed_images

    def optimize_prompt_with_vlm(
        self,
        images: Union[PIL.Image.Image, list[PIL.Image.Image], list[str]],
        instruction: str,
        prefix: str = " It is editing task.",
    ) -> str:
        """
        Optimize prompt using VLM model.

        Args:
            images: Input images (PIL Images or paths)
            instruction: User instruction/prompt
            prefix: Optional prefix to add to instruction

        Returns:
            Optimized prompt string
        """
        # Load VLM model if not already loaded
        if self.vlm_model is None or self.vlm_processor is None:
            self.load_vlm_model()

        if self.vlm_model is None or self.vlm_processor is None:
            logging.warning("VLM model not loaded. Returning original prompt.")
            return instruction
        print("vlm_model device ", self.vlm_model.device)
        processed_images = self._prepare_images_for_vlm(images)
        # Prepare messages for VLM
        content = []
        for img in processed_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": instruction + prefix})

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process inputs with images
        # The processor accepts PIL Images directly
        inputs = self.vlm_processor(
            text=[text],
            images=processed_images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        device = next(self.vlm_model.parameters()).device
        inputs = inputs.to(device)

        # Generate optimized prompt
        with torch.inference_mode():
            generated_ids = self.vlm_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=4096,
            )

        # Decode generated text
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]
        output_text = self.vlm_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        optimized_prompt = self._extract_gen_content(output_text[0])
        logging.info(f"Original prompt: {instruction}")
        logging.info(f"Optimized prompt: {optimized_prompt}")
        return optimized_prompt

    def prepare_embeddings(self, batch, stage="fit"):
        """
        Prepare embeddings with VLM prompt optimization and cumulative offset support.

        DreamOmni2-specific modifications:
        1. VLM prompt optimization before encoding (optional)
        2. Cumulative offsets in control_ids for multi-image positioning

        Note: Cannot simply call super() because control_ids need cumulative offsets
        applied BEFORE concatenation, but parent concatenates them directly.
        """
        # =============================================================================
        # PART 1: VLM Prompt Optimization (DreamOmni2-specific)
        # =============================================================================
        should_optimize = self._should_use_vlm_for_stage(stage)
        if should_optimize and "prompt" in batch:
            print("optimize prompt with vlm")
            num_additional_controls = (
                batch["n_controls"] if isinstance(batch["n_controls"], int) else batch["n_controls"][0]
            )
            # Collect and normalize control images for VLM
            control_images = []
            if "control" in batch:
                control_images.append(self._tensor2pil(batch["control"][0]))

            for i in range(num_additional_controls):
                control_key = f"control_{i + 1}"
                if control_key in batch:
                    control_images.append(self._tensor2pil(batch["control"][0]))

            # Optimize prompt with VLM and update batch
            if control_images:
                original_prompt = batch["prompt"][0] if isinstance(batch["prompt"], list) else batch["prompt"]
                optimized_prompt = self.optimize_prompt_with_vlm(
                    images=control_images,
                    instruction=original_prompt,
                    prefix="It is editing task.",
                )
                if isinstance(batch["prompt"], list):
                    batch["prompt"] = [optimized_prompt] * len(batch["prompt"])
                else:
                    batch["prompt"] = optimized_prompt

        # =============================================================================
        # PART 2: Standard Processing (same as parent, call super)
        # =============================================================================
        batch = super().prepare_embeddings(batch, stage)
        return batch
