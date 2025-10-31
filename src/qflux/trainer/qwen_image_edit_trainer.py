import copy
import gc
import logging
from typing import Any

import numpy as np
import PIL
import torch
import torch.nn.functional as F  # NOQA
from diffusers import AutoencoderKLQwenImage
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    QwenImageEditPipeline,
    randn_tensor,
    retrieve_latents,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from tqdm.auto import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from qflux.models.load_model import load_qwenvl, load_vae
from qflux.models.transformer_qwenimage import QwenImageTransformer2DModel

# calculate_dimensions,
# from qflux.loss.edit_mask_loss import map_mask_to_latent
from qflux.trainer.base_trainer import BaseTrainer
from qflux.utils.images import calculate_best_resolution, make_image_devisible, make_image_shape_devisible
from qflux.utils.tools import extract_batch_field, infer_image_tensor, pad_latents_for_multi_res


# image_adjust_best_resolution


class QwenImageEditTrainer(BaseTrainer):
    """Trainer class based on QwenImageEditPipeline"""

    def __init__(self, config):
        """
        the image process passed to vae
        import numpy as np
        def customized_process(img: np.ndarray) -> torch.Tensor:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = img.unsqueeze(2)
            img = img / 255
            img = img*2 -1
            return img
        image to the prompt encoder should be [B,C,1,H,W] in range [-1,1]
        image to the text encoder
            [B,C,H,W] tensor in range [0,255], uint8, or PIL.Image RGB mode
        """
        super().__init__(config)
        self.config = config

        # Component attributes
        self.vae: AutoencoderKLQwenImage
        self.text_encoder: Qwen2_5_VLForConditionalGeneration
        self.dit: QwenImageTransformer2DModel
        self.tokenizer: Qwen2Tokenizer
        self.scheduler: FlowMatchEulerDiscreteScheduler

        # Parameters obtained from VAE configuration
        self.vae_scale_factor: int
        self.vae_latent_mean: float
        self.vae_latent_std: float
        self.vae_z_dim: int
        self.adapter_name = config.model.lora.adapter_name

    def get_pipeline_class(self):
        return QwenImageEditPipeline

    # Static methods: directly reference QwenImageEditPipeline methods
    _pack_latents = staticmethod(QwenImageEditPipeline._pack_latents)
    _unpack_latents = staticmethod(QwenImageEditPipeline._unpack_latents)

    def load_model(self, text_encoder_device=None):
        """Load and separate components from QwenImageEditPipeline"""
        logging.info("Loading QwenImageEditPipeline and separating components...")

        # Load complete model using pipeline
        pipe = QwenImageEditPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype,
            transformer=None,
            vae=None,
        )
        pipe.to("cpu")
        logging.info(f"excution device: {pipe._execution_device}")

        # Separate individual components

        self.vae = load_vae("Qwen/Qwen-Image-Edit", weight_dtype=self.weight_dtype)  # use original one
        # same to model constructed from vae self.vae = pipe.vae
        self.text_encoder = load_qwenvl("Qwen/Qwen-Image-Edit", weight_dtype=self.weight_dtype)  # use original one
        logging.info(f"text_encoder device: {self.text_encoder.device}")
        # self.dit = pipe.transformer this is same as the following, verified
        from qflux.models.load_model import load_transformer

        self.dit = load_transformer(
            self.config.model.pretrained_model_name_or_path,  # could use quantized version
            weight_dtype=self.weight_dtype,
        )
        # load_transformer is same as pipe.transformer

        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

        self.processor: Qwen2VLProcessor = pipe.processor
        self.tokenizer: Qwen2Tokenizer = pipe.tokenizer
        # Create two schedulers: one for training, one for sampling/validation
        self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
        import copy

        self.sampling_scheduler: FlowMatchEulerDiscreteScheduler = copy.deepcopy(
            self.scheduler
        )  # Independent scheduler for validation/sampling
        # Initialize image processor (for predict method)
        from diffusers.image_processor import VaeImageProcessor

        # Set VAE-related parameters
        self.vae_scale_factor = 2 ** len(self.vae.temperal_downsample)
        self.vae_latent_mean = self.vae.config.latents_mean
        self.vae_latent_std = self.vae.config.latents_std
        self.vae_z_dim = self.vae.config.z_dim

        # Attributes copied from original pipeline
        self.latent_channels = self.vae.config.z_dim
        self._guidance_scale = 1.0
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        self.prompt_template_encode = pipe.prompt_template_encode
        self.prompt_template_encode_start_idx = pipe.prompt_template_encode_start_idx

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.num_channels_latents = self.dit.config.in_channels // 4

        # Set models to training/evaluation mode
        self.text_encoder.requires_grad_(False).eval()
        self.vae.requires_grad_(False).eval()
        self.dit.requires_grad_(False).eval()
        torch.cuda.empty_cache()

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def preprocess_image_for_vae_encoder(self, image):
        """
        the image process passed to vae
        import numpy as np
        def customized_process(img: np.ndarray) -> torch.Tensor:
            img = torch.from_numpy(img)
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(0)
            img = img.unsqueeze(2)
            img = img / 255
            img = img*2 -1
            return img
        image to the vae encoder should be [B,C,1,H,W] in range [-1,1]
        """
        # suppose image input is [0,1] [B,C,H,W] after dataloader -> [B,C,1,H,W]
        tensor_info = infer_image_tensor(image)
        if tensor_info["layout"] == "HW":
            image = image.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif tensor_info["layout"] == "CHW":
            image = image.unsqueeze(0).unsqueeze(2)
        elif tensor_info["layout"] == "HWC":
            image = image.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        elif tensor_info["layout"] == "BCHW":
            image = image.unsqueeze(2)
        elif tensor_info["layout"] == "BHWC":
            image = image.permute(0, 3, 1, 2).unsqueeze(2)
        else:
            raise ValueError(f"Invalid image layout: {tensor_info['layout']},{image.shape}")

        if tensor_info["range"] == "0-255":
            image = image / 255.0
            image = 2 * image - 1
        elif tensor_info["range"] == "0-1":
            image = 2 * image - 1
        elif tensor_info["range"] == "-1-1":
            pass
        else:
            raise ValueError(f"Invalid image range: {tensor_info['range']}, {image.min()}, {image.max()}")

        return image

    def preprocess_image_for_text_encoder(self, image, best_resolution=None) -> torch.Tensor:
        """
        the image process passed to text encoder
        image to the text encoder
            [B,C,H,W] tensor in range [0,255], uint8, or PIL.Image RGB mode
        """
        tensor_info = infer_image_tensor(image)
        if tensor_info["layout"] == "HW":
            image = image.unsqueeze(0).unsqueeze(0)
        elif tensor_info["layout"] == "CHW":
            image = image.unsqueeze(0)
        elif tensor_info["layout"] == "HWC":
            image = image.permute(2, 0, 1).unsqueeze(0)
        elif tensor_info["layout"] == "BCHW":
            pass
        elif tensor_info["layout"] == "BHWC":
            image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Invalid image layout: {tensor_info['layout']}")
        if tensor_info["range"] == "0-255":
            pass
        elif tensor_info["range"] == "0-1":
            image = image * 255
        elif tensor_info["range"] == "-1-1":
            image = (image + 1) / 2 * 255
        else:
            raise ValueError(f"Invalid image range: {tensor_info['range']}")
        if best_resolution is not None:
            new_width, new_height = calculate_best_resolution(image.shape[3], image.shape[2], best_resolution)
            image = F.interpolate(image, size=(new_height, new_width), mode="bilinear")
        return image

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        generator=None,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        device = self.vae.device

        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(
                    f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                )
            else:
                image_latents = torch.cat([image_latents], dim=0)

            image_latent_height, image_latent_width = image_latents.shape[3:]
            image_latents = self._pack_latents(
                image_latents,
                batch_size,
                num_channels_latents,
                image_latent_height,
                image_latent_width,
            )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    def setup_model_device_train_mode(self, stage="fit", cache=False):
        """Set model device allocation and train mode."""
        assert stage in [
            "fit",
            "cache",
            "predict",
        ], f"stage must be one of ['fit', 'cache', 'predict'], but got {stage}"
        if stage == "fit":
            assert hasattr(self, "accelerator"), "accelerator must be set before setting model devices"

        if self.cache_exist and self.use_cache and stage == "fit":
            # Cache mode: only need transformer
            self.text_encoder.cpu()
            torch.cuda.empty_cache()
            self.vae.cpu()
            torch.cuda.empty_cache()
            # del self.text_encoder

            if not self.config.validation.enabled:
                del self.vae
                del self.text_encoder
            else:
                self.vae.requires_grad_(False).eval()

            gc.collect()
            # self.dit.to(self.accelerator.device)
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
            # self.dit.to(self.accelerator.device)
            self.vae.decoder.to("cpu")

            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
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

            torch.cuda.synchronize()
            self.dit.cpu()
            torch.cuda.empty_cache()
            del self.dit
            gc.collect()
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            logging.info("cache mode device setting")
            print("vae device ", next(self.vae.parameters()).device)

        elif stage == "predict":
            # Predict mode: allocate to different GPUs according to configuration
            devices = self.config.predict.devices
            self.vae.to(devices.vae)
            self.text_encoder.to(devices.text_encoder)
            self.dit.to(devices.dit)
            self.vae.requires_grad_(False).eval()
            self.text_encoder.requires_grad_(False).eval()
            self.dit.requires_grad_(False).eval()

    def prepare_embeddings(self, batch, stage="fit"):
        """
        used in: _training_step_compute, cache, and predict
        for cache: use prepare_cached_embeddings
        Qwen-Edit
            - image_latent : target used for training
            - control_latent: concatenated version
            - height_image, width_image
            - height_control, width_control: no batch dim
            - height_control_1, width_control_1: no batch dim
            ...
        for control image, need to create two copies:
            1. processed for vae encoder
            2. processed for text encoder
        for additional control images, only pass to vae encoder
        predict mode:

            - no `image` key
            - extra: negative_prompt_embeds, negative_prompt_embeds_mask

        cache mode:
            - extra: empty_prompt_embeds, empty_prompt_embeds_mask
        """

        if "image" in batch:
            batch["image"] = self.preprocess_image_for_vae_encoder(batch["image"])  # B,3,1,H,W
            batch["width"] = batch["image"].shape[4]
            batch["height"] = batch["image"].shape[3]

        if "control" in batch:
            prompt_control = self.preprocess_image_for_text_encoder(batch["control"])  # [B,C,H,W], uint8, range [0,255]
            batch["prompt_control"] = prompt_control
            batch["control"] = self.preprocess_image_for_vae_encoder(
                batch["control"]
            )  # [B,C,1,H,W], float, range [-1,1]
            batch["width_control"] = batch["control"].shape[4]
            batch["height_control"] = batch["control"].shape[3]

        if isinstance(batch["n_controls"], int):
            num_additional_controls = batch["n_controls"]
        else:
            num_additional_controls = batch["n_controls"][0]

        for i in range(num_additional_controls):
            additional_control_key = f"control_{i + 1}"
            if additional_control_key in batch:
                batch[additional_control_key] = self.preprocess_image_for_vae_encoder(batch[additional_control_key])
                batch[f"width_control_{i + 1}"] = batch[additional_control_key].shape[4]
                batch[f"height_control_{i + 1}"] = batch[additional_control_key].shape[3]

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=batch["prompt"],
            image=batch["prompt_control"],
        )
        batch["prompt_embeds_mask"] = prompt_embeds_mask
        batch["prompt_embeds"] = prompt_embeds

        if stage == "cache":
            empty_prompt_embeds, empty_prompt_embeds_mask = self.encode_prompt(
                prompt=[""],
                image=batch["prompt_control"],
            )
            batch["empty_prompt_embeds_mask"] = empty_prompt_embeds_mask
            batch["empty_prompt_embeds"] = empty_prompt_embeds

        if "negative_prompt" in batch and batch["true_cfg_scale"] > 1:
            # only for predict stage
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt=batch["negative_prompt"],
                image=batch["prompt_control"],
            )
            batch["negative_prompt_embeds_mask"] = negative_prompt_embeds_mask
            batch["negative_prompt_embeds"] = negative_prompt_embeds

        # get latents
        if "image" in batch:
            image = batch["image"]
            batch_size = image.shape[0]
            height_image, width_image = batch["height"], batch["width"]
            _, image_latents = self.prepare_latents(
                image,
                batch_size,
                self.num_channels_latents,
                height_image,
                width_image,
                self.weight_dtype,
            )
            batch["image_latents"] = image_latents

        if "control" in batch:
            control = batch["control"]
            batch_size = control.shape[0]
            height_control, width_control = batch["height_control"], batch["width_control"]
            _, control_latents = self.prepare_latents(
                control,
                batch_size,
                self.num_channels_latents,
                height_control,
                width_control,
                self.weight_dtype,
            )
            batch["control_latents"] = [control_latents]

        for i in range(1, num_additional_controls + 1):
            control_key = f"control_{i}"
            control = batch[control_key]
            batch_size = control.shape[0]
            height_control, width_control = batch[f"height_control_{i}"], batch[f"width_control_{i}"]
            _, control_latents = self.prepare_latents(
                control,
                batch_size,
                self.num_channels_latents,
                height_control,
                width_control,
                self.weight_dtype,
            )
            batch["control_latents"].append(control_latents)

        if "control_latents" in batch:
            batch["control_latents"] = torch.cat(batch["control_latents"], dim=1)

        # if self.config.loss.mask_loss and "mask" in batch:
        #     mask = batch["mask"]
        #     height_image, width_image = batch["height"], batch["width"]
        #     batch["mask"] = resize_bhw(mask, height_image, width_image)
        #     batch["mask"] = map_mask_to_latent(batch["mask"])
        # need to convert img_shapes from pixel space to latent shapce

        img_shapes = batch["img_shapes"]
        img_shapes = self.convert_img_shapes_to_latent_space(img_shapes)
        batch["img_shapes"] = img_shapes
        return batch

    def cache_step(
        self,
        data: dict,
    ):
        """
        cache image embedding and vae embedding.
        which is calculated in prepare_embeddings()
        img_shapes (list[int]) is also cached without batch dimension, need to add batch dimension when use it
        """

        image_latents = data["image_latents"].detach().cpu()[0]
        control_latents = data["control_latents"].detach().cpu()[0]
        prompt_embeds = data["prompt_embeds"].detach().cpu()[0]
        prompt_embeds_mask = data["prompt_embeds_mask"].detach().cpu()[0]
        empty_prompt_embeds = data["empty_prompt_embeds"].detach().cpu()[0]
        empty_prompt_embeds_mask = data["empty_prompt_embeds_mask"].detach().cpu()[0]
        # img_shapes now is calculated in dataset, do not need any more
        cache_embeddings = {
            "image_latents": image_latents,
            "control_latents": control_latents,
            "prompt_embeds_mask": prompt_embeds_mask,
            "prompt_embeds": prompt_embeds,
            "empty_prompt_embeds_mask": empty_prompt_embeds_mask,
            "empty_prompt_embeds": empty_prompt_embeds,
            # "img_shapes": img_shapes,
        }
        map_keys = {
            "image_latents": "image_hash",
            "control_latents": "controls_sum_hash",
            "prompt_embeds_mask": "prompt_hash",
            "prompt_embeds": "prompt_hash",
            "empty_prompt_embeds_mask": "prompt_hash",
            "empty_prompt_embeds": "prompt_hash",
            # "img_shapes": "main_hash",
        }
        self.cache_manager.save_cache_embedding(cache_embeddings, map_keys, data["file_hashes"])

    def prepare_cached_embeddings(self, batch):
        """
        Prepare cached embeddings from dataloader batch.
        Expects v2.0 cache format with img_shapes field.
        """
        img_shapes = batch["img_shapes"]
        img_shapes = self.convert_img_shapes_to_latent_space(img_shapes)
        batch["img_shapes"] = img_shapes
        return batch

    def _postprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Post-process output images"""
        image = image_tensor.cpu().float()
        image = image.squeeze(2).squeeze(0)  # [C,H,W]
        image = (image / 2 + 0.5).clamp(0, 1)  # Convert from [-1,1] to [0,1]
        image_ = image.permute(1, 2, 0).numpy()  # [H,W,C]
        image_ = (image_ * 255).astype(np.uint8)
        return image_

    def convert_img_shapes_to_latent_space(
        self, img_shapes: list[list[tuple[int, int, int]]]
    ) -> list[list[tuple[int, int, int]]]:
        """Convert image shapes from pixel space to latent space
        img_shapes = [
                [
                    (1, height_image // self.vae_scale_factor // 2, width_image // self.vae_scale_factor // 2),
                    (1, height_control // self.vae_scale_factor // 2, width_control // self.vae_scale_factor // 2),
                    (1, height_control_1 // self.vae_scale_factor // 2, width_control_1 // self.vae_scale_factor // 2),
                ]
        ]
        """
        new_shapes = []
        for shape_per_batch in img_shapes:
            new_shape_per_batch = []
            for shape in shape_per_batch:
                new_shape_per_batch.append(
                    (1, shape[1] // self.vae_scale_factor // 2, shape[2] // self.vae_scale_factor // 2)
                )
            new_shapes.append(new_shape_per_batch)
        return new_shapes

    def _get_image_shapes_multi_resolution(self, embeddings: dict, batch_size: int) -> list[list[tuple[int, int, int]]]:
        """Get image shapes for multi-resolution mode (per-sample shapes)

        This method extends _get_image_shapes to support per-sample different
        resolutions, which is required for multi-resolution training.

        Args:
            embeddings: Batch embeddings containing shape information
            batch_size: Number of samples in the batch

        Returns:
            List[List[Tuple]] where:
            - Outer list: batch dimension (length = batch_size)
            - Inner list: images in this sample [target, control, control_1, ...]
            - Tuple: (1, H_latent, W_latent) for each image

        Example:
            >>> embeddings = {
            ...     "img_shapes": [
            ...         [(1, 32, 64), (1, 32, 64)],  # Sample 0: target + control
            ...         [(1, 48, 48), (1, 48, 48)],  # Sample 1: different size
            ...     ]
            ... }
            >>> shapes = trainer._get_image_shapes_multi_resolution(embeddings, 2)
            >>> shapes  # [[(1, 32, 64), (1, 32, 64)], [(1, 48, 48), (1, 48, 48)]]
        """
        # Priority 1: Use img_shapes from collate/cache
        if "img_shapes" in embeddings:
            # Format: List[List[(1, H', W')]]
            img_shapes = embeddings["img_shapes"]

            # Handle different formats (list, tensor, etc.)
            if isinstance(img_shapes, torch.Tensor):
                # Convert tensor to nested list
                # Assume shape: [batch_size, num_images, 3]
                img_shapes = img_shapes.tolist()
                # Restructure to nested list format
                result = []
                for b in range(batch_size):
                    sample_shapes = [tuple(img_shapes[b][i]) for i in range(len(img_shapes[b]))]
                    result.append(sample_shapes)
                return result

            # Already in correct format: List[List[Tuple]]
            if len(img_shapes) == batch_size:
                return img_shapes

            # Fallback: single shape, replicate for all samples
            if len(img_shapes) == 1:
                return img_shapes * batch_size

        # Priority 2: Reconstruct from height/width fields (runtime, non-cache mode)
        result = []
        for b in range(batch_size):
            img_shapes_sample = []

            # Target image
            height = extract_batch_field(embeddings, "height", b)
            width = extract_batch_field(embeddings, "width", b)
            img_shapes_sample.append(
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                )
            )

            # Control images (if exist)
            if "height_control" in embeddings and "width_control" in embeddings:
                height_ctrl = extract_batch_field(embeddings, "height_control", b)
                width_ctrl = extract_batch_field(embeddings, "width_control", b)
                img_shapes_sample.append(
                    (
                        1,
                        height_ctrl // self.vae_scale_factor // 2,
                        width_ctrl // self.vae_scale_factor // 2,
                    )
                )

                # Additional control images
                num_controls = extract_batch_field(embeddings, "n_controls", b) if "n_controls" in embeddings else 0
                for i in range(num_controls):
                    h_key = f"height_control_{i + 1}"
                    w_key = f"width_control_{i + 1}"
                    if h_key in embeddings and w_key in embeddings:
                        h_i = extract_batch_field(embeddings, h_key, b)
                        w_i = extract_batch_field(embeddings, w_key, b)
                        img_shapes_sample.append(
                            (
                                1,
                                h_i // self.vae_scale_factor // 2,
                                w_i // self.vae_scale_factor // 2,
                            )
                        )

            result.append(img_shapes_sample)

        return result

    def _forward_qwen_multi_resolution(
        self,
        packed_input: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        timesteps: torch.Tensor,
        img_shapes: list,
        txt_seq_lens: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Qwen model in multi-resolution mode.

        This method handles variable-resolution images by:
        1. Padding latents to uniform length
        2. Generating per-sample RoPE frequencies
        3. Creating attention mask to ignore padding
        4. Calling model with pre-computed RoPE

        Args:
            packed_input: Noisy input + control latents [B, seq_i, C] (may be variable length)
            prompt_embeds: Text embeddings [B, txt_len, D]
            prompt_embeds_mask: Text attention mask [B, txt_len]
            timesteps: Timesteps for diffusion [B]
            img_shapes: Per-sample image shapes List[List[Tuple]]
            txt_seq_lens: Text sequence lengths [B]

        Returns:
            model_pred: Model predictions [B, max_seq, C]
            img_attention_mask: Image attention mask [B, max_seq] (True=valid, False=padded)
        """
        device = packed_input.device
        # dtype = packed_input.dtype
        batch_size = packed_input.shape[0]

        # Calculate sequence lengths for each sample
        seq_lens = [shapes[0][1] * shapes[0][2] * 2 for shapes in img_shapes]  # *2 for target+control
        max_seq = max(seq_lens)

        # Pad packed_input if needed
        if packed_input.shape[1] < max_seq:
            # Convert to list for padding
            packed_list = [packed_input[i, : seq_lens[i]] for i in range(batch_size)]
            padded_packed, img_attention_mask = pad_latents_for_multi_res(packed_list, max_seq)
        else:
            padded_packed = packed_input
            # Create attention mask
            img_attention_mask = torch.zeros(batch_size, max_seq, device=device, dtype=torch.bool)
            for i in range(batch_size):
                img_attention_mask[i, : seq_lens[i]] = True

        # Generate per-sample RoPE frequencies
        img_freqs_list = []
        txt_freqs = None

        for b in range(batch_size):
            # Generate RoPE for this sample
            img_freqs_b, txt_freqs_b = self.dit.pos_embed([img_shapes[b]], [txt_seq_lens[b]], device=device)

            # Pad img_freqs to max_seq
            rope_dim = img_freqs_b.shape[-1]
            img_freqs_padded = torch.zeros(max_seq, rope_dim, device=device, dtype=img_freqs_b.dtype)
            img_freqs_padded[: img_freqs_b.shape[0]] = img_freqs_b
            img_freqs_list.append(img_freqs_padded)

            # txt_freqs is the same for all samples (if txt_seq_lens are the same)
            if txt_freqs is None:
                txt_freqs = txt_freqs_b

        # Stack per-sample RoPE into 3D tensor (B, max_seq, rope_dim)
        img_freqs_batched = torch.stack(img_freqs_list, dim=0)
        image_rotary_emb = (img_freqs_batched, txt_freqs)

        # Create 4D attention mask for scaled_dot_product_attention
        # Shape: (B, 1, 1, txt_len + max_seq)
        txt_len = prompt_embeds.shape[1]
        joint_seq = txt_len + max_seq
        attention_mask = torch.zeros(batch_size, joint_seq, device=device, dtype=torch.bool)

        for i in range(batch_size):
            attention_mask[i, : txt_seq_lens[i]] = True  # text tokens
            attention_mask[i, txt_len : txt_len + seq_lens[i]] = True  # valid image tokens

        # Convert to 4D: (B, 1, 1, joint_seq)
        attention_mask_4d = attention_mask.view(batch_size, 1, 1, joint_seq)

        # Forward through model with pre-computed RoPE
        model_pred = self.dit(
            hidden_states=padded_packed,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            image_rotary_emb=image_rotary_emb,  # Pre-computed per-sample RoPE
            attention_kwargs={"attention_mask": attention_mask_4d},
            return_dict=False,
        )[0]

        return model_pred, img_attention_mask

    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        assert self.accelerator is not None, "accelerator is not set"
        device = self.accelerator.device
        # dtype = self.weight_dtype
        image_latents = embeddings["image_latents"].to(self.weight_dtype).to(device)
        control_latents = embeddings["control_latents"].to(self.weight_dtype).to(device)
        prompt_embeds = embeddings["prompt_embeds"].to(self.weight_dtype).to(device)
        prompt_embeds_mask = embeddings["prompt_embeds_mask"].to(dtype=torch.int64).to(device)
        img_shapes = embeddings["img_shapes"]
        batch_size = image_latents.shape[0]
        if "edit_mask" in embeddings:  # already has edit_mask in collate function
            edit_mask = embeddings["edit_mask"]
        else:
            edit_mask = None

        # ✅ Use BaseTrainer's multi-resolution detection
        is_multi_res = False  # self.should_use_multi_resolution_mode(embeddings)
        # TODO: implement multi-resolution mode

        with torch.no_grad():
            noise = torch.randn_like(image_latents, device=device, dtype=self.weight_dtype)

            # Sample timesteps
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
            packed_input = torch.cat([noisy_model_input, control_latents], dim=1)
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        # ✅ Multi-resolution mode: use padding + per-sample RoPE
        if is_multi_res:
            model_pred, img_attention_mask = self._forward_qwen_multi_resolution(
                packed_input=packed_input,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                timesteps=timesteps,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )
        else:
            # Original concatenation mode
            model_pred = self.dit(
                hidden_states=packed_input,
                timestep=timesteps / 1000,
                guidance=None,
                encoder_hidden_states_mask=prompt_embeds_mask,
                encoder_hidden_states=prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]
            img_attention_mask = None

        model_pred = model_pred[:, : image_latents.size(1)]
        weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
        target = noise - image_latents

        if is_multi_res and img_attention_mask is not None:
            raise NotImplementedError("Multi-resolution mode is not implemented for image edit")
        else:
            # pred shape [B, seq, C], target shape [B, seq, C]
            loss = self.forward_loss(model_pred, target, weighting, edit_mask)

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

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator = None):
        # generator is None by default
        if isinstance(generator, list):
            _image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]),
                    generator=generator[i],
                    sample_mode="argmax",
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(_image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents

    def encode_prompt(
        self,
        prompt: str | list[str],
        image: torch.Tensor | None = None,
    ):
        r"""
        get the embedding of prompt and image via qwen_vl. Support batch inference. For batch inference,
        will pad to largest length of prompt in batch.
        It got grad by default, lets add the inference mode first

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = self.text_encoder.device
        num_images_per_prompt = 1  # 固定为1，支持单图像生成
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(image, torch.Tensor):
            # make sure it is identical with diffuser pipelines
            # when remove this, got relative error 0.49% in prompt embeds
            new_images = []
            for i in range(image.shape[0]):
                img = image[i].permute(1, 2, 0).cpu().numpy().astype("uint8")
                img = PIL.Image.fromarray(img)
                new_images.append(img)
            image = new_images

        with torch.inference_mode():
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
        return prompt_embeds, prompt_embeds_mask

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str] | None = None,
        image: torch.Tensor | None = None,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        assert prompt is not None, "prompt is required"
        assert image is not None, "image is required"
        assert device is not None, "device is required"
        assert dtype is not None, "dtype is required"
        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        model_inputs = self.processor(
            text=txt,
            images=image,
            padding=True,
            return_tensors="pt",
        ).to(device)

        with torch.cuda.device(model_inputs.input_ids.device):  # 作用域内的 'cuda' 都指向同一张卡
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]

        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def prepare_predict_batch_data(
        self,
        image: PIL.Image.Image | list[PIL.Image.Image],
        prompt: str | list[str],
        controls_size: list[int] | None = None,  # first one is for main control
        height: int | None = None,
        width: int | None = None,
        negative_prompt: None | str | list[str] = None,
        guidance_scale: float = None,
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        image_latents: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        prompt_embeds_mask: torch.Tensor = None,
        use_native_size: bool = False,
        best_resolution_size: bool = False,
        weight_dtype=torch.bfloat16,
        **kwargs,
    ) -> dict:
        """Prepare predict batch data.
        prepare the data to batch dict that can be used to prepare embeddings similar in the training step.
        We want to reuse the same data preparation code in the training step.
        Processing logic:
            if height, width is None: choose the width from prompt_image
        if controls_size is None, use same processing as that in the processor in the config
        if use_native_size = True:
            the prompt image and additional controls will use their own size. That is they will not be resized.
        """
        if not isinstance(image, list):
            image = [image]
        # image = [make_image_devisible(image, self.vae_scale_factor) for image in image]
        prompt_image = [image[0]]
        additional_controls = [image[1:]] if len(image) > 1 else None

        assert prompt_image is not None, "prompt_image is required"
        assert prompt is not None, "prompt is required"
        if isinstance(prompt_image, PIL.Image.Image):
            prompt_image = [prompt_image]
        prompt_image = [make_image_devisible(image, self.vae_scale_factor) for image in prompt_image]

        if isinstance(prompt, str):
            prompt = [prompt]

        self.weight_dtype = weight_dtype

        data = {}
        control = []
        img_shapes = []

        if use_native_size:
            controls_size_list: list[list[int]] = [[prompt_image[0].size[1], prompt_image[0].size[0]]]
            if additional_controls:
                controls_size_list.extend([[ctrl.size[1], ctrl.size[0]] for ctrl in additional_controls[0]])
            controls_size = controls_size_list  # type: ignore[assignment]

        if best_resolution_size and controls_size:
            controls_size = [
                list(calculate_best_resolution(c_size[0], c_size[1], 1024 * 1024))  # type: ignore
                for c_size in controls_size
            ]
            logging.info(f"controls_size after best resolution  {controls_size}")

        logging.info(f"controls_size for processing {controls_size}")
        for img in prompt_image:
            # for each image, need to make one copy for text_encoder, another for image_encoder
            # convert to [C,H,W] in range [0,1]
            img = self.preprocessor.preprocess({"control": img}, controls_size=controls_size)["control"]
            control.append(img)
        control = torch.stack(control, dim=0)  # B,C,H,W
        print("control shape", control.shape)
        data["control"] = control
        data["prompt"] = prompt
        data["height"] = height if height is not None else control.shape[2]
        data["width"] = width if width is not None else control.shape[3]
        img_shapes.append((3, height, width))
        img_shapes.append((3, control.shape[2], control.shape[3]))
        print("width height", width, height)

        if height is None or width is None:
            width, height = control.shape[3], control.shape[2]
        else:
            width, height = make_image_shape_devisible(width, height, self.vae_scale_factor)

        logging.info(f"target shape for generation {width}, {height}")

        if additional_controls:
            n_controls = len(additional_controls[0])
            new_controls: dict[str, list] = {f"control_{i + 1}": [] for i in range(n_controls)}
            # [control_1_batch1, control1_batch2, ..], [control2_batch1, control2_batch2, ..]

            for controls in additional_controls:
                controls = self.preprocessor.preprocess({"controls": controls}, controls_size=controls_size)["controls"]
                for i, control in enumerate(controls):
                    new_controls[f"control_{i + 1}"].append(control)
                    img_shapes.append((3, control.shape[1], control.shape[2]))
            for k, v in new_controls.items():
                print(k, type(v), type(v[0]), type(v[0][0]))
            for i in range(n_controls):
                control_stack = torch.stack(new_controls[f"control_{i + 1}"], dim=0)
                print("new controls", control_stack.shape, f"control_{i + 1}")
                data[f"control_{i + 1}"] = control_stack
            data["n_controls"] = n_controls
        else:
            data["n_controls"] = 0

        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            assert len(negative_prompt) == len(data["prompt"]), (
                "the number of negative_prompt should be same of control"
            )  # NOQA
            data["negative_prompt"] = negative_prompt
        data["num_inference_steps"] = num_inference_steps
        data["true_cfg_scale"] = true_cfg_scale
        data["guidance"] = guidance_scale
        data["img_shapes"] = [img_shapes] * control.shape[0]
        print("data keys", data.keys())
        for k, v in data.items():
            print(k, type(v))
        return data

    def sampling_from_embeddings(self, embeddings: dict) -> torch.Tensor:
        """Sampling from embeddings. Only handle the latent diffusion steps. Output the final latents. Need
        to decode the latents to images.
        """
        num_inference_steps = embeddings["num_inference_steps"]
        true_cfg_scale = embeddings["true_cfg_scale"]
        control_latents = embeddings["control_latents"]
        prompt_embeds = embeddings["prompt_embeds"]
        prompt_embeds_mask = embeddings["prompt_embeds_mask"]
        batch_size = embeddings["control_latents"].shape[0]
        # img_shapes = self._get_image_shapes(embeddings, batch_size)
        img_shapes = embeddings["img_shapes"]
        height_image = embeddings["height"]
        width_image = embeddings["width"]

        negative_prompt = embeddings["negative_prompt"] if "negative_prompt" in embeddings else None

        has_neg_prompt = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        print("do true cfg", do_true_cfg, "has neg prompt", has_neg_prompt, "true_cfg_scale", true_cfg_scale)
        device = self.dit.device

        if do_true_cfg:
            # 清理显存以确保有足够空间进行 CFG
            torch.cuda.empty_cache()
            logging.info(f"negative_prompt: {negative_prompt}")

            # 临时将 positive prompt embeddings 移到 CPU 以节省显存
            negative_prompt_embeds = embeddings["negative_prompt_embeds"]
            negative_prompt_embeds_mask = embeddings["negative_prompt_embeds_mask"]
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device, dtype=torch.int64)

        logging.info(f"mask shape: {prompt_embeds_mask.shape}, dtype: {prompt_embeds_mask.dtype}")
        logging.info(f"prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}")

        # 5. Prepare latent variables
        num_channels_latents = self.dit.config.in_channels // 4

        height_latent = 2 * (int(height_image) // (self.vae_scale_factor * 2))
        width_latent = 2 * (int(width_image) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height_latent, width_latent)
        if "latents" in embeddings:
            latents = embeddings["latents"].to(device, dtype=self.weight_dtype)
        else:
            latents = randn_tensor(
                shape,
                generator=None,
                device=device,
                dtype=self.weight_dtype,
            )
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height_latent, width_latent)
        image_seq_len = latents.shape[1]

        print("shape of latents", latents.shape, control_latents.shape)

        logging.info(f"image latent got grad: {control_latents.requires_grad}")
        logging.info(f"latents shape: {latents.shape}")
        logging.info(f"num_channels_latents: {num_channels_latents}")
        logging.info(f"image-latent shape: {control_latents.shape}")

        logging.info(f"shape of img_shapes: {img_shapes}")
        logging.info(f"self.vae_scale_factor: {self.vae_scale_factor}")
        logging.info(f"height: {height_image}")
        logging.info(f"width: {width_image}")
        logging.info(f"image_seq_len: {image_seq_len}")

        timesteps, num_inference_steps = self.prepare_predict_timesteps(
            num_inference_steps, image_seq_len, scheduler=self.sampling_scheduler
        )

        self._num_timesteps = len(timesteps)

        # 处理guidance
        guidance_scale = embeddings["guidance"]
        if self.dit.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if do_true_cfg and negative_prompt_embeds_mask is not None
            else None
        )

        # 7. 降噪循环 (遵循原始pipeline逻辑)
        self.sampling_scheduler.set_begin_index(0)
        self.attention_kwargs: dict[str, Any] = {}

        # set to proper device
        prompt_embeds = prompt_embeds.to(device, dtype=self.weight_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(device, dtype=torch.int64)
        control_latents = control_latents.to(device, dtype=self.weight_dtype)

        if do_true_cfg:
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(device, dtype=torch.int64)
            negative_prompt_embeds = negative_prompt_embeds.to(device, dtype=self.weight_dtype)

        logging.info(f"timesteps: {timesteps}")
        logging.info(f"num_inference_steps: {num_inference_steps}")
        with torch.inference_mode():
            # progress_bar = tqdm(enumerate(timesteps), total=num_inference_steps, desc="Generating")
            # for i, t in progress_bar:
            for _, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Generating"):
                # progress_bar.set_postfix({'timestep': f'{t:.1f}'})
                # progress_bar.update()

                self._current_timestep = t
                latents = latents.to(device, dtype=self.weight_dtype)

                latent_model_input = latents
                if control_latents is not None:
                    latent_model_input = torch.cat([latents, control_latents], dim=1)

                # broadcast to batch dimension
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Usecache_context (如果transformer支持)
                with torch.inference_mode():  # 外层关掉梯度 & 减元数据
                    with self.dit.cache_context("cond"):
                        noise_pred = self.dit(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_pred[:, : latents.size(1)]
                if do_true_cfg:
                    # 临时释放正面推理结果的显存，避免两次推理同时占用显存
                    noise_pred_cpu = noise_pred.cpu()
                    torch.cuda.empty_cache()
                    with torch.inference_mode():  # 外层关掉梯度 & 减元数据
                        with self.dit.cache_context("uncond"):
                            neg_noise_pred = self.dit(
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                                encoder_hidden_states=negative_prompt_embeds,
                                img_shapes=img_shapes,
                                txt_seq_lens=negative_txt_seq_lens,
                                attention_kwargs=self.attention_kwargs,
                                return_dict=False,
                            )[0]
                            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

                    # 将正面推理结果移回 GPU 进行合并
                    noise_pred = noise_pred_cpu.to(device)
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                    # 释放中间结果显存
                    del neg_noise_pred, comb_pred, cond_norm, noise_norm
                    torch.cuda.empty_cache()

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.sampling_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)
        self._current_timestep = None
        return latents

    def decode_vae_latent(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # 8. decode final latents
        latents = latents.to(self.vae.device, dtype=self.weight_dtype)
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        final_image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        # 后处理
        final_image = self.image_processor.postprocess(final_image, output_type="pt")
        return final_image
