import copy
import os
import shutil
import torch
import random
import torch.nn.functional as F  # NOQA
import PIL
import gc
import json
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
from peft.utils import get_peft_model_state_dict
from peft import LoraConfig

from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    QwenImageEditPipeline,
    calculate_shift,
    calculate_dimensions,
    retrieve_latents,
    randn_tensor,
)
from src.utils.logger import get_logger
from src.data.cache_manager import check_cache_exists
import logging
from src.loss.edit_mask_loss import map_mask_to_latent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d] %(funcName)s: %(message)s",
    datefmt="%H:%M:%S",
)


logger = get_logger(__name__, log_level="INFO")


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


def classify(lora_weight):
    import safetensors.torch
    import re

    sd = safetensors.torch.load_file(lora_weight)
    keys = list(sd.keys())
    peft = any(re.search(r"\.lora_[AB](\.|$)", k) for k in keys)
    diff = any(".lora.down.weight" in k or ".lora.up.weight" in k for k in keys)
    proc = any(".processor" in k for k in keys)
    if peft and not diff:
        return "PEFT"
    if diff:
        return "DIFFUSERS(attn-processor)" if proc else "DIFFUSERS"
    return "UNKNOWN"


def resize_bhw(x, h, w, mode="bilinear"):
    x = x.unsqueeze(1)  # [B, 1, H, W]
    x = F.interpolate(
        x,
        size=(h, w),
        mode=mode,
        align_corners=False if mode in {"bilinear", "bicubic"} else None,
        antialias=(
            True
            if mode in {"bilinear", "bicubic"} and (h < x.shape[-2] or w < x.shape[-1])
            else False
        ),
    )
    return x.squeeze(1)


class QwenImageEditTrainer:
    """Trainer class based on QwenImageEditPipeline"""

    def __init__(self, config):
        self.config = config
        self.accelerator = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0

        # Component attributes
        self.vae = None  # AutoencoderKLQwenImage
        self.text_encoder = None  # Qwen2_5_VLForConditionalGeneration (text_encoder)
        self.transformer = None  # QwenImageTransformer2DModel
        self.tokenizer = None  # Qwen2Tokenizer
        self.scheduler = None  # FlowMatchEulerDiscreteScheduler

        # Cache-related attributes
        self.use_cache = config.cache.use_cache
        self.cache_exist = check_cache_exists(config.cache.cache_dir)
        self.cache_dir = config.cache.cache_dir

        # Other configurations
        self.quantize = config.model.quantize
        self.weight_dtype = torch.bfloat16
        self.batch_size = config.data.batch_size
        self.prompt_image_dropout_rate = config.data.init_args.get(
            "prompt_image_dropout_rate", 0.1
        )

        # Parameters obtained from VAE configuration
        self.vae_scale_factor = None
        self.vae_latent_mean = None
        self.vae_latent_std = None
        self.vae_z_dim = None
        self.adapter_name = config.model.lora.adapter_name

    def set_criterion(self):
        import torch.nn as nn

        if self.config.loss.mask_loss:
            from src.loss.edit_mask_loss import MaskEditLoss

            self.criterion = MaskEditLoss(
                forground_weight=self.config.loss.forground_weight,
                background_weight=self.config.loss.background_weight,
            )
        else:
            self.criterion = nn.MSELoss()
        self.criterion.to(self.accelerator.device)

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

        from src.models.load_model import load_vae, load_qwenvl

        self.vae = load_vae(
            "Qwen/Qwen-Image-Edit", weight_dtype=self.weight_dtype  # use original one
        )
        # same to model constructed from vae self.vae = pipe.vae
        self.text_encoder = load_qwenvl(
            "Qwen/Qwen-Image-Edit", weight_dtype=self.weight_dtype  # use original one
        )
        logging.info(f"text_encoder device: {self.text_encoder.device}")
        # self.transformer = pipe.transformer this is same as the following, verified
        from src.models.load_model import load_transformer

        self.transformer = load_transformer(
            self.config.model.pretrained_model_name_or_path,  # could use quantized version
            weight_dtype=self.weight_dtype,
        )
        # load_transformer is same as pipe.transformer

        from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
            FlowMatchEulerDiscreteScheduler,
        )

        self.processor: Qwen2VLProcessor = pipe.processor
        self.tokenizer: Qwen2Tokenizer = pipe.tokenizer
        self.scheduler: FlowMatchEulerDiscreteScheduler = pipe.scheduler
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

        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.num_channels_latents = self.transformer.config.in_channels // 4

        # Set models to training/evaluation mode
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        torch.cuda.empty_cache()

        logging.info(
            f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}"
        )

    def setup_accelerator(self):
        """Initialize accelerator and logging configuration"""
        # Setup versioned logging directory
        self.setup_versioned_logging_dir()

        # Set logging_dir to the versioned output directory directly
        # Use project_dir as logging_dir to avoid extra subdirectory creation
        accelerator_project_config = ProjectConfiguration(
            project_dir=self.config.logging.output_dir,
            logging_dir=self.config.logging.output_dir,
        )

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.train.gradient_accumulation_steps,
            mixed_precision=self.config.train.mixed_precision,
            log_with=self.config.logging.report_to,
            project_config=accelerator_project_config,
        )

        # Initialize tracker with empty project name to avoid subdirectory
        if self.config.logging.report_to == "tensorboard":
            # Create a simple config dict with only basic types for TensorBoard
            try:
                simple_config = {
                    "learning_rate": float(
                        self.config.optimizer.init_args.get("lr", 0.0001)
                    ),
                    "batch_size": int(self.config.data.batch_size),
                    "max_train_steps": int(self.config.train.max_train_steps),
                    "num_epochs": int(self.config.train.num_epochs),
                    "gradient_accumulation_steps": int(
                        self.config.train.gradient_accumulation_steps
                    ),
                    "mixed_precision": str(self.config.train.mixed_precision),
                    "lora_r": int(self.config.model.lora.r),
                    "lora_alpha": int(self.config.model.lora.lora_alpha),
                    "model_name": str(self.config.model.pretrained_model_name_or_path),
                    "checkpointing_steps": int(self.config.train.checkpointing_steps),
                }
                self.accelerator.init_trackers("", config=simple_config)
            except Exception as e:
                logging.warning(f"Failed to initialize trackers with config: {e}")
                # Initialize without config if there's an error
                self.accelerator.init_trackers("")
        logging.info(
            f"Number of devices used in DDP training: {self.accelerator.num_processes}"
        )

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

    def quantize_model(self, model, device):
        from src.models.quantize import quantize_model_to_fp8

        model = quantize_model_to_fp8(
            model,
            engine="bnb",
            verbose=True,
            device=device,
        )
        model = model.to(device)
        return model

    def add_lora_adapter(self):
        """Add LoRA adapter to transformer"""
        lora_config = LoraConfig(
            r=self.config.model.lora.r,
            lora_alpha=self.config.model.lora.lora_alpha,
            init_lora_weights=self.config.model.lora.init_lora_weights,
            target_modules=self.config.model.lora.target_modules,
        )
        self.transformer.add_adapter(lora_config, adapter_name=self.adapter_name)
        self.transformer.set_adapter(self.adapter_name)

    def set_lora(self):
        """Set LoRA configuration"""
        if self.quantize:
            self.transformer = self.quantize_model(
                self.transformer, self.accelerator.device
            )
        else:
            self.transformer.to(self.accelerator.device)

        lora_config = LoraConfig(
            r=self.config.model.lora.r,
            lora_alpha=self.config.model.lora.lora_alpha,
            init_lora_weights=self.config.model.lora.init_lora_weights,
            target_modules=self.config.model.lora.target_modules,
        )

        if (
            hasattr(self.config.model.lora, "pretrained_weight")
            and self.config.model.lora.pretrained_weight
        ):
            lora_type = classify(self.config.model.lora.pretrained_weight)
            # DIFFUSERS can be loaded directly, otherwise, need to add lora first
            if lora_type != "PEFT":
                self.transformer.load_lora_adapter(
                    self.config.model.lora.pretrained_weight,
                    adapter_name=self.adapter_name,
                )
                logging.info(
                    f"set_lora: {lora_type}  {self.config.model.lora.pretrained_weight} {self.adapter_name}"
                )
            else:
                # add lora first
                # Configure model
                import safetensors.torch

                self.transformer.add_adapter(
                    lora_config, adapter_name=self.adapter_name
                )
                self.transformer.set_adapter(self.adapter_name)
                missing, unexpected = self.transformer.load_state_dict(
                    safetensors.torch.load_file(
                        self.config.model.lora.pretrained_weight
                    ),
                    strict=False,
                )
                if len(unexpected) > 0:
                    raise ValueError(f"Unexpected keys: {unexpected}")
                logging.info(
                    f"set_lora: {lora_type} {self.config.model.lora.pretrained_weight} {self.adapter_name}"
                )
                logging.info(f"missing keys: {len(missing)}, {missing[0]}")
                # self.load_lora(self.config.model.lora.pretrained_weight)
            logging.info(
                f"set_lora: Loaded lora from {self.config.model.lora.pretrained_weight}"
            )
        else:
            self.transformer.add_adapter(lora_config, adapter_name=self.adapter_name)
            self.transformer.set_adapter(self.adapter_name)

        self.transformer.to(self.accelerator.device)

        self.transformer.requires_grad_(False)
        self.transformer.train()

        # 根据配置决定是否启用梯度检查点
        if self.config.train.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            logging.info("梯度检查点已启用，将节省显存但可能增加计算时间")

        # Train only LoRA parameters
        trainable_params = 0
        for name, param in self.transformer.named_parameters():
            if "lora" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False

        logging.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    def load_lora(self, pretrained_weight, adapter_name="default"):
        """Load pretrained LoRA weights"""
        lora_type = classify(self.config.model.lora.pretrained_weight)
        # DIFFUSERS can be loaded directly, otherwise, need to add lora first
        if lora_type != "PEFT":
            self.transformer.load_lora_adapter(
                pretrained_weight, adapter_name=adapter_name
            )
            logging.info(f"Loaded LoRA weights from {pretrained_weight}")
        else:
            import safetensors.torch

            self.add_lora_adapter()
            missing, unexpected = self.transformer.load_state_dict(
                safetensors.torch.load_file(pretrained_weight),
                strict=False,
            )
            if len(unexpected) > 0:
                raise ValueError(f"Unexpected keys: {unexpected}")
            logging.info(
                f"set_lora: {lora_type} {self.config.model.lora.pretrained_weight} {self.adapter_name}"
            )
            logging.info(f"missing keys: {len(missing)}, {missing[0]}")

    def save_lora(self, save_path):
        """Save LoRA weights"""
        unwrapped_transformer = self.accelerator.unwrap_model(self.transformer)
        if is_compiled_module(unwrapped_transformer):
            unwrapped_transformer = unwrapped_transformer._orig_mod

        lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(
                unwrapped_transformer, adapter_name=self.adapter_name
            )
        )

        QwenImageEditPipeline.save_lora_weights(
            save_path, lora_state_dict, safe_serialization=True
        )
        logging.info(f"Saved LoRA weights to {save_path}")

    def setup_versioned_logging_dir(self):
        """设置版本化的日志目录"""
        base_output_dir = self.config.logging.output_dir
        project_name = self.config.logging.tracker_project_name

        # 创建项目目录结构: output_dir/project_name/v0
        project_dir = os.path.join(base_output_dir, project_name)

        # 如果项目目录不存在，直接使用 v0
        if not os.path.exists(project_dir):
            versioned_dir = os.path.join(project_dir, "v0")
            self.config.logging.output_dir = versioned_dir
            logging.info(f"创建新的训练版本目录: {versioned_dir}")
            return

        # 查找现有版本
        existing_versions = []
        for item in os.listdir(project_dir):
            item_path = os.path.join(project_dir, item)
            if os.path.isdir(item_path) and item.startswith("v") and item[1:].isdigit():
                version_num = int(item[1:])
                existing_versions.append((version_num, item_path))

        # 清理无效版本（训练步数 < 5）
        valid_versions = []
        for version_num, version_path in existing_versions:
            if self._is_valid_training_version(version_path):
                valid_versions.append(version_num)
            else:
                logging.info(f"移除无效训练版本: {version_path}")
                try:
                    shutil.rmtree(version_path)
                except Exception as e:
                    logging.info(f"移除无效训练版本失败: {version_path}, {e}")

        # 确定新版本号
        if valid_versions:
            next_version = max(valid_versions) + 1
        else:
            next_version = 0

        # 创建新版本目录
        versioned_dir = os.path.join(project_dir, f"v{next_version}")
        self.config.logging.output_dir = versioned_dir
        logging.info(f"使用训练版本目录: {versioned_dir}")

    def _is_valid_training_version(self, version_path):
        """检查版本是否包含有效的训练数据（步数 >= 5）"""
        # 检查 checkpoint 目录
        checkpoints = []
        if os.path.exists(version_path):
            for item in os.listdir(version_path):
                if item.startswith("checkpoint-") and os.path.isdir(
                    os.path.join(version_path, item)
                ):
                    try:
                        # 从 checkpoint-{epoch}-{global_step} 中提取 global_step
                        parts = item.split("-")
                        if len(parts) >= 3:
                            global_step = int(parts[2])
                            checkpoints.append(global_step)
                    except (ValueError, IndexError):
                        continue

        # 检查 tensorboard 日志（现在直接在版本目录下）
        has_logs = False
        if os.path.exists(version_path):
            # 查找 tensorboard 事件文件，可能在项目名子目录中
            for root, dirs, files in os.walk(version_path):
                log_files = [f for f in files if f.startswith("events.out.tfevents")]
                if log_files:
                    has_logs = True
                    break

        # 如果有检查点且最大步数 >= 5，或者只有日志文件但没有检查点，认为有效
        if checkpoints:
            return max(checkpoints) >= 5
        elif has_logs:
            # 如果只有日志但没有检查点，可能是训练刚开始就中断了，认为无效
            return False
        else:
            # 既没有检查点也没有日志，认为无效
            return False

    def merge_lora(self):
        """Merge LoRA weights into base model"""
        self.transformer.merge_adapter()
        logging.info("Merged LoRA weights into base model")

    def set_model_devices(self, mode="train"):
        """Set model device allocation based on different modes"""
        if mode == "train":
            assert hasattr(
                self, "accelerator"
            ), "accelerator must be set before setting model devices"

        if self.cache_exist and self.use_cache and mode == "train":
            # Cache mode: only need transformer
            self.text_encoder.cpu()
            torch.cuda.empty_cache()
            self.vae.cpu()
            torch.cuda.empty_cache()
            del self.text_encoder
            del self.vae
            gc.collect()
            self.transformer.to(self.accelerator.device)

        elif not self.use_cache and mode == "train":
            # Non-cache mode: need encoders
            self.vae.to(self.accelerator.device)
            self.vae.decoder.cpu()
            torch.cuda.empty_cache()
            gc.collect()
            self.vae.encoder.to(self.accelerator.device)
            self.text_encoder.to(self.accelerator.device)
            self.transformer.to(self.accelerator.device)

        elif mode == "cache":
            # Cache mode: need encoders, don't need transformer
            self.vae = self.vae.to(
                self.config.cache.vae_encoder_device, non_blocking=True
            )
            self.text_encoder = self.text_encoder.to(
                self.config.cache.text_encoder_device, non_blocking=True
            )

            torch.cuda.synchronize()
            self.transformer.cpu()
            torch.cuda.empty_cache()
            del self.transformer
            gc.collect()
            self.vae.decoder.cpu()
            torch.cuda.empty_cache()
            gc.collect()

        elif mode == "predict":
            # Predict mode: allocate to different GPUs according to configuration
            devices = self.config.predict.devices
            self.vae.to(devices["vae"])
            self.text_encoder.to(devices["text_encoder"])
            self.transformer.to(devices["transformer"])

    def decode_vae_latent(self, latents):
        """Decode VAE latent vectors to RGB images"""
        latents = latents.to(self.vae.device, dtype=self.weight_dtype)

        # Reverse normalization
        latents_mean = (
            torch.tensor(self.vae_latent_mean, dtype=self.weight_dtype)
            .view(1, 1, self.vae_z_dim, 1, 1)
            .to(latents.device)
        )
        latents_std = (
            torch.tensor(self.vae_latent_std, dtype=self.weight_dtype)
            .view(1, 1, self.vae_z_dim, 1, 1)
            .to(latents.device)
        )
        latents = latents * latents_std + latents_mean

        # Convert dimension format
        latents = latents.permute(0, 2, 1, 3, 4)

        # Decode image
        image = self.vae.decode(latents).sample

        # Post-processing
        image = self._postprocess_image(image)
        return image

    # Static methods: directly reference QwenImageEditPipeline methods
    _pack_latents = staticmethod(QwenImageEditPipeline._pack_latents)
    _unpack_latents = staticmethod(QwenImageEditPipeline._unpack_latents)

    def _preprocess_image_for_cache(
        self, image: torch.Tensor, adaptive_resolution=True
    ):
        """Preprocess images for caching"""
        # Convert to PIL image
        image = Image.fromarray(image.permute(1, 2, 0).cpu().numpy().astype("uint8"))
        if adaptive_resolution:
            calculated_width, calculated_height, _ = calculate_dimensions(
                1024 * 1024, image.size[0] / image.size[1]
            )
            # Use processor's resize method
            image = self.processor.image_processor.resize(
                image, calculated_height, calculated_width
            )
        return image

    def _vae_image_standardization(self, image: PIL.Image):
        """VAE image standardization"""
        image = np.array(image).astype("float32")
        image = (image / 127.5) - 1
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image = image.unsqueeze(0)
        pixel_values = image.unsqueeze(2)
        return pixel_values

    def _postprocess_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Post-process output images"""
        image = image_tensor.cpu().float()
        image = image.squeeze(2).squeeze(0)  # [C,H,W]
        image = (image / 2 + 0.5).clamp(0, 1)  # Convert from [-1,1] to [0,1]
        image = image.permute(1, 2, 0).numpy()  # [H,W,C]
        image = (image * 255).astype(np.uint8)
        return image

    def training_step(self, batch):
        """Execute a single training step"""
        # Check if cached data is available
        if (
            "prompt_embed" in batch
            and "pixel_latent" in batch
            and "control_latent" in batch
        ):
            return self._training_step_cached(batch)
        else:
            return self._training_step_compute(batch)

    def _training_step_cached(self, batch):
        """Training step using cached embeddings"""
        pixel_latents = batch["pixel_latent"].to(
            self.accelerator.device, dtype=self.weight_dtype
        )
        control_latents = batch["control_latent"].to(
            self.accelerator.device, dtype=self.weight_dtype
        )
        prompt_embeds = batch["prompt_embed"].to(self.accelerator.device)
        prompt_embeds_mask = batch["prompt_embeds_mask"].to(self.accelerator.device)
        prompt_embeds_mask = prompt_embeds_mask.to(
            torch.int64
        )  # original is int64 dtype

        image = batch["image"]  # torch.tensor: B,C,H,W
        image = image[0].cpu().numpy()
        image = image.transpose(1, 2, 0)
        image = Image.fromarray(image)
        image = [image]

        _, _, height, width = self.adjust_image_size(image)

        if self.config.loss.mask_loss:
            edit_mask = batch["mask"]  # torch.tensor: B,H,W
            edit_mask = resize_bhw(edit_mask, height, width)
            edit_mask = map_mask_to_latent(edit_mask)
        else:
            edit_mask = None

        return self._compute_loss(
            pixel_latents,
            control_latents,
            prompt_embeds,
            prompt_embeds_mask,
            height,
            width,
            edit_mask=edit_mask,
        )

    def _training_step_compute(self, batch):
        """Training step with embedding computation (no cache)"""
        image, control, prompt = batch["image"], batch["control"], batch["prompt"]
        # convert to list
        image = [x.cpu().numpy().transpose(1, 2, 0) for x in image]
        control = [x.cpu().numpy().transpose(1, 2, 0) for x in control]
        image = [Image.fromarray(x) for x in image]
        control = [Image.fromarray(x) for x in control]

        # doing preprocessing

        control_tensor, prompt_image_processed, height_control, width_control = (
            self.adjust_image_size(control)
        )
        image_tensor, _, height_image, width_image = self.adjust_image_size(image)

        if self.config.loss.mask_loss:
            edit_mask = batch["mask"]  # torch.tensor: B,H,W
            edit_mask = resize_bhw(edit_mask, height_image, width_image)
            edit_mask = map_mask_to_latent(edit_mask)
        else:
            edit_mask = None

        # Encode prompt
        new_prompts = []
        # # random drop the prompt to be empty string
        for p in prompt:
            if random.random() < self.config.data.init_args.get(
                "caption_dropout_rate", 0.1
            ):
                new_prompts.append("")
            else:
                new_prompts.append(p)

        prompt = new_prompts

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_image_processed,
            prompt=prompt,
            device=self.accelerator.device,
        )
        batch_size = image_tensor.shape[0]
        _, image_latents = self.prepare_latents(
            image_tensor,
            batch_size,
            self.num_channels_latents,
            height_image,
            width_image,
            torch.bfloat16,
            self.accelerator.device,
            generator=None,
            latents=None,
        )

        _, control_latents = self.prepare_latents(
            control_tensor,
            batch_size,
            self.num_channels_latents,
            height_control,
            width_control,
            torch.bfloat16,
            self.accelerator.device,
            generator=None,
            latents=None,
        )

        return self._compute_loss(
            image_latents,
            control_latents,
            prompt_embeds,
            prompt_embeds_mask,
            height_image,
            width_image,
            edit_mask=edit_mask,
        )

    def _compute_loss(
        self,
        pixel_latents,
        control_latents,
        prompt_embeds,
        prompt_embeds_mask,
        height,
        width,
        edit_mask=None,
    ):
        """calculate the flowmatching loss
        pixel_latents: is the packed latent, shape is
          [batch_size, (height // 2) * (width // 2), num_channels_latents * 4]
        will ignore the pack latent operation in the loss calculation
        prompt_mask is int64 tensor, shape is [batch_size, max_length]
        prompt_embeds is bfloat16 tensor, shape is [batch_size, max_length, hidden_size]
        """

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
            timesteps = self.scheduler.timesteps[indices].to(
                device=pixel_latents.device
            )

            sigmas = self._get_sigmas(
                timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype
            )
            noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise

            # image shape
            img_shapes = [
                [
                    (
                        1,
                        height // self.vae_scale_factor // 2,
                        width // self.vae_scale_factor // 2,
                    ),
                    (
                        1,
                        height // self.vae_scale_factor // 2,
                        width // self.vae_scale_factor // 2,
                    ),
                ]
            ] * batch_size

            packed_input = torch.cat([noisy_model_input, control_latents], dim=1)
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
            # prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)

        # timesteps tensor([374., 749.], device='cuda:0') packedinput shape torch.Size([2, 8208, 64])
        # prompt_embeds shape torch.Size([4, 1071, 3584])
        # prompt_embeds_mask shape torch.Size([2, 1071])
        # img_shapes [[(1, 54, 76), (1, 54, 76)], [(1, 54, 76), (1, 54, 76)]]
        # txt_seq_lens [814, 1071]

        model_pred = self.transformer(
            hidden_states=packed_input,
            timestep=timesteps / 1000,
            guidance=None,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        model_pred = model_pred[:, : pixel_latents.size(1)]

        # Calculate loss
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme="none", sigmas=sigmas
        )
        target = noise - pixel_latents
        # pred shape [2, 4104, 64], target shape [2, 4104, 64]
        # target = target.permute(0, 2, 1, 3, 4)
        loss = self.forward_loss(model_pred, target, weighting, edit_mask)
        return loss

    def forward_loss(self, model_pred, target, weighting, edit_mask=None):
        if edit_mask is None:
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
            num_warmup_steps=self.config.lr_scheduler.warmup_steps
            * self.accelerator.num_processes,
            num_training_steps=self.config.train.max_train_steps
            * self.accelerator.num_processes,
        )

    def accelerator_prepare(self, train_dataloader):
        """Prepare accelerator"""
        lora_layers_model = AttnProcsLayers(get_lora_layers(self.transformer))

        # 根据配置决定是否启用梯度检查点
        if self.config.train.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        if self.config.train.resume_from_checkpoint is not None:
            # self.accelerator.load_state(self.config.train.resume_from_checkpoint)
            self.optimizer.load_state_dict(
                torch.load(
                    os.path.join(
                        self.config.train.resume_from_checkpoint, "optimizer.bin"
                    )
                )
            )
            self.lr_scheduler.load_state_dict(
                torch.load(
                    os.path.join(
                        self.config.train.resume_from_checkpoint, "scheduler.bin"
                    )
                )
            )
            logging.info(
                f"Loaded optimizer and scheduler from {self.config.train.resume_from_checkpoint}"
            )

        lora_layers_model, optimizer, train_dataloader, lr_scheduler = (
            self.accelerator.prepare(
                lora_layers_model, self.optimizer, train_dataloader, self.lr_scheduler
            )
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Trackers already initialized in setup_accelerator
        return train_dataloader

    def save_checkpoint(self, epoch, global_step):
        """Save checkpoint"""
        if not self.accelerator.is_main_process:
            return
        save_path = os.path.join(
            self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}"
        )
        self.accelerator.save_state(save_path)
        with open(os.path.join(save_path, "state.json"), "w") as f:
            json.dump({"global_step": global_step, "epoch": epoch}, f)

        # save_path = os.path.join(
        #     self.config.logging.output_dir,
        #     f"checkpoint-{epoch}-{global_step}"
        # )
        # os.makedirs(save_path, exist_ok=True)
        # self.save_lora(save_path)
        #     self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}"
        # )
        #     if len(checkpoints) >= self.config.train.checkpoints_total_limit:
        #         num_to_remove = len(checkpoints) - self.config.train.checkpoints_total_limit + 1
        #         removing_checkpoints = checkpoints[0:num_to_remove]

        #         for removing_checkpoint in removing_checkpoints:
        #             removing_checkpoint = os.path.join(
        #                 self.config.logging.output_dir, removing_checkpoint
        #             )
        #             shutil.rmtree(removing_checkpoint)

        # save_path = os.path.join(
        #     self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}"
        # )
        # os.makedirs(save_path, exist_ok=True)

        # # Save LoRA weights
        # self.save_lora(save_path)

    def adjust_image_size(
        self, image: Union[Image.Image, List[Image.Image]]
    ) -> Tuple[torch.Tensor, Image.Image]:
        image_size = image[0].size if isinstance(image, list) else image.size

        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        height = calculated_height
        width = calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        image = self.image_processor.resize(image, calculated_height, calculated_width)
        prompt_image_processed = image
        image = self.image_processor.preprocess(
            image, calculated_height, calculated_width
        )
        image = image.unsqueeze(2)
        return image, prompt_image_processed, height, width

    def cache_step(self, data: dict, vae_encoder_device: str, text_encoder_device: str):
        """cache vae latent and prompt embedding, including empty prompt"""
        image, control, prompt = data["image"], data["control"], data["prompt"]
        # image from dataset is the RGB numpy image, in format C,H,W
        # convert to PIL image first such that it can utilize the method from pipeline
        image = image.transpose(1, 2, 0)
        control = control.transpose(1, 2, 0)

        image = Image.fromarray(image)
        control = Image.fromarray(control)
        image = [image]
        control = [control]
        prompt = [prompt]

        control_tensor, prompt_image_processed, height_control, width_control = (
            self.adjust_image_size(control)
        )
        image_tensor, _, height_image, width_image = self.adjust_image_size(image)

        file_hashes = data["file_hashes"]
        image_hash = file_hashes["image_hash"]
        control_hash = file_hashes["control_hash"]
        prompt_hash = file_hashes["prompt_hash"]
        empty_prompt_hash = file_hashes["empty_prompt_hash"]

        # calculate prompt embedding
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_image_processed,
            prompt=prompt,
            device=text_encoder_device,
        )

        empty_prompt_embed, empty_prompt_embeds_mask = self.encode_prompt(
            image=prompt_image_processed,
            prompt=[""],
            device=text_encoder_device,
        )

        # calculate latents of vae encoder for image and control

        _, image_latents = self.prepare_latents(
            image_tensor,
            1,
            self.num_channels_latents,
            height_image,
            width_image,
            torch.bfloat16,
            vae_encoder_device,
            generator=None,
            latents=None,
        )
        # first output of the latent is the noise that used in denoising steps, we dont need

        _, control_latents = self.prepare_latents(
            control_tensor,
            1,
            self.num_channels_latents,
            height_control,
            width_control,
            torch.bfloat16,
            vae_encoder_device,
            generator=None,
            latents=None,
        )

        """shape for different embeddings/latents with some real examples
        shape of prompt_embeds torch.Size([1, 900, 3584])
        shape of prompt_embeds_mask torch.Size([1, 900])
        shape of empty_prompt_embed torch.Size([1, 637, 3584])
        shape of empty_prompt_embeds_mask torch.Size([1, 637])
        shape of image_latents torch.Size([1, 4104, 64])
        shape of control_latents torch.Size([1, 4104, 64])
        """

        # unload from gpu and remove batch dimension
        image_latents = image_latents[0].cpu()
        control_latents = control_latents[0].cpu()
        prompt_embeds = prompt_embeds[0].cpu()
        prompt_embeds_mask = prompt_embeds_mask[0].cpu()
        empty_prompt_embed = empty_prompt_embed[0].cpu()
        empty_prompt_embeds_mask = empty_prompt_embeds_mask[0].cpu()

        self.cache_manager.save_cache("pixel_latent", image_hash, image_latents)
        self.cache_manager.save_cache("control_latent", control_hash, control_latents)
        self.cache_manager.save_cache("prompt_embed", prompt_hash, prompt_embeds)
        self.cache_manager.save_cache(
            "prompt_embeds_mask", prompt_hash, prompt_embeds_mask
        )
        self.cache_manager.save_cache(
            "empty_prompt_embed", empty_prompt_hash, empty_prompt_embed
        )
        self.cache_manager.save_cache(
            "empty_prompt_embeds_mask", empty_prompt_hash, empty_prompt_embeds_mask
        )

    def cache(self, train_dataloader):
        """Pre-compute and cache embeddings"""
        from tqdm.rich import tqdm


        self.cache_manager = train_dataloader.cache_manager
        vae_encoder_device = self.config.cache.vae_encoder_device
        text_encoder_device = self.config.cache.text_encoder_device

        logging.info("Starting embedding caching process...")

        # load models
        self.load_model(text_encoder_device=text_encoder_device)
        self.text_encoder.eval()
        self.vae.eval()
        self.set_model_devices(mode="cache")

        # cache for each item
        dataset = train_dataloader.dataset
        for data in tqdm(dataset, total=len(dataset), desc="cache_embeddings"):
            self.cache_step(data, vae_encoder_device, text_encoder_device)

        logging.info("Cache completed")

        # Clean up models
        self.text_encoder.cpu()
        self.vae.cpu()
        del self.text_encoder
        del self.vae

    def fit(self, train_dataloader):
        """Main training loop"""
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

        self.set_lora()
        self.text_encoder.eval()
        self.vae.eval()
        self.configure_optimizers()
        self.set_model_devices(mode="train")
        self.set_criterion()
        train_dataloader = self.accelerator_prepare(train_dataloader)

        logging.info("***** Running training *****")
        logging.info(f"  Instantaneous batch size per device = {self.batch_size}")
        logging.info(
            f"  Gradient Accumulation steps = {self.config.train.gradient_accumulation_steps}"
        )
        logging.info(f"  Use cache: {self.use_cache}, Cache exists: {self.cache_exist}")

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
        # 添加validation sampling setup (如果启用)
        validation_sampler = None
        if (
            hasattr(self.config.logging, "sampling")
            and self.config.logging.sampling.enable
        ):
            from src.validation.validation_sampler import ValidationSampler

            validation_sampler = ValidationSampler(
                config=self.config.logging.sampling,
                accelerator=self.accelerator,
                weight_dtype=self.weight_dtype,
            )

            # Setup validation dataset
            validation_sampler.setup_validation_dataset(train_dataloader.dataset)

            # Cache embeddings for validation using trainer methods
            embeddings_config = {
                "cache_vae_embeddings": True,  # Cache VAE latents
                "cache_text_embeddings": True,  # Cache Qwen text embeddings
            }
            validation_sampler.cache_embeddings(self, embeddings_config)
        # Progress bar
        progress_bar = tqdm(
            range(self.global_step, self.config.train.max_train_steps),
            desc="train",
            disable=not self.accelerator.is_local_main_process,
        )

        for epoch in range(start_epoch, self.config.train.num_epochs):
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
                    train_loss += (
                        avg_loss.item() / self.config.train.gradient_accumulation_steps
                    )
                    running_loss = train_loss

                    # Log metrics
                    self.accelerator.log(
                        {"train_loss": train_loss}, step=self.global_step
                    )
                    train_loss = 0.0

                    # Save checkpoint
                    if self.global_step % self.config.train.checkpointing_steps == 0:
                        self.save_checkpoint(epoch, self.global_step)
                    # 添加validation sampling
                    if validation_sampler and validation_sampler.should_run_validation(
                        self.global_step
                    ):
                        try:
                            validation_sampler.run_validation_loop(
                                global_step=self.global_step,
                                trainer=self,  # 传入trainer实例
                            )
                        except Exception as e:
                            self.accelerator.print(f"Validation sampling failed: {e}")

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

    def prepare_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator=None,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, 1, num_channels_latents, height, width)

        image_latents = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
            if (
                batch_size > image_latents.shape[0]
                and batch_size % image_latents.shape[0] == 0
            ):
                # expand init_latents for batch_size
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat(
                    [image_latents] * additional_image_per_prompt, dim=0
                )
            elif (
                batch_size > image_latents.shape[0]
                and batch_size % image_latents.shape[0] != 0
            ):
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
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
            latents = self._pack_latents(
                latents, batch_size, num_channels_latents, height, width
            )
        else:
            latents = latents.to(device=device, dtype=dtype)

        return latents, image_latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator = None):
        # generator is None by default
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]),
                    generator=generator[i],
                    sample_mode="argmax",
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(
                self.vae.encode(image), generator=generator, sample_mode="argmax"
            )
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

    def setup_predict(self):
        """Setup prediction mode"""
        self.load_model()
        if self.quantize:
            self.transformer = self.quantize_model(
                self.transformer, self.config.predict.devices["transformer"]
            )

        # Load LoRA weights (if available)
        if (
            hasattr(self.config.model.lora, "pretrained_weight")
            and self.config.model.lora.pretrained_weight
        ):
            self.load_lora(
                self.config.model.lora.pretrained_weight, adapter_name=self.adapter_name
            )

        # Set evaluation mode
        self.transformer.eval()
        self.vae.eval()
        self.text_encoder.eval()

        # Allocate devices
        self.set_model_devices(mode="predict")

        # Quantize (if enabled)
        if self.quantize:
            self.transformer = self.quantize_model(
                self.transformer, self.config.predict.devices["transformer"]
            )

    def predict(
        self,
        prompt_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Union[str, List[str]],
        negative_prompt: Union[None, str, List[str]] = None,
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        image_latents: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        prompt_embeds_mask: torch.Tensor = None,
        weight_dtype=torch.bfloat16,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Prediction method - follows original QwenImageEditPipeline.__call__ logic, supports batch processing

        Args:
            prompt_image: PIL.Image.Image or List[PIL.Image.Image], input prompt images
            prompt: str or List[str], text prompts (if list, must match prompt_image length)
            negative_prompt: str or List[str], negative text prompts, default empty
            num_inference_steps: int, number of inference steps, default 20
            true_cfg_scale: float, true CFG guidance strength, default 4.0

        Returns:
            Union[np.ndarray, List[np.ndarray]]: Generated images in RGB format
        """
        import logging

        logging.info(
            f"Starting prediction with {num_inference_steps} steps, CFG scale: {true_cfg_scale}"
        )
        assert prompt_image is not None, "prompt_image is required"
        assert prompt is not None, "prompt is required"

        # Process input format
        if isinstance(prompt_image, PIL.Image.Image):
            image = [prompt_image]
        else:
            image = prompt_image
        self.weight_dtype = weight_dtype

        # 1. Calculate image dimensions (follows original pipeline logic)
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        height = calculated_height
        width = calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 2. 定义批次参数 (遵循原始pipeline逻辑)

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        device_text_encoder = self.config.predict.devices["text_encoder"]
        device_transformer = self.config.predict.devices["transformer"]
        device_vae = self.config.predict.devices["vae"]

        # 3. 预处理图像 (遵循原始pipeline逻辑)
        if image is not None and not (
            isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels
        ):
            image = [
                self.image_processor.resize(xx, calculated_height, calculated_width)
                for xx in image
            ]
            prompt_image_processed = image
            image = self.image_processor.preprocess(
                image, calculated_height, calculated_width
            )
            image = image.unsqueeze(2)

        has_neg_prompt = negative_prompt is not None
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # 4. encode prompt
        self.text_encoder.to(device_text_encoder)
        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds.unsqueeze(0)
            prompt_embeds_mask = prompt_embeds_mask.unsqueeze(0).to(torch.int64)

        else:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                image=prompt_image_processed,
                prompt=prompt,
                device=device_text_encoder,
            )
            prompt_embeds_mask = prompt_embeds_mask.to(torch.int64)
            logging.info(f"prompt_embeds_mask shape: {prompt_embeds_mask.shape}")

        if do_true_cfg:
            # 清理显存以确保有足够空间进行 CFG
            torch.cuda.empty_cache()
            logging.info(f"negative_prompt: {negative_prompt}")

            # 临时将 positive prompt embeddings 移到 CPU 以节省显存
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=prompt_image_processed,
                prompt=negative_prompt,
                device=device_text_encoder,
            )
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(
                device_transformer, dtype=torch.int64
            )

        logging.info(
            f"mask shape: {prompt_embeds_mask.shape}, dtype: {prompt_embeds_mask.dtype}"
        )
        logging.info(
            f"prompt_embeds shape: {prompt_embeds.shape}, dtype: {prompt_embeds.dtype}"
        )

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        if image_latents is not None:
            image_latents = image_latents.unsqueeze(0)
            height_latent = 2 * (int(height) // (self.vae_scale_factor * 2))
            width_latent = 2 * (int(width) // (self.vae_scale_factor * 2))

            shape = (batch_size, 1, num_channels_latents, height_latent, width_latent)
            latents = randn_tensor(
                shape,
                generator=None,
                device=device_transformer,
                dtype=prompt_embeds.dtype,
            )
            latents = self._pack_latents(
                latents, batch_size, num_channels_latents, height_latent, width_latent
            )
            latents = latents.to(device=device_transformer, dtype=prompt_embeds.dtype)
        else:
            latents, image_latents = self.prepare_latents(
                image,
                batch_size,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device_vae,
                generator=None,
                latents=None,
            )
        logging.info(f"image latent got grad: {image_latents.requires_grad}")
        logging.info(f"latents shape: {latents.shape}")
        logging.info(f"num_channels_latents: {num_channels_latents}")
        logging.info(f"image-latent shape: {image_latents.shape}")

        img_shapes = [
            [
                (
                    1,
                    height // self.vae_scale_factor // 2,
                    width // self.vae_scale_factor // 2,
                ),
                (
                    1,
                    calculated_height // self.vae_scale_factor // 2,
                    calculated_width // self.vae_scale_factor // 2,
                ),
            ]
        ] * batch_size

        logging.info(f"shape of img_shapes: {img_shapes}")
        logging.info(f"self.vae_scale_factor: {self.vae_scale_factor}")
        logging.info(f"height: {height}")
        logging.info(f"width: {width}")
        logging.info(f"calculated_height: {calculated_height}")
        logging.info(f"calculated_width: {calculated_width}")

        # 6. 准备时间步 (遵循原始pipeline逻辑)
        import numpy as np
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
            retrieve_timesteps,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device_transformer,
            sigmas=sigmas,
            mu=mu,
        )

        self._num_timesteps = len(timesteps)

        # 处理guidance
        guidance_scale = 1.0
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device_transformer, dtype=torch.float32
            )
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None
        logging.info(
            f"prompt_embeds_mask.sum(dim=1): {prompt_embeds_mask.sum(dim=1)}",
            f"prompt_embeds_mask[:2]: {prompt_embeds_mask[:2]}",
            f"prompt_embeds_mask[:2]: {prompt_embeds_mask[:2]}",
            f"prompt_embeds_mask.sum(dim=1).tolist(): {prompt_embeds_mask.sum(dim=1).tolist()}",
        )
        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist()
            if do_true_cfg and negative_prompt_embeds_mask is not None
            else None
        )

        # 7. 降噪循环 (遵循原始pipeline逻辑)
        self.scheduler.set_begin_index(0)
        self.attention_kwargs = {}

        # set to proper device
        prompt_embeds = prompt_embeds.to(device_transformer, dtype=self.weight_dtype)
        prompt_embeds_mask = prompt_embeds_mask.to(
            device_transformer, dtype=torch.int64
        )

        if do_true_cfg:
            negative_prompt_embeds_mask = negative_prompt_embeds_mask.to(
                device_transformer, dtype=torch.int64
            )
            negative_prompt_embeds = negative_prompt_embeds.to(
                device_transformer, dtype=self.weight_dtype
            )

        logging.info(f"timesteps: {timesteps}")
        logging.info(f"num_inference_steps: {num_inference_steps}")
        logging.info(f"sigmas: {sigmas}")
        with torch.inference_mode():
            # progress_bar = tqdm(enumerate(timesteps), total=num_inference_steps, desc="Generating")
            # for i, t in progress_bar:
            for i, t in tqdm(enumerate(timesteps), desc="Generating"):
                # progress_bar.set_postfix({'timestep': f'{t:.1f}'})
                # progress_bar.update()

                self._current_timestep = t
                latents = latents.to(device_transformer, dtype=self.weight_dtype)

                latent_model_input = latents
                if image_latents is not None:
                    image_latents = image_latents.to(
                        device_transformer, dtype=self.weight_dtype
                    )
                    latent_model_input = torch.cat([latents, image_latents], dim=1)

                # broadcast to batch dimension
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                # Usecache_context (如果transformer支持)
                if i == 0:
                    logging.info(f"latent_model_input: {latent_model_input.shape}")
                    logging.info(f"timestep: {timestep}")
                    logging.info(f"guidance: {guidance}")
                    logging.info(f"prompt_embeds_mask: {prompt_embeds_mask.shape}")
                    logging.info(f"prompt_embeds: {prompt_embeds.shape}")
                    logging.info(f"img_shapes: {img_shapes}")
                    logging.info(f"txt_seq_lens: {txt_seq_lens}")
                    logging.info(f"attention_kwargs: {self.attention_kwargs}")
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
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

                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer(
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
                    noise_pred = noise_pred_cpu.to(device_transformer)
                    comb_pred = neg_noise_pred + true_cfg_scale * (
                        noise_pred - neg_noise_pred
                    )

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                    # 释放中间结果显存
                    del neg_noise_pred, comb_pred, cond_norm, noise_norm
                    torch.cuda.empty_cache()

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

        self._current_timestep = None
        # 8. decode final latents
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(self.vae.dtype)
        latents = latents.to(device_vae)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        final_image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        # 后处理
        final_image = self.image_processor.postprocess(final_image, output_type="pil")
        return final_image

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
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
        num_images_per_prompt = 1  # 固定为1，支持单图像生成
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        with torch.inference_mode():
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt, image, device
            )
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(
            batch_size * num_images_per_prompt, seq_len
        )
        return prompt_embeds, prompt_embeds_mask

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
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

        with torch.cuda.device(
            model_inputs.input_ids.device
        ):  # 作用域内的 'cuda' 都指向同一张卡
            outputs = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(
            hidden_states, model_inputs.attention_mask
        )
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [
            torch.ones(e.size(0), dtype=torch.long, device=e.device)
            for e in split_hidden_states
        ]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
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

    # Validation sampling methods
    def _encode_vae_image_for_validation(self, image):
        """Encode image for validation using existing VAE encoding logic"""
        if isinstance(image, str):  # 如果是路径，先加载图片
            from PIL import Image

            image = Image.open(image)

        # 转换为tensor并预处理
        image_array = np.array(image)
        if len(image_array.shape) == 2:  # 灰度图转RGB
            image_array = np.stack([image_array] * 3, axis=-1)

        # 使用现有的图像预处理和VAE编码逻辑
        image_tensor, _, height, width = self.adjust_image_size([image])
        _, image_latents = self.prepare_latents(
            image_tensor,
            1,
            self.vae_z_dim,
            height,
            width,
            self.weight_dtype,
            self.accelerator.device,
        )
        return image_latents[0].cpu()  # 返回CPU上的latents以节省显存

    def _encode_prompt_for_validation(self, prompt, control_image=None):
        """Encode prompt for validation using existing Qwen VL encoding logic"""
        # 准备control image用于prompt encoding
        if isinstance(control_image, str):
            from PIL import Image

            control_image = Image.open(control_image)

        control_image_processed = [control_image] if control_image else None

        # 复用现有的encode_prompt方法
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=control_image_processed,
            prompt=[prompt],
            device=self.accelerator.device,
        )
        return {
            "prompt_embeds": prompt_embeds[0].cpu(),  # 返回CPU上的embeddings
            "prompt_embeds_mask": prompt_embeds_mask[0].cpu(),
        }

    def _generate_latents_for_validation(self, cached_sample):
        """Generate latents using cached embeddings and current model"""
        # 使用缓存的embeddings进行推理生成
        text_embeddings = cached_sample["text_embeddings"]
        control_latents = cached_sample.get("control_latents")

        # 移动embeddings到GPU
        prompt_embeds = (
            text_embeddings["prompt_embeds"].unsqueeze(0).to(self.accelerator.device)
        )
        prompt_embeds_mask = (
            text_embeddings["prompt_embeds_mask"]
            .unsqueeze(0)
            .to(self.accelerator.device)
        )

        if control_latents is not None:
            control_latents = control_latents.unsqueeze(0).to(self.accelerator.device)

        # 简化的推理逻辑
        batch_size = 1
        height, width = 512, 512  # 固定尺寸用于validation

        # 准备latents
        latents, _ = self.prepare_latents(
            None,
            batch_size,
            self.vae_z_dim,
            height,
            width,
            self.weight_dtype,
            self.accelerator.device,
        )

        # 使用简化的推理步骤 (fewer steps for validation)
        num_inference_steps = 10
        timesteps = torch.linspace(
            1000, 0, num_inference_steps, device=self.accelerator.device
        )

        with torch.no_grad():
            for t in timesteps:
                # 准备模型输入
                if control_latents is not None:
                    packed_input = torch.cat([latents, control_latents], dim=1)
                else:
                    packed_input = latents

                # 准备其他输入
                img_shapes = [
                    [
                        (
                            1,
                            height // self.vae_scale_factor // 2,
                            width // self.vae_scale_factor // 2,
                        )
                    ]
                    * 2
                ] * batch_size
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

                noise_pred = self.transformer(
                    hidden_states=packed_input,
                    timestep=t.expand(latents.shape[0]) / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]

                # 只取前面对应latents的部分
                noise_pred = noise_pred[:, : latents.size(1)]

                # 简化的调度器步骤
                latents = latents - 0.1 * noise_pred  # 简化的更新规则

        return latents

    def _decode_latents_for_validation(self, latents):
        """Decode latents to image using existing VAE decoder"""
        # 复用现有的VAE解码逻辑
        latents = self._unpack_latents(latents, 512, 512, self.vae_scale_factor)

        # 反标准化
        latents_mean = (
            torch.tensor(self.vae_latent_mean)
            .view(1, self.vae_z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae_latent_std)
            .view(1, self.vae_z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        # 调整维度格式
        latents = latents.permute(0, 2, 1, 3, 4)

        # 临时将VAE decoder移到GPU
        original_device = self.vae.device
        self.vae.to(self.accelerator.device)

        try:
            with torch.no_grad():
                image = self.vae.decode(latents).sample
                # 后处理
                image = self._postprocess_image(image)
                # 转换为PIL格式
                from PIL import Image as PILImage

                image = PILImage.fromarray(image)
        finally:
            # 恢复VAE设备
            self.vae.to(original_device)

        return image
