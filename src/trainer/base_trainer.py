"""
Abstract Base Trainer for all trainer implementations.
Defines the core interface that all trainers must implement.
"""

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from src.data.cache_manager import EmbeddingCacheManager
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
import os
import shutil
import json
import numpy as np
from src.utils.sampling import calculate_shift, retrieve_timesteps
from tqdm import tqdm


import logging
import PIL
from src.data.config import Config
from src.utils.model_summary import print_model_summary_table
from src.utils.lora_utils import FpsLogger
from src.utils.tools import get_git_info
from src.utils.tools import instantiate_class
from src.scheduler.custom_flowmatch_scheduler import FlowMatchEulerDiscreteScheduler

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainer implementations.
    Defines the core interface that all trainers must implement.
    """

    def __init__(self, config: Config):
        """Initialize trainer with configuration."""
        self.config = config
        self.accelerator: Optional[Accelerator] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.scheduler: FlowMatchEulerDiscreteScheduler = None

        # Common attributes that all trainers should have
        self.weight_dtype = torch.bfloat16
        self.batch_size = self.config.data.batch_size
        self.use_cache = self.config.cache.use_cache
        self.cache_dir = self.config.cache.cache_dir
        self.fps_logger = FpsLogger()
        self.cache_manager = EmbeddingCacheManager(self.cache_dir)
        self.cache_exist = self.cache_manager.exist(self.cache_dir)
        self.quantize = self.config.model.quantize
        self.adapter_name = self.config.model.lora.adapter_name

        self.log_model_info()
        self.load_preprocessor()

    def load_preprocessor(self):
        class_path = self.config.data.init_args.processor.class_path
        init_args = self.config.data.init_args.processor.init_args
        self.preprocessor = instantiate_class(class_path, init_args)

    def __repr__(self) -> str:
        msg = f"{self.__class__.__name__}(config={self.config})"
        return msg

    def log_model_info(self):
        """Log model information."""
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Use Cache: {self.use_cache}")

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

    def accelerator_prepare(self, train_dataloader):
        """Prepare accelerator"""
        from diffusers.loaders import AttnProcsLayers
        from src.utils.lora_utils import get_lora_layers

        lora_layers_model = AttnProcsLayers(get_lora_layers(self.dit))

        # 根据配置决定是否启用梯度检查点
        if self.config.train.gradient_checkpointing:
            self.dit.enable_gradient_checkpointing()
        if self.config.resume is not None:
            # self.accelerator.load_state(self.config.train.resume_from_checkpoint)
            self.optimizer.load_state_dict(
                torch.load(os.path.join(self.config.resume, "optimizer.bin"))
            )
            self.lr_scheduler.load_state_dict(
                torch.load(os.path.join(self.config.resume, "scheduler.bin"))
            )
            logging.info(f"Loaded optimizer and scheduler from {self.config.resume}")

        lora_layers_model, optimizer, train_dataloader, lr_scheduler = (
            self.accelerator.prepare(
                lora_layers_model, self.optimizer, train_dataloader, self.lr_scheduler
            )
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Trackers already initialized in setup_accelerator
        return train_dataloader

    def merge_lora(self):
        """Merge LoRA weights into base model"""
        self.dit.merge_adapter()
        logging.info("Merged LoRA weights into base model")

    def cache_step_add_batch_dim(self, data: dict):
        """Add batch dimension to the data"""
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = value.unsqueeze(0)
        return data

    def cache(self, train_dataloader):
        """Pre-compute and cache embeddings/latents for training efficiency."""
        logging.info("Starting embedding caching process...")
        self.load_model()
        self.setup_model_device_train_mode(stage="cache", cache=True)
        # Cache for each item (same loop structure as QwenImageEditTrainer)
        dataset = train_dataloader.dataset
        for data in tqdm(dataset, total=len(dataset), desc="cache_embeddings"):
            data = self.cache_step_add_batch_dim(data)
            data = self.prepare_embeddings(data, stage="cache")
            self.cache_step(data)
        self.destroy_models()
        logging.info("Cache completed")

    def destroy_models(self):
        import gc

        if hasattr(self, "text_encoder"):
            self.text_encoder.cpu()
            del self.text_encoder
        if hasattr(self, "text_encoder_2"):
            self.text_encoder_2.cpu()
            del self.text_encoder_2
        if hasattr(self, "vae"):
            self.vae.cpu()
            del self.vae
        if hasattr(self, "vit"):
            self.vit.cpu()
            del self.vit
        torch.cuda.empty_cache()
        gc.collect()

    def setup_validation(self, train_dataloader):
        """Setup validation"""
        self.validation_sampler = None
        # TODO: do it later

    def clip_gradients(self):
        """Clip gradients"""
        if self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(
                self.dit.parameters(),
                self.config.train.max_grad_norm,
            )

    def training_step(self, batch: dict) -> torch.Tensor:
        """Training step"""
        if all(batch["cached"]):
            return self._training_step_cached(batch)
        return self._training_step_compute(batch)

    def _training_step_cached(self, batch: dict) -> torch.Tensor:
        """Training step with cached data"""
        embeddings = self.prepare_cached_embeddings(batch)
        return self._compute_loss(embeddings)

    def _training_step_compute(self, batch: dict) -> torch.Tensor:
        """Training step with real-time data"""
        embeddings = self.prepare_embeddings(batch, stage="fit")
        return self._compute_loss(embeddings)

    @abstractmethod
    def _compute_loss(self, embeddings: dict) -> torch.Tensor:
        """Compute loss. returned the flow matching loss tensor"""
        pass

    def forward_loss(self, model_pred, target, weighting=None, edit_mask=None):
        if edit_mask is None:
            if weighting is None:
                loss = torch.nn.functional.mse_loss(
                    model_pred, target, reduction="mean"
                )
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

    def train_epoch(self, epoch, train_dataloader):
        for _, batch in enumerate(train_dataloader):
            with self.accelerator.accumulate(self.dit):
                loss = self.training_step(batch)
                self.fps_logger.update(
                    batch_size=self.batch_size * self.accelerator.num_processes,
                    num_tokens=None,
                )
                self.accelerator.backward(loss)
                self.clip_gradients()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            if self.accelerator.sync_gradients:
                avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean()
                self.train_loss = (
                    avg_loss.item() / self.config.train.gradient_accumulation_steps
                )
                self.running_loss = 0.9 * self.running_loss + 0.1 * self.train_loss
                self.update_progressbar(
                    logs={
                        "loss": self.train_loss,
                        "smooth_loss": self.running_loss,
                        "lr": self.lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "fps": self.fps_logger.total_fps(),
                    }
                )
                self.save_checkpoint(epoch, self.global_step)
                if (
                    self.validation_sampler
                    and self.validation_sampler.should_run_validation(self.global_step)
                ):
                    self.fps_logger.pause()
                    try:
                        self.validation_sampler.run_validation_loop(
                            global_step=self.global_step,
                            trainer=self,  # 传入trainer实例
                        )
                    except Exception as e:
                        self.accelerator.print(f"Validation sampling failed: {e}")
                    self.fps_logger.resume()

    def setup_progressbar(self):
        self.train_loss = 0.0
        self.running_loss = 0.0
        if self.config.resume is not None:
            with open(os.path.join(self.config.resume, "state.json")) as f:
                st = json.load(f)
            self.global_step = st["global_step"]
            self.start_epoch = st["epoch"]
        else:
            self.global_step = 0
            self.start_epoch = 0
        self.progress_bar = tqdm(
            range(self.global_step, self.config.train.max_train_steps),
            desc="train",
            disable=(not self.accelerator.is_local_main_process),
        )
        self.num_epochs = int(
            self.config.train.max_train_steps
            / self.batch_size
            / self.accelerator.num_processes
        )

    def update_progressbar(self, logs: dict):
        self.accelerator.log(logs, step=self.global_step)
        logs = {
            "loss": f"{logs['loss']:.3f}",
            "smooth_loss": f"{logs['smooth_loss']:.3f}",
            "lr": f"{logs['lr']:.1e}",
            "epoch": logs['epoch'],
            "fps": f"{logs['fps']:.2f}",
        }
        self.progress_bar.update(1)
        self.global_step += 1
        self.progress_bar.set_postfix(logs)

    def fit(self, train_dataloader):
        """Main training loop implementation."""
        self.setup_accelerator()
        self.load_model()
        if self.config.resume is not None:
            # add the checkpoint in lora.pretrained_weight config
            self.config.model.lora.pretrained_weight = os.path.join(
                self.config.resume, "model.safetensors"
            )
            logging.info(
                f"Loaded checkpoint from {self.config.model.lora.pretrained_weight}"
            )
        self.__class__.load_pretrain_lora_model(self.dit, self.config, self.adapter_name)

        self.setup_model_device_train_mode(stage="fit", cache=self.use_cache)
        self.configure_optimizers()
        self.setup_criterion()
        self.setup_validation(train_dataloader)

        train_dataloader = self.accelerator_prepare(train_dataloader)
        logging.info("***** Running training *****")
        print_model_summary_table(self.dit)
        self.fps_logger.start()
        self.save_train_config()
        self.setup_progressbar()
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train_epoch(epoch, train_dataloader)
        self.fps_logger.stop()
        logging.info(f"FPS: {self.fps_logger.get_fps()}")
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()

    def setup_criterion(self):
        if self.config.loss.mask_loss:
            from src.loss.edit_mask_loss import MaskEditLoss

            self.criterion = MaskEditLoss(
                forground_weight=self.config.loss.forground_weight,
                background_weight=self.config.loss.background_weight,
            )
        else:
            self.criterion = nn.MSELoss()
        self.criterion.to(self.accelerator.device)

    def setup_predict(
        self,
    ):
        if not hasattr(self, "dit") or self.vae is None:
            logging.info("Loading model...")
            self.load_model()
        self.load_pretrain_lora_model(
            self.dit, self.config, self.config.lora_adapter_name, stage="predict"
        )
        if self.config.model.quantize:
            self.dit = self.quantize_model(
                self.dit,
                self.config.predict.devices.dit,
            )
        self.setup_model_device_train_mode(stage="predict")
        logging.info("setup_model_device_train_mode done")
        print_model_summary_table(self.dit)
        logging.info(f"setup_predict done")

    @abstractmethod
    def prepare_predict_batch_data(self, *args, **kwargs) -> dict:
        """Prepare predict batch data.
        prepare the data to batch dict that can be used to prepare embeddings similar in the training step.
        We want to reuse the same data preparation code in the training step.
        """
        pass

    def predict(
        self,
        **kwargs,
    ):
        """Inference/prediction method.
        Prepare the data, prepare the embeddings, sample the latents, decode the latents to images.
        """
        self.setup_predict()
        batch = self.prepare_predict_batch_data(**kwargs)
        embeddings: dict = self.prepare_embeddings(batch, stage="predict")
        target_height = embeddings["height"]
        target_width = embeddings["width"]
        latents = self.sampling_from_embeddings(embeddings)
        image = self.decode_vae_latent(latents, target_height, target_width)
        output_type = kwargs.get("output_type", "pil")
        if output_type == "pil":
            image = image.detach().permute(0, 2, 3, 1).float().cpu().numpy()
            image = (image * 255).round().astype("uint8")
            if image.shape[-1] == 1:
                # special case for grayscale (single channel) images
                pil_images = [
                    PIL.Image.fromarray(image.squeeze(), mode="L") for image in image
                ]
            else:
                pil_images = [PIL.Image.fromarray(image) for image in image]

            return pil_images
        return image

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

    def save_checkpoint(self, epoch, global_step):
        """Save checkpoint"""
        self.fps_logger.pause()
        if self.global_step % self.config.train.checkpointing_steps != 0:
            self.fps_logger.resume()
            return
        if self.accelerator.is_main_process:
            logging.info(f"Saving checkpoint to {self.config.logging.output_dir}")
            save_path = os.path.join(
                self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}"
            )
            self.accelerator.save_state(save_path)
            state_info = {"global_step": global_step, "epoch": epoch}
            git_info = get_git_info()
            state_info.update(git_info)
            with open(os.path.join(save_path, "state.json"), "w") as f:
                json.dump(state_info, f)
        self.fps_logger.resume()

    def save_train_config(self):
        import yaml

        d_json = self.config.model_dump(mode="json", exclude_none=True)
        train_yaml_file = os.path.join(
            self.config.logging.output_dir, "train_config.yaml"
        )
        with open(train_yaml_file, "w") as f:
            yaml.dump(d_json, f, default_flow_style=False, sort_keys=False)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        from diffusers.optimization import get_scheduler

        trainable_named_params = [
            (name, param)
            for name, param in self.dit.named_parameters()
            if param.requires_grad
        ]
        lora_layers = [param for _, param in trainable_named_params]

        # Log how many parameters are in training and show one example
        if (
            getattr(self, "accelerator", None) is None
        ) or self.accelerator.is_main_process:
            total_elements = sum(p.numel() for p in lora_layers)
            logging.info(
                f"Trainable parameters: {len(lora_layers)} tensors, total elements: {total_elements}"
            )
            if len(trainable_named_params) > 0:
                example_name, example_param = trainable_named_params[0]
                logging.info(
                    f"Example trainable param: {example_name}, shape={tuple(example_param.shape)}"
                )
                logging.info(f"Example dtype: {example_param.dtype}")

        # Use optimizer parameters from configuration

        import importlib

        optimizer_config = self.config.optimizer.init_args
        class_path = self.config.optimizer.class_path
        module_name, class_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        logging.info(f"Using optimizer: {cls}, {class_path}")
        # cls = getattr(importlib.import_module(class_path), class_path)
        self.optimizer = cls(
            lora_layers,
            **optimizer_config,
        )

        self.lr_scheduler = get_scheduler(
            self.config.lr_scheduler.scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.lr_scheduler.warmup_steps
            * self.accelerator.num_processes,
            num_training_steps=self.config.train.max_train_steps
            * self.accelerator.num_processes,
        )

    @classmethod
    def quantize_model(cls, model, device):
        from src.models.quantize import quantize_model_to_fp8

        model = quantize_model_to_fp8(
            model,
            engine="bnb",
            verbose=True,
            device=device,
        )
        model = model.to(device)
        return model

    @classmethod
    def add_lora_adapter(cls, transformer: torch.nn.Module, config, adapter_name: str):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            init_lora_weights=config.model.lora.init_lora_weights,
            target_modules=config.model.lora.target_modules,
        )
        logging.info(f"add_lora_adapter: {lora_config}, {adapter_name}")
        transformer.add_adapter(lora_config, adapter_name=adapter_name)
        transformer.set_adapter(adapter_name)

    @classmethod
    def load_pretrain_lora_model(
        cls, transformer: torch.nn.Module, config, adapter_name: str, stage='fit',
    ):
        from src.utils.lora_utils import classify_lora_weight

        pretrained_weight = getattr(config.model.lora, "pretrained_weight", None)
        if pretrained_weight:
            lora_type = classify_lora_weight(pretrained_weight)
            # DIFFUSERS can be loaded directly, otherwise, need to add lora first
            if lora_type != "PEFT":
                transformer.load_lora_adapter(
                    pretrained_weight, adapter_name=adapter_name
                )
                logging.info(
                    f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}"
                )
            else:
                # add lora first
                # Configure model
                cls.add_lora_adapter(transformer, config, adapter_name)
                import safetensors.torch

                missing, unexpected = transformer.load_state_dict(
                    safetensors.torch.load_file(pretrained_weight),
                    strict=False,
                )
                if len(unexpected) > 0:
                    raise ValueError(f"Unexpected keys: {unexpected}")
                logging.info(
                    f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}"
                )
                logging.info(f"missing keys: {len(missing)}, {missing[0]}")
                # self.load_lora(self.config.model.lora.pretrained_weight)
            logging.info(f"set_lora: Loaded lora from {pretrained_weight}")

        elif stage == 'fit':
            cls.add_lora_adapter(transformer, config, adapter_name)

    def normalize_image(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image from [0,1] to [-1,1]"""
        image = image.to(self.weight_dtype)
        return image * 2.0 - 1.0

    def prepare_predict_timesteps(
        self, num_inference_steps: int, image_seq_len: int
    ) -> torch.Tensor:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        device = next(self.dit.parameters()).device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    @abstractmethod
    def load_model(self, **kwargs):
        """Load and initialize model components."""
        pass

    @abstractmethod
    def save_lora(self, save_path):
        """Save LoRA weights"""
        pass

    @abstractmethod
    def encode_prompt(self, *args, **kwargs):
        """Encode text prompts to embeddings. Qwen-Edit pass image and prompt, Flux-Kontext pass prompt"""
        pass

    @abstractmethod
    def prepare_latents(self, *args, **kwargs):
        """Prepare latents for fit & predict. Input usually be control images"""
        pass

    @abstractmethod
    def prepare_embeddings(
        self, batch: dict, stage: str = "fit"
    ) -> Dict[str, torch.Tensor]:
        """Prepare embeddings for prediction. Call vae encoder and prompt encoder
        to get the embeddings. Used in fit & predict
        Update the embeddings keys in batch dict
        """
        return batch

    @abstractmethod
    def prepare_cached_embeddings(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Prepare cached embeddings for prediction. Loaded the cached embeddings from cache.
        Used in fit
        Update the embeddings keys in batch dict
        """
        return batch

    @abstractmethod
    def decode_vae_latent(self, latents: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        """Decode VAE latent vectors to RGB images. In range [0,1]"""
        pass

    @abstractmethod
    def sampling_from_embeddings(
        self, embeddings: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Sampling from embeddings. Only handle the latent diffusion steps. Output the final latents. Need
        to decode the latents to images.
        """
        pass


    @abstractmethod
    def cache_step(self, data: dict):
        """Cache step"""
        pass

    @abstractmethod
    def setup_model_device_train_mode(self, stage="fit", cache=False):
        pass
