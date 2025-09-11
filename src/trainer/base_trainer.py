"""
Abstract Base Trainer for all trainer implementations.
Defines the core interface that all trainers must implement.
"""

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import torch
import os
import shutil
import json
import logging

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainer implementations.
    Defines the core interface that all trainers must implement.
    """

    def __init__(self, config):
        """Initialize trainer with configuration."""
        self.config = config
        self.accelerator: Optional[Accelerator] = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0

        # Common attributes that all trainers should have
        self.weight_dtype = torch.bfloat16
        self.batch_size = config.data.batch_size
        self.use_cache = config.cache.use_cache
        self.cache_dir = config.cache.cache_dir

        logger.info(f"Initialized {self.__class__.__name__} with config")

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

    def merge_lora(self):
        """Merge LoRA weights into base model"""
        self.transformer.merge_adapter()
        logging.info("Merged LoRA weights into base model")

    @abstractmethod
    def load_model(self, **kwargs):
        """Load and initialize model components."""
        pass

    @abstractmethod
    def cache(self, train_dataloader):
        """Pre-compute and cache embeddings/latents for training efficiency."""
        pass

    @abstractmethod
    def fit(self, train_dataloader):
        """Main training loop implementation."""
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Inference/prediction method."""
        pass

    @abstractmethod
    def set_model_devices(self, mode: str = "train"):
        """Set model device allocation based on different modes."""
        pass

    def setup_predict(self,):
        pass

    @abstractmethod
    def encode_prompt(self, *args, **kwargs):
        """Encode text prompts to embeddings."""
        pass

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
        if not self.accelerator.is_main_process:
            return
        save_path = os.path.join(
            self.config.logging.output_dir, f"checkpoint-{epoch}-{global_step}"
        )
        self.accelerator.save_state(save_path)
        with open(os.path.join(save_path, "state.json"), "w") as f:
            json.dump({"global_step": global_step, "epoch": epoch}, f)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        from diffusers.optimization import get_scheduler

        trainable_named_params = [
            (name, param)
            for name, param in self.transformer.named_parameters()
            if param.requires_grad
        ]
        lora_layers = [param for _, param in trainable_named_params]

        # Log how many parameters are in training and show one example
        if (getattr(self, "accelerator", None) is None) or self.accelerator.is_main_process:
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

    def get_model_type(self) -> str:
        """Get the model type identifier."""
        return getattr(self.config.model, "model_type", "unknown")

    def get_precision_info(self) -> Dict[str, Any]:
        """Get precision and quantization information."""
        return {
            "weight_dtype": str(self.weight_dtype),
            "mixed_precision": getattr(self.config.train, "mixed_precision", "none"),
            "quantize": getattr(self.config.model, "quantize", False),
        }

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
        cls, transformer: torch.nn.Module, config, adapter_name: str
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
        else:
            cls.add_lora_adapter(transformer, config, adapter_name)

    def log_model_info(self):
        """Log model information."""
        logger.info(f"Model Type: {self.get_model_type()}")
        logger.info(f"Precision Info: {self.get_precision_info()}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Use Cache: {self.use_cache}")

    @abstractmethod
    def save_lora(self, save_path):
        """Save LoRA weights"""
        pass
