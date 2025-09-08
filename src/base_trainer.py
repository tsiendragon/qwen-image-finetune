"""
Abstract Base Trainer for all trainer implementations.
Defines the core interface that all trainers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import torch
import os
import shutil
import json
from accelerate import Accelerator
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
            if os.path.isdir(item_path) and item.startswith('v') and item[1:].isdigit():
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
                if item.startswith("checkpoint-") and os.path.isdir(os.path.join(version_path, item)):
                    try:
                        # 从 checkpoint-{epoch}-{global_step} 中提取 global_step
                        parts = item.split('-')
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
                log_files = [f for f in files if f.startswith('events.out.tfevents')]
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
                torch.load(os.path.join(self.config.train.resume_from_checkpoint, "optimizer.bin"))
            )
            self.lr_scheduler.load_state_dict(
                torch.load(os.path.join(self.config.train.resume_from_checkpoint, "scheduler.bin"))
            )
            logging.info(f"Loaded optimizer and scheduler from {self.config.train.resume_from_checkpoint}")

        lora_layers_model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
            lora_layers_model, self.optimizer, train_dataloader, self.lr_scheduler
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Trackers already initialized in setup_accelerator
        return train_dataloader

    def configure_optimizers(self):
        from diffusers.optimization import get_scheduler

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
            num_warmup_steps=self.config.lr_scheduler.warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.config.train.max_train_steps * self.accelerator.num_processes,
        )


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

    @abstractmethod
    def encode_prompt(self, *args, **kwargs):
        """Encode text prompts to embeddings."""
        pass

    # Common methods that can be shared across implementations
    def setup_accelerator(self):
        """Initialize accelerator and logging configuration."""
        # This will be implemented by child classes with specific logic
        # but can contain common initialization code
        pass

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
        """Configure optimizer and learning rate scheduler."""
        # Common optimizer configuration can be implemented here
        pass

    def setup_versioned_logging_dir(self):
        """Set up versioned logging directory."""
        # Common logging directory setup can be implemented here
        pass

    def get_model_type(self) -> str:
        """Get the model type identifier."""
        return getattr(self.config.model, 'model_type', 'unknown')

    def get_precision_info(self) -> Dict[str, Any]:
        """Get precision and quantization information."""
        return {
            'weight_dtype': str(self.weight_dtype),
            'mixed_precision': getattr(self.config.train, 'mixed_precision', 'none'),
            'quantize': getattr(self.config.model, 'quantize', False)
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
    def add_lora_adapter(cls, transformer: torch.nnModule, config, adapter_name: str):
        from peft import LoraConfig

        lora_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            init_lora_weights=config.model.lora.init_lora_weights,
            target_modules=config.model.lora.target_modules,
        )
        transformer.add_adapter(lora_config, adapter_name=adapter_name)
        transformer.set_adapter(adapter_name)

    @classmethod
    def load_pretrain_lora_model(cls, transformer: torch.nn.Module, config, adapter_name: str):
        from src.utils.lora_utils import classify_lora_weight
        pretrained_weight = getattr(config.model.lora, 'pretrained_weight', None)
        if pretrained_weight:
            lora_type = classify_lora_weight(pretrained_weight)
            # DIFFUSERS can be loaded directly, otherwise, need to add lora first
            if lora_type != "PEFT":
                transformer.load_lora_adapter(pretrained_weight, adapter_name=adapter_name)
                logging.info(f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}")
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
                logging.info(f"set_lora: {lora_type} Loaded lora from {pretrained_weight} for {adapter_name}")
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
