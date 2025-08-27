"""
Configuration module for Qwen Image Fine-tuning
按 model、data、logging、optimizer、train 分块组织配置，支持 YAML 文件加载和验证
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import yaml


@dataclass
class PredictConfig:
    """预测相关配置"""

    devices: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """验证预测配置"""

        if not isinstance(self.devices, dict):
            raise ValueError(f"devices must be a dictionary, got {type(self.devices)}")

        for device_type, device_id in self.devices.items():
            if not device_id.startswith("cuda:"):
                raise ValueError(f"device_id must start with 'cuda:', got {device_id}")


@dataclass
class LoraConfig:
    """LoRA 相关配置"""

    r: int = 16  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha
    init_lora_weights: str = "gaussian"  # 初始化方式
    target_modules: List[str] = field(
        default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"]
    )
    pretrained_weight: Optional[str] = None

    def __post_init__(self):
        """验证 LoRA 配置"""
        if not isinstance(self.r, int) or self.r <= 0:
            raise ValueError(f"r must be a positive integer, got {self.r}")

        if not isinstance(self.lora_alpha, int) or self.lora_alpha <= 0:
            raise ValueError(
                f"lora_alpha must be a positive integer, got {self.lora_alpha}"
            )

        if self.init_lora_weights not in ["gaussian", "normal", "zero"]:
            raise ValueError(
                f"init_lora_weights must be one of ['gaussian', 'normal', 'zero'], got {self.init_lora_weights}"
            )

        if not isinstance(self.target_modules, list) or not self.target_modules:
            raise ValueError(
                f"target_modules must be a non-empty list, got {self.target_modules}"
            )

        if self.pretrained_weight is not None:
            if not isinstance(self.pretrained_weight, str) or not os.path.exists(
                self.pretrained_weight
            ):
                raise ValueError(
                    f"pretrained_weight must be a valid file path, got {self.pretrained_weight}"
                )


@dataclass
class ModelConfig:
    """模型相关配置"""

    pretrained_model_name_or_path: str = "Qwen/Qwen-Image-Edit"
    rank: int = 16  # LoRA rank (为了兼容性保留)
    lora: LoraConfig = field(default_factory=LoraConfig)
    quantize: bool = False

    def __post_init__(self):
        """验证模型配置"""
        if not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError(f"rank must be a positive integer, got {self.rank}")
        if not isinstance(self.quantize, bool):
            raise ValueError(f"quantize must be a boolean, got {self.quantize}")

        # 确保 lora.r 与 rank 保持一致
        if self.lora.r != self.rank:
            self.lora.r = self.rank
            self.lora.lora_alpha = self.rank


@dataclass
class DataConfig:
    """数据相关配置"""

    class_path: str = "torch.utils.data.Dataset"
    init_args: Dict[str, Any] = field(default_factory=dict)
    batch_size: int = 1
    num_workers: int = 1
    shuffle: bool = True

    def __post_init__(self):
        """验证数据配置"""
        # 验证 class_path
        if not isinstance(self.class_path, str) or not self.class_path:
            raise ValueError(
                f"class_path must be a non-empty string, got {self.class_path}"
            )

        # 验证 init_args
        if not isinstance(self.init_args, dict):
            raise ValueError(
                f"init_args must be a dictionary, got {type(self.init_args)}"
            )


@dataclass
class LoggingConfig:
    """日志和输出相关配置"""

    output_dir: str = "./output"
    logging_dir: str = "logs"
    report_to: str = "tensorboard"  # tensorboard, wandb, all, none
    tracker_project_name: str = "qwen-image-finetune"

    def __post_init__(self):
        """验证日志配置"""
        if self.report_to not in ["tensorboard", "wandb", "all", "none"]:
            raise ValueError(
                f"report_to must be one of ['tensorboard', 'wandb', 'all', 'none'], got {self.report_to}"
            )


@dataclass
class LRSchedulerConfig:
    """学习率调度器相关配置，适配 get_scheduler 函数"""

    scheduler_type: str = "constant"  # 调度器类型名称，传给 get_scheduler 的第一个参数
    warmup_steps: int = 0  # 预热步数
    num_cycles: float = 0.5  # cosine_with_restarts 的循环数
    power: float = 1.0  # polynomial 的幂次

    def __post_init__(self):
        """验证学习率调度器配置"""
        # 验证调度器类型
        valid_schedulers = [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ]
        if self.scheduler_type not in valid_schedulers:
            raise ValueError(
                f"scheduler_type must be one of {valid_schedulers}, got {self.scheduler_type}"
            )

        # 验证预热步数
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be a non-negative integer, got {self.warmup_steps}"
            )

        # 验证 num_cycles
        if not isinstance(self.num_cycles, (int, float)) or self.num_cycles <= 0:
            raise ValueError(
                f"num_cycles must be a positive number, got {self.num_cycles}"
            )

        # 验证 power
        if not isinstance(self.power, (int, float)) or self.power <= 0:
            raise ValueError(f"power must be a positive number, got {self.power}")


@dataclass
class OptimizerConfig:
    """优化器相关配置"""

    class_path: str = "torch.optim.AdamW"
    init_args: Dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-4,
            # "weight_decay": 1e-2,
            "betas": [0.9, 0.999],
            # "eps": 1e-8
        }
    )

    def __post_init__(self):
        """验证优化器配置"""
        # 验证 class_path
        if not isinstance(self.class_path, str) or not self.class_path:
            raise ValueError(
                f"class_path must be a non-empty string, got {self.class_path}"
            )

        # 验证 init_args
        if not isinstance(self.init_args, dict):
            raise ValueError(
                f"init_args must be a dictionary, got {type(self.init_args)}"
            )

        # 验证学习率（如果存在）
        if "lr" in self.init_args:
            lr = self.init_args["lr"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError(f"lr in init_args must be a positive number, got {lr}")

        # 验证权重衰减（如果存在）
        if "weight_decay" in self.init_args:
            wd = self.init_args["weight_decay"]
            if not isinstance(wd, (int, float)) or wd < 0:
                raise ValueError(
                    f"weight_decay in init_args must be non-negative, got {wd}"
                )

        # 验证 betas（如果存在）
        if "betas" in self.init_args:
            betas = self.init_args["betas"]
            if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                raise ValueError(
                    f"betas in init_args must be a list/tuple of 2 elements, got {betas}"
                )
            if not all(isinstance(b, (int, float)) and 0 <= b < 1 for b in betas):
                raise ValueError(f"betas values must be in [0, 1), got {betas}")


@dataclass
class CacheConfig:
    """缓存相关配置"""

    vae_encoder_device: Optional[str] = None  # VAE 编码器设备 ID
    text_encoder_device: Optional[str] = None  # 文本编码器设备 ID
    use_cache: bool = True
    cache_dir: str = "/data/lilong/experiment/id_card_qwen_image_lora/cache"

    def __post_init__(self):
        """验证缓存配置"""
        # 验证设备 ID
        if self.vae_encoder_device is not None:
            if not isinstance(self.vae_encoder_device, str):
                raise ValueError(
                    f"vae_encoder_device must be a non-negative string or None, got {self.vae_encoder_device}"
                )

        if self.text_encoder_device is not None:
            if not isinstance(self.text_encoder_device, str):
                raise ValueError(
                    f"text_encoder_device must be a non-negative string or None, got {self.text_encoder_device}"
                )

        if not isinstance(self.use_cache, bool):
            raise ValueError(f"use_cache must be a boolean, got {self.use_cache}")

        if not isinstance(self.cache_dir, str) or not self.cache_dir:
            raise ValueError(
                f"cache_dir must be a non-empty string, got {self.cache_dir}"
            )


@dataclass
class TrainConfig:
    """训练相关配置"""

    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1000
    num_epochs: int = 3
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"  # fp16, bf16, no
    gradient_checkpointing: bool = True  # 启用梯度检查点以节省显存

    def __post_init__(self):
        """验证训练配置"""
        # 验证正整数参数
        int_params = {
            "train_batch_size": self.train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_train_steps": self.max_train_steps,
            "num_epochs": self.num_epochs,
            "checkpointing_steps": self.checkpointing_steps,
        }

        for param_name, param_value in int_params.items():
            if not isinstance(param_value, int) or param_value <= 0:
                raise ValueError(
                    f"{param_name} must be a positive integer, got {param_value}"
                )

        # 验证 checkpoints_total_limit
        if self.checkpoints_total_limit is not None:
            if (
                not isinstance(self.checkpoints_total_limit, int)
                or self.checkpoints_total_limit <= 0
            ):
                raise ValueError(
                    f"checkpoints_total_limit must be a positive integer or None, got {self.checkpoints_total_limit}"
                )

        # 验证梯度裁剪
        if not isinstance(self.max_grad_norm, (int, float)) or self.max_grad_norm <= 0:
            raise ValueError(
                f"max_grad_norm must be positive, got {self.max_grad_norm}"
            )

        # 验证混合精度
        if self.mixed_precision not in ["fp16", "bf16", "no"]:
            raise ValueError(
                f"mixed_precision must be one of ['fp16', 'bf16', 'no'], got {self.mixed_precision}"
            )

        # 验证梯度检查点
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError(
                f"gradient_checkpointing must be a boolean, got {self.gradient_checkpointing}"
            )


@dataclass
class Config:
    """完整配置类"""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)

    def to_flat_dict(self) -> Dict[str, Any]:
        """将配置转换为扁平字典格式，与 train.py 中的 args 兼容"""
        flat_config = {}

        # Model 配置
        flat_config.update(
            {
                "pretrained_model_name_or_path": self.model.pretrained_model_name_or_path,
                "rank": self.model.rank,
                "lora_r": self.model.lora.r,
                "lora_alpha": self.model.lora.lora_alpha,
                "lora_init_weights": self.model.lora.init_lora_weights,
                "lora_target_modules": self.model.lora.target_modules,
            }
        )

        # Data 配置
        flat_config.update(
            {
                "data_class_path": self.data.class_path,
                "data_init_args": self.data.init_args,
            }
        )

        # Logging 配置
        flat_config.update(
            {
                "output_dir": self.logging.output_dir,
                "logging_dir": self.logging.logging_dir,
                "report_to": self.logging.report_to,
                "tracker_project_name": self.logging.tracker_project_name,
            }
        )

        # Optimizer 配置
        flat_config.update(
            {
                "optimizer_class_path": self.optimizer.class_path,
                "optimizer_init_args": self.optimizer.init_args,
            }
        )

        # LR Scheduler 配置
        flat_config.update(
            {
                "lr_scheduler": self.lr_scheduler.scheduler_type,
                "lr_warmup_steps": self.lr_scheduler.warmup_steps,
                "lr_num_cycles": self.lr_scheduler.num_cycles,
                "lr_power": self.lr_scheduler.power,
            }
        )

        # Train 配置
        flat_config.update(
            {
                "train_batch_size": self.train.train_batch_size,
                "gradient_accumulation_steps": self.train.gradient_accumulation_steps,
                "max_train_steps": self.train.max_train_steps,
                "num_epochs": self.train.num_epochs,
                "checkpointing_steps": self.train.checkpointing_steps,
                "checkpoints_total_limit": self.train.checkpoints_total_limit,
                "max_grad_norm": self.train.max_grad_norm,
                "mixed_precision": self.train.mixed_precision,
                "gradient_checkpointing": self.train.gradient_checkpointing,
            }
        )

        # Cache 配置
        flat_config.update(
            {
                "vae_encoder_device": self.cache.vae_encoder_device,
                "text_encoder_device": self.cache.text_encoder_device,
            }
        )

        return flat_config


def load_config_from_yaml(yaml_path: str) -> Config:
    """
    从 YAML 文件加载配置

    Args:
        yaml_path: YAML 配置文件路径

    Returns:
        Config: 验证后的配置对象

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置参数类型错误或值无效
        Exception: OmegaConf 加载错误
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    try:
        # 使用 OmegaConf.load 直接加载 YAML 文件
        yaml_config = OmegaConf.load(yaml_path)
    except Exception as e:
        raise Exception(f"Error loading YAML file {yaml_path}: {e}")

    # 转换为普通字典
    config_dict = OmegaConf.to_container(yaml_config, resolve=True)

    # 处理空配置文件的情况
    if config_dict is None:
        config_dict = {}

    # 创建配置对象，使用 dataclass 默认值处理缺失的配置块
    model_dict = config_dict.get("model", {})
    # 处理 lora 子配置
    if "lora" in model_dict:
        lora_config = LoraConfig(**model_dict.pop("lora"))
        model_config = ModelConfig(lora=lora_config, **model_dict)
    else:
        model_config = ModelConfig(**model_dict)

    config = Config(
        model=model_config,
        data=DataConfig(**config_dict.get("data", {})),
        logging=LoggingConfig(**config_dict.get("logging", {})),
        optimizer=OptimizerConfig(**config_dict.get("optimizer", {})),
        lr_scheduler=LRSchedulerConfig(**config_dict.get("lr_scheduler", {})),
        train=TrainConfig(**config_dict.get("train", {})),
        cache=CacheConfig(**config_dict.get("cache", {})),
        predict=PredictConfig(**config_dict.get("predict", {})),
    )

    return config


def create_sample_config(output_path: str = "config_sample.yaml") -> None:
    """
    创建示例配置文件

    Args:
        output_path: 输出配置文件路径
    """
    sample_config = {
        "model": {
            "pretrained_model_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
            "rank": 16,
            "lora": {
                "r": 16,
                "lora_alpha": 16,
                "init_lora_weights": "gaussian",
                "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            },
        },
        "data": {
            "class_path": "src.data.dataset.QwenImageDataset",
            "init_args": {
                "dataset_path": "/path/to/dataset",
                "batch_size": 4,
                "num_workers": 4,
            },
        },
        "logging": {
            "output_dir": "./output",
            "logging_dir": "logs",
            "report_to": "tensorboard",
            "tracker_project_name": "qwen-image-finetune",
        },
        "optimizer": {
            "class_path": "torch.optim.AdamW",
            "init_args": {
                "lr": 0.0001,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
        "lr_scheduler": {
            "scheduler_type": "constant",
            "warmup_steps": 0,
            "num_cycles": 0.5,
            "power": 1.0,
        },
        "train": {
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_train_steps": 1000,
            "num_epochs": 3,
            "checkpointing_steps": 500,
            "checkpoints_total_limit": 3,
            "max_grad_norm": 1.0,
            "mixed_precision": "bf16",
        },
        "cache": {"vae_encoder_device": 1, "text_encoder_device": 2},
        "predict": {
            "devices": {
                "vae": "cuda:1",
                "text_encoder": "cuda:2",
                "transformer": "cuda:3",
            }
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            sample_config, f, default_flow_style=False, allow_unicode=True, indent=2
        )

    print(f"Sample configuration file created at: {output_path}")
