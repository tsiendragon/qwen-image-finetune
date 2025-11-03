"""
Configuration module for Qwen Image Fine-tuning (Pydantic BaseModel version)
按 model、data、logging、optimizer、train 分块组织配置，支持 YAML 文件加载和验证
"""

import os
import re
from enum import Enum
from typing import Any, Self, Union

import torch
import yaml
from omegaconf import OmegaConf
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)


# ----------------------------
# Common helpers / types
# ----------------------------

DeviceLike = Union[str, torch.device]


def _normalize_cache_dir(v: str | None) -> str | None:
    if v is None:
        return v
    v = os.path.expanduser(os.path.expandvars(str(v)))
    if "://" in v:  # 保留如 s3://, http://
        scheme, rest = v.split("://", 1)
        rest = re.sub(r"/+", "/", rest)
        v = f"{scheme}://{rest}"
    else:
        v = re.sub(r"/+", "/", v)  # 将 '//' '///' 压成 '/'
    # 可选：去掉尾部斜杠（非根目录）
    if len(v) > 1:
        v = v.rstrip("/")
    return v


def _normalize_device(x: DeviceLike | None) -> torch.device | None:
    if x is None:
        return None
    d = torch.device(x)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"CUDA not available but got device={d}.")
    if d.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS not available but got device='mps'.")
    return d


class DeviceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    vae: DeviceLike | None = None
    vae_encoder: DeviceLike | None = None
    text_encoder: DeviceLike | None = None
    text_encoder_2: DeviceLike | None = None
    dit: DeviceLike | None = None
    prompt_enhancer: DeviceLike | None = None  # VLM model for prompt optimization

    # 设备规范化（字符串/torch.device -> torch.device）
    @field_validator(
        "vae",
        "vae_encoder",
        "text_encoder",
        "text_encoder_2",
        "dit",
        "prompt_enhancer",
        mode="after",
    )
    @classmethod
    def _norm(cls, v: DeviceLike | None) -> torch.device | None:
        return _normalize_device(v)

    @field_serializer(
        "vae",
        "text_encoder",
        "text_encoder_2",
        "dit",
        "vae_encoder",
        "prompt_enhancer",
        when_used="always",  # 只在 JSON 导出时生效；若想在 Python dict 也转字符串，用 "always"
    )
    def _ser_dev(self, v: DeviceLike | None):
        return None if v is None else str(v)

    # if vae is None and vae_encoder is not None, set vae to vae_encoder
    @field_validator("vae")
    @classmethod
    def _set_vae(cls, v: DeviceLike | None) -> DeviceLike | None:
        if v is None and cls.vae_encoder is not None:
            return cls.vae_encoder
        return v


# ----------------------------
# Image Processor
# ----------------------------


class ImageProcessorInitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    process_type: str = "center_crop"  # resize, _padding, center_crop
    resize_mode: str = "bilinear"
    target_size: list[int] | None = None
    controls_size: list[list[int]] | None = None  # None -> use target_size
    controls_pixels: list[int] | None = None
    target_pixels: int | None = None
    # Multi-resolution training: list of target pixel candidates
    # If present, enables multi-resolution mode automatically
    # Format 1 (simple): List of pixels - applies to all images
    #   multi_resolutions: ["1024*1024", "512*512"]
    # Format 2 (advanced): Dict with separate configs for each image type
    #   multi_resolutions:
    #     target: ["1024*1024", "512*512"]
    #     controls: [["512*512", "256*256"], ["256*256"], ["512*512"]]
    #   where controls[0] is the first control, controls[1:] are additional controls
    multi_resolutions: list[int | str] | dict[str, Any] | None = None
    max_aspect_ratio: float | None = 3.0  # Maximum aspect ratio limit
    # If true, resize control and mask to match image size before further processing
    resize_controls_mask_to_image: bool = False

    @field_validator("process_type")
    @classmethod
    def _check_process_type(cls, v: str) -> str:
        allowed = {"resize", "center_padding", "right_padding", "center_crop", "fixed_pixels"}
        if v not in allowed:
            raise ValueError(f"process_type must be one of {allowed}")
        return v

    # 解析像素表达式，例如 "512*512" -> 262144
    @staticmethod
    def _eval_pixel_expr(expr: str) -> int:
        s = str(expr).strip()
        # 仅允许 非负整数 或 形如 a*b 的简单乘法表达式，避免执行任意代码
        if re.fullmatch(r"\d+", s):
            return int(s)
        if re.fullmatch(r"\d+\s*\*\s*\d+", s):
            a, b = re.split(r"\*", s)
            return int(a.strip()) * int(b.strip())
        raise ValueError(f"Invalid pixel expression: {expr}")

    @field_validator("target_pixels", mode="before")
    @classmethod
    def _parse_target_pixels(cls, v):
        if v is None:
            return v
        if isinstance(v, (int,)):
            return int(v)
        if isinstance(v, str):
            return cls._eval_pixel_expr(v)
        raise ValueError(f"target_pixels must be int or string expression like '512*512', got {type(v)}")

    @field_validator("controls_pixels", mode="before")
    @classmethod
    def _parse_controls_pixels(cls, v: Any) -> Any:
        if v is None:
            return v
        # 允许单个整数/表达式，或列表形式
        if isinstance(v, (int,)):
            return int(v)
        if isinstance(v, str):
            return cls._eval_pixel_expr(v)
        if isinstance(v, list):
            parsed: list[int] = []
            for item in v:
                if isinstance(item, (int,)):
                    parsed.append(int(item))
                elif isinstance(item, str):
                    parsed.append(cls._eval_pixel_expr(item))
                else:
                    raise ValueError("controls_pixels list items must be int or string expression like '512*512'")
            return parsed
        raise ValueError("controls_pixels must be int, string expression, or list of them")

    @field_validator("multi_resolutions", mode="before")
    @classmethod
    def _parse_multi_resolutions(cls, v: Any) -> Any:
        """Parse multi_resolutions from various formats

        Supports two formats:
        1. Simple list: ["1024*1024", "512*512"] -> applies to all images
        2. Advanced dict: {target: [...], controls: [[...], [...], [...]]}
           where controls[0] is first control, controls[1:] are additional controls
        """
        if v is None:
            return v

        # Format 1: Simple list (backward compatible)
        if isinstance(v, list):
            parsed: list[int] = []
            for item in v:
                if isinstance(item, (int,)):
                    parsed.append(int(item))
                elif isinstance(item, str):
                    parsed.append(cls._eval_pixel_expr(item))
                else:
                    raise ValueError("multi_resolutions items must be int or string expression like '512*512'")

            if len(parsed) == 0:
                raise ValueError("multi_resolutions must not be empty")

            # Validate all values are positive
            for p in parsed:
                if p <= 0:
                    raise ValueError(f"multi_resolutions values must be positive, got {p}")

            return parsed

        # Format 2: Advanced dict with per-image-type configs
        if isinstance(v, dict):
            parsed_dict: dict[str, list[int] | list[list[int]]] = {}

            # Parse 'target' key
            if "target" in v:
                target_list: list[int] = []
                for item in v["target"]:
                    if isinstance(item, (int,)):
                        target_list.append(int(item))
                    elif isinstance(item, str):
                        target_list.append(cls._eval_pixel_expr(item))
                    else:
                        raise ValueError("target resolutions must be int or string")
                parsed_dict["target"] = target_list

            # Parse 'controls' key - list of lists where controls[0] is first control
            if "controls" in v:
                controls_lists: list[list[int]] = []
                for control_idx, control_res in enumerate(v["controls"]):
                    if not isinstance(control_res, list):
                        raise ValueError(f"controls[{control_idx}] must be a list of resolutions")
                    control_list: list[int] = []
                    for item in control_res:
                        if isinstance(item, (int,)):
                            control_list.append(int(item))
                        elif isinstance(item, str):
                            control_list.append(cls._eval_pixel_expr(item))
                        else:
                            raise ValueError(f"controls[{control_idx}] items must be int or string")
                    controls_lists.append(control_list)
                parsed_dict["controls"] = controls_lists

            # Validate at least one key is present
            if len(parsed_dict) == 0:
                raise ValueError("multi_resolutions dict must contain at least one of: 'target', 'controls'")

            # Validate all values are positive
            for key, values in parsed_dict.items():
                if key == "controls":
                    # controls is list[list[int]]
                    if not isinstance(values, list):
                        continue
                    for idx, control_vals in enumerate(values):
                        if not isinstance(control_vals, list):
                            continue
                        for p in control_vals:
                            if p <= 0:
                                raise ValueError(f"controls[{idx}] values must be positive, got {p}")
                else:
                    # target is list[int]
                    if not isinstance(values, list):
                        continue
                    for item in values:
                        if not isinstance(item, int):
                            continue
                        if item <= 0:
                            raise ValueError(f"{key} values must be positive, got {item}")

            return parsed_dict

        raise ValueError("multi_resolutions must be a list or dict, got " + str(type(v)))

    @field_validator("max_aspect_ratio")
    @classmethod
    def _check_max_aspect_ratio(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError("max_aspect_ratio must be positive")
        return v


class ImageProcessorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_path: str = "qflux.data.preprocess.ImageProcessor"
    init_args: ImageProcessorInitArgs = Field(default_factory=ImageProcessorInitArgs)


# ----------------------------
# Predict
# ----------------------------


class PredictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    devices: DeviceConfig = Field(default_factory=DeviceConfig)


# ----------------------------
# LoRA / Model
# ----------------------------


class LoraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    r: int = 16
    lora_alpha: int = 16
    init_lora_weights: str = "gaussian"  # 'gaussian' | 'normal' | 'zero'
    target_modules: str | list[str] = Field(default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"])
    pretrained_weight: str | None = None
    adapter_name: str = "default"

    @field_validator("r")
    @classmethod
    def _check_r(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("r must be > 0")
        return v

    @field_validator("lora_alpha")
    @classmethod
    def _check_alpha(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("lora_alpha must be > 0")
        return v

    @field_validator("init_lora_weights")
    @classmethod
    def _check_init(cls, v: str) -> str:
        allowed = {"gaussian", "normal", "zero"}
        if v not in allowed:
            raise ValueError(f"init_lora_weights must be in {allowed}")
        return v

    @field_validator("target_modules")
    @classmethod
    def _check_target_modules(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str) and not v:
            raise ValueError("target_modules must be non-empty")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("target_modules must be non-empty")
        return v

    # @field_validator("pretrained_weight")
    # @classmethod
    # def _check_weight_path(cls, v: str | None) -> str | None:
    #     if v is not None and not os.path.exists(v):
    #         raise ValueError(f"pretrained_weight path not found: {v}")
    #     return v

    @field_validator("adapter_name")
    @classmethod
    def _check_adapter_name(cls, v: str | None) -> str | None:
        if v is not None and not isinstance(v, str):
            raise ValueError("adapter_name must be a string")
        return v


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pretrained_model_name_or_path: str = "Qwen/Qwen-Image-Edit"
    pretrained_embeddings: dict | None = None  # if want to load different embeeding model vs main model (dit)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    quantize: bool = False
    use_vlm_prompt_enhancer: bool = False  # Enable VLM-based prompt enhancement (DreamOmni2 trainer)


# ----------------------------
# Data
# ----------------------------


class DatasetInitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_path: str | list | None = None
    caption_dropout_rate: float = 0.0
    prompt_image_dropout_rate: float = 0.0
    cache_dir: str | None = None
    use_cache: bool = True
    use_edit_mask: bool = False
    selected_control_indexes: list[int] | None = None
    prompt_empty_drop_keys: list[str] | None = None
    processor: ImageProcessorConfig = Field(default_factory=ImageProcessorConfig)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_path: str = "torch.utils.data.Dataset"
    init_args: DatasetInitArgs = Field(default_factory=DatasetInitArgs)
    batch_size: int = 1
    num_workers: int = 1
    shuffle: bool = True

    @field_validator("class_path")
    @classmethod
    def _check_class_path(cls, v: str) -> str:
        if not v:
            raise ValueError("class_path must be non-empty")
        return v

    @field_validator("batch_size", "num_workers")
    @classmethod
    def _check_pos_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive integer")
        return v


# ----------------------------
# Logging / Sampling
# ----------------------------


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enable: bool = False
    validation_steps: int = 100
    num_samples: int = 4
    seed: int = 42
    validation_data: str | list[dict[str, str]] | None = None

    @model_validator(mode="after")
    def _check_when_enabled(self) -> Self:
        if self.enable:
            if self.validation_steps <= 0:
                raise ValueError("validation_steps must be positive when sampling is enabled")
            if self.num_samples <= 0:
                raise ValueError("num_samples must be positive when sampling is enabled")
        return self

    @field_validator("validation_data")
    @classmethod
    def _check_validation_data(cls, v: str | list[dict[str, str]] | None) -> str | list[dict[str, str]] | None:
        if v is None:
            return v
        if isinstance(v, list):
            for item in v:
                if not isinstance(item, dict) or "control" not in item or "prompt" not in item:
                    raise ValueError("Each validation_data item must be a dict with 'control' and 'prompt' keys")
        return v


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_dir: str = "./output"
    report_to: str = "tensorboard"  # tensorboard, wandb, swanlab, none
    tracker_project_name: str | None = None  # will get the value from trainer
    tags: list[str] | None = None
    notes: str | None = None

    @field_validator("report_to")
    @classmethod
    def _check_report_to(cls, v: str) -> str:
        allowed = {"tensorboard", "wandb", "swanlab", "none"}
        if v not in allowed:
            raise ValueError(f"report_to must be one of {allowed}")
        return v

    @field_validator("output_dir")
    @classmethod
    def _check_output_dir(cls, v: str) -> str | None:
        return _normalize_cache_dir(v)

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, v: list[str] | None) -> list[str] | None:
        if v is not None and not isinstance(v, list):
            raise ValueError("tags must be a list of strings")
        return v

    @field_validator("notes")
    @classmethod
    def _check_notes(cls, v: str | None) -> str | None:
        if v is not None and not isinstance(v, str):
            raise ValueError("notes must be a string")
        return v


# ----------------------------
# LR Scheduler
# ----------------------------


class LRSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scheduler_type: str = "constant"
    warmup_steps: int = 0
    num_cycles: float = 0.5
    power: float = 1.0

    @field_validator("scheduler_type")
    @classmethod
    def _check_type(cls, v: str) -> str:
        valid = {
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        }
        if v not in valid:
            raise ValueError(f"scheduler_type must be one of {valid}")
        return v

    @field_validator("warmup_steps")
    @classmethod
    def _check_warmup(cls, v: int) -> int:
        if v < 0:
            raise ValueError("warmup_steps must be >= 0")
        return v

    @field_validator("num_cycles", "power")
    @classmethod
    def _check_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("must be positive")
        return v


# ----------------------------
# Optimizer
# ----------------------------


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    class_path: str = "torch.optim.AdamW"
    init_args: dict[str, Any] = Field(
        default_factory=lambda: {
            "lr": 1e-4,
            "betas": [0.9, 0.999],
        }
    )

    @field_validator("class_path")
    @classmethod
    def _check_class_path(cls, v: str) -> str:
        if not v:
            raise ValueError("class_path must be non-empty")
        return v

    @field_validator("init_args")
    @classmethod
    def _check_init_args(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "lr" in v:
            lr = v["lr"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                raise ValueError(f"init_args.lr must be positive, got {lr}")
        if "weight_decay" in v:
            wd = v["weight_decay"]
            if not isinstance(wd, (int, float)) or wd < 0:
                raise ValueError(f"init_args.weight_decay must be >= 0, got {wd}")
        if "betas" in v:
            betas = v["betas"]
            if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                raise ValueError("init_args.betas must be a list/tuple of length 2")
            if not all(isinstance(b, (int, float)) and 0 <= b < 1 for b in betas):
                raise ValueError("each beta must be in [0, 1)")
        return v


# ----------------------------
# Device / Cache
# ----------------------------


class CacheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    devices: DeviceConfig = Field(default_factory=DeviceConfig)
    use_cache: bool = True
    cache_dir: str = "./cache/"
    prompt_empty_drop_keys: list[str] = Field(default_factory=lambda: ["prompt_embed", "prompt_embeds_mask"])

    # 规范化：展开 ~ / 环境变量，压缩多余斜杠，保留 scheme://
    @field_validator("cache_dir", mode="before")
    @classmethod
    def format_dir(cls, v: str) -> str | None:
        return _normalize_cache_dir(v)

    @field_validator("cache_dir")
    @classmethod
    def _check_cache_dir(cls, v: str) -> str | None:
        if not v:
            raise ValueError("cache_dir must be non-empty")
        return v


# ----------------------------
# Train / Loss
# ----------------------------
class TrainerKind(str, Enum):
    QwenImageEdit = "QwenImageEdit"
    QwenImageEditPlus = "QwenImageEditPlus"
    FluxKontext = "FluxKontext"
    DreamOmni2 = "DreamOmni2"


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 1000
    num_epochs: int = 3
    checkpointing_steps: int = 500
    checkpoints_total_limit: int | None = None
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"  # fp16 | bf16 | no
    gradient_checkpointing: bool = True
    low_memory: bool = False
    # 指定精细设备布局（低显存模式时生效）
    fit_device: DeviceConfig | None = None

    @field_validator(
        "train_batch_size",
        "gradient_accumulation_steps",
        "max_train_steps",
        "num_epochs",
        "checkpointing_steps",
    )
    @classmethod
    def _pos_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be a positive integer")
        return v

    @field_validator("checkpoints_total_limit")
    @classmethod
    def _check_total_limit(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ValueError("checkpoints_total_limit must be positive or None")
        return v

    @field_validator("max_grad_norm")
    @classmethod
    def _check_grad_norm(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("max_grad_norm must be positive")
        return v

    @field_validator("mixed_precision")
    @classmethod
    def _check_mp(cls, v: str) -> str:
        allowed = {"fp16", "bf16", "no"}
        if v not in allowed:
            raise ValueError(f"mixed_precision must be one of {allowed}")
        return v


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Legacy fields for backward compatibility
    mask_loss: bool = False
    forground_weight: float = 2.0
    background_weight: float = 1.0

    # New flexible configuration (optional)
    class_path: str | None = None
    init_args: dict[str, Any] | None = None

    @field_validator("forground_weight", "background_weight")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("weight must be >= 0")
        return v

    @field_validator("class_path")
    @classmethod
    def _check_class_path(cls, v: str | None) -> str | None:
        if v is not None and not v:
            raise ValueError("class_path must be non-empty if provided")
        return v


# ----------------------------
# Validation Config
# ----------------------------


class ValidationSample(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str
    images: list[str]
    controls_size: list[list[int]] | None = None
    height: int | None = None
    width: int | None = None
    negative_prompt: str | None = None
    guidance_scale: float | None = None
    num_inference_steps: int | None = 20


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    steps: int = 100
    max_samples: int = 4
    seed: int = 42
    dataset: DataConfig | None = None
    samples: list[ValidationSample] | None = None

    @model_validator(mode="after")
    def _check_when_enabled(self) -> Self:
        if self.enabled:
            if self.steps <= 0:
                raise ValueError("steps must be positive when validation is enabled")
            if self.max_samples <= 0:
                raise ValueError("max_samples must be positive when validation is enabled")
            if self.dataset is None and self.samples is None:
                raise ValueError("either dataset or samples must be provided when validation is enabled")

            # 验证samples的格式
            if self.samples is not None:
                for i, sample in enumerate(self.samples):
                    if not sample.images:
                        raise ValueError(f"Sample {i} must have at least one image")
                    if not sample.prompt:
                        raise ValueError(f"Sample {i} must have a prompt")

                    # 如果没有指定controls_size，确保在运行时计算
                    if sample.controls_size is not None and len(sample.controls_size) != len(sample.images):
                        raise ValueError(
                            f"Sample {i} has {len(sample.images)} images but {len(sample.controls_size)} control sizes"
                        )
        return self


# ----------------------------
# Root Config
# ----------------------------
class TrMode(str, Enum):
    cache = "cache"
    fit = "fit"
    predict = "predict"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trainer: TrainerKind = TrainerKind.QwenImageEdit
    resume: str | None = None
    mode: TrMode = TrMode.predict
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = Field(default_factory=LRSchedulerConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    predict: PredictConfig = Field(default_factory=PredictConfig)
    loss: LossConfig = Field(default_factory=LossConfig)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def trainer_type(self) -> str:
        return self.trainer

    @computed_field  # type: ignore[prop-decorator]
    @property
    def use_cache(self) -> bool:
        return self.cache.use_cache

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cache_dir(self) -> str:
        return self.cache.cache_dir

    @computed_field  # type: ignore[prop-decorator]
    @property
    def model_name(self) -> str:
        return self.model.pretrained_model_name_or_path

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lora_adapter_name(self) -> str:
        return self.model.lora.adapter_name

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lora_r(self) -> int:
        return self.model.lora.r

    @computed_field  # type: ignore[prop-decorator]
    @property
    def lora_lora_alpha(self) -> int:
        return self.model.lora.lora_alpha

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target_size(self) -> list[int] | None:
        return self.data.init_args.processor.init_args.target_size

    @computed_field  # type: ignore[prop-decorator]
    @property
    def caption_dropout_rate(self) -> float:
        return self.data.init_args.caption_dropout_rate

    # 1) 纯计算，不读 computed 字段，不改状态
    def _compute_quantization_type(self) -> str:
        name = (self.model_name or "").lower()
        if "fp4" in name or "4bit" in name:
            return "pretrain_fp4"
        if "fp8" in name:  # 注意不要把 "8bit/int8" 当作 fp8
            return "pretrain_fp8"
        if bool(getattr(self.model, "quantize", False)):
            return "fp8_online"
        return "pretrain_fp16"

    # 2) computed_field 只读 —— 任何时候访问都不会改状态
    @computed_field  # type: ignore[prop-decorator]
    @property
    def quantization_type(self) -> str:
        return self._compute_quantization_type()

    @model_validator(mode="after")
    def _wire_cross_defaults(self) -> "Config":
        self.data.init_args.cache_dir = self.cache.cache_dir
        self.data.init_args.use_cache = self.cache.use_cache
        self.data.init_args.prompt_empty_drop_keys = self.cache.prompt_empty_drop_keys
        self.train.train_batch_size = self.data.batch_size
        if self.quantization_type in {"pretrain_fp4", "pretrain_fp8", "pretrain_fp16"}:
            self.model.quantize = False
        return self


# ----------------------------
# I/O helpers
# ----------------------------


def load_config_from_yaml(yaml_path: str) -> Config:
    """
    从 YAML 文件加载配置并验证
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    try:
        yaml_config = OmegaConf.load(yaml_path)
    except Exception as e:
        raise Exception(f"Error loading YAML file {yaml_path}: {e}")

    data = OmegaConf.to_container(yaml_config, resolve=True) or {}

    # 直接一次性校验嵌套结构
    # 若希望保持你之前对 lora/rank 的手动对齐，这里已在 ModelConfig 的 model_validator 中处理
    config = Config.model_validate(data)
    return config


if __name__ == "__main__":
    # config = load_config_from_yaml("configs/example_fluxkontext_fp16.yaml")
    config_file = "tests/test_configs/test_example_qwen_image_edit_plus_fp4_dynamic_shapes.yaml"
    config = load_config_from_yaml(config_file)
    print(config)
    x = config.model_dump_json(indent=2, exclude_none=True)
    print("type of x", type(x))
    print(config.model_dump_json(indent=2, exclude_none=True))
    d_json = config.model_dump(mode="json", exclude_none=True)
    print(d_json, type(d_json))
    with open("test_config.yaml", "w") as f:
        yaml.dump(d_json, f, default_flow_style=False, sort_keys=False)
