# Configuration Guide

This guide documents the up-to-date configuration for Qwen Image Finetune. It reflects the current Pydantic-based schema in `qflux.data.config` and the trainer behavior.

## Configuration File Structure

The framework uses YAML configuration files. Here's the complete structure aligned with current code:

```yaml
# Trainer / Mode / Resume
trainer: QwenImageEdit              # QwenImageEdit | QwenImageEditPlus | FluxKontext
resume: null                        # path to checkpoint or null
mode: fit                           # fit | cache | predict (usually inferred from CLI)

# Model Configuration
model:
  pretrained_model_name_or_path: "Qwen/Qwen-Image-Edit"
  quantize: false                   # true -> runtime FP8 quantization (bnb)
  pretrained_embeddings: null       # optional different embedding models
  lora:
    r: 16
    lora_alpha: 16
    init_lora_weights: gaussian     # gaussian | normal | zero
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
    pretrained_weight: null         # path to LoRA weights (optional)
    adapter_name: default

# Data Configuration
data:
  class_path: "qflux.data.dataset.ImageDataset"
  init_args:
    dataset_path: "/path/to/dataset_or.csv"   # str | [str] | {repo_id, split}
    caption_dropout_rate: 0.0
    prompt_image_dropout_rate: 0.0
    cache_dir: ${cache.cache_dir}
    use_cache: ${cache.use_cache}
    use_edit_mask: false
    selected_control_indexes: null             # e.g. [1,3]
    prompt_empty_drop_keys: ${cache.prompt_empty_drop_keys}
    processor:
      class_path: "qflux.data.preprocess.ImageProcessor"
      init_args:
        process_type: center_crop              # resize | center_padding | right_padding | center_crop | fixed_pixels
        resize_mode: bilinear                  # nearest|linear|bilinear|bicubic|lanczos|area
        target_size: [576, 1024]              # [H, W]; divisible by 16 after normalization
        controls_size: [[576, 1024], [213, 192]]
        target_pixels: null                   # e.g. "768*768" or 589824; normalized to divisible-by-16 area
        controls_pixels: null                 # int | str | [int|str]
  batch_size: 1
  num_workers: 1
  shuffle: true

# Logging & Sampling
logging:
  output_dir: "./output"
  report_to: tensorboard                  # tensorboard | wandb | all | none
  tracker_project_name: null
  sampling:
    enable: false
    validation_steps: 100
    num_samples: 4
    seed: 42
    validation_data: null                 # optional list of {control, prompt}

# Optimizer Configuration
optimizer:
  class_path: torch.optim.AdamW          # torch.optim.AdamW | bitsandbytes.optim.Adam8bit | prodigyopt.Prodigy | torch.optim.SGD
  init_args:
    lr: 1.0e-4
    betas: [0.9, 0.999]
    # weight_decay: 0.01                 # AdamW/SGD only
    # eps: 1.0e-8                        # AdamW only
    # momentum: 0.9                      # SGD only
    # use_bias_correction: true          # Prodigy only
    # safeguard_warmup: true             # Prodigy only

# Learning Rate Scheduler
lr_scheduler:
  scheduler_type: constant               # linear | cosine | cosine_with_restarts | polynomial | constant | constant_with_warmup
  warmup_steps: 0
  num_cycles: 0.5
  power: 1.0

# Training Configuration
train:
  train_batch_size: ${data.batch_size}
  gradient_accumulation_steps: 4
  max_train_steps: 1000
  num_epochs: 3
  checkpointing_steps: 500
  checkpoints_total_limit: null
  max_grad_norm: 1.0
  mixed_precision: bf16                 # fp16 | bf16 | no
  gradient_checkpointing: true
  low_memory: false
  fit_device: null                      # per-module device mapping when low_memory=true

# Cache Configuration
cache:
  devices:
    vae: null
    vae_encoder: null
    text_encoder: null
    text_encoder_2: null
    dit: null
  use_cache: true
  cache_dir: "./cache"
  # When using caption dropout to empty, recommend replacing with cached empty-embeds
  prompt_empty_drop_keys: ["empty_prompt_embeds", "empty_prompt_embeds_mask"]

# Predict Configuration
predict:
  devices:
    vae: null
    vae_encoder: null
    text_encoder: null
    text_encoder_2: null
    dit: null

# Loss (task-specific)
loss:
  mask_loss: false
  forground_weight: 2.0
  background_weight: 1.0
```

## Field Options and Valid Values

### trainer
- Allowed: `QwenImageEdit`, `QwenImageEditPlus`, `FluxKontext`

### mode
- Allowed: `fit`, `cache`, `predict`（通常通过命令行 `--cache` 切换；默认运行训练会设置为 `fit`）

### model
- `pretrained_model_name_or_path`: 任意可被相应 Trainer 加载的权重名称或本地路径，名称中包含 `4bit`/`fp4`/`fp8` 会自动识别量化类型
- `quantize`: `true|false`，为 `true` 时启用运行时 FP8 量化（bnb 引擎），若使用预量化模型则会被强制设为 `false`
- `pretrained_embeddings`: 可选，单独指定与主体模型不同的嵌入模型
- `lora.init_lora_weights`: `gaussian | normal | zero`
- `lora.target_modules`: `str | [str]`，如 `["to_q","to_k","to_v","to_out.0"]`
- `lora.pretrained_weight`: 现有 LoRA 权重路径（可选）
- `lora.adapter_name`: 适配器名称（如需多适配器场景）

### data
- `class_path`: 例如 `qflux.data.dataset.ImageDataset`（可替换为你实现的 Dataset）
- `init_args.dataset_path` 支持：
  - 本地目录，形如包含 `training_images/` 与 `control_images/`
  - CSV 文件路径，包含列：`path_target`, `path_control_*`, `prompt`, 可选 `path_mask`
  - Hugging Face 仓库：`{repo_id: "org/name", split: "train"}`
- `use_edit_mask`: `true|false`，数据集中若存在 mask 会返回并参与预处理
- `selected_control_indexes`: 选择多控制图中的子集（1-based）
- `processor.init_args.process_type`：`resize | center_padding | right_padding | center_crop | fixed_pixels`
- `processor.init_args.resize_mode`：`nearest|linear|bilinear|bicubic|lanczos|area`
- `target_size / controls_size`：形如 `[H,W]`，会被规范到可被 16 整除
- `target_pixels / controls_pixels`：`int` 或表达式字符串如 `"768*768"`，会被规范到 16×16 的可分解面积，并按比例选择最合适的 `(H,W)`

### logging
- `report_to`: `tensorboard | wandb | all | none`
- `sampling.enable`: `true|false`；为 `true` 时需确保 `validation_steps > 0` 且 `num_samples > 0`

### optimizer
- 通过 `class_path` 指定优化器实现，当前已验证可用：
  - `torch.optim.AdamW`
  - `bitsandbytes.optim.Adam8bit`
  - `prodigyopt.Prodigy`
  - `torch.optim.SGD`
- 对应 `init_args` 示例：

```yaml
# AdamW（默认）
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1e-4
    betas: [0.9, 0.999]
    weight_decay: 0.01
    eps: 1e-8
```

```yaml
# 8-bit Adam（显存友好）
optimizer:
  class_path: bitsandbytes.optim.Adam8bit
  init_args:
    lr: 1.0e-4
    betas: [0.9, 0.999]
```

```yaml
# Prodigy（自适应学习率）
optimizer:
  class_path: prodigyopt.Prodigy
  init_args:
    lr: 1.0
    betas: [0.9, 0.999]
    use_bias_correction: true
    safeguard_warmup: true
```

```yaml
# SGD（经典）
optimizer:
  class_path: torch.optim.SGD
  init_args:
    lr: 0.01
    momentum: 0.9
    weight_decay: 1e-4
```

校验规则：`lr>0`；`weight_decay>=0`；`betas` 长度为 2 且每个在 `[0,1)`。

### lr_scheduler
- `scheduler_type`: `linear | cosine | cosine_with_restarts | polynomial | constant | constant_with_warmup`
- 其他字段：`warmup_steps>=0`，`num_cycles>0`，`power>0`

### train
- `mixed_precision`: `fp16 | bf16 | no`
- `low_memory`: `true|false`；为 `true` 时可通过 `fit_device` 为 `vae/text_encoder/dit` 指定设备
- `checkpoints_total_limit`: `null` 或 正整数

### cache
- `devices`：可为 `vae | vae_encoder | text_encoder | text_encoder_2 | dit` 分别指定设备，形如 `cuda:0`
- `use_cache`: `true|false`
- `cache_dir`: 任意可写目录，路径会被标准化
- `prompt_empty_drop_keys`: 用于空提示替换时丢弃的键

### predict
- `devices`：与 `cache.devices` 字段一致

### loss
- `mask_loss`: `true|false`
- `forground_weight` 与 `background_weight`: 非负数

## Notes
- 当 `pretrained_model_name_or_path` 包含 `fp4/4bit/fp8` 关键词时，会自动推断量化类型，并将 `model.quantize` 设为 `false`（避免重复量化）。
- CLI 中 `--cache` 会将运行模式切换为 `cache`，并自动调整批大小与打乱策略，以预构建缓存。
- 训练时优化器与调度器由 `BaseTrainer.configure_optimizers` 动态实例化，`lr_scheduler.scheduler_type` 通过 `diffusers.optimization.get_scheduler` 创建。
