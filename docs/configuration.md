# Configuration Guide

This guide covers the configuration options for Qwen Image Finetune training.

## Configuration File Structure

The framework uses YAML configuration files. Here's the complete structure:

```yaml
# Model Configuration
model:
  pretrained_model_name_or_path: "Qwen/Qwen-Image-Edit"  # or "ovedrive/qwen-image-edit-4bit"
  quantize: true  # Runtime FP8 quantization
  lora:
    r: 16                    # LoRA rank (8, 16, 32)
    lora_alpha: 16           # Usually equal to r
    init_lora_weights: "gaussian"
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
    pretrained_weight: null  # Path to pretrained LoRA weights

# Data Configuration
data:
  class_path: "src.data.dataset.ImageDataset"
  init_args:
    dataset_path: "/path/to/dataset"
    image_size: [864, 1048]
    caption_dropout_rate: 0.05
    prompt_image_dropout_rate: 0.05
    cache_dir: ${cache.cache_dir}
    use_cache: ${cache.use_cache}
    cache_drop_rate: 0.1
    random_crop: false
    crop_size: 1024
    crop_scale: [0.8, 1.0]
    center_crop: true
    center_crop_ratio: 1.0
  batch_size: 2
  num_workers: 2
  shuffle: true

# Training Configuration
train:
  gradient_accumulation_steps: 1
  max_train_steps: 6000
  num_epochs: 100
  checkpointing_steps: 100
  checkpoints_total_limit: 20
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  gradient_checkpointing: true

# Optimizer Configuration
optimizer:
  class_path: "bnb.optim.Adam8bit"  # 8-bit Adam for memory efficiency
  init_args:
    lr: 0.0001
    betas: [0.9, 0.999]

# Learning Rate Scheduler
lr_scheduler:
  scheduler_type: "cosine"
  warmup_steps: 50
  num_cycles: 0.5
  power: 1.0

# Cache Configuration
cache:
  vae_encoder_device: "cuda:0"
  text_encoder_device: "cuda:1"
  cache_dir: "/path/to/cache"
  use_cache: true

# Prediction Configuration
predict:
  devices:
    vae: "cuda:0"
    text_encoder: "cuda:0"
    transformer: "cuda:0"

# Logging Configuration
logging:
  output_dir: "/path/to/output"
  logging_dir: "logs"
  report_to: "tensorboard"
  tracker_project_name: "qwen_image_finetune"

# Resume Configuration
resume_from_checkpoint: "latest"

# Validation Configuration (optional)
validation:
  enabled: false
  validation_steps: 200
  num_validation_samples: 4
```

## Key Configuration Sections

### Model Configuration

#### Quantization Options
```yaml
# Runtime FP8 quantization
model:
  pretrained_model_name_or_path: "Qwen/Qwen-Image-Edit"
  quantize: true

# Pre-quantized FP4 model
model:
  pretrained_model_name_or_path: "ovedrive/qwen-image-edit-4bit"
  quantize: false
```

#### LoRA Settings
```yaml
model:
  lora:
    r: 16                    # Rank: 8 (small), 16 (medium), 32 (large)
    lora_alpha: 16           # Usually equal to r
    init_lora_weights: "gaussian"
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]
```

### Data Configuration

#### Dataset Settings
```yaml
data:
  init_args:
    dataset_path: "/path/to/dataset"
    image_size: [864, 1048]    # [height, width]
    caption_dropout_rate: 0.05  # Text prompt dropout
    prompt_image_dropout_rate: 0.05  # Image dropout
```

#### Crop Settings
```yaml
data:
  init_args:
    random_crop: false       # Use center crop instead
    crop_size: 1024         # Square crop size
    crop_scale: [0.8, 1.0]  # Scale range
    center_crop: true
    center_crop_ratio: 1.0
```

### Training Configuration

#### Memory Optimization
```yaml
train:
  gradient_checkpointing: true  # Enable for memory savings
  mixed_precision: "bf16"       # Use bfloat16
  gradient_accumulation_steps: 1 # Increase for larger effective batch
```

#### Training Steps
```yaml
train:
  max_train_steps: 6000    # Total training steps
  num_epochs: 100          # Number of epochs
  checkpointing_steps: 100 # Save frequency
```

### Optimizer Configuration

#### 8-bit Adam (Recommended)
```yaml
optimizer:
  class_path: "bnb.optim.Adam8bit"
  init_args:
    lr: 0.0001
    betas: [0.9, 0.999]
```

#### Standard AdamW (Alternative)
```yaml
optimizer:
  class_path: "torch.optim.AdamW"
  init_args:
    lr: 0.0001
    weight_decay: 0.01
    betas: [0.9, 0.999]
    eps: 1e-8
```

### Cache Configuration

#### Single GPU Setup
```yaml
cache:
  vae_encoder_device: "cuda:0"
  text_encoder_device: "cuda:0"
  use_cache: true
```

#### Multi-GPU Setup
```yaml
cache:
  vae_encoder_device: "cuda:0"
  text_encoder_device: "cuda:1"
  use_cache: true
```

## Example Configurations

### Small Dataset (Runtime FP8)
```yaml
model:
  pretrained_model_name_or_path: "Qwen/Qwen-Image-Edit"
  quantize: true
  lora:
    r: 8
    lora_alpha: 8

data:
  batch_size: 1

train:
  max_train_steps: 1000
  num_epochs: 50
```

### Large Dataset (Pre-quantized FP4)
```yaml
model:
  pretrained_model_name_or_path: "ovedrive/qwen-image-edit-4bit"
  quantize: false
  lora:
    r: 32
    lora_alpha: 32

data:
  batch_size: 4

train:
  max_train_steps: 10000
  num_epochs: 20
```

## Configuration Tips

### Memory Optimization
1. Use 8-bit optimizer: `bnb.optim.Adam8bit`
2. Enable gradient checkpointing: `gradient_checkpointing: true`
3. Use mixed precision: `mixed_precision: "bf16"`
4. Use pre-quantized models for maximum memory savings

### Performance Optimization
1. Enable caching: `use_cache: true`
2. Adjust batch size based on memory
3. Use multiple workers: `num_workers: 2-4`
4. Place encoders on different GPUs if available

### Quality Optimization
1. Use cosine scheduler: `scheduler_type: "cosine"`
2. Add warmup steps: `warmup_steps: 50`
3. Use appropriate LoRA rank (16 for most cases)
4. Enable validation monitoring when available

This configuration guide provides all essential settings for successful training with the Qwen Image Finetune framework.