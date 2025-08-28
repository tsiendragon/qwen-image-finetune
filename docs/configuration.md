# Configuration Guide

This guide covers all configuration options available in the Qwen Image Finetune framework.

## Configuration File Structure

The framework uses YAML configuration files located in the `configs/` directory. Here's the complete structure:

```yaml
# Model Configuration
model:
  pretrained_model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
  quantize: false
  lora:
    enabled: true
    rank: 16
    alpha: 32
    target_modules: ["to_q", "to_v", "to_k", "to_out.0"]
    dropout: 0.1

# Data Configuration
data:
  class_path: "src.data.dataset.ImageDataset"
  init_args:
    dataset_path: "/path/to/dataset"
    image_size: [832, 576]
    batch_size: 4
    cache_dir: "/path/to/cache"
    use_cache: true
    num_workers: 4
    pin_memory: true

# Training Configuration
train:
  num_epochs: 10
  learning_rate: 1e-4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  mixed_precision: "bf16"
  max_grad_norm: 1.0
  save_every_n_epochs: 2
  eval_every_n_epochs: 1

# Cache Configuration
cache:
  use_cache: true
  cache_dir: "/path/to/cache"
  vae_encoder_device: "cuda:1"
  text_encoder_device: "cuda:2"

# Prediction Configuration
predict:
  devices:
    vae: "cuda:0"
    text_encoder: "cuda:1"
    transformer: "cuda:2"

# Optimizer Configuration
optimizer:
  class_path: "torch.optim.AdamW"
  init_args:
    lr: 1e-4
    weight_decay: 0.01
    betas: [0.9, 0.999]

# Learning Rate Scheduler
lr_scheduler:
  class_path: "transformers.get_cosine_schedule_with_warmup"
  init_args:
    num_warmup_steps: 100
    num_training_steps: 1000

# Logging Configuration
logging:
  use_wandb: false
  wandb_project: "qwen-image-finetune"
  log_every_n_steps: 10
  output_dir: "./outputs"
```

## Configuration Sections

### Model Configuration

#### Basic Model Settings
```yaml
model:
  pretrained_model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"  # Base model path
  quantize: false                                              # Enable FP8 quantization
```

#### LoRA Configuration
```yaml
model:
  lora:
    enabled: true                                    # Enable LoRA fine-tuning
    rank: 16                                        # LoRA rank (lower = fewer parameters)
    alpha: 32                                       # LoRA scaling factor
    target_modules:                                 # Modules to apply LoRA
      - "to_q"                                     # Query projection
      - "to_v"                                     # Value projection
      - "to_k"                                     # Key projection
      - "to_out.0"                                 # Output projection
    dropout: 0.1                                   # LoRA dropout rate
```

### Data Configuration

#### Dataset Settings
```yaml
data:
  class_path: "src.data.dataset.ImageDataset"      # Dataset class
  init_args:
    dataset_path: "/path/to/dataset"               # Dataset directory
    image_size: [832, 576]                         # Target image size [width, height]
    batch_size: 4                                  # Training batch size
    shuffle: true                                  # Shuffle training data
    drop_last: true                               # Drop incomplete batches
```

#### Data Loading Optimization
```yaml
data:
  init_args:
    num_workers: 4                                 # Parallel data loading workers
    pin_memory: true                               # Pin memory for faster GPU transfer
    prefetch_factor: 2                            # Prefetch batches
    persistent_workers: true                       # Keep workers alive between epochs
```

#### Cache Settings
```yaml
data:
  init_args:
    cache_dir: "/path/to/cache"                   # Cache directory
    use_cache: true                               # Enable embedding cache
```

### Training Configuration

#### Basic Training Settings
```yaml
train:
  num_epochs: 10                                  # Number of training epochs
  learning_rate: 1e-4                            # Base learning rate
  gradient_accumulation_steps: 4                 # Accumulate gradients over N steps
  max_grad_norm: 1.0                            # Gradient clipping threshold
```

#### Memory Optimization
```yaml
train:
  gradient_checkpointing: true                   # Enable gradient checkpointing
  mixed_precision: "bf16"                        # Mixed precision mode ("bf16", "fp16", "no")
  dataloader_num_workers: 4                      # Data loading parallelism
```

#### Model Compilation and Optimization
```yaml
train:
  compile_model: true                            # Enable PyTorch 2.0 compilation
  use_memory_efficient_attention: true          # Memory efficient attention
```

#### Checkpointing
```yaml
train:
  save_every_n_epochs: 2                        # Save checkpoint frequency
  save_top_k: 3                                 # Keep top K checkpoints
  monitor_metric: "val_loss"                    # Metric to monitor for best model
```

### Cache Configuration

#### Cache Devices
```yaml
cache:
  use_cache: true                                # Enable cache system
  cache_dir: "/path/to/cache"                   # Cache storage directory
  vae_encoder_device: "cuda:1"                  # Device for VAE encoding
  text_encoder_device: "cuda:2"                # Device for text encoding
```

#### Cache Behavior
```yaml
cache:
  force_rebuild: false                          # Force cache rebuild
  cache_batch_size: 8                          # Batch size for caching
  validate_cache: true                         # Validate cache integrity
```

### Prediction Configuration

#### Device Allocation
```yaml
predict:
  devices:
    vae: "cuda:0"                               # VAE encoder/decoder device
    text_encoder: "cuda:1"                      # Text encoder device
    transformer: "cuda:2"                       # Main transformer device
```

#### Inference Settings
```yaml
predict:
  enable_memory_efficient_attention: true      # Memory efficient attention
  enable_vae_slicing: true                     # VAE slicing for large images
  enable_cpu_offload: false                   # Offload unused components to CPU
```

### Optimizer Configuration

#### AdamW (Recommended)
```yaml
optimizer:
  class_path: "torch.optim.AdamW"
  init_args:
    lr: 1e-4                                    # Learning rate
    weight_decay: 0.01                          # Weight decay for regularization
    betas: [0.9, 0.999]                        # Adam beta parameters
    eps: 1e-8                                   # Epsilon for numerical stability
```

#### Alternative Optimizers
```yaml
# SGD
optimizer:
  class_path: "torch.optim.SGD"
  init_args:
    lr: 1e-3
    momentum: 0.9
    weight_decay: 1e-4

# AdaFactor (memory efficient)
optimizer:
  class_path: "transformers.optimization.Adafactor"
  init_args:
    lr: 1e-4
    scale_parameter: false
    relative_step: false
```

### Learning Rate Scheduler

#### Cosine with Warmup (Recommended)
```yaml
lr_scheduler:
  class_path: "transformers.get_cosine_schedule_with_warmup"
  init_args:
    num_warmup_steps: 100                       # Warmup steps
    num_training_steps: 1000                    # Total training steps
```

#### Alternative Schedulers
```yaml
# Linear with warmup
lr_scheduler:
  class_path: "transformers.get_linear_schedule_with_warmup"
  init_args:
    num_warmup_steps: 100
    num_training_steps: 1000

# Constant with warmup
lr_scheduler:
  class_path: "transformers.get_constant_schedule_with_warmup"
  init_args:
    num_warmup_steps: 100

# Polynomial decay
lr_scheduler:
  class_path: "transformers.get_polynomial_decay_schedule_with_warmup"
  init_args:
    num_warmup_steps: 100
    num_training_steps: 1000
    power: 1.0
```

### Logging Configuration

#### Weights & Biases
```yaml
logging:
  use_wandb: true                               # Enable W&B logging
  wandb_project: "qwen-image-finetune"          # W&B project name
  wandb_name: "experiment-1"                    # Run name
  wandb_tags: ["lora", "qwen", "image-edit"]   # Tags for organization
```

#### TensorBoard
```yaml
logging:
  use_tensorboard: true                         # Enable TensorBoard
  tensorboard_dir: "./logs/tensorboard"         # TensorBoard log directory
```

#### Basic Logging
```yaml
logging:
  log_every_n_steps: 10                        # Log frequency
  output_dir: "./outputs"                       # Output directory
  log_level: "INFO"                            # Log level (DEBUG, INFO, WARNING, ERROR)
```

## Configuration Templates

### LoRA Fine-tuning (Recommended)
```yaml
# configs/lora_config.yaml
model:
  lora:
    enabled: true
    rank: 16
    alpha: 32
    dropout: 0.1

train:
  learning_rate: 1e-4
  gradient_checkpointing: true
  mixed_precision: "bf16"
  batch_size: 4
  gradient_accumulation_steps: 4
```

### Full Fine-tuning
```yaml
# configs/full_finetune_config.yaml
model:
  lora:
    enabled: false

train:
  learning_rate: 1e-5                          # Lower LR for full fine-tuning
  gradient_checkpointing: true                 # Essential for memory
  mixed_precision: "bf16"
  batch_size: 2                               # Smaller batch size
  gradient_accumulation_steps: 8              # Higher accumulation
```

### Fast Training with Cache
```yaml
# configs/cached_training_config.yaml
data:
  init_args:
    use_cache: true
    cache_dir: "/fast/ssd/cache"

cache:
  use_cache: true
  vae_encoder_device: "cuda:1"
  text_encoder_device: "cuda:2"

train:
  gradient_checkpointing: false                # Can disable with cache
  batch_size: 8                               # Larger batch with cache
```

### Multi-GPU Inference
```yaml
# configs/inference_config.yaml
predict:
  devices:
    vae: "cuda:0"
    text_encoder: "cuda:1"
    transformer: "cuda:2"

model:
  quantize: true                              # Enable quantization for speed
```

## Environment Variables

You can override configuration values using environment variables:

```bash
# Override learning rate
export LEARNING_RATE=1e-5

# Override batch size
export BATCH_SIZE=8

# Override cache directory
export CACHE_DIR="/custom/cache/path"

# Override output directory
export OUTPUT_DIR="/custom/output/path"
```

## Configuration Best Practices

### Memory Management
1. **Enable gradient checkpointing** for large models
2. **Use mixed precision** (bf16) for memory and speed
3. **Adjust batch size** based on available memory
4. **Use gradient accumulation** to simulate larger batches

### Performance Optimization
1. **Pre-compute embeddings** with cache system
2. **Use multiple data workers** for faster loading
3. **Enable model compilation** with PyTorch 2.0
4. **Distribute components** across multiple GPUs

### Training Stability
1. **Use learning rate warmup** for stable training
2. **Apply gradient clipping** to prevent explosion
3. **Monitor validation metrics** to detect overfitting
4. **Save checkpoints regularly** for recovery

### Quality Optimization
1. **Start with proven hyperparameters** and adjust gradually
2. **Use validation set** to guide hyperparameter tuning
3. **Log comprehensive metrics** for analysis
4. **Experiment systematically** with one change at a time

This configuration guide should help you optimize your training setup for the best results with your specific requirements and hardware.
