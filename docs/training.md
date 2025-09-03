# Training Guide

This guide covers the complete training workflow for Qwen Image Finetune, from dataset preparation to model training with various quantization options.

## Overview

The Qwen Image Finetune framework supports multiple training configurations optimized for different hardware setups and performance requirements:

- **LoRA Fine-tuning**: Parameter-efficient fine-tuning with low memory usage
- **Full Fine-tuning**: Complete model parameter updates
- **Quantized Training**: FP8/FP4 quantization for memory optimization
- **Cached Training**: Accelerated training with pre-computed embeddings
- **Multi-GPU Training**: Distributed training across multiple devices
- **Pretrained LoRA**: Initialize training with existing LoRA weights

## Dataset Preparation

### Dataset Structure

Organize your dataset following this structure:

```
data/your_dataset/
├── control_images/          # Input/control images
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── training_images/         # Target images and text prompts
    ├── image_001.png        # Target image
    ├── image_001.txt        # Text prompt for image_001
    ├── image_002.png
    ├── image_002.txt
    └── ...
```

### Data Requirements

- **Image Formats**: JPG, PNG (RGB format)
- **Image Resolution**: Flexible (automatically resized during training)
- **Text Format**: Plain text files (.txt) containing prompts
- **Naming Convention**: Consistent naming between control images, target images, and text files
- **Minimum Dataset Size**: At least 10-20 image pairs for basic training

### Example Dataset (Face Segmentation)

The repository includes a toy dataset for testing:

```bash
# Check the provided toy dataset
ls data/face_seg/control_images/    # 20 control images
ls data/face_seg/training_images/   # 20 target images + 20 text files

# Verify dataset completeness
echo "Dataset statistics:"
echo "Control images: $(ls data/face_seg/control_images/*.jpg | wc -l)"
echo "Target images: $(ls data/face_seg/training_images/*.png | wc -l)"
echo "Text files: $(ls data/face_seg/training_images/*.txt | wc -l)"
```

### Creating Your Own Dataset

1. **Collect image pairs**: Prepare input (control) and target images
2. **Write prompts**: Create descriptive text prompts for each image pair
3. **Organize files**: Follow the directory structure above
4. **Validate data**: Ensure all files are properly paired and accessible

Example text prompt content:
```txt
Transform the face in the image by applying a smooth skin segmentation mask while preserving facial features and expression.
```

## Configuration Setup

### Basic Training Configuration

Create a configuration file for your training setup:

```yaml
# configs/my_training_config.yaml

# Model configuration
model:
  pretrained_model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
  quantize: false  # Set to true for quantized training
  lora:
    r: 16
    lora_alpha: 32
    init_lora_weights: true
    target_modules: ["to_q", "to_v", "to_k", "to_out.0"]

# Dataset configuration
data:
  class_path: "src.data.dataset.ImageDataset"
  init_args:
    dataset_path: "data/face_seg"  # Path to your dataset
    batch_size: 2
    caption_dropout_rate: 0.1
    prompt_image_dropout_rate: 0.1

# Training parameters
train:
  num_epochs: 10
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  mixed_precision: "bf16"
  max_grad_norm: 1.0
  max_train_steps: 1000
  checkpointing_steps: 500
  checkpoints_total_limit: 3

# Optimizer configuration
optimizer:
  class_path: "torch.optim.AdamW"
  init_args:
    lr: 1e-4
    betas: [0.9, 0.999]
    weight_decay: 0.01
    eps: 1e-8

# Learning rate scheduler
lr_scheduler:
  scheduler_type: "cosine"
  warmup_steps: 100

# Cache configuration
cache:
  use_cache: true
  cache_dir: "cache/face_seg"
  vae_encoder_device: "cuda:1"
  text_encoder_device: "cuda:2"

# Logging configuration
logging:
  output_dir: "output/face_seg_training"  # Base directory, versions created automatically
  logging_dir: "logs"
  report_to: "tensorboard"
  tracker_project_name: "qwen_image_finetune"
```

### Quantization Configurations

#### Runtime FP8 Quantization

For memory-efficient training with minimal quality loss:

```yaml
# configs/fp8_training_config.yaml
model:
  pretrained_model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
  quantize: true  # Enable runtime FP8 quantization
  lora:
    r: 16
    lora_alpha: 32
    target_modules: ["to_q", "to_v", "to_k", "to_out.0"]

train:
  gradient_checkpointing: true  # Recommended with quantization
  mixed_precision: "bf16"

data:
  init_args:
    batch_size: 4  # Can use larger batch size with quantization
```

#### Pre-quantized FP4 Model

For maximum memory savings using pre-quantized models:

```yaml
# configs/fp4_pretrained_config.yaml
model:
  pretrained_model_name_or_path: "path/to/fp4_quantized_model"  # Pre-quantized FP4 model
  quantize: false  # No runtime quantization needed
  lora:
    r: 32  # Higher rank recommended for quantized models
    lora_alpha: 64
    target_modules: ["to_q", "to_v", "to_k", "to_out.0"]

train:
  gradient_checkpointing: true
  mixed_precision: "bf16"
  gradient_accumulation_steps: 8  # May need larger accumulation for stability

data:
  init_args:
    batch_size: 6  # Can use even larger batch size with pre-quantized models
```

### Multi-GPU Configuration

For distributed training across multiple GPUs:

```yaml
# configs/multi_gpu_config.yaml
cache:
  vae_encoder_device: "cuda:1"
  text_encoder_device: "cuda:2"

predict:
  devices:
    vae: "cuda:0"
    text_encoder: "cuda:1"
    transformer: "cuda:2"

train:
  gradient_accumulation_steps: 2  # Adjust based on GPU count
```

## Training Execution

### Automatic Version Management

The framework automatically manages training versions to prevent data loss and enable easy comparison:

**Directory Structure:**
```
output_dir/
└── {tracker_project_name}/
    ├── v0/                 # First training run
    │   ├── events.out.tfevents.*
    │   ├── checkpoint-0-100/
    │   │   └── pytorch_lora_weights.safetensors
    │   └── checkpoint-0-200/
    │       └── pytorch_lora_weights.safetensors
    ├── v1/                 # Second training run
    │   ├── events.out.tfevents.*
    │   └── checkpoints...
    └── v2/                 # Third training run
        └── ...
```

**Features:**
- **Auto-versioning**: Creates `v0`, `v1`, `v2`... for each training run
- **Invalid cleanup**: Removes versions with < 5 training steps (failed runs)
- **Safe restart**: Never overwrites existing valid training data

**Real Example:**
```
/raid/lilong/data/experiment/qwen-edit-face_seg_lora_fp4/
└── face_segmentation_lora/
    ├── v0/
    │   ├── events.out.tfevents.1756887994.workspace-dgx3-lilong-559b7bd5d5-n5x66.616211.0
    │   ├── 1756887994.3818905
    │   ├── 1756887994.383021
    │   ├── checkpoint-0-100/
    │   │   └── pytorch_lora_weights.safetensors
    │   └── checkpoint-0-200/
    │       └── pytorch_lora_weights.safetensors
    └── v1/
        └── (next training run)
```

Note: `{tracker_project_name}` comes from your config's `logging.tracker_project_name` setting.

### Basic Training Workflow

```bash
# 1. Copy and modify configuration
cp configs/face_seg_fp4_4090.yaml configs/my_config.yaml
# Edit my_config.yaml with your parameters

# 2. Pre-compute embeddings (recommended for faster training)
CUDA_VISIBLE_DEVICES=1,2 python -m src.main --config configs/my_config.yaml --cache

# 3. Start training (automatically creates new version)
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml

# 4. Monitor training progress
tensorboard --logdir output_dir/{tracker_project_name}/ --port 6006
# Or check TensorBoard logs directly:
# ls output_dir/{tracker_project_name}/v*/
```

### Single GPU Training

```bash
# Basic single GPU training
CUDA_VISIBLE_DEVICES=0 python -m src.main --config configs/my_config.yaml

# With accelerate (recommended)
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml
```

### Multi-GPU Training

```bash
# Configure accelerate for multi-GPU
accelerate config

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml
```

### Cache-Accelerated Training

Pre-computing embeddings significantly speeds up training:

```bash
# Step 1: Cache embeddings
CUDA_VISIBLE_DEVICES=1,2 python -m src.main --config configs/my_config.yaml --cache

# Step 2: Training will automatically use cached embeddings
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml
```

**Cache Benefits:**
- 2-3x faster training after first epoch
- 30-50% memory reduction during training
- Consistent embeddings across epochs

## Training Modes

### 1. LoRA Fine-tuning (Recommended)

Most efficient approach for fine-tuning:

```yaml
model:
  lora:
    r: 16                    # Rank (higher = more parameters)
    lora_alpha: 32           # Scaling factor
    target_modules:
      - "to_q"
      - "to_v"
      - "to_k"
      - "to_out.0"

train:
  gradient_checkpointing: true
```

**Benefits:**
- 90% reduction in trainable parameters
- Faster convergence
- Lower memory requirements
- Easy to merge and save

### 2. Quantized Training

There are two types of quantization supported:

#### Runtime FP8 Quantization
This applies FP8 quantization during training runtime:

```yaml
model:
  quantize: true              # Enable runtime FP8 quantization
  lora:
    r: 16
    lora_alpha: 32

train:
  gradient_checkpointing: true
  mixed_precision: "bf16"
```

**Use Case**: When you want to reduce memory usage during training while keeping full precision model weights.

#### Pre-quantized FP4 Model Loading
This loads a pre-quantized FP4 model directly:

```yaml
model:
  pretrained_model_name_or_path: "path/to/fp4_quantized_model"  # Pre-quantized model
  quantize: false             # No runtime quantization needed
  lora:
    r: 32                     # Higher rank recommended for quantized models
    lora_alpha: 64

train:
  gradient_checkpointing: true
  mixed_precision: "bf16"
```

**Use Case**: When you have a pre-quantized model and want maximum memory savings.


## Memory Optimization

### Gradient Checkpointing
```yaml
train:
  gradient_checkpointing: true  # 20-50% memory reduction
```

### Mixed Precision Training
```yaml
train:
  mixed_precision: "bf16"       # 50% memory reduction
```

### Gradient Accumulation
```yaml
train:
  gradient_accumulation_steps: 8  # Simulate larger batch size
```

### Quantization
```yaml
# Runtime FP8 quantization
model:
  quantize: true               # 30-50% memory reduction

# Or use pre-quantized FP4 model
model:
  pretrained_model_name_or_path: "path/to/fp4_model"  # 50-70% memory reduction
  quantize: false
```

## Using Pretrained LoRA Weights

You can initialize training with pretrained LoRA weights by specifying the `pretrained_weight` parameter in your configuration.

### Configuration

```yaml
model:
  pretrained_model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
  lora:
    r: 16
    lora_alpha: 32
    target_modules: ["to_q", "to_v", "to_k", "to_out.0"]
    pretrained_weight: "/path/to/pytorch_lora_weights.safetensors"

data:
  init_args:
    dataset_path: "data/your_dataset"
    batch_size: 2

train:
  num_epochs: 5
  max_train_steps: 500
```

### Usage

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/your_config.yaml
```

The framework will automatically load the specified LoRA weights before starting training. Ensure that the LoRA rank (`r`) and target modules match those used when the pretrained weights were created.

## Performance Optimization

### Data Loading Optimization
```yaml
data:
  init_args:
    num_workers: 4           # Parallel data loading
    prefetch_factor: 2       # Prefetch batches
    persistent_workers: true # Keep workers alive
```

### Batch Size Optimization
Start with small batch size and increase gradually:
```yaml
data:
  init_args:
    batch_size: 1           # Start small
    # Increase to 2, 4, 8 based on memory availability
```

### Cache Configuration
```yaml
cache:
  use_cache: true
  cache_dir: "/fast/ssd/cache"  # Use fast storage
  vae_encoder_device: "cuda:1"
  text_encoder_device: "cuda:2"
```

## Troubleshooting

### Out of Memory (OOM)
Solutions in order of preference:
1. Enable gradient checkpointing
2. Use pre-quantized FP4 model (maximum memory savings)
3. Enable runtime FP8 quantization (`quantize: true`)
4. Reduce batch size
5. Increase gradient accumulation steps
6. Use mixed precision (`mixed_precision: "bf16"`)

### Slow Training
Solutions:
1. Pre-compute embeddings with cache
2. Increase `num_workers` in data configuration
3. Use faster storage (SSD) for dataset and cache
4. Optimize batch size for your hardware

### Poor Convergence
Solutions:
1. Lower learning rate (try 5e-5, 1e-5)
2. Add learning rate warmup
3. Increase LoRA rank and alpha
4. Check data quality and diversity
5. Increase training steps

### Cache Issues
```bash
# Clear corrupted cache
rm -rf /path/to/cache/*

# Rebuild cache
CUDA_VISIBLE_DEVICES=1,2 python -m src.main --config configs/my_config.yaml --cache
```

### Version Management
```bash
# Check existing versions
ls output_dir/{tracker_project_name}/  # Shows: v0/ v1/ v2/ ...

# Compare different versions
tensorboard --logdir output_dir/{tracker_project_name}/ --port 6006

# Access specific version checkpoints
ls output_dir/{tracker_project_name}/v1/checkpoint-*

# Access specific version TensorBoard logs
ls output_dir/{tracker_project_name}/v1/

# Remove all versions and start fresh
rm -rf output_dir/{tracker_project_name}/v*
```