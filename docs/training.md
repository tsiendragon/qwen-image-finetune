# Training Guide

This comprehensive guide covers the complete training workflow for Qwen Image Finetune, from data preparation to model deployment.

## Overview

The Qwen Image Finetune framework supports multiple training modes optimized for different scenarios:

- **Full Fine-tuning**: Complete model parameter updates
- **LoRA Fine-tuning**: Parameter-efficient fine-tuning
- **Cached Training**: Accelerated training with pre-computed embeddings
- **Multi-GPU Training**: Distributed training across multiple devices

## Quick Start

### Basic Training Workflow

```bash
# 1. 使用提供的toy数据集
# 数据已经在 data/face_seg/ 目录下准备好

# 2. 配置训练
cp configs/face_seg_config.yaml configs/my_config.yaml
# 编辑 my_config.yaml 以设置您的参数

# 3. 预计算嵌入（推荐）
CUDA_VISIBLE_DEVICES=1 python -m src.main --config configs/my_config.yaml --cache

# 4. 开始训练
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml

# 5. 监控训练
python script/check_storage.py
```

## Data Preparation

### Dataset Structure

Organize your dataset in the following structure:

```
data/face_seg/               # Toy数据集示例
├── control_images/          # 输入图像
│   ├── 060002_4_028450_FEMALE_30.jpg
│   ├── 060003_4_028451_FEMALE_65.jpg
│   └── ...
└── training_images/         # 目标图像和文本
    ├── 060002_4_028450_FEMALE_30.png  # 目标图像
    ├── 060002_4_028450_FEMALE_30.txt  # 文本提示
    ├── 060003_4_028451_FEMALE_65.png
    ├── 060003_4_028451_FEMALE_65.txt
    └── ...
```

### Data Requirements

- **Image Format**: JPG, PNG (RGB)
- **Image Size**: Flexible (will be resized during training)
- **Text Format**: Plain text files (.txt)
- **Naming**: Consistent naming between images and prompts

### 使用Toy数据集

```bash
# Toy数据集已经准备好，位于:
ls data/face_seg/control_images/    # 20个输入图像
ls data/face_seg/training_images/   # 20个目标图像 + 20个文本文件

# 验证数据集完整性
echo "数据集统计:"
echo "输入图像: $(ls data/face_seg/control_images/*.jpg | wc -l)"
echo "训练图像: $(ls data/face_seg/training_images/*.png | wc -l)"
echo "文本文件: $(ls data/face_seg/training_images/*.txt | wc -l)"
```

## Configuration

### Basic Configuration

Create or modify a configuration file:

```yaml
# configs/my_training_config.yaml

# Model settings
model:
  pretrained_model_name_or_path: "Qwen/Qwen2-VL-7B-Instruct"
  quantize: false
  lora:
    enabled: true
    rank: 16
    alpha: 32
    target_modules: ["to_q", "to_v", "to_k", "to_out.0"]
    dropout: 0.1

# Data settings
data:
  class_path: "src.data.dataset.ImageDataset"
  init_args:
    dataset_path: "/path/to/your/dataset"
    image_size: [832, 576]
    cache_dir: "/path/to/cache"
    use_cache: true
    batch_size: 4

# Training settings
train:
  num_epochs: 10
  learning_rate: 1e-4
  gradient_accumulation_steps: 4
  gradient_checkpointing: true
  mixed_precision: "bf16"
  max_grad_norm: 1.0
  save_every_n_epochs: 2

# Cache settings
cache:
  use_cache: true
  cache_dir: "/path/to/cache"
  vae_encoder_device: "cuda:1"
  text_encoder_device: "cuda:2"

# Optimizer settings
optimizer:
  class_path: "torch.optim.AdamW"
  init_args:
    lr: 1e-4
    weight_decay: 0.01
    betas: [0.9, 0.999]

# Learning rate scheduler
lr_scheduler:
  class_path: "transformers.get_cosine_schedule_with_warmup"
  init_args:
    num_warmup_steps: 100
    num_training_steps: 1000
```

### Advanced Configuration Options

#### Memory Optimization
```yaml
train:
  gradient_checkpointing: true    # Reduce memory usage by 20-50%
  gradient_accumulation_steps: 8  # Effective larger batch size
  dataloader_num_workers: 4       # Parallel data loading
  pin_memory: true               # Faster GPU transfer
```

#### Multi-GPU Configuration
```yaml
predict:
  devices:
    vae: "cuda:0"
    text_encoder: "cuda:1"
    transformer: "cuda:2"

cache:
  vae_encoder_device: "cuda:1"
  text_encoder_device: "cuda:2"
```

#### Performance Tuning
```yaml
data:
  init_args:
    prefetch_factor: 2
    persistent_workers: true
    drop_last: true

train:
  compile_model: true           # PyTorch 2.0 compilation
  use_memory_efficient_attention: true
```

## Training Modes

### 1. LoRA Fine-tuning (Recommended)

LoRA (Low-Rank Adaptation) provides efficient fine-tuning with minimal memory requirements:

```yaml
model:
  lora:
    enabled: true
    rank: 16                    # Lower rank = fewer parameters
    alpha: 32                   # Scaling factor
    target_modules:
      - "to_q"
      - "to_v"
      - "to_k"
      - "to_out.0"
    dropout: 0.1
```

**Benefits:**
- 90% reduction in trainable parameters
- Faster training and convergence
- Lower memory requirements
- Easy to merge and deploy

### 2. Full Fine-tuning

Complete model parameter updates:

```yaml
model:
  lora:
    enabled: false

train:
  learning_rate: 1e-5           # Lower LR for full fine-tuning
  gradient_checkpointing: true  # Essential for memory
```

**Use Cases:**
- Large datasets (100k+ samples)
- Domain-specific adaptations
- Maximum model customization

### 3. Cached Training

Pre-compute embeddings for significant speed improvements:

```bash
# 预计算嵌入
CUDA_VISIBLE_DEVICES=1 python -m src.main --config configs/my_config.yaml --cache

# 训练将自动检测并使用缓存
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml
```

**Performance Benefits:**
- 2-3x faster training after first epoch
- 30-50% memory reduction
- Consistent embeddings across epochs

## Training Execution

### Single GPU Training

```bash
# 基本训练
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml

# 使用自定义参数
python -m src.main \
    --config configs/my_config.yaml \
    --cache  # 缓存模式
```

### Multi-GPU Training

```bash
# 使用accelerate
accelerate config  # 配置分布式训练
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml

# 使用torchrun
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 -m src.main --config configs/my_config.yaml
```

### Distributed Training

For cluster environments:

```bash
# SLURM
sbatch scripts/train_slurm.sh

# Kubernetes
kubectl apply -f k8s/training-job.yaml
```

## Monitoring and Logging

### Built-in Monitoring

The framework provides comprehensive monitoring:

```python
# Training metrics logged automatically
{
    "epoch": 1,
    "step": 100,
    "loss": 0.045,
    "learning_rate": 1e-4,
    "grad_norm": 0.8,
    "memory_used": "18.5GB",
    "cache_hit_rate": 0.95
}
```

### Storage Monitoring

Monitor disk usage during training:

```bash
# Real-time storage monitoring
python script/check_storage.py

# Automated alerts
python script/check_storage.py --alert-threshold 90
```

### External Logging

#### Weights & Biases Integration

```yaml
logging:
  use_wandb: true
  wandb_project: "qwen-image-finetune"
  wandb_name: "experiment-1"
  log_every_n_steps: 10
```

#### TensorBoard Integration

```yaml
logging:
  use_tensorboard: true
  tensorboard_dir: "./logs/tensorboard"
```

### Progress Tracking

```python
# Custom progress tracking
from src.utils.logger import TrainingLogger

logger = TrainingLogger(
    log_dir="./logs",
    project_name="my_experiment"
)

# Log custom metrics
logger.log_metrics({
    "custom_metric": value,
    "validation_score": score
})
```

## Optimization Strategies

### Memory Optimization

#### 1. Gradient Checkpointing
```yaml
train:
  gradient_checkpointing: true  # 20-50% memory reduction
```

#### 2. Mixed Precision Training
```yaml
train:
  mixed_precision: "bf16"       # 50% memory reduction
```

#### 3. Gradient Accumulation
```yaml
train:
  gradient_accumulation_steps: 8  # Simulate larger batch size
```

#### 4. Optimizer State Sharding
```yaml
optimizer:
  use_8bit: true               # 8-bit optimizer states
```

### Speed Optimization

#### 1. 嵌入缓存
```bash
# 预计算以获得最大速度
CUDA_VISIBLE_DEVICES=1 python -m src.main --config configs/my_config.yaml --cache
```

#### 2. Data Loading Optimization
```yaml
data:
  init_args:
    num_workers: 8             # Parallel data loading
    prefetch_factor: 4         # Prefetch batches
    persistent_workers: true   # Keep workers alive
```

#### 3. Model Compilation
```yaml
train:
  compile_model: true          # PyTorch 2.0 compilation
```

### Quality Optimization

#### 1. Learning Rate Scheduling
```yaml
lr_scheduler:
  class_path: "transformers.get_cosine_schedule_with_warmup"
  init_args:
    num_warmup_steps: 500      # Warm-up for stability
    num_training_steps: 10000
```

#### 2. Gradient Clipping
```yaml
train:
  max_grad_norm: 1.0          # Prevent gradient explosion
```

#### 3. Validation Monitoring
```yaml
validation:
  eval_every_n_epochs: 2
  eval_dataset_path: "/path/to/validation"
  metrics: ["loss", "fid", "lpips"]
```

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)
```bash
# Solutions (in order of preference):
1. Enable gradient checkpointing
2. Reduce batch size
3. Increase gradient accumulation steps
4. Use mixed precision (bf16)
5. Use LoRA instead of full fine-tuning
```

#### Slow Training
```bash
# Solutions:
1. Pre-compute embeddings (cache_embeddings.py)
2. Increase num_workers in dataloader
3. Use faster storage (SSD)
4. Enable model compilation
5. Optimize batch size
```

#### Poor Convergence
```bash
# Solutions:
1. Lower learning rate
2. Add learning rate warmup
3. Check data quality
4. Increase training steps
5. Adjust LoRA parameters (rank, alpha)
```

#### 缓存问题
```bash
# 如果缓存损坏，清除缓存
rm -rf /path/to/cache/*

# 重新构建缓存
CUDA_VISIBLE_DEVICES=1 python -m src.main --config configs/my_config.yaml --cache
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
# 调试训练
python -m src.main \
    --config configs/my_config.yaml
# 注: 调试选项需要在配置文件中设置
```

### Profiling

Profile training performance:

```python
# 启用性能分析
python -m src.main \
    --config configs/my_config.yaml
# 注: 性能分析选项需要在配置文件中设置
```

## Validation and Testing

### Validation During Training

```yaml
validation:
  enabled: true
  eval_every_n_epochs: 2
  eval_dataset_path: "/path/to/validation"
  metrics: ["loss", "fid", "lpips", "clip_score"]
  save_best_model: true
```

### Manual Validation

```python
# Validate trained model
python src/validate.py \
    --model_path /path/to/checkpoint \
    --test_data /path/to/test \
    --output_dir /path/to/results
```

### Quality Metrics

The framework supports various quality metrics:

- **Loss**: Training loss
- **FID**: Fréchet Inception Distance
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **CLIP Score**: Image-text alignment
- **Custom Metrics**: User-defined evaluation functions

## Model Export and Deployment

### Save Trained Model

```python
# Automatic saving during training
trainer.save_checkpoint(epoch=5, global_step=1000)

# Manual saving
trainer.save_model("/path/to/save")
trainer.save_lora("/path/to/lora")  # For LoRA models
```

### Export Formats

#### 1. PyTorch Format
```python
# Save complete model
torch.save(model.state_dict(), "model.pth")

# Save LoRA weights only
trainer.save_lora("lora_weights.pth")
```

#### 2. Hugging Face Format
```python
# Save as Hugging Face model
model.save_pretrained("/path/to/hf/model")
tokenizer.save_pretrained("/path/to/hf/model")
```

#### 3. ONNX Export
```python
# Export to ONNX for deployment
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11
)
```

### Deployment Options

#### 1. Local Inference
```python
from src.qwen_image_edit_trainer import QwenImageEditTrainer

trainer = QwenImageEditTrainer.from_pretrained("/path/to/model")
trainer.setup_predict()

result = trainer.predict(image, prompt)
```

#### 2. API Server
```python
# FastAPI server
from fastapi import FastAPI
from src.inference_server import InferenceServer

app = FastAPI()
server = InferenceServer("/path/to/model")

@app.post("/predict")
def predict(image: bytes, prompt: str):
    return server.predict(image, prompt)
```

#### 3. Production Deployment
```bash
# Docker deployment
docker build -t qwen-inference .
docker run -p 8000:8000 qwen-inference

# Kubernetes deployment
kubectl apply -f k8s/inference-deployment.yaml
```

## Best Practices

### Training Best Practices

1. **Start Small**: Begin with a small dataset and LoRA
2. **Use Cache**: Always pre-compute embeddings for multi-epoch training
3. **Monitor Memory**: Use gradient checkpointing for large models
4. **Validate Regularly**: Set up validation to catch overfitting
5. **Save Frequently**: Save checkpoints regularly

### Performance Best Practices

1. **SSD Storage**: Use fast storage for datasets and cache
2. **Batch Size**: Find optimal batch size for your hardware
3. **Data Loading**: Optimize num_workers and prefetch_factor
4. **Mixed Precision**: Use bf16 for speed and memory benefits
5. **Distributed Training**: Use multiple GPUs when available

### Quality Best Practices

1. **Data Quality**: Ensure high-quality, diverse training data
2. **Learning Rate**: Start with proven learning rates and adjust
3. **Regularization**: Use dropout and weight decay appropriately
4. **Evaluation**: Use multiple metrics to assess model quality
5. **Hyperparameter Tuning**: Systematically optimize hyperparameters

This comprehensive training guide should help you achieve optimal results with the Qwen Image Finetune framework. For additional help, consult the [API Reference](api/README.md) or open an issue on GitHub.
