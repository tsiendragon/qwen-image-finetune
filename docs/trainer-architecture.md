# QwenImageEditTrainer Architecture

`QwenImageEditTrainer` is a specialized training framework designed for the Qwen Image Edit model. It provides comprehensive training, inference, and caching capabilities with intelligent device management and optimization features.

## Overview

The trainer is built on top of the existing `trainer.py` architecture and extends it with multimodal image editing capabilities. It separates components from `QwenImageEditPipeline` to provide fine-grained control over training and inference processes.

## Core Architecture

### Main Components

```python
class QwenImageEditTrainer:
    def __init__(self, config):
        # Core model components (separated from pipeline)
        self.vae = None                   # AutoencoderKLQwenImage
        self.qwen_vl = None               # Qwen2_5_VLForConditionalGeneration (text_encoder)
        self.transformer = None           # QwenImageTransformer2DModel
        self.tokenizer = None             # Qwen2Tokenizer
        self.processor = None             # Qwen2VLProcessor
        self.scheduler = None             # FlowMatchEulerDiscreteScheduler

        # Training infrastructure
        self.accelerator = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0

        # Cache and optimization
        self.use_cache = config.cache.use_cache
        self.cache_exist = check_cache_exists(config.cache.cache_dir)
        self.quantize = config.model.quantize
        self.weight_dtype = torch.bfloat16
```

### Key Features

- **Component Separation**: All pipeline components are individually accessible
- **Smart Caching**: Intelligent embedding cache system for training acceleration
- **Device Management**: Flexible multi-GPU device allocation strategies
- **Mixed Training Modes**: Support for both cached and real-time computation
- **LoRA Support**: Comprehensive LoRA fine-tuning capabilities
- **Gradient Checkpointing**: Memory optimization with gradient checkpointing

## Core Methods

### 1. Model Loading and Setup

#### `load_model()`
Loads and separates all components from QwenImageEditPipeline:

```python
def load_model(self):
    pipe = QwenImageEditPipeline.from_pretrained(
        self.config.model.pretrained_model_name_or_path,
        torch_dtype=self.weight_dtype
    )
    # Separate all components
    self.vae = pipe.vae
    self.qwen_vl = pipe.text_encoder
    self.transformer = pipe.transformer
    self.tokenizer = pipe.tokenizer
    self.processor = pipe.processor
    self.scheduler = pipe.scheduler
```

#### Setup Methods
- `setup_accelerator()`: Initialize accelerator and mixed precision
- `set_lora()`: Configure LoRA training parameters and gradient checkpointing
- `configure_optimizers()`: Setup optimizer and learning rate scheduler

### 2. Training Capabilities

#### Dual Training Modes

**Cached Mode** (`use_cache=True, cache_exist=True`):
- Automatically detects cached embeddings in batch
- Moves non-training components to CPU for memory optimization
- Uses only transformer for training
- Significantly faster training with pre-computed embeddings

**Real-time Mode** (`use_cache=False`):
- Computes all embeddings during training
- Keeps encoders on GPU for real-time computation
- Supports end-to-end training
- More memory intensive but more flexible

```python
def training_step(self, batch):
    # Automatic mode detection
    if 'prompt_embed' in batch and 'pixel_latent' in batch:
        return self._training_step_cached(batch)
    else:
        return self._training_step_compute(batch)
```

#### Training Methods
- `fit(train_dataloader)`: Main training loop with epoch management
- `training_step(batch)`: Single training step with automatic mode detection
- `_training_step_cached(batch)`: Optimized training with cached embeddings
- `_training_step_compute(batch)`: Real-time computation training
- `_compute_loss(...)`: Loss computation with noise prediction

### 3. Caching System

#### Cache Methods
```python
cache_embeddings(train_dataloader)      # Pre-compute embeddings for entire dataset
cache_step(batch, vae_device, text_device) # Single batch caching
_encode_empty_prompt(image, device)     # Encode empty prompt for CFG
```

#### Cached Content Types
- **`pixel_latent`**: Image latent vectors from VAE encoder
- **`control_latent`**: Control image latent vectors
- **`prompt_embed`**: Text prompt embeddings (multimodal)
- **`prompt_embeds_mask`**: Attention masks for prompt embeddings
- **`empty_prompt_embed`**: Empty prompt embeddings for CFG
- **`empty_prompt_embeds_mask`**: Empty prompt attention masks

#### Cache Implementation Benefits
- **Memory Efficiency**: Non-training components moved to CPU during cached training
- **Speed Improvement**: 2-3x training acceleration after first epoch
- **Consistency**: Identical embeddings across epochs ensure stable training
- **Flexibility**: Automatic fallback to real-time computation when cache unavailable

### 4. Inference and Prediction

#### Prediction Setup
```python
def setup_predict(self):
    # Configure devices according to config
    self.vae.to(config.predict.devices.vae)
    self.qwen_vl.to(config.predict.devices.text_encoder)
    self.transformer.to(config.predict.devices.transformer)
```

#### Prediction Method
```python
def predict(
    self,
    prompt_image: np.ndarray,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 20,
    true_cfg_scale: float = 4.0
) -> np.ndarray:
    # Multi-GPU inference with CFG support
```

#### Prediction Features
- **Multi-GPU Support**: Components distributed across different GPUs
- **CFG Support**: Classifier-Free Guidance for better generation quality
- **Dynamic Scheduling**: Adaptive mu value computation for flow matching
- **Complete Pipeline**: Full encoding-to-decoding inference workflow

### 5. Device Management Strategies

#### Training Device Allocation

**Cached Mode**:
```python
# Only transformer needs to be on training GPU
self.qwen_vl.cpu()        # Text encoder to CPU
self.vae.cpu()            # VAE to CPU
self.transformer.to(accelerator.device)  # Transformer on GPU
```

**Real-time Mode**:
```python
# Encoders needed for real-time computation
self.vae.decoder.cpu()    # VAE decoder to CPU (not needed for training)
self.vae.encoder.to(accelerator.device)   # VAE encoder on GPU
self.qwen_vl.to(accelerator.device)       # Text encoder on GPU
self.transformer.to(accelerator.device)   # Transformer on GPU
```

#### Prediction Device Allocation
```python
# Flexible multi-GPU allocation from config
self.vae.to(config.predict.devices.vae)              # e.g., cuda:0
self.qwen_vl.to(config.predict.devices.text_encoder) # e.g., cuda:1
self.transformer.to(config.predict.devices.transformer) # e.g., cuda:2
```

#### Caching Device Allocation
```python
# Specialized devices for cache computation
self.qwen_vl.to(config.cache.text_encoder_device)    # e.g., cuda:1
self.vae.to(config.cache.vae_encoder_device)         # e.g., cuda:2
self.transformer.cpu()  # Not needed during caching
```

### 6. LoRA and Optimization Features

#### LoRA Configuration
```python
def set_lora(self):
    # Configure LoRA parameters
    lora_config = LoraConfig(
        r=self.config.model.lora.rank,
        lora_alpha=self.config.model.lora.alpha,
        target_modules=self.config.model.lora.target_modules,
        lora_dropout=self.config.model.lora.dropout,
    )

    # Enable gradient checkpointing if configured
    if self.config.train.gradient_checkpointing:
        self.transformer.enable_gradient_checkpointing()
```

#### LoRA Methods
- `load_lora(path)`: Load pre-trained LoRA weights
- `save_lora(path)`: Save current LoRA weights
- `merge_lora()`: Merge LoRA weights into base model

#### Gradient Checkpointing
- **Memory Optimization**: 20-50% memory reduction when enabled
- **Speed Trade-off**: Slight computational overhead for memory savings
- **Configurable**: Easy enable/disable through configuration
- **Smart Usage**: Recommended for large models or limited memory scenarios

### 7. Encoding and Decoding Methods

#### Image Processing
```python
encode_image_embedding(image, device)           # Encode single image to latent
encode_prompt_image_embedding(prompt, image, device) # Multimodal encoding
decode_vae_latent(latents)                      # Decode latents to image
```

#### Multimodal Processing
The trainer handles complex multimodal inputs where text prompts are combined with image context:

```python
# Example of multimodal encoding
prompt_embeds, prompt_embeds_mask = self.encode_prompt_image_embedding(
    prompt="Make the sky more dramatic",
    image=input_image,
    device=self.qwen_vl.device
)
```

### 8. Advanced Features

#### Prompt Embedding Padding
The system automatically handles variable-length prompt embeddings:

```python
# Automatic padding for batch processing
# Different prompt lengths and image sizes result in different embedding sizes
# The system automatically pads to batch maximum length
prompt_embeds_mask[:,-1100:-900]
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...],  # Sample 1: all valid
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, ...]], # Sample 2: padded sections
       device='cuda:1')
```

#### FP8 Quantization Support
```python
def quantize_model(self, model, device):
    # FP8 quantization for inference acceleration
    from optimum.quanto import quantize, qfloat8, freeze
    quantize(model, weights=qfloat8)
    freeze(model)
```

## Configuration Integration

### Complete Configuration Support
The trainer supports comprehensive configuration through YAML files:

```yaml
# Training configuration
train:
  gradient_checkpointing: true    # Memory optimization
  mixed_precision: "bf16"         # Mixed precision training
  max_grad_norm: 1.0              # Gradient clipping

# Cache configuration
cache:
  vae_encoder_device: cuda:1      # VAE encoder device
  text_encoder_device: cuda:2     # Text encoder device
  cache_dir: "/path/to/cache"     # Cache directory
  use_cache: true                 # Enable caching

# Prediction configuration
predict:
  devices:
    vae: cuda:5                   # VAE device for inference
    text_encoder: cuda:6          # Text encoder device for inference
    transformer: cuda:7           # Transformer device for inference

# LoRA configuration
model:
  lora:
    rank: 16                      # LoRA rank
    alpha: 32                     # LoRA alpha
    target_modules: ["to_q", "to_v", "to_k", "to_out.0"]
    dropout: 0.1                  # LoRA dropout
```

## Usage Examples

### Basic Training
```python
from qwen_image_edit_trainer import QwenImageEditTrainer
from data.config import load_config_from_yaml

# Load configuration
config = load_config_from_yaml("configs/qwen_image_edit_config.yaml")

# Create trainer
trainer = QwenImageEditTrainer(config)

# Train model
trainer.fit(train_dataloader)
```

### Cache-Accelerated Training
```python
# Pre-compute embeddings for faster training
trainer.cache_embeddings(train_dataloader)

# Training will automatically use cached embeddings
trainer.fit(train_dataloader)  # 2-3x faster after first epoch
```

### Multi-GPU Inference
```python
# Setup for inference with device allocation
trainer.setup_predict()

# Generate image with multi-GPU inference
result = trainer.predict(
    prompt_image=input_image,
    prompt="Transform into winter scene",
    negative_prompt="blurry, low quality",
    num_inference_steps=20,
    true_cfg_scale=4.0
)
```

### LoRA Fine-tuning
```python
# Configure and train with LoRA
trainer.set_lora()
trainer.fit(train_dataloader)

# Save LoRA weights
trainer.save_lora("/path/to/lora/weights")

# Load LoRA for inference
trainer.load_lora("/path/to/lora/weights")
trainer.setup_predict()
```

## Performance Optimizations

### Memory Optimization Strategies
1. **Gradient Checkpointing**: Reduce memory usage by 20-50%
2. **Smart Device Allocation**: Move unused components to CPU
3. **Mixed Precision**: Use bfloat16 for memory and speed optimization
4. **Batch Size Adjustment**: Automatic batch size optimization

### Speed Optimization Features
1. **Embedding Cache**: 2-3x training acceleration
2. **Multi-GPU Distribution**: Parallel processing across devices
3. **Optimized Attention**: Efficient attention computation
4. **Dynamic Compilation**: JIT compilation where applicable

### Quality Assurance
1. **Numerical Stability**: FP16 clipping to prevent overflow
2. **Consistent Embeddings**: Cache ensures reproducible training
3. **Validation Monitoring**: Built-in validation loop support
4. **Checkpoint Management**: Automatic model checkpointing

## Best Practices

### Training Recommendations
1. **Use Cache**: Always pre-compute embeddings for multi-epoch training
2. **Monitor Memory**: Use gradient checkpointing for large models
3. **Device Planning**: Distribute components optimally across available GPUs
4. **Batch Size**: Start with smaller batches and increase gradually

### Inference Optimization
1. **Multi-GPU Setup**: Distribute components to maximize throughput
2. **Quantization**: Use FP8 quantization for production inference
3. **Batch Processing**: Process multiple images simultaneously when possible
4. **Memory Management**: Monitor GPU memory usage during inference

This architecture provides a robust, scalable, and efficient framework for training and deploying Qwen Image Edit models with state-of-the-art optimization techniques.
