# Flux Kontext LoRA Training Implementation

## Overview

This document outlines the implementation plan for integrating Flux Kontext LoRA training capabilities into the existing Qwen Image Edit training framework. The implementation follows an incremental approach to ensure backward compatibility while adding new functionality.

## Architecture Design

### Core Principles
- **Non-invasive Integration**: Preserve existing Qwen Image Edit functionality without modifications
- **Modular Design**: Implement Flux Kontext as separate, independent modules
- **Incremental Development**: Build new features that are compatible with existing infrastructure
- **Configuration-driven**: Support multiple model variants through configuration files

### System Components
```
src/
├── base_trainer.py (new - abstract base class)
├── qwen_image_edit_trainer.py (existing - unchanged)
├── flux_kontext_trainer.py (new - inherits from BaseTrainer)
├── models/
│   ├── load_model.py (existing - unchanged)
│   └── flux_kontext_loader.py (new)
└── ...

tests/
├── models/
│   └── test_flux_kontext_loader.py (new)
├── test_flux_kontext_trainer.py (new)
├── test_base_trainer.py (new)
└── ...

configs/
├── flux_kontext_bf16.yaml (new - highest quality)
├── flux_kontext_fp8.yaml (new - balanced performance)
└── flux_kontext_fp4.yaml (new - resource efficient)

docs/
├── architecture/
│   └── flux_kontext_architecture.md (new - detailed model analysis)
└── flux_kontext_configuration_guide.md (new - config selection guide)
```

## Implementation Details

### 1. QwenImageEditTrainer Refactoring

#### Migration to Abstract Base Class
As part of implementing the Flux Kontext trainer, we will refactor the existing `QwenImageEditTrainer` to inherit from the new `BaseTrainer` abstract class. This ensures consistency and maintainability across all trainer implementations.

#### Refactoring Steps
1. **Extract Common Methods**: Move shared functionality from `QwenImageEditTrainer` to `BaseTrainer`
2. **Maintain Interface**: Ensure all existing method signatures remain unchanged
3. **Preserve Functionality**: All existing training, caching, and prediction behavior remains identical
4. **Add Type Hints**: Enhance type safety with proper annotations

#### Backward Compatibility
- All existing configurations and scripts continue to work without modification
- Method signatures remain exactly the same
- Training behavior and results are identical
- Cache files and checkpoint formats remain compatible

### 2. Abstract Base Trainer (`src/base_trainer.py`)

#### Design Philosophy
To ensure consistency across different trainer implementations and maintain a unified interface, we introduce an abstract base class that defines the core training protocol. This approach follows the existing `QwenImageEditTrainer` structure while providing extensibility for new model architectures.

#### Abstract Base Class Definition
```python
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Any
import torch
import PIL
from accelerate import Accelerator

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

    # Common methods that can be shared across implementations
    def setup_accelerator(self):
        """Initialize accelerator and logging configuration."""
        # Common accelerator setup logic
        pass

    def save_checkpoint(self, epoch: int, global_step: int):
        """Save model checkpoint."""
        pass

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        pass

    @abstractmethod
    def set_model_devices(self, mode: str = "train"):
        """Set model device allocation based on different modes."""
        pass

    @abstractmethod
    def encode_prompt(self, *args, **kwargs):
        """Encode text prompts to embeddings."""
        pass
```

#### Implementation Benefits
- **Consistency**: All trainers follow the same interface
- **Maintainability**: Common functionality is centralized
- **Extensibility**: Easy to add new trainer types
- **Type Safety**: Clear method signatures and return types
- **Documentation**: Standardized docstring format

### 2. Model Loading Module (`src/models/flux_kontext_loader.py`)

#### Supported Components
The Flux Kontext model consists of the following components:
- **Text Encoder (CLIP)**: Primary text encoding
- **Text Encoder 2 (T5)**: Secondary text encoding for enhanced understanding
- **VAE**: Variational Autoencoder for image encoding/decoding
- **Transformer**: Core diffusion transformer model
- **Tokenizers**: CLIP and T5 tokenizers

#### Loading Functions
```python
def load_flux_kontext_clip(model_path: str, weight_dtype: torch.dtype, device_map: str = "cpu")
def load_flux_kontext_t5(model_path: str, weight_dtype: torch.dtype, device_map: str = "cpu")
def load_flux_kontext_vae(model_path: str, weight_dtype: torch.dtype, device_map: str = "cpu")
def load_flux_kontext_transformer(model_path: str, weight_dtype: torch.dtype, device_map: str = "cpu")
def load_flux_kontext_tokenizers(model_path: str)
```

#### Model Validation
Create comprehensive tests to validate that individually loaded components match the pipeline-loaded components:

```python
# Reference implementation for validation
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
)
clip = pipe.text_encoder
t5 = pipe.text_encoder_2
tokenizer_clip = pipe.tokenizer
tokenizer_t5 = pipe.tokenizer_2
transformer = pipe.transformer
vae = pipe.vae
```

**Validation Requirements**:
- Parameter shape consistency
- Parameter value consistency (within numerical tolerance)
- Memory footprint comparison
- Loading time benchmarks

### 3. FluxKontextLoraTrainer (`src/flux_kontext_trainer.py`)

#### Class Structure and Inheritance
Following the `QwenImageEditTrainer` design pattern, the `FluxKontextLoraTrainer` inherits from `BaseTrainer` and implements all required abstract methods:

```python
class FluxKontextLoraTrainer(BaseTrainer):
    """
    Flux Kontext LoRA Trainer implementation following QwenImageEditTrainer patterns.
    Inherits from BaseTrainer to ensure consistent interface.
    """

    def __init__(self, config):
        """Initialize Flux Kontext trainer with configuration."""
        super().__init__(config)

        # Flux Kontext specific components (similar to QwenImageEditTrainer structure)
        self.vae = None                    # FluxVAE
        self.text_encoder = None           # CLIP text encoder
        self.text_encoder_2 = None         # T5 text encoder
        self.transformer = None            # Flux transformer
        self.tokenizer = None              # CLIP tokenizer
        self.tokenizer_2 = None            # T5 tokenizer
        self.scheduler = None              # FlowMatchEulerDiscreteScheduler

        # Cache-related attributes (following QwenImageEditTrainer pattern)
        self.cache_exist = check_cache_exists(config.cache.cache_dir)

        # Flux-specific configurations
        self.quantize = config.model.quantize
        self.prompt_image_dropout_rate = config.data.init_args.get('prompt_image_dropout_rate', 0.1)

        # VAE parameters (similar to QwenImageEditTrainer)
        self.vae_scale_factor = None
        self.vae_latent_mean = None
        self.vae_latent_std = None
        self.vae_z_dim = None

    def load_model(self, text_encoder_device=None, text_encoder_2_device=None):
        """
        Load and separate components from FluxKontextPipeline.
        Follows QwenImageEditTrainer.load_model() pattern exactly.
        """
        logging.info("Loading FluxKontextPipeline and separating components...")

        # Load complete model using pipeline (similar to QwenImageEditTrainer)
        pipe = FluxKontextPipeline.from_pretrained(
            self.config.model.pretrained_model_name_or_path,
            torch_dtype=self.weight_dtype,
        )
        pipe.to('cpu')

        # Separate individual components using flux_kontext_loader
        from qflux.models.flux_kontext_loader import (
            load_flux_kontext_vae, load_flux_kontext_clip,
            load_flux_kontext_t5, load_flux_kontext_transformer
        )

        self.vae = load_flux_kontext_vae(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )
        self.text_encoder = load_flux_kontext_clip(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )
        self.text_encoder_2 = load_flux_kontext_t5(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )
        self.transformer = load_flux_kontext_transformer(
            self.config.model.pretrained_model_name_or_path,
            weight_dtype=self.weight_dtype
        )

        # Load tokenizers and scheduler from pipeline
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.scheduler = pipe.scheduler

        # Set VAE-related parameters (following QwenImageEditTrainer pattern)
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # Additional Flux-specific VAE configuration

        # Set models to training/evaluation mode (same as QwenImageEditTrainer)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        torch.cuda.empty_cache()

        logging.info(f"Components loaded successfully. VAE scale factor: {self.vae_scale_factor}")

    def cache(self, train_dataloader):
        """
        Pre-compute and cache embeddings (exactly same signature as QwenImageEditTrainer).
        Implements dual text encoder caching for CLIP + T5.
        """
        from tqdm import tqdm



        self.cache_manager = train_dataloader.cache_manager
        vae_encoder_device = self.config.cache.vae_encoder_device
        text_encoder_device = self.config.cache.text_encoder_device
        text_encoder_2_device = self.config.cache.get('text_encoder_2_device', text_encoder_device)

        logging.info("Starting embedding caching process...")

        # Load models (following QwenImageEditTrainer pattern)
        self.load_model(
            text_encoder_device=text_encoder_device,
            text_encoder_2_device=text_encoder_2_device
        )
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.vae.eval()
        self.set_model_devices(mode="cache")

        # Cache for each item (same loop structure as QwenImageEditTrainer)
        dataset = train_dataloader.dataset
        for data in tqdm(dataset, total=len(dataset), desc="cache_embeddings"):
            self.cache_step(data, vae_encoder_device, text_encoder_device, text_encoder_2_device)

        logging.info("Cache completed")

        # Clean up models (same as QwenImageEditTrainer)
        self.text_encoder.cpu()
        self.text_encoder_2.cpu()
        self.vae.cpu()
        del self.text_encoder
        del self.text_encoder_2
        del self.vae

    def fit(self, train_dataloader):
        """
        Main training loop (exactly same signature as QwenImageEditTrainer).
        Implements Flux Kontext specific training with LoRA.
        """
        logging.info("Starting training process...")

        # Setup components (same order as QwenImageEditTrainer)
        self.setup_accelerator()
        self.load_model()

        self.set_lora()  # Flux-specific LoRA setup
        self.text_encoder.eval()
        self.text_encoder_2.eval()
        self.vae.eval()
        self.configure_optimizers()
        self.set_model_devices(mode="train")
        train_dataloader = self.accelerator_prepare(train_dataloader)

        # Training loop implementation (following QwenImageEditTrainer structure)
        # ... (detailed implementation follows QwenImageEditTrainer.fit() pattern)

    def predict(
        self,
        prompt_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
        prompt: Union[str, List[str]],
        negative_prompt: Union[None, str, List[str]] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Prediction method (similar signature to QwenImageEditTrainer.predict).
        Implements Flux Kontext specific inference pipeline.
        """
        logging.info(f"Starting prediction with {num_inference_steps} steps, guidance scale: {guidance_scale}")

        # Implementation follows FluxKontextPipeline.__call__ logic
        # but uses pre-loaded model components
        # ... (detailed implementation)

    # Additional methods following QwenImageEditTrainer patterns
    def set_model_devices(self, mode="train"):
        """Set model device allocation (same signature as QwenImageEditTrainer)."""
        # Flux-specific multi-device allocation logic
        pass

    def encode_prompt(self, prompt, image=None, device=None):
        """
        Encode prompts using both CLIP and T5 encoders.
        Returns combined embeddings compatible with Flux Kontext.
        """
        # Dual encoder implementation for CLIP + T5
        pass

    def get_clip_prompt_embeds(self, prompt: str):
        """Get CLIP text embeddings."""
        pass

    def get_t5_prompt_embeds(self, prompt: str):
        """Get T5 text embeddings."""
        pass
```

#### Caching System Implementation
Following the `QwenImageEditTrainer.cache_step()` pattern, the Flux Kontext trainer implements dual-encoder caching:

**Cached Components** (following QwenImageEditTrainer structure):
- `pixel_latent`: VAE-encoded source images (same as QwenImageEditTrainer)
- `control_latent`: VAE-encoded target/control images (same as QwenImageEditTrainer)
- `clip_prompt_embed`: CLIP text embeddings (new for Flux)
- `clip_prompt_embeds_mask`: CLIP attention masks (new for Flux)
- `t5_prompt_embed`: T5 text embeddings (new for Flux)
- `t5_prompt_embeds_mask`: T5 attention masks (new for Flux)
- `empty_clip_prompt_embed`: Empty CLIP embeddings for CFG (new for Flux)
- `empty_t5_prompt_embed`: Empty T5 embeddings for CFG (new for Flux)

**Cache Implementation** (following QwenImageEditTrainer.cache_step pattern):
```python
def cache_step(self, data: dict, vae_encoder_device: str, text_encoder_device: str, text_encoder_2_device: str):
    """
    Cache VAE latents and dual prompt embeddings.
    Follows QwenImageEditTrainer.cache_step() structure exactly.
    """
    image, control, prompt = data["image"], data["control"], data["prompt"]

    # Image processing (same as QwenImageEditTrainer)
    image = image.transpose(1, 2, 0)
    control = control.transpose(1, 2, 0)
    image = Image.fromarray(image)
    control = Image.fromarray(control)

    # Calculate embeddings for both encoders
    clip_embeds, clip_mask = self.encode_clip_prompt(prompt, device=text_encoder_device)
    t5_embeds, t5_mask = self.encode_t5_prompt(prompt, device=text_encoder_2_device)

    # Empty prompt embeddings for CFG
    empty_clip_embeds, empty_clip_mask = self.encode_clip_prompt("", device=text_encoder_device)
    empty_t5_embeds, empty_t5_mask = self.encode_t5_prompt("", device=text_encoder_2_device)

    # VAE encoding (same logic as QwenImageEditTrainer)
    _, image_latents = self.prepare_latents(...)
    _, control_latents = self.prepare_latents(...)

    # Save to cache (following QwenImageEditTrainer pattern)
    file_hashes = data["file_hashes"]
    self.cache_manager.save_cache("pixel_latent", file_hashes["image_hash"], image_latents[0].cpu())
    self.cache_manager.save_cache("control_latent", file_hashes["control_hash"], control_latents[0].cpu())
    self.cache_manager.save_cache("clip_prompt_embed", file_hashes["prompt_hash"], clip_embeds[0].cpu())
    self.cache_manager.save_cache("t5_prompt_embed", file_hashes["prompt_hash"], t5_embeds[0].cpu())
    # ... additional cache saves
```

**Cache Strategy** (same as QwenImageEditTrainer):
- Support both cached and non-cached training modes
- Use existing cache_manager from QwenImageEditTrainer
- Memory-efficient loading with same batch processing logic
- Compatible with existing cache directory structure

#### Training Implementation (Following QwenImageEditTrainer Patterns)

**Training Step Structure** (same as QwenImageEditTrainer):
```python
def training_step(self, batch):
    """Execute a single training step (same signature as QwenImageEditTrainer)."""
    # Check if cached data is available (same logic as QwenImageEditTrainer)
    if 'clip_prompt_embed' in batch and 't5_prompt_embed' in batch and 'pixel_latent' in batch:
        return self._training_step_cached(batch)
    else:
        return self._training_step_compute(batch)

def _training_step_cached(self, batch):
    """Training step using cached embeddings (follows QwenImageEditTrainer pattern)."""
    pixel_latents = batch["pixel_latent"].to(self.accelerator.device, dtype=self.weight_dtype)
    control_latents = batch["control_latent"].to(self.accelerator.device, dtype=self.weight_dtype)

    # Dual encoder embeddings (Flux-specific)
    clip_embeds = batch["clip_prompt_embed"].to(self.accelerator.device)
    t5_embeds = batch["t5_prompt_embed"].to(self.accelerator.device)

    # Combine embeddings for Flux Kontext (model-specific logic)
    combined_embeds = self.combine_text_embeddings(clip_embeds, t5_embeds)

    return self._compute_loss(pixel_latents, control_latents, combined_embeds, ...)

def _compute_loss(self, pixel_latents, control_latents, prompt_embeds, ...):
    """Calculate the flow matching loss (same structure as QwenImageEditTrainer)."""
    # Same flow matching loss implementation as QwenImageEditTrainer
    # but adapted for Flux Kontext transformer input format
    pass
```

**Loss Function** (consistent with QwenImageEditTrainer):
- Flow matching loss for diffusion training (same as QwenImageEditTrainer)
- `compute_density_for_timestep_sampling` (same utility functions)
- `compute_loss_weighting_for_sd3` (same weighting strategy)
- Sigma calculation and noise scheduling (adapted for Flux)

**LoRA Configuration** (following QwenImageEditTrainer.set_lora() pattern):
```python
def set_lora(self):
    """Set LoRA configuration (same structure as QwenImageEditTrainer)."""
    if self.quantize:
        self.transformer = self.quantize_model(self.transformer, self.accelerator.device)
    else:
        self.transformer.to(self.accelerator.device)

    # Same LoRA config structure as QwenImageEditTrainer
    lora_config = LoraConfig(
        r=self.config.model.lora.r,
        lora_alpha=self.config.model.lora.lora_alpha,
        init_lora_weights=self.config.model.lora.init_lora_weights,
        target_modules=self.config.model.lora.target_modules,  # Flux-specific modules
    )

    self.transformer.add_adapter(lora_config)
    # Same gradient setup as QwenImageEditTrainer
    self.transformer.requires_grad_(False)
    # ... (same pattern as QwenImageEditTrainer.set_lora())
```

**Multi-Device Support** (same as QwenImageEditTrainer.set_model_devices()):
- Cache mode: Only encoders, no transformer
- Train mode: Allocate based on cache availability
- Predict mode: Configurable device mapping
- Same memory management and cleanup patterns

#### Prediction Pipeline (Following QwenImageEditTrainer.predict() Structure)

**Method Signature** (consistent with QwenImageEditTrainer):
```python
def predict(
    self,
    prompt_image: Union[PIL.Image.Image, List[PIL.Image.Image]],
    prompt: Union[str, List[str]],
    negative_prompt: Union[None, str, List[str]] = None,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,  # renamed from true_cfg_scale for consistency
    **kwargs
) -> Union[np.ndarray, List[np.ndarray]]:
```

**Implementation Structure** (same flow as QwenImageEditTrainer.predict()):
1. **Input Processing**: Same image size calculation and batching logic
2. **Device Allocation**: Use predict mode device mapping
3. **Prompt Encoding**: Dual encoder (CLIP + T5) instead of single Qwen encoder
4. **Latent Preparation**: Same `prepare_latents()` logic adapted for Flux
5. **Denoising Loop**: Same scheduler and timestep handling
6. **CFG Implementation**: Same classifier-free guidance pattern
7. **VAE Decoding**: Same latent-to-image conversion

**Core Helper Functions** (adapted from QwenImageEditTrainer):
```python
def encode_prompt(self, prompt, image=None, device=None):
    """
    Dual encoder prompt encoding (CLIP + T5).
    Follows QwenImageEditTrainer.encode_prompt() signature.
    """
    # Combine CLIP and T5 embeddings similar to how QwenImageEditTrainer
    # combines text and image embeddings
    pass

def prepare_latents(self, image, batch_size, num_channels_latents, height, width, dtype, device, generator=None, latents=None):
    """Same signature and logic as QwenImageEditTrainer.prepare_latents()"""
    # Flux-specific latent preparation but same overall structure
    pass

def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator = None):
    """Same signature as QwenImageEditTrainer._encode_vae_image()"""
    # Flux VAE encoding with same normalization patterns
    pass
```

**CFG Implementation** (same pattern as QwenImageEditTrainer):
- Same memory management for dual inference passes
- Same guidance scale application and norm preservation
- Same device allocation strategy for positive/negative prompts

### 4. Data Processing

#### Image Handling
- **No Resizing**: Use original dataset image dimensions
- **Latent Scaling**: Extract scaling factors from FluxKontext pipeline configuration
- **Aspect Ratio**: Maintain original aspect ratios to preserve image quality

#### Text Processing
- **Dual Encoding**: Support both CLIP and T5 text encoders
- **Prompt Engineering**: Handle context-aware prompting for image editing tasks
- **Empty Prompts**: Support for unconditional generation in CFG

### 5. Configuration Management

#### Supported Model Variants
1. **BF16 Model**: `black-forest-labs/FLUX.1-Kontext-dev`
   - **Config**: `configs/flux_kontext_bf16.yaml`
   - **Quality**: Highest (reference level)
   - **VRAM**: 24GB training, 12GB inference
   - **Use Case**: Production, research, benchmarking

2. **FP8 Model**: `AlekseyCalvin/Flux_Kontext_Dev_fp8_scaled_diffusers`
   - **Config**: `configs/flux_kontext_fp8.yaml`
   - **Quality**: 95% of BF16, 1.5x faster training
   - **VRAM**: 18GB training, 8GB inference
   - **Use Case**: Balanced performance and quality

3. **FP4 Model**: `eramth/flux-kontext-4bit-fp4`
   - **Config**: `configs/flux_kontext_fp4.yaml`
   - **Quality**: 85% of BF16, 2.5x faster training
   - **VRAM**: 12GB training, 5GB inference
   - **Use Case**: Consumer GPUs, rapid prototyping

For detailed configuration analysis and selection guidance, see:
- **Model Architecture**: `docs/architecture/flux_kontext_architecture.md`
- **Configuration Guide**: `docs/flux_kontext_configuration_guide.md`

#### Configuration Structure (Following QwenImageEditTrainer Config Pattern)
```yaml
model:
  pretrained_model_name_or_path: "black-forest-labs/FLUX.1-Kontext-dev"
  model_type: "flux_kontext"  # new field to distinguish trainer types
  quantize: false  # same as QwenImageEditTrainer
  lora:
    r: 16
    lora_alpha: 16
    init_lora_weights: "gaussian"  # same as QwenImageEditTrainer
    target_modules: ["to_k", "to_q", "to_v", "to_out.0"]  # Flux-specific
    pretrained_weight: null  # same as QwenImageEditTrainer

data:
  class_path: "qflux.data.dataset.ImageDataset"  # same dataset class
  init_args:
    dataset_path: "/path/to/flux_dataset/"
    image_size: [1024, 1024]  # Flux typical size
    caption_dropout_rate: 0.05
    prompt_image_dropout_rate: 0.05
    cache_dir: ${cache.cache_dir}
    use_cache: ${cache.use_cache}
  batch_size: 1  # Flux models typically need smaller batches
  num_workers: 2
  shuffle: true

logging:
  output_dir: "/path/to/flux_experiments/"
  logging_dir: "logs"
  report_to: "tensorboard"
  tracker_project_name: "flux_kontext_lora"

optimizer:
  class_path: "torch.optim.AdamW"  # same as QwenImageEditTrainer
  init_args:
    lr: 0.0001
    betas: [0.9, 0.999]
    weight_decay: 0.01
    eps: 1e-8

lr_scheduler:
  scheduler_type: "cosine"
  warmup_steps: 100
  num_cycles: 0.5
  power: 1.0

train:
  gradient_accumulation_steps: 4  # larger due to smaller batch size
  max_train_steps: 5000
  num_epochs: 50
  checkpointing_steps: 100
  checkpoints_total_limit: 10
  max_grad_norm: 1.0
  mixed_precision: "bf16"
  gradient_checkpointing: true
  low_memory: true
  vae_encoder_device: cuda:0
  text_encoder_device: cuda:1  # CLIP encoder
  text_encoder_2_device: cuda:1  # T5 encoder (new field)

cache:
  vae_encoder_device: cuda:0
  text_encoder_device: cuda:1
  text_encoder_2_device: cuda:1  # new field for T5
  cache_dir: "/path/to/flux_cache"
  use_cache: true

predict:
  devices:
    vae: cuda:0
    text_encoder: cuda:1
    text_encoder_2: cuda:1  # new field
    transformer: cuda:0

# Same as QwenImageEditTrainer
resume_from_checkpoint: "latest"

validation:
  enabled: false
  validation_steps: 200
  num_validation_samples: 4
```

## Testing Strategy

### 1. Unit Tests
- **Model Loading**: Validate each component loads correctly
- **Parameter Consistency**: Ensure loaded models match pipeline versions
- **Memory Management**: Test device allocation and cleanup
- **Configuration Parsing**: Validate config file processing

### 2. Integration Tests
- **End-to-End Training**: Complete training pipeline validation
- **Cache System**: Verify caching improves performance without affecting results
- **Multi-Device**: Test distributed component loading
- **Quantization**: Validate different precision models work correctly

### 3. Performance Tests
- **Benchmark Comparison**: Trainer vs. direct pipeline performance
- **Memory Profiling**: Track memory usage across training phases
- **Speed Analysis**: Measure training and inference throughput
- **Quality Metrics**: Compare output quality with reference implementations

### 4. Compatibility Tests
- **CFG Modes**: Test with and without classifier-free guidance
- **Batch Processing**: Validate different batch sizes
- **Long Training**: Stability tests for extended training sessions

## Implementation Timeline

### Phase 1: Foundation & Refactoring (Week 1-2)
- [ ] Create `BaseTrainer` abstract class
- [ ] Refactor `QwenImageEditTrainer` to inherit from `BaseTrainer`
- [ ] Validate backward compatibility of refactored `QwenImageEditTrainer`
- [ ] Implement Flux Kontext model loading functions (`flux_kontext_loader.py`)
- [ ] Set up unit tests for model loading and base trainer
- [ ] Validate model component consistency with pipeline

### Phase 2: Core Flux Kontext Implementation (Week 3-4)
- [ ] Create `FluxKontextLoraTrainer` inheriting from `BaseTrainer`
- [ ] Implement dual-encoder caching system (CLIP + T5)
- [ ] Add training loop with Flux-specific LoRA support
- [ ] Implement loss computation following QwenImageEditTrainer patterns
- [ ] Create prediction pipeline with CFG support

### Phase 3: Integration & Configuration (Week 5-6)
- [ ] Develop Flux Kontext configuration management
- [ ] Add multi-device support for dual encoders
- [ ] Implement quantization support (fp8, fp4)
- [ ] Create configuration files for different model variants
- [ ] Performance optimization and memory efficiency improvements

### Phase 4: Testing & Validation (Week 7-8)
- [ ] Comprehensive test suite for both trainers
- [ ] Performance benchmarking (Flux vs Qwen, trainer vs pipeline)
- [ ] Quality metrics validation
- [ ] Integration testing with existing datasets
- [ ] Documentation completion and examples

## Quality Assurance

### Code Standards
- **PEP 8 Compliance**: Follow Python coding standards
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings for all public methods
- **Error Handling**: Robust exception handling and logging

### Testing Requirements
- **Pytest Framework**: All tests use pytest conventions
- **Coverage**: Minimum 80% test coverage
- **Directory Structure**: Mirror src/ structure in tests/
- **Automated Testing**: CI/CD integration for continuous validation

### Performance Criteria
- **Memory Efficiency**: No memory leaks during extended training
- **Speed Requirements**: Training speed within 10% of reference implementation
- **Quality Metrics**: Generated images meet quality benchmarks
- **Resource Usage**: Efficient GPU utilization across devices

## Risk Mitigation

### Technical Risks
- **Model Availability**: Verify all referenced models are accessible
- **API Changes**: Monitor diffusers library updates for breaking changes
- **Hardware Compatibility**: Test across different GPU configurations
- **Quantization Issues**: Validate numerical stability with reduced precision

### Mitigation Strategies
- **Fallback Options**: Alternative model sources for each precision level
- **Version Pinning**: Lock dependency versions to ensure stability
- **Comprehensive Testing**: Extensive validation across hardware configurations
- **Documentation**: Clear troubleshooting guides for common issues

## Dependencies

### Core Dependencies
- `torch >= 1.13.0`
- `diffusers >= 0.24.0`
- `transformers >= 4.35.0`
- `accelerate >= 0.24.0`
- `peft >= 0.6.0`

### Development Dependencies
- `pytest >= 7.0.0`
- `pytest-cov >= 4.0.0`
- `black >= 23.0.0`
- `isort >= 5.12.0`

### Compatibility Notes
- All dependencies verified compatible with existing Qwen Image Edit requirements
- No version conflicts with current environment
- Incremental updates supported for future compatibility

---

## Conclusion

This implementation plan provides a comprehensive approach to integrating Flux Kontext LoRA training while maintaining system stability and performance. The modular design ensures easy maintenance and future extensibility, while the thorough testing strategy guarantees reliability and quality.
