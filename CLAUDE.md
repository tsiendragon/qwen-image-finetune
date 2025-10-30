# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

This is a PyTorch-based framework for fine-tuning vision-language models (Qwen-Image-Edit and FLUX Kontext) using LoRA (Low-Rank Adaptation). The codebase follows a modular trainer pattern:

- **Base Trainer System**: `src/trainer/base_trainer.py` defines the abstract interface
- **Model-specific Trainers**:
  - `QwenImageEditTrainer` in `src/trainer/qwen_image_edit_trainer.py`
  - `FluxKontextLoraTrainer` in `src/trainer/flux_kontext_trainer.py`
- **Dynamic Trainer Loading**: `src/main.py` imports trainers based on config `train.trainer` field

### Key Components

- **Cache System** (`src/data/cache_manager.py`): Embedding cache for 2-3x training speedup
- **Edit Mask Loss** (`src/loss/edit_mask_loss.py`): Mask-weighted loss for focused training
- **Quantization Support** (`src/models/quantize.py`): FP4/FP8/FP16 quantization
- **Validation Sampling** (`src/validation/validation_sampler.py`): Real-time progress monitoring
- **Model Loaders**: Separate loaders for different model architectures

## Essential Commands

### Environment Setup
```bash
# Automated setup with conda
./setup.sh [BASE_PATH] [HF_TOKEN]

# Manual setup
pip install -r requirements.txt
```

### Training Commands

**Build cache first (recommended for speed):**
```bash
CUDA_VISIBLE_DEVICES=1 python -m qflux.main --config configs/my_config.yaml --cache
```

**Single GPU training:**
```bash
CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file accelerate_config.yaml -m qflux.main --config configs/my_config.yaml
```

**Multi-GPU training:**
```bash
CUDA_VISIBLE_DEVICES=1,2,4 accelerate launch --config_file accelerate_config.yaml -m qflux.main --config configs/my_config.yaml
```

**RTX 4090 specific:**
```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config.yaml -m qflux.main --config configs/my_config.yaml
```

### Testing
```bash
python run_tests.py  # Run model comparison tests
python -m pytest tests/  # Run specific test files
```

## Configuration System

All training is configured via YAML files in `configs/`. Key config sections:

- `model`: Model path, LoRA settings, quantization
- `data`: Dataset path, batch size, image processing
- `train`: Training parameters, optimizer, scheduler
- `logging`: Output directory, TensorBoard, validation sampling
- `cache`: Cache settings for performance optimization

### Important Config Files
- `face_seg_config.yaml`: Standard BF16 training (48.6GB VRAM)
- `face_seg_fp4_config.yaml`: FP4 quantized (22.5GB VRAM)
- `face_seg_flux_kontext_fp16.yaml`: FLUX Kontext FP16 (50GB VRAM)
- `face_seg_flux_kontext_fp4.yaml`: FLUX Kontext FP4 (24GB VRAM)

## Multi-GPU Configuration

Edit `accelerate_config.yaml`:
- Set `distributed_type: MULTI_GPU` for multi-GPU training
- Set `num_processes` to match number of GPUs
- Use `NO` for single GPU training

## Key Development Patterns

**Adding a new trainer:**
1. Inherit from `BaseTrainer` in `src/trainer/base_trainer.py`
2. Implement required abstract methods
3. Add trainer import case in `src/main.py:import_trainer()`
4. Set `train.trainer` in config YAML

**Cache system usage:**
- Always run with `--cache` mode first to build embeddings cache
- Cache is stored in `config.cache.cache_dir`
- Provides 2-3x training speedup by pre-computing embeddings

**Model quantization:**
- Set `model.quantize: true` in config for FP4/FP8
- Use pre-quantized model paths (e.g., `ovedrive/qwen-image-edit-4bit`)
- Quantization applies to base model, LoRA adapters remain in higher precision

## Version Management

When updating versions (per `.cursor/rules/general.mdc`):
1. Update `VERSION` file
2. Add changes to `docs/CHANGELOG.md`
3. Update `docs/TODO.md`
4. Update README.md if new features
5. Commit and push changes

## Dataset Integration

- Supports local datasets and HuggingFace datasets
- Use `qfluxutils.hugginface.load_editing_dataset()` for HF datasets
- Dataset structure: control images, target images, prompts
- Supports empty control/image for conditional training
