# Qwen-Image-Edit Fine-tuning Framework

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)  [![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-green.svg)](https://developer.nvidia.com/cuda-downloads)  [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)  [![Framework](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)

## Overview

This repository provides a comprehensive framework for fine-tuning advanced vision-language models for specialized image editing and understanding tasks. The framework supports both **Qwen-Image-Edit** and **FLUX Kontext** model architectures, offering flexible precision levels (FP16/FP8/FP4) to accommodate different hardware capabilities. Our implementation focuses on efficient training through LoRA (Low-Rank Adaptation) and features an optimized embedding cache system that achieves 2-3x training acceleration.

## Key Features

- **Dual Model Support**: Complete support for both Qwen-Image-Edit and FLUX Kontext model architectures
- **Multi-Precision Training**: FP16, FP8, and FP4 quantization levels for different hardware requirements
- **Efficient Fine-tuning**: LoRA-based parameter-efficient fine-tuning with minimal memory footprint
- **Edit Mask Loss Support**: Advanced mask-weighted loss function for focused training on edit regions
- **Embedding Cache System**: Proprietary caching mechanism for 2-3x training acceleration
- **Resume Training**: Seamless training resumption from checkpoints with full state recovery
- **Multi-GPU Support**: Distributed training capabilities with gradient accumulation
- **Quantization Support**: FP4/FP8/FP16 quantization for reduced memory usage and performance optimization
- **Flexible Architecture**: Modular design supporting various vision-language tasks
- **Production Ready**: Comprehensive testing suite and deployment configurations

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Inference](#inference)
- [Performance Benchmarks](#performance-benchmarks)
- [Citation](#citation)
- [License](#license)

## Dataset

We recommend using the curated dataset hosted on Hugging Face instead of keeping samples in-repo. The dataset used in our examples and configs:

- Face segmentation dataset: [`TsienDragon/face_segmentation_20`](https://huggingface.co/datasets/TsienDragon/face_segmentation_20)

Quick usage:

```python
from src.utils.hugginface import load_editing_dataset

dd = load_editing_dataset("TsienDragon/face_segmentation_20")
sample = dd["train"][0]
```

Dataset structure reference and upload/download instructions are in [`docs/huggingface-dataset.md`](docs/huggingface-dataset.md). We will remove the dataset copies under this repository and rely on Hugging Face going forward.

## Installation

### Prerequisites
- Python 3.12+
- CUDA 12.0+ (for GPU training)
- 16GB+ RAM, 8GB+ VRAM recommended

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/qwen-image-finetune.git
cd qwen-image-finetune

# Automated setup
./setup.sh

# Or with custom path and HF token
./setup.sh /your/path hf_your_token_here
```
Refer [`docs/speed_optimization.md`](docs/speed_optimization.md) to install `flash-atten` to speed-up the training.

### Train with Toy Dataset
1. prepare the datasets or use Hugging Face dataset (**recommended**). Refer `configs/face_seg_flux_kontext_fp16_huggingface_dataset.yaml`

2. prepare your config. Now suppose you have the config
Chose your model, optimizer, etc.
```bash
configs/my_config.yaml
```

3. (Optional) build cache first to speed up training (**recommended**)
It save the GPU memory since in the training, you dont need image encoder and prompt encoder anymore if you have the cache.
```bash
CUDA_VISIBLE_DEVICES=1 python -m src.main --config configs/my_config.yaml --cache
```
4. start training
Prepaare a `accelerate_config` to specify single gpu training or multi-gpu training
```bash
# three gpu training using accelerate
CUDA_VISIBLE_DEVICES=1,2,4 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml
```

5. resume training (add resume_from_checkpoint: to config)
Add the checkpoint path in `train.resume_from_checkpoint` in the config and resume the training
```
CUDA_VISIBLE_DEVICES=1,2,4 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/my_config.yaml
```

### Configuration Guide

The framework provides various pre-configured training setups for different models and hardware requirements:

| Config File | Model | Precision | Key Features | GPU Memory | Recommended GPU |
|------------|-------|-----------|--------------|------------|-----------------|
| `face_seg_config.yaml` | Qwen-Image-Edit | BF16 | Standard training, best quality | ~48.6GB | A100 |
| `face_seg_fp4_config.yaml` | Qwen-Image-Edit | FP4 | 4-bit quantized, memory efficient | ~22.5GB | A100/RTX4090 |
| `face_seg_fp4_4090.yaml` | Qwen-Image-Edit | FP4 | Optimized for RTX 4090 | ~23.3GB | RTX 4090 |
| `face_seg_flux_kontext_fp16.yaml` | FLUX Kontext | FP16 | Full precision, best quality | ~50GB | A100 |
| `face_seg_flux_kontext_fp16_huggingface_dataset.yaml` | FLUX Kontext | FP16 | HuggingFace dataset support | ~50GB | A100 |
| `face_seg_flux_kontext_fp8.yaml` | FLUX Kontext | FP8 | 8-bit quantized, balanced | ~35GB | A100 |
| `face_seg_flux_kontext_fp4.yaml` | FLUX Kontext | FP4 | 4-bit quantized, most efficient | ~24GB | RTX 4090/A100 |

**Key Configuration Differences:**

- **Qwen-Image-Edit Configs**:
  - Standard BF16: Uses `Qwen/Qwen-Image-Edit` base model
  - FP4 versions: Use `ovedrive/qwen-image-edit-4bit` quantized model
  - All use LoRA rank 16 for efficient fine-tuning

- **FLUX Kontext Configs**:
  - FP16: Uses standard FLUX Kontext or `camenduru/flux1-kontext-dev_fp8_e4m3fn_diffusers`
  - FP8: Uses `camenduru/flux1-kontext-dev_fp8_e4m3fn_diffusers`
  - FP4: Uses `eramth/flux-kontext-4bit-fp4`
  - Includes additional `text_encoder_2` for T5 model

**Usage Example:**
```bash
# For FLUX Kontext FP4 training on RTX 4090
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file accelerate_config.yaml -m src.main --config configs/face_seg_flux_kontext_fp4.yaml
```

#### Training with RTX4090

```
Config Exampe
configs/face_seg_fp4_4090.yaml
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config.yaml -m src.main --config $config_file
```
For multi-gpu training, need to set
```
distributed_type: MULTI_GPU  #for multi-gpu training
```
in the `accelerate_config.yaml`


#### 🎯 Qwen-Image-Edit LoRA Fine-tuning Results

This project demonstrates fine-tuning the Qwen-VL model for face segmentation tasks. Below shows the comparison between pre and post fine-tuning results:

##### Image used in Training

<div align="center">
  <table>
    <tr>
      <th>Input Image</th>
      <th>Base Model Results</th>
      <th>LoRA Fine-tuned Results</th>
    </tr>
    <tr>
      <td align="center">
        <img src="docs/images/input_image.jpg" alt="Original Input Image" width="300"/>
        <br><em>Original input image</em>
      </td>
      <td align="center">
        <img src="docs/images/result_base_model.jpg" alt="Base Model Results" width="300"/>
        <br><em>Base Qwen-Image-Edit model</em>
      </td>
      <td align="center">
        <img src="docs/images/result_lora_model.jpg" alt="LoRA Fine-tuned Model Results" width="300"/>
        <br><em>LoRA fine-tuned model</em>
      </td>
    </tr>
  </table>
  <p><strong>Comparison:</strong> The LoRA fine-tuned model shows significantly improved accuracy and details in face segmentation compared to the base model.</p>
</div>

##### Image not used in Training

<div align="center">
  <table>
    <tr>
      <th>Input Image</th>
      <th>Base Model</th>
      <th>LoRA Fine-tuned Result</th>
    </tr>
    <tr>
      <td align="center">
        <img src="docs/images/20250829_160238/input_image.jpg" alt="Input Image BF16" width="300"/>
        <br><em>Original input</em>
      </td>
      <td align="center">
        <img src="docs/images/20250829_160238/result_base_model.jpg" alt="Base Model BF16" width="300"/>
        <br><em>Base Qwen-Image-Edit model (bfloat16)</em>
      </td>
      <td align="center">
        <img src="docs/images/20250829_160238/result_lora_model.jpg" alt="LoRA BF16 Results" width="300"/>
        <br><em><strong>LoRA fine-tuned model in bf16</strong></em>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="docs/images/20250829_155502/input_image.jpg" alt="Input Image FP4" width="300"/>
        <br><em>Original input</em>
      </td>
      <td align="center">
        <img src="docs/images/20250829_155502/result_base_model.jpg" alt="Base Model FP4" width="300"/>
        <br><em>Base Qwen-Image-Edit model（fp4)</em>
      </td>
      <td align="center">
        <img src="docs/images/20250829_155502/result_lora_model.jpg" alt="LoRA FP4 Results" width="300"/>
        <br><em><strong>LoRA fine-tuned model (Base fp4,Lora bf16)</strong></em>
      </td>
    </tr>
  </table>
</div>

**Experiment Details:**
  - **Prompt:** "change the image from the face to the face segmentation mask"
  - **Row 1 - BF16 LoRA:** Base model (BF16) + LoRA adapters (BF16) - Checkpoint 900 steps on 20 samples
  - **Row 2 - FP4 LoRA:** Base model (BF16) + LoRA adapters (FP4 quantized) - Checkpoint 1000 steps on 20 samples
  - **Inference Steps:** 20, **CFG Scale:** 2.5

  **Key Observations:**
  - Both LoRA variants significantly outperform the base model
  - BF16 LoRA shows slightly better detail preservation
  - FP4 quantized LoRA maintains competitive quality while being more memory efficient
  - Base model uses BF16 precision in both experiments; only the LoRA adapters differ in quantization

#### 🔥 Flux Kontext Compare：FP16, FP8, FP4 LoRA Fine-tuning Results

<div align="center">
  <table>
    <tr>
      <th>Input Image</th>
      <th>精度类型</th>
      <th>Base Model (无LoRA)</th>
      <th>LoRA Fine-tuned Model</th>
    </tr>
    <tr>
      <td align="center" rowspan="3">
        <img src="docs/images/20250829_155502/input_image.jpg" alt="Input Image" width="250"/>
        <br><em>Original input image</em>
      </td>
      <td align="center"><strong>FP16</strong></td>
      <td align="center">
        <img src="docs/images/image-7.png" alt="FP16 Base Model Results" width="250"/>
        <br><em>Base Flux Kontext model (FP16)</em>
      </td>
      <td align="center">
        <img src="docs/images/image-6.png" alt="FP16 LoRA Fine-tuned Results" width="250"/>
        <br><em><strong>FP16 LoRA fine-tuned model</strong></em>
      </td>
    </tr>
    <tr>
      <td align="center"><strong>FP8</strong></td>
      <td align="center">
        <img src="docs/images/image-10.png" alt="FP8 Base Model Results" width="250"/>
        <br><em>FP8 base model (无LoRA)</em>
      </td>
      <td align="center">
        <img src="docs/images/image-11.png" alt="FP8 LoRA Fine-tuned Results" width="250"/>
        <br><em><strong>FP8 LoRA fine-tuned model</strong></em>
      </td>
    </tr>
    <tr>
      <td align="center"><strong>FP4</strong></td>
      <td align="center">
        <img src="docs/images/image-9.png" alt="FP4 Base Model Results" width="250"/>
        <br><em>FP4 base model (无LoRA)</em>
      </td>
      <td align="center">
        <img src="docs/images/image-8.png" alt="FP4 LoRA Fine-tuned Results" width="250"/>
        <br><em><strong>FP4 LoRA fine-tuned model</strong></em>
      </td>
    </tr>
  </table>
</div>

### Lora Training Performance

|cache|Batch Size|Quantization|Gradient Checkpoint|Flash Attention|Device|GPU Used| Training Speed|
|---|---|---|---|---|---|---|---|
|cache|2| bf16| True| False|A100|48.6 G | 18.3 s/it|
|cache|2 | fp4| True| False|A100 |22.47 | 10.6 s/it|
|cache| 2 | fbf16| True | True | A100 | 50.2 G | 10.34 s/it|
|cache| 2 | fp4 | True | True | A100 | 23.7 G | 10.8 s/it|
|cache| 2| fp4| True| True| rtx4090| 23.3/22.8G | 12.8 s/it|
|non-cache|2| fp4| True| True|A100|54.8/53.9G| 20.1 s/it|
|cache| 1| fp4| True| True| rtx4090| 18.8/17.9G | 6.34 s/it|

### Inference
```python
# Inference with trained LoRA model
from src.qwen_image_edit_trainer import QwenImageEditTrainer
from src.data.config import load_config_from_yaml
from PIL import Image

# Load configuration
config = load_config_from_yaml("configs/face_seg_config.yaml")

# Initialize trainer
trainer = QwenImageEditTrainer(config)

# Load trained LoRA weights
trainer.load_lora("/path/to/your/lora/weights")

# Setup for inference
trainer.setup_predict()

# Load input image
input_image = Image.open("data/face_seg/control_images/060002_4_028450_FEMALE_30.jpg")

# Generate face segmentation
result = trainer.predict(
    prompt_image=input_image,
    prompt="change the image from the face to the face segmentation mask",
    num_inference_steps=20,
    true_cfg_scale=4.0
)
# show the image
result[0]
# Save result
result[0].save("output_segmentation.png")
print("Generated face segmentation saved as output_segmentation.png")
```
## Edit Mask Loss

[Edit Mask Loss feature documentation in `docs/prd/image_edit_mask_loss.md`](docs/prd/image_edit_mask_loss.md)

## Speed Optimization

[Speed Optimization including quantilizationand flash attention in `docs/speed_optimization.md`](docs/speed_optimization.md)

## Debug
[Record of bugs encountered in `docs/debug.md`](docs/debug.md)

## 🎯 Key Features Covered

### Training Optimization
- **Edit Mask Loss**: Advanced mask-weighted loss computation for focused training on edit regions
- **Embedding Cache System**: 2-3x training acceleration
- **LoRA Fine-tuning**: Memory-efficient parameter updates
- **Multi-GPU Support**: Distributed training capabilities
- **Gradient Checkpointing**: Memory optimization techniques

### Model Capabilities
- **Multimodal Processing**: Combined text and image understanding
- **Flexible Architecture**: Support for various image editing tasks
- **Quality Control**: Advanced inference parameters and optimization
- **Production Ready**: Deployment guides and API examples

### Developer Tools
- **Comprehensive Testing**: Full test suite and validation tools
- **Monitoring**: Real-time storage and performance monitoring
- **Configuration**: Flexible YAML-based configuration system
- **Documentation**: Complete API reference and examples

## 🤝 Contributing to Documentation

We welcome contributions to improve this documentation:

1. **Found an Error?** Open an issue or submit a PR
2. **Missing Information?** Suggest additions or improvements
3. **Want to Help?** Contact the maintainers for contribution guidelines

### Documentation Standards
- Use clear, concise language
- Include practical examples
- Provide complete code snippets
- Add troubleshooting sections
- Keep content up to date

## 📞 Getting Help

### Documentation Issues
- **Missing Information**: Check if it's covered in another section
- **Outdated Content**: Open an issue to report outdated information
- **Unclear Instructions**: Suggest improvements via issues or PRs

### Technical Support
- **Training Issues**: See [Training Guide](docs/training.md) troubleshooting
- **Data Preparation**: See [Data Preparation Guide](docs/data-preparation.md) for dataset setup
- **HuggingFace Datasets**: See [HuggingFace Dataset Guide](docs/huggingface-dataset.md) for cloud datasets
- **Optimizer Selection**: See [Training Guide](docs/training.md#optimizer-selection) for available optimizers
- **FLUX Kontext Training**: See [Training Guide](docs/training.md#flux-kontext-lora-training) for multi-precision training
- **Setup Problems**: Check [Setup Guide](docs/setup.md) common issues
- **Performance**: Review [Cache System](docs/cache-system.md) optimization
- **General Questions**: Open a GitHub issue with detailed description

## 📚 External Resources

### Related Projects
- [Qwen Official Repository](https://github.com/QwenLM/Qwen)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)

### Community
- [GitHub Discussions](../../discussions) - General discussions and Q&A
- [Issues](../../issues) - Bug reports and feature requests
- [Pull Requests](../../pulls) - Code contributions

---

**📝 Note**: This documentation is continuously updated. Last updated: 2025/08/28

**⭐ Tip**: Use the navigation links above to jump to specific topics, or browse sequentially for a complete understanding of the framework.
