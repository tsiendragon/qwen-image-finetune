# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [3.0.0] - 2025-10-19

### Added
- **Multi-Resolution Mixed Training Support**: Complete implementation of mixed resolution training capabilities
  - Support for training with multiple resolution candidates in a single training session
  - Intelligent resolution selection based on original image aspect ratio
  - Two configuration formats: Simple list mode (shared across all images) and Advanced dict mode (per-image-type configuration)
  - Automatic aspect ratio preservation with configurable maximum aspect ratio limit
  - Dynamic pixel-based target selection from candidate resolutions
  - Example configuration at `tests/test_configs/test_example_fluxkontext_multiresolution.yaml`

### Changed
- **Data Processing Enhancement**: Extended `ImageProcessor` with multi-resolution mode support
  - New `multi_resolutions` configuration parameter supporting list or dict formats
  - `_parse_multi_resolution_config()` method for parsing resolution candidates
  - `_select_pixels_candidate()` method for intelligent resolution selection
- **Configuration Schema**: Added `multi_resolutions` field to `ImageProcessorInitArgs`
  - Supports simple list format: `["1024*1024", "512*512"]` (applies to all images)
  - Supports advanced dict format: `{target: [...], controls: [[...], [...]]}`
  - Includes configurable `max_aspect_ratio` parameter for aspect ratio constraints

### Technical Details
- Resolution candidates are parsed from string expressions (e.g., "512*512") or integers
- Aspect ratio-aware selection ensures optimal resolution choice for each input
- Compatible with all existing training modes (single control, multi-control)
- Works seamlessly with cache system, quantization, and all supported model architectures
- Full backward compatibility: existing configs without `multi_resolutions` continue to work unchanged

### Breaking Changes
- None - This is a backwards-compatible addition. Existing configurations without `multi_resolutions` will function exactly as before.

---

## [2.4.1] - 2025-10-04

### Changed
- Documentation improvements and clarifications
  - README license badge updated to MIT, added License section linking to LICENSE
  - Data preparation guide updated: now documents Folder/Hugging Face/CSV dataset sources
  - Data preparation guide language switched to English
  - Updated dynamic shapes usage and configuration descriptions

### Fixed
- Minor docs fixes and typos

---

## [2.4.0] - 2025-10-01

### Added
- Dynamic shape support for Qwen-Image-Edit-Plus training and preprocessing
  - Introduced pixel-based dynamic sizing: `target_pixels`, `controls_pixels`
  - Accept integer or expression (e.g., `512*512`) with validation in `src/data/config.py`
  - Example config at `tests/test_configs/test_example_qwen_image_edit_plus_fp4_dynamic_shapes.yaml`

### Changed
- Updated README with v2.4.0 notes and config examples
- Refined configuration guide to include dynamic shape options

### Fixed
- Minor wording and formatting issues across docs

---

## [2.3.0] - 2025-09-26

### Added
- **Qwen-Image-Edit-Plus Support**: Complete support for the enhanced Qwen-Image-Edit-Plus (2509) model architecture
  - Native support for multiple image composition without additional preprocessing
  - Enhanced character composition capabilities with improved quality
  - Full compatibility with existing LoRA training pipeline
  - Support for FP4 quantization training with optimized memory usage
  - Comprehensive test configurations and example notebooks provided

### Changed
- **Model Architecture Enhancement**: Extended framework to support the latest Qwen-Image-Edit-Plus model
- **Training Configuration**: Added new configuration templates for Qwen-Image-Edit-Plus training scenarios
- **Documentation**: Updated README.md with Qwen-Image-Edit-Plus examples and usage instructions

### Technical Details
- Qwen-Image-Edit-Plus model provides native multi-image composition support
- Optimized training configurations for character composition and face segmentation tasks
- Pretrained model available at `TsienDragon/qwen-image-edit-plus-lora-face-seg`
- Complete integration with existing caching and quantization systems

---

## [2.2.0] - 2025-09-24

### Added
- **CSV数据格式支持**: 新增对CSV元数据文件的数据集支持，提供更灵活的数据集结构管理
  - 支持基于CSV文件的数据集结构定义，可自定义图像路径、提示文本等字段
  - 兼容混合图像格式（JPG、PNG等）和灵活的目录结构
  - 保持与现有数据集格式的完全兼容性

### Changed
- **数据集处理能力**: 扩展了数据集支持范围，现在同时支持传统目录结构和CSV元数据文件

### Technical Details
- CSV数据集格式支持自定义列名映射，适应不同的数据集结构
- 自动检测并处理混合图像格式和路径结构
---

## [2.1.0] - 2025-09-19

### Added
- **FSDP Training Support**: Implemented Fully Sharded Data Parallel (FSDP) training capabilities for enhanced memory efficiency on multi-GPU setups
  - 支持 FSDP 分布式训练模式，显著降低单卡显存占用
  - 自动参数分片和梯度同步，支持大规模模型训练
  - 提供 FSDP 配置示例和性能对比数据

### Changed
- **Training Configuration**: Updated default FSDP configuration settings for optimal performance
- **Documentation Enhancement**: Added comprehensive FSDP training guides and multi-control image examples

### Technical Details
- FSDP memory optimization: Single GPU memory usage reduced from 43G to 22.2G on A100 with 3-GPU setup
- Complete FSDP vs DDP performance comparison documentation

---

## [2.0.0] - 2025-09-18
Update to v2.0.0 with the refactored codes
## [1.6.0] - 2025-09-18

### Added
- **Save Final Checkpoint on Completion**: Automatically save the final checkpoint when training completes or is interrupted, ensuring training progress is preserved
- **HuggingFace Compatible LoRA Format**: LoRA weights now use HuggingFace-compatible state dict filenames and format, enabling direct loading with diffusers pipeline: `pipe.load_lora_weights(lora_weight, prefix="", adapter_name="lora_edit")`
- **Automatic LoRA Weight Upload**: Support automatic upload of trained LoRA weights to HuggingFace model repository

### Changed
- **Training Workflow Optimization**: Improved weight saving and management workflow after training completion
- **Enhanced Compatibility**: Improved integration with HuggingFace ecosystem for easier model sharing and deployment

### Technical Details
- LoRA state dict format fully compatible with HuggingFace standards
- Automatic checkpoint saving mechanism ensures training robustness
- Support one-click upload of training results to HuggingFace Hub

---

## [1.5.3] - 2025-09-15

### Fixed
- **Documentation Corrections**: Updated additional control image naming rule descriptions to ensure consistency with actual code implementation
  - Fixed control image naming rules in `docs/huggingface-dataset.md` from `<base>_1.*` to `<base>_control_1.*`
  - Updated example directory structure to match actual naming conventions
  - Fixed comments in `src/data/dataset.py` to align with code implementation
  - Corrected additional control image naming description in `docs/prd/multi_control.md`
  - Fixed control image collection logic description in `src/utils/hugginface.py`

### Technical Details
- Unified additional control image naming convention to `<base>_control_1.*`, `<base>_control_2.*` format
- Ensured all documentation matches code implementation to avoid user confusion

---

## [1.5.2] - 2025-09-11

### Added
- **HuggingFace Dataset Support**: Implemented seamless integration with HuggingFace datasets
  - 在 `ImageDataset` 类中添加智能路径检测，自动区分本地路径和 HF 仓库 ID
  - 实现懒加载机制，避免大型数据集的枚举开销
  - 支持混合使用本地数据集和 HF 数据集
  - 提供 `face_seg_flux_kontext_fp16_huggingface_dataset.yaml` 配置示例
  - 完整的数据格式转换，确保与现有预处理流程兼容

### Changed
- **Data Loading Performance**: Significantly improved large dataset loading performance through lazy loading mechanism
- **Configuration System**: Maintained full backward compatibility, existing config files require no modifications
- **Documentation**: Added `docs/huggingface-dataset.md` detailed guide including upload, download, and usage instructions

### Technical Details
- Path detection logic: Automatically distinguishes between local paths and HF repository ID formats
- Data format conversion: HF dataset PIL Image → RGB numpy array
- Unified interface: `__getitem__` method provides consistent handling for both local and HF data
- Complete design documentation: `docs/prd/v1.5.2_add_huggingface_dataset.md`

---

## [1.5.1] - 2025-01-11

### Changed
- **Optimizer Selection Documentation**: Added new optimizer selection section in `docs/training.md`
  - 支持四种优化器：AdamW（默认）、8-bit Adam（内存高效）、Prodigy（自动学习率）、SGD（经典）
  - 提供详细的 YAML 配置示例和使用场景说明
  - 添加快速参考表格，对比各优化器的内存占用、自动学习率功能和最佳使用场景
  - 在 README.md 技术支持部分添加优化器选择的快速链接

### Fixed
- Fixed bitsandbytes optimizer class path (from `bnb.optim.Adam8bit` to `bitsandbytes.optim.Adam8bit`)

---

## [1.5.0] - 2025-01-10

### Added
- **FLUX Kontext LoRA Training Support**: Complete implementation of LoRA fine-tuning functionality for FLUX Kontext models
  - 支持三种精度级别：FP16（最高质量）、FP8（平衡性能）、FP4（最大效率）
  - 提供预配置的训练配置文件：`face_seg_flux_kontext_fp16.yaml`、`face_seg_flux_kontext_fp8.yaml`、`face_seg_flux_kontext_fp4.yaml`
  - 实现了 `FluxKontextLoraTrainer` 类，继承自 `BaseTrainer` 确保接口一致性
  - 支持双文本编码器架构（CLIP + T5），优化多设备分配策略
  - 完整的缓存系统支持，包括 VAE、CLIP 和 T5 编码器的独立设备配置

### Changed
- **Documentation Enhancement**: Added detailed FLUX Kontext training guidance section in `docs/training.md`
  - 详细的精度对比表格，包含质量、训练速度、显存需求和使用场景
  - 完整的训练工作流程和多GPU训练配置
  - 设备分配策略和内存优化建议
  - FLUX Kontext 推理代码示例和最佳实践
- **框架描述更新**: 更新 README.md 以反映双模型架构支持
  - 强调对 Qwen-Image-Edit 和 FLUX Kontext 的完整支持
  - 突出多精度训练能力（FP16/FP8/FP4）
  - 更新技术支持部分，添加 FLUX Kontext 训练指南链接

### Technical Details
- FLUX Kontext model architecture: Transformer-based diffusion model supporting joint understanding of images and text
- Precision performance comparison: FP16 (reference quality), FP8 (95% quality, 1.5x speed), FP4 (85% quality, 2.5x speed)
- Memory requirement optimization: FP16 (24GB training/12GB inference) → FP8 (18GB/8GB) → FP4 (12GB/5GB)
- 支持的预训练模型：
  - `black-forest-labs/FLUX.1-Kontext-dev` (FP16)
  - `camenduru/flux1-kontext-dev_fp8_e4m3fn_diffusers` (FP8)
  - `eramth/flux-kontext-4bit-fp4` (FP4)

---

## [1.4.0] - 2025-01-03

### Added
- **Edit Mask Loss Support**: Implemented mask-based loss function to improve image editing training effectiveness
  - 新增 `MaskEditLoss` 类，支持对编辑区域和背景区域的差异化权重
  - 实现 `map_mask_to_latent` 函数，将图像空间掩码映射到潜在空间序列权重
  - 支持可配置的前景权重 (`foreground_weight`) 和背景权重 (`background_weight`)
  - 完全向后兼容，现有训练管道在 `mask_loss: false` 时保持不变

### Changed
- **Dataset Support**: Extended dataset structure to support optional edit mask files (`*_mask.png`)
- **Configuration System**: Added `loss.mask_loss`, `loss.foreground_weight`, `loss.background_weight` configuration options
- **Training Effectiveness**: Improved model convergence speed and quality in key editing regions through focused loss computation

### Technical Details
- Mask processing pipeline: Image space → VAE downsampling → 2×2 patch merging → Latent space sequence alignment
- Dual loss computation: Combination of original loss + mask-weighted loss
- Support for joint application of timestep weights and mask weights
- Detailed implementation documentation: `docs/image_edit_mask_loss.md`

---

## [0.2.0] - 2025-08-27

### Added
- **QwenImageEditTrainer.predict() Method**: Implemented complete image generation inference functionality
  - 支持单张或批量图像处理
  - 完整的 CFG (Classifier-Free Guidance) 支持，包括负面提示
  - 多设备 GPU 分布式推理支持
  - 遵循原始 QwenImageEditPipeline 的完整推理逻辑
  - 支持可配置的推理步数和引导强度
  - 包含完整的去噪循环和时间步调度
  - 输出高质量的 RGB 格式图像

### Changed
- **Inference Performance**: New predict method now correctly generates high-quality image samples
- **Device Management**: Optimized device allocation strategy for multi-GPU inference
- **Memory Efficiency**: Implemented transformer cache context to improve inference efficiency

### Technical Details
- Complete rewrite of predict method ensuring consistent behavior with QwenImageEditPipeline
- Implemented correct latent vector processing and image decoding pipeline
- Added complete batch processing support
- Integrated progress bar display to track inference progress
![alt text](images/image.png)
---

## Historical Versions

### [0.1.0] - 2025-08-27
- Initial version release
- Basic training functionality implementation
- Cache system support
