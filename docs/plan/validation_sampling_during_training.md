# Validation Sampling During Training

**Last Updated:** October 26, 2025
**Status:** Planning
**Version Target:** 3.1.0

## Overview

This plan outlines the implementation of a validation sampling process during model training. The feature will enable periodic sampling from validation data to visualize generation quality in TensorBoard, providing real-time feedback on training progress.

## Motivation

- Enable real-time monitoring of model generation quality during training
- Provide visual feedback on how the model is improving over time
- Help identify potential issues or degradation early in the training process
- Support better hyperparameter tuning through immediate visual feedback

## Technical Design

### 1. Core Components

#### 1.1 BaseTrainer Enhancements

Add the following methods to `BaseTrainer`:

```python
import torch

def setup_validation_data(self):
    """
    Prepare validation data for periodic sampling during training.
    Called during trainer initialization.
    """
    if not self.config.validation.enabled:
        return

    # Load validation dataset
    validation_dataset = self.load_validation_dataset()

    # Limit to max_validation_samples if specified
    max_samples = self.config.validation.max_samples
    if max_samples and max_samples < len(validation_dataset):
        validation_dataset = validation_dataset[:max_samples]

    # Prepare validation embeddings
    self.validation_embeddings = []
    for sample in validation_dataset:
        # Process sample to match predict_batch format
        batch = self.prepare_validation_sample(sample)
        # Get embeddings
        embeddings = self.prepare_embeddings(batch, stage="predict")

        # Move embeddings to CPU to save GPU memory
        cpu_embeddings = {}
        for k, v in embeddings.items():
            if isinstance(v, torch.Tensor):
                cpu_embeddings[k] = v.cpu()
            else:
                cpu_embeddings[k] = v

        self.validation_embeddings.append(cpu_embeddings)

    logger.info(f"Prepared {len(self.validation_embeddings)} validation samples for periodic sampling")

def prepare_validation_sample(self, sample):
    """
    Convert a validation dataset sample to the format expected by prepare_embeddings.
    """
    # Implementation depends on dataset format
    pass

def run_validation(self):
    """
    Run validation sampling and log results to TensorBoard.
    Called periodically during training.
    """
    if not self.config.validation.enabled or not hasattr(self, "validation_embeddings"):
        return

    logger.info(f"Running validation sampling at step {self.global_step}")

    # Move VAE to device temporarily
    self.vae.to(self.accelerator.device)

    # Process each validation sample
    for i, cpu_embeddings in enumerate(self.validation_embeddings):
        # Only process samples assigned to this device in distributed training
        if self.accelerator.is_main_process or i % self.accelerator.num_processes == self.accelerator.process_index:
            # Move embeddings to current device
            embeddings = {}
            for k, v in cpu_embeddings.items():
                if isinstance(v, torch.Tensor):
                    embeddings[k] = v.to(self.accelerator.device)
                else:
                    embeddings[k] = v

            # Sample from embeddings
            with torch.no_grad():
                latents = self.sampling_from_embeddings(embeddings)

                # Decode latents to images
                images = self.decode_vae_latent(
                    latents,
                    embeddings["height"],
                    embeddings["width"]
                )

                # Log to TensorBoard
                if self.accelerator.is_main_process:
                    self.log_validation_images(
                        images,
                        embeddings,
                        sample_idx=i
                    )

    # Move VAE back to CPU to save memory
    self.vae.to("cpu")
    torch.cuda.empty_cache()

def log_validation_images(self, images, embeddings, sample_idx):
    """
    Log validation images and metadata to TensorBoard.
    """
    # Get prompt for display
    prompt = embeddings.get("prompt", f"Sample {sample_idx}")

    # Get TensorBoard writer
    if hasattr(self.accelerator.trackers[0], "writer"):
        writer = self.accelerator.trackers[0].writer
    else:
        logger.warning("TensorBoard writer not found in accelerator trackers")
        return

    # Log generated images
    for idx, img in enumerate(images):
        # 确保图像是正确的格式 [C, H, W]
        if img.ndim == 4:
            img = img[0]  # 如果是批量图像，取第一个
        if img.ndim == 3 and img.shape[0] == 3:  # 已经是 [C, H, W] 格式
            pass
        elif img.ndim == 3:  # 可能是 [H, W, C] 格式
            img = img.permute(2, 0, 1)

        # 确保在CPU上并且是正确的值范围
        img = img.cpu()
        if img.max() > 1.0:
            img = img / 255.0

        writer.add_image(
            f"validation/generated_images/sample_{sample_idx}_{idx}",
            img,
            global_step=self.global_step
        )

    # Log control images if available
    if "control_images" in embeddings:
        for idx, img in enumerate(embeddings["control_images"]):
            # 确保图像是正确的格式 [C, H, W]
            if img.ndim == 4:
                img = img[0]  # 如果是批量图像，取第一个
            if img.ndim == 3 and img.shape[0] == 3:  # 已经是 [C, H, W] 格式
                pass
            elif img.ndim == 3:  # 可能是 [H, W, C] 格式
                img = img.permute(2, 0, 1)

            # 确保在CPU上并且是正确的值范围
            img = img.cpu()
            if img.max() > 1.0:
                img = img / 255.0

            writer.add_image(
                f"validation/control_images/sample_{sample_idx}_{idx}",
                img,
                global_step=self.global_step
            )

    # Log prompt text
    writer.add_text(
        f"validation/prompt/sample_{sample_idx}",
        prompt,
        global_step=self.global_step
    )

```

#### 1.2 Training Loop Integration

Modify the `train` method to call validation at specified intervals:

```python
def train(self):
    # Existing training setup code...

    # Setup validation data if enabled
    self.setup_validation_data()

    # Training loop
    for step in range(first_epoch, max_train_steps):
        # Existing training step code...

        # Run validation at specified intervals
        if (
            self.config.validation.enabled and
            step > 0 and
            step % self.config.validation.steps == 0
        ):
            self.run_validation()

        # Existing checkpoint and logging code...
```

### 2. Configuration

Add validation configuration to the YAML schema:

```yaml
# 1. 基础验证配置
validation:
  enabled: true       # 启用验证采样
  steps: 500          # 每N步执行一次验证
  max_samples: 5      # 最多使用的验证样本数量
  seed: 42            # 可选：用于采样的固定随机种子

  # 2. HuggingFace数据集配置
  dataset:
    class_path: "qflux.data.datasets.HuggingFaceDataset"
    init_args:
      dataset_name: "namespace/dataset_name"  # HuggingFace数据集名称
      split: "validation"                    # 使用的数据集分割
      image_column: "control_image"          # 控制图像列名
      caption_column: "prompt"               # 提示文本列名
      additional_control_columns:            # 可选：额外控制图像列名
        - "control_image_depth"
        - "control_image_canny"

# 3. CSV文件数据集配置
validation:
  enabled: true
  steps: 500
  dataset:
    class_path: "qflux.data.datasets.CSVDataset"
    init_args:
      csv_file: "path/to/validation.csv"     # CSV文件路径
      image_dir: "path/to/images"           # 图像目录路径
      image_column: "control_image"         # 控制图像列名
      caption_column: "prompt"              # 提示文本列名
      additional_control_columns:           # 可选：额外控制图像列名
        - "depth_map"
        - "canny_edge"

# 4. 直接列表配置
validation:
  enabled: true
  steps: 500
  samples:
    - prompt: "一只可爱的猫咪坐在窗台上"
      images:
        - "path/to/cat_control.png"
        - "path/to/cat_depth.png"
        - "path/to/cat_canny.png"
      controls_size: [[512, 512], [512, 512], [512, 512]]  # 每个输入图像的目标尺寸 [height, width]
      height: 512  # 生成图像高度
      width: 512   # 生成图像宽度
    - prompt: "日落时分的城市天际线"
      images:
        - "path/to/skyline_control.png"
      controls_size: [[768, 512]]  # 单个控制图像的目标尺寸 [height, width]
      height: 768
      width: 512

```

### 3. Distributed Training Considerations

- In distributed training, each process should handle a subset of validation samples
- Only the main process should log to TensorBoard
- Ensure proper synchronization between processes

## Implementation Plan

### Phase 1: Core Implementation

1. Add validation configuration schema
2. Implement `setup_validation_data` and `prepare_validation_sample` methods
3. Implement `run_validation` method
4. Implement `log_validation_images` and helper methods
5. Integrate validation calls into the training loop
6. Add memory management for VAE (CPU/GPU movement)

### Phase 2: Testing and Optimization

1. Test with single-GPU training
2. Test with multi-GPU distributed training
3. Optimize memory usage during validation
4. Ensure TensorBoard logging works correctly
5. Add validation metrics (optional)

### Phase 3: Documentation and Release

1. Update documentation with validation feature details
2. Add example validation configuration to templates
3. Create changelog entry
4. Release as part of version 3.1.0

## Memory Considerations

- VAE should be moved to GPU only during validation and back to CPU afterward
- Validation should process one sample at a time to minimize memory usage
- In distributed settings, samples should be distributed across GPUs

## Future Extensions

- Add quantitative metrics for validation (FID, CLIP score, etc.)
- Support custom validation callbacks
- Allow validation dataset filtering and selection
- Enable validation image export to disk

## 特定训练器的考虑因素

### 1. 不同训练器的兼容性分析

当前实现设计兼容所有现有训练器，因为它使用了所有训练器共有的基础方法：

| 方法 | BaseTrainer | QwenImageEditTrainer | QwenImageEditPlusTrainer | FluxKontextLoraTrainer |
| --- | :---: | :---: | :---: | :---: |
| `prepare_embeddings` | ✓ (抽象) | ✓ | ✓ | ✓ |
| `sampling_from_embeddings` | ✓ (抽象) | ✓ | 继承 | ✓ |
| `decode_vae_latent` | ✓ (抽象) | ✓ | 继承 | ✓ |

#### 参数兼容性分析

`prepare_predict_batch_data` 方法的参数在不同训练器之间有细微差异：

| 参数 | QwenImageEditTrainer | FluxKontextLoraTrainer | 注意事项 |
| --- | --- | --- | --- |
| `controls_size` | `list[int] 类型` | `list[list[int]] 类型` | FluxKontext需要列表嵌套列表格式 [[h1,w1], [h2,w2]] |
| `prompt` | 必需 | 可选 | FluxKontext支持双提示词模式 |
| `use_multi_resolution` | 不支持 | 支持 | 仅FluxKontext支持多分辨率模式 |

为确保兼容性，验证采样实现将使用最通用的参数格式，并在必要时进行转换。

### 2. 特定训练器的注意事项

- **QwenImageEditTrainer/QwenImageEditPlusTrainer**:
  - 使用 `AutoencoderKLQwenImage` 作为VAE
  - 控制图像格式为标准RGB图像
  - 支持单一分辨率输入

- **FluxKontextLoraTrainer**:
  - 支持多分辨率训练和采样
  - 当前实现将仅使用标准的 `sampling_from_embeddings` 方法

### 3. 实现策略

为确保兼容性，验证采样实现将：

1. 在 `BaseTrainer` 中实现通用逻辑
2. 允许特定训练器子类覆盖 `prepare_validation_sample` 方法以处理特定格式
3. 对所有训练器类型使用标准的 `sampling_from_embeddings` 方法

## 开放问题

1. 是否应该使用固定种子进行可重现的比较？
2. 是否应该支持像训练数据一样缓存验证数据集？
3. 如何在后期支持FluxKontextLoraTrainer的多分辨率验证采样？
